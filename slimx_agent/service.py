"""The standalone SlimX-Agent service: the engine loop in its own container.

Runs :mod:`slimx_agent.engine` over :class:`~slimx_agent.http_store.HttpRunStore` and the
remote tool registry — every store operation, step tool, and the run-end epilogue call back
to the HOST's internal agent-host API. The container therefore needs NO database, NO
provider credentials, and NO host code: its entire world is ``SLIMX_AGENT_HOST_URL`` plus
the shared ``SLIMX_AGENT_INTERNAL_TOKEN`` (one value authenticates both directions).

Surface (the wire contract is ``docs/service-contract.md``):

- ``GET /health`` — liveness + ``version`` + ``auth_enabled`` + ``mode: "standalone"`` so the
  host's deep health can flag token, topology, and version drift.
- ``POST /agent/runs/{run_id}/execute`` — drive an already-claimed run to its next stop.
- ``POST /agent/runs/{run_id}/execute/stream`` — same, streaming each durable event as an SSE
  ``data:`` line with ``:``-comment keepalives while a step is quiet. The stream only
  observes: a disconnected observer does not cancel the drive, which continues to its next
  durable stop. Cancellation is the host's run-status authority, re-read before every step.
- ``POST /internal/run-check`` — absent unless ``SLIMX_AGENT_ENABLE_RUN_CHECK`` is set, and
  then refused without a configured token. See :func:`_run_bounded_check` for its limits.

Planning and the system map stay host-side on purpose: they are host-shaped (context
manifests, provider profile resolution, extraction services). The loop is the portable part.

Requires the ``service`` extra (fastapi/uvicorn/httpx).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import selectors
import signal
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from slimx_agent import __version__, engine
from slimx_agent.host_client import ExecutionAttempt, HostClient, HostError, bounded_detail
from slimx_agent.http_store import HttpRunStore, RunSnapshot
from slimx_agent.http_tools import build_remote_registry
from slimx_agent.runtime import RunProfile
from slimx_agent.tools import StepOutcomeUnknown

logger = logging.getLogger("slimx_agent.service")

KEEPALIVE_COMMENT = ": keepalive\n\n"
DEFAULT_KEEPALIVE_SECONDS = 15.0

# The legacy check runner is opt-in: set this to 1/true/yes/on to expose /internal/run-check.
RUN_CHECK_ENABLED_ENV = "SLIMX_AGENT_ENABLE_RUN_CHECK"

# Opaque run identifiers appear in callback paths (quoted again there) and, for run-check, in a
# workspace path: no separators, no leading dot, bounded length.
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
# A plain assignment (not a ``type`` alias) so FastAPI reads the Path metadata at runtime.
RunIdPath = Annotated[str, Path(pattern=RUN_ID_PATTERN)]

_READ_CHUNK_BYTES = 65_536


class ProfileBody(BaseModel):
    """The host-resolved execution profile (egress already enforced host-side) plus the lease.

    Additional host-only profile fields (ControlRoom also sends safe provider settings and
    profile identity) are accepted and ignored: the host re-resolves them from its own records
    at every callback. ``provider``/``model``/``base_url`` are forwarded exactly.
    """

    model_config = ConfigDict(extra="ignore")

    provider: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=512)
    base_url: str | None = Field(default=None, max_length=2048)
    lease_job_id: UUID | None = None
    lease_token: UUID | None = None
    lease_generation: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_complete_lease(self) -> ProfileBody:
        supplied = (
            self.lease_job_id is not None,
            self.lease_token is not None,
            self.lease_generation is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("Incomplete agent execution lease")
        return self

    def to_profile(self) -> RunProfile:
        return RunProfile(self.provider, self.model, self.base_url)

    def execution_attempt(self) -> ExecutionAttempt:
        """Return the required attempt fence for an execute endpoint.

        ProfileBody keeps the all-absent shape parseable for wire compatibility with code
        that constructs profiles outside this service.  This standalone app has no planning
        endpoint, however, so both execute surfaces fail closed when the attempt is absent.
        """
        if self.lease_job_id is None or self.lease_token is None or self.lease_generation is None:
            raise HTTPException(status_code=422, detail="Agent execution lease is required")
        return ExecutionAttempt(
            job_id=str(self.lease_job_id),
            token=str(self.lease_token),
            generation=self.lease_generation,
        )


class RunCheckBody(BaseModel):
    """One already-allowlisted check command for the legacy run-check runner."""

    model_config = ConfigDict(extra="forbid")

    argv: list[Annotated[str, Field(min_length=1, max_length=4096)]] = Field(
        min_length=1, max_length=64
    )
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    timeout_seconds: StrictFloat = Field(default=120.0, gt=0, le=600, allow_inf_nan=False)
    output_cap: StrictInt = Field(default=20_000, ge=1, le=100_000)

    @field_validator("argv")
    @classmethod
    def _no_nul_bytes(cls, value: list[str]) -> list[str]:
        if any("\x00" in part for part in value):
            raise ValueError("argv must not contain NUL bytes")
        return value


@dataclass(frozen=True)
class CheckResult:
    exit_code: int | None
    timed_out: bool
    output: bytes
    truncated: bool


def _keepalive_seconds() -> float:
    try:
        value = float(os.environ.get("SLIMX_AGENT_KEEPALIVE_SECONDS", DEFAULT_KEEPALIVE_SECONDS))
    except ValueError:
        return DEFAULT_KEEPALIVE_SECONDS
    return value if value > 0 else DEFAULT_KEEPALIVE_SECONDS


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _token_matches(authorization: str | None, token: str) -> bool:
    """Constant-time bearer comparison over bytes, so a non-ASCII header is a 401, not a 500."""
    if authorization is None:
        return False
    supplied = authorization.encode("utf-8", "surrogateescape")
    expected = f"Bearer {token}".encode("utf-8", "surrogateescape")
    return secrets.compare_digest(supplied, expected)


def _host_http_error(exc: HostError) -> HTTPException:
    """Preserve a host's fail-closed 4xx (notably a stale-lease 409) at the service edge."""
    status_code = exc.status_code if 400 <= exc.status_code < 500 else 502
    return HTTPException(status_code=status_code, detail=bounded_detail(str(exc)))


def _outcome_unknown_http_error(exc: StepOutcomeUnknown) -> HTTPException:
    """A drive that ended on an unobserved step outcome: relay the host's own refusal status
    when there was one, else 502. Either way the host's durable records own resolution."""
    if isinstance(exc.__cause__, HostError):
        return _host_http_error(exc.__cause__)
    return HTTPException(status_code=502, detail=bounded_detail(str(exc)))


def _not_permitted_http_error(exc: engine.RunningStepNotPermitted) -> HTTPException:
    """A step left running lost its grant: the run's state needs the host, so 409, not a fault."""
    return HTTPException(status_code=409, detail=bounded_detail(str(exc)))


def _sanitized_validation_errors(errors: Sequence[Any]) -> list[dict[str, object]]:
    """Validation errors without the offending input or context: never echo request material
    (which could carry a credential), and never fail to render a non-finite number."""
    sanitized: list[dict[str, object]] = []
    for error in errors:
        loc = error.get("loc", ()) if isinstance(error, dict) else ()
        sanitized.append(
            {
                "type": str(error.get("type", "")) if isinstance(error, dict) else "",
                "loc": [part if isinstance(part, str | int) else str(part) for part in loc],
                "msg": bounded_detail(str(error.get("msg", ""))) if isinstance(error, dict) else "",
            }
        )
    return sanitized


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    return JSONResponse(status_code=422, content={"detail": _sanitized_validation_errors(errors)})


def create_app(host_client: HostClient | None = None) -> FastAPI:
    app = FastAPI(title="SlimX-Agent", version=__version__)
    app.add_exception_handler(RequestValidationError, _validation_error)
    registry = build_remote_registry()
    state: dict[str, HostClient | None] = {"client": host_client}
    client_lock = threading.Lock()

    def client() -> HostClient:
        # Built lazily so importing the module (and /health) never requires the host URL.
        with client_lock:
            existing = state["client"]
            if existing is None:
                try:
                    existing = HostClient(
                        os.environ.get("SLIMX_AGENT_HOST_URL"),
                        token=os.environ.get("SLIMX_AGENT_INTERNAL_TOKEN") or None,
                    )
                except ValueError as exc:
                    raise HTTPException(
                        status_code=503, detail="SLIMX_AGENT_HOST_URL is not configured"
                    ) from exc
                state["client"] = existing
            return existing

    def require_internal_token(authorization: str | None = Header(default=None)) -> None:
        token = os.environ.get("SLIMX_AGENT_INTERNAL_TOKEN") or ""
        if not token:
            return  # documented local-first compatibility: no configured token means no auth
        if not _token_matches(authorization, token):
            raise HTTPException(status_code=401, detail="Missing or invalid internal service token")

    def store_and_run(run_id: str, callback_client: HostClient) -> tuple[HttpRunStore, RunSnapshot]:
        store = HttpRunStore(callback_client)
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Agent run not found")
        return store, run

    def on_run_end(
        profile: RunProfile, callback_client: HostClient
    ) -> Callable[[RunSnapshot, str], None]:
        def hook(run: RunSnapshot, status: str) -> None:
            # The epilogue is the host's concern (and flag-gated there); a callback failure
            # must never jeopardize the finished run's own result.
            try:
                callback_client.run_end(run.id, status, profile)
            except Exception:
                logger.exception("run-end callback failed for run %s", run.id)

        return hook

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "slimx-agent",
            "version": __version__,
            "mode": "standalone",
            "auth_enabled": bool(os.environ.get("SLIMX_AGENT_INTERNAL_TOKEN")),
        }

    if _env_flag(RUN_CHECK_ENABLED_ENV):

        def require_run_check_token(authorization: str | None = Header(default=None)) -> None:
            # Unlike the loop endpoints, command execution has no tokenless local mode.
            token = os.environ.get("SLIMX_AGENT_INTERNAL_TOKEN") or ""
            if not token:
                raise HTTPException(
                    status_code=503,
                    detail="run-check requires SLIMX_AGENT_INTERNAL_TOKEN on this service",
                )
            if not _token_matches(authorization, token):
                raise HTTPException(
                    status_code=401, detail="Missing or invalid internal service token"
                )

        @app.post("/internal/run-check")
        def run_check(
            body: RunCheckBody, _: None = Depends(require_run_check_token)
        ) -> dict[str, Any]:
            """Run ONE host-allowlisted check command in a run's mutable workspace directory.

            The host owns the allowlist decision; this endpoint re-enforces mechanical bounds
            (see :func:`_run_bounded_check`). It never interprets or expands the command. It is
            NOT an exact-snapshot runner, so it cannot back a digest-bound check receipt."""
            root = os.path.realpath(os.environ.get("AGENT_WORKSPACE_ROOT", "/workspaces"))
            cwd = os.path.realpath(os.path.join(root, body.run_id))
            if os.path.dirname(cwd) != root or not os.path.isdir(cwd):
                raise HTTPException(
                    status_code=404, detail="run workspace not found on this volume"
                )
            try:
                result = _run_bounded_check(
                    body.argv,
                    cwd,
                    timeout_seconds=body.timeout_seconds,
                    output_cap=body.output_cap,
                )
            except OSError as exc:
                return {
                    "ok": False,
                    "exit_code": None,
                    "timed_out": False,
                    "output": f"could not start the check command: {exc.strerror or 'error'}",
                    "output_truncated": False,
                }
            return {
                "ok": result.exit_code == 0,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "output": result.output.decode("utf-8", "replace"),
                "output_truncated": result.truncated,
            }

    @app.post("/agent/runs/{run_id}/execute")
    def execute_run(
        run_id: RunIdPath, body: ProfileBody, _: None = Depends(require_internal_token)
    ) -> dict[str, Any]:
        attempt = body.execution_attempt()
        callback_client = client().for_execution(attempt)
        profile = body.to_profile()
        try:
            store, run = store_and_run(run_id, callback_client)
            final = engine.execute_run(
                store,
                registry,
                run,
                profile=profile,
                on_run_end=on_run_end(profile, callback_client),
            )
        except HostError as exc:
            raise _host_http_error(exc) from exc
        except StepOutcomeUnknown as exc:
            raise _outcome_unknown_http_error(exc) from exc
        except engine.RunningStepNotPermitted as exc:
            raise _not_permitted_http_error(exc) from exc
        return {"run_id": str(final.id), "status": final.status}

    @app.post("/agent/runs/{run_id}/execute/stream")
    def execute_run_stream(
        run_id: RunIdPath, body: ProfileBody, _: None = Depends(require_internal_token)
    ) -> StreamingResponse:
        attempt = body.execution_attempt()
        callback_client = client().for_execution(attempt)
        try:
            store, run = store_and_run(run_id, callback_client)
        except HostError as exc:
            raise _host_http_error(exc) from exc
        profile = body.to_profile()

        def events() -> Iterator[engine.EngineEvent]:
            try:
                yield from engine.execute_run_events(
                    store,
                    registry,
                    run,
                    profile=profile,
                    on_run_end=on_run_end(profile, callback_client),
                )
            except (HostError, StepOutcomeUnknown, engine.RunningStepNotPermitted) as exc:
                # The drive ended without an authoritative stop (the host became unreachable,
                # refused a callback, a step outcome was not observed, or a step left running
                # lost its grant). The durable rows hold the truth; the stream just ends and the
                # host reconciles from its records.
                logger.warning("run %s drive ended early: %s", run_id, bounded_detail(str(exc)))

        async def sse() -> AsyncIterator[str]:
            async for item in _bridge(events, keepalive_seconds=_keepalive_seconds()):
                if item is None:
                    yield KEEPALIVE_COMMENT
                else:
                    _kind, payload = item
                    yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


async def _bridge[T](
    generator_factory: Callable[[], Iterator[T]], *, keepalive_seconds: float
) -> AsyncIterator[T | None]:
    """Run a sync generator in a worker thread, yielding its items to the event loop and
    ``None`` (a keepalive) whenever it stays quiet past the interval.

    The worker owns the drive. When the observer goes away (a client disconnect, or the event
    loop closing), the worker stops publishing but keeps draining the generator, so the run
    reaches its next durable stop instead of being abandoned mid-step. A process exit still
    stops the daemon worker; the host's lease/reclaim path owns that case."""
    queue: asyncio.Queue[tuple[T] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    observer_gone = threading.Event()

    def publish(entry: tuple[T] | None) -> None:
        if observer_gone.is_set():
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, entry)
        except RuntimeError:  # the event loop closed: nobody is observing any more
            observer_gone.set()

    def work() -> None:
        try:
            for item in generator_factory():
                publish((item,))
        except Exception:
            logger.exception("standalone run drive failed unexpectedly")
        finally:
            publish(None)

    threading.Thread(target=work, daemon=True, name="slimx-agent-run").start()
    try:
        while True:
            try:
                entry = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
            except TimeoutError:
                yield None
                continue
            if entry is None:
                return
            yield entry[0]
    finally:
        observer_gone.set()


def _run_bounded_check(
    argv: list[str], cwd: str, *, timeout_seconds: float, output_cap: int
) -> CheckResult:
    """Run ``argv`` with mechanical bounds; this is NOT an isolation boundary.

    Enforced: no shell, a scrubbed environment (``PATH`` and ``HOME`` only), no stdin, a pinned
    working directory, a wall-clock timeout, output capture bounded to ``output_cap`` bytes while
    reading (excess output is drained and discarded, never buffered), and SIGKILL of the whole
    process group on every exit path. Not enforced: filesystem, network, CPU, or memory
    containment, and a descendant that starts its own session escapes the group kill. The
    working directory is the run's mutable workspace, not an exact snapshot.
    """
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp"}
    deadline = time.monotonic() + timeout_seconds
    # argv is host-allowlisted and never shell-expanded.
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        start_new_session=True,
    )
    stdout = process.stdout
    if stdout is None:  # pragma: no cover - stdout=PIPE always provides a pipe
        raise OSError("check process has no output pipe")
    retained = bytearray()
    truncated = False
    timed_out = False
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                if not selector.select(timeout=remaining):
                    continue
                chunk = os.read(stdout.fileno(), _READ_CHUNK_BYTES)
                if not chunk:
                    break  # every writer, descendants included, closed the pipe
                room = output_cap - len(retained)
                if room > 0:
                    retained += chunk[:room]
                if len(chunk) > max(room, 0):
                    truncated = True
        if not timed_out:
            timed_out = not _group_leader_exited_by(process.pid, deadline)
    finally:
        # The leader is not reaped yet, so its process-group id cannot have been reused.
        _kill_process_group(process.pid)
        stdout.close()
        returncode = process.wait()
    return CheckResult(
        exit_code=None if timed_out else returncode,
        timed_out=timed_out,
        output=bytes(retained),
        truncated=truncated,
    )


def _group_leader_exited_by(pid: int, deadline: float) -> bool:
    """Wait, without reaping, until the leader exits or the deadline passes."""
    while True:
        if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _kill_process_group(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


app = create_app()
