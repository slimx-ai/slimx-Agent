"""The standalone SlimX-Agent service: the engine loop in its own container.

Runs :mod:`slimx_agent.engine` over :class:`~slimx_agent.http_store.HttpRunStore` and the
remote tool registry — every store operation, step tool, and the run-end epilogue call back
to the HOST's internal agent-host API. The container therefore needs NO database, NO
provider credentials, and NO host code: its entire world is ``SLIMX_AGENT_HOST_URL`` plus
the shared ``SLIMX_AGENT_INTERNAL_TOKEN`` (one value authenticates both directions).

Surface (mirrors the host's embedded service app where they overlap):
- ``GET /health`` — liveness + ``auth_enabled`` + ``mode: "standalone"`` so the host's deep
  health can flag token/topology drift.
- ``POST /agent/runs/{run_id}/execute`` — drive a run to its next stop (blocking).
- ``POST /agent/runs/{run_id}/execute/stream`` — same, streaming each durable event as an
  SSE ``data:`` line with ``:``-comment keepalives while a step is quiet.

Planning and the system map stay host-side on purpose: they are host-shaped (context
manifests, provider profile resolution, extraction services). The loop is the portable part.

Requires the ``service`` extra (fastapi/uvicorn/httpx).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from slimx_agent import __version__, engine
from slimx_agent.host_client import HostClient, HostError
from slimx_agent.http_store import HttpRunStore
from slimx_agent.http_tools import build_remote_registry
from slimx_agent.runtime import RunProfile

logger = logging.getLogger("slimx_agent.service")

KEEPALIVE_COMMENT = ": keepalive\n\n"


class ProfileBody(BaseModel):
    """The host-resolved execution profile (egress already enforced host-side)."""

    provider: str
    model: str
    base_url: str | None = None

    def to_profile(self) -> RunProfile:
        return RunProfile(self.provider, self.model, self.base_url)


def _keepalive_seconds() -> float:
    return float(os.environ.get("SLIMX_AGENT_KEEPALIVE_SECONDS", "15"))


def create_app(host_client: HostClient | None = None) -> FastAPI:
    app = FastAPI(title="SlimX-Agent", version=__version__)
    registry = build_remote_registry()
    state: dict[str, HostClient | None] = {"client": host_client}

    def client() -> HostClient:
        # Built lazily so importing the module (and /health) never requires the host URL.
        if state["client"] is None:
            state["client"] = HostClient(
                os.environ.get("SLIMX_AGENT_HOST_URL"),
                token=os.environ.get("SLIMX_AGENT_INTERNAL_TOKEN") or None,
            )
        return state["client"]

    def require_internal_token(authorization: str | None = Header(default=None)) -> None:
        token = os.environ.get("SLIMX_AGENT_INTERNAL_TOKEN") or ""
        if not token:
            return
        expected = f"Bearer {token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid internal service token")

    def store_and_run(run_id: str) -> tuple[HttpRunStore, Any]:
        store = HttpRunStore(client())
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Agent run not found")
        return store, run

    def on_run_end(profile: RunProfile):
        def hook(run: Any, status: str) -> None:
            # The epilogue is the host's concern (and flag-gated there); a callback failure
            # must never jeopardize the finished run's own result.
            try:
                client().run_end(run.id, status, profile)
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

    @app.post("/internal/run-check")
    def run_check(
        body: dict,
        _: None = Depends(require_internal_token),
    ) -> dict[str, Any]:
        """Isolated check execution (Phase 5 hardening): run ONE host-allowlisted check command
        inside THIS container — which holds no DB, no credentials, no ControlRoom code — instead
        of the api container. The HOST still owns the allowlist decision (it only sends commands
        it already validated); this endpoint re-enforces the mechanical bounds: shell=False,
        scrubbed env, pinned cwd under the shared workspace volume, timeout, output cap. It never
        interprets or expands the command."""
        import os
        import subprocess

        argv = body.get("argv")
        run_id = str(body.get("run_id") or "").strip()
        timeout = min(float(body.get("timeout_seconds") or 120.0), 600.0)
        output_cap = min(int(body.get("output_cap") or 20_000), 100_000)
        workspace_root = os.environ.get("AGENT_WORKSPACE_ROOT", "/workspaces")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise HTTPException(status_code=422, detail="argv must be a non-empty list of strings")
        if not run_id or "/" in run_id or ".." in run_id:
            raise HTTPException(status_code=422, detail="run_id must be a plain identifier")
        cwd = os.path.join(workspace_root, run_id)
        if not os.path.isdir(cwd):
            raise HTTPException(status_code=404, detail="run workspace not found on this volume")
        scrubbed_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp"}
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=scrubbed_env,
                shell=False,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "exit_code": None, "timed_out": True, "output": ""}
        except FileNotFoundError as exc:
            return {"ok": False, "exit_code": None, "timed_out": False, "output": str(exc)[:500]}
        output = (completed.stdout + completed.stderr).decode("utf-8", "replace")[:output_cap]
        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "timed_out": False,
            "output": output,
        }

    @app.post("/agent/runs/{run_id}/execute")
    def execute_run(
        run_id: str, body: ProfileBody, _: None = Depends(require_internal_token)
    ) -> dict[str, Any]:
        store, run = store_and_run(run_id)
        profile = body.to_profile()
        try:
            final = engine.execute_run(
                store, registry, run, profile=profile, on_run_end=on_run_end(profile)
            )
        except HostError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"run_id": str(final.id), "status": final.status}

    @app.post("/agent/runs/{run_id}/execute/stream")
    def execute_run_stream(
        run_id: str, body: ProfileBody, _: None = Depends(require_internal_token)
    ) -> StreamingResponse:
        store, run = store_and_run(run_id)
        profile = body.to_profile()

        def events() -> Iterator[tuple[str, dict[str, Any]]]:
            try:
                yield from engine.execute_run_events(
                    store, registry, run, profile=profile, on_run_end=on_run_end(profile)
                )
            except HostError:
                # The host became unreachable mid-run: the durable rows hold the truth; the
                # stream just ends (the host client reconciles from the DB, as on any drop).
                logger.exception("host callback failed mid-stream for run %s", run_id)

        async def sse() -> AsyncIterator[str]:
            async for item in _bridge(events, keepalive_seconds=_keepalive_seconds()):
                if item is None:
                    yield KEEPALIVE_COMMENT
                else:
                    _kind, payload = item
                    yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


async def _bridge(
    generator_factory, *, keepalive_seconds: float
) -> AsyncIterator[tuple[str, dict[str, Any]] | None]:
    """Run a sync generator in a worker thread, yielding its items to the event loop and
    ``None`` (a keepalive) whenever it stays quiet past the interval."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    done = object()

    def work() -> None:
        try:
            for item in generator_factory():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=work, daemon=True, name="slimx-agent-run").start()
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
        except TimeoutError:
            yield None
            continue
        if item is done:
            return
        yield item


app = create_app()
