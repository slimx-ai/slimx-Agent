"""HostClient: the standalone service's ONE path back to its host.

The engine loop runs in the SlimX-Agent container; everything host-owned — persistence,
step-tool execution, model transport, provider/egress policy, the run-end epilogue — stays
behind the host's internal callback API (``/internal/agent-host/*`` in ControlRoom). This
client is that boundary: :class:`~slimx_agent.http_store.HttpRunStore` drives store
operations through it, the remote tool registry invokes steps through it, and it is the
opaque ``handler_context`` handed to those handlers.

Auth is a single shared bearer token (``SLIMX_AGENT_INTERNAL_TOKEN``) — the same value
authenticates the host to the service and the service back to the host. An
:class:`ExecutionAttempt` adds the per-execution lease headers to every callback.

Failure vocabulary (all :class:`HostError`): a non-2xx answer keeps its status code; a
transport failure with no observed response is :class:`HostUnavailable`; an answer outside the
documented wire shape is :class:`HostProtocolError`. The client NEVER retries a callback: after
a transport failure a mutating callback may or may not have been applied, and only the host's
durable records can say which. Internal callbacks ignore ambient proxy and netrc settings.

httpx is imported lazily so the core package (contracts/engine/planning) stays importable
without the ``service`` extra installed.
"""

from __future__ import annotations

import json as _json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote

from slimx_agent.runtime import ProfileView, RunProfile
from slimx_agent.store import UNSET, EventPayload, HostId, OutputRefs, UnsetType

# Step invocation legitimately takes minutes (local models, sub-agent joins); reads and
# state writes should not. Connects fail fast either way.
CONNECT_TIMEOUT_SECONDS = 10.0
STORE_TIMEOUT_SECONDS = 60.0
INVOKE_TIMEOUT_SECONDS = 3600.0

_BASE_PATH = "/internal/agent-host"

# Host error details are relayed to logs and to the service's own caller; keep them bounded.
MAX_DETAIL_CHARS = 500

# An execution attempt is host authority, not ambient service state.  Keep the values on
# every callback request so the host can reject a stale worker at each persistence/tool edge.
# These names are part of the standalone host wire contract; see docs/service-contract.md.
LEASE_JOB_ID_HEADER = "X-SlimX-Agent-Lease-Job-Id"
LEASE_TOKEN_HEADER = "X-SlimX-Agent-Lease-Token"
LEASE_GENERATION_HEADER = "X-SlimX-Agent-Lease-Generation"


def bounded_detail(text: str, limit: int = MAX_DETAIL_CHARS) -> str:
    """``text`` clipped to ``limit`` characters; an ellipsis marks a cut."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class ExecutionAttempt:
    """The host-issued lease identifying exactly one standalone execution attempt."""

    job_id: str
    # The lease token is authority material: keep it out of reprs and logs.
    token: str = field(repr=False)
    generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id.strip():
            raise ValueError("Execution attempt job id must not be empty")
        if not isinstance(self.token, str) or not self.token.strip():
            raise ValueError("Execution attempt token must not be empty")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("Execution attempt generation must be a non-negative integer")

    def headers(self) -> dict[str, str]:
        return {
            LEASE_JOB_ID_HEADER: self.job_id,
            LEASE_TOKEN_HEADER: self.token,
            LEASE_GENERATION_HEADER: str(self.generation),
        }


class HostError(RuntimeError):
    """A host callback failed: a non-2xx answer that isn't part of a method's contract."""

    def __init__(self, status_code: int, detail: str) -> None:
        detail = bounded_detail(detail)
        super().__init__(f"agent host error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class HostUnavailable(HostError):
    """No response was observed (connect/read/write failure or timeout).

    ``status_code`` is a synthetic 503. For a mutating callback the host may or may not have
    applied the request; it is never replayed automatically.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(503, f"host unavailable: {detail}")


class HostProtocolError(HostError):
    """The host answered, but not in the documented wire shape (synthetic 502)."""

    def __init__(self, detail: str) -> None:
        super().__init__(502, f"malformed host response: {detail}")


class HttpResponse(Protocol):
    """The response surface the client reads (satisfied by ``httpx.Response``)."""

    @property
    def status_code(self) -> int: ...

    @property
    def content(self) -> bytes: ...

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...


class HttpClient(Protocol):
    """The request surface the client uses (satisfied by ``httpx.Client`` and FastAPI's
    ``TestClient``). Transport keyword values are httpx's own open types."""

    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = ...,
        params: Any = ...,
        headers: Any = ...,
        timeout: Any = ...,
    ) -> HttpResponse: ...


class _NotFound:
    """Marker for an allowed 404 — distinct from a JSON ``null`` body."""


_NOT_FOUND = _NotFound()


def profile_wire(profile: ProfileView) -> dict[str, str | None]:
    """The provider/model/base_url wire shape. Values are forwarded exactly, never defaulted:
    the host compares them with its own authoritative routing at every callback."""
    return {
        "provider": profile.provider,
        "model": profile.model,
        "base_url": getattr(profile, "base_url", None),
    }


class HostClient:
    """Typed client for the host's internal agent-host callback API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        client: HttpClient | None = None,
    ) -> None:
        # Request headers are owned by this wrapper, never installed on a shared client.  A
        # per-execution wrapper can therefore share the connection pool without one run's
        # lease becoming another concurrent run's ambient default.
        self._request_headers: dict[str, str] = (
            {"Authorization": f"Bearer {token}"} if token else {}
        )
        if client is not None:
            # An injected httpx-compatible client (tests use the host app's TestClient),
            # which owns its own timeout policy — per-request timeouts are skipped for it.
            self._client: HttpClient = client
            self._per_request_timeouts = False
            return
        if not base_url:
            raise ValueError("HostClient needs a base_url (SLIMX_AGENT_HOST_URL) or a client")
        import httpx

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(STORE_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
            # Service-to-host callbacks carry the bearer token and lease: never route them
            # through ambient proxy environment variables or netrc credentials.
            trust_env=False,
        )
        self._per_request_timeouts = True

    def for_execution(self, attempt: ExecutionAttempt) -> HostClient:
        """Return an attempt-bound wrapper over this client's transport/connection pool.

        The base client and the injected/shared HTTP client's default headers are untouched.
        Attempt headers are copied into the derived wrapper and then supplied explicitly on
        every request, which keeps concurrent executions isolated.
        """
        derived = HostClient.__new__(HostClient)
        derived._client = self._client
        derived._per_request_timeouts = self._per_request_timeouts
        derived._request_headers = {**self._request_headers, **attempt.headers()}
        return derived

    # --- plumbing -----------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, object] | None = None,
        params: Mapping[str, object] | None = None,
        timeout: float | None = None,
        allow_404: bool = False,
    ) -> object:
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = dict(json)
        if params is not None:
            kwargs["params"] = dict(params)
        if timeout is not None and self._per_request_timeouts:
            kwargs["timeout"] = timeout
        if self._request_headers:
            # Pass a fresh mapping so neither httpx nor an injected test/client adapter can
            # mutate the wrapper's immutable-by-convention request context.
            kwargs["headers"] = dict(self._request_headers)
        try:
            response = self._client.request(method, f"{_BASE_PATH}{path}", **kwargs)
        except Exception as exc:
            if not _is_transport_error(exc):
                raise
            # No response was observed. A mutating callback may or may not have been applied;
            # it is never replayed here — the host's durable records decide.
            raise HostUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if response.status_code == 404 and allow_404:
            return _NOT_FOUND
        if response.status_code >= 400:
            raise HostError(response.status_code, _error_detail(response))
        if response.status_code == 204 or not response.content:
            return None
        try:
            body: object = response.json()
        except ValueError as exc:
            raise HostProtocolError("the response body is not JSON") from exc
        return body

    # --- store operations (see slimx_agent.store.RunStore) --------------------------

    def get_run(self, run_id: HostId) -> dict[str, Any] | None:
        data = self._request("GET", f"/runs/{_segment(run_id)}", allow_404=True)
        if isinstance(data, _NotFound):
            return None
        return _expect_object(data, "run snapshot")

    def get_steps(self, run_id: HostId) -> list[dict[str, Any]]:
        data = self._request("GET", f"/runs/{_segment(run_id)}/steps")
        return _expect_object_list(data, "step list")

    def get_step(self, step_id: HostId) -> dict[str, Any] | None:
        data = self._request("GET", f"/steps/{_segment(step_id)}", allow_404=True)
        if isinstance(data, _NotFound):
            return None
        return _expect_object(data, "step snapshot")

    def set_run_status(self, run_id: HostId, status: str) -> dict[str, Any]:
        data = self._request("POST", f"/runs/{_segment(run_id)}/status", json={"status": status})
        return _expect_object(data, "run snapshot")

    def set_step_state(
        self,
        step_id: HostId,
        status: str,
        *,
        error: str | None | UnsetType = UNSET,
        output_refs: OutputRefs | None | UnsetType = UNSET,
    ) -> dict[str, Any]:
        # UNSET travels as an explicit presence flag: set=False leaves the field untouched
        # host-side; set=True replaces it (value None clears).
        body: dict[str, object] = {
            "status": status,
            "error_set": error is not UNSET,
            "output_refs_set": output_refs is not UNSET,
        }
        if not isinstance(error, UnsetType):
            body["error"] = error
        if not isinstance(output_refs, UnsetType):
            body["output_refs"] = output_refs
        data = self._request("POST", f"/steps/{_segment(step_id)}/state", json=body)
        return _expect_object(data, "step snapshot")

    def append_event(
        self,
        run_id: HostId,
        type: str,
        *,
        step_id: HostId | None = None,
        payload: dict[str, Any] | None = None,
    ) -> EventPayload:
        data = self._request(
            "POST",
            f"/runs/{_segment(run_id)}/events",
            json={
                "type": type,
                "step_id": str(step_id) if step_id is not None else None,
                "payload": payload,
            },
        )
        return _expect_event(data)

    def next_sequence(self, run_id: HostId) -> int:
        data = _expect_object(
            self._request("GET", f"/runs/{_segment(run_id)}/events/next-sequence"),
            "next-sequence answer",
        )
        value = data.get("next_sequence")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HostProtocolError("next_sequence is not a non-negative integer")
        return value

    def events_after(self, run_id: HostId, after_sequence: int) -> list[EventPayload]:
        data = self._request(
            "GET", f"/runs/{_segment(run_id)}/events", params={"after": after_sequence}
        )
        return [_expect_event(item) for item in _expect_object_list(data, "event list")]

    # --- host capabilities ----------------------------------------------------------

    def invoke_step(self, run_id: HostId, step_id: HostId, profile: ProfileView) -> dict[str, Any]:
        """Execute one step's tool ON THE HOST. Returns the invocation outcome envelope:
        ``{"outcome": "completed", "output_refs": {...}}`` / ``{"outcome": "skipped",
        "reason": ...}`` / ``{"outcome": "prepared", "reason": ...}`` / ``{"outcome": "failed",
        "error": ...}``. Envelope semantics are interpreted by ``http_tools``."""
        wire = profile_wire(profile)
        data = self._request(
            "POST",
            f"/runs/{_segment(run_id)}/steps/{_segment(step_id)}/invoke",
            json={"profile": wire},
            timeout=INVOKE_TIMEOUT_SECONDS,
        )
        return _expect_object(data, "invocation outcome")

    def run_end(self, run_id: HostId, status: str, profile: ProfileView) -> None:
        """Fire the host's run-end epilogue (e.g. bounded auto-iterate). May run long when
        the epilogue plans+executes a follow-up run host-side."""
        self._request(
            "POST",
            f"/runs/{_segment(run_id)}/run-end",
            json={"status": status, "profile": profile_wire(profile)},
            timeout=INVOKE_TIMEOUT_SECONDS,
        )


def profile_from_wire(data: Mapping[str, object]) -> RunProfile:
    """Parse the provider/model/base_url wire shape, refusing anything but exact strings."""
    provider = data.get("provider")
    model = data.get("model")
    base_url = data.get("base_url")
    if not isinstance(provider, str) or not provider:
        raise ValueError("profile provider must be a non-empty string")
    if not isinstance(model, str) or not model:
        raise ValueError("profile model must be a non-empty string")
    if base_url is not None and not isinstance(base_url, str):
        raise ValueError("profile base_url must be a string or null")
    return RunProfile(provider=provider, model=model, base_url=base_url)


def _segment(value: HostId) -> str:
    """One URL path segment for a host identity: quoted, so an identity can never add path."""
    return quote(str(value), safe="")


def _expect_object(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HostProtocolError(f"the {what} is not a JSON object")
    return value


def _expect_object_list(value: object, what: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise HostProtocolError(f"the {what} is not a JSON array of objects")
    return value


def _expect_event(value: object) -> EventPayload:
    event = _expect_object(value, "event")
    sequence = event.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise HostProtocolError("an event has no integer sequence")
    return event


def _error_detail(response: HttpResponse) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or "no detail"
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        if detail is not None:
            return _json.dumps(detail, default=str)
    return response.text or "no detail"


def _is_transport_error(exc: BaseException) -> bool:
    try:
        import httpx
    except ImportError:  # pragma: no cover - an injected client without httpx installed
        return False
    return isinstance(exc, httpx.HTTPError)
