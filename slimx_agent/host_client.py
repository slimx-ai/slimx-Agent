"""HostClient: the standalone service's ONE path back to its host.

The engine loop runs in the SlimX-Agent container; everything host-owned — persistence,
step-tool execution, model transport, provider/egress policy, the run-end epilogue — stays
behind the host's internal callback API (``/internal/agent-host/*`` in ControlRoom). This
client is that boundary: :class:`~slimx_agent.http_store.HttpRunStore` drives store
operations through it, the remote tool registry invokes steps through it, and it is the
opaque ``handler_context`` handed to those handlers.

Auth is a single shared bearer token (``SLIMX_AGENT_INTERNAL_TOKEN``) — the same value
authenticates the host to the service and the service back to the host.

httpx is imported lazily so the core package (contracts/engine/planning) stays importable
without the ``service`` extra installed.
"""

from __future__ import annotations

from typing import Any

from slimx_agent.runtime import RunProfile
from slimx_agent.store import UNSET

# Step invocation legitimately takes minutes (local models, sub-agent joins); reads and
# state writes should not. Connects fail fast either way.
CONNECT_TIMEOUT_SECONDS = 10.0
STORE_TIMEOUT_SECONDS = 60.0
INVOKE_TIMEOUT_SECONDS = 3600.0

_BASE_PATH = "/internal/agent-host"


class HostError(RuntimeError):
    """A host callback failed (non-2xx that isn't part of a method's contract)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"agent host error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def profile_wire(profile: Any) -> dict[str, Any]:
    """The provider/model/base_url wire shape for a duck-typed profile object."""
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
        client: Any | None = None,
    ) -> None:
        if client is not None:
            # An injected httpx-compatible client (tests use the host app's TestClient),
            # which owns its own timeout policy — per-request timeouts are skipped for it.
            self._client = client
            self._per_request_timeouts = False
            if token:
                self._client.headers["Authorization"] = f"Bearer {token}"
            return
        if not base_url:
            raise ValueError("HostClient needs a base_url (SLIMX_AGENT_HOST_URL) or a client")
        import httpx

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=httpx.Timeout(STORE_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS),
        )
        self._per_request_timeouts = True

    # --- plumbing -----------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
        allow_404: bool = False,
    ) -> Any | None:
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = json
        if params is not None:
            kwargs["params"] = params
        if timeout is not None and self._per_request_timeouts:
            kwargs["timeout"] = timeout
        response = self._client.request(method, f"{_BASE_PATH}{path}", **kwargs)
        if response.status_code == 404 and allow_404:
            return None
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail") or response.text
            except ValueError:
                detail = response.text
            raise HostError(response.status_code, str(detail))
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    # --- store operations (see slimx_agent.store.RunStore) --------------------------

    def get_run(self, run_id: Any) -> dict[str, Any] | None:
        return self._request("GET", f"/runs/{run_id}", allow_404=True)

    def get_steps(self, run_id: Any) -> list[dict[str, Any]]:
        return self._request("GET", f"/runs/{run_id}/steps") or []

    def get_step(self, step_id: Any) -> dict[str, Any] | None:
        return self._request("GET", f"/steps/{step_id}", allow_404=True)

    def set_run_status(self, run_id: Any, status: str) -> dict[str, Any]:
        return self._request("POST", f"/runs/{run_id}/status", json={"status": status})

    def set_step_state(
        self,
        step_id: Any,
        status: str,
        *,
        error: Any = UNSET,
        output_refs: Any = UNSET,
    ) -> dict[str, Any]:
        # UNSET travels as an explicit presence flag: set=False leaves the field untouched
        # host-side; set=True replaces it (value None clears).
        body: dict[str, Any] = {
            "status": status,
            "error_set": error is not UNSET,
            "output_refs_set": output_refs is not UNSET,
        }
        if error is not UNSET:
            body["error"] = error
        if output_refs is not UNSET:
            body["output_refs"] = output_refs
        return self._request("POST", f"/steps/{step_id}/state", json=body)

    def append_event(
        self,
        run_id: Any,
        type: str,
        *,
        step_id: Any | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runs/{run_id}/events",
            json={
                "type": type,
                "step_id": str(step_id) if step_id is not None else None,
                "payload": payload,
            },
        )

    def next_sequence(self, run_id: Any) -> int:
        data = self._request("GET", f"/runs/{run_id}/events/next-sequence") or {}
        return int(data["next_sequence"])

    def events_after(self, run_id: Any, after_sequence: int) -> list[dict[str, Any]]:
        return (
            self._request("GET", f"/runs/{run_id}/events", params={"after": after_sequence}) or []
        )

    # --- host capabilities ----------------------------------------------------------

    def invoke_step(self, run_id: Any, step_id: Any, profile: Any) -> dict[str, Any]:
        """Execute one step's tool ON THE HOST. Returns the invocation outcome envelope:
        ``{"outcome": "completed", "output_refs": {...}}`` /
        ``{"outcome": "skipped", "reason": ...}`` / ``{"outcome": "failed", "error": ...}``."""
        return (
            self._request(
                "POST",
                f"/runs/{run_id}/steps/{step_id}/invoke",
                json={"profile": profile_wire(profile)},
                timeout=INVOKE_TIMEOUT_SECONDS,
            )
            or {}
        )

    def run_end(self, run_id: Any, status: str, profile: Any) -> None:
        """Fire the host's run-end epilogue (e.g. bounded auto-iterate). May run long when
        the epilogue plans+executes a follow-up run host-side."""
        self._request(
            "POST",
            f"/runs/{run_id}/run-end",
            json={"status": status, "profile": profile_wire(profile)},
            timeout=INVOKE_TIMEOUT_SECONDS,
        )


def profile_from_wire(data: dict[str, Any]) -> RunProfile:
    return RunProfile(
        provider=str(data["provider"]),
        model=str(data["model"]),
        base_url=data.get("base_url"),
    )
