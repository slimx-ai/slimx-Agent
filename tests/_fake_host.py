"""Deterministic fakes for the standalone-service boundary tests.

``FakeHost`` is an in-memory FastAPI app implementing the internal agent-host callback API (the
ControlRoom wire contract). ``HostClient`` talks to it through FastAPI's TestClient, so
serialization, UNSET presence flags, outcome envelopes, headers, and error statuses are all
exercised over the real wire — with no network, database, or model provider.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from slimx_agent import engine
from slimx_agent.host_client import LEASE_GENERATION_HEADER, HostClient
from slimx_agent.http_store import HttpRunStore
from slimx_agent.http_tools import build_remote_registry
from slimx_agent.runtime import RunProfile

PROFILE = RunProfile("ollama", "llama3.2", None)


class FakeHost:
    """In-memory run/step/event tables + per-step-type invocation behaviors."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.steps: dict[str, dict[str, Any]] = {}
        self.step_order: dict[str, list[str]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.behaviors: dict[str, Any] = {}
        self.run_end_calls: list[tuple[str, str]] = []
        self.run_end_profiles: list[dict[str, Any]] = []
        self.invoked_profiles: list[dict[str, Any]] = []
        self.callback_requests: list[tuple[str, dict[str, str]]] = []
        self.callback_log: list[tuple[str, str]] = []
        self.state_bodies: list[tuple[str, dict[str, Any]]] = []
        self._callback_lock = threading.Lock()
        self.invoke_barrier: threading.Barrier | None = None
        self.invoke_delay_seconds = 0.0
        # When set, a callback carrying any other lease generation is refused with 409 — the
        # host's stale-attempt fence.
        self.current_lease_generation: int | None = None

    def add_run(self, run_id: str, **overrides: Any) -> dict[str, Any]:
        run = {
            "id": run_id,
            "status": "planned",
            "approval_policy": "auto_complete",
            "auto_approve": True,
            "allowed_tools_json": None,
            **overrides,
        }
        self.runs[run_id] = run
        self.step_order[run_id] = []
        self.events[run_id] = []
        return run

    def add_step(self, run_id: str, step_id: str, step_type: str, **overrides: Any) -> dict:
        step = {
            "id": step_id,
            "agent_run_id": run_id,
            "type": step_type,
            "title": f"{step_type} step",
            "status": "pending",
            "requires_approval": False,
            "error": None,
            "output_refs_json": None,
            **overrides,
        }
        self.steps[step_id] = step
        self.step_order[run_id].append(step_id)
        return step

    def event_types(self, run_id: str) -> list[str]:
        return [event["type"] for event in self.events[run_id]]

    def app(self) -> FastAPI:
        api = FastAPI()
        host = self

        @api.middleware("http")
        async def record_and_fence(request: Request, call_next):
            if request.url.path.startswith("/internal/agent-host/"):
                headers = {key.lower(): value for key, value in request.headers.items()}
                with host._callback_lock:
                    host.callback_requests.append((request.url.path, headers))
                    host.callback_log.append((request.method, request.url.path))
                expected = host.current_lease_generation
                if expected is not None and headers.get(LEASE_GENERATION_HEADER.lower()) != str(
                    expected
                ):
                    return JSONResponse(
                        status_code=409,
                        content={"detail": "Agent execution attempt is no longer valid"},
                    )
            return await call_next(request)

        @api.get("/internal/agent-host/runs/{run_id}")
        def get_run(run_id: str) -> dict:
            run = host.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="not found")
            return run

        @api.get("/internal/agent-host/runs/{run_id}/steps")
        def get_steps(run_id: str) -> list[dict]:
            return [host.steps[sid] for sid in host.step_order.get(run_id, [])]

        @api.get("/internal/agent-host/steps/{step_id}")
        def get_step(step_id: str) -> dict:
            step = host.steps.get(step_id)
            if step is None:
                raise HTTPException(status_code=404, detail="not found")
            return step

        @api.post("/internal/agent-host/runs/{run_id}/status")
        def set_run_status(run_id: str, body: dict) -> dict:
            host.runs[run_id]["status"] = body["status"]
            return host.runs[run_id]

        @api.post("/internal/agent-host/steps/{step_id}/state")
        def set_step_state(step_id: str, body: dict) -> dict:
            with host._callback_lock:
                host.state_bodies.append((step_id, dict(body)))
            step = host.steps[step_id]
            step["status"] = body["status"]
            if body.get("error_set"):
                step["error"] = body.get("error")
            if body.get("output_refs_set"):
                step["output_refs_json"] = body.get("output_refs")
            return step

        @api.post("/internal/agent-host/runs/{run_id}/events")
        def append_event(run_id: str, body: dict) -> dict:
            events = host.events[run_id]
            payload = {
                "id": f"evt-{len(events) + 1}",
                "agent_run_id": run_id,
                "agent_step_id": body.get("step_id"),
                "sequence": len(events) + 1,
                "type": body["type"],
                "payload_json": body.get("payload"),
                "created_at": None,
            }
            events.append(payload)
            return payload

        @api.get("/internal/agent-host/runs/{run_id}/events/next-sequence")
        def next_sequence(run_id: str) -> dict:
            return {"next_sequence": len(host.events[run_id]) + 1}

        @api.get("/internal/agent-host/runs/{run_id}/events")
        def events_after(run_id: str, after: int = 0) -> list[dict]:
            return [e for e in host.events[run_id] if e["sequence"] > after]

        @api.post("/internal/agent-host/runs/{run_id}/steps/{step_id}/invoke")
        def invoke(run_id: str, step_id: str, body: dict) -> Any:
            if host.invoke_barrier is not None:
                host.invoke_barrier.wait(timeout=10)
            if host.invoke_delay_seconds:
                time.sleep(host.invoke_delay_seconds)
            host.invoked_profiles.append(body["profile"])
            step = host.steps[step_id]
            behavior = host.behaviors.get(
                step["type"], {"outcome": "completed", "output_refs": None}
            )
            return behavior() if callable(behavior) else behavior

        @api.post("/internal/agent-host/runs/{run_id}/run-end", status_code=204)
        def run_end(run_id: str, body: dict) -> None:
            host.run_end_calls.append((run_id, body["status"]))
            host.run_end_profiles.append(body["profile"])

        return api

    def transport(self) -> TestClient:
        return TestClient(self.app())

    def boundary(
        self, *, token: str | None = None, client: Any | None = None
    ) -> tuple[HttpRunStore, HostClient]:
        host_client = HostClient(client=client or self.transport(), token=token)
        return HttpRunStore(host_client), host_client

    def callbacks_after(self, path_suffix: str) -> list[tuple[str, str]]:
        """Every callback recorded after the first one whose path ends with ``path_suffix``."""
        for index, (_method, path) in enumerate(self.callback_log):
            if path.endswith(path_suffix):
                return self.callback_log[index + 1 :]
        raise AssertionError(f"no callback ending with {path_suffix!r}")


class LossyClient:
    """Wrap a TestClient and lose ONE matching callback at the transport level.

    ``when="after"`` forwards the request (the host applies it) and then raises a read timeout,
    so the client never observes the answer — the ambiguous case. ``when="before"`` refuses
    the connection without reaching the host. Only the ``nth`` matching request is lost;
    ``matching_requests`` proves nothing replays it.
    """

    def __init__(
        self, inner: Any, *, method: str, path_suffix: str, when: str = "after", nth: int = 1
    ) -> None:
        self.inner = inner
        self.method = method
        self.path_suffix = path_suffix
        self.when = when
        self.nth = nth
        self.matching_requests = 0

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        if method == self.method and url.endswith(self.path_suffix):
            self.matching_requests += 1
            if self.matching_requests == self.nth:
                if self.when == "after":
                    self.inner.request(method, url, **kwargs)
                    raise httpx.ReadTimeout("simulated: the response was lost")
                raise httpx.ConnectError("simulated: connection refused")
        return self.inner.request(method, url, **kwargs)


def execution_body(seed: int = 1, **profile: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "provider": "ollama",
        "model": "llama3.2",
        "base_url": None,
        "lease_job_id": str(uuid.UUID(int=seed)),
        "lease_token": str(uuid.UUID(int=seed + 10_000)),
        "lease_generation": seed,
    }
    body.update(profile)
    return body


def expected_attempt_headers(body: dict[str, Any]) -> dict[str, str]:
    return {
        "x-slimx-agent-lease-job-id": body["lease_job_id"],
        "x-slimx-agent-lease-token": body["lease_token"],
        "x-slimx-agent-lease-generation": str(body["lease_generation"]),
    }


def assert_attempt_headers(headers: dict[str, str], body: dict[str, Any]) -> None:
    for key, value in expected_attempt_headers(body).items():
        assert headers[key] == value


def drive(host: FakeHost, run_id: str, *, client: Any | None = None):
    store, _client = host.boundary(client=client)
    run = store.get_run(run_id)
    assert run is not None
    final = engine.execute_run(
        store,
        build_remote_registry(),
        run,
        profile=PROFILE,
        on_run_end=lambda r, status: host.run_end_calls.append((str(r.id), f"hook:{status}")),
    )
    return store, final
