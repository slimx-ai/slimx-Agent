"""The standalone-service boundary: the real engine driving a fake host over the real wire.

The fake host is an in-memory FastAPI app implementing the internal agent-host callback API
(the ControlRoom contract); ``HostClient`` talks to it through FastAPI's TestClient, so
serialization, UNSET presence flags, and outcome envelopes are all exercised end to end.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from slimx_agent import contracts, engine
from slimx_agent.host_client import HostClient
from slimx_agent.http_store import HttpRunStore
from slimx_agent.http_tools import build_remote_registry


class FakeHost:
    """In-memory run/step/event tables + per-step-type invocation behaviors."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.steps: dict[str, dict[str, Any]] = {}
        self.step_order: dict[str, list[str]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.behaviors: dict[str, dict[str, Any]] = {}
        self.run_end_calls: list[tuple[str, str]] = []
        self.invoked_profiles: list[dict[str, Any]] = []

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

    def app(self) -> FastAPI:
        api = FastAPI()
        host = self

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
        def invoke(run_id: str, step_id: str, body: dict) -> dict:
            host.invoked_profiles.append(body["profile"])
            step = host.steps[step_id]
            return host.behaviors.get(step["type"], {"outcome": "completed", "output_refs": None})

        @api.post("/internal/agent-host/runs/{run_id}/run-end", status_code=204)
        def run_end(run_id: str, body: dict) -> None:
            host.run_end_calls.append((run_id, body["status"]))

        return api

    def boundary(self) -> tuple[HttpRunStore, Any]:
        client = HostClient(client=TestClient(self.app()))
        return HttpRunStore(client), client


PROFILE = type("P", (), {"provider": "ollama", "model": "llama3.2", "base_url": None})()


def _drive(host: FakeHost, run_id: str):
    store, _client = host.boundary()
    run = store.get_run(run_id)
    assert run is not None
    registry = build_remote_registry()
    final = engine.execute_run(
        store,
        registry,
        run,
        profile=PROFILE,
        on_run_end=lambda r, status: host.run_end_calls.append((str(r.id), f"hook:{status}")),
    )
    return store, final


def test_engine_completes_a_run_over_the_wire():
    host = FakeHost()
    host.add_run("r1")
    host.add_step("r1", "s1", "model_call")
    host.add_step("r1", "s2", "create_synthesis")
    host.behaviors["model_call"] = {
        "outcome": "completed",
        "output_refs": {"run_group_ids": ["g1"]},
    }

    _store, final = _drive(host, "r1")

    assert final.status == "completed"
    assert host.steps["s1"]["status"] == "completed"
    assert host.steps["s1"]["output_refs_json"] == {"run_group_ids": ["g1"]}
    assert host.steps["s2"]["status"] == "completed"
    types = [e["type"] for e in host.events["r1"]]
    assert types.count(contracts.STEP_STARTED) == 2
    assert types.count(contracts.STEP_COMPLETED) == 2
    assert types[-1] == contracts.RUN_COMPLETED
    assert ("r1", "hook:completed") in host.run_end_calls
    # The resolved profile travelled to every host invocation.
    assert host.invoked_profiles[0] == {
        "provider": "ollama",
        "model": "llama3.2",
        "base_url": None,
    }


def test_permission_gate_skips_ungranted_external_tool():
    host = FakeHost()
    host.add_run("r1", allowed_tools_json=None)  # legacy: nothing optional granted
    host.add_step("r1", "s1", "web_search")
    host.add_step("r1", "s2", "model_call")

    _store, final = _drive(host, "r1")

    assert final.status == "completed"
    assert host.steps["s1"]["status"] == "skipped"
    skipped = next(e for e in host.events["r1"] if e["type"] == contracts.STEP_SKIPPED)
    assert "not enabled for this run" in skipped["payload_json"]["reason"]
    # The gate never reached the host: web_search was not invoked.
    assert all(p == host.invoked_profiles[0] for p in host.invoked_profiles)
    assert host.steps["s2"]["status"] == "completed"


def test_hard_gate_stops_even_in_auto_complete():
    host = FakeHost()
    host.add_run("r1", allowed_tools_json=["mcp_tools"])
    host.add_step("r1", "s1", "mcp_call")

    _store, final = _drive(host, "r1")

    assert final.status == "awaiting_approval"
    assert host.steps["s1"]["status"] == "awaiting_approval"
    assert any(e["type"] == contracts.APPROVAL_REQUIRED for e in host.events["r1"])
    assert host.run_end_calls == []  # a gate stop is not a run end


def test_host_reported_skip_and_failure_map_to_engine_transitions():
    host = FakeHost()
    host.add_run("r1")
    host.add_step("r1", "s1", "rag_retrieve")
    host.behaviors["rag_retrieve"] = {"outcome": "skipped", "reason": "no documents indexed"}
    host.add_step("r1", "s2", "model_call")
    host.behaviors["model_call"] = {"outcome": "failed", "error": "provider unreachable"}

    _store, final = _drive(host, "r1")

    assert host.steps["s1"]["status"] == "skipped"
    assert final.status == "failed"
    assert host.steps["s2"]["status"] == "failed"
    assert host.steps["s2"]["error"] == "provider unreachable"
    assert ("r1", "hook:failed") in host.run_end_calls


def test_streamed_events_are_ordered_and_wire_shaped():
    host = FakeHost()
    host.add_run("r1")
    host.add_step("r1", "s1", "model_call")
    store, _client = host.boundary()
    run = store.get_run("r1")

    items = list(engine.execute_run_events(store, build_remote_registry(), run, profile=PROFILE))

    assert all(kind == "event" for kind, _ in items)
    sequences = [payload["sequence"] for _, payload in items]
    assert sequences == sorted(sequences)
    assert {payload["type"] for _, payload in items} >= {
        contracts.STEP_STARTED,
        contracts.STEP_COMPLETED,
        contracts.RUN_COMPLETED,
    }


def test_service_app_executes_and_reports_health(monkeypatch):
    from slimx_agent.service import create_app

    host = FakeHost()
    host.add_run("r1")
    host.add_step("r1", "s1", "model_call")
    monkeypatch.delenv("SLIMX_AGENT_INTERNAL_TOKEN", raising=False)
    service = TestClient(create_app(host_client=HostClient(client=TestClient(host.app()))))

    health = service.get("/health").json()
    assert health["mode"] == "standalone"
    assert health["auth_enabled"] is False

    body = {"provider": "ollama", "model": "llama3.2", "base_url": None}
    done = service.post("/agent/runs/r1/execute", json=body)
    assert done.status_code == 200
    assert done.json() == {"run_id": "r1", "status": "completed"}
    assert host.steps["s1"]["status"] == "completed"
    assert host.run_end_calls == [("r1", "completed")]

    assert service.post("/agent/runs/nope/execute", json=body).status_code == 404


def test_service_app_enforces_internal_token(monkeypatch):
    from slimx_agent.service import create_app

    host = FakeHost()
    host.add_run("r1")
    monkeypatch.setenv("SLIMX_AGENT_INTERNAL_TOKEN", "sekrit")
    service = TestClient(create_app(host_client=HostClient(client=TestClient(host.app()))))

    body = {"provider": "ollama", "model": "llama3.2", "base_url": None}
    assert service.post("/agent/runs/r1/execute", json=body).status_code == 401
    ok = service.post(
        "/agent/runs/r1/execute", json=body, headers={"Authorization": "Bearer sekrit"}
    )
    assert ok.status_code == 200
    assert service.get("/health").json()["auth_enabled"] is True


def test_service_stream_emits_sse_data_lines(monkeypatch):
    from slimx_agent.service import create_app

    host = FakeHost()
    host.add_run("r1")
    host.add_step("r1", "s1", "model_call")
    monkeypatch.delenv("SLIMX_AGENT_INTERNAL_TOKEN", raising=False)
    service = TestClient(create_app(host_client=HostClient(client=TestClient(host.app()))))

    body = {"provider": "ollama", "model": "llama3.2", "base_url": None}
    payloads = []
    with service.stream("POST", "/agent/runs/r1/execute/stream", json=body) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: ") :]))
    assert [p["sequence"] for p in payloads] == sorted(p["sequence"] for p in payloads)
    assert payloads[-1]["type"] == contracts.RUN_COMPLETED
    assert host.runs["r1"]["status"] == "completed"
