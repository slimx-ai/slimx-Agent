"""The standalone-service boundary: the real engine driving a fake host over the real wire.

The fake host (``tests/_fake_host.py``) is an in-memory FastAPI app implementing the internal
agent-host callback API (the ControlRoom contract); ``HostClient`` talks to it through FastAPI's
TestClient, so serialization, UNSET presence flags, and outcome envelopes are all exercised end
to end. ``tests/test_service_contract.py`` extends this with the negative-path matrix.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
from typing import Any

from _fake_host import PROFILE, FakeHost
from _fake_host import assert_attempt_headers as _assert_attempt_headers
from _fake_host import drive as _drive
from _fake_host import execution_body as _execution_body
from _fake_host import expected_attempt_headers as _expected_attempt_headers
from fastapi.testclient import TestClient

from slimx_agent import contracts, engine
from slimx_agent.host_client import HostClient
from slimx_agent.http_tools import build_remote_registry


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


def test_host_reported_preparation_reenters_the_approval_gate_without_false_terminal_event():
    host = FakeHost()
    host.add_run("r1", approval_policy="manual_review")
    host.add_step("r1", "s1", "model_call", status="approved", requires_approval=True)

    def prepared_behavior() -> dict[str, Any]:
        host.steps["s1"]["status"] = "awaiting_approval"
        host.runs["r1"]["status"] = "awaiting_approval"
        return {"outcome": "prepared", "reason": "candidate ready"}

    host.behaviors["model_call"] = prepared_behavior
    _store, final = _drive(host, "r1")

    assert final.status == "awaiting_approval"
    assert host.steps["s1"]["status"] == "awaiting_approval"
    types = [event["type"] for event in host.events["r1"]]
    assert contracts.APPROVAL_GRANTED not in types
    assert contracts.STEP_COMPLETED not in types
    assert contracts.STEP_FAILED not in types
    assert contracts.STEP_SKIPPED not in types


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
    host_transport = TestClient(host.app())
    service = TestClient(create_app(host_client=HostClient(client=host_transport)))

    health = service.get("/health").json()
    assert health["mode"] == "standalone"
    assert health["auth_enabled"] is False

    body = _execution_body()
    done = service.post("/agent/runs/r1/execute", json=body)
    assert done.status_code == 200
    assert done.json() == {"run_id": "r1", "status": "completed"}
    assert host.steps["s1"]["status"] == "completed"
    assert host.run_end_calls == [("r1", "completed")]

    # The attempt fence is present at every callback category, including the final tool edge.
    paths = [path for path, _headers in host.callback_requests]
    assert any(path.endswith("/steps/s1/state") for path in paths)
    assert any(path.endswith("/events") for path in paths)
    assert any(path.endswith("/steps/s1/invoke") for path in paths)
    assert any(path.endswith("/run-end") for path in paths)
    for _path, headers in host.callback_requests:
        _assert_attempt_headers(headers, body)
    # Deriving the execution client never installs lease authority on the shared transport.
    for key in _expected_attempt_headers(body):
        assert key not in {header.lower() for header in host_transport.headers}

    assert service.post("/agent/runs/nope/execute", json=body).status_code == 404


def test_service_execute_requires_a_complete_attempt_before_any_callback(monkeypatch):
    from slimx_agent.service import create_app

    host = FakeHost()
    host.add_run("r1")
    monkeypatch.delenv("SLIMX_AGENT_INTERNAL_TOKEN", raising=False)
    service = TestClient(create_app(host_client=HostClient(client=TestClient(host.app()))))

    profile_only = {"provider": "ollama", "model": "llama3.2", "base_url": None}
    missing = service.post("/agent/runs/r1/execute", json=profile_only)
    assert missing.status_code == 422
    assert missing.json()["detail"] == "Agent execution lease is required"

    partial = service.post(
        "/agent/runs/r1/execute",
        json={**profile_only, "lease_job_id": _execution_body()["lease_job_id"]},
    )
    assert partial.status_code == 422
    assert "Incomplete agent execution lease" in str(partial.json()["detail"])

    stream_missing = service.post("/agent/runs/r1/execute/stream", json=profile_only)
    assert stream_missing.status_code == 422
    assert host.callback_requests == []


def test_concurrent_service_executions_keep_attempt_headers_isolated(monkeypatch):
    from slimx_agent.service import create_app

    host = FakeHost()
    host.add_run("r1")
    host.add_step("r1", "step-r1", "model_call")
    host.add_run("r2")
    host.add_step("r2", "step-r2", "model_call")
    # Force both executions to overlap at the tool edge. A shared-header implementation
    # would make at least one run's later state/event/run-end callbacks carry the other lease.
    host.invoke_barrier = threading.Barrier(2)
    monkeypatch.delenv("SLIMX_AGENT_INTERNAL_TOKEN", raising=False)
    shared_transport = TestClient(host.app())
    service = TestClient(create_app(host_client=HostClient(client=shared_transport)))
    bodies = {"r1": _execution_body(101), "r2": _execution_body(202)}

    def execute(run_id: str):
        return service.post(f"/agent/runs/{run_id}/execute", json=bodies[run_id])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(execute, ("r1", "r2")))

    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json()["status"] for response in responses] == ["completed", "completed"]
    assert any(path.endswith("/steps/step-r1/invoke") for path, _ in host.callback_requests)
    assert any(path.endswith("/steps/step-r2/invoke") for path, _ in host.callback_requests)

    for path, headers in host.callback_requests:
        if "/runs/r1" in path or "/steps/step-r1" in path:
            _assert_attempt_headers(headers, bodies["r1"])
        elif "/runs/r2" in path or "/steps/step-r2" in path:
            _assert_attempt_headers(headers, bodies["r2"])
        else:  # Every callback in this test must be attributable to exactly one run.
            raise AssertionError(f"unexpected callback path {path}")

    for key in _expected_attempt_headers(bodies["r1"]):
        assert key not in {header.lower() for header in shared_transport.headers}


def test_service_app_enforces_internal_token(monkeypatch):
    from slimx_agent.service import create_app

    host = FakeHost()
    host.add_run("r1")
    monkeypatch.setenv("SLIMX_AGENT_INTERNAL_TOKEN", "sekrit")
    service = TestClient(create_app(host_client=HostClient(client=TestClient(host.app()))))

    body = _execution_body()
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

    body = _execution_body()
    payloads = []
    with service.stream("POST", "/agent/runs/r1/execute/stream", json=body) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[len("data: ") :]))
    assert [p["sequence"] for p in payloads] == sorted(p["sequence"] for p in payloads)
    assert payloads[-1]["type"] == contracts.RUN_COMPLETED
    assert host.runs["r1"]["status"] == "completed"
