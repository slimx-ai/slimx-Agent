"""The standalone service's negative-path contract over the real wire (``tests/_fake_host.py``).

Covers authentication and the documented tokenless local mode, execution-attempt authority,
every invocation-outcome envelope, UNSET/null presence on the wire, transport failures without
replay, exact model identity, and SSE observation semantics — including that a disconnected
observer is not cancellation authority.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

import pytest
from _fake_host import FakeHost, LossyClient, drive, execution_body
from fastapi import HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from slimx_agent import contracts, service
from slimx_agent.host_client import HostClient, HostUnavailable
from slimx_agent.service import create_app
from slimx_agent.tools import StepOutcomeUnknown

TERMINAL = {contracts.STEP_COMPLETED, contracts.STEP_FAILED, contracts.STEP_SKIPPED}
TOKEN = "service-token-value"


def _service(host: FakeHost, monkeypatch, *, token=None, client=None, callback_token=None):
    if token is None:
        monkeypatch.delenv("SLIMX_AGENT_INTERNAL_TOKEN", raising=False)
    else:
        monkeypatch.setenv("SLIMX_AGENT_INTERNAL_TOKEN", token)
    host_client = HostClient(client=client or host.transport(), token=callback_token)
    return TestClient(create_app(host_client=host_client))


def _one_step_host(step_type: str = "model_call", **run: Any) -> FakeHost:
    host = FakeHost()
    host.add_run("r1", **run)
    host.add_step("r1", "s1", step_type)
    return host


def _raising(exc: Exception):
    def behavior():
        raise exc

    return behavior


# --- authentication -----------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/agent/runs/r1/execute", "/agent/runs/r1/execute/stream"])
@pytest.mark.parametrize(
    "authorization",
    [None, "Bearer wrong", f"Basic {TOKEN}", f"Bearer {TOKEN} ", b"Bearer \xe9\xe9", "bearer"],
)
def test_a_configured_token_is_required_on_both_execute_surfaces(monkeypatch, path, authorization):
    host = _one_step_host()
    client = _service(host, monkeypatch, token=TOKEN)
    headers = {} if authorization is None else {"Authorization": authorization}
    response = client.post(path, json=execution_body(), headers=headers)
    assert response.status_code == 401
    assert TOKEN not in response.text
    assert host.callback_requests == []


def test_the_right_token_opens_both_surfaces(monkeypatch):
    host = _one_step_host()
    client = _service(host, monkeypatch, token=TOKEN)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    assert client.post("/agent/runs/r1/execute", json=execution_body(), headers=headers).json() == {
        "run_id": "r1",
        "status": "completed",
    }


def test_tokenless_mode_is_the_documented_local_compatibility(monkeypatch):
    host = _one_step_host()
    client = _service(host, monkeypatch)
    assert client.get("/health").json()["auth_enabled"] is False
    response = client.post(
        "/agent/runs/r1/execute", json=execution_body(), headers={"Authorization": "anything"}
    )
    assert response.status_code == 200


def test_the_token_never_appears_in_health_or_error_bodies(monkeypatch):
    host = _one_step_host()
    host.behaviors["model_call"] = _raising(HTTPException(500, detail="host crashed"))
    client = _service(host, monkeypatch, token=TOKEN, callback_token=TOKEN)
    health = client.get("/health")
    failed = client.post(
        "/agent/runs/r1/execute",
        json=execution_body(),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert health.json()["auth_enabled"] is True
    assert failed.status_code == 502
    for body in (health.text, failed.text):
        assert TOKEN not in body


def test_callbacks_carry_the_service_bearer_and_lease_per_request(monkeypatch):
    host = _one_step_host()
    shared = host.transport()
    client = _service(host, monkeypatch, client=shared, callback_token="callback-token")
    body = execution_body(7)
    assert client.post("/agent/runs/r1/execute", json=body).status_code == 200
    assert host.callback_requests
    for _path, headers in host.callback_requests:
        assert headers["authorization"] == "Bearer callback-token"
        assert headers["x-slimx-agent-lease-generation"] == "7"
    assert "authorization" not in {key.lower() for key in shared.headers}


def test_a_lazily_built_client_without_a_host_url_is_a_503(monkeypatch):
    monkeypatch.delenv("SLIMX_AGENT_HOST_URL", raising=False)
    monkeypatch.delenv("SLIMX_AGENT_INTERNAL_TOKEN", raising=False)
    response = TestClient(create_app()).post("/agent/runs/r1/execute", json=execution_body())
    assert response.status_code == 503
    assert "SLIMX_AGENT_HOST_URL" in response.json()["detail"]


# --- execution-attempt authority --------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"lease_job_id": "not-a-uuid"},
        {"lease_token": 123},
        {"lease_generation": "1"},
        {"lease_generation": True},
        {"lease_generation": -1},
        {"lease_generation": 1.5},
        {"lease_token": None},
        {"provider": ""},
        {"model": ""},
        {"model": "m" * 513},
        {"base_url": "http://x/" + "p" * 2100},
    ],
)
def test_malformed_leases_and_profiles_are_refused_before_any_callback(monkeypatch, override):
    host = _one_step_host()
    client = _service(host, monkeypatch)
    for path in ("/agent/runs/r1/execute", "/agent/runs/r1/execute/stream"):
        assert client.post(path, json=execution_body(**override)).status_code == 422
    assert host.callback_requests == []


@pytest.mark.parametrize("run_id", ["bad%20id", ".hidden", "x" * 200])
def test_run_ids_that_are_not_plain_identifiers_are_refused_before_any_callback(
    monkeypatch, run_id
):
    host = _one_step_host()
    client = _service(host, monkeypatch)
    assert client.post(f"/agent/runs/{run_id}/execute", json=execution_body()).status_code == 422
    assert host.callback_requests == []


@pytest.mark.parametrize("path", ["/agent/runs/r1/execute", "/agent/runs/r1/execute/stream"])
def test_a_stale_lease_is_refused_at_the_first_callback_and_nothing_follows(monkeypatch, path):
    host = _one_step_host()
    host.current_lease_generation = 2
    client = _service(host, monkeypatch)
    response = client.post(path, json=execution_body(1))
    assert response.status_code == 409
    assert host.callback_log == [("GET", "/internal/agent-host/runs/r1")]
    assert host.runs["r1"]["status"] == "planned"


def test_lease_loss_at_the_tool_edge_stops_without_a_false_failed_step(monkeypatch):
    host = _one_step_host()

    def lose_the_lease():
        host.current_lease_generation = 99
        raise HTTPException(409, detail="Agent execution attempt is no longer valid")

    host.behaviors["model_call"] = lose_the_lease
    client = _service(host, monkeypatch)
    response = client.post("/agent/runs/r1/execute", json=execution_body(1))
    assert response.status_code == 409
    assert "no longer valid" in response.json()["detail"]
    assert host.callbacks_after("/steps/s1/invoke") == []
    assert host.steps["s1"]["status"] == "running"
    assert not TERMINAL & set(host.event_types("r1"))
    assert host.run_end_calls == []


# --- invocation-outcome envelopes -------------------------------------------------------


@pytest.mark.parametrize(
    ("envelope", "status", "error", "refs"),
    [
        ({"outcome": "completed", "output_refs": {"a": 1}}, "completed", None, {"a": 1}),
        ({"outcome": "completed", "output_refs": None}, "completed", None, None),
        ({"outcome": "completed"}, "completed", None, None),
        ({"outcome": "skipped", "reason": "nothing indexed"}, "skipped", None, None),
        ({"outcome": "skipped", "reason": 5}, "skipped", None, None),
        (
            {"outcome": "failed", "error": "provider unreachable"},
            "failed",
            "provider unreachable",
            None,
        ),
        ({"outcome": "failed"}, "failed", "step failed on the host", None),
    ],
)
def test_authoritative_envelopes_map_to_exactly_one_terminal_state(envelope, status, error, refs):
    host = _one_step_host()
    host.behaviors["model_call"] = envelope
    drive(host, "r1")
    step = host.steps["s1"]
    assert (step["status"], step["error"], step["output_refs_json"]) == (status, error, refs)
    terminal = [event for event in host.events["r1"] if event["type"] in TERMINAL]
    assert len(terminal) == 1
    if status == "skipped":
        expected_reason = (
            envelope["reason"] if isinstance(envelope["reason"], str) else "step not applicable"
        )
        assert terminal[0]["payload_json"]["reason"] == expected_reason


@pytest.mark.parametrize(
    "behavior",
    [
        {"outcome": "completed", "output_refs": ["not", "an", "object"]},
        {"outcome": "completed", "output_refs": "refs"},
        {"outcome": "succeeded"},
        {"outcome": None},
        {},
        lambda: ["completed"],
        lambda: PlainTextResponse("completed"),
        lambda: PlainTextResponse(""),
        _raising(HTTPException(500, detail="host crashed mid-tool")),
        _raising(HTTPException(409, detail="refused")),
        _raising(HTTPException(422, detail="bad profile")),
    ],
)
def test_non_authoritative_answers_never_produce_a_terminal_step(behavior):
    host = _one_step_host()
    host.behaviors["model_call"] = behavior
    with pytest.raises(StepOutcomeUnknown):
        drive(host, "r1")
    assert host.steps["s1"]["status"] == "running"
    assert not TERMINAL & set(host.event_types("r1"))
    assert contracts.RUN_COMPLETED not in host.event_types("r1")
    assert host.callbacks_after("/steps/s1/invoke") == []
    assert host.run_end_calls == []


def test_unset_and_null_travel_as_explicit_presence_flags_end_to_end():
    host = FakeHost()
    host.add_run("r1")
    host.add_step("r1", "s1", "model_call", error="kept while running")
    host.add_step("r1", "s2", "rag_retrieve")
    host.add_step("r1", "s3", "create_synthesis")
    host.behaviors["model_call"] = {"outcome": "completed", "output_refs": {"x": 1}}
    host.behaviors["rag_retrieve"] = {"outcome": "skipped", "reason": "no index"}
    host.behaviors["create_synthesis"] = {"outcome": "failed", "error": "boom"}

    drive(host, "r1")

    bodies = {(step_id, body["status"]): body for step_id, body in host.state_bodies}
    assert bodies[("s1", "running")] == {
        "status": "running",
        "error_set": False,
        "output_refs_set": False,
    }
    assert bodies[("s1", "completed")] == {
        "status": "completed",
        "error_set": True,
        "error": None,
        "output_refs_set": True,
        "output_refs": {"x": 1},
    }
    assert bodies[("s2", "skipped")] == {
        "status": "skipped",
        "error_set": True,
        "error": None,
        "output_refs_set": False,
    }
    assert bodies[("s3", "failed")] == {
        "status": "failed",
        "error_set": True,
        "error": "boom",
        "output_refs_set": False,
    }


# --- transport failures: no replay, no false effects ------------------------------------


def test_an_ambiguous_invocation_is_never_replayed():
    host = _one_step_host()
    lossy = LossyClient(host.transport(), method="POST", path_suffix="/steps/s1/invoke")
    with pytest.raises(StepOutcomeUnknown):
        drive(host, "r1", client=lossy)
    assert lossy.matching_requests == 1
    assert len(host.invoked_profiles) == 1  # the host ran the tool exactly once
    assert host.steps["s1"]["status"] == "running"
    assert host.callbacks_after("/steps/s1/invoke") == []


def test_a_lost_terminal_write_is_not_replayed():
    host = _one_step_host()
    # The second /state write for s1 is its completed projection (the first is "running").
    lossy = LossyClient(host.transport(), method="POST", path_suffix="/steps/s1/state", nth=2)
    with pytest.raises(HostUnavailable):
        drive(host, "r1", client=lossy)
    assert lossy.matching_requests == 2
    assert host.steps["s1"]["status"] == "completed"  # the host applied it; the client never saw it
    state_writes = [entry for entry in host.callback_log if entry[1].endswith("/steps/s1/state")]
    assert len(state_writes) == 2  # running, then the lost completed write — never a third
    assert host.callback_log[-1] == ("POST", "/internal/agent-host/steps/s1/state")
    assert contracts.STEP_COMPLETED not in host.event_types("r1")


def test_a_failed_read_produces_no_false_effects():
    host = _one_step_host()
    lossy = LossyClient(host.transport(), method="GET", path_suffix="/runs/r1/steps", when="before")
    with pytest.raises(HostUnavailable):
        drive(host, "r1", client=lossy)
    assert host.runs["r1"]["status"] == "running"
    assert host.steps["s1"]["status"] == "pending"
    assert host.invoked_profiles == []
    assert host.events["r1"] == []


@pytest.mark.parametrize(
    ("behavior", "status", "detail_part"),
    [
        (_raising(HTTPException(500, detail="host crashed")), 502, "agent host error 500"),
        (_raising(HTTPException(409, detail="refused")), 409, "refused"),
        ({"outcome": "succeeded"}, 502, "unrecognized invocation outcome"),
    ],
)
def test_the_service_relays_unknown_outcomes_with_bounded_status(
    monkeypatch, behavior, status, detail_part
):
    host = _one_step_host()
    host.behaviors["model_call"] = behavior
    response = _service(host, monkeypatch).post("/agent/runs/r1/execute", json=execution_body())
    assert response.status_code == status
    assert detail_part in response.json()["detail"]
    assert host.steps["s1"]["status"] == "running"


def test_a_transport_failure_is_a_bounded_502(monkeypatch):
    host = _one_step_host()
    lossy = LossyClient(host.transport(), method="GET", path_suffix="/runs/r1/steps", when="before")
    response = _service(host, monkeypatch, client=lossy).post(
        "/agent/runs/r1/execute", json=execution_body()
    )
    assert response.status_code == 502
    assert "host unavailable" in response.json()["detail"]


def test_host_error_details_are_bounded_at_the_service_edge(monkeypatch):
    host = _one_step_host()
    host.behaviors["model_call"] = _raising(HTTPException(409, detail="x" * 20_000))
    response = _service(host, monkeypatch).post("/agent/runs/r1/execute", json=execution_body())
    assert response.status_code == 409
    assert len(response.json()["detail"]) <= 500


# --- exact model identity -----------------------------------------------------------------


@pytest.mark.parametrize(
    "model", ["qwen3:8b", "hf.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF:Q4_K_M"]
)
def test_the_host_selected_model_travels_exactly_to_every_callback(monkeypatch, model):
    host = FakeHost()
    host.add_run("r1")
    host.add_step("r1", "s1", "model_call")
    host.add_step("r1", "s2", "create_synthesis")
    body = execution_body(
        provider="ollama",
        model=model,
        base_url="http://host.docker.internal:11434",
        settings={"num_ctx": 8192},  # host-only extras are accepted and not echoed
        provider_profile_id="pp-1",
        profile_name="Local Qwen",
    )
    assert _service(host, monkeypatch).post("/agent/runs/r1/execute", json=body).status_code == 200
    expected = {
        "provider": "ollama",
        "model": model,
        "base_url": "http://host.docker.internal:11434",
    }
    assert host.invoked_profiles == [expected, expected]
    assert host.run_end_profiles == [expected]
    assert {profile["model"] for profile in host.invoked_profiles + host.run_end_profiles} == {
        model
    }
    assert "llama3" not in json.dumps(host.invoked_profiles + host.run_end_profiles)


# --- SSE observation --------------------------------------------------------------------


def _sse_lines(client: TestClient, path: str, body: dict[str, Any]) -> list[str]:
    with client.stream("POST", path, json=body) as response:
        assert response.status_code == 200
        return list(response.iter_lines())


def test_the_stream_keeps_quiet_steps_alive_and_orders_events(monkeypatch):
    monkeypatch.setenv("SLIMX_AGENT_KEEPALIVE_SECONDS", "0.05")
    host = _one_step_host()
    host.invoke_delay_seconds = 0.4
    lines = _sse_lines(
        _service(host, monkeypatch), "/agent/runs/r1/execute/stream", execution_body()
    )
    assert ": keepalive" in lines
    payloads = [json.loads(line[len("data: ") :]) for line in lines if line.startswith("data: ")]
    assert [p["sequence"] for p in payloads] == sorted(p["sequence"] for p in payloads)
    assert payloads[-1]["type"] == contracts.RUN_COMPLETED


def test_the_stream_ends_cleanly_when_a_step_outcome_is_unknown(monkeypatch, caplog):
    host = _one_step_host()
    host.behaviors["model_call"] = _raising(HTTPException(500, detail="host crashed"))
    with caplog.at_level(logging.WARNING, logger="slimx_agent.service"):
        lines = _sse_lines(
            _service(host, monkeypatch), "/agent/runs/r1/execute/stream", execution_body()
        )
    assert not any("Traceback" in line for line in lines)
    assert contracts.RUN_COMPLETED not in "".join(lines)
    assert host.steps["s1"]["status"] == "running"
    assert "drive ended early" in caplog.text


def test_cancellation_is_the_hosts_run_status_authority(monkeypatch):
    host = FakeHost()
    host.add_run("r1")
    host.add_step("r1", "s1", "model_call")
    host.add_step("r1", "s2", "create_synthesis")

    def complete_then_cancel():
        host.runs["r1"]["status"] = "cancelled"  # the host's cancel route landed mid-step
        return {"outcome": "completed", "output_refs": None}

    host.behaviors["model_call"] = complete_then_cancel
    lines = _sse_lines(
        _service(host, monkeypatch), "/agent/runs/r1/execute/stream", execution_body()
    )
    types = [json.loads(line[6:])["type"] for line in lines if line.startswith("data: ")]
    assert contracts.RUN_COMPLETED not in types
    assert len(host.invoked_profiles) == 1  # s2 never ran
    assert host.runs["r1"]["status"] == "cancelled"


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _drive_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "slimx-agent-run"]


def test_an_observer_disconnect_does_not_cancel_the_drive(monkeypatch):
    """Drive the ASGI app directly so the client can really disconnect after the first event
    (FastAPI's TestClient buffers whole bodies). The drive must still reach its durable stop."""
    host = FakeHost()
    host.add_run("r1")
    for index in range(1, 4):
        host.add_step("r1", f"s{index}", "model_call")
    host.invoke_delay_seconds = 0.2
    monkeypatch.delenv("SLIMX_AGENT_INTERNAL_TOKEN", raising=False)
    app = create_app(host_client=HostClient(client=host.transport()))
    body = json.dumps(execution_body()).encode()

    async def stream_then_disconnect() -> list[dict[str, Any]]:
        sent: list[dict[str, Any]] = []
        first_event = asyncio.Event()
        request_delivered = False

        async def receive() -> dict[str, Any]:
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await first_event.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)
            if message["type"] == "http.response.body" and message.get("body", b"").startswith(
                b"data: "
            ):
                first_event.set()

        path = "/agent/runs/r1/execute/stream"
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 5000),
            "server": ("testserver", 80),
        }
        await app(scope, receive, send)
        return sent

    sent = asyncio.run(stream_then_disconnect())
    observed = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert b"agent.run.completed" not in observed  # the observer left before the end

    assert _wait_for(lambda: host.runs["r1"]["status"] == "completed")
    assert [host.steps[f"s{i}"]["status"] for i in range(1, 4)] == ["completed"] * 3
    assert host.run_end_calls == [("r1", "completed")]
    assert _wait_for(lambda: not _drive_threads(), timeout=5.0)


def test_the_bridge_keeps_draining_after_its_event_loop_closes():
    produced: list[int] = []
    finished = threading.Event()

    def source():
        try:
            for index in range(5):
                time.sleep(0.05)
                produced.append(index)
                yield index
        finally:
            finished.set()

    async def take_first() -> int | None:
        stream = service._bridge(source, keepalive_seconds=5)
        first = await anext(stream)
        await stream.aclose()
        return first

    assert asyncio.run(take_first()) == 0
    assert finished.wait(5)
    assert produced == [0, 1, 2, 3, 4]


def test_the_bridge_logs_an_unexpected_drive_failure_and_ends(caplog):
    def source():
        yield "first"
        raise RuntimeError("engine bug")

    async def collect() -> list[str | None]:
        return [item async for item in service._bridge(source, keepalive_seconds=5)]

    with caplog.at_level(logging.ERROR, logger="slimx_agent.service"):
        assert asyncio.run(collect()) == ["first"]
    assert "failed unexpectedly" in caplog.text


@pytest.mark.parametrize(
    ("value", "expected"), [("abc", 15.0), ("0", 15.0), ("-1", 15.0), ("0.5", 0.5)]
)
def test_keepalive_interval_parsing_is_fail_safe(monkeypatch, value, expected):
    monkeypatch.setenv("SLIMX_AGENT_KEEPALIVE_SECONDS", value)
    assert service._keepalive_seconds() == expected
