"""HostClient and HttpRunStore at the wire trust boundary.

Strict parsing (a wrong JSON type is refused, never coerced), bounded error details, explicit
UNSET presence flags, path-safe identities, per-request authority, and no replay after a
transport failure.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from slimx_agent.host_client import (
    INVOKE_TIMEOUT_SECONDS,
    LEASE_GENERATION_HEADER,
    LEASE_JOB_ID_HEADER,
    LEASE_TOKEN_HEADER,
    MAX_DETAIL_CHARS,
    ExecutionAttempt,
    HostClient,
    HostError,
    HostProtocolError,
    HostUnavailable,
    profile_from_wire,
    profile_wire,
)
from slimx_agent.http_store import HttpRunStore, RunSnapshot, StepSnapshot
from slimx_agent.runtime import RunProfile

RUN = {"id": "r1", "status": "running", "approval_policy": "auto_complete", "auto_approve": False}
STEP = {"id": "s1", "type": "model_call", "title": "t", "status": "pending"}
QWEN = RunProfile("ollama", "qwen3:8b", "http://host.docker.internal:11434")


class Recorder:
    """A scripted HttpClient: answers per (method, url) — a response, an exception to raise,
    or a callable — and records every request with its keyword arguments."""

    def __init__(self, answers: dict[tuple[str, str], Any] | None = None, default: Any = None):
        self.answers = answers or {}
        self.default = default
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.requests.append({"method": method, "url": url, **kwargs})
        answer = self.answers.get((method, url), self.default)
        if isinstance(answer, BaseException):
            raise answer
        return answer() if callable(answer) else answer


def ok(body: Any) -> httpx.Response:
    return httpx.Response(200, json=body)


def raw(status: int, content: bytes = b"", content_type: str = "text/plain") -> httpx.Response:
    return httpx.Response(status, content=content, headers={"content-type": content_type})


# --- authority on every request ------------------------------------------------------------


def test_bearer_and_lease_travel_per_request_never_as_client_defaults():
    recorder = Recorder(default=ok(RUN))
    base = HostClient(client=recorder, token="tkn")
    bound = base.for_execution(ExecutionAttempt("job-1", "lease-secret", 3))

    bound.get_run("r1")
    sent = recorder.requests[-1]["headers"]
    assert sent == {
        "Authorization": "Bearer tkn",
        LEASE_JOB_ID_HEADER: "job-1",
        LEASE_TOKEN_HEADER: "lease-secret",
        LEASE_GENERATION_HEADER: "3",
    }
    sent["Authorization"] = "tampered"  # each request gets a fresh mapping
    bound.get_run("r1")
    assert recorder.requests[-1]["headers"]["Authorization"] == "Bearer tkn"

    base.get_run("r1")  # the base client never acquired the lease
    assert recorder.requests[-1]["headers"] == {"Authorization": "Bearer tkn"}


def test_no_token_means_no_authorization_header():
    recorder = Recorder(default=ok(RUN))
    HostClient(client=recorder).get_run("r1")
    assert "headers" not in recorder.requests[-1]


@pytest.mark.parametrize(
    ("job_id", "token", "generation"),
    [("", "t", 0), (" ", "t", 0), ("j", "", 0), ("j", "t", -1), ("j", "t", True), ("j", "t", "1")],
)
def test_execution_attempt_rejects_malformed_values(job_id, token, generation):
    with pytest.raises(ValueError):
        ExecutionAttempt(job_id, token, generation)


def test_execution_attempt_repr_never_reveals_the_lease_token():
    assert "lease-secret" not in repr(ExecutionAttempt("job-1", "lease-secret", 0))


def test_identities_are_quoted_into_exactly_one_path_segment():
    recorder = Recorder(default=ok(STEP))
    HostClient(client=recorder).get_step("../runs/r2/status?x=1")
    assert (
        recorder.requests[-1]["url"] == "/internal/agent-host/steps/..%2Fruns%2Fr2%2Fstatus%3Fx%3D1"
    )


def test_a_real_client_ignores_ambient_proxy_and_netrc_settings(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    client = HostClient("http://api:8000/", token="tkn")
    inner = client._client
    assert isinstance(inner, httpx.Client)
    assert inner.trust_env is False
    assert str(inner.base_url) == "http://api:8000"
    assert "authorization" not in {key.lower() for key in inner.headers}


def test_a_client_needs_a_url_or_an_injected_transport():
    with pytest.raises(ValueError, match="SLIMX_AGENT_HOST_URL"):
        HostClient(None)


def test_only_step_invocation_and_run_end_use_the_long_timeout():
    client = HostClient("http://api:8000")
    recorder = Recorder(
        answers={
            ("POST", "/internal/agent-host/runs/r1/steps/s1/invoke"): ok({"outcome": "completed"}),
            ("POST", "/internal/agent-host/runs/r1/run-end"): httpx.Response(204),
        },
        default=ok(RUN),
    )
    client._client = recorder
    client.get_run("r1")
    client.invoke_step("r1", "s1", QWEN)
    client.run_end("r1", "completed", QWEN)
    timeouts = [request.get("timeout") for request in recorder.requests]
    assert timeouts == [None, INVOKE_TIMEOUT_SECONDS, INVOKE_TIMEOUT_SECONDS]


# --- failure vocabulary -------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        raw(200, b"null", "application/json"),
        raw(200),
        ok(["not", "an", "object"]),
        raw(200, b"<html>not json</html>", "text/html"),
    ],
)
def test_null_empty_and_malformed_bodies_are_protocol_errors_not_not_found(answer):
    with pytest.raises(HostProtocolError) as caught:
        HostClient(client=Recorder(default=answer)).get_run("r1")
    assert caught.value.status_code == 502


def test_an_allowed_404_is_the_only_not_found():
    assert HostClient(client=Recorder(default=httpx.Response(404))).get_run("r1") is None
    with pytest.raises(HostError) as caught:
        HostClient(client=Recorder(default=httpx.Response(404))).get_steps("r1")
    assert caught.value.status_code == 404


@pytest.mark.parametrize(
    ("answer", "status", "detail"),
    [
        (httpx.Response(409, json={"detail": "stale lease"}), 409, "stale lease"),
        (httpx.Response(422, json={"detail": [{"loc": ["x"]}]}), 422, '[{"loc": ["x"]}]'),
        (raw(500, b"boom"), 500, "boom"),
        (raw(502), 502, "no detail"),
        (httpx.Response(409, json=["unexpected"]), 409, '["unexpected"]'),
    ],
)
def test_http_errors_keep_their_status_and_a_readable_detail(answer, status, detail):
    with pytest.raises(HostError) as caught:
        HostClient(client=Recorder(default=answer)).set_run_status("r1", "running")
    assert (caught.value.status_code, caught.value.detail) == (status, detail)


def test_error_details_are_bounded():
    answer = httpx.Response(409, json={"detail": "x" * 10_000})
    with pytest.raises(HostError) as caught:
        HostClient(client=Recorder(default=answer)).get_run("r1")
    assert len(caught.value.detail) == MAX_DETAIL_CHARS
    assert caught.value.detail.endswith("…")


@pytest.mark.parametrize(
    "failure",
    [httpx.ReadTimeout("timed out"), httpx.ConnectError("refused"), httpx.RemoteProtocolError("x")],
)
def test_transport_failures_are_host_unavailable_and_never_replayed(failure):
    recorder = Recorder(default=failure)
    with pytest.raises(HostUnavailable) as caught:
        HostClient(client=recorder).set_step_state("s1", "completed", error=None)
    assert isinstance(caught.value, HostError)
    assert caught.value.status_code == 503
    assert len(recorder.requests) == 1


def test_non_transport_exceptions_are_not_disguised_as_host_failures():
    with pytest.raises(RuntimeError, match="client bug"):
        HostClient(client=Recorder(default=RuntimeError("client bug"))).get_run("r1")


# --- wire shapes ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "body"),
    [
        ({}, {"status": "running", "error_set": False, "output_refs_set": False}),
        (
            {"error": None},
            {"status": "running", "error_set": True, "error": None, "output_refs_set": False},
        ),
        (
            {"error": "boom"},
            {"status": "running", "error_set": True, "error": "boom", "output_refs_set": False},
        ),
        (
            {"output_refs": None},
            {"status": "running", "error_set": False, "output_refs_set": True, "output_refs": None},
        ),
        (
            {"output_refs": {"a": 1}},
            {
                "status": "running",
                "error_set": False,
                "output_refs_set": True,
                "output_refs": {"a": 1},
            },
        ),
    ],
)
def test_unset_and_null_travel_as_explicit_presence_flags(kwargs, body):
    recorder = Recorder(default=ok(STEP))
    HostClient(client=recorder).set_step_state("s1", "running", **kwargs)
    assert recorder.requests[-1]["json"] == body


def test_step_invocation_forwards_the_exact_profile():
    recorder = Recorder(default=ok({"outcome": "completed"}))
    HostClient(client=recorder).invoke_step("r1", "s1", QWEN)
    assert recorder.requests[-1]["json"] == {
        "profile": {
            "provider": "ollama",
            "model": "qwen3:8b",
            "base_url": "http://host.docker.internal:11434",
        }
    }
    assert profile_wire(QWEN) == recorder.requests[-1]["json"]["profile"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"next_sequence": 4}, 4),
        ({"next_sequence": 0}, 0),
        ({"next_sequence": True}, HostProtocolError),
        ({"next_sequence": "4"}, HostProtocolError),
        ({"next_sequence": -1}, HostProtocolError),
        ({}, HostProtocolError),
        ([4], HostProtocolError),
    ],
)
def test_next_sequence_must_be_a_non_negative_integer(body, expected):
    client = HostClient(client=Recorder(default=ok(body)))
    if isinstance(expected, int):
        assert client.next_sequence("r1") == expected
    else:
        with pytest.raises(expected):
            client.next_sequence("r1")


@pytest.mark.parametrize(
    "body",
    [{"sequence": 1}, [{"sequence": "1"}], [{"type": "x"}], [{"sequence": True}], [1], None],
)
def test_events_must_be_an_array_of_objects_with_integer_sequences(body):
    with pytest.raises(HostProtocolError):
        HostClient(client=Recorder(default=ok(body))).events_after("r1", 0)


def test_valid_event_pages_and_step_lists_pass_through():
    events = [{"sequence": 1, "type": "agent.step.started"}, {"sequence": 2, "type": "x"}]
    assert HostClient(client=Recorder(default=ok(events))).events_after("r1", 0) == events
    assert HostClient(client=Recorder(default=ok([STEP]))).get_steps("r1") == [STEP]
    with pytest.raises(HostProtocolError):
        HostClient(client=Recorder(default=ok([STEP, 5]))).get_steps("r1")


@pytest.mark.parametrize(
    "data",
    [
        {"provider": "", "model": "m"},
        {"provider": "p", "model": ""},
        {"provider": "p"},
        {"provider": 1, "model": "m"},
        {"provider": "p", "model": "m", "base_url": 5},
    ],
)
def test_profile_from_wire_refuses_anything_but_exact_strings(data):
    with pytest.raises(ValueError):
        profile_from_wire(data)


def test_profile_from_wire_round_trips_the_exact_identity():
    assert profile_from_wire(profile_wire(QWEN)) == QWEN


# --- strict snapshots -------------------------------------------------------------------


def test_complete_snapshots_parse_exactly():
    run = RunSnapshot.from_wire(
        {
            **RUN,
            "allowed_tools_json": ["web_search"],
            "budget_max_steps": 5,
            "budget_max_wall_seconds": None,
            "preapproved_tools": ["web_search"],
            "workspace_id": "w",
        }
    )
    assert (run.id, run.status, run.auto_approve, run.budget_max_steps) == (
        "r1",
        "running",
        False,
        5,
    )
    assert run.allowed_tools_json == ["web_search"] and run.preapproved_tools == ["web_search"]
    assert run.raw["workspace_id"] == "w"
    minimal = RunSnapshot.from_wire({"id": "r1", "status": "planned"})
    assert (minimal.approval_policy, minimal.auto_approve, minimal.allowed_tools_json) == (
        None,
        False,
        None,
    )
    step = StepSnapshot.from_wire({**STEP, "title": None})
    assert (step.title, step.requires_approval) == ("", False)


@pytest.mark.parametrize(
    "override",
    [
        {"id": None},
        {"id": ""},
        {"id": 5},
        {"status": None},
        {"status": 3},
        {"approval_policy": 5},
        {"auto_approve": "false"},  # bool("false") would have been True
        {"auto_approve": 1},
        {"allowed_tools_json": "web_search"},
        {"allowed_tools_json": [1]},
        {"allowed_tools_json": {"web_search": True}},
        {"budget_max_steps": "5"},  # a malformed budget must not become unbounded
        {"budget_max_steps": True},
        {"budget_max_wall_seconds": 1.5},
        {"preapproved_tools": "web_search"},
        {"preapproved_tools": {"web_search": True}},
    ],
)
def test_malformed_run_snapshots_are_refused_not_coerced(override):
    with pytest.raises(HostProtocolError):
        RunSnapshot.from_wire({**RUN, **override})


@pytest.mark.parametrize(
    "override",
    [
        {"id": None},
        {"type": None},
        {"type": 7},
        {"status": None},
        {"title": 5},
        {"requires_approval": "false"},
    ],
)
def test_malformed_step_snapshots_are_refused_not_coerced(override):
    with pytest.raises(HostProtocolError):
        StepSnapshot.from_wire({**STEP, **override})


def test_the_http_store_surfaces_malformed_host_state_as_a_protocol_error():
    store = HttpRunStore(HostClient(client=Recorder(default=ok({**RUN, "auto_approve": "false"}))))
    with pytest.raises(HostProtocolError):
        store.get_run("r1")
    assert (
        HttpRunStore(HostClient(client=Recorder(default=httpx.Response(404)))).get_run("r1") is None
    )
