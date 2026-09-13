"""The legacy run-check runner: absent by default, token-only, strictly validated, confined to
one run workspace, and mechanically bounded (timeout, captured output, process group).

It is not an isolation boundary and not an exact-snapshot runner; these tests pin exactly what
it does enforce, and ``test_the_runner_documents_what_it_does_not_isolate`` pins what it does not.
"""

from __future__ import annotations

import inspect
import os
import sys
import textwrap
import time

import pytest
from fastapi.testclient import TestClient

from slimx_agent import service
from slimx_agent.service import RUN_CHECK_ENABLED_ENV, create_app

TOKEN = "check-token"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspaces"
    (root / "run-1").mkdir(parents=True)
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(root))
    return root


@pytest.fixture
def check_client(workspace, monkeypatch):
    monkeypatch.setenv(RUN_CHECK_ENABLED_ENV, "true")
    monkeypatch.setenv("SLIMX_AGENT_INTERNAL_TOKEN", TOKEN)
    return TestClient(create_app())


@pytest.fixture
def runner_spy(monkeypatch):
    calls: list[list[str]] = []
    real = service._run_bounded_check

    def spy(argv, cwd, **kwargs):
        calls.append(argv)
        return real(argv, cwd, **kwargs)

    monkeypatch.setattr(service, "_run_bounded_check", spy)
    return calls


def _post(client, body, *, token: str | bytes | None = TOKEN):
    headers = {"Authorization": f"Bearer {token}" if isinstance(token, str) else token}
    return client.post("/internal/run-check", json=body, headers={} if token is None else headers)


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(code)]


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


# --- the surface ------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [None, "", "false", "0", "no"])
def test_run_check_is_absent_unless_explicitly_enabled(workspace, monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv(RUN_CHECK_ENABLED_ENV, raising=False)
    else:
        monkeypatch.setenv(RUN_CHECK_ENABLED_ENV, flag)
    monkeypatch.setenv("SLIMX_AGENT_INTERNAL_TOKEN", TOKEN)
    client = TestClient(create_app())
    assert _post(client, {"argv": ["true"], "run_id": "run-1"}).status_code == 404


def test_enabled_run_check_has_no_tokenless_mode(workspace, monkeypatch, runner_spy):
    monkeypatch.setenv(RUN_CHECK_ENABLED_ENV, "1")
    monkeypatch.delenv("SLIMX_AGENT_INTERNAL_TOKEN", raising=False)
    response = _post(TestClient(create_app()), {"argv": ["true"], "run_id": "run-1"}, token=None)
    assert response.status_code == 503
    assert runner_spy == []


@pytest.mark.parametrize(
    "authorization",
    [None, "Bearer wrong", TOKEN, f"Basic {TOKEN}", f"Bearer {TOKEN} ", b"Bearer \xe9"],
)
def test_enabled_run_check_refuses_every_wrong_credential(check_client, runner_spy, authorization):
    headers = {} if authorization is None else {"Authorization": authorization}
    response = check_client.post(
        "/internal/run-check", json={"argv": ["true"], "run_id": "run-1"}, headers=headers
    )
    assert response.status_code == 401
    assert runner_spy == []


# --- validation before any process starts -----------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"argv": [], "run_id": "run-1"},
        {"argv": "pytest -q", "run_id": "run-1"},
        {"argv": ["pytest", 3], "run_id": "run-1"},
        {"argv": ["a\x00b"], "run_id": "run-1"},
        {"argv": [""], "run_id": "run-1"},
        {"argv": ["x"] * 65, "run_id": "run-1"},
        {"argv": ["x" * 4097], "run_id": "run-1"},
        {"argv": ["true"]},
        {"argv": ["true"], "run_id": "."},
        {"argv": ["true"], "run_id": ".."},
        {"argv": ["true"], "run_id": "a/b"},
        {"argv": ["true"], "run_id": ""},
        {"argv": ["true"], "run_id": ".hidden"},
        {"argv": ["true"], "run_id": "run 1"},
        {"argv": ["true"], "run_id": "x" * 129},
        {"argv": ["true"], "run_id": "run-1", "timeout_seconds": 0},
        {"argv": ["true"], "run_id": "run-1", "timeout_seconds": -1},
        {"argv": ["true"], "run_id": "run-1", "timeout_seconds": 601},
        {"argv": ["true"], "run_id": "run-1", "timeout_seconds": "soon"},
        {"argv": ["true"], "run_id": "run-1", "output_cap": 0},
        {"argv": ["true"], "run_id": "run-1", "output_cap": -5},
        {"argv": ["true"], "run_id": "run-1", "output_cap": 100_001},
        {"argv": ["true"], "run_id": "run-1", "output_cap": "10"},
        {"argv": ["true"], "run_id": "run-1", "output_cap": True},
        {"argv": ["true"], "run_id": "run-1", "env": {"SECRET": "x"}},
        {"argv": ["true"], "run_id": "run-1", "cwd": "/"},
    ],
)
def test_malformed_requests_are_422_and_start_nothing(check_client, runner_spy, body):
    assert _post(check_client, body).status_code == 422
    assert runner_spy == []


def test_a_non_finite_timeout_is_refused(check_client, runner_spy):
    response = check_client.post(
        "/internal/run-check",
        content=b'{"argv": ["true"], "run_id": "run-1", "timeout_seconds": NaN}',
        headers={"Authorization": f"Bearer {TOKEN}", "content-type": "application/json"},
    )
    assert response.status_code == 422
    assert runner_spy == []


def test_the_run_is_confined_to_one_existing_workspace(
    check_client, workspace, tmp_path, runner_spy
):
    (tmp_path / "outside").mkdir()
    (workspace / "escape").symlink_to(tmp_path / "outside", target_is_directory=True)
    assert _post(check_client, {"argv": ["true"], "run_id": "missing"}).status_code == 404
    assert _post(check_client, {"argv": ["true"], "run_id": "escape"}).status_code == 404
    assert runner_spy == []


# --- what the runner enforces -----------------------------------------------------------


def test_a_passing_check_runs_in_the_run_workspace(check_client, workspace):
    body = {"argv": _py("import os; print(os.getcwd())"), "run_id": "run-1"}
    result = _post(check_client, body).json()
    assert result == {
        "ok": True,
        "exit_code": 0,
        "timed_out": False,
        "output": f"{os.path.realpath(workspace / 'run-1')}\n",
        "output_truncated": False,
    }


def test_a_failing_check_reports_its_exit_code(check_client):
    result = _post(check_client, {"argv": _py("import sys; sys.exit(3)"), "run_id": "run-1"}).json()
    assert (result["ok"], result["exit_code"], result["timed_out"]) == (False, 3, False)


def test_the_environment_is_scrubbed_and_stdin_is_closed(check_client, monkeypatch):
    monkeypatch.setenv("SECRET_API_KEY", "must-not-leak")
    code = """
        import os, sys
        print(sorted(os.environ))
        print(repr(sys.stdin.read()))
    """
    result = _post(check_client, {"argv": _py(code), "run_id": "run-1"}).json()
    names_line, stdin_line = result["output"].splitlines()
    assert set(eval(names_line)) <= {"PATH", "HOME", "LC_CTYPE"}  # LC_CTYPE: PEP 538 coercion
    assert stdin_line == "''"
    assert "must-not-leak" not in result["output"] and TOKEN not in result["output"]


def test_a_timeout_kills_the_whole_process_group(check_client, tmp_path):
    marker = tmp_path / "grandchild-survived"
    code = f"""
        import subprocess, sys, time
        subprocess.Popen(
            [sys.executable, "-c",
             "import pathlib, time; time.sleep(1.5); pathlib.Path({str(marker)!r}).write_text('x')"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(30)
    """
    started = time.monotonic()
    result = _post(
        check_client, {"argv": _py(code), "run_id": "run-1", "timeout_seconds": 1.0}
    ).json()
    assert time.monotonic() - started < 10
    assert (result["ok"], result["exit_code"], result["timed_out"]) == (False, None, True)
    time.sleep(2.5)
    assert not marker.exists()


def test_descendants_are_killed_even_when_the_leader_exits_first(check_client, tmp_path):
    marker = tmp_path / "orphan-survived"
    code = f"""
        import subprocess, sys
        subprocess.Popen(
            [sys.executable, "-c",
             "import pathlib, time; time.sleep(1.5); pathlib.Path({str(marker)!r}).write_text('x')"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    """
    result = _post(check_client, {"argv": _py(code), "run_id": "run-1"}).json()
    assert (result["ok"], result["exit_code"], result["timed_out"]) == (True, 0, False)
    time.sleep(2.5)
    assert not marker.exists()


def test_output_is_capped_while_it_is_read(check_client):
    code = "import sys; sys.stdout.write('x' * 2_000_000); sys.stdout.flush()"
    result = _post(check_client, {"argv": _py(code), "run_id": "run-1", "output_cap": 1000}).json()
    assert result["ok"] is True
    assert result["output"] == "x" * 1000
    assert result["output_truncated"] is True


def test_output_exactly_at_the_cap_is_not_marked_truncated(workspace):
    result = service._run_bounded_check(
        _py("import sys; sys.stdout.write('y' * 64)"),
        str(workspace / "run-1"),
        timeout_seconds=10,
        output_cap=64,
    )
    assert (result.output, result.truncated) == (b"y" * 64, False)


def test_the_runner_never_buffers_the_full_output():
    source = inspect.getsource(service._run_bounded_check)
    assert "capture_output" not in source and ".communicate(" not in source


def test_a_missing_or_unrunnable_command_is_reported_not_raised(check_client, workspace):
    missing = _post(check_client, {"argv": ["definitely-not-a-command-xyz"], "run_id": "run-1"})
    assert missing.status_code == 200
    assert missing.json()["ok"] is False
    assert missing.json()["output"].startswith("could not start the check command")
    script = workspace / "run-1" / "not-executable.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o644)
    refused = _post(check_client, {"argv": [str(script)], "run_id": "run-1"}).json()
    assert refused["ok"] is False and refused["exit_code"] is None


def test_invalid_utf8_output_is_replaced_not_fatal(check_client):
    code = "import sys; sys.stdout.buffer.write(b'\\xff\\xfeok')"
    result = _post(check_client, {"argv": _py(code), "run_id": "run-1"}).json()
    assert result["ok"] is True and result["output"].endswith("ok")


def test_the_runner_documents_what_it_does_not_isolate():
    doc = inspect.getdoc(service._run_bounded_check) or ""
    for limitation in ("NOT an isolation boundary", "filesystem, network", "not an exact snapshot"):
        assert limitation in doc
