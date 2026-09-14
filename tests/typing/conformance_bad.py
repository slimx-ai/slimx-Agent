"""Static conformance, negative half: invalid host adapters are REJECTED.

Every line marked ``# E: <code>`` must produce exactly that mypy error code, and no other line
may produce one; ``tests/test_static_conformance.py`` fails on any missing, extra, or changed
diagnostic. Bodies raise instead of using ``...`` so the only errors are the marked ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from conformance_ok import MemoryStore, Run, Session, Step, model_call

from slimx_agent import engine
from slimx_agent.host_client import HostClient
from slimx_agent.http_store import HttpRunStore, RunSnapshot
from slimx_agent.http_tools import build_remote_registry
from slimx_agent.runtime import RunProfile
from slimx_agent.store import UNSET, EventPayload, OutputRefs, RunStore, RunView, UnsetType
from slimx_agent.tools import ToolRegistry


class _StoreCore:
    """Every RunStore member except set_step_state, next_sequence, and drained_events."""

    @property
    def handler_context(self) -> Session:
        raise NotImplementedError

    def get_run(self, run_id: str) -> Run | None:
        raise NotImplementedError

    def get_steps(self, run_id: str) -> list[Step]:
        raise NotImplementedError

    def get_step(self, step_id: str) -> Step | None:
        raise NotImplementedError

    def set_run_status(self, run: Run, status: str) -> Run:
        raise NotImplementedError

    def rollback(self) -> None:
        raise NotImplementedError

    def append_event(
        self,
        run_id: str,
        type: str,
        *,
        step_id: str | None = None,
        payload: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> object:
        raise NotImplementedError


class MissingDrain(_StoreCore):
    def set_step_state(
        self,
        step_id: str,
        status: str,
        *,
        error: str | None | UnsetType = UNSET,
        output_refs: OutputRefs | None | UnsetType = UNSET,
    ) -> Step:
        raise NotImplementedError

    def next_sequence(self, run_id: str) -> int:
        raise NotImplementedError


class NoOutputRefsKeyword(_StoreCore):
    def set_step_state(
        self, step_id: str, status: str, *, error: str | None | UnsetType = UNSET
    ) -> Step:
        raise NotImplementedError

    def next_sequence(self, run_id: str) -> int:
        raise NotImplementedError

    def drained_events(self, run_id: str, after_sequence: int) -> list[EventPayload]:
        raise NotImplementedError


class TextSequence(_StoreCore):
    def set_step_state(
        self,
        step_id: str,
        status: str,
        *,
        error: str | None | UnsetType = UNSET,
        output_refs: OutputRefs | None | UnsetType = UNSET,
    ) -> Step:
        raise NotImplementedError

    def next_sequence(self, run_id: str) -> str:
        raise NotImplementedError

    def drained_events(self, run_id: str, after_sequence: int) -> list[EventPayload]:
        raise NotImplementedError


@dataclass
class IntStatusRun:
    id: str
    status: int
    approval_policy: str | None
    auto_approve: bool
    allowed_tools_json: list[str] | None


@dataclass
class ModelOnly:
    model: str


def three_args(session: Session, run: Run, step: Step) -> OutputRefs:
    raise NotImplementedError


def returns_list(session: Session, run: Run, step: Step, profile: RunProfile) -> list[str]:
    raise NotImplementedError


def wants_client(client: HostClient, run: Run, step: Step, profile: RunProfile) -> OutputRefs:
    raise NotImplementedError


def end_for_snapshot(run: RunSnapshot, status: str) -> None:
    raise NotImplementedError


def rejected(run: Run, snapshot: RunSnapshot, client: HostClient) -> None:
    p = RunProfile("ollama", "qwen3:8b")
    reg: ToolRegistry[Session, Run, Step, RunProfile] = ToolRegistry()
    reg.register("model_call", model_call)
    good = MemoryStore(run, [])
    remote = build_remote_registry()
    http = HttpRunStore(client)

    engine.execute_run(MissingDrain(), reg, run, profile=p)  # E: arg-type
    engine.execute_run(NoOutputRefsKeyword(), reg, run, profile=p)  # E: arg-type
    engine.execute_run(TextSequence(), reg, run, profile=p)  # E: arg-type
    reg.register("three", three_args)  # E: arg-type
    reg.register("list", returns_list)  # E: arg-type
    reg.register("client", wants_client)  # E: arg-type
    # A registry whose context/rows differ from the store's, and a profile lacking provider and
    # base_url: mypy rejects both by refusing to infer the engine's type parameters.
    engine.execute_run(good, remote, run, profile=p)  # E: misc
    engine.execute_run(http, remote, snapshot, profile=ModelOnly("qwen3:8b"))  # E: misc
    engine.execute_run(good, reg, run, profile=p, on_run_end=end_for_snapshot)  # E: arg-type
    good.set_step_state("s1", "running", output_refs=object())  # E: arg-type
    bad_view: RunView = IntStatusRun("r1", 1, None, False, None)  # E: assignment
    wrong_rows: RunStore[Run, Step, Session] = HttpRunStore(client)  # E: assignment
    del bad_view, wrong_rows
