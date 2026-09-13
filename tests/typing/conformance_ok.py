"""Static conformance, positive half: valid host adapters type-check under ``mypy --strict``.

Never imported at runtime — ``tests/test_static_conformance.py`` runs mypy over this file and
``conformance_bad.py`` and requires zero diagnostics here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from slimx_agent import engine
from slimx_agent.host_client import HostClient
from slimx_agent.http_store import HttpRunStore, RunSnapshot, StepSnapshot
from slimx_agent.http_tools import build_remote_registry
from slimx_agent.runtime import ProfileView, RunProfile
from slimx_agent.store import (
    UNSET,
    EventPayload,
    OutputRefs,
    RunStore,
    RunView,
    StepView,
    UnsetType,
)
from slimx_agent.tools import ToolHandler, ToolRegistry


@dataclass
class Run:
    """A host's own run row (think: an ORM model with plain attributes)."""

    id: str
    status: str = "planned"
    approval_policy: str | None = "auto_complete"
    auto_approve: bool = False
    allowed_tools_json: list[str] | None = None


@dataclass
class Step:
    id: str
    type: str
    title: str = "step"
    status: str = "pending"
    requires_approval: bool = False


class Session:
    """A host's handler context (think: a database session)."""


@dataclass
class MemoryStore:
    run: Run
    steps: list[Step]
    events: list[EventPayload] = field(default_factory=list)
    session: Session = field(default_factory=Session)

    @property
    def handler_context(self) -> Session:
        return self.session

    def get_run(self, run_id: str) -> Run | None:
        return self.run if self.run.id == run_id else None

    def get_steps(self, run_id: str) -> list[Step]:
        return list(self.steps)

    def get_step(self, step_id: str) -> Step | None:
        return next((step for step in self.steps if step.id == step_id), None)

    def set_run_status(self, run: Run, status: str) -> Run:
        self.run.status = status
        return self.run

    def set_step_state(
        self,
        step_id: str,
        status: str,
        *,
        error: str | None | UnsetType = UNSET,
        output_refs: OutputRefs | None | UnsetType = UNSET,
    ) -> Step:
        step = self.get_step(step_id)
        if step is None:
            raise KeyError(step_id)
        step.status = status
        return step

    def rollback(self) -> None:
        return None

    def append_event(
        self,
        run_id: str,
        type: str,
        *,
        step_id: str | None = None,
        payload: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> EventPayload:
        event: EventPayload = {"sequence": len(self.events) + 1, "type": type}
        self.events.append(event)
        return event

    def next_sequence(self, run_id: str) -> int:
        return len(self.events) + 1

    def drained_events(self, run_id: str, after_sequence: int) -> list[EventPayload]:
        return [event for event in self.events if event["sequence"] > after_sequence]


def model_call(session: Session, run: Run, step: Step, profile: RunProfile) -> OutputRefs:
    return {"ref": step.id}


def on_run_end(run: Run, status: str) -> None:
    return None


def drive_in_process() -> Run:
    registry: ToolRegistry[Session, Run, Step, RunProfile] = ToolRegistry()
    registry.register("model_call", model_call)
    store = MemoryStore(Run("r1"), [Step("s1", "model_call")])
    profile = RunProfile("ollama", "qwen3:8b")
    return engine.execute_run(store, registry, store.run, profile=profile, on_run_end=on_run_end)


def drive_standalone(client: HostClient, run: RunSnapshot) -> RunSnapshot:
    store: RunStore[RunSnapshot, StepSnapshot, HostClient] = HttpRunStore(client)
    return engine.execute_run(store, build_remote_registry(), run, profile=RunProfile("a", "b"))


def views_accept_plain_host_objects(run: Run, step: Step, snapshot: RunSnapshot) -> None:
    run_view: RunView = run
    step_view: StepView = step
    snapshot_view: RunView = snapshot
    profile_view: ProfileView = RunProfile("ollama", "qwen3:8b", None)
    handler: ToolHandler[Session, Run, Step, RunProfile] = model_call
    del run_view, step_view, snapshot_view, profile_view, handler
