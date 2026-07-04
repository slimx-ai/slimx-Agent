"""Engine semantics over an in-memory RunStore — the behavior contract hosts rely on."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from slimx_agent import contracts, engine
from slimx_agent.store import UNSET
from slimx_agent.tools import StepExecutionError, StepNotApplicable, ToolRegistry


@dataclass
class FakeRun:
    id: str
    status: str = "planned"
    approval_policy: str | None = "auto_complete"
    auto_approve: bool = False
    allowed_tools_json: list[str] | None = None


@dataclass
class FakeStep:
    id: str
    type: str
    title: str = "step"
    status: str = "pending"
    requires_approval: bool = False
    error: str | None = None
    output_refs: dict[str, Any] | None = None


@dataclass
class MemoryStore:
    run: FakeRun
    steps: list[FakeStep]
    events: list[dict[str, Any]] = field(default_factory=list)
    rollbacks: int = 0
    _seq: itertools.count = field(default_factory=lambda: itertools.count(1))

    handler_context: Any = "host-context"

    def get_run(self, run_id):
        return self.run if self.run.id == run_id else None

    def get_steps(self, run_id):
        return list(self.steps)

    def get_step(self, step_id):
        return next((s for s in self.steps if s.id == step_id), None)

    def set_run_status(self, run, status):
        self.run.status = status
        return self.run

    def set_step_state(self, step_id, status, *, error=UNSET, output_refs=UNSET):
        step = self.get_step(step_id)
        assert step is not None
        step.status = status
        if error is not UNSET:
            step.error = error
        if output_refs is not UNSET:
            step.output_refs = output_refs
        return step

    def rollback(self):
        self.rollbacks += 1

    def append_event(self, run_id, type, *, step_id=None, payload=None, commit=True):
        event = {
            "sequence": next(self._seq),
            "type": type,
            "agent_step_id": step_id,
            "payload_json": payload,
        }
        self.events.append(event)
        return event

    def next_sequence(self, run_id):
        return len(self.events) + 1

    def drained_events(self, run_id, after_sequence):
        return [e for e in self.events if e["sequence"] > after_sequence]


def _registry(handler=None):
    registry = ToolRegistry()
    registry.register("model_call", handler or (lambda ctx, run, step, profile: {"ref": "r1"}))
    registry.register("web_search", lambda ctx, run, step, profile: {"result_count": 1})
    return registry


def _types(store):
    return [e["type"] for e in store.events]


def test_happy_path_completes_and_streams_every_event():
    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "model_call"), FakeStep("s2", "model_call")])
    seen_context = {}

    def handler(ctx, run, step, profile):
        seen_context["ctx"] = ctx
        return {"ref": step.id}

    yielded = list(
        engine.execute_run_events(store, _registry(handler), store.run, profile=object())
    )
    assert store.run.status == "completed"
    assert [s.status for s in store.steps] == ["completed", "completed"]
    assert store.steps[0].output_refs == {"ref": "s1"}
    assert seen_context["ctx"] == "host-context"
    assert _types(store) == [
        contracts.STEP_STARTED,
        contracts.STEP_COMPLETED,
        contracts.STEP_STARTED,
        contracts.STEP_COMPLETED,
        contracts.RUN_COMPLETED,
    ]
    # Every persisted event was also yielded, in order.
    assert [p["sequence"] for _, p in yielded] == [e["sequence"] for e in store.events]


def test_ungranted_tool_skips_before_the_approval_gate():
    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "web_search"), FakeStep("s2", "model_call")])
    run = engine.execute_run(store, _registry(), store.run, profile=object())
    assert run.status == "completed"
    assert store.steps[0].status == "skipped"
    assert contracts.APPROVAL_REQUIRED not in _types(store)
    skip = next(e for e in store.events if e["type"] == contracts.STEP_SKIPPED)
    assert "Web search" in skip["payload_json"]["reason"]


def test_hard_gate_parks_even_in_auto_complete_and_resumes_after_approval():
    store = MemoryStore(
        FakeRun("r", allowed_tools_json=["web_search"]), [FakeStep("s1", "web_search")]
    )
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.run.status == "awaiting_approval"
    assert store.steps[0].status == "awaiting_approval"
    assert contracts.APPROVAL_REQUIRED in _types(store)

    # Host approves (its route), then re-executes: the approved step runs to completion.
    store.set_step_state("s1", "approved")
    store.set_run_status(store.run, "planned")
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.run.status == "completed"
    assert store.steps[0].status == "completed"


def test_legacy_policy_honors_planner_flag_and_auto_approve():
    gated = FakeStep("s1", "model_call", requires_approval=True)
    store = MemoryStore(FakeRun("r", approval_policy=None), [gated])
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.run.status == "awaiting_approval"

    # auto_approve clears the planner flag — with the trail showing why it proceeded.
    gated2 = FakeStep("s1", "model_call", requires_approval=True)
    store2 = MemoryStore(FakeRun("r", approval_policy=None, auto_approve=True), [gated2])
    engine.execute_run(store2, _registry(), store2.run, profile=object())
    assert store2.run.status == "completed"
    granted = next(e for e in store2.events if e["type"] == contracts.APPROVAL_GRANTED)
    assert granted["payload_json"] == {"auto": True}


def test_step_failure_fails_the_run_and_fires_on_run_end():
    def boom(ctx, run, step, profile):
        raise StepExecutionError("model exploded")

    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "model_call")])
    ends: list[str] = []
    engine.execute_run(
        store, _registry(boom), store.run, profile=object(), on_run_end=lambda r, s: ends.append(s)
    )
    assert store.run.status == "failed"
    assert store.steps[0].error == "model exploded"
    assert contracts.RUN_FAILED in _types(store)
    assert ends == ["failed"]


def test_unexpected_handler_crash_rolls_back_and_fails_the_step():
    def crash(ctx, run, step, profile):
        raise RuntimeError("db went away")

    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "model_call")])
    engine.execute_run(store, _registry(crash), store.run, profile=object())
    assert store.rollbacks == 1
    assert store.steps[0].status == "failed"
    assert "RuntimeError" in (store.steps[0].error or "")


def test_not_applicable_skips_and_the_run_continues():
    def skip(ctx, run, step, profile):
        raise StepNotApplicable("nothing to do")

    registry = ToolRegistry()
    registry.register("model_call", skip)
    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "model_call")])
    run = engine.execute_run(store, registry, store.run, profile=object())
    assert run.status == "completed"
    assert store.steps[0].status == "skipped"


def test_cancel_during_the_last_step_is_not_overwritten_by_completion():
    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "model_call")])

    def cancelling(ctx, run, step, profile):
        store.run.status = "cancelled"
        return {}

    ends: list[str] = []
    engine.execute_run(
        store,
        _registry(cancelling),
        store.run,
        profile=object(),
        on_run_end=lambda r, s: ends.append(s),
    )
    assert store.run.status == "cancelled"
    assert contracts.RUN_COMPLETED not in _types(store)
    assert ends == []  # user decisions are never auto-extended


def test_terminal_run_is_a_no_op():
    store = MemoryStore(FakeRun("r", status="completed"), [FakeStep("s1", "model_call")])
    assert list(engine.execute_run_events(store, _registry(), store.run, profile=object())) == []
    assert store.events == []


def test_unregistered_step_type_fails_honestly():
    registry = ToolRegistry()
    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "model_call")])
    engine.execute_run(store, registry, store.run, profile=object())
    assert store.steps[0].status == "failed"
    assert "Unsupported step type" in (store.steps[0].error or "")


def test_running_transition_leaves_prior_error_untouched():
    """The UNSET sentinel contract: marking a step running must not clear a prior error the
    host chose to keep (rerun flows clear it explicitly)."""
    step = FakeStep("s1", "model_call", error="previous failure")
    store = MemoryStore(FakeRun("r"), [step])
    engine.run_step(store, _registry(), store.run, step, profile=object())
    # completed clears it; but the intermediate 'running' write must not have been the one
    # to do it — assert via a fresh store where the handler fails BEFORE completion.
    step2 = FakeStep("s2", "model_call", error="previous failure")
    store2 = MemoryStore(FakeRun("r"), [step2])

    def crash(ctx, run, step, profile):
        assert step.error == "previous failure"  # still present while running
        raise StepExecutionError("new failure")

    engine.run_step(store2, _registry(crash), store2.run, step2, profile=object())
    assert step2.error == "new failure"


def test_policies_reexports_cover_the_gate_api():
    from slimx_agent import policies

    assert policies.classify_step(FakeStep("s", "web_search"))[0] == policies.HARD_GATED
    assert policies.requires_stop("auto_complete", policies.AUTO_SAFE, False) is False
    assert policies.normalize_grants(["web_search", "nope", "web_search"]) == ["web_search"]
