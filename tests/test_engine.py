"""Engine semantics over an in-memory RunStore — the behavior contract hosts rely on."""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from slimx_agent import contracts, engine
from slimx_agent.store import UNSET
from slimx_agent.tools import (
    StepActionPrepared,
    StepExecutionError,
    StepNotApplicable,
    StepOutcomeUnknown,
    ToolRegistry,
)


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


def test_the_step_status_vocabulary_is_closed_and_exact():
    assert contracts.STEP_STATUSES == frozenset(
        {"pending", "awaiting_approval", "approved", "running", "completed", "failed", "skipped"}
    )


@pytest.mark.parametrize("status", ["queued", "Pending", "prepared", "cancelled", ""])
def test_an_unrecognized_step_status_is_refused_not_dispatched_past_the_gates(status):
    # An ungranted, hard-gated tool: a status that slipped past both gates would dispatch it.
    store = MemoryStore(
        FakeRun("r", allowed_tools_json=None), [FakeStep("s1", "web_search", status=status)]
    )
    with pytest.raises(engine.UnknownStepStatus) as caught:
        engine.execute_run(store, _registry(), store.run, profile=object())
    assert caught.value.step_id == "s1"
    assert store.steps[0].status == status
    assert store.events == []


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


def test_prepared_action_stops_cleanly_and_next_drive_applies_policy():
    calls = 0
    store = MemoryStore(
        FakeRun("r", approval_policy="auto_complete"),
        [FakeStep("s1", "model_call", status="approved", requires_approval=True)],
    )

    def prepare(ctx, run, step, profile):
        nonlocal calls
        calls += 1
        step.status = "awaiting_approval"
        step.requires_approval = False
        store.run.status = "awaiting_approval"
        raise StepActionPrepared("candidate ready")

    engine.execute_run(store, _registry(prepare), store.run, profile=object())

    assert calls == 1
    assert store.run.status == "awaiting_approval"
    assert store.steps[0].status == "awaiting_approval"
    assert contracts.STEP_COMPLETED not in _types(store)
    assert contracts.STEP_FAILED not in _types(store)
    assert contracts.STEP_SKIPPED not in _types(store)
    assert contracts.APPROVAL_GRANTED not in _types(store)

    store.set_run_status(store.run, "planned")
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.run.status == "completed"
    assert store.steps[0].status == "completed"
    assert contracts.APPROVAL_GRANTED in _types(store)


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


# --- Deep-research loop (0.9): mid-run plan extension + engine-enforced budgets ---


def test_steps_appended_mid_run_join_the_same_drive():
    """The re-fetching loop's contract: a handler that APPENDS steps (research_iterate's plan
    extension) sees them executed in this same drive, in order, and the run completes."""
    store = MemoryStore(
        FakeRun("r"), [FakeStep("s1", "model_call"), FakeStep("s2", "research_iterate")]
    )
    executed: list[str] = []

    def extend(ctx, run, step, profile):
        executed.append(step.id)
        store.steps.append(FakeStep("s3", "model_call", title="follow-up"))
        return {"added_steps": 1}

    registry = ToolRegistry()
    registry.register(
        "model_call", lambda ctx, run, step, profile: executed.append(step.id) or {"ref": step.id}
    )
    registry.register("research_iterate", extend)
    engine.execute_run(store, registry, store.run, profile=object())
    assert executed == ["s1", "s2", "s3"]
    assert store.run.status == "completed"
    assert [s.status for s in store.steps] == ["completed", "completed", "completed"]


def test_step_budget_pauses_the_run_honestly():
    """Exhausting budget_max_steps stops BEFORE the next step with a BUDGET_EXHAUSTED event and
    a paused (resumable) run — never a silent truncation, never a fake completion."""
    run = FakeRun("r")
    run.budget_max_steps = 1
    store = MemoryStore(run, [FakeStep("s1", "model_call"), FakeStep("s2", "model_call")])
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.run.status == "paused"
    assert [s.status for s in store.steps] == ["completed", "pending"]
    types = _types(store)
    assert contracts.BUDGET_EXHAUSTED in types
    assert contracts.RUN_PAUSED in types
    assert contracts.RUN_COMPLETED not in types
    exhausted = next(e for e in store.events if e["type"] == contracts.BUDGET_EXHAUSTED)
    assert "step budget" in exhausted["payload_json"]["reason"]


def test_wall_budget_pauses_between_steps(monkeypatch):
    run = FakeRun("r")
    run.budget_max_wall_seconds = 10
    store = MemoryStore(run, [FakeStep("s1", "model_call"), FakeStep("s2", "model_call")])
    # anchor(0) -> first check within budget(1) -> s1 runs -> second check exceeds(100)
    clock = iter([0.0, 1.0, 100.0, 200.0])
    monkeypatch.setattr(engine, "_now", lambda: next(clock))
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.run.status == "paused"
    assert [s.status for s in store.steps] == ["completed", "pending"]
    exhausted = next(e for e in store.events if e["type"] == contracts.BUDGET_EXHAUSTED)
    assert "time budget" in exhausted["payload_json"]["reason"]


def test_budget_less_runs_are_unbounded_and_unchanged():
    """Legacy hosts/rows carry no budget attributes at all — getattr defaults keep them running."""
    store = MemoryStore(FakeRun("r"), [FakeStep(f"s{i}", "model_call") for i in range(5)])
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.run.status == "completed"
    assert contracts.BUDGET_EXHAUSTED not in _types(store)


def test_research_iterate_classifies_auto_safe():
    from slimx_agent import policies

    tier, reason = policies.classify_step(FakeStep("s", "research_iterate"))
    assert tier == policies.AUTO_SAFE
    assert "extend the plan" in reason
    assert policies.required_grant("research_iterate") is None


def test_preapproved_web_search_clears_the_hard_gate_with_an_audited_grant():
    """Scoped pre-approval (0.14): a run whose duck-typed ``preapproved_tools`` names
    web_search runs it WITHOUT parking, and the APPROVAL_GRANTED trail says why."""

    @dataclass
    class PreapprovedRun(FakeRun):
        preapproved_tools: list[str] | None = None

    run = PreapprovedRun("r", allowed_tools_json=["web_search"], preapproved_tools=["web_search"])
    store = MemoryStore(run, [FakeStep("s1", "web_search"), FakeStep("s2", "model_call")])
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.run.status == "completed"
    assert [s.status for s in store.steps] == ["completed", "completed"]
    assert contracts.APPROVAL_REQUIRED not in _types(store)
    granted = next(e for e in store.events if e["type"] == contracts.APPROVAL_GRANTED)
    assert granted["payload_json"]["preapproved"] is True
    assert granted["payload_json"]["classification"] == "hard_gated"

    # manual policy still stops — pre-approval only downgrades the HARD gate, it never
    # overrides plan-review-first.
    run2 = PreapprovedRun(
        "r",
        approval_policy="manual",
        allowed_tools_json=["web_search"],
        preapproved_tools=["web_search"],
    )
    store2 = MemoryStore(run2, [FakeStep("s1", "web_search")])
    engine.execute_run(store2, _registry(), store2.run, profile=object())
    assert store2.run.status == "awaiting_approval"


def test_preapproval_allowlist_is_enforced_in_the_engine():
    """A stored pre-approval outside PREAPPROVABLE_STEP_TYPES (e.g. mcp_call) can never
    clear a hard gate — the engine checks membership itself, not just the host route."""
    from slimx_agent import policies

    assert "web_search" in policies.PREAPPROVABLE_STEP_TYPES
    assert "mcp_call" not in policies.PREAPPROVABLE_STEP_TYPES

    @dataclass
    class PreapprovedRun(FakeRun):
        preapproved_tools: list[str] | None = None

    registry = ToolRegistry()
    registry.register("mcp_call", lambda ctx, run, step, profile: {"ok": True})
    run = PreapprovedRun("r", allowed_tools_json=["mcp_tools"], preapproved_tools=["mcp_call"])
    store = MemoryStore(run, [FakeStep("s1", "mcp_call")])
    engine.execute_run(store, registry, store.run, profile=object())
    # Parked at the gate (or skipped by the permission gate) — never auto-run.
    assert store.run.status != "completed" or store.steps[0].status != "completed"
    assert not any(
        (e["payload_json"] or {}).get("preapproved")
        for e in store.events
        if e["type"] == contracts.APPROVAL_GRANTED
    )


# --- 0.20: grants at dispatch, prepared generations, races, and unknown outcomes ---------

_TERMINAL_STEP_EVENTS = {contracts.STEP_COMPLETED, contracts.STEP_FAILED, contracts.STEP_SKIPPED}


def _counting_registry(step_type: str, handler):
    registry = ToolRegistry()
    registry.register(step_type, handler)
    return registry


def test_approved_step_whose_grant_was_revoked_is_skipped_not_dispatched():
    """Approval never bypasses a missing grant: the permission gate re-checks an already
    approved step against the fresh run before dispatch."""
    calls: list[str] = []
    registry = _counting_registry("mcp_call", lambda c, r, s, p: calls.append(s.id) or {})
    store = MemoryStore(
        FakeRun("r", allowed_tools_json=[]), [FakeStep("s1", "mcp_call", status="approved")]
    )
    engine.execute_run(store, registry, store.run, profile=object())
    assert calls == []
    assert store.steps[0].status == "skipped"
    assert contracts.APPROVAL_REQUIRED not in _types(store)
    skipped = next(e for e in store.events if e["type"] == contracts.STEP_SKIPPED)
    assert "Connector tools (MCP)" in skipped["payload_json"]["reason"]


def test_approved_step_that_keeps_its_grant_runs_without_re_gating():
    calls: list[str] = []
    registry = _counting_registry("mcp_call", lambda c, r, s, p: calls.append(s.id) or {})
    store = MemoryStore(
        FakeRun("r", allowed_tools_json=["mcp_tools"]),
        [FakeStep("s1", "mcp_call", status="approved")],
    )
    engine.execute_run(store, registry, store.run, profile=object())
    assert calls == ["s1"]
    assert store.steps[0].status == "completed"
    assert contracts.APPROVAL_REQUIRED not in _types(store)


@pytest.mark.parametrize("policy", [None, "manual", "review_checkpoints", "auto_complete"])
def test_permission_gate_precedes_approval_under_every_policy(policy):
    store = MemoryStore(
        FakeRun("r", approval_policy=policy, allowed_tools_json=None),
        [FakeStep("s1", "mcp_call", requires_approval=True)],
    )
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.steps[0].status == "skipped"
    assert contracts.APPROVAL_REQUIRED not in _types(store)


def test_a_junk_string_preapproval_never_clears_a_hard_gate():
    run = FakeRun("r", allowed_tools_json=["web_search"])
    run.preapproved_tools = "web_search"  # a string, not a list: grants no pre-approval
    store = MemoryStore(run, [FakeStep("s1", "web_search")])
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.run.status == "awaiting_approval"
    assert not any(
        (e["payload_json"] or {}).get("preapproved")
        for e in store.events
        if e["type"] == contracts.APPROVAL_GRANTED
    )


def test_a_prepared_generation_never_reuses_the_earlier_approval():
    """The earlier generation was approved; preparing a new one re-parks it as pending, and the
    same drive re-applies the hard gate to it instead of dispatching on the old approval."""
    calls = 0
    store = MemoryStore(
        FakeRun("r", allowed_tools_json=["mcp_tools"]),
        [FakeStep("s1", "mcp_call", status="approved")],
    )

    def prepare(ctx, run, step, profile):
        nonlocal calls
        calls += 1
        store.steps[0].status = "pending"
        raise StepActionPrepared("new action generation")

    engine.execute_run(store, _counting_registry("mcp_call", prepare), store.run, profile=object())
    assert calls == 1
    assert store.steps[0].status == "awaiting_approval"
    assert store.run.status == "awaiting_approval"
    types = _types(store)
    assert types.count(contracts.APPROVAL_REQUIRED) == 1
    assert not (_TERMINAL_STEP_EVENTS | {contracts.APPROVAL_GRANTED}) & set(types)


@pytest.mark.parametrize(
    ("policy", "initial_status", "expected_calls", "expected_status"),
    [
        ("auto_complete", "pending", 2, "completed"),
        ("manual", "approved", 1, "awaiting_approval"),
        ("review_checkpoints", "approved", 1, "awaiting_approval"),
    ],
)
def test_prepared_file_work_continues_only_where_policy_allows(
    policy, initial_status, expected_calls, expected_status
):
    calls = 0
    store = MemoryStore(
        FakeRun("r", approval_policy=policy),
        [FakeStep("s1", "write_file", status=initial_status)],
    )

    def two_stage(ctx, run, step, profile):
        nonlocal calls
        calls += 1
        if calls == 1:
            store.steps[0].status = "pending"
            raise StepActionPrepared("content prepared")
        return {"path": "index.html"}

    engine.execute_run(
        store, _counting_registry("write_file", two_stage), store.run, profile=object()
    )
    assert calls == expected_calls
    assert store.steps[0].status == expected_status
    assert _types(store).count(contracts.STEP_COMPLETED) == (expected_calls - 1)


def test_pause_between_preparation_and_dispatch_is_honored():
    calls = 0
    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "write_file")])

    def prepare_then_pause(ctx, run, step, profile):
        nonlocal calls
        calls += 1
        store.steps[0].status = "pending"
        store.run.status = "paused"
        raise StepActionPrepared("prepared while the user paused")

    ends: list[str] = []
    engine.execute_run(
        store,
        _counting_registry("write_file", prepare_then_pause),
        store.run,
        profile=object(),
        on_run_end=lambda r, s: ends.append(s),
    )
    assert calls == 1
    assert store.run.status == "paused"
    assert store.steps[0].status == "pending"
    assert not _TERMINAL_STEP_EVENTS & set(_types(store))
    assert ends == []


def test_pause_during_the_last_step_is_not_overwritten_by_completion():
    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "model_call")])

    def pausing(ctx, run, step, profile):
        store.run.status = "paused"
        return {"ref": "r1"}

    ends: list[str] = []
    engine.execute_run(
        store,
        _registry(pausing),
        store.run,
        profile=object(),
        on_run_end=lambda r, s: ends.append(s),
    )
    assert store.steps[0].status == "completed"
    assert store.run.status == "paused"
    assert contracts.RUN_COMPLETED not in _types(store)
    assert ends == []


@dataclass
class PauseAfterLoopRead(MemoryStore):
    """A pause that lands after the loop's last run read but before the completion read."""

    reads: int = 0

    def get_run(self, run_id):
        self.reads += 1
        if self.reads == 3:  # 1: first loop pass, 2: loop pass that finds no step, 3: final
            self.run.status = "paused"
        return super().get_run(run_id)


def test_a_pause_landing_after_the_last_loop_read_is_still_honored():
    store = PauseAfterLoopRead(FakeRun("r"), [FakeStep("s1", "model_call")])
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.steps[0].status == "completed"
    assert store.run.status == "paused"
    assert contracts.RUN_COMPLETED not in _types(store)


def test_an_unknown_outcome_ends_the_drive_without_terminal_writes_or_retry():
    calls = 0

    def unobserved(ctx, run, step, profile):
        nonlocal calls
        calls += 1
        raise StepOutcomeUnknown("the invocation response was lost")

    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "model_call"), FakeStep("s2", "model_call")])
    ends: list[str] = []
    with pytest.raises(StepOutcomeUnknown):
        engine.execute_run(
            store,
            _registry(unobserved),
            store.run,
            profile=object(),
            on_run_end=lambda r, s: ends.append(s),
        )
    assert calls == 1
    assert [s.status for s in store.steps] == ["running", "pending"]
    assert store.steps[0].error is None
    assert _types(store) == [contracts.STEP_STARTED]
    assert store.run.status == "running"
    assert store.rollbacks == 0
    assert ends == []


def test_a_reported_step_failure_is_not_reported_again_on_a_re_drive():
    """A host re-opens a failed run without resetting the failed step. The next drive fails the
    run again, but the failure was already reported: no handler runs, no second RUN_FAILED is
    appended, and the epilogue does not fire again."""
    calls: list[str] = []

    def fail_second(ctx, run, step, profile):
        calls.append(step.id)
        if step.id == "s2":
            raise StepExecutionError("model exploded")
        return {}

    store = MemoryStore(
        FakeRun("r"),
        [FakeStep("s1", "model_call"), FakeStep("s2", "model_call"), FakeStep("s3", "model_call")],
    )
    ends: list[str] = []

    def drive():
        engine.execute_run(
            store,
            _registry(fail_second),
            store.run,
            profile=object(),
            on_run_end=lambda r, s: ends.append(s),
        )

    drive()
    assert store.run.status == "failed" and ends == ["failed"] and calls == ["s1", "s2"]
    first_drive = _types(store)
    assert first_drive.count(contracts.RUN_FAILED) == 1

    store.run.status = "planned"  # the host re-opens the run; s2 stays failed
    drive()

    assert store.run.status == "failed"
    assert calls == ["s1", "s2"] and ends == ["failed"]
    assert _types(store) == first_drive


def test_a_step_failure_nobody_reported_ends_the_run_with_its_terminal_event_and_hook():
    """BUG-03: a step the host itself marked failed, with no ``agent.run.failed`` in the log, used
    to end the run silently. The run now gets exactly one terminal event and one hook call."""
    calls: list[str] = []
    store = MemoryStore(
        FakeRun("r"),
        [
            FakeStep("s1", "model_call", status="completed"),
            FakeStep("s2", "model_call", status="failed", error="host resolved it as failed"),
            FakeStep("s3", "model_call"),
        ],
    )
    ends: list[str] = []
    streamed = [
        payload["type"]
        for _kind, payload in engine.execute_run_events(
            store,
            _registry(lambda c, r, s, p: calls.append(s.id) or {}),
            store.run,
            profile=object(),
            on_run_end=lambda r, s: ends.append(s),
        )
    ]

    assert store.run.status == "failed"
    assert calls == [] and ends == ["failed"]
    assert _types(store) == [contracts.RUN_FAILED] == streamed
    assert store.events[0]["agent_step_id"] == "s2"
    assert [s.status for s in store.steps] == ["completed", "failed", "pending"]
    assert store.steps[1].error == "host resolved it as failed"

    store.run.status = "planned"  # re-opened again: that failure is now reported
    engine.execute_run(
        store, _registry(), store.run, profile=object(), on_run_end=lambda r, s: ends.append(s)
    )
    assert _types(store) == [contracts.RUN_FAILED] and ends == ["failed"]


def test_a_step_the_host_fails_mid_drive_ends_the_run_with_its_terminal_event_and_hook():
    store = MemoryStore(FakeRun("r"), [FakeStep("s1", "model_call"), FakeStep("s2", "model_call")])

    def fails_the_next_step(ctx, run, step, profile):
        store.steps[1].status = "failed"  # an out-of-band host write, not an engine outcome
        return {}

    ends: list[str] = []
    engine.execute_run(
        store,
        _registry(fails_the_next_step),
        store.run,
        profile=object(),
        on_run_end=lambda r, s: ends.append(s),
    )

    assert store.run.status == "failed" and ends == ["failed"]
    assert _types(store) == [contracts.STEP_STARTED, contracts.STEP_COMPLETED, contracts.RUN_FAILED]
    assert store.events[-1]["agent_step_id"] == "s2"


def test_a_reported_failure_is_matched_by_step_id_text_not_object_identity():
    """ControlRoom holds UUID step ids while its wire payload carries their string form."""
    step_id = uuid.UUID("00000000-0000-0000-0000-0000000000a2")
    store = MemoryStore(FakeRun("r"), [FakeStep(step_id, "model_call", status="failed")])  # type: ignore[arg-type]
    store.append_event("r", contracts.RUN_FAILED, step_id=str(step_id))
    ends: list[str] = []

    engine.execute_run(
        store, _registry(), store.run, profile=object(), on_run_end=lambda r, s: ends.append(s)
    )

    assert store.run.status == "failed" and ends == []
    assert _types(store) == [contracts.RUN_FAILED]


def test_a_failure_reported_for_another_step_does_not_silence_this_one():
    store = MemoryStore(
        FakeRun("r"),
        [
            FakeStep("s1", "model_call", status="skipped"),
            FakeStep("s2", "model_call", status="failed"),
        ],
    )
    store.append_event("r", contracts.RUN_FAILED, step_id="s1")
    ends: list[str] = []

    engine.execute_run(
        store, _registry(), store.run, profile=object(), on_run_end=lambda r, s: ends.append(s)
    )

    assert ends == ["failed"]
    assert [e["agent_step_id"] for e in store.events] == ["s1", "s2"]


@dataclass
class TerminalWriteCountingStore(MemoryStore):
    """Counts the run-status writes that end a run, so every exit can be checked for pairing."""

    terminal_writes: list[str] = field(default_factory=list)

    def set_run_status(self, run, status):
        if status in ("completed", "failed"):
            self.terminal_writes.append(status)
        return super().set_run_status(run, status)


def _raise(exc: BaseException):
    def handler(ctx, run, step, profile):
        raise exc

    return handler


def _pause(ctx, run, step, profile):
    run.status = "paused"
    return {}


def _cancel(ctx, run, step, profile):
    run.status = "cancelled"
    return {}


@pytest.mark.parametrize(
    ("name", "run", "steps", "handler", "terminal"),
    [
        ("completes", {}, [FakeStep("a", "model_call")], None, "completed"),
        ("all skipped", {}, [FakeStep("a", "web_search")], None, "completed"),
        ("no steps", {}, [], None, "completed"),
        (
            "fails in the drive",
            {},
            [FakeStep("a", "model_call")],
            _raise(StepExecutionError("boom")),
            "failed",
        ),
        (
            "crashes in the drive",
            {},
            [FakeStep("a", "model_call")],
            _raise(ValueError("x")),
            "failed",
        ),
        ("unsupported type", {}, [FakeStep("a", "never_registered")], None, "failed"),
        (
            "already failed, unreported",
            {},
            [FakeStep("a", "model_call", status="failed")],
            None,
            "failed",
        ),
        ("paused by the last step", {}, [FakeStep("a", "model_call")], _pause, None),
        ("cancelled by the last step", {}, [FakeStep("a", "model_call")], _cancel, None),
        (
            "unknown outcome",
            {},
            [FakeStep("a", "model_call")],
            _raise(StepOutcomeUnknown("lost")),
            None,
        ),
        (
            "stops at the approval gate",
            {"approval_policy": "manual"},
            [FakeStep("a", "model_call", requires_approval=True)],
            None,
            None,
        ),
        (
            "step budget exhausted",
            {"budget_max_steps": 1},
            [FakeStep("a", "model_call", status="completed"), FakeStep("b", "model_call")],
            None,
            None,
        ),
        ("already terminal", {"status": "cancelled"}, [FakeStep("a", "model_call")], None, None),
        (
            "running step lost its grant",
            {"allowed_tools_json": []},
            [FakeStep("a", "web_search", status="running")],
            None,
            None,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_engine_exit_pairs_a_terminal_status_with_one_event_and_one_hook_call(
    name, run, steps, handler, terminal
):
    """The completion invariant: a drive writes a terminal run status exactly when it appends
    that status's terminal event and calls ``on_run_end`` once. Pause, cancel, an approval stop,
    a budget pause, an unknown outcome and a refused re-entry write none of the three."""
    budget = {k: run.pop(k) for k in list(run) if k.startswith("budget_")}
    fake_run = FakeRun("r", **run)
    for key, value in budget.items():
        setattr(fake_run, key, value)
    store = TerminalWriteCountingStore(fake_run, steps)
    ends: list[str] = []

    try:
        engine.execute_run(
            store,
            _registry(handler),
            store.run,
            profile=object(),
            on_run_end=lambda r, s: ends.append(s),
        )
    except (StepOutcomeUnknown, engine.RunningStepNotPermitted):
        pass

    terminal_events = [
        {contracts.RUN_COMPLETED: "completed", contracts.RUN_FAILED: "failed"}[t]
        for t in _types(store)
        if t in (contracts.RUN_COMPLETED, contracts.RUN_FAILED)
    ]
    expected = [terminal] if terminal else []
    assert store.terminal_writes == expected, name
    assert terminal_events == expected, name
    assert ends == expected, name


def test_an_awaiting_step_is_auto_approved_after_a_mid_run_policy_change():
    store = MemoryStore(
        FakeRun("r", approval_policy="auto_complete"),
        [FakeStep("s1", "compare_models", status="awaiting_approval")],
    )
    registry = _counting_registry("compare_models", lambda c, r, s, p: {"ok": True})
    engine.execute_run(store, registry, store.run, profile=object())
    granted = next(e for e in store.events if e["type"] == contracts.APPROVAL_GRANTED)
    assert granted["payload_json"] == {
        "auto": True,
        "policy": "auto_complete",
        "classification": "review_recommended",
    }
    assert store.steps[0].status == "completed"


@dataclass
class RefusingReentryStore(MemoryStore):
    """A host that refuses to re-enter a step whose earlier attempt may have crossed its entry
    boundary — the obligation ``RunStore.set_step_state`` documents for ``running``."""

    def set_step_state(self, step_id, status, *, error=UNSET, output_refs=UNSET):
        current = self.get_step(step_id)
        if status == "running" and current is not None and current.status == "running":
            raise RuntimeError("the earlier attempt already crossed the host entry boundary")
        return super().set_step_state(step_id, status, error=error, output_refs=output_refs)


def test_an_interrupted_running_step_is_re_entered_only_through_the_running_transition():
    """Recorded contract: the engine has no ledger, so it delegates the at-most-once decision
    for a step an interrupted drive left ``running`` to the store's ``running`` transition. A
    refusal propagates before the handler runs and before any other write."""
    calls: list[str] = []
    store = RefusingReentryStore(FakeRun("r"), [FakeStep("s1", "model_call", status="running")])
    with pytest.raises(RuntimeError, match="entry boundary"):
        engine.execute_run(
            store,
            _registry(lambda c, r, s, p: calls.append(s.id) or {}),
            store.run,
            profile=object(),
        )
    assert calls == []
    assert store.events == []
    assert store.steps[0].status == "running"

    # A host that knows the earlier attempt never entered (no ledger row) may allow it: the
    # step is dispatched exactly once more.
    permissive = MemoryStore(FakeRun("r"), [FakeStep("s1", "model_call", status="running")])
    engine.execute_run(
        permissive,
        _registry(lambda c, r, s, p: calls.append(s.id) or {}),
        permissive.run,
        profile=object(),
    )
    assert calls == ["s1"]
    assert permissive.steps[0].status == "completed"


@pytest.mark.parametrize("grants", [None, [], ["code_read"]], ids=["legacy", "empty", "other"])
def test_a_running_step_whose_grant_is_gone_is_neither_re_entered_nor_skipped(grants):
    """RISK-02: an interrupted drive left a granted step ``running`` and the grant is now absent.
    Approval never bypasses a missing grant, so the handler is not re-entered. The earlier attempt
    may already have run, so no terminal ``skipped`` is written either: the drive ends with
    nothing written for the step, even on a store that would accept every write."""
    calls: list[str] = []
    store = MemoryStore(
        FakeRun("r", allowed_tools_json=grants),
        [FakeStep("s1", "web_search", status="running"), FakeStep("s2", "model_call")],
    )
    ends: list[str] = []
    with pytest.raises(engine.RunningStepNotPermitted) as refused:
        engine.execute_run(
            store,
            _counting_registry("web_search", lambda c, r, s, p: calls.append(s.id) or {}),
            store.run,
            profile=object(),
            on_run_end=lambda r, s: ends.append(s),
        )

    assert refused.value.step_id == "s1"
    assert "not enabled for this run" in refused.value.reason
    assert calls == [] and ends == []
    assert store.events == []
    assert [s.status for s in store.steps] == ["running", "pending"]
    assert store.steps[0].error is None
    assert store.run.status == "running"


def test_a_running_step_that_keeps_its_grant_is_still_re_entered_through_the_store():
    calls: list[str] = []
    store = MemoryStore(
        FakeRun("r", allowed_tools_json=["web_search"]),
        [FakeStep("s1", "web_search", status="running")],
    )
    engine.execute_run(
        store,
        _counting_registry("web_search", lambda c, r, s, p: calls.append(s.id) or {}),
        store.run,
        profile=object(),
    )
    assert calls == ["s1"]
    assert store.steps[0].status == "completed" and store.run.status == "completed"


def test_a_running_step_that_needs_no_grant_is_unaffected_by_the_run_grants():
    calls: list[str] = []
    store = MemoryStore(
        FakeRun("r", allowed_tools_json=[]), [FakeStep("s1", "model_call", status="running")]
    )
    engine.execute_run(
        store, _registry(lambda c, r, s, p: calls.append(s.id) or {}), store.run, profile=object()
    )
    assert calls == ["s1"] and store.run.status == "completed"


@pytest.mark.parametrize(
    ("policy", "expected_step_status"),
    [("auto_complete", "failed"), ("manual", "awaiting_approval")],
)
def test_an_unknown_step_type_cannot_execute(policy, expected_step_status):
    store = MemoryStore(FakeRun("r", approval_policy=policy), [FakeStep("s1", "rm_rf")])
    engine.execute_run(store, _registry(), store.run, profile=object())
    assert store.steps[0].status == expected_step_status
    if expected_step_status == "failed":
        assert "Unsupported step type 'rm_rf'" in (store.steps[0].error or "")
