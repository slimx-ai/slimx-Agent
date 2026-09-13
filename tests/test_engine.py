"""Engine semantics over an in-memory RunStore — the behavior contract hosts rely on."""

from __future__ import annotations

import itertools
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


def test_a_run_with_an_earlier_failed_step_is_marked_failed_on_the_next_drive():
    """Recorded legacy behavior: no handler runs, no second RUN_FAILED event is appended, and
    the epilogue does not fire again."""
    calls: list[str] = []
    store = MemoryStore(
        FakeRun("r"),
        [
            FakeStep("s1", "model_call", status="completed"),
            FakeStep("s2", "model_call", status="failed"),
            FakeStep("s3", "model_call"),
        ],
    )
    ends: list[str] = []
    engine.execute_run(
        store,
        _registry(lambda c, r, s, p: calls.append(s.id) or {}),
        store.run,
        profile=object(),
        on_run_end=lambda r, s: ends.append(s),
    )
    assert store.run.status == "failed"
    assert calls == [] and ends == []
    assert _types(store) == []


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
