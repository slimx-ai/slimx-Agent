"""The agent run engine: the deliberately boring dispatch loop.

For each runnable step it applies the tool-permission gate (ungranted tools skip honestly),
the deterministic approval gate (hard gates stop even in Auto-complete), dispatches through
the :class:`~slimx_agent.tools.ToolRegistry` (the ONLY path to a tool implementation),
persists transitions through a :class:`~slimx_agent.store.RunStore`, and emits the durable
event vocabulary from :mod:`slimx_agent.contracts`. Invariants hosts rely on: gate ordering
(permission BEFORE approval), fresh run re-reads so mid-run pause/cancel/policy changes are
honored (including during the LAST step), legacy ``approval_policy IS NULL`` behavior, event
payload shapes, and no terminal step state without an authoritative outcome — a prepared
action generation re-enters the gates, an unknown outcome ends the drive untouched, and a step
in an unrecognized status is refused rather than dispatched ungated.

The engine holds no model transport, no persistence, and no host capabilities — hosts
provide those via the registry's handlers (which receive ``store.handler_context``), and may
observe run completion/failure through ``on_run_end`` (e.g. ControlRoom's bounded
auto-iterate epilogue). It never retries a step on its own.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence

from slimx_agent import contracts, policies
from slimx_agent.store import EventPayload, HostId, RunStore, RunView, StepView
from slimx_agent.tools import (
    StepActionPrepared,
    StepExecutionError,
    StepNotApplicable,
    StepOutcomeUnknown,
    ToolRegistry,
)

# Run statuses a run cannot transition out of.
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


class UnknownStepStatus(RuntimeError):
    """A host step's status is outside :data:`~slimx_agent.contracts.STEP_STATUSES`.

    The gates key on exact statuses, so the engine refuses such a step instead of dispatching it
    ungated, and ends the drive without writing anything for it. The host owns the repair.
    """

    def __init__(self, step_id: HostId, status: str) -> None:
        super().__init__(
            f"step {step_id!r} has unrecognized status {str(status)[:64]!r}; it was not dispatched"
        )
        self.step_id = step_id
        self.status = status


# Monotonic clock for the wall budget, as a module alias so tests can stub it.
_now = time.monotonic

# ``on_run_end(run, status)`` — called once when the loop finishes a run as "completed" or
# "failed" (after the terminal event is appended, before the final drain, so anything the
# hook appends still reaches a live stream). Never called for pause/cancel: those are user
# decisions the host must not auto-extend. Never called when a drive ends on an unknown outcome.
type RunEndHook[RunT] = Callable[[RunT, str], None]

# What a drive yields: ``("event", payload)`` for each newly persisted durable event.
type EngineEvent = tuple[str, EventPayload]


def execute_run[RunT: RunView, StepT: StepView, ContextT, ProfileT](
    store: RunStore[RunT, StepT, ContextT],
    registry: ToolRegistry[ContextT, RunT, StepT, ProfileT],
    run: RunT,
    *,
    profile: ProfileT,
    on_run_end: RunEndHook[RunT] | None = None,
) -> RunT:
    """Drive a run to its next stop (approval gate / pause / cancel / failure / completion).

    Thin drain of :func:`execute_run_events` so streaming and non-streaming execution share
    ONE path. Raises :class:`~slimx_agent.tools.StepOutcomeUnknown` when a step's outcome could
    not be observed."""
    for _ in execute_run_events(store, registry, run, profile=profile, on_run_end=on_run_end):
        pass
    refreshed = store.get_run(run.id)
    return refreshed if refreshed is not None else run


def execute_run_events[RunT: RunView, StepT: StepView, ContextT, ProfileT](
    store: RunStore[RunT, StepT, ContextT],
    registry: ToolRegistry[ContextT, RunT, StepT, ProfileT],
    run: RunT,
    *,
    profile: ProfileT,
    on_run_end: RunEndHook[RunT] | None = None,
) -> Iterator[EngineEvent]:
    """Generator form of :func:`execute_run`: identical side effects, but yields each newly
    persisted progress event as ``("event", payload)`` so an SSE endpoint can stream live
    step progress. Events are tailed from a per-run sequence cursor, so the live stream and
    a polled timeline stay consistent."""
    cursor = store.next_sequence(run.id) - 1

    def drain() -> Iterator[EngineEvent]:
        nonlocal cursor
        for payload in store.drained_events(run.id, cursor):
            cursor = payload["sequence"]
            yield "event", payload

    if run.status in TERMINAL_RUN_STATUSES:
        return
    store.set_run_status(run, "running")
    yield from drain()
    drive_started = _now()

    # A re-fetching loop (not a snapshot iteration) so steps a handler APPENDS mid-run — a
    # research_iterate plan extension — join this same drive: each pass re-reads the step list
    # and takes the first step that is not yet completed/skipped, exactly the order the old
    # snapshot loop executed. Termination is structural: run_step always leaves its step in a
    # terminal state, re-parked at a gate, or raises — so every pass makes progress or exits.
    while True:
        current = store.get_run(run.id)
        if current is not None and current.status in ("paused", "cancelled"):
            yield from drain()
            return
        steps = store.get_steps(run.id)
        step = next((s for s in steps if s.status not in ("completed", "skipped")), None)
        if step is None:
            break  # every step is done — fall through to the completion block
        if step.status not in contracts.STEP_STATUSES:
            # Fail closed: every gate below keys on exact statuses, so an unrecognized one would
            # otherwise reach dispatch with neither the permission nor the approval gate applied.
            raise UnknownStepStatus(step.id, step.status)
        if step.status == "failed":
            store.set_run_status(run, "failed")
            yield from drain()
            return
        # Budget gate — before any work on the next step. Exhaustion is an honest PAUSE (event
        # with the reason, then the paused status), never a silent truncation: the user raises
        # the budget and re-executes, or accepts the partial result.
        budget_reason = _budget_exhausted_reason(current or run, steps, drive_started)
        if budget_reason is not None:
            store.append_event(
                run.id, contracts.BUDGET_EXHAUSTED, payload={"reason": budget_reason}
            )
            store.set_run_status(run, "paused")
            store.append_event(run.id, contracts.RUN_PAUSED, payload={"reason": budget_reason})
            yield from drain()
            return
        # Tool-permission gate — runs BEFORE the approval gate so an ungranted external tool
        # is skipped honestly rather than stopping the run for an approval it could never
        # satisfy. The fresh run's grants are authoritative, and an already-approved step is
        # re-checked too: approval never bypasses a grant that is missing or was revoked.
        if step.status in ("pending", "awaiting_approval", "approved"):
            permit_reason = policies.permission_block_reason(step, current or run)
            if permit_reason is not None:
                _skip_step(store, run, step.id, step.type, permit_reason)
                yield from drain()
                continue
        # Approval gate — deterministic, host-enforced policy. Read the run fresh so a
        # mid-run policy/auto_approve toggle is honored. ``approval_policy is None`` keeps
        # the exact legacy behavior (gate only planner-flagged steps; ``auto_approve``
        # clears them); a set policy uses the classifier + policy matrix, so hard gates stop
        # even in Auto-complete while additive steps run without a manual click.
        policy = current.approval_policy if current is not None else run.approval_policy
        auto_approve = current.auto_approve if current is not None else run.auto_approve
        # Scoped pre-approval (0.14): duck-typed run attribute, read fresh so a mid-run
        # grant/revoke is honored on the very next step. Normalized: only allowlisted
        # read-only types in a real list can ever clear a gate.
        preapproved_tools = policies.normalize_preapproved(
            getattr(current if current is not None else run, "preapproved_tools", None)
        )
        if step.status in ("pending", "awaiting_approval"):
            classification, reason, stop = resolve_gate(
                step,
                policy=policy,
                auto_approve=auto_approve,
                preapproved_tools=preapproved_tools,
            )
            if stop:
                if step.status == "pending":
                    gate_for_approval(store, run, step, reason=reason)
                store.set_run_status(run, "awaiting_approval")
                yield from drain()
                return
            # Not stopping: if the step was gated (planner-flagged, already awaiting, or a
            # hard gate cleared by scoped pre-approval), record the auto-approval that clears
            # it — the trail shows WHY it proceeded.
            preapproved_gate = (
                classification == policies.HARD_GATED and step.type in preapproved_tools
            )
            if step.requires_approval or step.status == "awaiting_approval" or preapproved_gate:
                store.set_step_state(step.id, "approved")
                payload: dict[str, object] = {"auto": True}
                if policy is not None:
                    payload |= {"policy": policy, "classification": classification}
                if preapproved_gate:
                    payload["preapproved"] = True
                store.append_event(
                    run.id, contracts.APPROVAL_GRANTED, step_id=step.id, payload=payload
                )

        ran = run_step(store, registry, run, step, profile=profile)
        yield from drain()
        if ran.status == "awaiting_approval":
            # A host-side two-stage action prepared a new generation while this handler was
            # admitted. End this drive at the durable review boundary. A later drive re-reads the
            # new action and applies normal manual/automatic policy, matching in-process parity.
            return
        if ran.status == "failed":
            store.append_event(run.id, contracts.RUN_FAILED, step_id=ran.id, commit=False)
            store.set_run_status(run, "failed")
            if on_run_end is not None:
                on_run_end(run, "failed")
            yield from drain()
            return

    # A pause/cancel can land while the LAST step runs (e.g. during a join fan-out);
    # re-read before declaring completion so it is honored instead of overwritten.
    final = store.get_run(run.id)
    if final is not None and final.status in ("paused", "cancelled"):
        yield from drain()
        return
    store.set_run_status(run, "completed")
    store.append_event(run.id, contracts.RUN_COMPLETED)
    if on_run_end is not None:
        on_run_end(run, "completed")
    yield from drain()


def run_step[RunT: RunView, StepT: StepView, ContextT, ProfileT](
    store: RunStore[RunT, StepT, ContextT],
    registry: ToolRegistry[ContextT, RunT, StepT, ProfileT],
    run: RunT,
    step: StepT,
    *,
    profile: ProfileT,
) -> StepT:
    """Execute one step. Returns the step as ``completed``/``failed``/``skipped``, or re-read
    at a gate after a prepared action; raises :class:`StepOutcomeUnknown` untouched."""
    step_id = step.id
    store.set_step_state(step_id, "running")
    store.append_event(run.id, contracts.STEP_STARTED, step_id=step_id, payload={"type": step.type})

    handler = registry.resolve(step.type)
    if handler is None:
        return _fail_step(store, run, step_id, step.type, f"Unsupported step type {step.type!r}")

    try:
        output_refs = handler(store.handler_context, run, step, profile)
    except StepActionPrepared:
        # The host committed a new action generation and re-parked the step. Re-read it rather
        # than writing a false completed/skipped/failed state; the outer loop will apply policy
        # to that exact prepared generation (automatic receipt or human gate).
        fresh = store.get_step(step_id)
        return fresh if fresh is not None else step
    except StepOutcomeUnknown:
        # No authoritative outcome reached the engine: write nothing terminal and retry nothing.
        # Ending the drive leaves the step exactly where the host's durable records put it.
        raise
    except StepNotApplicable as exc:
        return _skip_step(store, run, step_id, step.type, str(exc))
    except StepExecutionError as exc:
        return _fail_step(store, run, step_id, step.type, str(exc))
    except Exception as exc:  # noqa: BLE001 — service bugs fail the step, not the request
        store.rollback()
        return _fail_step(store, run, step_id, step.type, _short_error(exc))

    fresh_step = store.set_step_state(
        step_id, "completed", error=None, output_refs=output_refs or None
    )
    store.append_event(
        run.id,
        contracts.STEP_COMPLETED,
        step_id=step_id,
        payload={"type": fresh_step.type, **(output_refs or {})},
    )
    return fresh_step


def _budget_exhausted_reason(
    run: RunView, steps: Sequence[StepView], drive_started: float
) -> str | None:
    """Why the run's budget forbids executing the next step, or ``None`` while within budget.

    Budgets are duck-typed run attributes (see ``contracts.RUN_BUDGET_FIELDS``); absent/None/
    non-positive values mean unbounded, so budget-less hosts and legacy rows never pause.
    ``budget_max_steps`` counts EXECUTED steps (completed or failed — skips did no work);
    ``budget_max_wall_seconds`` is wall clock for THIS drive only (a resume starts fresh)."""
    max_steps = getattr(run, "budget_max_steps", None)
    if isinstance(max_steps, int) and max_steps > 0:
        executed = sum(1 for s in steps if s.status in ("completed", "failed"))
        if executed >= max_steps:
            return f"step budget reached ({executed}/{max_steps} steps executed)"
    max_wall = getattr(run, "budget_max_wall_seconds", None)
    if isinstance(max_wall, int) and max_wall > 0:
        elapsed = _now() - drive_started
        if elapsed >= max_wall:
            return f"time budget reached ({int(elapsed)}s of {max_wall}s for this execution)"
    return None


def resolve_gate(
    step: StepView,
    *,
    policy: str | None,
    auto_approve: bool,
    preapproved_tools: object = None,
) -> tuple[str | None, str, bool]:
    """Decide whether execution stops at ``step``. Returns ``(classification, reason, stop)``.

    Legacy path (``policy is None``): only planner-flagged steps gate, and ``auto_approve``
    clears them — byte-for-byte the old behavior. Otherwise the deterministic classifier +
    policy matrix in :mod:`slimx_agent.policies` decide. ``preapproved_tools`` (the run's
    duck-typed scoped pre-authorization, 0.14) downgrades an allowlisted read-only hard gate
    — :func:`policies.normalize_preapproved` enforces membership in
    ``policies.PREAPPROVABLE_STEP_TYPES``, so a stored value outside the allowlist (or a junk
    non-list value) can never clear a gate."""
    if policy is None:
        return None, "", bool(step.requires_approval) and not auto_approve
    classification, reason = policies.classify_step(step)
    preapproved = step.type in policies.normalize_preapproved(preapproved_tools)
    stop = policies.requires_stop(
        policy, classification, step.requires_approval, preapproved=preapproved
    )
    return classification, reason, stop


def gate_for_approval[RunT: RunView, StepT: StepView](
    store: RunStore[RunT, StepT, object], run: RunT, step: StepT, *, reason: str = ""
) -> None:
    """Park a step at the human gate and record why."""
    store.set_step_state(step.id, "awaiting_approval")
    payload: dict[str, object] = {"title": step.title}
    if reason:
        payload["reason"] = reason
    store.append_event(run.id, contracts.APPROVAL_REQUIRED, step_id=step.id, payload=payload)


def _fail_step[RunT: RunView, StepT: StepView](
    store: RunStore[RunT, StepT, object], run: RunT, step_id: HostId, step_type: str, message: str
) -> StepT:
    step = store.set_step_state(step_id, "failed", error=message)
    store.append_event(
        run.id,
        contracts.STEP_FAILED,
        step_id=step_id,
        payload={"type": step_type, "error": message},
    )
    return step


def _skip_step[RunT: RunView, StepT: StepView](
    store: RunStore[RunT, StepT, object], run: RunT, step_id: HostId, step_type: str, reason: str
) -> StepT:
    step = store.set_step_state(step_id, "skipped", error=None)
    store.append_event(
        run.id,
        contracts.STEP_SKIPPED,
        step_id=step_id,
        payload={"type": step_type, "reason": reason},
    )
    return step


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]
