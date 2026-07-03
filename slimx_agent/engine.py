"""The agent run engine: the deliberately boring dispatch loop, extracted from ControlRoom.

For each runnable step it applies the tool-permission gate (ungranted tools skip honestly),
the deterministic approval gate (hard gates stop even in Auto-complete), dispatches through
the :class:`~slimx_agent.tools.ToolRegistry` (the ONLY path to a tool implementation),
persists transitions through a :class:`~slimx_agent.store.RunStore`, and emits the durable
event vocabulary from :mod:`slimx_agent.contracts`. Semantics preserved verbatim from the
host implementation (Stage I): gate ordering (permission BEFORE approval), fresh run
re-reads so mid-run pause/cancel/policy changes are honored (including during the LAST
step), legacy ``approval_policy IS NULL`` behavior, and event payload shapes.

The engine holds no model transport, no persistence, and no host capabilities — hosts
provide those via the registry's handlers (which receive ``store.handler_context``), and may
observe run completion/failure through ``on_run_end`` (e.g. ControlRoom's bounded
auto-iterate epilogue).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from slimx_agent import contracts
from slimx_agent import policies
from slimx_agent.tools import StepExecutionError, StepNotApplicable, ToolRegistry

# Run statuses a run cannot transition out of.
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

# ``on_run_end(run, status)`` — called once when the loop finishes a run as "completed" or
# "failed" (after the terminal event is appended, before the final drain, so anything the
# hook appends still reaches a live stream). Never called for pause/cancel: those are user
# decisions the host must not auto-extend.
RunEndHook = Callable[[Any, str], None]


def execute_run(
    store: Any,
    registry: ToolRegistry,
    run: Any,
    *,
    profile: Any,
    on_run_end: RunEndHook | None = None,
) -> Any:
    """Drive a run to its next stop (approval gate / pause / cancel / failure / completion).

    Thin drain of :func:`execute_run_events` so streaming and non-streaming execution share
    ONE path."""
    for _ in execute_run_events(store, registry, run, profile=profile, on_run_end=on_run_end):
        pass
    refreshed = store.get_run(run.id)
    return refreshed if refreshed is not None else run


def execute_run_events(
    store: Any,
    registry: ToolRegistry,
    run: Any,
    *,
    profile: Any,
    on_run_end: RunEndHook | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Generator form of :func:`execute_run`: identical side effects, but yields each newly
    persisted progress event as ``("event", payload)`` so an SSE endpoint can stream live
    step progress. Events are tailed from a per-run sequence cursor, so the live stream and
    a polled timeline stay consistent."""
    cursor = store.next_sequence(run.id) - 1

    def drain() -> Iterator[tuple[str, dict[str, Any]]]:
        nonlocal cursor
        for payload in store.drained_events(run.id, cursor):
            cursor = payload["sequence"]
            yield "event", payload

    if run.status in TERMINAL_RUN_STATUSES:
        return
    store.set_run_status(run, "running")
    yield from drain()

    for step in store.get_steps(run.id):
        current = store.get_run(run.id)
        if current is not None and current.status in ("paused", "cancelled"):
            yield from drain()
            return
        if step.status in ("completed", "skipped"):
            continue
        if step.status == "failed":
            store.set_run_status(run, "failed")
            yield from drain()
            return
        # Tool-permission gate — runs BEFORE the approval gate so an ungranted external tool
        # is skipped honestly rather than stopping the run for an approval it could never
        # satisfy. The grant list is set at create time; the fresh run is authoritative.
        if step.status in ("pending", "awaiting_approval"):
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
        if step.status in ("pending", "awaiting_approval"):
            classification, reason, stop = resolve_gate(
                step, policy=policy, auto_approve=auto_approve
            )
            if stop:
                if step.status == "pending":
                    gate_for_approval(store, run, step, reason=reason)
                store.set_run_status(run, "awaiting_approval")
                yield from drain()
                return
            # Not stopping: if the step was gated (planner-flagged or already awaiting),
            # record the auto-approval that clears it — the trail shows WHY it proceeded.
            if step.requires_approval or step.status == "awaiting_approval":
                store.set_step_state(step.id, "approved")
                payload: dict[str, Any] = {"auto": True}
                if policy is not None:
                    payload |= {"policy": policy, "classification": classification}
                store.append_event(
                    run.id, contracts.APPROVAL_GRANTED, step_id=step.id, payload=payload
                )

        ran = run_step(store, registry, run, step, profile=profile)
        yield from drain()
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


def run_step(store: Any, registry: ToolRegistry, run: Any, step: Any, *, profile: Any) -> Any:
    """Execute one step. Always returns the step (status ``completed``/``failed``/``skipped``)."""
    step_id = step.id
    store.set_step_state(step_id, "running")
    store.append_event(
        run.id, contracts.STEP_STARTED, step_id=step_id, payload={"type": step.type}
    )

    handler = registry.resolve(step.type)
    if handler is None:
        return _fail_step(store, run, step_id, step.type, f"Unsupported step type {step.type!r}")

    try:
        output_refs = handler(store.handler_context, run, step, profile)
    except StepNotApplicable as exc:
        return _skip_step(store, run, step_id, step.type, str(exc))
    except StepExecutionError as exc:
        return _fail_step(store, run, step_id, step.type, str(exc))
    except Exception as exc:  # underlying service blew up — fail the step, not the request
        store.rollback()
        return _fail_step(store, run, step_id, step.type, _short_error(exc))

    fresh = store.set_step_state(
        step_id, "completed", error=None, output_refs=output_refs or None
    )
    store.append_event(
        run.id,
        contracts.STEP_COMPLETED,
        step_id=step_id,
        payload={"type": fresh.type, **(output_refs or {})},
    )
    return fresh


def resolve_gate(
    step: Any, *, policy: str | None, auto_approve: bool
) -> tuple[str | None, str, bool]:
    """Decide whether execution stops at ``step``. Returns ``(classification, reason, stop)``.

    Legacy path (``policy is None``): only planner-flagged steps gate, and ``auto_approve``
    clears them — byte-for-byte the old behavior. Otherwise the deterministic classifier +
    policy matrix in :mod:`slimx_agent.policies` decide."""
    if policy is None:
        return None, "", bool(step.requires_approval) and not auto_approve
    classification, reason = policies.classify_step(step)
    stop = policies.requires_stop(policy, classification, step.requires_approval)
    return classification, reason, stop


def gate_for_approval(store: Any, run: Any, step: Any, *, reason: str = "") -> None:
    """Park a step at the human gate and record why."""
    store.set_step_state(step.id, "awaiting_approval")
    payload: dict[str, Any] = {"title": step.title}
    if reason:
        payload["reason"] = reason
    store.append_event(run.id, contracts.APPROVAL_REQUIRED, step_id=step.id, payload=payload)


def _fail_step(store: Any, run: Any, step_id: Any, step_type: str, message: str) -> Any:
    step = store.set_step_state(step_id, "failed", error=message)
    store.append_event(
        run.id,
        contracts.STEP_FAILED,
        step_id=step_id,
        payload={"type": step_type, "error": message},
    )
    return step


def _skip_step(store: Any, run: Any, step_id: Any, step_type: str, reason: str) -> Any:
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
