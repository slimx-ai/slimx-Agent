"""The RunStore protocol — the persistence boundary the agent engine drives runs through.

The engine (``slimx_agent.engine``) owns run/step/event ORDERING and semantics; the store
owns HOW they persist (a host DB, an in-memory fake, eventually the standalone service's own
store). Runs/steps are duck-typed host objects — the engine only reads the fields named in
each method's contract (``run.id/status/approval_policy/auto_approve``, ``step.id/type/
title/status/requires_approval``) and never mutates them directly: every write goes through
a store method so hosts control transactions, timestamps, and refresh semantics.
"""

from __future__ import annotations

from typing import Any, Protocol

# Sentinel distinguishing "leave the step's error untouched" from "clear it (None)".
UNSET: Any = object()


class RunStore(Protocol):
    """Persistence operations the engine's run loop requires."""

    @property
    def handler_context(self) -> Any:
        """Opaque host state handed to tool handlers as their first argument (e.g. a DB
        session). The engine never inspects it."""
        ...

    # --- reads (fresh, not cached — the loop re-reads to honor concurrent mutations) ---

    def get_run(self, run_id: Any) -> Any | None: ...

    def get_steps(self, run_id: Any) -> list[Any]:
        """The run's steps in execution order."""
        ...

    def get_step(self, step_id: Any) -> Any | None: ...

    # --- writes (persist + commit; return the fresh row) ---

    def set_run_status(self, run: Any, status: str) -> Any: ...

    def set_step_state(
        self,
        step_id: Any,
        status: str,
        *,
        error: Any = UNSET,
        output_refs: Any = UNSET,
    ) -> Any:
        """Persist a step transition. ``error``/``output_refs`` replace the step's fields
        when given (``None`` clears); left UNSET they stay untouched."""
        ...

    def rollback(self) -> None:
        """Discard uncommitted host state after a handler blew up mid-transaction."""
        ...

    # --- durable events (append-only, per-run monotonic sequence) ---

    def append_event(
        self,
        run_id: Any,
        type: str,
        *,
        step_id: Any | None = None,
        payload: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> Any: ...

    def next_sequence(self, run_id: Any) -> int: ...

    def drained_events(self, run_id: Any, after_sequence: int) -> list[dict[str, Any]]:
        """Events past ``after_sequence`` as JSON-ready payload dicts (the wire shape a host
        streams over SSE), ordered by sequence."""
        ...
