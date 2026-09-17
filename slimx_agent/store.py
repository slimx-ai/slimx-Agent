"""The RunStore protocol — the persistence boundary the agent engine drives runs through.

The engine (``slimx_agent.engine``) owns run/step/event ORDERING and semantics; the store
owns HOW they persist (a host DB, an in-memory fake, the standalone service's callback
client). Runs and steps are host objects the engine only reads through the structural views
below (:class:`RunView`, :class:`StepView`); it never mutates them directly — every write goes
through a store method so hosts control transactions, timestamps, and refresh semantics.

The protocol is generic in the host's run type, step type, and handler context, so a type
checker can prove that one host's store, tool registry, and run objects agree. Identities are
the one deliberately open part: they are host-chosen (UUIDs in-process, strings on the wire),
passed back to the store unchanged, and never inspected by the engine.
"""

from __future__ import annotations

import enum
from collections.abc import Hashable, Sequence
from typing import Any, Final, Protocol

# A host-chosen identity value (a UUID in-process, a string over the standalone wire). The
# engine only passes identities back to the store that produced them, so the store decides.
type HostId = Any

# Small, JSON-ready references a completed step produced (ids, paths, counts) — never model
# output or evidence text. An open mapping on purpose: each tool defines its own keys.
type OutputRefs = dict[str, Any]

# One durable event in its JSON-ready wire shape. The engine reads only ``sequence`` (a
# per-run monotonic integer); the rest is the host's streaming projection.
type EventPayload = dict[str, Any]


class UnsetType(enum.Enum):
    """The type of :data:`UNSET`: "leave this field untouched", distinct from ``None``."""

    UNSET = "UNSET"

    def __repr__(self) -> str:
        return "UNSET"


# Sentinel distinguishing "leave the step's error untouched" from "clear it (None)". Compare
# with ``is``; it is a singleton, and ``None`` is a real value meaning "clear".
UNSET: Final = UnsetType.UNSET


class RunView(Protocol):
    """The run fields the engine and policies read.

    Optional engine inputs — ``budget_max_steps``, ``budget_max_wall_seconds`` (see
    ``contracts.RUN_BUDGET_FIELDS``) and ``preapproved_tools`` — are read with ``getattr``
    defaults, so hosts that predate them need not declare them.
    """

    @property
    def id(self) -> Hashable: ...

    @property
    def status(self) -> str: ...

    @property
    def approval_policy(self) -> str | None: ...

    @property
    def auto_approve(self) -> bool: ...

    @property
    def allowed_tools_json(self) -> Sequence[str] | None: ...


class StepView(Protocol):
    """The step fields the engine and policies read."""

    @property
    def id(self) -> Hashable: ...

    @property
    def type(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def requires_approval(self) -> bool: ...


class RunStore[RunT: RunView, StepT: StepView, ContextT](Protocol):
    """Persistence operations the engine's run loop requires.

    ``RunT``/``StepT`` are the host's own run and step types; ``ContextT`` is the opaque
    value handed to tool handlers (for example a database session or a callback client).
    """

    @property
    def handler_context(self) -> ContextT:
        """Opaque host state handed to tool handlers as their first argument (e.g. a DB
        session). The engine never inspects it."""
        ...

    # --- reads (fresh, not cached — the loop re-reads to honor concurrent mutations) ---

    def get_run(self, run_id: HostId) -> RunT | None: ...

    def get_steps(self, run_id: HostId) -> list[StepT]:
        """The run's steps in execution order."""
        ...

    def get_step(self, step_id: HostId) -> StepT | None: ...

    # --- writes (persist + commit; return the fresh row) ---

    def set_run_status(self, run: RunT, status: str) -> RunT: ...

    def set_step_state(
        self,
        step_id: HostId,
        status: str,
        *,
        error: str | None | UnsetType = UNSET,
        output_refs: OutputRefs | None | UnsetType = UNSET,
    ) -> StepT:
        """Persist a step transition. ``error``/``output_refs`` replace the step's fields
        when given (``None`` clears); left UNSET they stay untouched.

        The ``running`` transition is the host's last chance to refuse dispatch. A drive that
        finds a step already ``running`` (an earlier drive was interrupted) requests
        ``running`` again before re-entering the handler; a host whose earlier attempt may
        already have crossed its entry boundary MUST refuse (raise) instead of allowing a
        second entry. The engine itself never retries. The engine asks for this transition only
        after its own permission gate passed: a ``running`` step whose grant is gone is refused
        with ``engine.RunningStepNotPermitted`` before any store write."""
        ...

    def rollback(self) -> None:
        """Discard uncommitted host state after a handler blew up mid-transaction."""
        ...

    # --- durable events (append-only, per-run monotonic sequence) ---

    def append_event(
        self,
        run_id: HostId,
        type: str,
        *,
        step_id: HostId | None = None,
        payload: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> object: ...

    def next_sequence(self, run_id: HostId) -> int: ...

    def drained_events(self, run_id: HostId, after_sequence: int) -> list[EventPayload]:
        """Events past ``after_sequence`` as JSON-ready payload dicts (the wire shape a host
        streams over SSE), ordered by sequence."""
        ...
