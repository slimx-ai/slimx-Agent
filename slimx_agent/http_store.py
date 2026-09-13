"""HttpRunStore: the RunStore protocol over the host's callback API.

Runs and steps arrive as JSON snapshots and are parsed into small dataclasses exposing
exactly the fields the engine and policies read (:class:`~slimx_agent.store.RunView` /
:class:`~slimx_agent.store.StepView`). Parsing is strict at this trust boundary: a field with
the wrong JSON type is a :class:`~slimx_agent.host_client.HostProtocolError`, never coerced —
``bool("false")`` must not turn into an automatic approval, and a malformed budget must not
turn into an unbounded one. Every write is one host call that persists and commits host-side;
``rollback`` is a no-op because a failed step invocation already rolled back inside the host's
own request transaction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from slimx_agent import contracts
from slimx_agent.host_client import HostClient, HostProtocolError
from slimx_agent.store import UNSET, EventPayload, HostId, OutputRefs, UnsetType


@dataclass
class RunSnapshot:
    """The engine-visible view of a host run row."""

    id: str
    status: str
    approval_policy: str | None
    auto_approve: bool
    allowed_tools_json: list[str] | None
    # Engine-enforced run budgets (0.9, contracts.RUN_BUDGET_FIELDS). None/absent = unbounded,
    # so a pre-budget host wire shape behaves exactly as before.
    budget_max_steps: int | None = None
    budget_max_wall_seconds: int | None = None
    # Scoped pre-approval (0.14). None/absent = nothing pre-authorized (pre-0.14 wire shape).
    preapproved_tools: list[str] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, data: Mapping[str, object]) -> RunSnapshot:
        what = "run snapshot"
        return cls(
            id=_required_str(data, "id", what),
            status=_required_str(data, "status", what),
            approval_policy=_optional_str(data, "approval_policy", what),
            auto_approve=_optional_bool(data, "auto_approve", what),
            allowed_tools_json=_optional_str_list(data, "allowed_tools_json", what),
            budget_max_steps=_optional_int(data, "budget_max_steps", what),
            budget_max_wall_seconds=_optional_int(data, "budget_max_wall_seconds", what),
            preapproved_tools=_optional_str_list(data, "preapproved_tools", what),
            raw=dict(data),
        )


@dataclass
class StepSnapshot:
    """The engine-visible view of a host step row."""

    id: str
    type: str
    title: str
    status: str
    requires_approval: bool
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, data: Mapping[str, object]) -> StepSnapshot:
        what = "step snapshot"
        status = _required_str(data, "status", what)
        if status not in contracts.STEP_STATUSES:
            # The gates key on exact statuses: an unrecognized one is refused, never dispatched.
            raise HostProtocolError(f"the {what} has an unrecognized status {status[:64]!r}")
        return cls(
            id=_required_str(data, "id", what),
            type=_required_str(data, "type", what),
            title=_optional_str(data, "title", what) or "",
            status=status,
            requires_approval=_optional_bool(data, "requires_approval", what),
            raw=dict(data),
        )


class HttpRunStore:
    """RunStore implementation for the standalone service (host persistence over HTTP)."""

    def __init__(self, client: HostClient) -> None:
        self._client = client

    @property
    def handler_context(self) -> HostClient:
        # Tool handlers receive the host client — the remote registry's handlers use it to
        # invoke the step on the host.
        return self._client

    # --- reads ----------------------------------------------------------------------

    def get_run(self, run_id: HostId) -> RunSnapshot | None:
        data = self._client.get_run(run_id)
        return RunSnapshot.from_wire(data) if data is not None else None

    def get_steps(self, run_id: HostId) -> list[StepSnapshot]:
        return [StepSnapshot.from_wire(item) for item in self._client.get_steps(run_id)]

    def get_step(self, step_id: HostId) -> StepSnapshot | None:
        data = self._client.get_step(step_id)
        return StepSnapshot.from_wire(data) if data is not None else None

    # --- writes ---------------------------------------------------------------------

    def set_run_status(self, run: RunSnapshot, status: str) -> RunSnapshot:
        return RunSnapshot.from_wire(self._client.set_run_status(run.id, status))

    def set_step_state(
        self,
        step_id: HostId,
        status: str,
        *,
        error: str | None | UnsetType = UNSET,
        output_refs: OutputRefs | None | UnsetType = UNSET,
    ) -> StepSnapshot:
        return StepSnapshot.from_wire(
            self._client.set_step_state(step_id, status, error=error, output_refs=output_refs)
        )

    def rollback(self) -> None:
        """No-op: a handler exception already rolled back inside the host's invoke request;
        there is no client-side transaction to discard."""

    # --- durable events ---------------------------------------------------------------

    def append_event(
        self,
        run_id: HostId,
        type: str,
        *,
        step_id: HostId | None = None,
        payload: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> EventPayload:
        # ``commit`` is accepted for protocol parity but each host call persists+commits on
        # its own: the engine's single ``commit=False`` site only deferred a commit boundary
        # (RUN_FAILED immediately before the run-status write); event ordering is unchanged.
        return self._client.append_event(run_id, type, step_id=step_id, payload=payload)

    def next_sequence(self, run_id: HostId) -> int:
        return self._client.next_sequence(run_id)

    def drained_events(self, run_id: HostId, after_sequence: int) -> list[EventPayload]:
        return self._client.events_after(run_id, after_sequence)


# --- strict wire-field parsing ------------------------------------------------------------


def _required_str(data: Mapping[str, object], key: str, what: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise HostProtocolError(f"{what} field {key!r} must be a non-empty string")
    return value


def _optional_str(data: Mapping[str, object], key: str, what: str) -> str | None:
    value = data.get(key)
    if value is None or isinstance(value, str):
        return value
    raise HostProtocolError(f"{what} field {key!r} must be a string or null")


def _optional_bool(data: Mapping[str, object], key: str, what: str) -> bool:
    value = data.get(key)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise HostProtocolError(f"{what} field {key!r} must be a boolean or null")


def _optional_int(data: Mapping[str, object], key: str, what: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise HostProtocolError(f"{what} field {key!r} must be an integer or null")


def _optional_str_list(data: Mapping[str, object], key: str, what: str) -> list[str] | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise HostProtocolError(f"{what} field {key!r} must be an array of strings or null")
