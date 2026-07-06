"""HttpRunStore: the RunStore protocol over the host's callback API.

Runs and steps arrive as JSON snapshots and are wrapped in small dataclasses exposing
exactly the duck-typed fields the engine and policies read (see ``store.RunStore``'s
contract). Every write is one host call that persists and commits host-side; ``rollback``
is a no-op because a failed step invocation already rolled back inside the host's own
request transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from slimx_agent.host_client import HostClient
from slimx_agent.store import UNSET


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
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> "RunSnapshot":
        return cls(
            id=str(data["id"]),
            status=str(data["status"]),
            approval_policy=data.get("approval_policy"),
            auto_approve=bool(data.get("auto_approve", False)),
            allowed_tools_json=data.get("allowed_tools_json"),
            budget_max_steps=_opt_int(data.get("budget_max_steps")),
            budget_max_wall_seconds=_opt_int(data.get("budget_max_wall_seconds")),
            raw=data,
        )


def _opt_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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
    def from_wire(cls, data: dict[str, Any]) -> "StepSnapshot":
        return cls(
            id=str(data["id"]),
            type=str(data["type"]),
            title=str(data.get("title") or ""),
            status=str(data["status"]),
            requires_approval=bool(data.get("requires_approval", False)),
            raw=data,
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

    def get_run(self, run_id: Any) -> RunSnapshot | None:
        data = self._client.get_run(run_id)
        return RunSnapshot.from_wire(data) if data is not None else None

    def get_steps(self, run_id: Any) -> list[StepSnapshot]:
        return [StepSnapshot.from_wire(item) for item in self._client.get_steps(run_id)]

    def get_step(self, step_id: Any) -> StepSnapshot | None:
        data = self._client.get_step(step_id)
        return StepSnapshot.from_wire(data) if data is not None else None

    # --- writes ---------------------------------------------------------------------

    def set_run_status(self, run: Any, status: str) -> RunSnapshot:
        return RunSnapshot.from_wire(self._client.set_run_status(run.id, status))

    def set_step_state(
        self,
        step_id: Any,
        status: str,
        *,
        error: Any = UNSET,
        output_refs: Any = UNSET,
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
        run_id: Any,
        type: str,
        *,
        step_id: Any | None = None,
        payload: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        # ``commit`` is accepted for protocol parity but each host call persists+commits on
        # its own: the engine's single ``commit=False`` site only deferred a commit boundary
        # (RUN_FAILED immediately before the run-status write); event ordering is unchanged.
        return self._client.append_event(run_id, type, step_id=step_id, payload=payload)

    def next_sequence(self, run_id: Any) -> int:
        return self._client.next_sequence(run_id)

    def drained_events(self, run_id: Any, after_sequence: int) -> list[dict[str, Any]]:
        return self._client.events_after(run_id, after_sequence)
