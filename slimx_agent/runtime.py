"""The SlimX-Agent runtime interface (Stage C of docs/slimx-agent-extraction-plan.md).

``AgentRuntime`` is the seam between the host (ControlRoom routes/UI) and the agent engine:

    ControlRoom route → AgentRuntime → implementation

The in-process implementation (``app.slimx_agent.inprocess``) wraps the existing
``services/agent`` modules verbatim today; Stage E adds an HTTP client implementation with
the same surface so the runtime can move into its own service/container without touching the
routes again.

Design rules:
- **Session ownership stays with the caller.** Every method takes the SQLModel ``Session``
  first; the runtime never opens/closes sessions (the SSE route owns its worker-thread
  session exactly as before). This is the documented transitional shared-DB design.
- **Domain semantics live here, transport stays in routes.** State-machine guards (what can
  be paused/approved/rerun) raise :class:`AgentRunConflict`; routes translate to HTTP 409
  with the same detail strings, keeping responses byte-identical. Record resolution (404s),
  authz, provider-profile resolution, cloud-egress enforcement, and review-packet creation
  remain host concerns (Stage D formalizes them as adapters).
- The interface is typed loosely (``Any`` for ORM rows) on purpose: the rows are ControlRoom's
  SQLModel objects until the RunStore protocol lands (Stage E); pinning the protocol to the
  ORM types now would couple the contract layer to the database.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID


class AgentRunConflict(Exception):
    """A lifecycle operation was applied in a state that forbids it (HTTP 409 at the route).

    ``str(exc)`` is the user-facing detail and must stay stable — the routes return it
    verbatim, and the frontend matches on some of these messages.
    """


@dataclass(frozen=True)
class RunProfile:
    """The provider/model a run's model-using operations execute against.

    The HOST resolves this (provider profile → provider/model/base_url) and enforces cloud
    egress BEFORE handing it to the runtime — the runtime never resolves credentials.
    """

    provider: str
    model: str
    base_url: str | None = None


class AgentRuntime(Protocol):
    """The run-lifecycle surface ControlRoom consumes. One method per route operation.

    All methods are synchronous and stateless; implementations must be safe to share across
    requests and threads (the streaming route calls ``execute_run_events`` from a worker
    thread with its own session).
    """

    # --- runs ---------------------------------------------------------------
    def get_run(self, session: Any, run_id: UUID) -> Any | None: ...
    def create_run(
        self,
        session: Any,
        *,
        workspace_id: UUID,
        conversation_id: UUID,
        user_goal: str,
        provider_profile_id: UUID | None,
        routing_policy: str,
        mode: str,
        approval_policy: str | None,
        allowed_tools: list[str] | None,
        context_bundle_id: UUID | None = None,
        expected_artifact_kind: str | None = None,
        planner_profile_id: UUID | None = None,
        depth: str = "standard",
    ) -> Any: ...
    def list_runs(self, session: Any, conversation_id: UUID) -> list[Any]: ...
    def get_steps(self, session: Any, run_id: UUID) -> list[Any]: ...
    def get_events(
        self, session: Any, run_id: UUID, after_sequence: int | None = None
    ) -> list[Any]: ...
    def append_event(
        self,
        session: Any,
        run_id: UUID,
        event_type: str,
        *,
        step_id: UUID | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any: ...
    def normalize_grants(self, allowed_tools: list[str] | None) -> list[str] | None: ...
    def plan_run(self, session: Any, run: Any, profile: RunProfile) -> Any: ...
    def claim_run_for_execution(self, session: Any, run_id: UUID) -> bool: ...
    def execute_run(self, session: Any, run: Any, profile: RunProfile) -> Any: ...
    def execute_run_events(self, session: Any, run: Any, profile: RunProfile) -> Iterator[Any]: ...
    def pause_run(self, session: Any, run: Any) -> Any: ...
    def cancel_run(self, session: Any, run: Any) -> Any: ...
    def resume_run(self, session: Any, run: Any) -> Any: ...
    def set_auto_approve(self, session: Any, run: Any, enabled: bool) -> Any: ...
    def set_approval_policy(self, session: Any, run: Any, policy: str) -> Any: ...
    def set_budgets(self, session: Any, run: Any, budgets: dict[str, int | None]) -> Any: ...
    def set_preapproved_tools(self, session: Any, run: Any, tools: list[str]) -> Any: ...
    def set_node_review(self, session: Any, run: Any, node_id: str, reviewed: bool) -> Any: ...
    def extract_system_map(self, session: Any, run: Any, profile: RunProfile) -> dict[str, Any]: ...
    def scan_code_into_run(self, session: Any, run: Any) -> dict[str, Any]: ...
    def build_outcome(self, session: Any, run: Any) -> Any: ...

    # --- steps --------------------------------------------------------------
    def update_step(self, session: Any, step: Any, fields: dict[str, Any]) -> Any: ...
    def approve_step(self, session: Any, run: Any, step: Any) -> Any: ...
    def skip_step(self, session: Any, run: Any, step: Any) -> Any: ...
    def rerun_step(self, session: Any, run: Any, step: Any) -> Any: ...

    # --- artifacts ----------------------------------------------------------
    def list_artifacts(self, session: Any, run_id: UUID) -> list[Any]: ...
    def get_artifact(self, session: Any, artifact_id: UUID) -> Any | None: ...
    def artifact_to_read(self, artifact: Any) -> Any: ...
    def load_artifact_bytes(self, artifact: Any) -> bytes: ...

    # --- templates ----------------------------------------------------------
    def get_template(self, session: Any, template_id: UUID) -> Any | None: ...
    def save_run_as_template(
        self, session: Any, run: Any, *, name: str, description: str | None
    ) -> Any: ...
    def list_templates(self, session: Any, workspace_id: UUID) -> list[Any]: ...
    def update_template(
        self, session: Any, template: Any, *, name: str | None, description: str | None
    ) -> Any: ...
    def delete_template(self, session: Any, template: Any) -> None: ...
    def instantiate_template(
        self,
        session: Any,
        template: Any,
        *,
        conversation_id: UUID,
        provider_profile_id: UUID | None,
        user_goal: str | None,
    ) -> Any: ...

    # --- Live System Map node comments ---------------------------------------
    def list_comments(
        self, session: Any, run_id: UUID, node_id: str | None = None
    ) -> list[Any]: ...
    def comment_to_read(self, session: Any, comment: Any) -> Any: ...
    def create_comment(
        self, session: Any, *, agent_run_id: UUID, node_id: str, body: str, author_user_id: UUID
    ) -> Any: ...
    def update_comment(
        self, session: Any, comment: Any, *, body: str | None, resolved: bool | None
    ) -> Any: ...
    def delete_comment(self, session: Any, comment: Any) -> None: ...
