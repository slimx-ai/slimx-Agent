"""The remote tool registry: every step type dispatches back to the host.

The standalone service owns the LOOP (gates, ordering, transitions, events); the host owns
the TOOLS (they need the host's database, capability services, model transport, and egress
policy). One generic handler per contract step type turns the host's invocation-outcome
envelope back into the engine's native vocabulary — ``StepNotApplicable`` for an honest
skip, ``StepExecutionError`` for a failure — so the engine applies exactly the same
transitions it would in-process.
"""

from __future__ import annotations

from typing import Any

from slimx_agent import contracts
from slimx_agent.host_client import HostClient
from slimx_agent.tools import StepExecutionError, StepNotApplicable, ToolRegistry


def _remote_handler(client: HostClient, run: Any, step: Any, profile: Any) -> dict[str, Any] | None:
    result = client.invoke_step(run.id, step.id, profile)
    outcome = result.get("outcome")
    if outcome == "completed":
        refs = result.get("output_refs")
        return refs if isinstance(refs, dict) else None
    if outcome == "skipped":
        raise StepNotApplicable(str(result.get("reason") or "step not applicable"))
    raise StepExecutionError(str(result.get("error") or "step failed on the host"))


def build_remote_registry(step_types: tuple[str, ...] | None = None) -> ToolRegistry:
    """A registry whose every handler invokes the step on the host. Defaults to the full
    contract vocabulary; an unknown type in stored data still resolves to ``None`` and fails
    the step honestly, exactly like the in-process registry."""
    registry = ToolRegistry()
    for step_type in step_types or contracts.ALLOWED_STEP_TYPES:
        registry.register(step_type, _remote_handler)
    return registry
