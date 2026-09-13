"""The remote tool registry: every step type dispatches back to the host.

The standalone service owns the LOOP (gates, ordering, transitions, events); the host owns
the TOOLS (they need the host's database, capability services, model transport, and egress
policy). One generic handler per contract step type turns the host's invocation-outcome
envelope back into the engine's native vocabulary:

=============================================  ======================================
Host answer                                    Engine outcome
=============================================  ======================================
``completed`` + object or null ``output_refs``  output references (step completes)
``skipped``                                    ``StepNotApplicable`` (honest skip)
``prepared``                                   ``StepActionPrepared`` (gates re-apply)
``failed``                                     ``StepExecutionError`` (step fails)
anything else, or no answer at all             ``StepOutcomeUnknown`` (drive ends)
=============================================  ======================================

"Anything else" covers an unrecognized outcome, a malformed envelope, a host refusal
(4xx/5xx), and a transport failure after which the host may already have run the tool. None of
those is an authoritative outcome, so the engine must not project one and must not retry.
"""

from __future__ import annotations

from collections.abc import Iterable

from slimx_agent import contracts
from slimx_agent.host_client import HostClient, HostError
from slimx_agent.http_store import RunSnapshot, StepSnapshot
from slimx_agent.runtime import ProfileView
from slimx_agent.store import OutputRefs
from slimx_agent.tools import (
    StepActionPrepared,
    StepExecutionError,
    StepNotApplicable,
    StepOutcomeUnknown,
    ToolRegistry,
)

type RemoteRegistry = ToolRegistry[HostClient, RunSnapshot, StepSnapshot, ProfileView]


def _remote_handler(
    client: HostClient, run: RunSnapshot, step: StepSnapshot, profile: ProfileView
) -> OutputRefs | None:
    try:
        result = client.invoke_step(run.id, step.id, profile)
    except HostError as exc:
        raise StepOutcomeUnknown(
            f"step {step.id} has no observed invocation outcome ({exc})"
        ) from exc
    outcome = result.get("outcome")
    if outcome == "completed":
        refs = result.get("output_refs")
        if refs is None:
            return None
        if isinstance(refs, dict):
            return refs
        raise StepOutcomeUnknown(f"step {step.id} completion carried malformed output references")
    if outcome == "skipped":
        raise StepNotApplicable(_message(result.get("reason"), "step not applicable"))
    if outcome == "prepared":
        raise StepActionPrepared(
            _message(result.get("reason"), "a new action generation was prepared")
        )
    if outcome == "failed":
        raise StepExecutionError(_message(result.get("error"), "step failed on the host"))
    raise StepOutcomeUnknown(f"step {step.id} returned an unrecognized invocation outcome")


def _message(value: object, default: str) -> str:
    return value if isinstance(value, str) and value.strip() else default


def build_remote_registry(step_types: Iterable[str] | None = None) -> RemoteRegistry:
    """A registry whose every handler invokes the step on the host. Defaults to the full
    contract vocabulary; an unknown type in stored data still resolves to ``None`` and fails
    the step honestly, exactly like the in-process registry."""
    registry: RemoteRegistry = ToolRegistry()
    for step_type in contracts.ALLOWED_STEP_TYPES if step_types is None else step_types:
        registry.register(step_type, _remote_handler)
    return registry
