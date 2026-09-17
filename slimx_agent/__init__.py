"""SlimX-Agent: the portable agent core of the SlimX platform.

This package owns:

- the dependency-light **contracts** every agent surface shares — step types, tool grants,
  run modes, approval policies, and the durable event vocabulary (``contracts``);
- the deterministic **policies** behind the permission and approval gates (``policies``);
- the typed engine boundary — the ``RunStore`` protocol with its run/step views (``store``),
  the ``ToolRegistry`` dispatch boundary and step-outcome vocabulary (``tools``) — and the
  **execution engine** itself (``engine``);
- the host-facing ``AgentRuntime`` protocol and ``RunProfile`` (``runtime``);
- portable **planning** schemas, repair, and validation (``planning``; needs pydantic);
- the optional **standalone service** (``service``): the engine loop in its own container,
  driving a host's callback API (install the ``service`` extra).

It deliberately does NOT own persistence, model transport, credentials, authorization, or any
host capability (evidence, synthesis, RAG, MCP, sandboxes): hosts keep those behind tool
handlers and the callback API. ControlRoom consumes the contracts, policies, and engine
directly; its active planner remains host-owned. No agent framework (LangChain/LangGraph/
CrewAI/AutoGen/OpenAI-Agents-SDK) is, or may ever become, a dependency of this package.
"""

from slimx_agent.contracts import (
    AGENT_MODES,
    ALLOWED_STEP_TYPES,
    APPROVAL_POLICIES,
    EVENT_TYPES,
    GRANTABLE_TOOLS,
    STEP_STATUSES,
)
from slimx_agent.engine import RunningStepNotPermitted, UnknownStepStatus
from slimx_agent.runtime import AgentRunConflict, AgentRuntime, ProfileView, RunProfile
from slimx_agent.store import UNSET, RunStore, RunView, StepView
from slimx_agent.tools import (
    AgentRunContext,
    StepActionPrepared,
    StepExecutionError,
    StepNotApplicable,
    StepOutcomeUnknown,
    ToolHandler,
    ToolRegistry,
)

__all__ = [
    "AGENT_MODES",
    "ALLOWED_STEP_TYPES",
    "APPROVAL_POLICIES",
    "EVENT_TYPES",
    "GRANTABLE_TOOLS",
    "STEP_STATUSES",
    "UNSET",
    "AgentRunConflict",
    "AgentRunContext",
    "AgentRuntime",
    "ProfileView",
    "RunProfile",
    "RunStore",
    "RunView",
    "RunningStepNotPermitted",
    "StepActionPrepared",
    "StepExecutionError",
    "StepNotApplicable",
    "StepOutcomeUnknown",
    "StepView",
    "ToolHandler",
    "ToolRegistry",
    "UnknownStepStatus",
    "__version__",
]

# The single maintained version source: pyproject reads it at build time
# (``[tool.setuptools.dynamic]``), and the health endpoint reports it.
__version__ = "0.21.0"
