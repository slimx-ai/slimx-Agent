"""SlimX-Agent: the portable core of the SlimX platform's agent layer.

Extracted from SlimX-AI ControlRoom (extraction plan Stage H). This package owns the
**dependency-light contracts** every agent surface shares — step types, tool grants, run
modes, approval policies, the durable event vocabulary, the ``ToolRegistry`` dispatch
boundary, the ``AgentRuntime`` protocol, and the pure planning schemas/prompt/validation.

It also owns the **execution engine** (the dispatch loop over the ``RunStore`` protocol,
Stage I) and the **standalone service** (``slimx_agent.service``, the loop in its own
container driving a host's internal agent-host callback API — install the ``service``
extra). It deliberately does NOT own persistence, model transport, or any host capability
(evidence, synthesis, RAG, MCP) — hosts keep those behind tool handlers / the callback API
and implement ``AgentRuntime``. No agent framework (LangChain/LangGraph/CrewAI/AutoGen/
OpenAI-Agents-SDK) is, or may ever become, a dependency of this package.

``contracts``/``tools``/``runtime`` are stdlib-only and byte-identical with the ControlRoom
copies until the host consumes this package directly (a parity guard on the host enforces
that); ``planning`` additionally needs pydantic.
"""

from slimx_agent.contracts import (
    AGENT_MODES as AGENT_MODES,
    ALLOWED_STEP_TYPES as ALLOWED_STEP_TYPES,
    APPROVAL_POLICIES as APPROVAL_POLICIES,
    EVENT_TYPES as EVENT_TYPES,
    GRANTABLE_TOOLS as GRANTABLE_TOOLS,
)
from slimx_agent.runtime import (
    AgentRunConflict as AgentRunConflict,
    AgentRuntime as AgentRuntime,
    RunProfile as RunProfile,
)
from slimx_agent.store import (
    RunStore as RunStore,
)
from slimx_agent.tools import (
    AgentRunContext as AgentRunContext,
    StepExecutionError as StepExecutionError,
    StepNotApplicable as StepNotApplicable,
    ToolRegistry as ToolRegistry,
)

__version__ = "0.4.0"
