"""SlimX-Agent: the portable core of the SlimX platform's agent layer.

Extracted from SlimX-AI ControlRoom (extraction plan Stage H). This package owns the
**dependency-light contracts** every agent surface shares — step types, tool grants, run
modes, approval policies, the durable event vocabulary, the ``ToolRegistry`` dispatch
boundary, the ``AgentRuntime`` protocol, and the pure planning schemas/prompt/validation.

It deliberately does NOT own the execution engine, persistence, model transport, or any
host capability (evidence, synthesis, RAG, MCP) — hosts register those as tool handlers
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
from slimx_agent.tools import (
    AgentRunContext as AgentRunContext,
    StepExecutionError as StepExecutionError,
    StepNotApplicable as StepNotApplicable,
    ToolRegistry as ToolRegistry,
)

__version__ = "0.1.0"
