"""SlimX-Agent tool & context contracts (Stage D of docs/slimx-agent-extraction-plan.md).

The explicit boundary between the agent engine and the host's capabilities:

- **Step errors** — the two outcomes a tool handler may signal besides success. Moved here
  from ``executor_service`` (which re-exports them) because they are the engine's vocabulary,
  not an implementation detail.
- **ToolHandler / ToolRegistry** — the executor dispatches step types ONLY through a registry
  the host populates. ControlRoom registers its 11 handlers (model calls, RAG, context,
  synthesis, evidence, web search, code tools, build sandbox); a future host registers its
  own. The registry fails loudly on duplicate registration and resolves unknown types to
  ``None`` (the dispatcher fails that step honestly).
- **AgentRunContext / ContextProvider** — the typed context package a host hands the engine
  for model-using steps. ControlRoom builds it from the run conversation's enabled context
  sources (review packet, attach_context, web_search, code, rag); the engine never reaches
  into host tables for context itself.

Dependency rule: stdlib only (same as ``contracts``), so this module moves verbatim into the
standalone package. A guard test enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class StepExecutionError(Exception):
    """A genuine failure: the model or an underlying service errored. Stops the run."""


class StepNotApplicable(Exception):
    """An additive step whose required inputs are absent (e.g. save_evidence with no anchor).

    Not a failure — the step is skipped and the run continues. Lets a planner that emits a
    step it can't supply inputs for degrade gracefully instead of killing an otherwise-good run.
    """


@dataclass
class AgentRunContext:
    """The context that grounds an agent model call, plus its provenance manifest.

    ``reference_text`` is the already-neutralized, budget-bounded reference block prepended to
    the step prompt; ``manifest`` lists where it came from (small refs: id/kind/name — never
    content). Empty means "run on the bare instruction", byte-for-byte the no-context behavior.
    """

    reference_text: str = ""
    manifest: list[dict[str, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.reference_text


class ToolHandler(Protocol):
    """One step-type handler: executes the step against host services and returns the
    ``output_refs`` dict (small references only). Signals via the step errors above."""

    def __call__(self, session: Any, run: Any, step: Any, profile: Any) -> dict[str, Any]: ...


class ContextProvider(Protocol):
    """Builds the :class:`AgentRunContext` for a run — the host's translation of its own
    workspace/document/conversation state into the engine's typed context package."""

    def __call__(self, session: Any, run: Any) -> AgentRunContext: ...


class ToolRegistry:
    """Step type → handler. The ONLY path from the dispatcher to a tool implementation."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, step_type: str, handler: ToolHandler) -> None:
        """Register a handler; a duplicate step type is a wiring bug and fails loudly."""
        if step_type in self._handlers:
            raise ValueError(f"Tool handler for {step_type!r} is already registered")
        self._handlers[step_type] = handler

    def resolve(self, step_type: str) -> ToolHandler | None:
        """The handler for ``step_type``, or None (dispatcher fails the step honestly)."""
        return self._handlers.get(step_type)

    @property
    def step_types(self) -> tuple[str, ...]:
        return tuple(self._handlers)
