"""SlimX-Agent tool contracts: step outcomes, the typed handler shape, and the registry.

The explicit boundary between the agent engine and a host's capabilities:

- **Step outcomes** — besides returning output references, a handler may signal a genuine
  failure (:class:`StepExecutionError`), an honest skip (:class:`StepNotApplicable`), a durably
  prepared new action generation (:class:`StepActionPrepared`), or that the step's outcome is
  unknown to the engine (:class:`StepOutcomeUnknown`). Only failure and skip produce terminal
  step states; the other two never do.
- **ToolHandler / ToolRegistry** — the engine dispatches step types ONLY through a registry the
  host populates. ControlRoom registers one governed handler per contract step type; the
  standalone service registers one remote handler per type (``http_tools``). The registry fails
  loudly on duplicate registration and resolves unknown types to ``None`` (the engine fails that
  step honestly).
- **AgentRunContext / ContextProvider** — the typed context package a host hands model-using
  handlers; the engine never reaches into host tables for context itself.

Dependency rule: the standard library plus this package only, so every host can import the
vocabulary without the ``service`` extra. A guard test enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from slimx_agent.store import OutputRefs


class StepExecutionError(Exception):
    """A genuine failure: the model or an underlying service errored. Stops the run."""


class StepNotApplicable(Exception):
    """An additive step whose required inputs are absent (e.g. save_evidence with no anchor).

    Not a failure — the step is skipped and the run continues. Lets a planner that emits a
    step it can't supply inputs for degrade gracefully instead of killing an otherwise-good run.
    """


class StepActionPrepared(Exception):
    """The host durably prepared a new action generation instead of executing the target.

    This is a control-plane outcome, not a failure or a skip. The engine re-reads the
    authoritative run and step and applies the gates to that prepared generation afresh without
    emitting a terminal step event; approval of an earlier generation never carries over.
    """


class StepOutcomeUnknown(Exception):
    """The engine could not observe an authoritative outcome for the step.

    Raised when a step may already have crossed the host's entry boundary but its outcome
    never reached the engine — for example the standalone service lost the invocation
    response, the host refused the callback, or the answer was not a recognizable outcome.
    It is neither a failure nor a skip: the engine writes no terminal step state or event, never
    retries the step, and re-raises this exception to end the drive. The host's durable
    invocation record, not the engine, decides what happened.
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


class ToolHandler[ContextT, RunT, StepT, ProfileT](Protocol):
    """One step-type handler: executes the step against host services and returns the
    ``output_refs`` dict (small references only, or ``None``). Signals via the step outcomes
    above. Parameters are positional: ``(handler_context, run, step, profile)``."""

    def __call__(
        self,
        context: ContextT,
        run: RunT,
        step: StepT,
        profile: ProfileT,
        /,
    ) -> OutputRefs | None: ...


class ContextProvider[ContextT, RunT](Protocol):
    """Builds the :class:`AgentRunContext` for a run — the host's translation of its own
    workspace/document/conversation state into the engine's typed context package."""

    def __call__(self, context: ContextT, run: RunT, /) -> AgentRunContext: ...


class ToolRegistry[ContextT, RunT, StepT, ProfileT]:
    """Step type → handler. The ONLY path from the dispatcher to a tool implementation.

    Parameterize it with the host's handler context, run, step, and profile types (for
    example ``ToolRegistry[Session, AgentRun, AgentStep, ExecProfile]``) so a type checker
    rejects a handler with the wrong shape at registration time.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler[ContextT, RunT, StepT, ProfileT]] = {}

    def register(
        self, step_type: str, handler: ToolHandler[ContextT, RunT, StepT, ProfileT]
    ) -> None:
        """Register a handler; a duplicate step type is a wiring bug and fails loudly."""
        if step_type in self._handlers:
            raise ValueError(f"Tool handler for {step_type!r} is already registered")
        self._handlers[step_type] = handler

    def resolve(self, step_type: str) -> ToolHandler[ContextT, RunT, StepT, ProfileT] | None:
        """The handler for ``step_type``, or None (dispatcher fails the step honestly)."""
        return self._handlers.get(step_type)

    @property
    def step_types(self) -> tuple[str, ...]:
        return tuple(self._handlers)
