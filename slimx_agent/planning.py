"""Planner schemas + validation (the portable planning layer).

Kept in sync with ControlRoom's ``services/agent/schemas.py`` (minus the host-specific
system-map inference) until the host consumes this module directly.

Two shapes, on purpose:

* ``SlimXAgentPlan`` / ``SlimXAgentPlanStep`` are plain dataclasses handed to SlimX's
  structured-output call (the same dataclass-schema convention as ``SlimXTagSuggestions``).
  They shape what the model is asked to produce.
* ``AgentPlan`` / ``AgentPlanStep`` are Pydantic models that *validate* the returned data
  dict before any row is created — structural safety, not prompt safety: an unknown step
  type, a missing field, or too many steps is rejected here, not trusted from the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from slimx_agent.contracts import ALLOWED_STEP_TYPES

# Keep v0.1 plans short and reviewable. Tunable later.
MAX_STEPS = 12


# --- Schema sent to SlimX structured output (dataclasses, like SlimXTagSuggestions) ---
#
# Every field carries a default ON PURPOSE: a small local model (e.g. llama3.2) frequently
# omits a field, and SlimX constructs the dataclass positionally — without defaults that
# raises ``TypeError: __init__() missing ... arguments`` and the whole plan call fails. With
# defaults the model's output always *constructs*; the real requirements (allowed step type,
# step count) are then enforced by the strict Pydantic ``AgentPlan`` below.


@dataclass
class SlimXAgentPlanStep:
    title: str = ""
    type: str = "model_call"
    instruction: str = ""
    expected_output: str = ""
    requires_approval: bool = False


@dataclass
class SlimXAgentPlan:
    steps: list[SlimXAgentPlanStep] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


# --- Validation models for the returned data dict ---


class AgentPlanStep(BaseModel):
    title: str
    type: str
    instruction: str = ""
    expected_output: str = ""
    requires_approval: bool = False
    input_refs: dict[str, list[str]] | None = None
    # Structured per-step params the executor needs (e.g. write_file's path/content). Kept permissive
    # (the executor validates per step type); persisted to AgentStep.params_json.
    params: dict[str, Any] | None = None

    @field_validator("type")
    @classmethod
    def _type_in_allowlist(cls, value: str) -> str:
        if value not in ALLOWED_STEP_TYPES:
            allowed = ", ".join(ALLOWED_STEP_TYPES)
            raise ValueError(f"Unsupported step type {value!r}; allowed: {allowed}")
        return value


class AgentPlan(BaseModel):
    assumptions: list[str] = []
    steps: list[AgentPlanStep]

    @field_validator("steps")
    @classmethod
    def _steps_bounded(cls, value: list[AgentPlanStep]) -> list[AgentPlanStep]:
        if not value:
            raise ValueError("Plan must contain at least one step")
        if len(value) > MAX_STEPS:
            raise ValueError(f"Plan has too many steps ({len(value)} > {MAX_STEPS})")
        return value


class PlanValidationError(Exception):
    """The model returned a plan that fails structural validation (-> 422)."""


class PlanGenerationError(Exception):
    """The planner model call itself failed or returned no data (-> 502)."""


def repair_plan_data(data: object) -> object:
    """Best-effort salvage of a small model's plan *before* strict validation: drop steps whose
    ``type`` isn't in the allowlist (rather than fail the whole plan — or, worse, run a mislabeled
    step) and fill an obviously-missing title. Non-dict input or a non-list ``steps`` is returned
    unchanged so ``validate_plan`` rejects it. Too-many-steps is deliberately NOT truncated here —
    that's left for ``validate_plan`` to reject so a retry-with-feedback can produce a shorter plan
    instead of silently dropping the tail. Keeps the strict validator as the real gate; this only
    widens what a flaky local model can produce and still have accepted."""
    if not isinstance(data, dict):
        return data
    # Schema echo: small models sometimes return the SCHEMA (e.g. assumptions =
    # {"type": "array", "items": …}) instead of values. Salvage what can be salvaged —
    # keep only string assumptions (else []) so a good steps list isn't rejected over
    # garbage assumptions; an echoed/missing steps value becomes [] so validation gives
    # the clean "at least one step" feedback the retry loop can act on.
    assumptions = data.get("assumptions")
    if not isinstance(assumptions, list):
        data = {**data, "assumptions": []}
    else:
        data = {**data, "assumptions": [a for a in assumptions if isinstance(a, str)]}
    steps = data.get("steps")
    if not isinstance(steps, list):
        return {**data, "steps": []}
    repaired: list[dict[str, Any]] = []
    for raw in steps:
        if not isinstance(raw, dict) or raw.get("type") not in ALLOWED_STEP_TYPES:
            continue
        title = str(raw.get("title") or "").strip()
        if not title:
            instruction = str(raw.get("instruction") or "").strip()
            raw = {**raw, "title": instruction[:60] or f"Step {len(repaired) + 1}"}
        repaired.append(raw)
    return {**data, "steps": repaired}


def validate_plan(data: object) -> AgentPlan:
    if not isinstance(data, dict):
        raise PlanValidationError("Planner output was not a JSON object")
    try:
        return AgentPlan.model_validate(data)
    except ValidationError as exc:
        raise PlanValidationError(str(exc)) from exc


def build_planner_prompt(
    goal: str,
    *,
    feedback: str | None = None,
    review_context: str | None = None,
    allowed_tools: list[str] | None = None,
    prior_results: bool = False,
) -> str:
    # Only the always-available assisted step types are advertised by default. web_search is an
    # external tool; it is offered to the planner ONLY when the run granted it (else the executor would
    # just skip a planned web_search, wasting a step) — and the "no external tools" instruction below is
    # relaxed accordingly.
    grants = set(allowed_tools or [])
    web_search_granted = "web_search" in grants
    code_read_granted = "code_read" in grants
    spawn_granted = "spawn_agents" in grants
    _gated_out = set()
    if not web_search_granted:
        _gated_out.add("web_search")
    if not code_read_granted:
        _gated_out |= {"code_search", "code_read"}
    if not spawn_granted:
        _gated_out |= {"spawn_run", "join_runs"}
    # mcp_call is never advertised to the planner: it needs structured params (connector_id/
    # tool/arguments) the plan schema cannot carry reliably — it enters plans via templates or
    # the API, and the executor honestly skips a bare planner-emitted one.
    _gated_out.add("mcp_call")
    advertised = [t for t in ALLOWED_STEP_TYPES if t not in _gated_out]
    allowed = ", ".join(advertised)
    prompt = (
        "You plan a short, supervised AI workflow. Output JSON ONLY, matching the schema, with "
        "2 to 5 steps. Return concrete VALUES — never the schema/type definitions themselves. "
        "EVERY step MUST include all of: title, type, instruction, "
        "expected_output, requires_approval.\n"
        f"`type` MUST be exactly one of: {allowed}. Prefer self-contained steps:\n"
        "- model_call: ask a model to analyze or write something (use for most steps).\n"
        "- compare_models: ask the same question across the workspace's models to compare answers.\n"
        "- create_synthesis: merge earlier results into one saved summary (good final step).\n"
        "- knowledge_retrieve: pull promoted decisions/syntheses from the project Knowledge Base "
        "so the run respects what the team already concluded.\n"
        "- attach_context: save earlier text as reusable context.\n"
        "Use rag_retrieve ONLY if the goal needs indexed documents, and save_evidence ONLY if a "
        "specific quote and its source are already known. "
    )
    if web_search_granted:
        prompt += (
            "Use web_search ONLY when the goal needs current or missing information the workspace does "
            "not already contain; it sends a query to an external service, so plan it sparingly and "
            "feed its results into a later model_call or create_synthesis. Never plan steps that delete "
            "data, run code, mutate files, or send messages. "
        )
    else:
        prompt += (
            "Never plan steps that delete data, run code, call external tools (including web search), "
            "mutate files, or send messages. "
        )
    if code_read_granted:
        prompt += (
            "You may inspect the codebase (read-only): code_search finds where something is, code_read "
            "reads one file by its repo-relative path. Use them to ground code analysis in real source, "
            "then feed findings into a model_call/create_synthesis. You CANNOT edit or run code. "
        )
    if spawn_granted:
        prompt += (
            "You may delegate to sub-agents: each spawn_run step creates one sub-agent for ONE focused "
            "sub-goal (write the sub-goal as that step's instruction; at most 3 spawn_run steps). Plan "
            "exactly one join_runs step AFTER them — it executes all sub-agents and saves their results "
            "as context — then finish with a model_call or create_synthesis that merges those results. "
        )
    prompt += "Set requires_approval=true for any step the user should review before it runs.\n"
    if review_context:
        # A durable review packet (highlights/comments/Ask branches/tags) is already attached as
        # conversation context, so the model sees it during execution. Steer the plan to USE it rather
        # than emit a generic single model_call — but never restate the packet text into the plan.
        prompt += (
            "\nThe user attached document-review context to this run: "
            f"{review_context} It is already available to every step as conversation context, so DO "
            "NOT paste it into instructions. Plan steps that WORK ON this review context — favor "
            "create_synthesis to synthesize the review, and use rag_retrieve / attach_context / "
            "save_evidence where they fit — instead of a single generic model_call.\n"
        )
    if prior_results:
        # Follow-up turns (v3 autonomy): earlier agent runs in this conversation already produced
        # results, persisted as context this run's steps will see. Build on them, don't redo them.
        prompt += (
            "\nEarlier agent results from this conversation are attached as context. Plan steps "
            "that BUILD ON them — verify, extend, or synthesize further — instead of redoing "
            "completed work.\n"
        )
    prompt += f"\nGoal: {goal}"
    if feedback:
        # Retry guidance: tell the model exactly why the last attempt was rejected so it can correct
        # the specific problem (wrong step type, missing field, too many steps) rather than reroll blind.
        prompt += (
            "\n\nYour previous attempt was rejected by validation: "
            f"{feedback}\nReturn corrected JSON ONLY, matching the schema."
        )
    return prompt


# --- Completion self-check (v3 autonomy) ----------------------------------------------------
# Defaulted dataclass for the structured verdict a completed run is checked against its goal
# with (AGENT_AUTO_ITERATE). Defaults make weak local models construct; ``satisfied=True`` is
# the fail-safe — a malformed verdict never triggers an iteration.


@dataclass
class SlimXOutcomeVerdict:
    satisfied: bool = True
    gaps: list[str] = field(default_factory=list)


def build_self_check_prompt(goal: str, result_text: str) -> str:
    return (
        "You verify whether an AI agent's result satisfies the user's goal. Output JSON ONLY, "
        "matching the schema: satisfied (boolean), gaps (list of strings). Set satisfied=true "
        "unless something MATERIAL the goal asked for is missing. List at most 3 concrete, "
        "actionable gaps (an unanswered part of the goal, missing analysis, unsupported claims). "
        "Never invent scope beyond the goal.\n\n"
        f"Goal: {goal}\n\nResult:\n{result_text}"
    )
