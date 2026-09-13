"""Portable planning over untrusted model output.

The chain under test is generation schema → structured-output coercion → repair → strict
validation. ``coerce_declared`` models the structured-output boundary as the host documents it
(the schema declares exactly the dataclass fields and coercion copies only declared fields), so
the tests exercise field fidelity at the risky boundary instead of injecting dicts after it.
The downstream host suite repeats the check against the real pinned SlimX coercion.

(No ``from __future__ import annotations`` here: the local legacy dataclasses below must carry
real annotations for ``typing.get_type_hints``.)
"""

import copy
import dataclasses
import itertools
import json
import typing
from types import SimpleNamespace
from typing import Any

import pytest

from slimx_agent import policies
from slimx_agent.contracts import ALLOWED_STEP_TYPES, GRANTABLE_TOOLS
from slimx_agent.planning import (
    MAX_STEPS,
    NEVER_ADVERTISED_STEP_TYPES,
    AgentPlanStep,
    PlanValidationError,
    SlimXAgentPlan,
    SlimXAgentPlanStep,
    SlimXOutcomeVerdict,
    SlimXResearchCheck,
    advertised_step_types,
    build_planner_prompt,
    build_self_check_prompt,
    repair_plan_data,
    validate_plan,
)


def coerce_declared(cls: Any, data: dict[str, Any]) -> Any:
    """Build ``cls`` from ``data`` copying ONLY declared dataclass fields, recursing into nested
    dataclasses and lists of them — undeclared keys are dropped, as a declared-schema coercion
    drops them."""
    hints = typing.get_type_hints(cls)
    return cls(
        **{
            f.name: _coerce(hints[f.name], data[f.name])
            for f in dataclasses.fields(cls)
            if f.name in data
        }
    )


def _coerce(tp: Any, value: Any) -> Any:
    if dataclasses.is_dataclass(tp) and isinstance(value, dict):
        return coerce_declared(tp, value)
    if typing.get_origin(tp) is list and isinstance(value, list):
        (item_type,) = typing.get_args(tp)
        return [_coerce(item_type, item) for item in value]
    return value


MODEL_OUTPUT: dict[str, Any] = {
    "assumptions": ["the sandbox starts empty"],
    "steps": [
        {
            "title": "Write the page",
            "type": "write_file",
            "instruction": "Create index.html",
            "expected_output": "one file",
            "requires_approval": False,
            "params": {"path": "index.html", "content": "<h1>hi</h1>", "nested": {"k": [1, 2]}},
            "input_refs": {"context": ["step-1", "step-2"]},
        },
        {
            "title": "Check",
            "type": "run_check",
            "instruction": "run the tests",
            "params": {"command": "pytest -q"},
        },
        {
            "title": "Summarize",
            "type": "create_synthesis",
            "instruction": "Summarize the result",
            "undeclared": "dropped by the declared schema",
        },
    ],
}


def _simple_step(index: int) -> dict[str, Any]:
    return {"title": f"s{index}", "type": "model_call", "instruction": f"do {index}"}


# --- field fidelity at the structured-output boundary ------------------------------------


def test_generation_schema_declares_every_validated_step_field():
    declared = {f.name for f in dataclasses.fields(SlimXAgentPlanStep)}
    assert set(AgentPlanStep.model_fields) <= declared
    blank = SlimXAgentPlanStep()
    assert blank.params is None and blank.input_refs is None


def test_executable_fields_survive_coercion_repair_and_validation():
    coerced = coerce_declared(SlimXAgentPlan, copy.deepcopy(MODEL_OUTPUT))
    wire = json.loads(json.dumps(dataclasses.asdict(coerced)))  # JSON-serializable end to end
    assert "undeclared" not in wire["steps"][2]

    plan = validate_plan(repair_plan_data(wire))

    assert plan.steps[0].params == MODEL_OUTPUT["steps"][0]["params"]
    assert plan.steps[0].input_refs == {"context": ["step-1", "step-2"]}
    assert plan.steps[1].params == {"command": "pytest -q"}
    assert (plan.steps[2].params, plan.steps[2].input_refs) == (None, None)
    assert plan.assumptions == ["the sandbox starts empty"]


def test_the_pre_0_20_generation_shape_silently_dropped_executable_fields():
    """Regression evidence for the correction: through the same boundary, the previous
    five-field step shape loses ``params``/``input_refs`` before validation can see them."""

    @dataclasses.dataclass
    class LegacyStep:
        title: str = ""
        type: str = "model_call"
        instruction: str = ""
        expected_output: str = ""
        requires_approval: bool = False

    @dataclasses.dataclass
    class LegacyPlan:
        steps: list[LegacyStep] = dataclasses.field(default_factory=list)
        assumptions: list[str] = dataclasses.field(default_factory=list)

    legacy = validate_plan(dataclasses.asdict(coerce_declared(LegacyPlan, MODEL_OUTPUT)))
    assert legacy.steps[0].params is None
    assert legacy.steps[0].input_refs is None


def test_new_generation_fields_are_optional_and_positionally_compatible():
    step = SlimXAgentPlanStep("t", "model_call", "i", "e", True)
    assert (step.title, step.requires_approval, step.params, step.input_refs) == (
        "t",
        True,
        None,
        None,
    )
    assert [f.name for f in dataclasses.fields(SlimXAgentPlanStep)][-2:] == [
        "params",
        "input_refs",
    ]


def test_research_checkpoint_extensions_carry_the_same_executable_fields():
    check = coerce_declared(
        SlimXResearchCheck,
        {"satisfied": False, "next_steps": [MODEL_OUTPUT["steps"][0]]},
    )
    assert check.next_steps[0].params == MODEL_OUTPUT["steps"][0]["params"]
    assert check.next_steps[0].input_refs == {"context": ["step-1", "step-2"]}


# --- the portable limit -------------------------------------------------------------------


def test_portable_limit_accepts_exactly_the_maximum_and_rejects_one_more():
    assert MAX_STEPS == 12  # the portable default; host ceilings are separate (AGENT-LIMIT-001)
    assert len(validate_plan({"steps": [_simple_step(i) for i in range(MAX_STEPS)]}).steps) == 12
    with pytest.raises(PlanValidationError, match="too many steps"):
        validate_plan({"steps": [_simple_step(i) for i in range(MAX_STEPS + 1)]})


def test_repair_never_truncates_an_over_limit_plan():
    repaired = repair_plan_data({"steps": [_simple_step(i) for i in range(MAX_STEPS + 1)]})
    assert isinstance(repaired, dict)
    assert len(repaired["steps"]) == MAX_STEPS + 1
    with pytest.raises(PlanValidationError, match="too many steps"):
        validate_plan(repaired)


def test_an_empty_plan_is_rejected_with_actionable_feedback():
    with pytest.raises(PlanValidationError, match="at least one step"):
        validate_plan({"steps": []})


# --- untrusted output: malformed, echoed, and aliased -------------------------------------


@pytest.mark.parametrize("data", [None, [], "a plan", 5, ["steps"]])
def test_non_object_output_is_rejected_and_left_untouched_by_repair(data):
    assert repair_plan_data(data) is data
    with pytest.raises(PlanValidationError, match="not a JSON object"):
        validate_plan(data)


def test_schema_echoes_become_clean_retry_feedback():
    echoed = {"assumptions": {"type": "array"}, "steps": {"type": "array", "items": {}}}
    repaired = repair_plan_data(echoed)
    assert repaired == {"assumptions": [], "steps": []}
    with pytest.raises(PlanValidationError, match="at least one step"):
        validate_plan(repaired)
    # An echoed assumptions value never sinks an otherwise good plan.
    good = repair_plan_data({"assumptions": {"type": "array"}, "steps": [_simple_step(1)]})
    assert validate_plan(good).assumptions == []


def test_repair_drops_non_objects_and_unknown_types_but_keeps_valid_steps():
    repaired = repair_plan_data(
        {
            "assumptions": ["kept", 3, None],
            "steps": ["text", 5, None, {"type": "rm_rf"}, {"title": "no type"}, _simple_step(1)],
        }
    )
    plan = validate_plan(repaired)
    assert [step.type for step in plan.steps] == ["model_call"]
    assert plan.assumptions == ["kept"]


def test_model_steps_need_an_instruction_but_aliases_are_adopted():
    with pytest.raises(PlanValidationError, match="non-empty 'instruction'"):
        validate_plan({"steps": [{"title": "t", "type": "model_call"}]})
    with pytest.raises(PlanValidationError, match="non-empty 'instruction'"):
        validate_plan({"steps": [{"title": "t", "type": "compare_models", "instruction": " "}]})
    aliased = repair_plan_data({"steps": [{"type": "model_call", "prompt": "use the alias"}]})
    plan = validate_plan(aliased)
    assert plan.steps[0].instruction == "use the alias"
    assert plan.steps[0].title == "use the alias"
    # Non-model steps may omit the instruction.
    assert validate_plan({"steps": [{"title": "t", "type": "attach_context"}]}).steps


def test_titles_are_repaired_from_name_instruction_or_position():
    repaired = repair_plan_data(
        {
            "steps": [
                {"type": "model_call", "name": "Named", "instruction": "x"},
                {"type": "model_call", "instruction": "y" * 80},
                {"type": "attach_context"},
            ]
        }
    )
    titles = [step.title for step in validate_plan(repaired).steps]
    assert titles == ["Named", "y" * 60, "Step 3"]


@pytest.mark.parametrize(
    "bad_field",
    [
        {"params": "path=index.html"},
        {"params": ["path", "index.html"]},
        {"input_refs": {"context": "step-1"}},
        {"input_refs": {"context": [1, 2]}},
        {"input_refs": ["step-1"]},
    ],
)
def test_malformed_executable_fields_are_rejected_not_coerced(bad_field):
    with pytest.raises(PlanValidationError):
        validate_plan({"steps": [{**_simple_step(1), **bad_field}]})


def test_repair_and_validation_never_mutate_the_callers_data():
    data = copy.deepcopy(MODEL_OUTPUT)
    data["assumptions"] = {"type": "array"}
    data["steps"] = [*data["steps"], {"type": "rm_rf"}, {"type": "model_call", "prompt": "p"}]
    snapshot = copy.deepcopy(data)
    validate_plan(repair_plan_data(data))
    assert data == snapshot


def test_preserved_params_never_authorize_a_step():
    """Validation is structural. A preserved ``params`` payload leaves the hard gate, the grant
    requirement, and the planner-advertisement rule exactly where they were."""
    plan = validate_plan(
        {
            "steps": [
                {
                    "title": "call",
                    "type": "mcp_call",
                    "params": {"connector_id": "c", "tool": "delete_everything"},
                    "requires_approval": False,
                }
            ]
        }
    )
    step = plan.steps[0]
    assert policies.classify_step(step)[0] == policies.HARD_GATED
    assert policies.permission_block_reason(step, SimpleNamespace(allowed_tools_json=[]))
    assert "mcp_call" not in advertised_step_types(GRANTABLE_TOOLS)


# --- grant-aware advertisement ----------------------------------------------------------


def test_advertisement_matches_the_permission_gate_for_every_grant_subset():
    for size in range(len(GRANTABLE_TOOLS) + 1):
        for subset in itertools.combinations(GRANTABLE_TOOLS, size):
            advertised = set(advertised_step_types(subset))
            for step_type in ALLOWED_STEP_TYPES:
                grant = policies.required_grant(step_type)
                executable = grant is None or grant in subset
                expected = executable and step_type not in NEVER_ADVERTISED_STEP_TYPES
                assert (step_type in advertised) is expected, (subset, step_type)


def _advertised_in_prompt(prompt: str) -> list[str]:
    line = next(line for line in prompt.splitlines() if line.startswith("`type` MUST be"))
    return line.split(":", 1)[1].split(". Prefer", 1)[0].strip().split(", ")


@pytest.mark.parametrize(
    "grants", [[], ["web_search"], ["evidence_write", "code_read"], list(GRANTABLE_TOOLS)]
)
def test_the_prompt_lists_exactly_the_advertised_types(grants):
    prompt = build_planner_prompt("Do the thing", allowed_tools=grants)
    assert _advertised_in_prompt(prompt) == list(advertised_step_types(grants))
    for never in NEVER_ADVERTISED_STEP_TYPES:
        assert never not in prompt


def test_the_ungranted_prompt_no_longer_offers_gated_or_never_advertised_types():
    bare = set(_advertised_in_prompt(build_planner_prompt("Do the thing")))
    for step_type in (
        "plugin_tool",
        "web_fetch",
        "create_work_item",
        "link_work_item",
        "promote_to_knowledge",
        "apply_patch_sandbox",
        "run_check",
        "package_patch",
        "propose_patch",
    ):
        assert step_type not in bare
    assert {"model_call", "evidence_query", "work_items_read", "compose_report"} <= bare


def test_junk_grant_values_advertise_nothing_extra():
    assert advertised_step_types(["web_search", 7, None]) == advertised_step_types(["web_search"])


def test_direct_validation_rejects_unknown_types_without_repair():
    with pytest.raises(PlanValidationError, match="Unsupported step type 'rm_rf'"):
        validate_plan({"steps": [{"title": "t", "type": "rm_rf"}]})


def test_optional_prompt_sections_appear_only_when_requested():
    bare = build_planner_prompt("Goal text")
    assert bare.endswith("Goal: Goal text")
    for marker in (
        "document-review context",
        "Earlier agent results",
        "rejected by validation",
        "EXACTLY as listed",
    ):
        assert marker not in bare
    full = build_planner_prompt(
        "Goal text",
        review_context="3 highlights",
        prior_results=True,
        feedback="too many steps",
        evidence_hint="tags: Risk (3).",
    )
    assert "document-review context to this run: 3 highlights" in full
    assert "Earlier agent results" in full
    assert "rejected by validation: too many steps" in full
    assert "tags: Risk (3)." in full


def test_grant_specific_guidance_follows_the_grants():
    granted = build_planner_prompt(
        "g",
        allowed_tools=["code_read", "evidence_write", "spawn_agents", "netops_read", "web_search"],
    )
    for marker in (
        "inspect the codebase",
        "save findings back",
        "delegate to sub-agents",
        "READ-ONLY network telemetry",
        "Use web_search ONLY",
    ):
        assert marker in granted
    assert "call external tools (including web search)" in build_planner_prompt("g")


def test_self_check_prompt_and_fail_safe_verdict():
    prompt = build_self_check_prompt("Ship it", "Result body")
    assert "Goal: Ship it" in prompt and prompt.endswith("Result:\nResult body")
    assert SlimXOutcomeVerdict().satisfied is True
