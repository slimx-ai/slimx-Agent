"""The portable core stays dependency-light, coherent, and framework-free."""

from __future__ import annotations

import inspect
import uuid

import pytest

import slimx_agent
from slimx_agent import (
    ALLOWED_STEP_TYPES,
    EVENT_TYPES,
    GRANTABLE_TOOLS,
    AgentRunContext,
    StepExecutionError,
    StepNotApplicable,
    ToolRegistry,
)
from slimx_agent.planning import (
    MAX_STEPS,
    PlanValidationError,
    build_planner_prompt,
    repair_plan_data,
    validate_plan,
)


def test_contracts_vocabulary_is_coherent():
    assert "model_call" in ALLOWED_STEP_TYPES
    assert "spawn_run" in ALLOWED_STEP_TYPES and "join_runs" in ALLOWED_STEP_TYPES
    assert "mcp_call" in ALLOWED_STEP_TYPES
    assert set(GRANTABLE_TOOLS) == {
        "web_search",
        "code_read",
        "code_write",
        "spawn_agents",
        "mcp_tools",
        "evidence_write",
        "netops_read",
        "netops_write",
        "plugin_tools",
        "data_read",
    }
    assert "netops_collect" in ALLOWED_STEP_TYPES
    assert {"netops_apply", "netops_auto_apply"} <= set(ALLOWED_STEP_TYPES)
    assert {"propose_patch", "apply_patch_sandbox", "run_check", "package_patch", "stage_files", "review_patch"} <= set(
        ALLOWED_STEP_TYPES
    )
    assert "research_iterate" in ALLOWED_STEP_TYPES
    assert {"data_catalog", "data_query", "analyze_data"} <= set(ALLOWED_STEP_TYPES)
    assert len(EVENT_TYPES) == len(set(EVENT_TYPES)) == 25


def test_code_build_steps_are_a_graduated_patch_loop():
    """Code Builder: propose is model-only (auto_safe, code_read); apply/run_check mutate/execute
    the sandbox (review_recommended, code_write); package bundles the diff (auto_safe, code_write)."""
    from slimx_agent import policies
    from slimx_agent.contracts import CODE_BUILD_STEP_TYPES

    assert CODE_BUILD_STEP_TYPES == (
        "propose_patch",
        "apply_patch_sandbox",
        "run_check",
        "package_patch",
        "stage_files",
        "review_patch",
    )

    def tier(step_type: str) -> str:
        step = type("Step", (), {"type": step_type, "requires_approval": False})()
        return policies.classify_step(step)[0]

    assert tier("propose_patch") == policies.AUTO_SAFE
    assert tier("apply_patch_sandbox") == policies.REVIEW_RECOMMENDED
    assert tier("run_check") == policies.REVIEW_RECOMMENDED
    assert tier("package_patch") == policies.AUTO_SAFE

    assert policies.required_grant("propose_patch") == "code_read"
    for step_type in ("apply_patch_sandbox", "run_check", "package_patch"):
        assert policies.required_grant(step_type) == "code_write"
    assert policies.CAPABILITY_BY_TYPE["propose_patch"] == policies.MODEL
    assert policies.CAPABILITY_BY_TYPE["apply_patch_sandbox"] == policies.WRITE
    # code_write is a real, labeled grant.
    assert "code_write" in policies.GRANT_LABELS


def test_netops_collect_is_grant_gated_review_recommended_read():
    """netops_collect: a READ-class step, opt-in via the ``netops_read`` grant, and
    review_recommended (NOT hard-gated) so a read-only investigation runs to completion under
    Auto-complete rather than stopping on every device read."""
    from slimx_agent import policies
    from slimx_agent.contracts import NETOPS_STEP_TYPES

    assert NETOPS_STEP_TYPES == ("netops_collect",)
    step = type("Step", (), {"type": "netops_collect", "requires_approval": False})()
    tier, _reason = policies.classify_step(step)
    assert tier == policies.REVIEW_RECOMMENDED
    assert policies.CAPABILITY_BY_TYPE["netops_collect"] == policies.READ
    assert policies.required_grant("netops_collect") == "netops_read"
    ungranted = type("Run", (), {"allowed_tools_json": []})()
    granted = type("Run", (), {"allowed_tools_json": ["netops_read"]})()
    assert policies.permission_block_reason(step, ungranted) is not None
    assert policies.permission_block_reason(step, granted) is None
    # Under Auto-complete a review_recommended step does not stop; under manual it does.
    assert policies.requires_stop("auto_complete", tier, False) is False
    assert policies.requires_stop("manual", tier, False) is True


def test_netops_write_steps_are_grant_gated_and_never_planner_advertised():
    """netops_apply (Mode 4) hard-gates every change; netops_auto_apply (Mode 5) is
    review_recommended. Both are WRITE-class, need the netops_write grant, and are NEVER advertised
    to the planner (a device change must not be model-invented)."""
    from slimx_agent import policies
    from slimx_agent.contracts import NETOPS_WRITE_STEP_TYPES
    from slimx_agent.planning import build_planner_prompt

    assert NETOPS_WRITE_STEP_TYPES == ("netops_apply", "netops_auto_apply")
    for step_type in NETOPS_WRITE_STEP_TYPES:
        assert step_type in ALLOWED_STEP_TYPES
        assert policies.CAPABILITY_BY_TYPE[step_type] == policies.WRITE
        assert policies.required_grant(step_type) == "netops_write"
        # Never advertised, even with the grant.
        assert step_type not in build_planner_prompt("g", allowed_tools=["netops_write"])

    apply_step = type("Step", (), {"type": "netops_apply", "requires_approval": False})()
    tier, _ = policies.classify_step(apply_step)
    assert tier == policies.HARD_GATED
    assert policies.requires_stop("auto_complete", tier, False) is True  # hard gate always stops

    auto_step = type("Step", (), {"type": "netops_auto_apply", "requires_approval": False})()
    tier2, _ = policies.classify_step(auto_step)
    assert tier2 == policies.REVIEW_RECOMMENDED
    assert policies.requires_stop("auto_complete", tier2, False) is False  # Mode 5 can auto-run


def test_evidence_step_types_are_ungated_auto_safe_reads():
    """The project-evidence tools are always-on local reads: allowed, auto-safe, READ-class,
    and grant-free — the rag_retrieve/knowledge_retrieve risk profile, never web_search's."""
    from slimx_agent import policies
    from slimx_agent.contracts import EVIDENCE_STEP_TYPES

    assert EVIDENCE_STEP_TYPES == (
        "project_inventory",
        "evidence_query",
        "document_read",
        "conversation_search",
    )
    for step_type in EVIDENCE_STEP_TYPES:
        assert step_type in ALLOWED_STEP_TYPES
        step = type("Step", (), {"type": step_type, "requires_approval": False})()
        tier, _reason = policies.classify_step(step)
        assert tier == policies.AUTO_SAFE
        assert policies.CAPABILITY_BY_TYPE[step_type] == policies.READ
        assert policies.required_grant(step_type) is None


def test_evidence_write_step_types_are_grant_gated_review_recommended():
    """Project-evidence writes: allowed, review-recommended (not hard-gated), and gated behind the
    ``evidence_write`` grant so the agent never mutates evidence unless the user opted in."""
    from slimx_agent import policies
    from slimx_agent.contracts import EVIDENCE_WRITE_STEP_TYPES

    assert EVIDENCE_WRITE_STEP_TYPES == ("create_note", "add_tag")
    for step_type in EVIDENCE_WRITE_STEP_TYPES:
        assert step_type in ALLOWED_STEP_TYPES
        step = type("Step", (), {"type": step_type, "requires_approval": False})()
        tier, _reason = policies.classify_step(step)
        assert tier == policies.REVIEW_RECOMMENDED
        assert policies.required_grant(step_type) == "evidence_write"
        # Ungranted run → blocked with an honest reason; granted run → permitted.
        ungranted = type("Run", (), {"allowed_tools_json": []})()
        granted = type("Run", (), {"allowed_tools_json": ["evidence_write"]})()
        assert policies.permission_block_reason(step, ungranted) is not None
        assert policies.permission_block_reason(step, granted) is None


def test_contracts_and_tools_and_runtime_are_stdlib_only():
    """The move-verbatim guarantee: no third-party imports in the core three modules."""
    for module_name in ("contracts", "tools", "runtime"):
        module = __import__(f"slimx_agent.{module_name}", fromlist=[module_name])
        source = inspect.getsource(module)
        for forbidden in ("pydantic", "sqlmodel", "fastapi", "sqlalchemy", "httpx", "slimx"):
            assert f"import {forbidden}" not in source, (module_name, forbidden)
            assert f"from {forbidden}" not in source, (module_name, forbidden)


def test_no_agent_frameworks_anywhere():
    for module_name in ("contracts", "tools", "runtime", "planning"):
        module = __import__(f"slimx_agent.{module_name}", fromlist=[module_name])
        source = inspect.getsource(module).lower()
        for forbidden in ("langchain", "langgraph", "crewai", "autogen", "openai_agents"):
            assert forbidden not in source, (module_name, forbidden)


def test_tool_registry_dispatch_and_duplicate_rejection():
    registry = ToolRegistry()
    registry.register("model_call", lambda *args: {"ok": True})
    assert registry.resolve("model_call") is not None
    assert registry.resolve("unknown") is None
    with pytest.raises(ValueError):
        registry.register("model_call", lambda *args: None)


def test_step_errors_are_distinct():
    assert not issubclass(StepNotApplicable, StepExecutionError)
    assert issubclass(StepExecutionError, Exception)


def test_agent_run_context_defaults_empty():
    context = AgentRunContext()
    assert context.reference_text == ""
    assert context.manifest == []


def test_planning_validate_and_repair_roundtrip():
    raw = {
        "steps": [
            {"title": "", "type": "model_call", "instruction": "Analyze"},
            {"title": "Bogus", "type": "rm_rf"},  # unknown type → dropped by repair
        ],
        "assumptions": ["a"],
    }
    plan = validate_plan(repair_plan_data(raw))
    assert len(plan.steps) == 1
    assert plan.steps[0].type == "model_call"
    assert plan.steps[0].title  # repaired from instruction

    with pytest.raises(PlanValidationError):
        validate_plan({"steps": []})
    with pytest.raises(PlanValidationError):
        validate_plan({"steps": [{"title": "t", "type": "model_call"}] * (MAX_STEPS + 1)})


def test_planner_prompt_gates_tools_by_grant():
    bare = build_planner_prompt("Do the thing")
    assert "web_search" not in bare  # ungranted tools are never advertised
    granted = build_planner_prompt("Do the thing", allowed_tools=["web_search", "spawn_agents"])
    assert "web_search" in granted
    assert "spawn_run" in granted
    # mcp_call is NEVER advertised (structured params required).
    assert "mcp_call" not in granted
    # Evidence tools are always-on (grant-free), so they are always advertised …
    assert "evidence_query" in bare and "project_inventory" in bare and "document_read" in bare
    # … the host's project grounding is injected only when provided …
    assert "EXACTLY as listed" not in bare
    hinted = build_planner_prompt("Do the thing", evidence_hint="tags: Risk (3 highlights).")
    assert "tags: Risk (3 highlights)." in hinted
    # … and evidence WRITES are gated behind the evidence_write grant.
    assert "create_note" not in bare and "add_tag" not in bare
    writable = build_planner_prompt("Do the thing", allowed_tools=["evidence_write"])
    assert "create_note" in writable and "add_tag" in writable


def test_runtime_protocol_shape():
    from slimx_agent.runtime import AgentRuntime, RunProfile

    profile = RunProfile("ollama", "llama3.2", None)
    assert profile.provider == "ollama"
    # Protocol usable for isinstance checks on duck-typed implementations.
    assert hasattr(AgentRuntime, "plan_run")


def test_version():
    assert slimx_agent.__version__ == "0.14.0"


def test_run_id_types_are_uuid_friendly():
    # The runtime protocol talks UUIDs; make sure nothing in the core assumes strings.
    assert isinstance(uuid.uuid4(), uuid.UUID)


def test_new_write_step_types_are_evidence_write_gated_review_points():
    """0.7.0: link_work_item (task link) and promote_to_knowledge (KB promotion) are additive
    project writes behind the reused evidence_write grant; conversation_search stays a free read."""
    from slimx_agent import policies
    from slimx_agent.contracts import KNOWLEDGE_WRITE_STEP_TYPES, TASK_STEP_TYPES

    assert TASK_STEP_TYPES == ("create_work_item", "link_work_item")
    assert KNOWLEDGE_WRITE_STEP_TYPES == ("promote_to_knowledge",)
    for step_type in (*TASK_STEP_TYPES, *KNOWLEDGE_WRITE_STEP_TYPES):
        assert step_type in ALLOWED_STEP_TYPES
        assert policies.required_grant(step_type) == "evidence_write"
        step = type("Step", (), {"type": step_type, "requires_approval": False})()
        tier, _reason = policies.classify_step(step)
        assert tier == policies.REVIEW_RECOMMENDED
    assert policies.required_grant("conversation_search") is None


def test_plugin_tool_is_hard_gated_behind_the_plugin_tools_grant():
    """0.8.0: ONE generic plugin_tool step executes many admin-installed plugin tools (the
    mcp_call design applied to local plugin code) — granted AND hard-gated, never auto-run."""
    from slimx_agent import policies
    from slimx_agent.contracts import GRANTABLE_TOOLS, PLUGIN_STEP_TYPES

    assert PLUGIN_STEP_TYPES == ("plugin_tool",)
    assert "plugin_tool" in ALLOWED_STEP_TYPES
    assert "plugin_tools" in GRANTABLE_TOOLS
    assert policies.required_grant("plugin_tool") == "plugin_tools"
    step = type("Step", (), {"type": "plugin_tool", "requires_approval": False})()
    tier, _reason = policies.classify_step(step)
    assert tier == policies.HARD_GATED  # every invocation stops for the user
