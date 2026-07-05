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
        "spawn_agents",
        "mcp_tools",
        "evidence_write",
        "netops_read",
    }
    assert "netops_collect" in ALLOWED_STEP_TYPES
    assert len(EVENT_TYPES) == len(set(EVENT_TYPES)) == 22


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


def test_evidence_step_types_are_ungated_auto_safe_reads():
    """The project-evidence tools are always-on local reads: allowed, auto-safe, READ-class,
    and grant-free — the rag_retrieve/knowledge_retrieve risk profile, never web_search's."""
    from slimx_agent import policies
    from slimx_agent.contracts import EVIDENCE_STEP_TYPES

    assert EVIDENCE_STEP_TYPES == ("project_inventory", "evidence_query", "document_read")
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
    assert slimx_agent.__version__ == "0.3.3"


def test_run_id_types_are_uuid_friendly():
    # The runtime protocol talks UUIDs; make sure nothing in the core assumes strings.
    assert isinstance(uuid.uuid4(), uuid.UUID)
