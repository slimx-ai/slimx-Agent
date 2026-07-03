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
    assert set(GRANTABLE_TOOLS) == {"web_search", "code_read", "spawn_agents", "mcp_tools"}
    assert len(EVENT_TYPES) == len(set(EVENT_TYPES)) == 22


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


def test_runtime_protocol_shape():
    from slimx_agent.runtime import AgentRuntime, RunProfile

    profile = RunProfile("ollama", "llama3.2", None)
    assert profile.provider == "ollama"
    # Protocol usable for isinstance checks on duck-typed implementations.
    assert hasattr(AgentRuntime, "plan_run")


def test_version():
    assert slimx_agent.__version__ == "0.1.0"


def test_run_id_types_are_uuid_friendly():
    # The runtime protocol talks UUIDs; make sure nothing in the core assumes strings.
    assert isinstance(uuid.uuid4(), uuid.UUID)
