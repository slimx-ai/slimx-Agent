"""The permission and approval gates as finite tables.

Every contract step type, policy, planner flag, and pre-approval state has one recorded,
deterministic answer, written out here independently of the implementation so an accidental
tier, grant, or predicate change fails loudly. Current semantics are recorded, not redesigned:
``manual`` and ``review_checkpoints`` share one predicate, and legacy ``approval_policy IS NULL``
runs consult only the planner flag.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from slimx_agent import engine, policies
from slimx_agent.contracts import ALLOWED_STEP_TYPES, APPROVAL_POLICIES, GRANTABLE_TOOLS

A, R, H = policies.AUTO_SAFE, policies.REVIEW_RECOMMENDED, policies.HARD_GATED

# The reviewed classification: step type -> (risk tier, required grant).
EXPECTED: dict[str, tuple[str, str | None]] = {
    "model_call": (A, None),
    "compare_models": (R, None),
    "rag_retrieve": (A, None),
    "knowledge_retrieve": (A, None),
    "attach_context": (A, None),
    "create_synthesis": (A, None),
    "compose_report": (A, None),
    "save_evidence": (A, None),
    "project_inventory": (A, None),
    "evidence_query": (A, None),
    "document_read": (A, None),
    "conversation_search": (A, None),
    "create_note": (R, "evidence_write"),
    "add_tag": (R, "evidence_write"),
    "create_work_item": (R, "evidence_write"),
    "link_work_item": (R, "evidence_write"),
    "work_items_read": (A, None),
    "promote_to_knowledge": (R, "evidence_write"),
    "web_search": (H, "web_search"),
    "web_fetch": (H, "web_search"),
    "mcp_call": (H, "mcp_tools"),
    "code_search": (A, "code_read"),
    "code_read": (A, "code_read"),
    "write_file": (R, None),
    "package_artifact": (R, None),
    "propose_patch": (A, "code_read"),
    "apply_patch_sandbox": (R, "code_write"),
    "run_check": (R, "code_write"),
    "package_patch": (A, "code_write"),
    "stage_files": (A, "code_read"),
    "review_patch": (A, "code_read"),
    "spawn_run": (A, "spawn_agents"),
    "join_runs": (R, "spawn_agents"),
    "netops_collect": (R, "netops_read"),
    "netops_apply": (H, "netops_write"),
    "netops_auto_apply": (R, "netops_write"),
    "plugin_tool": (H, "plugin_tools"),
    "research_iterate": (A, None),
    "data_catalog": (R, "data_read"),
    "data_query": (R, "data_read"),
    "analyze_data": (A, "data_read"),
}

# (tier, planner flag, pre-approved) -> stops?  Written out, not derived.
_CHECKPOINT_TABLE: dict[tuple[str, bool, bool], bool] = {
    (A, False, False): False,
    (A, True, False): True,
    (A, False, True): False,
    (A, True, True): True,
    (R, False, False): True,
    (R, True, False): True,
    (R, False, True): True,
    (R, True, True): True,
    (H, False, False): True,
    (H, True, False): True,
    (H, False, True): True,  # pre-approval never skips review under a review policy
    (H, True, True): True,
}
STOP_TABLE: dict[str, dict[tuple[str, bool, bool], bool]] = {
    "auto_complete": {
        (A, False, False): False,
        (A, True, False): False,  # the planner flag does not stop auto_complete
        (A, False, True): False,
        (A, True, True): False,
        (R, False, False): False,
        (R, True, False): False,
        (R, False, True): False,
        (R, True, True): False,
        (H, False, False): True,
        (H, True, False): True,
        (H, False, True): False,  # the one gate pre-approval can lower
        (H, True, True): False,
    },
    "review_checkpoints": _CHECKPOINT_TABLE,
    # Recorded known divergence: manual is currently the same predicate as review_checkpoints.
    "manual": _CHECKPOINT_TABLE,
}


def _step(step_type: str, requires_approval: bool = False) -> SimpleNamespace:
    return SimpleNamespace(type=step_type, requires_approval=requires_approval)


def _run(grants: object) -> SimpleNamespace:
    return SimpleNamespace(allowed_tools_json=grants)


def test_the_reviewed_table_covers_exactly_the_contract_vocabulary():
    assert len(ALLOWED_STEP_TYPES) == len(set(ALLOWED_STEP_TYPES)) == 41
    assert set(EXPECTED) == set(ALLOWED_STEP_TYPES)
    assert APPROVAL_POLICIES == ("manual", "review_checkpoints", "auto_complete")
    assert set(STOP_TABLE) == set(APPROVAL_POLICIES)


@pytest.mark.parametrize("step_type", ALLOWED_STEP_TYPES)
def test_every_step_type_has_its_reviewed_tier_grant_and_capability(step_type):
    tier, grant = EXPECTED[step_type]
    classified, reason = policies.classify_step(_step(step_type))
    assert classified == tier
    assert "Unrecognized" not in reason  # an explicit entry, never the unknown-type fallback
    assert policies.required_grant(step_type) == grant
    assert policies.CAPABILITY_BY_TYPE[step_type] in {
        policies.READ,
        policies.MODEL,
        policies.EXTERNAL,
        policies.WRITE,
        policies.PERSISTENT,
        policies.ORCHESTRATION,
    }


def test_policy_tables_have_no_stale_missing_or_unlabeled_entries():
    assert set(policies.CAPABILITY_BY_TYPE) == set(ALLOWED_STEP_TYPES)
    assert set(policies._TIER_BY_TYPE) == set(ALLOWED_STEP_TYPES)
    assert set(policies._GRANT_BY_TYPE) <= set(ALLOWED_STEP_TYPES)
    assert set(policies._GRANT_BY_TYPE.values()) == set(GRANTABLE_TOOLS)
    assert set(policies.GRANT_LABELS) == set(GRANTABLE_TOOLS)
    assert policies.PREAPPROVABLE_STEP_TYPES == frozenset({"web_search", "web_fetch"})
    for step_type in policies.PREAPPROVABLE_STEP_TYPES:
        assert EXPECTED[step_type][0] == H
        assert policies.CAPABILITY_BY_TYPE[step_type] == policies.EXTERNAL


@pytest.mark.parametrize("policy", APPROVAL_POLICIES)
def test_requires_stop_is_the_recorded_truth_table(policy):
    for (tier, flag, preapproved), expected in STOP_TABLE[policy].items():
        got = policies.requires_stop(policy, tier, flag, preapproved=preapproved)
        assert got is expected, (policy, tier, flag, preapproved)


@pytest.mark.parametrize(
    "policy", ["", "strict", "manual_review", "AUTO_COMPLETE", "auto-complete"]
)
def test_unknown_policy_strings_are_treated_like_manual(policy):
    for (tier, flag, preapproved), expected in STOP_TABLE["manual"].items():
        assert policies.requires_stop(policy, tier, flag, preapproved=preapproved) is expected


@pytest.mark.parametrize("step_type", ALLOWED_STEP_TYPES)
def test_every_step_type_resolves_through_the_table_under_every_policy(step_type):
    tier = EXPECTED[step_type][0]
    for policy, flag in itertools.product(APPROVAL_POLICIES, (False, True)):
        classification, _reason, stop = engine.resolve_gate(
            _step(step_type, flag), policy=policy, auto_approve=False
        )
        assert classification == tier
        assert stop is STOP_TABLE[policy][(tier, flag, False)], (policy, flag)


@pytest.mark.parametrize("step_type", ALLOWED_STEP_TYPES)
def test_legacy_null_policy_consults_only_the_planner_flag(step_type):
    """Recorded legacy behavior (``approval_policy IS NULL``): the portable engine ignores the
    risk tier — a hard-gated type without a planner flag does not stop. Hosts that must
    hard-gate legacy runs enforce it at their own dispatch boundary (ControlRoom requires an
    approval receipt there). This release does not reinterpret stored null policies."""
    for flag, auto_approve in itertools.product((False, True), repeat=2):
        classification, reason, stop = engine.resolve_gate(
            _step(step_type, flag), policy=None, auto_approve=auto_approve
        )
        assert (classification, reason) == (None, "")
        assert stop is (flag and not auto_approve)


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (None, set()),
        ([], set()),
        (["web_search"], {"web_search"}),
        (("web_fetch", "web_search"), {"web_fetch", "web_search"}),
        ({"web_search"}, {"web_search"}),
        (
            [
                "mcp_call",
                "plugin_tool",
                "netops_apply",
                "netops_auto_apply",
                "run_check",
                "write_file",
                "web_search",
            ],
            {"web_search"},
        ),
        ("web_search", set()),  # a junk string is not a list: no substring matching
        ("prefix web_fetch suffix", set()),
        ({"web_search": True}, set()),  # a mapping is not a list
        ([["web_search"], 1, None, b"web_fetch"], set()),
        (["WEB_SEARCH", " web_search"], set()),
    ],
)
def test_stored_preapproval_can_only_name_allowlisted_read_only_types(stored, expected):
    assert policies.normalize_preapproved(stored) == frozenset(expected)


@pytest.mark.parametrize(
    "step_type", sorted(set(ALLOWED_STEP_TYPES) - policies.PREAPPROVABLE_STEP_TYPES)
)
def test_no_other_step_type_can_lower_its_gate_through_stored_preapproval(step_type):
    everything = [*ALLOWED_STEP_TYPES, "web_search web_fetch"]
    for policy, flag in itertools.product(APPROVAL_POLICIES, (False, True)):
        step = _step(step_type, flag)
        junk = engine.resolve_gate(
            step, policy=policy, auto_approve=True, preapproved_tools=everything
        )
        clean = engine.resolve_gate(step, policy=policy, auto_approve=True, preapproved_tools=None)
        assert junk == clean, (policy, flag)


@pytest.mark.parametrize("step_type", sorted(policies.PREAPPROVABLE_STEP_TYPES))
def test_preapproval_lowers_only_the_auto_complete_hard_gate(step_type):
    step = _step(step_type)
    stops = {
        policy: engine.resolve_gate(
            step, policy=policy, auto_approve=False, preapproved_tools=[step_type]
        )[2]
        for policy in APPROVAL_POLICIES
    }
    assert stops == {"auto_complete": False, "review_checkpoints": True, "manual": True}
    # Without it — or with a junk non-list value — the hard gate stops everywhere.
    for stored in (None, [], step_type, {step_type: True}):
        for policy in APPROVAL_POLICIES:
            assert engine.resolve_gate(
                step, policy=policy, auto_approve=True, preapproved_tools=stored
            )[2]


@pytest.mark.parametrize(
    ("stored", "granted"),
    [
        (None, set()),
        ([], set()),
        (["web_search"], {"web_search"}),
        (("code_read",), {"code_read"}),
        ("web_search", set()),  # a junk string grants nothing
        ({"web_search": True}, set()),  # a mapping grants nothing (it used to grant its keys)
        (["web_search", 7, None, ["mcp_tools"]], {"web_search"}),
    ],
)
def test_stored_grants_are_read_fail_closed(stored, granted):
    assert policies.granted_tools(_run(stored)) == granted


@pytest.mark.parametrize("step_type", ALLOWED_STEP_TYPES)
def test_permission_decision_for_every_step_type(step_type):
    grant = EXPECTED[step_type][1]
    if grant is None:
        for stored in (None, [], "junk", {"x": 1}, list(GRANTABLE_TOOLS)):
            assert policies.permission_block_reason(_step(step_type), _run(stored)) is None
        return
    others = [g for g in GRANTABLE_TOOLS if g != grant]
    for stored in (None, [], others, grant, {grant: True}, ["future_grant"]):
        reason = policies.permission_block_reason(_step(step_type), _run(stored))
        assert reason is not None
        assert policies.GRANT_LABELS[grant] in reason
    assert policies.permission_block_reason(_step(step_type), _run([grant])) is None


def test_client_grant_lists_are_normalized_without_reinterpreting_legacy_none():
    assert policies.normalize_grants(None) is None
    assert policies.normalize_grants([]) == []
    assert policies.normalize_grants(
        ["web_search", " code_read ", "nope", "", "web_search", "mcp_tools"]
    ) == ["web_search", "code_read", "mcp_tools"]
    assert policies.normalize_grants("web_search") == []  # characters are not grant keys


def test_unknown_step_types_are_review_recommended_and_need_no_grant():
    """Recorded current behavior. Such a step cannot execute: no registry maps a type outside
    the contract, so the engine fails it honestly (tests/test_engine.py)."""
    tier, reason = policies.classify_step(_step("rm_rf"))
    assert tier == R
    assert "Unrecognized step type 'rm_rf'" in reason
    assert policies.required_grant("rm_rf") is None
    assert policies.permission_block_reason(_step("rm_rf"), _run(None)) is None
    assert policies.normalize_preapproved(["rm_rf"]) == frozenset()
