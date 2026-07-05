"""Deterministic run policies — the portable halves of the agent's two orthogonal gates.

* **Approval policy** (`classify_step` / `requires_stop`): whether a *permitted* step stops
  for human review. Type-based and deterministic; a ``hard_gated`` step stops even in
  Auto-complete.
* **Tool policy** (`required_grant` / `permission_block_reason` / `normalize_grants`):
  whether a step is *permitted at all* for a run, based on its per-run grant list.

Moved verbatim from ControlRoom's ``services/agent/{approval_policy,tool_policy}.py``
(Stage I); the host modules are re-export shims. Steps/runs are duck-typed (``.type``,
``.requires_approval``, ``.allowed_tools_json``) so no ORM dependency exists here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from slimx_agent.contracts import GRANTABLE_TOOLS

# Risk tiers.
AUTO_SAFE = "auto_safe"
REVIEW_RECOMMENDED = "review_recommended"
HARD_GATED = "hard_gated"

# Base tier per step type. Everything on the current allowlist is additive/non-destructive, so the
# strongest tier used here is ``review_recommended`` (cost/fanout worth a checkpoint). Unknown types
# fall back to ``review_recommended`` (conservative — surface it rather than silently auto-run).
_TIER_BY_TYPE: dict[str, str] = {
    "model_call": AUTO_SAFE,
    "rag_retrieve": AUTO_SAFE,
    "knowledge_retrieve": AUTO_SAFE,
    "attach_context": AUTO_SAFE,
    "create_synthesis": AUTO_SAFE,
    # save_evidence is additive; the executor already self-skips when it has no quote/anchor, so
    # it never produces misleading output — safe to auto-run.
    "save_evidence": AUTO_SAFE,
    # compare_models fans out across several models (slower/costlier, possibly extra providers) —
    # a meaningful review point, though additive.
    "compare_models": REVIEW_RECOMMENDED,
    # web_search leaves the machine (external MCP egress). Hard-gated so each external send stops for
    # the user even in Auto-complete — the grant makes it *available*, this makes each use *confirmed*.
    "web_search": HARD_GATED,
    # mcp_call invokes an arbitrary (possibly write-capable) connector tool — every call is a
    # safety checkpoint, on top of the mcp_tools grant and the connector's allowedTools list.
    "mcp_call": HARD_GATED,
    # Read-only codebase tools: local, bounded, no execution, no egress — additive like rag_retrieve.
    # They only run when the run granted the ``code_read`` tool, so auto-running when present is safe.
    "code_search": AUTO_SAFE,
    "code_read": AUTO_SAFE,
    # Project-evidence reads: local, bounded queries over the active project's own documents/
    # tags/highlights/comments — additive like rag_retrieve/knowledge_retrieve.
    "project_inventory": AUTO_SAFE,
    "evidence_query": AUTO_SAFE,
    "document_read": AUTO_SAFE,
    # Project-evidence writes: additive/reversible in-project mutations behind the evidence_write
    # grant — a meaningful review point (they change what the user sees on their evidence board),
    # but never a hard gate (local, reversible, no egress).
    "create_note": REVIEW_RECOMMENDED,
    "add_tag": REVIEW_RECOMMENDED,
    # Build-agent steps touch a sandboxed per-run workspace; they self-skip outside build mode.
    "write_file": REVIEW_RECOMMENDED,
    "package_artifact": REVIEW_RECOMMENDED,
    # Code Builder patch loop: propose_patch is model text only (a diff) → auto_safe; applying the
    # diff and running an allowlisted check mutate/execute in the sandbox → review_recommended;
    # packaging the diff as an artifact is safe.
    "propose_patch": AUTO_SAFE,
    "apply_patch_sandbox": REVIEW_RECOMMENDED,
    "run_check": REVIEW_RECOMMENDED,
    "package_patch": AUTO_SAFE,
    # spawn_run only creates + plans a child (additive; nothing executes yet — every child plan
    # is inspectable before join). join_runs is the fan-out execution point (each child is its
    # own sequence of model calls) — a meaningful review checkpoint, like compare_models.
    "spawn_run": AUTO_SAFE,
    "join_runs": REVIEW_RECOMMENDED,
    # netops_collect reaches external infra (read-only telemetry), so it is opt-in via the
    # netops_read grant — but review_recommended, NOT hard_gated: a read-only investigation should
    # run to completion under Auto-complete after one plan approval, not stop on every device read.
    "netops_collect": REVIEW_RECOMMENDED,
    # netops_apply CHANGES a device — hard_gated so every change stops for explicit human approval
    # even in Auto-complete (Mode 4), on top of the netops_write grant + the bridge's own gates.
    "netops_apply": HARD_GATED,
    # netops_auto_apply is bounded auto-remediation (Mode 5): review_recommended so it can run under
    # Auto-complete, but the host fences it behind a flag + a low-risk change-type allowlist and
    # auto-rolls-back on failed validation — it is never reachable without that explicit opt-in.
    "netops_auto_apply": REVIEW_RECOMMENDED,
}

_REASON_BY_TIER: dict[str, str] = {
    AUTO_SAFE: "Additive, reversible step — runs automatically in Auto-complete.",
    REVIEW_RECOMMENDED: (
        "Runs several models (slower/costlier and may use extra providers) — a good place to review."
    ),
    HARD_GATED: "Safety checkpoint — always requires approval, even in Auto-complete.",
}

# A more specific reason than the generic tier text, where the step type warrants it.
_REASON_BY_TYPE: dict[str, str] = {
    "web_search": (
        "Sends your query to an external web-search service — always asks first, even in Auto-complete."
    ),
}


def classify_step(step: Any) -> tuple[str, str]:
    """Classify a step into a risk tier with a plain-language reason. Type-based and deterministic."""
    tier = _TIER_BY_TYPE.get(step.type, REVIEW_RECOMMENDED)
    reason = _REASON_BY_TYPE.get(step.type) or _REASON_BY_TIER[tier]
    if tier == REVIEW_RECOMMENDED and step.type not in _TIER_BY_TYPE:
        reason = f"Unrecognized step type {step.type!r} — review before it runs."
    return tier, reason


def requires_stop(policy: str, classification: str, requires_approval: bool) -> bool:
    """Whether execution must stop at this step for the given policy.

    * ``hard_gated`` always stops (safety checkpoint), regardless of policy.
    * ``auto_complete`` — stops only for hard gates.
    * ``review_checkpoints`` — stops for review-recommended (or planner-flagged) steps.
    * ``manual`` — stops for anything reviewable or planner-flagged (most conservative).
    """
    if classification == HARD_GATED:
        return True
    if policy == "auto_complete":
        return False
    if policy == "review_checkpoints":
        return classification == REVIEW_RECOMMENDED or requires_approval
    # manual (and any unknown/other policy string → treat conservatively like manual)
    return classification == REVIEW_RECOMMENDED or requires_approval


# Capability class per step type — the read / external / write / persistent distinction the product
# asks for. Informational today (it drives UI copy and future policy); the load-bearing decision is the
# grant check below. Model steps go through the (separately egress-gated) provider, so they are "model".
READ = "read"
MODEL = "model"
EXTERNAL = "external"
WRITE = "write"
PERSISTENT = "persistent"
ORCHESTRATION = "orchestration"

CAPABILITY_BY_TYPE: dict[str, str] = {
    "model_call": MODEL,
    "compare_models": MODEL,
    "create_synthesis": MODEL,
    "rag_retrieve": READ,
    "attach_context": READ,
    "save_evidence": PERSISTENT,
    "web_search": EXTERNAL,
    "mcp_call": EXTERNAL,
    "code_search": READ,
    "code_read": READ,
    "project_inventory": READ,
    "evidence_query": READ,
    "document_read": READ,
    "create_note": PERSISTENT,
    "add_tag": WRITE,
    "write_file": WRITE,
    "package_artifact": WRITE,
    # Code Builder: propose reads+generates (model); apply/check/package mutate or execute (write).
    "propose_patch": MODEL,
    "apply_patch_sandbox": WRITE,
    "run_check": WRITE,
    "package_patch": WRITE,
    # Master agent: spawn creates+plans child runs; join executes them (their own steps stay
    # individually gated by the children's grants/policies — never a bypass of either).
    "spawn_run": ORCHESTRATION,
    "join_runs": ORCHESTRATION,
    # NetOps telemetry collection is a bounded read (no device mutation).
    "netops_collect": READ,
    # NetOps changes are writes (bounded, reversible, dry-run-planned — but still writes).
    "netops_apply": WRITE,
    "netops_auto_apply": WRITE,
}

# The grant key a step type requires before it may run. Only gated tools appear here; a type not in the
# map needs no grant. One grant may gate several step types — the map is the single source. Both
# read-only code steps share the ``code_read`` grant.
_GRANT_BY_TYPE: dict[str, str] = {
    "web_search": "web_search",
    "mcp_call": "mcp_tools",
    "code_search": "code_read",
    "code_read": "code_read",
    # Code Builder: propose reads source (code_read); apply/check/package mutate the sandbox
    # (code_write). The Code Builder pack requires both grants.
    "propose_patch": "code_read",
    "apply_patch_sandbox": "code_write",
    "run_check": "code_write",
    "package_patch": "code_write",
    "spawn_run": "spawn_agents",
    "join_runs": "spawn_agents",
    "create_note": "evidence_write",
    "add_tag": "evidence_write",
    "netops_collect": "netops_read",
    "netops_apply": "netops_write",
    "netops_auto_apply": "netops_write",
}

# User-facing label per grant key, for honest UI/skip copy.
GRANT_LABELS: dict[str, str] = {
    "web_search": "Web search",
    "code_read": "Codebase (read-only)",
    "code_write": "Edit code (sandboxed patches)",
    "spawn_agents": "Sub-agents",
    "mcp_tools": "Connector tools (MCP)",
    "evidence_write": "Save notes & tags",
    "netops_read": "Network telemetry (read-only)",
    "netops_write": "Apply network changes",
}


def required_grant(step_type: str) -> str | None:
    """The grant key ``step_type`` needs to run, or ``None`` when it needs no explicit grant."""
    return _GRANT_BY_TYPE.get(step_type)


def granted_tools(run: Any) -> set[str]:
    """The set of tool grants on a run. ``None`` (legacy) grants nothing optional."""
    raw = run.allowed_tools_json or []
    return {str(item) for item in raw if isinstance(item, str)}


def permission_block_reason(step: Any, run: Any) -> str | None:
    """Why ``step`` is not permitted for ``run``, or ``None`` if it is.

    Returns ``None`` for any step needing no grant (all core assisted/build steps). For a gated step,
    returns a plain-language reason when the run did not grant the required tool — the executor turns
    that into an honest skip, never a hard failure and never a silent run.
    """
    grant = required_grant(step.type)
    if grant is None:
        return None
    if grant in granted_tools(run):
        return None
    label = GRANT_LABELS.get(grant, grant)
    return f"'{label}' was not enabled for this run — skipping (turn it on in the run's Tools to use it)."


def normalize_grants(raw: Iterable[str] | None) -> list[str] | None:
    """Validate a client-supplied grant list: drop unknown/blank keys, dedupe, keep a stable order.

    ``None`` passes through as ``None`` (legacy — no optional tools), which is distinct from ``[]``
    (explicitly no tools). Both behave identically at runtime; preserving the distinction keeps a
    programmatic/legacy caller byte-for-byte unchanged.
    """
    if raw is None:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        key = str(item).strip()
        if key and key in GRANTABLE_TOOLS and key not in seen:
            seen.add(key)
            out.append(key)
    return out
