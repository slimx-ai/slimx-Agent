"""SlimX-Agent contracts: the stable constants shared by every agent surface.

The single source of truth for step types, run modes, approval policies, tool grants, and the
durable event vocabulary. Historically these lived in ``app.models.agent_run`` and
``app.services.agent.events``; those modules now re-export from here so every existing import
path keeps working (Stage B of ``docs/slimx-agent-extraction-plan.md``).

**Dependency rule:** this module imports nothing beyond the standard library — no ORM, no
FastAPI, no other ``app.*`` modules — so it can move verbatim into the standalone
``slimx-agent`` package. A guard test enforces this.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- step types
# Step-type allowlist. The executor dispatches ONLY these types; the planner rejects anything
# else at validation time (structural safety, not prompt safety). The first six are the
# always-on assisted-workflow set. The build types are additive but GATED: only usable when the
# run is mode="build_agent" AND ENABLE_AGENT_BUILD_MODE is on, and they only ever touch the
# per-run sandbox workspace — never arbitrary host paths, shells, or generic tool calls.
ASSISTED_STEP_TYPES: tuple[str, ...] = (
    "model_call",
    "compare_models",
    "rag_retrieve",
    "knowledge_retrieve",
    "attach_context",
    "create_synthesis",
    "save_evidence",
)
# External tools reach the network via the one bounded MCP boundary (services/mcp_runtime).
# They are GATED twice over: usable only when the run's ``allowed_tools_json`` grants them
# (else the executor skips the step), and classified ``hard_gated`` for approval so each
# external call still stops for the user even in Auto-complete. ``web_search`` was the first;
# ``mcp_call`` is the generic (write-capable) connector invocation — TRIPLE-gated: run grant
# (``mcp_tools``) + per-call hard-gate approval + the connector's own ``allowedTools``
# allowlist (connectors permit nothing by default). Not advertised to the planner (structured
# params required); it enters plans via templates or the API.
EXTERNAL_STEP_TYPES: tuple[str, ...] = ("web_search", "mcp_call")
# Read-only codebase tools, confined to ``AGENT_CODE_SCAN_ROOT`` (never executes code, never
# edits; code EDIT/RUN are deliberately NOT provided — they would be arbitrary code execution,
# which the product forbids). Gated by the ``code_read`` grant; self-skip without a scan root.
CODE_STEP_TYPES: tuple[str, ...] = ("code_search", "code_read")
# Project-evidence tools: always-on, read-only views over the ACTIVE PROJECT's own material
# (documents, tags, highlights, comments). Local queries only — no egress, and no write beyond
# the visible context source each saves — so they need no grant, like rag_retrieve.
# ``project_inventory`` maps the project (documents/tags/evidence counts); ``evidence_query``
# retrieves highlights/comments by tag/kind/keyword as a cited evidence pack; ``document_read``
# reads one project document's extracted text (bounded). All three self-skip honestly when the
# run has no project scope or the project has no matching material.
EVIDENCE_STEP_TYPES: tuple[str, ...] = (
    "project_inventory",
    "evidence_query",
    "document_read",
    # ``conversation_search`` searches the active project's conversations by keyword and saves a
    # bounded, cited excerpt pack as a context source — same read discipline as evidence_query
    # (project-scoped, local-only, no grant, honest skip when nothing matches).
    "conversation_search",
)
# Project-evidence WRITES: additive, reversible, in-scope-to-the-user's-own-project mutations —
# ``create_note`` saves a comment (Note) anchored in the active project; ``add_tag`` attaches a
# label to an existing project highlight/comment. GATED by the ``evidence_write`` grant (off by
# default, like every write path) so the agent never mutates a user's evidence unless they opt in,
# and classified ``review_recommended`` so they still stop for review under manual/review policies.
# Never destructive: no deletes, no cross-project writes, no external egress.
EVIDENCE_WRITE_STEP_TYPES: tuple[str, ...] = ("create_note", "add_tag")
# Task WRITES: additive, reversible, in-project task mutations — ``create_work_item`` turns a run
# finding into a durable Work Item (task); ``link_work_item`` attaches an existing task to a
# document/conversation so the task carries its evidence trail. GATED by the ``evidence_write``
# grant (reused: it is the same "save additive project state" opt-in as create_note/add_tag) and
# classified ``review_recommended``. Never destructive: no deletes, no cross-workspace/project
# writes, no egress.
TASK_STEP_TYPES: tuple[str, ...] = ("create_work_item", "link_work_item")
# Knowledge WRITES: ``promote_to_knowledge`` promotes a synthesis this run produced into the
# project Knowledge Base (curated trust; optionally as a decision). Additive and reversible in the
# knowledge UI, GATED by the same ``evidence_write`` opt-in, classified ``review_recommended``.
KNOWLEDGE_WRITE_STEP_TYPES: tuple[str, ...] = ("promote_to_knowledge",)
BUILD_STEP_TYPES: tuple[str, ...] = (
    "write_file",
    "package_artifact",
)
# Code Builder patch loop (Stage 4c): a professional propose→apply→check→package cycle over the
# per-run sandbox, so "code this" produces a reviewable patch instead of a blind file dump.
# ``propose_patch`` generates a unified diff over named sandbox files (model text only — no
# mutation, so classified AUTO_SAFE and gated by ``code_read``); ``apply_patch_sandbox`` applies
# that diff inside the sandbox (WRITE, review_recommended); ``run_check`` runs one ALLOWLISTED
# check command (pytest/ruff/npm test …) in the sandbox (host-fenced behind its own flag; a
# non-allowlisted command is refused, never gated); ``package_patch`` bundles the diff as a
# ``code_patch`` artifact. The three mutating/executing steps are gated by the ``code_write`` grant.
CODE_BUILD_STEP_TYPES: tuple[str, ...] = (
    "propose_patch",
    "apply_patch_sandbox",
    "run_check",
    "package_patch",
)
# Master-agent orchestration: ``spawn_run`` creates AND plans a child run (a "sub-agent") for
# one focused sub-goal; ``join_runs`` executes the spawned children and collects their outcomes
# as context for later steps. Gated by the ``spawn_agents`` grant and bounded by depth/width
# caps; children never inherit the grant, so a run cannot fan out recursively.
ORCHESTRATION_STEP_TYPES: tuple[str, ...] = ("spawn_run", "join_runs")
# NetOps vertical pack (SlimX-NetOps): ``netops_collect`` runs a structured bundle of READ-ONLY
# network telemetry reads (SSH show / SNMP / Prometheus / Alertmanager / logs) through the one
# bounded MCP boundary against the NetOps bridge connector, and saves the result as a visible
# context source later steps reason over. GATED by the ``netops_read`` grant (external egress to
# infra, opt-in per run) but classified ``review_recommended`` — NOT hard-gated — so a whole
# read-only investigation runs to completion under Auto-complete after one plan approval, rather
# than stopping on every device read. It never mutates a device; the bridge enforces read-only.
NETOPS_STEP_TYPES: tuple[str, ...] = ("netops_collect",)
# NetOps WRITE path (Stage 4/5) — device-changing, so OFF by default at every layer: gated by the
# ``netops_write`` grant, the bridge's own ``NETOPS_ENABLE_WRITE`` + writelist, and the connector's
# allowedTools. Never advertised to the planner (a change enters via a template/API with structured
# params). ``netops_apply`` is HARD-GATED (every change stops for explicit human approval — Mode 4);
# ``netops_auto_apply`` is ``review_recommended`` for bounded auto-remediation (Mode 5), which the
# host additionally fences behind a flag + a low-risk change-type allowlist. Every change is
# dry-run-planned, rollback-carrying, validated after, and recorded — never a blind mutation.
NETOPS_WRITE_STEP_TYPES: tuple[str, ...] = ("netops_apply", "netops_auto_apply")
# Trusted Tool Plugins (admin-installed, reviewed, code-bearing host plugins — the mcp_call
# design applied to local plugin code): ONE generic step type executes many plugin-declared
# tools; the step's params name the plugin + tool. GATED by the ``plugin_tools`` grant AND
# HARD-GATED (every invocation stops for explicit human approval, even in Auto-complete) because
# a plugin runs beyond-contract code the host cannot classify. The host additionally fences the
# whole path behind its own feature flag and loads plugins only from an explicit configured
# directory — never from the browser. Never advertised to the planner (structured params only).
PLUGIN_STEP_TYPES: tuple[str, ...] = ("plugin_tool",)
ALLOWED_STEP_TYPES: tuple[str, ...] = (
    ASSISTED_STEP_TYPES
    + EVIDENCE_STEP_TYPES
    + EVIDENCE_WRITE_STEP_TYPES
    + TASK_STEP_TYPES
    + KNOWLEDGE_WRITE_STEP_TYPES
    + EXTERNAL_STEP_TYPES
    + CODE_STEP_TYPES
    + BUILD_STEP_TYPES
    + CODE_BUILD_STEP_TYPES
    + ORCHESTRATION_STEP_TYPES
    + NETOPS_STEP_TYPES
    + NETOPS_WRITE_STEP_TYPES
    + PLUGIN_STEP_TYPES
)

# --------------------------------------------------------------------------- grants & modes
# Grant keys a run may list in ``allowed_tools_json`` to opt into an optional/external tool.
# Kept separate from step types so one grant can gate one-or-more step types (and so the UI
# has a stable vocabulary). Core assisted steps are always available and are NOT gated by a
# grant. Workspace knowledge / rag is always-on and intentionally not listed here.
GRANTABLE_TOOLS: tuple[str, ...] = (
    "web_search",
    "code_read",
    "code_write",
    "spawn_agents",
    "mcp_tools",
    "evidence_write",
    "netops_read",
    "netops_write",
    "plugin_tools",
)

# What kind of agent run this is — drives the planner prompt, allowed step types, and the UI.
AGENT_MODES: tuple[str, ...] = ("assisted_workflow", "build_agent", "research_agent")

# Deterministic, backend-enforced approval policy (supersedes the planner-set per-step
# ``requires_approval`` flag). ``manual`` gates every planner-flagged/reviewable step;
# ``review_checkpoints`` runs auto-safe steps and stops only at meaningful review points;
# ``auto_complete`` runs to completion and stops ONLY at true safety checkpoints (hard gates).
# ``None`` on a run means "legacy behavior": honor the old ``auto_approve`` boolean + planner
# gate, so pre-existing runs are byte-for-byte unchanged.
APPROVAL_POLICIES: tuple[str, ...] = ("manual", "review_checkpoints", "auto_complete")

# --------------------------------------------------------------------------- event types
# Canonical durable event types, kept in one place so routes, executor, UI reducers, and tests
# agree. Convention: payloads carry only small references (ids, paths, counts) — never model
# output, retrieved chunks, traces, or evidence text. Events are append-only with a unique
# per-run ``sequence`` so a run can be replayed from scratch and a live stream resumed from a
# known cursor.
RUN_CREATED = "agent.run.created"
PLAN_CREATED = "agent.plan.created"
PLAN_APPROVED = "agent.plan.approved"
STEP_CREATED = "agent.step.created"
STEP_STARTED = "agent.step.started"
STEP_COMPLETED = "agent.step.completed"
STEP_FAILED = "agent.step.failed"
STEP_SKIPPED = "agent.step.skipped"
APPROVAL_REQUIRED = "agent.approval.required"
APPROVAL_GRANTED = "agent.approval.granted"
ARTIFACT_CREATED = "agent.artifact.created"
EVIDENCE_LINKED = "agent.evidence.linked"
RUN_COMPLETED = "agent.run.completed"
RUN_FAILED = "agent.run.failed"
RUN_PAUSED = "agent.run.paused"
RUN_CANCELLED = "agent.run.cancelled"
RUN_RESUMED = "agent.run.resumed"
# System-map inference (additive): a model-declared element of the system the goal implies
# (component/file/api/entity/test/risk/decision …). The payload carries kind/label/layer/
# rationale and is always source="agent_declared" — an inference from the model's narration,
# never tool- or test-confirmed. One event per element; a summary MAP_EXTRACTED follows.
SYSTEM_ELEMENT = "agent.system.element"
MAP_EXTRACTED = "agent.map.extracted"
# Build Agent (additive): a file was written into the per-run sandbox workspace.
# Reference-only (path + size); the bytes live in the sandbox, never in the event.
FILE_WRITTEN = "agent.file.written"
# Master agent (additive): a sub-agent run was created+planned by a spawn_run step / the
# spawned children were executed by a join_runs step. Payloads carry child run ids and
# statuses only — the children's own event trails hold their detail.
RUN_SPAWNED = "agent.run.spawned"
RUN_JOINED = "agent.run.joined"

EVENT_TYPES: tuple[str, ...] = (
    RUN_CREATED,
    PLAN_CREATED,
    PLAN_APPROVED,
    STEP_CREATED,
    STEP_STARTED,
    STEP_COMPLETED,
    STEP_FAILED,
    STEP_SKIPPED,
    APPROVAL_REQUIRED,
    APPROVAL_GRANTED,
    ARTIFACT_CREATED,
    EVIDENCE_LINKED,
    RUN_COMPLETED,
    RUN_FAILED,
    RUN_PAUSED,
    RUN_CANCELLED,
    RUN_RESUMED,
    SYSTEM_ELEMENT,
    MAP_EXTRACTED,
    FILE_WRITTEN,
    RUN_SPAWNED,
    RUN_JOINED,
)
