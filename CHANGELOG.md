# Changelog

Notable changes to `slimx-agent`. Before 1.0, a minor release may change behavior; every such
change is listed here with a migration note. The newest release heading must equal
`slimx_agent.__version__` (`scripts/check_version.py` enforces it). A version is a source
identity only until a tag or published artifact exists; see [`docs/release.md`](docs/release.md).

## 0.21.0 — unreleased (not tagged or published)

Opens the development line after 0.20.0. ControlRoom consumes 0.20.0 by exact source commit
(`dc04d360`), so every later behavior change carries this new version instead of altering what
0.20.0 names. Behavior changes are listed here as each one merges; until then 0.21.0 behaves
exactly as 0.20.0.

### Behavior changes

- **A step failure nobody reported no longer ends a run silently.** When a drive finds its first
  unfinished step already `failed`, the engine still fails the run without dispatching anything.
  It now also appends `agent.run.failed` for that step and calls `on_run_end(run, "failed")`,
  unless the run's log already holds `agent.run.failed` for the same step. Previously this path
  wrote only the `failed` run status, so a step the host had failed itself ended the run with no
  terminal event, no hook and, over the standalone service, no `/run-end` callback. A re-drive
  of a run whose failure an earlier drive reported is unchanged: no second event, no second
  hook. The check reads the run's events once, from sequence 0, on this path only.
  *Migration:* a host that writes a `failed` step itself, leaves the run non-terminal and then
  drives it now receives the terminal event and the hook once. A host that already appends its
  own `agent.run.failed` for that step sees no change.

## 0.20.0 — unreleased (not tagged or published)

A hardening release built on the 0.18.0/0.19.0 standalone-fencing commits
(`6791590`, `61ae7af`, `fc6f4c5`), which it contains unchanged in history.

### Behavior changes (fail-closed)

- **Unknown step outcomes are no longer projected as failures.** A new engine outcome,
  `StepOutcomeUnknown`, ends the drive without writing a terminal step state or event and without
  retrying. The standalone remote handler raises it when the invocation callback fails
  (transport failure, host 4xx/5xx), or returns an unrecognized outcome or malformed envelope.
  Previously these became an ordinary `failed` step, which a host had to refuse when the tool
  might already have run. The service maps the drive end to the host's own 4xx, else 502.
- **Approval never bypasses a missing grant.** The permission gate now also re-checks an
  already `approved` step, so a grant revoked after approval produces an honest skip instead of
  a dispatch.
- **Unrecognized step statuses are refused.** The engine recognizes exactly `pending`,
  `awaiting_approval`, `approved`, `running`, `completed`, `failed`, and `skipped`
  (`contracts.STEP_STATUSES`). A step in any other status raises `engine.UnknownStepStatus`,
  and the drive ends with nothing written for it. Previously a status such as `queued` or
  `Pending` matched neither gate and was dispatched. The standalone store refuses such a
  snapshot as a `HostProtocolError`.
- **Stored grant and pre-approval values are read fail-closed.** Only a list/tuple/set of
  strings counts. A junk string no longer pre-approves by substring, and a mapping no longer
  grants its keys. Pre-approval remains limited to `web_search`/`web_fetch`.
- **Strict standalone wire parsing.** Run/step snapshots, event pages, `next_sequence`, and
  response bodies must have their documented JSON types; anything else is a `HostProtocolError`
  (a synthetic 502). Previously `bool("false")` could become an automatic approval and a
  malformed budget an unbounded one.
- **`/internal/run-check` is off by default.** It is registered only when
  `SLIMX_AGENT_ENABLE_RUN_CHECK` is set, and then refuses requests unless
  `SLIMX_AGENT_INTERNAL_TOKEN` is configured (there is no tokenless mode for command
  execution). Requests are validated strictly, and `timeout_seconds` must be a JSON number (422,
  with no process started); `run_id` must be
  a plain identifier whose resolved directory is a direct child of the workspace root. Output is
  bounded while reading, stdin is closed, and the whole process group is killed on every exit.
  The response gains `output_truncated`. This runner is still not an isolation boundary or an
  exact-snapshot runner.
- **The portable planner prompt advertises only executable types.** A type is offered only when
  its required grant is present, and `plugin_tool` is now never advertised (joining `mcp_call`,
  the netops writes, and `research_iterate`). The portable `MAX_STEPS` stays 12.
- **Execute endpoints validate their inputs.** Run ids must be plain identifiers; the provider,
  model, and base URL are bounded non-empty strings; `lease_generation` must be a JSON integer.
- **Hardening of the callback client.** It never routes callbacks through ambient proxy
  environment variables (`trust_env=False`), while a CA bundle configured through
  `SSL_CERT_FILE`/`SSL_CERT_DIR` is still honored exactly as before. It quotes host identities
  into single path segments, refuses empty and dot-only identities (which HTTP clients would
  normalize out of the path), and bounds error details to 500 characters.

### Added

- Typed engine boundary: `RunView`, `StepView`, `ProfileView`, and generic `RunStore`,
  `ToolHandler`, and `ToolRegistry` protocols (PEP 695), plus the `UNSET`/`UnsetType` sentinel
  and the `OutputRefs`/`EventPayload`/`HostId` aliases. The package ships `py.typed`.
- `SlimXAgentPlanStep.params` and `.input_refs` (optional, default `None`), so structured
  output keeps executable step data. `planning.advertised_step_types()` and
  `NEVER_ADVERTISED_STEP_TYPES`.
- `HostUnavailable` and `HostProtocolError` (both `HostError`); `policies.normalize_preapproved`.
- `contracts.STEP_STATUSES` and `engine.UnknownStepStatus`.
- `CAPABILITY_BY_TYPE` entries for `knowledge_retrieve` and `compose_report`, and a
  `plugin_tools` grant label.
- CI: a Python 3.12/3.13 matrix at the exact PR head, a version-agreement check, ruff lint and
  format, strict mypy, static-conformance fixtures, branch coverage with per-module floors, and
  build/install verification of the wheel outside the source tree.

### Fixed

- A non-ASCII `Authorization` header returned 500 instead of 401.
- An unset `SLIMX_AGENT_HOST_URL` returned 500 instead of 503.
- A disconnected SSE observer, or a closed event loop, could crash the drive thread. The drive
  now continues to its next durable stop.
- Documentation described core host adoption as future work, and the Dockerfile omitted the
  license file from the installed service image.

### Migration notes for hosts

- mypy users must parameterize registries they build, e.g.
  `ToolRegistry[Session, AgentRun, AgentStep, ExecProfile]`: an unannotated `ToolRegistry()`
  is reported as needing an annotation.
- A host that relied on the standalone engine writing `failed` after an invocation callback
  error now receives no terminal write. Its invocation ledger (or reclaim path) owns that step.
- `AgentRuntime.instantiate_template` is a documented compatibility exception (an actorless
  root launcher). Hosts may omit it, and it is expected to leave this protocol in a future minor.

## 0.19.0 — source `fc6f4c5` (never tagged)

- The standalone remote handler maps a host `prepared` outcome to `StepActionPrepared`, and the
  engine stops cleanly at a prepared action generation instead of projecting a terminal state.
- Consumed by ControlRoom by exact source archive; the manifest and `__version__` both read
  `0.19.0` at this commit.

## 0.18.0 — source `6791590` (never tagged)

- Standalone callbacks carry the execution-attempt lease (`X-SlimX-Agent-Lease-*`) on every
  request, per-execution and never as shared client defaults. The execute endpoints require a
  complete lease, and host 4xx refusals are preserved at the service edge.

## 0.17.0 and earlier

See the Git history of `main` through `bf227a6`. At that commit the manifest read `0.17.0` while
the runtime `__version__` still read `0.16.0`. 0.20.0 removes that class of drift by deriving the
manifest version from `__version__`.
