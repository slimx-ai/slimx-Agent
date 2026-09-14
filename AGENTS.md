# AGENTS.md — SlimX-Agent

Rules for coding agents and contributors working in this repository.

## Ownership

- This is a public, MIT-licensed, dependency-light package, and each tier has an allowlist:
  - The **core** modules are `contracts`, `tools`, `runtime`, `store`, `policies`, `engine`,
    `host_client`, `http_store`, and `http_tools`. They import only the standard library and
    this package at module level.
  - `planning` may add `pydantic`.
  - `service` may add the `service` extra.

  A guard test enforces these allowlists exactly.
- Host integration is owned by the consuming host, not here. For SlimX-AI ControlRoom, these
  own adoption, sequencing, the cross-repository register, and dependency pins:
  - its `AGENTS.md`;
  - its sole roadmap, `docs/roadmap.md`;
  - `docs/architecture/agent-runtime.md`;
  - `docs/compatibility-matrix.md`.

  Do not copy host plans or status into this repository.
- Never add an agent framework (LangChain, LangGraph, CrewAI, AutoGen, the OpenAI Agents SDK)
  as a dependency.

## Invariants

- **The step vocabulary is closed.** Every step type needs:
  - a risk tier;
  - a capability class;
  - a grant decision in `policies.py`;
  - a row in `tests/test_policy_matrix.py`;
  - a host handler, coordinated with the host.
- **Never weaken the gates or the boundary:**
  - permission-before-approval, including re-checking approved steps;
  - hard gates;
  - the read-only pre-approval allowlist;
  - no terminal step state without an authoritative outcome;
  - no automatic retry;
  - per-request lease fencing;
  - strict wire parsing.
- **Policy semantics are persisted.** Changing what a stored `approval_policy` means is a new,
  versioned contract with a migration plan, never a silent edit.
- **The portable planning limit stays independent.** `planning.MAX_STEPS` changes only through
  a shared, versioned limit contract.
- **`/internal/run-check` is not an isolation boundary.** Keep it off by default and token-only.
  Never describe it as exact-snapshot evidence.

## Workflow

- Change the version only in `slimx_agent/__init__.py`, together with a matching `CHANGELOG.md`
  heading. Never tag, publish, or move a release without explicit release authorization; see
  `docs/release.md`.
- Run the CI gate locally before proposing a change:

  ```bash
  pip install -e '.[dev,service]'
  python scripts/check_version.py
  ruff check . && ruff format --check .
  mypy
  pytest --cov --cov-report=json:coverage.json && python scripts/check_coverage.py coverage.json
  python scripts/verify_distribution.py
  ```

- Tests stay offline and deterministic: no provider, database, host service, credentials, or
  GPU.
