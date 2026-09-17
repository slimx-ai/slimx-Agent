# SlimX-Agent

The portable agent core of the SlimX platform:

- the step, grant, policy, and event **contracts**;
- the deterministic **permission and approval policies**;
- a typed **execution engine** that drives runs through a host-supplied store and tool registry;
- portable **planning** primitives;
- an optional **standalone service** that runs the same engine in its own container.

| Layer | Repository | Job |
| --- | --- | --- |
| Model execution | `slimx` | providers, payloads, retries, structured output, parallel fan-out |
| Knowledge / retrieval | `SlimX-RAG` | ingest, chunk, embed, index, retrieve, cite |
| Connector transport | `SlimX-MCP` | SSRF-guarded, capped MCP transport |
| **Agent core** | **`SlimX-Agent`** | **contracts, policies, typed engine, planning primitives, standalone service** |
| Reasoning workspace | `slimx-CR` (SlimX-AI ControlRoom) | UI, persistence, authorization, capabilities, host orchestration |

## Current adoption

SlimX-AI ControlRoom consumes this package directly, pinned to an exact source commit (its
compatibility matrix names the commit):

- `contracts`, `tools`, and `policies` — ControlRoom's modules of the same names are thin
  re-exports of these objects (identity, not copies).
- `engine` — ControlRoom's executor drives every in-process run through
  `engine.execute_run_events`, over its SQL `RunStore` and its governed `HOST_TOOLS` registry.
- `service` — ControlRoom's optional standalone container is built from this repository and
  runs the same engine against the host's `/internal/agent-host/*` callback API.

These stay with the host by design:

- persistence and its SQL store;
- authorization and workspace scope;
- provider-profile resolution and egress policy;
- every tool handler;
- approval receipts and the invocation ledger;
- the actor-aware root-launch service;
- the active planner.

**Planning ownership.** ControlRoom's active planner is host-owned: its schema, prompt,
readiness-aware shortlisting, repair, and 30-step validation ceiling. `slimx_agent.planning` is
the portable planning API, with a 12-step default ceiling. Unifying the two needs a shared,
versioned plan-limit contract, which is tracked in the host's roadmap. Do not edit either
ceiling to match the other.

## One engine, two placements

- **In-process:** the host calls `engine.execute_run_events(store, registry, run, profile=...)`
  with its own store, registry, and profile type.
- **Standalone:** `slimx_agent.service` runs the loop in its own container. Every store
  operation and tool invocation is an authenticated, lease-fenced callback to the host. The
  container holds no database, credentials, or host code. The wire contract is
  [`docs/service-contract.md`](docs/service-contract.md).

## Guarantees and their limits

- **Gate order.** Permission, then budget, then approval. Grants never imply approval, and
  approval never bypasses a missing grant: an already-approved step is re-checked against the
  run's current grants.
  - The permission gate precedes the budget gate because an honest skip does no work. An
    exhausted budget pauses the run before the next step that would work, never before one that
    would only be skipped.
  - A step that an interrupted drive left `running` is re-checked too. If its grant is gone, the
    engine raises `RunningStepNotPermitted` and writes nothing for the step. It does not skip it,
    because the earlier attempt may already have run. The standalone service answers 409.
  - Whether a still-permitted `running` step may be re-entered stays the store's decision, at
    the `running` transition.
- **Risk tiers × policies.** The table lives in `policies.py` and is written out as finite
  tests in `tests/test_policy_matrix.py`.
  - `manual` and `review_checkpoints` currently share one predicate.
  - Legacy runs with `approval_policy` null consult only the planner flag; hosts must hard-gate
    them at their own dispatch boundary.
  - A strictly distinct approve-every-step policy would be a new, versioned contract.
- **Pre-approval.** Only `web_search` and `web_fetch` can be pre-approved, only under
  `auto_complete`, and only from a real list; a junk stored value pre-approves nothing.
- **Outcomes.**
  - `completed`, `skipped`, and `failed` are the only terminal step states the engine writes.
  - What a handler raises decides which. This is the whole contract, and
    `tests/test_engine.py` pins it on a store that accepts every write:

    | A handler raises | The engine | Meaning |
    | --- | --- | --- |
    | `StepExecutionError` | writes `failed`, fails the run, calls the hook | a genuine failure |
    | `StepNotApplicable` | writes `skipped`, continues | required inputs were absent |
    | `StepActionPrepared` | re-reads the step, re-enters the gates | the host prepared a new action generation |
    | `StepOutcomeUnknown` | writes no terminal state or event, ends the drive, re-raises | the outcome was not observed |
    | any other `Exception` | rolls back, writes `failed`, fails the run | a bug in the handler |
    | a `BaseException` that is not an `Exception` | writes nothing further, propagates | a host control signal |

  - A host that means "I cannot tell whether this ran" (an admission ledger conflict, a lost
    lease, an entry already in progress) must raise `StepOutcomeUnknown` or a subclass of it.
    An ordinary exception is projected as a failure unless the host's store refuses the write,
    and a guarantee that depends on the store refusing is not one the engine makes.
  - A prepared action generation re-enters the gates, and an earlier approval never carries
    over.
  - An unknown outcome (`StepOutcomeUnknown`) ends the drive with no terminal write and no
    retry; the host's durable invocation record decides what happened.
- **Run completion.** A drive writes a terminal run status exactly when it appends that
  status's terminal event (`agent.run.completed` or `agent.run.failed`) and calls `on_run_end`
  once. Pause, cancel, an approval stop, a budget pause and an unknown outcome write none of the
  three.
  - A step failure is reported once per run, across drives. When a drive finds its first
    unfinished step already `failed`, it fails the run and reports the failure only if the log
    holds no `agent.run.failed` for that step; a host that re-opens a failed run gets no second
    event and no second hook call.
  - The match is by step id. A host that resets a failed step for retry and later fails it again
    outside the engine must append its own terminal event.
- **"Completed" means the loop finished.** A `completed` run status means every step reached
  `completed` or `skipped` without a stop. It is not a verified deliverable. Criterion-level
  acceptance, checked artifacts, and exact-snapshot executable checks are host
  responsibilities.
- **The check runner is not an isolation boundary.** `/internal/run-check` is absent unless
  explicitly enabled. It runs a host-allowlisted command in a run's mutable workspace, bounded
  by timeout, output cap, and process-group kill. It does not contain the filesystem, network,
  CPU, or memory, and it cannot back an exact-snapshot check receipt: ControlRoom refuses
  service-mode checks for exactly that reason. An isolated exact-snapshot runner is a separate,
  future deliverable.

## Static contracts

The engine boundary is typed and checked with `mypy --strict`, and the package ships `py.typed`:

```python
from slimx_agent import engine
from slimx_agent.tools import ToolRegistry

registry: ToolRegistry[Session, AgentRun, AgentStep, ExecProfile] = ToolRegistry()
registry.register("model_call", handle_model_call)  # a wrong handler shape is a type error
engine.execute_run(SqlRunStore(session), registry, run, profile=profile)
```

- **Structural views.** `RunView`, `StepView`, and `ProfileView` describe exactly the fields
  the engine reads.
- **Generic boundaries.** `RunStore[RunT, StepT, ContextT]` and
  `ToolRegistry[ContextT, RunT, StepT, ProfileT]` tie one host's store, handlers, and run
  objects together.
- **Enforced by tests.** `tests/typing/` holds conformance fixtures that must pass, and
  deliberately wrong adapters that must fail with exact diagnostics.

Static types do not validate hostile input at runtime. The standalone wire boundary also
parses strictly at runtime.

## Install

There is no package-index release yet. Consume an exact source commit:

```bash
pip install "slimx-agent @ https://github.com/slimx-ai/SlimX-Agent/archive/<commit>.tar.gz"
pip install "slimx-agent[service] @ https://github.com/slimx-ai/SlimX-Agent/archive/<commit>.tar.gz"
```

The core install needs only pydantic. The `service` extra adds FastAPI, uvicorn, and httpx.
Release identity and the gated release procedure are in [`docs/release.md`](docs/release.md);
changes are in [`CHANGELOG.md`](CHANGELOG.md).

## Development

These commands are the CI gate, run on Python 3.12 and 3.13 at the exact PR head:

```bash
pip install -e '.[dev,service]'
python scripts/check_version.py        # one version: source, pyproject, changelog, metadata, health
ruff check . && ruff format --check .
mypy                                   # strict, package + scripts
pytest --cov --cov-report=json:coverage.json
python scripts/check_coverage.py coverage.json
python scripts/verify_distribution.py  # build sdist+wheel; install and import outside the tree
```

Tests are offline: no provider, database, or host service. Contributor rules for this repository
are in [`AGENTS.md`](AGENTS.md).

## License

MIT — see [LICENSE](LICENSE).
