# SlimX-Agent service contract (v1)

This is the HTTP surface of the standalone agent service, and the host callback API that service
drives. The service owns the engine loop only. It has no database or provider credentials, and it
drives the host's `/internal/agent-host/*` callback API for persistence and tool execution.
Version 0.20.0 tightens validation and failure semantics without changing any successful wire
shape.

## Auth

`SLIMX_AGENT_INTERNAL_TOKEN` is the shared bearer token. One value drives both directions.

- **Configured token.** The execute endpoints require `Authorization: Bearer <token>`. The
  comparison is constant-time and byte-wise, so a malformed header is a 401, never a 500.
- **No token.** Leaving it empty turns execute auth off. This is the documented local-first
  compatibility mode.
- **Run-check.** `/internal/run-check` has no tokenless mode.
- **Health.** `GET /health` is open and reports `auth_enabled`, so the host's deep health can
  flag one-sided configuration. It never returns the token.

The service's own callbacks carry the same bearer token, attached per request. They ignore
ambient proxy environment variables and honor a CA bundle configured through `SSL_CERT_FILE` or
`SSL_CERT_DIR`.

## Endpoints

| Endpoint | Body | Semantics |
| --- | --- | --- |
| `GET /health` | — | `{status: "ok", service: "slimx-agent", version, mode: "standalone", auth_enabled}`; `version` equals the installed distribution version |
| `POST /agent/runs/{id}/execute` | execute body (below) | Drive an **already-claimed** run to its next stop; `200 {run_id, status}` |
| `POST /agent/runs/{id}/execute/stream` | same | Same drive, streamed as server-sent events (below) |
| `POST /internal/run-check` | run-check body (below) | Absent (404) unless `SLIMX_AGENT_ENABLE_RUN_CHECK` is set; see [Run-check](#run-check) |

`{id}` must be a plain identifier: `^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$`.

### Execute body

| Field | Rule |
| --- | --- |
| `provider` | non-empty string, ≤128 characters |
| `model` | non-empty provider-native model id (for example `qwen3:8b`), ≤512 characters |
| `base_url` | string ≤2048 characters, or null |
| `lease_job_id`, `lease_token` | UUIDs; required |
| `lease_generation` | JSON integer ≥0 (not a string or boolean); required |

The host may send additional host-only profile fields; ControlRoom sends safe settings and the
profile id and name. They are accepted and ignored. `provider`, `model`, and `base_url` are
forwarded **exactly** on every callback, never defaulted or substituted. The host compares them
with its own authoritative routing and re-resolves everything else from its records.

### Execute responses

| Status | Meaning |
| --- | --- |
| 200 | The drive reached a stop: approval gate, pause, cancel, failure, or completion |
| 401 | Missing or invalid bearer token |
| 404 | The host has no such run |
| 409, other 4xx | The host refused a callback (for example a stale execution lease); its status and bounded detail are relayed |
| 422 | Invalid body or run id, or missing/incomplete execution lease; no callback was made |
| 502 | The host returned 5xx, a transport failure left no observed response, the host answered outside the wire shape, or a step outcome was not observed |
| 503 | `SLIMX_AGENT_HOST_URL` is not configured |

Error details are bounded to 500 characters.

## Execution-attempt callback fence

The three lease fields are all required on both execute endpoints. The service copies them to
explicit headers on **every** callback it makes for that execution:

| Execute body field | Callback header |
| --- | --- |
| `lease_job_id` | `X-SlimX-Agent-Lease-Job-Id` |
| `lease_token` | `X-SlimX-Agent-Lease-Token` |
| `lease_generation` | `X-SlimX-Agent-Lease-Generation` |

The host must validate that tuple against its current execution-attempt ledger on each callback.
The headers are per-request context, never shared client defaults, so concurrent runs cannot leak
or overwrite one another's lease. The bearer token authenticates the service; the lease tuple
separately proves that this worker still owns this particular execution attempt.

## Host callback API

The service calls these endpoints under `/internal/agent-host`. Every response is parsed
strictly: a field with the wrong JSON type is a protocol error, never coerced. Identities are
quoted into single path segments.

| Callback | Request | Response |
| --- | --- | --- |
| `GET /runs/{run}` | — | Run snapshot; 404 means no such run |
| `GET /runs/{run}/steps` | — | Array of step snapshots in execution order |
| `GET /steps/{step}` | — | Step snapshot; 404 means no such step |
| `POST /runs/{run}/status` | `{status}` | Run snapshot |
| `POST /steps/{step}/state` | `{status, error_set, error?, output_refs_set, output_refs?}` | Step snapshot |
| `POST /runs/{run}/events` | `{type, step_id, payload}` | Event |
| `GET /runs/{run}/events/next-sequence` | — | `{next_sequence}`: an integer ≥0 |
| `GET /runs/{run}/events?after=N` | — | Array of events with `sequence` > N |
| `POST /runs/{run}/steps/{step}/invoke` | `{profile: {provider, model, base_url}}` | Outcome envelope |
| `POST /runs/{run}/run-end` | `{status, profile}` | Any 2xx (ControlRoom returns 204) |

A **run snapshot** has these fields:

- `id` and `status`: non-empty strings;
- `approval_policy`: a string or null;
- `auto_approve`: a boolean or null (null means false);
- `allowed_tools_json` and `preapproved_tools`: arrays of strings, or null;
- `budget_max_steps` and `budget_max_wall_seconds`: integers or null.

A **step snapshot** has these fields:

- `id`, `type`, and `status`: non-empty strings;
- `title`: a string or null;
- `requires_approval`: a boolean or null.

An **event** is an object with an integer `sequence`.

**Presence flags.** `error_set`/`output_refs_set` make "leave unchanged" (`false`) distinct
from "clear" (`true` with `null`).

### Outcome envelopes

| Host answer | Engine outcome |
| --- | --- |
| `{"outcome": "completed", "output_refs": {...} \| null}` | step `completed` |
| `{"outcome": "skipped", "reason": "..."}` | step `skipped` (honest skip) |
| `{"outcome": "prepared", "reason": "..."}` | no terminal state; the prepared generation re-enters the gates |
| `{"outcome": "failed", "error": "..."}` | step `failed` |
| anything else | **no observed outcome** |

"Anything else" covers these cases:

- an unrecognized or missing `outcome`;
- non-object `output_refs`;
- a non-object or non-JSON body;
- a host 4xx or 5xx;
- a transport failure after which the host may already have run the tool.

In every one of these cases the engine writes no terminal step state or event, does not retry,
and ends the drive. The host's durable invocation record owns resolution.

### No automatic retries

The service never replays a callback. A lost response to a mutating callback may or may not have
been applied, and only the host can tell. A failed read ends the drive without side effects.

## Streaming

`/execute/stream` emits one `data: <event json>` line per durable event, in `sequence` order.
While a step is quiet, it emits a `: keepalive` SSE comment every
`SLIMX_AGENT_KEEPALIVE_SECONDS` (default 15). The stream simply ends when the drive stops,
including when it stops early on a host error or an unobserved outcome. The durable event table
is the source of truth for anything the stream did not show.

The stream observes; it does not own the run:

- **Disconnects are not cancellation.** When an observer disconnects, the drive continues in
  the service to its next durable stop.
- **Cancellation belongs to the host.** Cancellation and pause are the host's run-status
  authority, and the engine re-reads that status before every step and before declaring
  completion.
- **Process exit.** Stopping the service process stops a running drive. The host's
  execution-lease reclaim path owns that case.

## Run-check

`POST /internal/run-check` runs one command that the host has already allowlisted, in the run's
workspace directory under `AGENT_WORKSPACE_ROOT`.

- **Enabling it.** The endpoint is registered only when `SLIMX_AGENT_ENABLE_RUN_CHECK` is `1`,
  `true`, `yes`, or `on`. Even then it returns 503 without a configured token and 401 without
  the right one.
- **Body.** Unknown fields are refused.

  | Field | Rule |
  | --- | --- |
  | `argv` | 1–64 non-empty strings, each ≤4096 characters, no NUL bytes |
  | `run_id` | Plain identifier; the resolved directory must be a direct child of the resolved workspace root |
  | `timeout_seconds` | Optional; finite, >0 and ≤600; default 120 |
  | `output_cap` | Optional; JSON integer from 1 to 100000; default 20000 |

- **Enforced.**
  - no shell;
  - an environment of only `PATH` and `HOME`;
  - closed stdin;
  - output bounded while it is read (the excess is drained and discarded);
  - a wall-clock timeout;
  - SIGKILL of the whole process group on every exit path.
- **Response.** `{ok, exit_code, timed_out, output, output_truncated}`. A command that cannot
  start returns `ok: false` with a short reason.
- **Not enforced.** Filesystem, network, CPU, and memory containment. A descendant that starts
  its own session escapes the group kill. The workspace is mutable, so a result is **not bound
  to an exact snapshot**, and hosts must not treat it as digest-bound check evidence. ControlRoom
  refuses service-mode checks for that reason. An isolated exact-snapshot runner is separate
  future work.

## Boundary rules

- **Egress.** The host resolves provider profiles and enforces cloud egress before any request
  reaches the service.
- **Host ownership.** The host owns claiming, authorization, artifacts, reads, and quick state
  mutations.
- **Events.** Events are append-only with a per-run monotonic `sequence`. Payloads carry small
  references, never content (see `slimx_agent.contracts.EVENT_TYPES`).
- **Host-side work.** Planning and system-map extraction remain host-side because they require
  host-shaped context. This service only executes already-planned runs.
