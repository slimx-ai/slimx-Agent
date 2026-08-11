# SlimX-Agent service contract (v1)

The HTTP surface the standalone agent service exposes to a host. The service built from this
repository owns the engine loop only: it has no database or provider credentials and drives the
host's `/internal/agent-host/*` callback API for persistence and tool execution.

## Auth

`SLIMX_AGENT_INTERNAL_TOKEN` — shared bearer token, constant-time compared when set;
empty = auth off (local-first). One value drives both sides. `GET /health` reports
`auth_enabled` so the host's deep health can flag one-sided tokens.

## Endpoints

| Endpoint | Body | Semantics |
| --- | --- | --- |
| `GET /health` | — | `{status, service: "slimx-agent", auth_enabled}` |
| `POST /agent/runs/{id}/execute` | `{provider, model, base_url?, lease_job_id, lease_token, lease_generation}` | Drive an **already-claimed** run to its next stop. Missing or incomplete execution-attempt fields are refused with 422 before any host callback. |
| `POST /agent/runs/{id}/execute/stream` | same | Same, streaming each durable progress event as a `data: <json>` SSE line with `:`-comment keepalives. |

Planning and system-map extraction remain host-side because they require host-shaped context. This
service only executes already-planned runs.

## Execution-attempt callback fence

The three lease fields are all required on both execute endpoints. The service copies them to
explicit headers on **every** callback it makes for that execution:

| Execute body field | Callback header |
| --- | --- |
| `lease_job_id` | `X-SlimX-Agent-Lease-Job-Id` |
| `lease_token` | `X-SlimX-Agent-Lease-Token` |
| `lease_generation` | `X-SlimX-Agent-Lease-Generation` |

The host must validate that tuple against the current execution-attempt ledger on each callback.
The headers are per-request context, never shared client defaults, so concurrent runs cannot leak
or overwrite one another's lease. The bearer token authenticates the service; the lease tuple
separately proves that this worker still owns this particular execution attempt.

## Boundary rules

- The host resolves provider profiles and enforces **cloud egress** before any request
  reaches the service; requests carry the already-resolved `RunProfile`.
- The host owns claiming, authz, artifacts, reads, and quick state mutations.
- Events are append-only with a per-run monotonic `sequence`; payloads carry small
  references, never content (see `slimx_agent.contracts.EVENT_TYPES`).
