# SlimX-Agent service contract (v1)

The HTTP surface the standalone agent service exposes to a host. ControlRoom's
`app/slimx_agent_service/main.py` implements it today (same api image, shared DB —
the documented transitional design); a service built from THIS repo must keep it
wire-compatible.

## Auth

`SLIMX_AGENT_INTERNAL_TOKEN` — shared bearer token, constant-time compared when set;
empty = auth off (local-first). One value drives both sides. `GET /health` reports
`auth_enabled` so the host's deep health can flag one-sided tokens.

## Endpoints

| Endpoint | Body | Semantics |
| --- | --- | --- |
| `GET /health` | — | `{status, service: "slimx-agent", auth_enabled}` |
| `POST /agent/runs/{id}/plan` | `{provider, model, base_url?}` | Generate/replace the run's plan. 409 conflict / 422 invalid plan / 502 generation failure — raw detail strings; the host re-raises its own exception types and adds user-facing wrapping exactly once. |
| `POST /agent/runs/{id}/execute` | same | Drive an **already-claimed** run to its next stop. The HOST claims (atomic) — the service never does. |
| `POST /agent/runs/{id}/execute/stream` | same | Same, streaming each durable progress event as a `data: <json>` SSE line with `:`-comment keepalives. |
| `POST /agent/runs/{id}/extract-map` | same | Model-declared system-map inference for the run. |

## Boundary rules

- The host resolves provider profiles and enforces **cloud egress** before any request
  reaches the service; requests carry the already-resolved `RunProfile`.
- The host owns claiming, authz, artifacts, reads, and quick state mutations.
- Events are append-only with a per-run monotonic `sequence`; payloads carry small
  references, never content (see `slimx_agent.contracts.EVENT_TYPES`).
