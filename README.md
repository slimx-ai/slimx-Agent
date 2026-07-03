# SlimX-Agent

The **portable agent core** of the SlimX platform — the contracts, tool-dispatch boundary,
runtime protocol, and planning layer extracted from SlimX-AI ControlRoom.

SlimX layering:

| Layer | Repo | Job |
| --- | --- | --- |
| Model execution | `slimx` | providers, payloads, retries, parallel fan-out |
| Knowledge / retrieval | `SlimX-RAG` | ingest, chunk, embed, index, retrieve, cite |
| Connector transport | `SlimX-MCP` | SSRF-guarded, capped MCP JSON-RPC |
| **Agent core** | **`SlimX-Agent`** | **step/grant/event contracts, ToolRegistry, AgentRuntime protocol, planning** |
| Reasoning workspace | `slimx-brainstorm` (ControlRoom) | UI, persistence, capabilities, orchestration |

## What it owns — and deliberately does not

**Owns:**
- `slimx_agent.contracts` — the single vocabulary: step types (assisted / external /
  code / build / orchestration), grantable tools, run modes, approval policies, and the
  durable event types. Stdlib-only.
- `slimx_agent.tools` — `ToolRegistry` (the ONLY dispatcher→tool path), the step error
  vocabulary (`StepExecutionError` fails a run; `StepNotApplicable` skips honestly), and the
  typed `AgentRunContext` hosts build from their own state. Stdlib-only.
- `slimx_agent.runtime` — the `AgentRuntime` protocol every host route consumes, plus
  `RunProfile` (the host resolves providers and enforces cloud egress BEFORE calling the
  runtime) and the stable conflict errors. Stdlib-only.
- `slimx_agent.planning` — plan schemas (defaulted dataclasses for structured output +
  strict pydantic validation), best-effort `repair_plan_data`, and the grant-aware
  `build_planner_prompt`.

**Does not own:** execution engines, persistence, model transport, credentials, or any host
capability (evidence, synthesis, RAG, MCP, sandboxes). Hosts register capabilities as
`ToolRegistry` handlers and implement `AgentRuntime`. **No agent framework**
(LangChain/LangGraph/CrewAI/AutoGen/OpenAI-Agents-SDK) is, or may ever become, a dependency.

## Consumption

ControlRoom currently keeps byte-identical copies of `contracts`/`tools`/`runtime` (a parity
guard test on the host compares them against this checkout) until this repo is published;
then the host modules become re-export shims of this package. The standalone agent service's
HTTP surface is documented in [`docs/service-contract.md`](docs/service-contract.md) so the
container can eventually be built from this repo (extraction plan §8: needs the RunStore +
host-tool HTTP boundary first).

## Development

```bash
pip install -e '.[dev]'
ruff check .
pytest
```
