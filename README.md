# mesh-dev (`kleinbem/mesh-dev-plugin`)

Lightweight, zero-GPU developer and test plugin for [Mesh-LLM](https://github.com/Mesh-LLM/mesh-llm).
Provides deterministic fake OpenAI-compatible inference models, controllable supervision fault
injection, and a live opt-in request/observation/exchange inspector at `< 5 MB` RSS — catching host,
proxy, routing, `exclusive_gpu` handoff, supervision, and plugin-contract bugs end-to-end without a
GPU or multi-GB GGUF models.

## Install

```bash
# From GitHub Releases (mesh-llm >= 0.80.0)
mesh-llm plugins install kleinbem/mesh-dev-plugin

# Or from a locally packaged archive
mesh-llm plugins install --archive target/dist/mesh-dev-$(uname -m)-unknown-linux-gnu.tar.gz --name mesh-dev
```

## Quick Configuration (`config.toml`)

### Basic fake inference + fault injection

```toml
[[plugin]]
name = "mesh-dev"
```

### With live host observation + OpenAI exchange inspection

```toml
[[plugin]]
name = "mesh-dev"
allow_inference_observation = true

[plugin.openai_exchange_grant]
endpoints = ["chat_completions", "completions", "responses"]
phases = ["request_received", "backend_selected", "exchange_finished"]
metadata = true
headers = ["x-mesh-target", "x-mesh-exclude", "x-request-id", "x-mesh-nonce", "x-mesh-dev-nonce"]
deadline_ms = 250
max_body_bytes = 1048576
max_queue_bytes = 4194304
max_in_flight = 32
```

### With `exclusive_gpu` VRAM handoff testing (tracking: `Mesh-LLM/mesh-llm#3174`)

```toml
[[plugin]]
name = "mesh-dev"
exclusive_gpu = true
```

## Built-in Model Catalog (`endpoint:inference/openai_compatible`)

| Model ID | Behavior | Useful For |
|---|---|---|
| `dev/echo` | Deterministic JSON echoing prompt, headers (`x-mesh-target`, nonces; auth redacted), timing, `plugin_pid`, and `plugin_instance_id`; supports `--token-delay-ms` pacing | Routing verification, header forwarding/stripping, restart detection, 2-node spillover A/B tests |
| `dev/slow-ttft-15s`, `dev/slow-ttft-45s`, `dev/slow-ttft-75s` (or any `dev/slow-ttft-<N>s`/`ms`) | Waits `N`s (or `x-mesh-dev-ttft-ms`) before first output on both non-streaming JSON and SSE; tracks disconnect cancellation (`cancelled_count`) | Long TTFT proxy timeout testing & disconnect cancellation |
| `dev/slow-sse-1tps` (or any `dev/slow-sse-<N>tps`) | Streams tokens at `N` tok/s (override via `--slow-sse-interval-ms`, `--token-delay-ms`, or `x-mesh-dev-interval-ms`) | SSE keep-alive, spillover routing, client-disconnect cancellation, `exclusive_gpu` stream hold |
| `dev/mid-stream-drop` | Flushes `K` valid SSE chunks (default 3) then aborts TCP chunked transfer without `[DONE]` | Truncated stream handling, exchange terminal recording |
| `dev/http-503` | HTTP `503 Service Unavailable` + `Retry-After: 1` | Peer failover, retry/circuit-breaker handling |
| `dev/http-429` | HTTP `429 Too Many Requests` + `Retry-After: 2` | Rate-limit handling, `Retry-After` propagation |
| `dev/malformed-sse` | Emits 1 valid chunk then invalid JSON and a truncated SSE frame without `[DONE]` | Proxy SSE parser resilience |
| `dev/tool-call-edge` | Split multi-chunk `<think>` with decoy `</tool_call>` + structured `tool_calls` JSON arguments containing `<think>` & `</tool_call>` (or raw-tag mode with `mode=raw-tag`) | Streaming/non-streaming parser edge cases & digest normalization |
| `dev/mem-probe`, `dev/mem-probe-hold-5s` (or any `dev/mem-probe-hold-<N>s`/`ms`) | Reads live `/proc/meminfo` (`MemAvailable`, `SwapFree`), self `VmRSS`, Tegra UMA, and optional GPU memory; `hold` variants hold the response/stream and return `before` + `after` snapshots | Deterministic `exclusive_gpu` handoff assertions (tracking: `Mesh-LLM/mesh-llm#3174`) and RSS checks |
| `dev/env-probe` | Reports strictly allowlisted non-secret `MESH_LLM_*` and cache-path env vars (never tokens/secrets) | Shared model-cache env propagation tests |
| `dev/virtual-echo` | Virtual model (`VirtualModelRouter`) that invokes nested `PluginContext::infer("dev/echo", ...)` for both JSON and SSE streaming | Virtual-model orchestration, progress lines, and nested inference |

## Full `mesh-llm 0.80.0` Plugin Contract Coverage

Beyond fake inference, `mesh-dev` exercises the entire v3 plugin contract:

- **Typed `[plugin.settings]`** validated via bundled `plugin-manifest.json` (`config_schema`)
- **Stapled HTTP routes**: buffered `GET /api/plugins/mesh-dev/http/ping` & `POST /api/plugins/mesh-dev/http/echo`, plus streamed SSE `GET /api/plugins/mesh-dev/http/stream` (`StreamMode::Http1`)
- **Full Streamable HTTP MCP (`POST /mcp`)**: tools (`mesh-dev.<op>`), exact resources (`mesh-dev://status`, `mesh-dev://models`), prefix resource template (`mesh-dev://inspect/{source}`), prompts (`mesh-dev.diagnose_model`), completions, and SEP-2663 tasks (`tasks/get`, `tasks/update`, `tasks/cancel`)
- **Host lifecycle & security capabilities**: `openai_exchange.v1` (admission allow/deny, `x-plugin-<hex>-*` response headers, `sha256-v1` wire-byte side streams), opt-in `request_transform.v1`, `plugin_status.v1`, `plugin_keys.v1`, `peer_blocks.v1`, `local_channels.v1` (`mesh-dev/bus`), and `mesh_event_*` subscriptions

See [`docs/CAPABILITY_AUDIT.md`](docs/CAPABILITY_AUDIT.md) for the complete `mesh-llm 0.80.0` capability matrix and confirmed upstream bugs/SDK gaps.

## Key CLI / `[[plugin]] args` Flags for A/B Testing

| Flag | Default | Purpose |
|---|---|---|
| `--startup-delay-secs <N>` | `0` | Keeps plugin in `state=loading` (`GET /v1/models` → `503 loading`) for `N`s after init before exposing models (tests `ENDPOINT_STARTUP_GRACE_SECS = 30s`) |
| `--max-concurrency <N>` | `1` | Synchronous concurrency limit returning `503` before SSE `200 OK` headers (tests spillover routing) |
| `--token-delay-ms <MS>` | `0` | Token pacing delay on `dev/echo` / `dev/slow-sse-1tps` (e.g. `500` ms for a 2 tok/s node vs fast peer) |
| `--enable-request-transform` | `false` | Advertise and register `request_transform.v1` pre-inference request transforms |
| `--no-virtual-model` | `false` | Disable the `dev/virtual-echo` virtual model |
| `--allow-public-mesh` | `false` | Allow running on a public mesh (refused by default via `STARTUP_DISABLED` to protect dev fault ops) |
| `--startup-disabled [REASON]` | unset | Returns `PluginError::startup_disabled` during `InitializeRequest` |
| `--init-delay-ms <MS>` | `0` | Delays `InitializeRequest` response |
| `--exit-on-start <CODE>` | unset | Exits immediately on startup for crash-loop backoff testing |
| `--unhealthy-on-start [DETAIL]` | unset | Boots with `HealthRequest` failing |

## Plugin Operations (`mesh-llm plugins call mesh-dev <op>`)

| Operation | Example `--input` | Purpose |
|---|---|---|
| `status` | `{}` | Instance ID, PID, uptime, `self_rss_kb`, `loading`, `max_concurrency`, `in_flight`, `received_settings`, active faults, negotiated capabilities |
| `env_probe` | `{}` | Strictly allowlisted non-secret `MESH_LLM_*` / cache-path env vars |
| `inject_crash` | `{"exit_code": 1, "delay_ms": 50}` | Trigger exit/abort for supervision restart backoff & gossip withdrawal |
| `inject_health` | `{"mode": "error", "detail": "boom"}` or `{"mode": "hang", "hang_ms": 30000}` | Test `/api/status` degradation and health-check timeouts |
| `inject_http_fault` | `{"mode": "models_503"}` / `{"mode": "models_empty"}` | Test `/v1/models` probe failure and gossip withdrawal |
| `set_model_override` | `{"model": "dev/echo", "behavior": "http503", "count": 2}` | Inject transient or permanent per-model faults |
| `set_exchange_policy` | `{"decision": "allow", "response_headers": [["x-plugin-6d6573682d646576-hook", "v1"]]}` | Control `openai_exchange.v1` admission (`abstain`/`allow`/`deny`), headers, and annotations |
| `set_transform_policy` | `{"mode": "tag_system", "tag": "[tagged]"}` | Control `request_transform.v1` (`unchanged`, `tag_system`, `reject`, `corrupt_model`) |
| `create_task` | `{"label": "probe"}` | Create a tracked SEP-2663 MCP task (`tasks/get`, `tasks/update`, `tasks/cancel`) |
| `publish_status` | `{"payload": {"note": "ok"}, "fresh_for_ms": 5000}` | Publish or withdraw (`{"withdraw": true}`) a `plugin_status.v1` snapshot in `/api/status` |
| `announce_key` | `{}` or `{"public_key_hex": ""}` | Announce or withdraw an Ed25519 key via `plugin_keys.v1` |
| `peer_block` | `{"action": "block", "peer_id": "<64-hex>", "length": "seven_days"}` | Exercise `peer_blocks.v1` (`allow_peer_blocks = true`) |
| `send_channel` | `{"channel": "mesh-dev/bus", "payload": {"ping": 1}}` | Broadcast on a declared `local_channels.v1` channel |
| `resubscribe_observation` | `{}` | Re-attempt `allow_inference_observation` subscription without restart |
| `reset_faults` | `{}` | Clear all injected health, HTTP probe, loading, exchange, transform, and model overrides |
| `inspect_recent` | `{"source": "all", "limit": 20}` | Query bounded ring buffers (`http`, `observation`, `exchange`, `events`) + live `in_flight` count |
| `stats` | `{}` | Aggregated counters (`cancelled_count`, `backpressure_503`, `side_streams_verified`, observation, exchange, events, RSS) |
| `clear` | `{"source": "all"}` | Reset inspector ring buffers and counters |

## Declarative E2E Test Suites

`mesh-dev test-suite` runs declarative TOML conformance suites against an isolated `mesh-llm`
instance (temp `HOME`, non-conflicting ports, strict `mesh-llm config validate` against bundled
`plugin-manifest.json`):

```bash
# Run all 5 suites (96 steps) against a real mesh-llm binary
nix develop -c just e2e-suites /path/to/mesh-llm

# Or run a specific suite directly (optionally installing a packaged release archive)
mesh-dev test-suite suites/02-plugin-contract-full.toml \
  --mesh-llm /path/to/mesh-llm \
  --archive target/dist/mesh-dev-x86_64-unknown-linux-gnu.tar.gz
```

Suites in [`suites/`](suites/):
- `01-core-inference-and-inspector.toml` — all `dev/*` inference models, dynamic overrides, env redaction, exchange inspector, and upstream probes
- `02-plugin-contract-full.toml` — full 40-step v3 plugin contract (settings, stapled buffered/streamed HTTP, full MCP + SEP-2663 tasks, virtual model, status/keys/blocks/channels, exchange admission + headers + `sha256-v1` side streams, crash-restart supervision)
- `03-request-transform.toml` — `request_transform.v1` (`unchanged`, `tag_system`, fail-closed `reject`, mutated-model rejection)
- `04-startup-grace-and-backpressure.toml` — `--startup-delay-secs` loading transition and `--max-concurrency 1` synchronous `503` backpressure
- `05-security-boundaries.toml` — least-privilege capability denial (`allow_inference_observation = false`, `allow_peer_blocks = false`), credential header stripping, `openai_exchange.v1` header spoofing / cross-plugin / CRLF defenses, admission error redaction, and secret env redaction

## Security Hardening

- **Private-mesh-only by default**: refuses `MeshVisibility::Public` with `STARTUP_DISABLED` unless `--allow-public-mesh` is explicitly passed.
- **Bounded resources**: 4 MiB HTTP request body cap (`413 Payload Too Large`), 16 MiB side-stream cap, bounded ring buffers, and capped SEP-2663 `TaskStore` (256 tasks, 256-byte labels).
- **Zero-secret echo/env**: `env_probe` uses a strict allowlist + denylist; `dev/echo` and the HTTP inspector redact any header matching `token`, `secret`, `key`, `auth`, `cookie`, `session`, `jwt`, `x-owner-*`, `x-mesh-owner-*`, or `x-mesh-control-*`.
- **Hardened side streams**: `0o600` Unix socket permissions + automatic cleanup on both `Http1` and `RawBytes` side streams, CSPRNG UUIDv4 `sha256-v1` auth tokens, and constant-time token comparison.

## Standalone Mode

```bash
mesh-dev serve --host 127.0.0.1 --port 8080
mesh-dev mem-probe
mesh-dev manifest
```

## Development

```bash
nix develop -c just check
nix develop -c just archive
nix develop -c just e2e-suites /path/to/mesh-llm
```

## License

Apache-2.0. See [LICENSE](LICENSE).
