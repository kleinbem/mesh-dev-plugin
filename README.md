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

## Key CLI / `[[plugin]] args` Flags for A/B Testing

| Flag | Default | Purpose |
|---|---|---|
| `--startup-delay-secs <N>` | `0` | Keeps plugin in `state=loading` (`GET /v1/models` → `503 loading`) for `N`s after init before exposing models (tests `ENDPOINT_STARTUP_GRACE_SECS = 30s`) |
| `--max-concurrency <N>` | `1` | Synchronous concurrency limit returning `503` before SSE `200 OK` headers (tests spillover routing) |
| `--token-delay-ms <MS>` | `0` | Token pacing delay on `dev/echo` / `dev/slow-sse-1tps` (e.g. `500` ms for a 2 tok/s node vs fast peer) |
| `--startup-disabled [REASON]` | unset | Returns `PluginError::startup_disabled` during `InitializeRequest` |
| `--init-delay-ms <MS>` | `0` | Delays `InitializeRequest` response |
| `--exit-on-start <CODE>` | unset | Exits immediately on startup for crash-loop backoff testing |
| `--unhealthy-on-start [DETAIL]` | unset | Boots with `HealthRequest` failing |

## Plugin Operations (`mesh-llm plugins call mesh-dev <op>`)

| Operation | Example `--input` | Purpose |
|---|---|---|
| `status` | `{}` | Instance ID, PID, uptime, `self_rss_kb`, `loading`, `max_concurrency`, `in_flight`, active faults, negotiated capabilities |
| `env_probe` | `{}` | Strictly allowlisted non-secret `MESH_LLM_*` / cache-path env vars |
| `inject_crash` | `{"exit_code": 1, "delay_ms": 50}` | Trigger exit/abort for supervision restart backoff & gossip withdrawal |
| `inject_health` | `{"mode": "error", "detail": "boom"}` or `{"mode": "hang", "hang_ms": 30000}` | Test `/api/status` degradation and health-check timeouts |
| `inject_http_fault` | `{"mode": "models_503"}` / `{"mode": "models_empty"}` | Test `/v1/models` probe failure and gossip withdrawal |
| `set_model_override` | `{"model": "dev/echo", "behavior": "http_503", "count": 2}` | Inject transient or permanent per-model faults |
| `publish_status` | `{"payload": {"note": "ok"}, "fresh_for_ms": 5000}` | Exercise `plugin_status.v1` fresh/stale/withdrawn transitions |
| `resubscribe_observation` | `{}` | Re-attempt `allow_inference_observation` subscription without restart |
| `reset_faults` | `{}` | Clear all injected health, HTTP probe, loading, and model overrides |
| `inspect_recent` | `{"source": "all", "limit": 20}` | Query bounded ring buffers (`http`, `observation`, `exchange`) + live `in_flight` count |
| `stats` | `{}` | Aggregated counters (`cancelled_count`, `backpressure_503`, observation, exchange, RSS) |
| `clear` | `{"source": "all"}` | Reset inspector ring buffers and counters |

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
```

## License

Apache-2.0. See [LICENSE](LICENSE).
