# Mesh-LLM `0.80.0` Plugin Capability Audit & Bug Report

This document audits every plugin capability defined in `mesh-llm 0.80.0` (`mesh-llm-plugin` and
`mesh-llm-host-runtime` at upstream commit `ea3d3e5c6`, byte-identical on `mesh-llm-plugin` to
release tag `v0.80.0` / `2b9e64ae`), maps each surface to declarative conformance tests in
`mesh-dev`, and records every bug and SDK/DSL gap verified live against the real `mesh-llm 0.80.0`
binary.

## How to Reproduce Everything

```bash
# 1. In-process Rust unit, protocol, lifecycle, security, and declarative `plugin!` DSL tests (45 tests)
nix develop --command cargo test --locked -- --test-threads 1

# 2. All 5 declarative TOML E2E suites live against real `mesh-llm 0.80.0` (96 steps)
nix develop --command just e2e-suites /path/to/mesh-llm
```

---

## Complete Capability Matrix (`mesh-llm 0.80.0`)

| # | Capability / Surface | `SimplePlugin` SDK | `plugin!` Macro DSL | Live E2E Status (`mesh-llm 0.80.0`) | Declarative Test Coverage |
|---|---|---|---|---|---|
| 1 | Managed OpenAI inference endpoint (`endpoint:inference/openai_compatible`) | Yes | Yes (`inference::provider`) | **PASS** (exact-match `/v1/models` routing; ~15s health re-poll) | `suites/01`, `suites/04` |
| 2 | Operations (`mesh-llm plugins call` + `/api/plugins/<name>/tools/<op>`) | Yes (`OperationRouter`) | Yes (`mcp::tool`) | **PASS** (`INVALID_PARAMS` maps to HTTP 400) | `suites/01`, `suites/02` |
| 3 | Typed `config_schema` & `[plugin.settings]` strict validation | Yes (`config_schema`) | Yes (`config: [...]`) | **PASS** (requires bundled `plugin-manifest.json` in archive) | `suites/02` Steps 1–3 |
| 4 | Stapled buffered HTTP routes (`/api/plugins/<name>/http/*`) | Yes (`http_get`/`http_post`) | Yes (`http::get`/`post`, **leaks MCP tools — Bug #1**) | **PASS** (`GET /ping`, `POST /echo`) | `suites/02` Steps 3–4, `tests/declarative_dsl.rs` |
| 5 | Stapled streamed HTTP routes (`StreamMode::Http1` side stream) | Yes (`on_open_stream` + `.streamed_response()`) | **No** (no `on_open_stream`) | **PASS** (`GET /stream` SSE over Unix side stream) | `suites/02` Step 5 |
| 6 | MCP Tools (`POST /mcp` Streamable HTTP `tools/list`, `tools/call`) | Yes | Yes | **PASS** (qualified as `<plugin>.<tool>`) | `suites/02` Steps 6–7 |
| 7 | MCP Exact Resources (`resources/list`, `resources/read`) | Yes (`ResourceRouter`) | Yes (`mcp::resource`) | **PASS** (`mesh-dev://status`, `mesh-dev://models`) | `suites/02` Steps 8–9 |
| 8 | MCP Resource Templates (`resources/templates/list`, prefix read) | Yes (`add_prefix_template`) | Yes (`mcp::resource_template`) | **PASS** (`mesh-dev://inspect/{source}`) | `suites/02` Steps 10–11 |
| 9 | MCP Prompts (`prompts/list`, `prompts/get`) | Yes (`PromptRouter`) | Partial (`mcp::prompt` **missing `.argument()` — Gap #5**) | **PASS** (`mesh-dev.diagnose_model`) | `suites/02` Steps 12–13 |
| 10 | MCP Argument Completions (`completion/complete`) | Yes (`CompletionRouter`) | Yes (`mcp::completion`) | **PASS** (`ref/prompt` prefix filtering) | `suites/02` Step 14 |
| 11 | MCP SEP-2663 Tasks (`tasks/get`, `tasks/update`, `tasks/cancel`) | Yes (`TaskRouter` + `TaskStore`) | **No** | **PASS** (requires client `io.modelcontextprotocol/tasks` extension) | `suites/02` Steps 15–21 |
| 12 | Virtual Models (`dev/virtual-echo`) + nested `PluginContext::infer` | Yes (`VirtualModelRouter`) | **No** | **PASS** (non-streaming & SSE streaming) | `suites/02` Steps 22–23 |
| 13 | `plugin_status.v1` (`/api/status` -> `plugins.<name>`) | Yes (`publish_status` / `withdraw_status`) | **No** | **PASS** (`state = "ready"`, custom JSON `data`, withdraw) | `suites/02` Steps 24–26 |
| 14 | `plugin_keys.v1` (`announce_plugin_key` -> `/api/plugin-keys`) | Yes (`PluginContext`) | Partial (`on_initialized` only) | **PASS** (announce, list, withdraw via empty key) | `suites/02` Steps 27–29 |
| 15 | `peer_blocks.v1` (`allow_peer_blocks = true`) | Yes (`request_peer_block`) | Partial (`PluginContext` only) | **PASS** (block 7d + unblock) | `suites/02` Steps 30–31 |
| 16 | `local_channels.v1` + `mesh_event_*` subscriptions | Yes (`on_channel_message`, `on_mesh_event`) | Yes (`mesh: [...]`, `events: [...]`) | **PASS** (`send_channel` on `mesh-dev/bus`) | `suites/02` Step 32, `tests/declarative_dsl.rs` |
| 17 | `openai_exchange.v1` (admission, response headers, `sha256-v1` wire side streams) | Yes (`with_openai_exchange_handler` + `on_open_stream`) | **No** | **PASS** (header injection, HTTP 403 deny, `sha256-v1` receipts verified) | `suites/01`, `suites/02` Steps 33–38 |
| 18 | `request_transform.v1` (host-private pre-inference transform) | Yes (`with_request_transform`) | **No** | **PASS** (`unchanged`, `tag_system`, `reject`, `corrupt_model` rejected) | `suites/03` (10/10 steps) |
| 19 | `local_inference_observation.v1` (`allow_inference_observation = true`) | Yes (`subscribe_inference_observation`) | **No** | **PARTIAL / GAP (Bug #3)** — subscribes cleanly, 0 events for plugin endpoints | `suites/01` Steps 16–17 |
| 20 | Host supervision, health checks, startup grace & crash restart | Yes (`with_health`, `STARTUP_DISABLED_ERROR_CODE`) | Partial (no `on_initialize`) | **PASS** (`--startup-delay-secs`, `inject_crash` restart recovery) | `suites/02` Steps 39–40, `suites/04` |
| 21 | `web_ui` static bundle & slots (`chat_message`, `logs_request`) | Yes (`web_ui(...)`) | Yes (`web_ui: [...]`) | Not exercised (headless test plugin) | — |

---

## Confirmed Upstream Bugs & Discrepancies

### Bug #1 — `plugin!` macro leaks synthetic stapled HTTP routes into MCP `tools/list` and `/api/plugins/<name>/tools`

- **Severity**: Medium (pollutes LLM MCP tool catalogs with internal HTTP endpoints)
- **Location**:
  - `mesh/crates/mesh-llm-plugin/src/dsl.rs:27-40` (`normalize_http_operation_name`)
  - `mesh/crates/mesh-llm-plugin/src/dsl.rs:882-947` (`HttpRouteBuilder::handle`)
  - `mesh/crates/mesh-llm-host-runtime/src/plugin/mcp/tool_dispatch.rs:199` (`PluginMcpServer::discover_tools`)
  - `mesh/crates/mesh-llm-host-runtime/src/api/routes/plugins.rs:265` (`handle_tools`)
- **Details**:
  When a plugin uses `plugin! { http: [ http::get("/ping")..., http::post("/echo")... ] }`,
  `HttpRouteBuilder::handle` synthesizes an operation name via `normalize_http_operation_name`
  (which also strips the leading `__` via `.trim_matches('_')`, turning `__http_get_/ping` into
  `http_get__ping`), appends it to `manifest.operations`, and registers it in `OperationRouter`.
  Neither the host MCP tool discovery nor `GET /api/plugins/<name>/tools` filters out HTTP-only
  bindings, so every stapled HTTP route leaks into LLM `tools/list`.
- **Upstream**: `upstream: Mesh-LLM/mesh-llm#3190`
- **Reproducer**: `cargo test --test declarative_dsl -- declarative_plugin_macro_exercises_all_arms_and_reproduces_http_tool_leak`

### Bug #2 — OpenAI ingress port routes unknown `/api/...` POSTs to a random plugin inference model instead of `404`

- **Severity**: Medium (confusing 503 errors & wasted inference attempts when callers hit the wrong port)
- **Location**: `mesh/crates/mesh-llm-host-runtime/src/network/openai/ingress.rs:270-310` (`resolve_auto_routed_model`)
- **Details**:
  Sending `POST /api/plugins/mesh-dev/tools/env_probe` with body `{}` to the OpenAI ingress port
  (`--port`, default `9337`) instead of the console port (`--console`, default `3131`) parses
  `request.model_name = None`, treats the request as automatic model routing, picks an arbitrary
  advertised plugin model (e.g. `dev/tool-call-edge` or `dev/mem-probe`), and forwards the unknown
  path to that plugin endpoint, returning:
  ```json
  HTTP 503 {"error":{"message":"plugin endpoint for model 'dev/mem-probe' failed","type":"server_error","param":null,"code":"service_unavailable"}}
  ```
  Unknown paths on the OpenAI ingress listener should be rejected with `404 Not Found` before
  auto-routing runs.
- **Upstream**: `tracking: Mesh-LLM/mesh-llm#3186`
- **Reproducer**: `suites/01-core-inference-and-inspector.toml` Step 18 (`XFAIL`).

### Bug #3 — `local_inference_observation.v1` does not observe plugin-hosted inference endpoints

- **Severity**: Medium (silent telemetry blind spot for plugins using `allow_inference_observation = true`)
- **Location**: `mesh/crates/mesh-llm-host-runtime/src/inference/skippy/{dedicated,mod}.rs`
- **Details**:
  `observe_frontend()` is only hooked into native Skippy inference execution. Plugin-hosted
  `openai_compatible` inference endpoints (`layerstream`, `flash-moe`, `mesh-dev`) are invisible
  to `local_inference_observation.v1`: subscribers succeed (`status = "subscribed"`), yet receive
  `0` execution events and `ready_models = []` even while `openai_exchange.v1` observes every
  request.
- **Upstream**: `tracking: Mesh-LLM/mesh-llm#3187`
- **Reproducer**: `suites/01-core-inference-and-inspector.toml` Steps 16–17 (`XFAIL`).

### Bug #4 — `PluginRuntime::handle_initialize` returns `-32600` (`INVALID_REQUEST`) for missing host capabilities before checking `host_protocol_version`, causing supervisor restart loops

- **Severity**: Medium (causes restart churn when running newer plugins on older hosts)
- **Location**:
  - `mesh/crates/mesh-llm-plugin/src/runtime.rs:1067-1077` (`PluginRuntime::handle_initialize`)
  - `mesh/crates/mesh-llm-host-runtime/src/plugin/runtime.rs:410-423` (`parse_initialize_response`)
- **Details**:
  `PluginRuntime::handle_initialize` validates `host_capabilities::unsupported_requirement(...)`
  (such as `local_channels.v1` or `request_transform.v1`) *before* invoking `plugin.initialize(...)`
  (where `host_protocol_version` is checked) and returns `-32600` (`INVALID_REQUEST`).
  On the host side, `parse_initialize_response` only calls `mark_disabled()` on
  `STARTUP_DISABLED_ERROR_CODE` (`-32010`), treating `-32600` as a crash that triggers supervisor
  restarts even though host capabilities cannot change without upgrading/restarting the host.
- **Upstream**: `upstream: Mesh-LLM/mesh-llm#3189`
- **Reproducer**: `cargo test --test declarative_dsl -- missing_local_channels_capability_returns_invalid_request_before_protocol_check`

---

## SDK & Declarative DSL Ergonomics Gaps

### Gap #5 — `plugin!` macro lags behind `SimplePlugin` by 9 v0.80.0 capabilities

`DeclarativePluginBuilder` / `plugin!` (`mesh/crates/mesh-llm-plugin/src/dsl.rs`) currently lacks
support for:
1. `VirtualModelRouter` (`virtual_model`)
2. `inference_observation` declaration / subscriber helper
3. `openai_exchange_hook` + `with_openai_exchange_handler`
4. `plugin_status`
5. `request_transform` + `with_request_transform`
6. `TaskRouter` (SEP-2663 `tasks/get`, `tasks/update`, `tasks/cancel`)
7. `on_open_stream` (needed for streamed stapled HTTP `StreamMode::Http1` and `sha256-v1` wire-byte side streams)
8. `on_initialize` (needed to read `[plugin.settings]` before `InitializeResponse`)
9. `McpPromptBuilder` (`dsl.rs:713-758`) has no `.argument(...)` method and hardcodes
   `arguments: None` in `prompts/list`, even though `mcp::completion("prompt.<name>.<arg>")`
   expects named prompt arguments.

### Gap #6 — Undocumented runtime contracts & silent failure modes

1. **Hex-encoded `openai_exchange.v1` response header prefix**:
   `mesh/crates/mesh-llm-host-runtime/src/plugin/exchange_metadata.rs:4` requires every
   hook-injected response header to match `format!("x-plugin-{}-", hex::encode(plugin_name.as_bytes()))`
   (e.g. `x-plugin-6d6573682d646576-hook` for `mesh-dev`) **and** be listed in both
   `manifest.openai_exchange_hook.headers` and `[plugin.openai_exchange_grant].headers`.
   `mesh-llm-plugin` does not export a `response_header_prefix()` helper.
2. **UUID-formatted `sha256-v1` side-stream token**:
   `mesh/crates/mesh-llm-host-runtime/src/plugin/exchange_streams.rs:70` validates that
   `OpenStreamResponse.token` parses as a 36-char RFC 4122 UUID (`uuid::Uuid::parse_str`). Under
   default `failure_policy = "best_effort"`, returning a non-UUID token silently trips the exchange
   circuit breaker after 3 requests and suppresses subsequent admission/header decisions without
   surfacing a client error.
3. **`plugin-manifest.json` required in archives for strict `[plugin.settings]`**:
   `mesh/crates/mesh-llm-plugin-manager/src/archive.rs:21` reads `<plugin>/plugin-manifest.json`
   at install time. Without it, `mesh-llm plugins install` succeeds silently, but any
   `[plugin.settings]` entry later fails `mesh-llm config validate` with `SchemaUnavailable`.
4. **`TaskStore<T>` unnecessary `T: Default` bound**:
   `mesh/crates/mesh-llm-plugin/src/helpers.rs:1086` derives `Default` on `TaskStore<T>` without
   providing `TaskStore::new()`, requiring custom task payload types to implement `Default`.
5. **Exact-match `/v1/models` routing & ~15s health re-poll cadence**:
   `mesh-llm` only routes OpenAI inference requests for model IDs returned by the plugin's
   `GET /v1/models` (returning `404 model_not_found` for unadvertised dynamic IDs), and after a
   `--startup-delay-secs` loading state (`GET /v1/models` returning `503`), `mesh-llm` picks up
   newly ready models on its ~15s health-probe cadence (observed: `14.9s` in Suite 04).

---

## Verified Corrections (Non-Bugs Confirmed Working Live)

- **`/api/status` plugin snapshots key**: Although the Rust struct field in
  `mesh/crates/mesh-llm-host-runtime/src/api/status.rs:400` is named `plugin_status`, it carries
  `#[serde(rename = "plugins")]`, so `GET /api/status` correctly serves snapshots under
  `plugins.<name>` matching `proto/plugin.proto:713`. Verified live in Suite 02 Step 25.
- **Header stripping on OpenAI proxy**: `mesh-llm` strips `Authorization`, `Cookie`, `X-Api-Key`,
  `X-Owner-*`, and `X-Mesh-Control-*` headers before forwarding to local plugin inference endpoints
  (verified live in Suite 01 Step 2 and Suite 05 Step 4).
- **Admission deny message redaction**: When an `openai_exchange.v1` hook denies a request,
  `mesh-llm` returns HTTP `403` with a fixed host-owned message
  (`"OpenAI plugin admission rejected the exchange"`, `code: "permission_denied"`) rather than
  reflecting untrusted plugin text to clients (verified live in Suite 02 Step 36 and Suite 05 Step 18).

---

## Security Hardening & Live Boundary Verification

### `mesh-dev` Defense-in-Depth (`tests/security.rs` — 5/5 PASS)

1. **Private-mesh-only by default**: `mesh-dev` refuses `MeshVisibility::Public` with
   `PluginError::startup_disabled` (`-32010`) unless `--allow-public-mesh` is explicitly passed,
   preventing accidental exposure of fault-injection and inspection ops on public meshes.
2. **Bounded HTTP & side-stream surfaces**: embedded Axum server enforces a `4 MiB` request body
   limit (`413 Payload Too Large`), `sha256-v1` side streams enforce a `16 MiB` byte cap, and
   `TaskStore` enforces `MAX_TRACKED_TASKS = 256` and `MAX_TASK_LABEL_LEN = 256`.
3. **Zero-secret echo & env probes**: `env_probe` combines an explicit allowlist with a secret-keyword
   denylist; `dev/echo` and the HTTP inspector redact headers matching `mesh_llm_config::safe_exchange_header`
   (`token`, `secret`, `key`, `auth`, `cookie`, `session`, `jwt`, `x-owner-*`, `x-mesh-owner-*`,
   `x-mesh-control-*`) and bound captured header counts/lengths (`64` headers / `1024` bytes).
4. **Hardened Unix side streams**: `0o600` socket permissions + RAII `SocketCleanupGuard` on both
   `StreamMode::Http1` and `StreamMode::RawBytes`, CSPRNG `uuid::Uuid::new_v4()` `sha256-v1` auth
   tokens, and constant-time token comparison.

### Live `mesh-llm 0.80.0` Security Boundaries (`suites/05-security-boundaries.toml` — 20/20 PASS)

Verified live against real `mesh-llm 0.80.0`:
- `allow_inference_observation = false` denies the host observation feed (`status = "not_granted"`).
- `allow_peer_blocks = false` rejects `peer_block` operations (`4xx/5xx`).
- `env_probe` never leaks `MESH_LLM_JOIN_TOKEN`, `OPENAI_API_KEY`, `AWS_SECRET_ACCESS_KEY`, or `GITHUB_TOKEN`.
- OpenAI proxy strips/redacts `Authorization`, `Cookie`, `X-Api-Key`, `X-Owner-Token`, and `X-Mesh-Control-Admin` while forwarding benign `X-Mesh-Dev-Nonce`.
- `openai_exchange.v1` blocks spoofed `Set-Cookie`, cross-plugin `x-plugin-<other>-*`, un-granted `x-plugin-<self>-*`, and CRLF-injected header values while still allowing validly granted `x-plugin-6d6573682d646576-hook`.
- Admission `Deny` returns HTTP `403 permission_denied` with a fixed host message and never reflects untrusted plugin reason strings.
- `sha256-v1` wire-byte side streams authenticate with CSPRNG UUIDv4 tokens and verify end-to-end.
