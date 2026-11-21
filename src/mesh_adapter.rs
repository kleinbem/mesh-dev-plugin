//! Native Mesh-LLM v3 plugin adapter hosting the embedded OpenAI-compatible dev server,
//! supervision fault-injection operations, live inference observation subscriber, and
//! OpenAI exchange lifecycle observer.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use mesh_llm_plugin::inference_observation::{InferenceObservationItem, inference_observation};
use mesh_llm_plugin::openai_exchange::{
    OpenAiAdmissionDecision, OpenAiExchangeDecision, OpenAiExchangeEvent, openai_exchange_hook,
};
use mesh_llm_plugin::plugin_status::plugin_status;
use mesh_llm_plugin::{
    OperationRouter, PROTOCOL_VERSION, PluginContext, PluginError, PluginMetadata, PluginRuntime,
    PluginStartupPolicy, SimplePlugin, capability, host_capabilities, json_schema_operation,
    openai_http_inference_endpoint, operation, plugin_manifest, plugin_server_info, proto,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;

use crate::cli::DevArgs;
use crate::env_probe::probe_env;
use crate::faults::{
    FaultController, HealthFaultMode, HttpFaultMode, ModelOverride, OverrideBehavior,
};
use crate::inspector::{InspectSource, Inspector};
use crate::mem_probe::read_self_rss_kb;
use crate::models::ModelDefaults;
use crate::server::{AppState, serve_with_state};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DEFAULT_PLUGIN_NAME: &str = "mesh-dev";
const ENDPOINT_ID: &str = "mesh-dev";

/// Aborts a spawned background task automatically when dropped.
pub struct AbortOnDrop(Option<JoinHandle<()>>);

impl AbortOnDrop {
    fn new(handle: JoinHandle<()>) -> Self {
        Self(Some(handle))
    }

    fn take(&mut self) -> Option<JoinHandle<()>> {
        self.0.take()
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct EmptyArgs {}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct InjectCrashArgs {
    /// Process exit code (default: `1`).
    #[serde(default = "default_exit_code")]
    pub exit_code: i32,
    /// Delay in milliseconds before exiting so the RPC response flushes first (default: `50`).
    #[serde(default = "default_crash_delay_ms")]
    pub delay_ms: u64,
    /// Exit synchronously before sending an RPC response.
    #[serde(default)]
    pub immediate: bool,
    /// Terminate via `std::process::abort()` instead of `std::process::exit()`.
    #[serde(default)]
    pub abort: bool,
}

impl Default for InjectCrashArgs {
    fn default() -> Self {
        Self {
            exit_code: default_exit_code(),
            delay_ms: default_crash_delay_ms(),
            immediate: false,
            abort: false,
        }
    }
}

fn default_exit_code() -> i32 {
    1
}

fn default_crash_delay_ms() -> u64 {
    50
}

#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct InjectHealthArgs {
    /// Target health mode: `"ok"`, `"error"`, or `"hang"`.
    #[serde(default = "default_mode_ok")]
    pub mode: String,
    /// Error detail when `mode == "error"`.
    #[serde(default)]
    pub detail: Option<String>,
    /// Hang duration in milliseconds when `mode == "hang"` (`null` hangs until shutdown).
    #[serde(default)]
    pub hang_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct InjectHttpFaultArgs {
    /// Target `/v1/models` fault mode: `"ok"`, `"models_503"`, `"models_empty"`, or `"models_hang"`.
    #[serde(default = "default_mode_ok")]
    pub mode: String,
    /// Error detail when `mode == "models_503"`.
    #[serde(default)]
    pub detail: Option<String>,
    /// Hang duration in milliseconds when `mode == "models_hang"`.
    #[serde(default)]
    pub hang_ms: Option<u64>,
}

fn default_mode_ok() -> String {
    "ok".to_string()
}

#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct SetModelOverrideArgs {
    /// Model ID to override (e.g. `"dev/echo"`).
    pub model: String,
    /// Behavior override (`default`, `echo`, `http_503`, `http_429`, `mid_stream_drop`, `malformed_sse`, `slow_ttft`, `slow_sse`).
    #[serde(default)]
    pub behavior: OverrideBehavior,
    /// Optional number of requests to apply this override before auto-reverting.
    #[serde(default)]
    pub count: Option<u32>,
    /// Optional delay/interval in milliseconds (used by `slow_ttft`, `slow_sse`, `echo`).
    #[serde(default)]
    pub delay_ms: Option<u64>,
    /// Optional chunk count before drop (used by `mid_stream_drop`).
    #[serde(default)]
    pub chunks: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct PublishStatusArgs {
    /// JSON object published under `/api/status` (`plugin_status.mesh-dev`).
    #[serde(default)]
    pub payload: Option<Value>,
    /// Freshness TTL in milliseconds (default: `5000`).
    #[serde(default = "default_fresh_for_ms")]
    pub fresh_for_ms: u64,
    /// Withdraw the previously published status snapshot instead of publishing.
    #[serde(default)]
    pub withdraw: bool,
}

fn default_fresh_for_ms() -> u64 {
    5_000
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct InspectRecentArgs {
    /// Source to inspect: `"all"`, `"http"`, `"observation"`, or `"exchange"`.
    #[serde(default)]
    pub source: InspectSource,
    /// Maximum records returned per source (default: `20`).
    #[serde(default = "default_inspect_limit")]
    pub limit: usize,
    /// Optional exact model ID filter.
    #[serde(default)]
    pub model: Option<String>,
}

impl Default for InspectRecentArgs {
    fn default() -> Self {
        Self {
            source: InspectSource::All,
            limit: default_inspect_limit(),
            model: None,
        }
    }
}

fn default_inspect_limit() -> usize {
    20
}

#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
pub struct ClearArgs {
    /// Source to clear: `"all"`, `"http"`, `"observation"`, or `"exchange"`.
    #[serde(default)]
    pub source: InspectSource,
}

#[derive(Debug, Clone, Serialize)]
struct NegotiatedCaps {
    inference_observation: bool,
    openai_exchange: bool,
    plugin_status: bool,
}

/// Build the static `PluginManifest` for `mesh-dev`.
pub fn build_manifest(endpoint_base: &str) -> proto::PluginManifest {
    let mut exchange_hook = openai_exchange_hook("observe");
    exchange_hook.required = false;
    exchange_hook.request_body = true;
    exchange_hook.effective_request_body = true;
    exchange_hook.response_body = true;
    exchange_hook.metadata = true;
    exchange_hook.headers = vec![
        "x-mesh-target".into(),
        "x-mesh-exclude".into(),
        "x-request-id".into(),
        "x-mesh-nonce".into(),
        "x-mesh-dev-nonce".into(),
    ];

    plugin_manifest()
        .item(capability("endpoint:inference"))
        .item(capability("endpoint:inference/openai_compatible"))
        .item(capability("backend:mesh-dev"))
        .item(
            openai_http_inference_endpoint(ENDPOINT_ID, endpoint_base)
                .managed_by_plugin(true)
                .supports_streaming(true)
                .protocol("openai_compatible"),
        )
        .item(inference_observation().required(false))
        .item(exchange_hook)
        .item(plugin_status(1).required(false))
        .item(operation::<EmptyArgs>(
            "status",
            "Return mesh-dev instance metadata, RSS, active faults, overrides, and negotiated capabilities.",
        ))
        .item(operation::<InjectCrashArgs>(
            "inject_crash",
            "Trigger a controllable process exit or abort to test host supervision and restart backoff.",
        ))
        .item(operation::<InjectHealthArgs>(
            "inject_health",
            "Switch HealthRequest between ok, error, and hang modes to test /api/status degradation and restart.",
        ))
        .item(operation::<InjectHttpFaultArgs>(
            "inject_http_fault",
            "Inject /v1/models endpoint faults (ok, models_503, models_empty, models_hang) to test gossip withdrawal.",
        ))
        .item(operation::<SetModelOverrideArgs>(
            "set_model_override",
            "Override a dev/* model's behavior permanently or for the next N requests.",
        ))
        .item(operation::<PublishStatusArgs>(
            "publish_status",
            "Publish or withdraw a plugin_status.v1 snapshot in /api/status.",
        ))
        .item(operation::<EmptyArgs>(
            "resubscribe_observation",
            "Re-attempt subscribing to the host local inference observation feed.",
        ))
        .item(operation::<EmptyArgs>(
            "reset_faults",
            "Reset all injected health, HTTP probe, and per-model overrides.",
        ))
        .item(operation::<InspectRecentArgs>(
            "inspect_recent",
            "Inspect recent HTTP requests, local inference observations, and OpenAI exchange lifecycle events.",
        ))
        .item(operation::<EmptyArgs>(
            "stats",
            "Return aggregated request, observation, exchange, and memory statistics.",
        ))
        .item(operation::<EmptyArgs>(
            "env_probe",
            "Report strictly allowlisted non-secret MESH_LLM_* and cache-path environment variables.",
        ))
        .item(operation::<ClearArgs>(
            "clear",
            "Clear bounded inspector ring buffers and reset counters for the selected source.",
        ))
        .build()
}

pub fn build_plugin(
    endpoint_base: String,
    state: AppState,
    dev_args: DevArgs,
    listener_slot: Arc<Mutex<Option<TcpListener>>>,
    server_handle_slot: Arc<Mutex<Option<AbortOnDrop>>>,
    observation_handle_slot: Arc<Mutex<Option<AbortOnDrop>>>,
) -> SimplePlugin {
    let plugin_name =
        std::env::var("MESH_LLM_PLUGIN_NAME").unwrap_or_else(|_| DEFAULT_PLUGIN_NAME.to_string());
    let manifest = build_manifest(&endpoint_base);
    let capabilities = manifest.capabilities.clone();
    let metadata = PluginMetadata::new(
        plugin_name.clone(),
        VERSION,
        plugin_server_info(
            plugin_name.clone(),
            VERSION,
            "Mesh-LLM Developer & Test Plugin",
            "Zero-GPU programmable fake OpenAI models, supervision fault injection, and live exchange/observation inspector.",
            Some(
                "Configure [[plugin]] name = \"mesh-dev\" (optionally with allow_inference_observation = true and [plugin.openai_exchange_grant]).",
            ),
        ),
    )
    .with_startup_policy(PluginStartupPolicy::Any)
    .with_capabilities(capabilities)
    .with_manifest(manifest);

    let negotiated_caps = Arc::new(Mutex::new(NegotiatedCaps {
        inference_observation: false,
        openai_exchange: false,
        plugin_status: false,
    }));

    let mut router = OperationRouter::new();
    register_operations(
        &mut router,
        &plugin_name,
        &endpoint_base,
        state.clone(),
        negotiated_caps.clone(),
        observation_handle_slot.clone(),
    );

    let init_startup_disabled = dev_args.startup_disabled.clone();
    let init_delay_ms = dev_args.init_delay_ms;
    let init_caps = negotiated_caps.clone();
    let init_state = state.clone();

    let health_state = state.clone();
    let health_endpoint = endpoint_base.clone();
    let health_server_slot = server_handle_slot.clone();

    let on_init_state = state.clone();
    let on_init_obs_slot = observation_handle_slot.clone();

    let exchange_inspector = state.inspector.clone();

    SimplePlugin::new(metadata)
        .with_operation_router(router)
        .on_initialize(move |request, _| {
            let startup_disabled = init_startup_disabled.clone();
            let init_caps = init_caps.clone();
            let init_state = init_state.clone();
            Box::pin(async move {
                if init_delay_ms > 0 {
                    tokio::select! {
                        biased;
                        _ = init_state.shutdown_notified() => {},
                        _ = tokio::time::sleep(Duration::from_millis(init_delay_ms)) => {},
                    }
                }
                if request.host_protocol_version != PROTOCOL_VERSION {
                    return Err(PluginError::startup_disabled(format!(
                        "incompatible Mesh plugin protocol: host={}, plugin={PROTOCOL_VERSION}",
                        request.host_protocol_version
                    )));
                }
                if let Some(reason) = startup_disabled {
                    return Err(PluginError::startup_disabled(reason));
                }
                let mut caps = init_caps.lock().await;
                caps.inference_observation = request
                    .host_capabilities
                    .iter()
                    .any(|c| c == host_capabilities::LOCAL_INFERENCE_OBSERVATION);
                caps.openai_exchange = request
                    .host_capabilities
                    .iter()
                    .any(|c| c == host_capabilities::OPENAI_EXCHANGE);
                caps.plugin_status = request
                    .host_capabilities
                    .iter()
                    .any(|c| c == host_capabilities::PLUGIN_STATUS);
                Ok(())
            })
        })
        .on_initialized(move |context| {
            let state = on_init_state.clone();
            let listener_slot = listener_slot.clone();
            let server_handle_slot = server_handle_slot.clone();
            let obs_slot = on_init_obs_slot.clone();
            let owned_ctx = context.owned();
            Box::pin(async move {
                // Ensure initial observation status is set before spawning tasks.
                if !owned_ctx.host_supports(host_capabilities::LOCAL_INFERENCE_OBSERVATION) {
                    state.inspector.set_observation_status(
                        "unsupported",
                        Some("host does not advertise local_inference_observation.v1".into()),
                    );
                } else {
                    state.inspector.set_observation_status("subscribing", None);
                }

                if let Some(listener) = listener_slot.lock().await.take() {
                    let serve_state = state.clone();
                    let handle = tokio::spawn(async move {
                        if let Err(err) = serve_with_state(listener, serve_state).await {
                            tracing::error!("mesh-dev HTTP server exited with error: {err:#}");
                        }
                    });
                    *server_handle_slot.lock().await = Some(AbortOnDrop::new(handle));
                }

                if owned_ctx.host_supports(host_capabilities::LOCAL_INFERENCE_OBSERVATION) {
                    let obs_handle = spawn_observation_task(owned_ctx, state.clone());
                    *obs_slot.lock().await = Some(AbortOnDrop::new(obs_handle));
                }

                Ok(())
            })
        })
        .with_health(move |_context| {
            let state = health_state.clone();
            let endpoint = health_endpoint.clone();
            let server_slot = health_server_slot.clone();
            Box::pin(async move {
                match state.faults.health_mode() {
                    HealthFaultMode::Ok => {}
                    HealthFaultMode::Error { detail } => {
                        anyhow::bail!("{detail}");
                    }
                    HealthFaultMode::Hang { hang_ms } => match hang_ms {
                        Some(ms) => {
                            tokio::select! {
                                biased;
                                _ = state.shutdown_notified() => {
                                    anyhow::bail!("mesh-dev shutting down during health hang");
                                }
                                _ = tokio::time::sleep(Duration::from_millis(ms)) => {}
                            }
                        }
                        None => {
                            state.shutdown_notified().await;
                            anyhow::bail!("mesh-dev shutting down during health hang");
                        }
                    },
                }

                if let Some(guard) = server_slot.lock().await.as_ref()
                    && guard.0.as_ref().is_some_and(JoinHandle::is_finished)
                {
                    anyhow::bail!("mesh-dev HTTP server task exited unexpectedly");
                }

                let rss_kb = read_self_rss_kb().unwrap_or(0);
                let loading_remaining_ms = state.faults.loading_remaining_ms();
                if loading_remaining_ms > 0 {
                    return Ok(format!(
                        "endpoint={endpoint} instance={} pid={} models=0 state=loading remaining_ms={loading_remaining_ms} rss_kb={rss_kb}",
                        state.faults.instance_id(),
                        state.faults.pid(),
                    ));
                }
                Ok(format!(
                    "endpoint={endpoint} instance={} pid={} models={} state=ready rss_kb={rss_kb}",
                    state.faults.instance_id(),
                    state.faults.pid(),
                    state.enabled_models.len()
                ))
            })
        })
        .with_openai_exchange_handler(move |_handler_name, event_json, _context| {
            let inspector = exchange_inspector.clone();
            Box::pin(async move {
                match OpenAiExchangeEvent::parse(&event_json) {
                    Ok(event) => {
                        inspector.record_exchange_event(&event);
                    }
                    Err(err) => {
                        tracing::warn!("failed to parse OpenAiExchangeEvent: {err}");
                    }
                }
                Ok(OpenAiExchangeDecision {
                    decision: OpenAiAdmissionDecision::Abstain,
                    ..Default::default()
                })
            })
        })
}

fn register_operations(
    router: &mut OperationRouter,
    plugin_name: &str,
    endpoint_base: &str,
    state: AppState,
    negotiated_caps: Arc<Mutex<NegotiatedCaps>>,
    observation_handle_slot: Arc<Mutex<Option<AbortOnDrop>>>,
) {
    // status
    {
        let state = state.clone();
        let plugin_name = plugin_name.to_string();
        let endpoint_base = endpoint_base.to_string();
        let caps = negotiated_caps.clone();
        router.add_json_default::<EmptyArgs, Value, _>(
            json_schema_operation::<EmptyArgs>(
                "status",
                "Return mesh-dev instance metadata, RSS, active faults, overrides, and negotiated capabilities.",
            ),
            move |_args, _ctx| {
                let state = state.clone();
                let plugin_name = plugin_name.clone();
                let endpoint_base = endpoint_base.clone();
                let caps = caps.clone();
                Box::pin(async move {
                    let negotiated = caps.lock().await.clone();
                    let loading_remaining_ms = state.faults.loading_remaining_ms();
                    let stats = state.inspector.stats();
                    Ok(json!({
                        "plugin_name": plugin_name,
                        "version": VERSION,
                        "pid": state.faults.pid(),
                        "instance_id": state.faults.instance_id(),
                        "uptime_ms": state.faults.uptime_ms(),
                        "self_rss_kb": read_self_rss_kb(),
                        "endpoint_url": endpoint_base,
                        "loading": loading_remaining_ms > 0,
                        "loading_remaining_ms": loading_remaining_ms,
                        "max_concurrency": state.max_concurrency,
                        "in_flight": stats.http.in_flight,
                        "models": state.enabled_models,
                        "faults": state.faults.snapshot(),
                        "negotiated_capabilities": negotiated,
                        "observation_status": stats.observation.status,
                    }))
                })
            },
        );
    }

    // inject_crash
    {
        let state = state.clone();
        router.add_json_default::<InjectCrashArgs, Value, _>(
            json_schema_operation::<InjectCrashArgs>(
                "inject_crash",
                "Trigger a controllable process exit or abort to test host supervision and restart backoff.",
            ),
            move |args, _ctx| {
                let state = state.clone();
                Box::pin(async move {
                    tracing::warn!(
                        exit_code = args.exit_code,
                        delay_ms = args.delay_ms,
                        immediate = args.immediate,
                        abort = args.abort,
                        "mesh-dev inject_crash triggered"
                    );
                    if args.immediate {
                        if args.abort {
                            std::process::abort();
                        } else {
                            std::process::exit(args.exit_code);
                        }
                    }
                    let exit_code = args.exit_code;
                    let delay_ms = args.delay_ms;
                    let abort = args.abort;
                    let instance_id = state.faults.instance_id().to_string();
                    let pid = state.faults.pid();
                    tokio::spawn(async move {
                        if delay_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        }
                        if abort {
                            std::process::abort();
                        } else {
                            std::process::exit(exit_code);
                        }
                    });
                    Ok(json!({
                        "scheduled": true,
                        "pid": pid,
                        "instance_id": instance_id,
                        "exit_code": exit_code,
                        "delay_ms": delay_ms,
                        "abort": abort,
                    }))
                })
            },
        );
    }

    // inject_health
    {
        let state = state.clone();
        router.add_json_default::<InjectHealthArgs, Value, _>(
            json_schema_operation::<InjectHealthArgs>(
                "inject_health",
                "Switch HealthRequest between ok, error, and hang modes.",
            ),
            move |args, _ctx| {
                let state = state.clone();
                Box::pin(async move {
                    let mode = match args.mode.to_ascii_lowercase().as_str() {
                        "ok" | "healthy" => HealthFaultMode::Ok,
                        "error" | "unhealthy" | "fail" => HealthFaultMode::Error {
                            detail: args
                                .detail
                                .unwrap_or_else(|| "injected health check failure".into()),
                        },
                        "hang" | "sleep" => HealthFaultMode::Hang {
                            hang_ms: args.hang_ms,
                        },
                        other => {
                            return Err(PluginError::invalid_request(format!(
                                "invalid inject_health mode '{other}'; expected ok, error, or hang"
                            )));
                        }
                    };
                    state.faults.set_health_mode(mode.clone());
                    Ok(json!({
                        "applied": true,
                        "health": mode,
                    }))
                })
            },
        );
    }

    // inject_http_fault
    {
        let state = state.clone();
        router.add_json_default::<InjectHttpFaultArgs, Value, _>(
            json_schema_operation::<InjectHttpFaultArgs>(
                "inject_http_fault",
                "Inject /v1/models endpoint faults (ok, models_503, models_empty, models_hang).",
            ),
            move |args, _ctx| {
                let state = state.clone();
                Box::pin(async move {
                    let mode = match args.mode.to_ascii_lowercase().as_str() {
                        "ok" => HttpFaultMode::Ok,
                        "models_503" | "503" => HttpFaultMode::Models503 {
                            detail: args
                                .detail
                                .unwrap_or_else(|| "injected /v1/models 503 failure".into()),
                        },
                        "models_empty" | "empty" => HttpFaultMode::ModelsEmpty,
                        "models_hang" | "hang" => HttpFaultMode::ModelsHang {
                            hang_ms: args.hang_ms,
                        },
                        other => {
                            return Err(PluginError::invalid_request(format!(
                                "invalid inject_http_fault mode '{other}'; expected ok, models_503, models_empty, or models_hang"
                            )));
                        }
                    };
                    state.faults.set_http_fault(mode.clone());
                    Ok(json!({
                        "applied": true,
                        "http": mode,
                    }))
                })
            },
        );
    }

    // set_model_override
    {
        let state = state.clone();
        router.add_json_default::<SetModelOverrideArgs, Value, _>(
            json_schema_operation::<SetModelOverrideArgs>(
                "set_model_override",
                "Override a dev/* model's behavior permanently or for the next N requests.",
            ),
            move |args, _ctx| {
                let state = state.clone();
                Box::pin(async move {
                    if args.model.trim().is_empty() {
                        return Err(PluginError::invalid_request("model must be non-empty"));
                    }
                    let ov = if args.behavior == OverrideBehavior::Default {
                        None
                    } else {
                        Some(ModelOverride {
                            behavior: args.behavior,
                            remaining: args.count,
                            delay_ms: args.delay_ms,
                            chunks: args.chunks,
                        })
                    };
                    state
                        .faults
                        .set_model_override(args.model.clone(), ov.clone());
                    Ok(json!({
                        "applied": true,
                        "model": args.model,
                        "override": ov,
                    }))
                })
            },
        );
    }

    // publish_status
    {
        let state = state.clone();
        router.add_json_default::<PublishStatusArgs, Value, _>(
            json_schema_operation::<PublishStatusArgs>(
                "publish_status",
                "Publish or withdraw a plugin_status.v1 snapshot in /api/status.",
            ),
            move |args, ctx| {
                let state = state.clone();
                Box::pin(async move {
                    if !ctx.host_supports(host_capabilities::PLUGIN_STATUS) {
                        return Err(PluginError::invalid_request(
                            "host does not advertise plugin_status.v1",
                        ));
                    }
                    if args.withdraw {
                        ctx.withdraw_status()
                            .await
                            .map_err(|e| PluginError::internal(e.to_string()))?;
                        return Ok(json!({ "withdrawn": true }));
                    }
                    let payload = args.payload.unwrap_or_else(|| {
                        json!({
                            "instance_id": state.faults.instance_id(),
                            "pid": state.faults.pid(),
                            "faults": state.faults.snapshot(),
                            "stats": state.inspector.stats(),
                        })
                    });
                    let resp = ctx
                        .publish_status(&payload, Duration::from_millis(args.fresh_for_ms))
                        .await
                        .map_err(|e| PluginError::internal(e.to_string()))?;
                    Ok(json!({
                        "published": true,
                        "received_at_unix_ms": resp.received_at_unix_ms,
                        "fresh_until_unix_ms": resp.fresh_until_unix_ms,
                    }))
                })
            },
        );
    }

    // resubscribe_observation
    {
        let state = state.clone();
        let obs_slot = observation_handle_slot.clone();
        router.add_json_default::<EmptyArgs, Value, _>(
            json_schema_operation::<EmptyArgs>(
                "resubscribe_observation",
                "Re-attempt subscribing to the host local inference observation feed.",
            ),
            move |_args, ctx| {
                let state = state.clone();
                let obs_slot = obs_slot.clone();
                let owned_ctx = ctx.owned();
                Box::pin(async move {
                    if !owned_ctx.host_supports(host_capabilities::LOCAL_INFERENCE_OBSERVATION) {
                        state.inspector.set_observation_status(
                            "unsupported",
                            Some("host does not advertise local_inference_observation.v1".into()),
                        );
                        return Ok(json!({
                            "subscribed": false,
                            "status": "unsupported",
                        }));
                    }
                    let (result_tx, result_rx) = oneshot::channel();
                    let handle =
                        spawn_observation_task_with_ack(owned_ctx, state.clone(), Some(result_tx));
                    *obs_slot.lock().await = Some(AbortOnDrop::new(handle));
                    match tokio::time::timeout(Duration::from_secs(5), result_rx).await {
                        Ok(Ok(Ok(()))) => Ok(json!({
                            "subscribed": true,
                            "status": "subscribed",
                        })),
                        Ok(Ok(Err(reason))) => Ok(json!({
                            "subscribed": false,
                            "status": "not_granted",
                            "detail": reason,
                        })),
                        _ => Ok(json!({
                            "subscribed": false,
                            "status": state.inspector.stats().observation.status,
                        })),
                    }
                })
            },
        );
    }

    // reset_faults
    {
        let state = state.clone();
        router.add_json_default::<EmptyArgs, Value, _>(
            json_schema_operation::<EmptyArgs>(
                "reset_faults",
                "Reset all injected health, HTTP probe, and per-model overrides.",
            ),
            move |_args, _ctx| {
                let state = state.clone();
                Box::pin(async move {
                    state.faults.reset_all();
                    Ok(json!({
                        "reset": true,
                        "faults": state.faults.snapshot(),
                    }))
                })
            },
        );
    }

    // inspect_recent
    {
        let state = state.clone();
        router.add_json_default::<InspectRecentArgs, Value, _>(
            json_schema_operation::<InspectRecentArgs>(
                "inspect_recent",
                "Inspect recent HTTP requests, local inference observations, and OpenAI exchange lifecycle events.",
            ),
            move |args, _ctx| {
                let state = state.clone();
                Box::pin(async move {
                    let resp =
                        state
                            .inspector
                            .inspect_recent(args.source, args.limit, args.model.as_deref());
                    serde_json::to_value(resp)
                        .map_err(|e| PluginError::internal(e.to_string()))
                })
            },
        );
    }

    // stats
    {
        let state = state.clone();
        router.add_json_default::<EmptyArgs, Value, _>(
            json_schema_operation::<EmptyArgs>(
                "stats",
                "Return aggregated request, observation, exchange, and memory statistics.",
            ),
            move |_args, _ctx| {
                let state = state.clone();
                Box::pin(async move {
                    Ok(json!({
                        "instance_id": state.faults.instance_id(),
                        "pid": state.faults.pid(),
                        "uptime_ms": state.faults.uptime_ms(),
                        "self_rss_kb": read_self_rss_kb(),
                        "inspector": state.inspector.stats(),
                    }))
                })
            },
        );
    }

    // env_probe
    {
        router.add_json_default::<EmptyArgs, Value, _>(
            json_schema_operation::<EmptyArgs>(
                "env_probe",
                "Report strictly allowlisted non-secret MESH_LLM_* and cache-path environment variables.",
            ),
            move |_args, _ctx| {
                Box::pin(async move {
                    serde_json::to_value(probe_env())
                        .map_err(|e| PluginError::internal(e.to_string()))
                })
            },
        );
    }

    // clear
    {
        let state = state.clone();
        router.add_json_default::<ClearArgs, Value, _>(
            json_schema_operation::<ClearArgs>(
                "clear",
                "Clear bounded inspector ring buffers and reset counters for the selected source.",
            ),
            move |args, _ctx| {
                let state = state.clone();
                Box::pin(async move {
                    state.inspector.clear(args.source);
                    Ok(json!({
                        "cleared": true,
                        "source": args.source,
                    }))
                })
            },
        );
    }
}

fn spawn_observation_task(context: PluginContext<'static>, state: AppState) -> JoinHandle<()> {
    spawn_observation_task_with_ack(context, state, None)
}

fn spawn_observation_task_with_ack(
    mut context: PluginContext<'static>,
    state: AppState,
    ack: Option<oneshot::Sender<Result<(), String>>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut feed = match context.subscribe_inference_observation().await {
            Ok(feed) => {
                state.inspector.apply_observation_snapshot(feed.snapshot());
                if let Some(tx) = ack {
                    let _ = tx.send(Ok(()));
                }
                feed
            }
            Err(err) => {
                let msg = err.to_string();
                tracing::info!("local inference observation not active: {msg}");
                state
                    .inspector
                    .set_observation_status("not_granted", Some(msg.clone()));
                if let Some(tx) = ack {
                    let _ = tx.send(Err(msg));
                }
                return;
            }
        };

        loop {
            tokio::select! {
                biased;
                _ = state.shutdown_notified() => {
                    state.inspector.set_observation_status("closed", Some("shutdown".into()));
                    return;
                }
                item = feed.next() => match item {
                    Some(InferenceObservationItem::Batch(batch)) => {
                        state.inspector.apply_observation_batch(batch);
                    }
                    Some(InferenceObservationItem::Overflowed) => {
                        state.inspector.record_observation_overflow();
                        match context.subscribe_inference_observation().await {
                            Ok(next) => {
                                state.inspector.apply_observation_snapshot(next.snapshot());
                                feed = next;
                            }
                            Err(err) => {
                                state.inspector.set_observation_status("closed", Some(err.to_string()));
                                return;
                            }
                        }
                    }
                    None => {
                        state.inspector.set_observation_status("closed", Some("feed closed".into()));
                        return;
                    }
                }
            }
        }
    })
}

pub fn build_app_state(args: &DevArgs) -> AppState {
    let enabled_models = args.resolved_models();
    let allow_dynamic_models = args.models.is_empty();
    let defaults = ModelDefaults {
        slow_ttft_ms: args.slow_ttft_ms,
        slow_sse_interval_ms: args.slow_sse_interval_ms,
        token_delay_ms: args.token_delay_ms,
        mid_stream_chunks: args.mid_stream_chunks,
        gpu_probe: args.gpu_probe,
    };
    let faults = FaultController::new(args.unhealthy_on_start.clone(), args.total_startup_delay());
    let inspector = Inspector::new(args.ring_capacity);
    AppState::new(
        enabled_models,
        allow_dynamic_models,
        defaults,
        args.max_concurrency,
        faults,
        inspector,
    )
}

pub async fn run_mesh_plugin(dev_args: DevArgs) -> Result<()> {
    if let Some(code) = dev_args.exit_on_start {
        tracing::warn!(exit_code = code, "exiting on start as requested");
        std::process::exit(code);
    }

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind ephemeral loopback port for mesh-dev plugin")?;
    let port = listener.local_addr()?.port();
    let endpoint_base = format!("http://127.0.0.1:{port}/v1");

    let state = build_app_state(&dev_args);
    let listener_slot = Arc::new(Mutex::new(Some(listener)));
    let server_handle_slot = Arc::new(Mutex::new(None));
    let observation_handle_slot = Arc::new(Mutex::new(None));

    let plugin = build_plugin(
        endpoint_base,
        state.clone(),
        dev_args,
        listener_slot,
        server_handle_slot.clone(),
        observation_handle_slot.clone(),
    );

    let result = tokio::select! {
        res = PluginRuntime::run(plugin) => res,
        _ = shutdown_signal() => Ok(()),
    };

    // Signal shutdown immediately so in-flight streams and background tasks exit promptly.
    state.request_shutdown();
    if let Some(mut obs_guard) = observation_handle_slot.lock().await.take()
        && let Some(handle) = obs_guard.take()
    {
        handle.abort();
    }
    if let Some(mut server_guard) = server_handle_slot.lock().await.take()
        && let Some(handle) = server_guard.take()
    {
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
    }
    result
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        if let Ok(mut term) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            tokio::select! {
                _ = term.recv() => {},
                _ = tokio::signal::ctrl_c() => {},
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_plugin::Plugin;

    #[test]
    fn manifest_declares_managed_openai_endpoint_and_optional_observers() {
        let state = build_app_state(&DevArgs::default());
        let plugin = build_plugin(
            "http://127.0.0.1:1/v1".to_string(),
            state,
            DevArgs::default(),
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
        );
        let manifest = plugin.manifest().expect("manifest");
        assert_eq!(manifest.endpoints.len(), 1);
        assert!(manifest.endpoints[0].managed_by_plugin);
        assert_eq!(
            manifest.endpoints[0].protocol.as_deref(),
            Some("openai_compatible")
        );
        assert!(manifest.inference_observation.is_some());
        assert!(!manifest.inference_observation.as_ref().unwrap().required);
        assert!(manifest.openai_exchange_hook.is_some());
        assert!(!manifest.openai_exchange_hook.as_ref().unwrap().required);
        assert!(manifest.plugin_status.is_some());
        assert!(!manifest.plugin_status.as_ref().unwrap().required);
        assert!(
            manifest
                .operations
                .iter()
                .any(|op| op.name == "inject_crash")
        );
        assert!(
            manifest
                .operations
                .iter()
                .any(|op| op.name == "inspect_recent")
        );
    }
}
