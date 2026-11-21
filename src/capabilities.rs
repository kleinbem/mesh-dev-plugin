//! Extended `mesh-llm 0.80.0` plugin capability implementations:
//! - Config schema (`[plugin.settings]`)
//! - MCP Resources, Resource Templates, Prompts, Completions, and SEP-2663 Tasks
//! - Virtual Model (`dev/virtual-echo`) orchestrating nested `PluginContext::infer`
//! - Request Transform (`request_transform.v1`)
//! - Negotiated Side Streams (`StreamMode::Http1` stapled streaming + `sha256-v1` OpenAI exchange wire receipts)

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use mesh_llm_plugin::request_transform::{RequestTransformInput, RequestTransformOutput};
use mesh_llm_plugin::{
    CompletionRouter, HostInferenceRequest, PluginConfigSchemaBuilder, PluginError, PluginResult,
    PromptRouter, ResourceRouter, TaskRouter, TaskStore, VirtualModelInvocation,
    VirtualModelResponse, VirtualModelRouter, bind_side_stream, complete_result, config_boolean,
    config_enum, config_integer, config_schema, config_setting, config_string, get_prompt_result,
    get_task_result, operation_with_schema, prompt, prompt_argument, proto, read_resource_result,
    resource_template, task, text_resource,
};
use rmcp::model::{
    ContentBlock, DetailedTask, PromptMessage, ResourceContents, Role, TaskPayload, TaskStatus,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Mutex;

use crate::faults::TransformMode;
use crate::inspector::{InspectSource, unix_now_ms};
use crate::mem_probe::read_self_rss_kb;
use crate::models::MODEL_ECHO;
use crate::server::AppState;

pub const VIRTUAL_MODEL_ID: &str = "dev/virtual-echo";
pub const VIRTUAL_MODEL_HANDLER: &str = "virtual_echo";
pub const LOCAL_BUS_CHANNEL: &str = "mesh-dev/bus";
pub const RFC8032_DEMO_KEY: [u8; 32] = [
    0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07, 0x3a,
    0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07, 0x51, 0x1a,
];

static STREAM_TOKEN_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct DevTaskState {
    pub label: String,
    pub payload: TaskPayload,
}

impl Default for DevTaskState {
    fn default() -> Self {
        Self {
            label: String::new(),
            payload: TaskPayload::Working,
        }
    }
}

pub type SharedTaskStore = Arc<Mutex<TaskStore<DevTaskState>>>;

pub fn build_config_schema(plugin_name: &str) -> PluginConfigSchemaBuilder {
    config_schema(plugin_name)
        .setting(
            config_setting("greeting", config_string())
                .description("Optional greeting marker echoed in status and stapled HTTP routes.")
                .default_value(&json!("hello-mesh-dev")),
        )
        .setting(
            config_setting("default_delay_ms", config_integer())
                .description("Optional default delay in milliseconds for diagnostics.")
                .default_value(&json!(0)),
        )
        .setting(
            config_setting("strict_mode", config_boolean())
                .description("Boolean setting exercising typed schema validation.")
                .default_value(&json!(false)),
        )
        .setting(
            config_setting("profile", config_enum(["default", "slow", "faulty"]))
                .description("Enum setting exercising allowed-values validation.")
                .default_value(&json!("default")),
        )
}

pub fn build_resource_router(state: AppState) -> ResourceRouter {
    let mut router = ResourceRouter::new();

    // Exact resource: mesh-dev://status
    {
        let state = state.clone();
        let mut res = text_resource("mesh-dev://status", "mesh-dev-status");
        res.description = Some("Live mesh-dev plugin status JSON".into());
        res.mime_type = Some("application/json".into());
        router.add_exact(res, move |req, _ctx| {
            let state = state.clone();
            Box::pin(async move {
                let body = json!({
                    "instance_id": state.faults.instance_id(),
                    "pid": state.faults.pid(),
                    "uptime_ms": state.faults.uptime_ms(),
                    "self_rss_kb": read_self_rss_kb(),
                    "models": state.enabled_models,
                    "faults": state.faults.snapshot(),
                    "received_settings": state.faults.received_settings(),
                })
                .to_string();
                Ok(read_resource_result(vec![ResourceContents::text(
                    body, req.uri,
                )]))
            })
        });
    }

    // Exact resource: mesh-dev://models
    {
        let state = state.clone();
        let mut res = text_resource("mesh-dev://models", "mesh-dev-models");
        res.description = Some("Advertised dev/* models JSON".into());
        res.mime_type = Some("application/json".into());
        router.add_exact(res, move |req, _ctx| {
            let state = state.clone();
            Box::pin(async move {
                let body = json!({ "models": state.enabled_models }).to_string();
                Ok(read_resource_result(vec![ResourceContents::text(
                    body, req.uri,
                )]))
            })
        });
    }

    // Prefix template resource: mesh-dev://inspect/{source}
    {
        let mut tpl = resource_template("mesh-dev://inspect/{source}", "mesh-dev-inspect");
        tpl.description = Some(
            "Inspect bounded ring buffers by source (all|http|observation|exchange|events)".into(),
        );
        tpl.mime_type = Some("application/json".into());
        router.add_prefix_template(tpl, "mesh-dev://inspect/", move |req, _ctx| {
            let state = state.clone();
            Box::pin(async move {
                let suffix = req.uri.strip_prefix("mesh-dev://inspect/").unwrap_or("all");
                let source = match suffix {
                    "http" => InspectSource::Http,
                    "observation" => InspectSource::Observation,
                    "exchange" => InspectSource::Exchange,
                    "events" => InspectSource::Events,
                    _ => InspectSource::All,
                };
                let slice = state.inspector.inspect_recent(source, 20, None);
                let body = serde_json::to_string(&slice)
                    .map_err(|e| PluginError::internal(e.to_string()))?;
                Ok(read_resource_result(vec![ResourceContents::text(
                    body, req.uri,
                )]))
            })
        });
    }

    router
}

pub fn build_prompt_router() -> PromptRouter {
    let mut router = PromptRouter::new();
    let def = prompt(
        "diagnose_model",
        "Generate a diagnostic prompt for testing a specific mesh-dev model.",
        Some(vec![prompt_argument(
            "model",
            "Target dev/* model ID",
            true,
        )]),
    );
    router.add(def, |req, _ctx| {
        Box::pin(async move {
            let model = req
                .arguments
                .as_ref()
                .and_then(|args| args.get("model"))
                .and_then(Value::as_str)
                .unwrap_or(MODEL_ECHO);
            let msg = PromptMessage::new(
                Role::User,
                ContentBlock::text(format!(
                    "Run diagnostic verification against mesh-dev model `{model}`."
                )),
            );
            Ok(get_prompt_result(vec![msg]))
        })
    });
    router
}

pub fn build_completion_router(state: AppState) -> CompletionRouter {
    let mut router = CompletionRouter::new();
    router.add_prompt_argument("diagnose_model", "model", move |req, _ctx| {
        let models = state.enabled_models.clone();
        Box::pin(async move {
            let prefix = req.argument.value;
            let matches: Vec<String> = models
                .into_iter()
                .filter(|m| m.starts_with(&prefix))
                .collect();
            complete_result(matches)
        })
    });
    router
}

pub fn build_task_router(store: SharedTaskStore) -> TaskRouter {
    let get_store = store.clone();
    let update_store = store.clone();
    let cancel_store = store;

    TaskRouter::new()
        .with_get(move |req, _ctx| {
            let store = get_store.clone();
            Box::pin(async move {
                let guard = store.lock().await;
                let record = guard.get(&req.task_id)?;
                let detailed =
                    DetailedTask::new(record.task.clone(), record.payload.payload.clone());
                Ok(get_task_result(detailed))
            })
        })
        .with_update(move |req, _ctx| {
            let store = update_store.clone();
            Box::pin(async move {
                let mut guard = store.lock().await;
                let record = guard.get_mut(&req.task_id)?;
                record.task.last_updated_at = unix_now_ms().to_string();
                let mut result_obj = Map::new();
                result_obj.insert("updated".into(), json!(true));
                result_obj.insert("label".into(), json!(record.payload.label));
                record.task.status = TaskStatus::Completed;
                record.payload.payload = TaskPayload::Completed { result: result_obj };
                Ok(())
            })
        })
        .with_cancel(move |req, _ctx| {
            let store = cancel_store.clone();
            Box::pin(async move {
                let mut guard = store.lock().await;
                let record = guard.get_mut(&req.task_id)?;
                record.task.last_updated_at = unix_now_ms().to_string();
                record.task.status = TaskStatus::Cancelled;
                record.payload.payload = TaskPayload::Cancelled;
                Ok(())
            })
        })
}

pub const MAX_TRACKED_TASKS: usize = 256;
pub const MAX_TASK_LABEL_LEN: usize = 256;
pub const MAX_SIDE_STREAM_BYTES: u64 = 16 * 1024 * 1024;

/// Constant-time byte-slice comparison to avoid timing side channels on auth tokens.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(unix)]
fn restrict_socket_permissions(path: &str) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn restrict_socket_permissions(_path: &str) {}

struct SocketCleanupGuard(String);

impl Drop for SocketCleanupGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        let _ = std::fs::remove_file(&self.0);
        #[cfg(not(unix))]
        let _ = &self.0;
    }
}

pub async fn create_tracked_task(store: &SharedTaskStore, label: String) -> PluginResult<Value> {
    if label.len() > MAX_TASK_LABEL_LEN {
        return Err(PluginError::invalid_params(format!(
            "task label exceeds maximum length of {MAX_TASK_LABEL_LEN} bytes"
        )));
    }
    let mut guard = store.lock().await;
    if guard.list().len() >= MAX_TRACKED_TASKS {
        return Err(PluginError::invalid_request(format!(
            "task store capacity ({MAX_TRACKED_TASKS}) reached"
        )));
    }
    let id = format!(
        "mesh-dev-task-{}",
        STREAM_TOKEN_SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let now = unix_now_ms().to_string();
    let t = task(id.clone(), TaskStatus::Working, now.clone(), now);
    guard.insert(
        t,
        DevTaskState {
            label: label.clone(),
            payload: TaskPayload::Working,
        },
    );
    Ok(json!({
        "task_id": id,
        "status": "working",
        "label": label,
    }))
}

pub fn build_virtual_model_router(state: AppState) -> VirtualModelRouter {
    let mut router = VirtualModelRouter::new();
    let op = operation_with_schema(
        VIRTUAL_MODEL_HANDLER,
        "Orchestrates nested host inference against dev/echo (or echoes directly with mode=direct).",
        Map::new(),
    );
    router.add_json::<VirtualModelInvocation, VirtualModelResponse, _>(op, move |inv, ctx| {
        let state = state.clone();
        Box::pin(async move {
            let prompt_text = inv
                .request
                .get("messages")
                .and_then(Value::as_array)
                .and_then(|arr| arr.last())
                .and_then(|m| m.get("content"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();

            let direct_mode = prompt_text.contains("mode=direct");
            let candidate_ids: Vec<String> = inv
                .candidates
                .iter()
                .map(|c| c.model_id.clone())
                .collect();

            let (nested_status, nested_served_by, nested_content) = if direct_mode {
                (None, None, json!({"direct": true, "prompt": prompt_text}))
            } else {
                let target_model = inv
                    .candidates
                    .iter()
                    .find(|c| c.model_id == MODEL_ECHO)
                    .map(|c| c.model_id.clone())
                    .unwrap_or_else(|| MODEL_ECHO.to_string());
                let nested_req = HostInferenceRequest {
                    model_id: target_model.clone(),
                    single_model_only: true,
                    target_node_id: None,
                    request: json!({
                        "model": target_model,
                        "messages": [{"role": "user", "content": format!("virtual->nested: {prompt_text}")}],
                        "stream": false,
                    }),
                    timeout_ms: Some(15_000),
                };
                match ctx.infer(nested_req).await {
                    Ok(resp) => {
                        let content = resp.body["choices"][0]["message"]["content"].clone();
                        (Some(resp.status_code), resp.served_by, content)
                    }
                    Err(err) => (
                        Some(502),
                        None,
                        json!({"nested_error": err.to_string()}),
                    ),
                }
            };

            let summary = json!({
                "virtual_model": VIRTUAL_MODEL_ID,
                "instance_id": state.faults.instance_id(),
                "candidates_count": candidate_ids.len(),
                "response_adapter": inv.response_adapter,
                "prompt": prompt_text,
                "nested_status": nested_status,
                "nested_served_by": nested_served_by,
                "nested_content": nested_content,
            })
            .to_string();

            let created = unix_now_ms() / 1000;
            Ok(VirtualModelResponse {
                status_code: 200,
                body: json!({
                    "id": format!("chatcmpl-vdev-{created}"),
                    "object": "chat.completion",
                    "created": created,
                    "model": VIRTUAL_MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": summary,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 16,
                        "total_tokens": 20,
                    }
                }),
                headers: vec![("x-mesh-dev-virtual".into(), "true".into())],
                event_stream: false,
            })
        })
    });
    router
}

pub fn apply_request_transform(
    state: &AppState,
    input: RequestTransformInput,
) -> PluginResult<RequestTransformOutput> {
    let policy = state.faults.transform_policy();
    let model = input
        .body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    state.inspector.record_runtime_event(
        "request_transform",
        format!("{:?}", policy.mode),
        json!({
            "path": input.path,
            "client_path": input.client_path,
            "model": model,
        }),
    );

    match policy.mode {
        TransformMode::Unchanged => Ok(RequestTransformOutput::unchanged()),
        TransformMode::TagSystem => {
            let mut body = input.body;
            if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
                for msg in messages.iter_mut() {
                    if msg.get("role").and_then(Value::as_str) == Some("user")
                        && let Some(content) = msg.get("content").and_then(Value::as_str)
                    {
                        msg["content"] = json!(format!("{} {}", policy.tag, content));
                        break;
                    }
                }
            }
            Ok(RequestTransformOutput::replaced(body))
        }
        TransformMode::Reject => Err(PluginError::invalid_request(policy.reason)),
        TransformMode::CorruptModel => {
            let mut body = input.body;
            body["model"] = json!("illegal/mutated-model");
            Ok(RequestTransformOutput::replaced(body))
        }
    }
}

/// Handle both:
/// 1. `StreamMode::Http1` — streamed stapled HTTP binding (`GET /api/plugins/mesh-dev/http/stream`).
/// 2. `StreamMode::RawBytes` (`sha256-v1`) — OpenAI exchange wire-byte commitment side streams.
pub async fn handle_open_stream(
    request: proto::OpenStreamRequest,
    state: AppState,
) -> PluginResult<Option<proto::OpenStreamResponse>> {
    let seq = STREAM_TOKEN_SEQ.fetch_add(1, Ordering::Relaxed);
    let short_id = format!("{}-{seq}", state.faults.pid());

    if request.mode == proto::StreamMode::Http1 as i32 {
        let listener = bind_side_stream("mdh", &short_id)
            .await
            .map_err(|e| PluginError::internal(e.to_string()))?;
        let endpoint = listener.endpoint();
        restrict_socket_permissions(&endpoint);
        let response = listener.open_stream_response(&request);
        let instance_id = state.faults.instance_id().to_string();
        tokio::spawn(async move {
            let _cleanup = SocketCleanupGuard(endpoint);
            let _ = tokio::time::timeout(Duration::from_secs(15), async move {
                let stream = listener.accept().await?;
                let (mut read, mut write) = stream.into_split();
                let mut req_buf = vec![0u8; 4096];
                let _ = read.read(&mut req_buf).await?;
                let body = format!(
                    "data: {{\"event\":\"stapled_stream\",\"chunk\":1,\"instance\":\"{instance_id}\"}}\n\n\
                     data: {{\"event\":\"stapled_stream\",\"chunk\":2,\"instance\":\"{instance_id}\"}}\n\n\
                     data: [DONE]\n\n"
                );
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                write.write_all(head.as_bytes()).await?;
                write.write_all(body.as_bytes()).await?;
                write.shutdown().await?;
                anyhow::Ok(())
            })
            .await;
        });
        return Ok(Some(response));
    }

    if request.mode == proto::StreamMode::RawBytes as i32 && request.bidirectional {
        let metadata: Value =
            serde_json::from_str(request.metadata_json.as_deref().unwrap_or("{}"))
                .map_err(|e| PluginError::invalid_request(e.to_string()))?;
        if metadata["receipt_protocol"] != "sha256-v1" {
            return Err(PluginError::invalid_request(
                "unsupported raw side-stream protocol",
            ));
        }
        let kind = metadata["kind"].as_str().unwrap_or("unknown").to_string();
        let exchange_id = metadata["exchange_id"]
            .as_str()
            .or(request.correlation_id.as_deref())
            .unwrap_or("unknown")
            .to_string();

        let listener = bind_side_stream("mdo", &short_id)
            .await
            .map_err(|e| PluginError::internal(e.to_string()))?;
        let endpoint = listener.endpoint();
        restrict_socket_permissions(&endpoint);
        // Cryptographically random 36-byte RFC 4122 UUIDv4 auth token for sha256-v1 handshake.
        let token = uuid::Uuid::new_v4().to_string();
        let mut response = listener.open_stream_response(&request);
        response.token = Some(token.clone());

        let inspector = state.inspector.clone();
        tokio::spawn(async move {
            let _cleanup = SocketCleanupGuard(endpoint);
            let _ = tokio::time::timeout(Duration::from_secs(30), async move {
                let stream = listener.accept().await?;
                let (mut read, mut write) = stream.into_split();
                let mut auth = vec![0u8; token.len()];
                read.read_exact(&mut auth).await?;
                anyhow::ensure!(
                    constant_time_eq(&auth, token.as_bytes()),
                    "side stream token mismatch"
                );
                let mut hasher = Sha256::new();
                let mut count = 0u64;
                let mut buf = [0u8; 8192];
                loop {
                    let n = read.read(&mut buf).await?;
                    if n == 0 {
                        break;
                    }
                    count = count.saturating_add(n as u64);
                    anyhow::ensure!(
                        count <= MAX_SIDE_STREAM_BYTES,
                        "side stream exceeded max body limit"
                    );
                    hasher.update(&buf[..n]);
                }
                let sha256 = hex::encode(hasher.finalize());
                inspector.record_side_stream_receipt(&exchange_id, kind, sha256.clone(), count);
                let receipt = json!({"sha256": sha256, "byte_count": count}).to_string() + "\n";
                write.write_all(receipt.as_bytes()).await?;
                write.shutdown().await?;
                anyhow::Ok(())
            })
            .await;
        });
        return Ok(Some(response));
    }

    Ok(None)
}
