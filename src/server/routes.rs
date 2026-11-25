//! Axum HTTP handlers for `/v1/models`, `/v1/chat/completions`, `/v1/completions`,
//! `/v1/responses`, and `/health`.
//!
//! Invariants:
//! - Concurrency permits are acquired synchronously *before* returning `200 OK` SSE headers,
//!   so full queues return `503 Service Unavailable` (never `200 OK` + `[DONE]`).
//! - Streaming responses hold a `CancelOnDrop` guard that records `ClientDisconnected` and
//!   releases the permit immediately when a client drops TCP.
//! - Healthy SSE streams strictly terminate with `data: [DONE]\n\n` as the final frame.

use std::collections::BTreeMap;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE, RETRY_AFTER};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::Stream;
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::env_probe::probe_env;
use crate::faults::HttpFaultMode;
use crate::inspector::{HttpOutcome, HttpRequestRecord, Inspector, unix_now_ms};
use crate::mem_probe::{MemorySnapshot, probe_memory};
use crate::models::{MODEL_ECHO, ResolvedBehavior, resolve_behavior, tool_call_edge};
use crate::server::AppState;
use crate::server::openai::{
    ApiSurface, ChatCompletionRequest, CompletionRequest, ResponsesRequest, SSE_DONE, TokenUsage,
    capture_headers, chat_chunk, chat_completion_json, chat_usage_chunk, content_value_to_string,
    estimate_tokens, extract_chat_prompt, format_sse_json, openai_error_json, responses_json,
    text_completion_chunk, text_completion_json,
};

struct NormalizedRequest {
    surface: ApiSurface,
    model: String,
    prompt_text: String,
    message_count: usize,
    stream: bool,
    max_tokens: Option<u32>,
    include_usage: bool,
}

/// Drop guard attached to every streaming HTTP response body.
/// Ensures client TCP disconnects release the semaphore permit immediately and record
/// `HttpOutcome::ClientDisconnected` if the stream did not reach a terminal completion state.
struct CancelOnDrop {
    inspector: Inspector,
    seq: u64,
    endpoint: &'static str,
    model: String,
    headers: BTreeMap<String, String>,
    prompt_preview: Option<String>,
    override_applied: Option<String>,
    started_at: Instant,
    timestamp_unix_ms: u64,
    ttft_ms: Option<u64>,
    tokens_emitted: u64,
    finished: bool,
    _permit: OwnedSemaphorePermit,
}

impl CancelOnDrop {
    fn note_first_token(&mut self) {
        if self.ttft_ms.is_none() {
            self.ttft_ms = Some(self.started_at.elapsed().as_millis() as u64);
        }
    }

    fn note_tokens(&mut self, count: u64) {
        self.tokens_emitted += count;
    }

    fn finish(mut self, outcome: HttpOutcome) {
        self.finished = true;
        self.emit(outcome);
    }

    fn emit(&self, outcome: HttpOutcome) {
        self.inspector.finish_http_request(HttpRequestRecord {
            seq: self.seq,
            timestamp_unix_ms: self.timestamp_unix_ms,
            endpoint: self.endpoint.to_string(),
            model: self.model.clone(),
            stream: true,
            headers: self.headers.clone(),
            http_status: 200,
            ttft_ms: self.ttft_ms,
            duration_ms: self.started_at.elapsed().as_millis() as u64,
            tokens_emitted: self.tokens_emitted,
            outcome,
            prompt_preview: self.prompt_preview.clone(),
            override_applied: self.override_applied.clone(),
        });
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !self.finished {
            self.emit(HttpOutcome::ClientDisconnected);
        }
    }
}

/// Wrapper stream that drops `ReceiverStream` (closing the channel so the producer exits
/// promptly) when the HTTP client disconnects.
struct GuardedByteStream {
    inner: ReceiverStream<io::Result<Bytes>>,
}

impl Stream for GuardedByteStream {
    type Item = io::Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let loading_remaining_ms = state.faults.loading_remaining_ms();
    let loading = loading_remaining_ms > 0;
    (
        StatusCode::OK,
        Json(json!({
            "status": if loading { "loading" } else { "ok" },
            "loading": loading,
            "loading_remaining_ms": loading_remaining_ms,
            "plugin": "mesh-dev",
            "instance_id": state.faults.instance_id(),
            "pid": state.faults.pid(),
            "models": if loading { 0 } else { state.enabled_models.len() },
            "max_concurrency": state.max_concurrency,
        })),
    )
}

pub async fn list_models(State(state): State<AppState>) -> Response {
    let loading_remaining_ms = state.faults.loading_remaining_ms();
    if loading_remaining_ms > 0 {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!("mesh-dev loading ({loading_remaining_ms}ms remaining)"),
            "server_error",
            "loading",
            Some(1),
        );
    }

    match state.faults.http_fault() {
        HttpFaultMode::Ok => {}
        HttpFaultMode::Models503 { detail } => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                &detail,
                "server_error",
                "models_probe_fault",
                Some(1),
            );
        }
        HttpFaultMode::ModelsEmpty => {
            return (
                StatusCode::OK,
                Json(json!({
                    "object": "list",
                    "data": [],
                })),
            )
                .into_response();
        }
        HttpFaultMode::ModelsHang { hang_ms } => {
            let _ = sleep_interruptible(&state, hang_ms).await;
            if state.is_shutting_down() {
                return error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "mesh-dev shutting down",
                    "server_error",
                    "shutting_down",
                    None,
                );
            }
        }
    }

    let created = unix_now_ms() / 1000;
    let data: Vec<Value> = state
        .enabled_models
        .iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "created": created,
                "owned_by": "mesh-dev",
                "availability": "ready",
                "workload_class": "causal_generation",
            })
        })
        .collect();

    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": data,
        })),
    )
        .into_response()
}

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let prompt_text = extract_chat_prompt(&req.messages);
    let model = if req.model.trim().is_empty() {
        state
            .enabled_models
            .first()
            .cloned()
            .unwrap_or_else(|| MODEL_ECHO.to_string())
    } else {
        req.model
    };
    let norm = NormalizedRequest {
        surface: ApiSurface::ChatCompletions,
        model,
        prompt_text,
        message_count: req.messages.len(),
        stream: req.stream,
        max_tokens: req.max_completion_tokens.or(req.max_tokens),
        include_usage: req.stream_options.as_ref().is_some_and(|o| o.include_usage),
    };
    handle_inference(state, headers, norm).await
}

pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let prompt_text = content_value_to_string(&req.prompt);
    let model = if req.model.trim().is_empty() {
        state
            .enabled_models
            .first()
            .cloned()
            .unwrap_or_else(|| MODEL_ECHO.to_string())
    } else {
        req.model
    };
    let norm = NormalizedRequest {
        surface: ApiSurface::Completions,
        model,
        prompt_text,
        message_count: 1,
        stream: req.stream,
        max_tokens: req.max_tokens,
        include_usage: req.stream_options.as_ref().is_some_and(|o| o.include_usage),
    };
    handle_inference(state, headers, norm).await
}

pub async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    let prompt_text = content_value_to_string(&req.input);
    let model = if req.model.trim().is_empty() {
        state
            .enabled_models
            .first()
            .cloned()
            .unwrap_or_else(|| MODEL_ECHO.to_string())
    } else {
        req.model
    };
    let norm = NormalizedRequest {
        surface: ApiSurface::Responses,
        model,
        prompt_text,
        message_count: 1,
        stream: req.stream,
        max_tokens: req.max_output_tokens,
        include_usage: false,
    };
    handle_inference(state, headers, norm).await
}

async fn handle_inference(state: AppState, headers: HeaderMap, req: NormalizedRequest) -> Response {
    let started_at = Instant::now();
    let timestamp_unix_ms = unix_now_ms();
    let seq = state.inspector.begin_http_request();
    let captured_headers = capture_headers(&headers);
    let prompt_preview = (!req.prompt_text.is_empty()).then(|| truncate_preview(&req.prompt_text));

    // Acquire concurrency permit synchronously BEFORE returning any `200 OK` SSE headers.
    let permit = match state.permits.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            state.inspector.finish_http_request(HttpRequestRecord {
                seq,
                timestamp_unix_ms,
                endpoint: req.surface.path().to_string(),
                model: req.model.clone(),
                stream: req.stream,
                headers: captured_headers,
                http_status: 503,
                ttft_ms: None,
                duration_ms: started_at.elapsed().as_millis() as u64,
                tokens_emitted: 0,
                outcome: HttpOutcome::Backpressure503,
                prompt_preview,
                override_applied: None,
            });
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "mesh-dev concurrency limit reached",
                "server_error",
                "queue_full",
                Some(1),
            );
        }
    };

    if !state.is_model_supported(&req.model) {
        drop(permit);
        state.inspector.finish_http_request(HttpRequestRecord {
            seq,
            timestamp_unix_ms,
            endpoint: req.surface.path().to_string(),
            model: req.model.clone(),
            stream: req.stream,
            headers: captured_headers,
            http_status: 404,
            ttft_ms: None,
            duration_ms: started_at.elapsed().as_millis() as u64,
            tokens_emitted: 0,
            outcome: HttpOutcome::InjectedError,
            prompt_preview,
            override_applied: None,
        });
        return error_response(
            StatusCode::NOT_FOUND,
            &format!("unknown mesh-dev model '{}'", req.model),
            "invalid_request_error",
            "model_not_found",
            None,
        );
    }

    let active_override = state.faults.take_model_override(&req.model);
    let override_applied = active_override
        .as_ref()
        .map(|ov| format!("{:?}", ov.behavior));
    let behavior = resolve_behavior(
        &req.model,
        &req.prompt_text,
        req.max_tokens,
        &headers,
        &state.defaults,
        active_override.as_ref(),
    );

    // Handle immediate HTTP error behaviors (503 / 429).
    if let ResolvedBehavior::HttpError {
        status,
        err_type,
        code,
        message,
        retry_after,
    } = behavior
    {
        drop(permit);
        state.inspector.finish_http_request(HttpRequestRecord {
            seq,
            timestamp_unix_ms,
            endpoint: req.surface.path().to_string(),
            model: req.model.clone(),
            stream: req.stream,
            headers: captured_headers,
            http_status: status,
            ttft_ms: None,
            duration_ms: started_at.elapsed().as_millis() as u64,
            tokens_emitted: 0,
            outcome: HttpOutcome::InjectedError,
            prompt_preview,
            override_applied,
        });
        let http_status = StatusCode::from_u16(status).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
        return error_response(http_status, &message, err_type, code, retry_after);
    }

    let req_id = format!("meshdev-{seq}");
    let created = timestamp_unix_ms / 1000;
    let prompt_tokens = estimate_tokens(&req.prompt_text);

    // Non-streaming path (note: `dev/mid-stream-drop` and `dev/malformed-sse` still emit
    // their fault behavior when `stream: true`, and return deterministic fault responses
    // when called with `stream: false`).
    if !req.stream {
        return handle_non_streaming(
            state,
            permit,
            req,
            behavior,
            seq,
            req_id,
            created,
            timestamp_unix_ms,
            started_at,
            captured_headers,
            prompt_preview,
            override_applied,
            prompt_tokens,
        )
        .await;
    }

    // For `SlowTtft` with `after_headers == false` (default), sleep before sending `200 OK`
    // so both response-header TTFT timeouts and pre-header client disconnects are exercised.
    if let ResolvedBehavior::SlowTtft {
        ttft_ms,
        after_headers: false,
    } = &behavior
    {
        // Guard ensures disconnect during pre-header sleep is recorded cleanly.
        struct PreHeaderDisconnectGuard {
            inspector: Inspector,
            seq: u64,
            endpoint: &'static str,
            model: String,
            headers: BTreeMap<String, String>,
            prompt_preview: Option<String>,
            override_applied: Option<String>,
            started_at: Instant,
            timestamp_unix_ms: u64,
            armed: bool,
        }
        impl Drop for PreHeaderDisconnectGuard {
            fn drop(&mut self) {
                if self.armed {
                    self.inspector.finish_http_request(HttpRequestRecord {
                        seq: self.seq,
                        timestamp_unix_ms: self.timestamp_unix_ms,
                        endpoint: self.endpoint.to_string(),
                        model: self.model.clone(),
                        stream: true,
                        headers: self.headers.clone(),
                        http_status: 499,
                        ttft_ms: None,
                        duration_ms: self.started_at.elapsed().as_millis() as u64,
                        tokens_emitted: 0,
                        outcome: HttpOutcome::ClientDisconnected,
                        prompt_preview: self.prompt_preview.clone(),
                        override_applied: self.override_applied.clone(),
                    });
                }
            }
        }
        let mut pre_guard = PreHeaderDisconnectGuard {
            inspector: state.inspector.clone(),
            seq,
            endpoint: req.surface.path(),
            model: req.model.clone(),
            headers: captured_headers.clone(),
            prompt_preview: prompt_preview.clone(),
            override_applied: override_applied.clone(),
            started_at,
            timestamp_unix_ms,
            armed: true,
        };
        if !sleep_interruptible(&state, Some(*ttft_ms)).await {
            pre_guard.armed = false;
            drop(permit);
            state.inspector.finish_http_request(HttpRequestRecord {
                seq,
                timestamp_unix_ms,
                endpoint: req.surface.path().to_string(),
                model: req.model.clone(),
                stream: true,
                headers: captured_headers,
                http_status: 503,
                ttft_ms: None,
                duration_ms: started_at.elapsed().as_millis() as u64,
                tokens_emitted: 0,
                outcome: HttpOutcome::ClientDisconnected,
                prompt_preview,
                override_applied,
            });
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "mesh-dev shutting down during TTFT wait",
                "server_error",
                "shutting_down",
                None,
            );
        }
        pre_guard.armed = false;
    }

    // For `dev/mem-probe`, capture the initial memory snapshot before headers so we can
    // attach `x-mesh-dev-mem-available-kb` on the HTTP response headers too.
    let initial_mem_probe = if matches!(behavior, ResolvedBehavior::MemProbe { .. }) {
        let mode = state.defaults.gpu_probe;
        Some(
            tokio::task::spawn_blocking(move || probe_memory(mode))
                .await
                .unwrap_or_else(|_| probe_memory(mode)),
        )
    } else {
        None
    };

    let guard = CancelOnDrop {
        inspector: state.inspector.clone(),
        seq,
        endpoint: req.surface.path(),
        model: req.model.clone(),
        headers: captured_headers.clone(),
        prompt_preview: prompt_preview.clone(),
        override_applied: override_applied.clone(),
        started_at,
        timestamp_unix_ms,
        ttft_ms: None,
        tokens_emitted: 0,
        finished: false,
        _permit: permit,
    };

    let (tx, rx) = mpsc::channel::<io::Result<Bytes>>(16);
    let extra_mem_kb = initial_mem_probe
        .as_ref()
        .and_then(|snap| snap.mem_available_kb);

    tokio::spawn(run_streaming_producer(
        state,
        tx,
        guard,
        req,
        behavior,
        req_id,
        created,
        timestamp_unix_ms,
        started_at,
        captured_headers,
        prompt_tokens,
        initial_mem_probe,
    ));

    let body = Body::from_stream(GuardedByteStream {
        inner: ReceiverStream::new(rx),
    });

    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .header(CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no");

    if let Some(kb) = extra_mem_kb
        && let Ok(val) = HeaderValue::from_str(&kb.to_string())
    {
        builder = builder.header(HeaderName::from_static("x-mesh-dev-mem-available-kb"), val);
    }

    builder
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

#[allow(clippy::too_many_arguments)]
async fn handle_non_streaming(
    state: AppState,
    permit: OwnedSemaphorePermit,
    req: NormalizedRequest,
    behavior: ResolvedBehavior,
    seq: u64,
    req_id: String,
    created: u64,
    timestamp_unix_ms: u64,
    started_at: Instant,
    captured_headers: BTreeMap<String, String>,
    prompt_preview: Option<String>,
    override_applied: Option<String>,
    prompt_tokens: u64,
) -> Response {
    // Non-streaming cancellation guard if client closes connection during a delay.
    let completed = Arc::new(AtomicBool::new(false));
    struct NonStreamCancelGuard {
        completed: Arc<AtomicBool>,
        inspector: Inspector,
        seq: u64,
        endpoint: &'static str,
        model: String,
        headers: BTreeMap<String, String>,
        prompt_preview: Option<String>,
        override_applied: Option<String>,
        started_at: Instant,
        timestamp_unix_ms: u64,
    }
    impl Drop for NonStreamCancelGuard {
        fn drop(&mut self) {
            if !self.completed.load(Ordering::Acquire) {
                self.inspector.finish_http_request(HttpRequestRecord {
                    seq: self.seq,
                    timestamp_unix_ms: self.timestamp_unix_ms,
                    endpoint: self.endpoint.to_string(),
                    model: self.model.clone(),
                    stream: false,
                    headers: self.headers.clone(),
                    http_status: 499,
                    ttft_ms: None,
                    duration_ms: self.started_at.elapsed().as_millis() as u64,
                    tokens_emitted: 0,
                    outcome: HttpOutcome::ClientDisconnected,
                    prompt_preview: self.prompt_preview.clone(),
                    override_applied: self.override_applied.clone(),
                });
            }
        }
    }
    let _cancel_guard = NonStreamCancelGuard {
        completed: completed.clone(),
        inspector: state.inspector.clone(),
        seq,
        endpoint: req.surface.path(),
        model: req.model.clone(),
        headers: captured_headers.clone(),
        prompt_preview: prompt_preview.clone(),
        override_applied: override_applied.clone(),
        started_at,
        timestamp_unix_ms,
    };

    let mut extra_headers: Vec<(HeaderName, HeaderValue)> = Vec::new();
    let mut tool_calls: Option<Value> = None;
    let mut finish_reason = "stop";
    let mut outcome = HttpOutcome::Completed;

    let content = match behavior {
        ResolvedBehavior::Echo {
            delay_ms,
            token_delay_ms,
        } => {
            let total_delay = delay_ms.saturating_add(token_delay_ms);
            if total_delay > 0 && !sleep_interruptible(&state, Some(total_delay)).await {
                completed.store(true, Ordering::Release);
                drop(permit);
                return error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "mesh-dev shutting down",
                    "server_error",
                    "shutting_down",
                    None,
                );
            }
            build_echo_payload(
                &state,
                &req,
                &req_id,
                timestamp_unix_ms,
                started_at,
                &captured_headers,
            )
            .to_string()
        }
        ResolvedBehavior::SlowTtft { ttft_ms, .. } => {
            if !sleep_interruptible(&state, Some(ttft_ms)).await {
                completed.store(true, Ordering::Release);
                drop(permit);
                return error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "mesh-dev shutting down during TTFT wait",
                    "server_error",
                    "shutting_down",
                    None,
                );
            }
            json!({
                "model": req.model,
                "request_id": req_id,
                "ttft_ms": started_at.elapsed().as_millis() as u64,
                "configured_ttft_ms": ttft_ms,
                "status": "completed",
            })
            .to_string()
        }
        ResolvedBehavior::SlowSse {
            interval_ms,
            token_count,
        } => {
            let mut parts = Vec::with_capacity(token_count as usize);
            for idx in 1..=token_count {
                if !sleep_interruptible(&state, Some(interval_ms)).await {
                    completed.store(true, Ordering::Release);
                    drop(permit);
                    return error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "mesh-dev shutting down during slow generation",
                        "server_error",
                        "shutting_down",
                        None,
                    );
                }
                parts.push(format!("tok_{idx:03}"));
            }
            parts.join(" ")
        }
        ResolvedBehavior::MidStreamDrop { .. } => {
            // In non-streaming mode, simulate an aborted body transfer via streaming body error.
            completed.store(true, Ordering::Release);
            let (tx, rx) = mpsc::channel::<io::Result<Bytes>>(2);
            let _ = tx.try_send(Ok(Bytes::from_static(br#"{"id":"partial","choices":[{"#)));
            let _ = tx.try_send(Err(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "mesh-dev injected mid-stream drop",
            )));
            drop(permit);
            state.inspector.finish_http_request(HttpRequestRecord {
                seq,
                timestamp_unix_ms,
                endpoint: req.surface.path().to_string(),
                model: req.model.clone(),
                stream: false,
                headers: captured_headers,
                http_status: 200,
                ttft_ms: Some(started_at.elapsed().as_millis() as u64),
                duration_ms: started_at.elapsed().as_millis() as u64,
                tokens_emitted: 0,
                outcome: HttpOutcome::MidStreamDrop,
                prompt_preview,
                override_applied,
            });
            return Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from_stream(ReceiverStream::new(rx)))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
        }
        ResolvedBehavior::MalformedSse => {
            outcome = HttpOutcome::MalformedSse;
            "data: {malformed-non-stream-fallback".to_string()
        }
        ResolvedBehavior::ToolCallEdge { raw_tag_mode } => {
            if raw_tag_mode {
                tool_call_edge::raw_tag_chunks().concat()
            } else {
                tool_calls = Some(tool_call_edge::non_streaming_tool_calls());
                finish_reason = "tool_calls";
                tool_call_edge::full_think_content()
            }
        }
        ResolvedBehavior::MemProbe { hold_ms } => {
            let mode = state.defaults.gpu_probe;
            let before = tokio::task::spawn_blocking(move || probe_memory(mode))
                .await
                .unwrap_or_else(|_| probe_memory(mode));
            if let Some(kb) = before.mem_available_kb
                && let Ok(val) = HeaderValue::from_str(&kb.to_string())
            {
                extra_headers.push((HeaderName::from_static("x-mesh-dev-mem-available-kb"), val));
            }
            if hold_ms > 0 {
                if !sleep_interruptible(&state, Some(hold_ms)).await {
                    completed.store(true, Ordering::Release);
                    drop(permit);
                    return error_response(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "mesh-dev shutting down during mem-probe hold",
                        "server_error",
                        "shutting_down",
                        None,
                    );
                }
                let after = tokio::task::spawn_blocking(move || probe_memory(mode))
                    .await
                    .unwrap_or_else(|_| probe_memory(mode));
                json!({
                    "model": req.model,
                    "instance_id": state.faults.instance_id(),
                    "pid": state.faults.pid(),
                    "hold_ms": hold_ms,
                    "before": before,
                    "after": after,
                })
                .to_string()
            } else {
                serde_json::to_string(&before).unwrap_or_else(|_| "{}".to_string())
            }
        }
        ResolvedBehavior::EnvProbe => {
            serde_json::to_string(&probe_env()).unwrap_or_else(|_| "{}".to_string())
        }
        ResolvedBehavior::HttpError { .. } => unreachable!("handled before dispatch"),
    };

    let completion_tokens = estimate_tokens(&content);
    let elapsed_ms = started_at.elapsed().as_millis() as u64;

    let body = match req.surface {
        ApiSurface::ChatCompletions => chat_completion_json(
            &req_id,
            created,
            &req.model,
            &content,
            tool_calls,
            finish_reason,
            TokenUsage {
                prompt_tokens,
                completion_tokens,
            },
        ),
        ApiSurface::Completions => text_completion_json(
            &req_id,
            created,
            &req.model,
            &content,
            finish_reason,
            prompt_tokens,
            completion_tokens,
        ),
        ApiSurface::Responses => responses_json(
            &req_id,
            created,
            &req.model,
            &content,
            prompt_tokens,
            completion_tokens,
        ),
    };

    completed.store(true, Ordering::Release);
    drop(permit);
    state.inspector.finish_http_request(HttpRequestRecord {
        seq,
        timestamp_unix_ms,
        endpoint: req.surface.path().to_string(),
        model: req.model.clone(),
        stream: false,
        headers: captured_headers,
        http_status: 200,
        ttft_ms: Some(elapsed_ms),
        duration_ms: elapsed_ms,
        tokens_emitted: completion_tokens,
        outcome,
        prompt_preview,
        override_applied,
    });

    let mut resp = (StatusCode::OK, Json(body)).into_response();
    for (name, val) in extra_headers {
        resp.headers_mut().insert(name, val);
    }
    resp
}

#[allow(clippy::too_many_arguments)]
async fn run_streaming_producer(
    state: AppState,
    tx: mpsc::Sender<io::Result<Bytes>>,
    mut guard: CancelOnDrop,
    req: NormalizedRequest,
    behavior: ResolvedBehavior,
    req_id: String,
    created: u64,
    timestamp_unix_ms: u64,
    started_at: Instant,
    captured_headers: BTreeMap<String, String>,
    prompt_tokens: u64,
    initial_mem_probe: Option<MemorySnapshot>,
) {
    let send_frame = |tx: &mpsc::Sender<io::Result<Bytes>>, frame: String| {
        let tx = tx.clone();
        async move { tx.send(Ok(Bytes::from(frame))).await.is_ok() }
    };

    let emit_delta = |surface: ApiSurface,
                      req_id: &str,
                      created: u64,
                      model: &str,
                      content: Option<&str>,
                      role: Option<&str>,
                      tool_calls: Option<Value>,
                      finish_reason: Option<&str>|
     -> String {
        match surface {
            ApiSurface::ChatCompletions => {
                let mut delta = serde_json::Map::new();
                if let Some(r) = role {
                    delta.insert("role".into(), json!(r));
                }
                if let Some(c) = content {
                    delta.insert("content".into(), json!(c));
                }
                if let Some(tc) = tool_calls {
                    delta.insert("tool_calls".into(), tc);
                }
                format_sse_json(&chat_chunk(
                    req_id,
                    created,
                    model,
                    Value::Object(delta),
                    finish_reason,
                ))
            }
            ApiSurface::Completions => format_sse_json(&text_completion_chunk(
                req_id,
                created,
                model,
                content.unwrap_or(""),
                finish_reason,
            )),
            ApiSurface::Responses => {
                if let Some(c) = content {
                    format_sse_json(&json!({
                        "type": "response.output_text.delta",
                        "response_id": req_id,
                        "delta": c,
                    }))
                } else {
                    format_sse_json(&json!({
                        "type": "response.completed",
                        "response": {
                            "id": req_id,
                            "status": "completed",
                            "model": model,
                        }
                    }))
                }
            }
        }
    };

    match behavior {
        ResolvedBehavior::Echo {
            delay_ms,
            token_delay_ms,
        } => {
            if delay_ms > 0 && !sleep_interruptible(&state, Some(delay_ms)).await {
                return;
            }
            let payload = build_echo_payload(
                &state,
                &req,
                &req_id,
                timestamp_unix_ms,
                started_at,
                &captured_headers,
            )
            .to_string();
            let tokens = estimate_tokens(&payload);
            if token_delay_ms > 0 {
                // Stream payload in fixed-size chunks paced by `token_delay_ms` so callers
                // can simulate slow vs fast peers on `dev/echo` in spillover A/B tests.
                let chunk_count = req.max_tokens.unwrap_or(4).clamp(1, 64) as usize;
                let bytes = payload.as_bytes();
                let step = bytes.len().div_ceil(chunk_count).max(1);
                let mut offset = 0usize;
                let mut idx = 0usize;
                while offset < bytes.len() {
                    if idx > 0 {
                        let disconnected = tokio::select! {
                            biased;
                            _ = tx.closed() => true,
                            ok = sleep_interruptible(&state, Some(token_delay_ms)) => !ok,
                        };
                        if disconnected {
                            return;
                        }
                    }
                    let mut end = (offset + step).min(bytes.len());
                    while end < bytes.len() && !payload.is_char_boundary(end) {
                        end += 1;
                    }
                    let slice = &payload[offset..end];
                    guard.note_first_token();
                    guard.note_tokens(1);
                    let role = (idx == 0).then_some("assistant");
                    if !send_frame(
                        &tx,
                        emit_delta(
                            req.surface,
                            &req_id,
                            created,
                            &req.model,
                            Some(slice),
                            role,
                            None,
                            None,
                        ),
                    )
                    .await
                    {
                        return;
                    }
                    offset = end;
                    idx += 1;
                }
            } else {
                guard.note_first_token();
                guard.note_tokens(tokens);
                if !send_frame(
                    &tx,
                    emit_delta(
                        req.surface,
                        &req_id,
                        created,
                        &req.model,
                        Some(&payload),
                        Some("assistant"),
                        None,
                        None,
                    ),
                )
                .await
                {
                    return;
                }
            }
            if !send_frame(
                &tx,
                emit_delta(
                    req.surface,
                    &req_id,
                    created,
                    &req.model,
                    None,
                    None,
                    None,
                    Some("stop"),
                ),
            )
            .await
            {
                return;
            }
            if req.include_usage
                && req.surface == ApiSurface::ChatCompletions
                && !send_frame(
                    &tx,
                    format_sse_json(&chat_usage_chunk(
                        &req_id,
                        created,
                        &req.model,
                        prompt_tokens,
                        tokens,
                    )),
                )
                .await
            {
                return;
            }
            if !send_frame(&tx, SSE_DONE.to_string()).await {
                return;
            }
            guard.finish(HttpOutcome::Completed);
        }

        ResolvedBehavior::SlowTtft {
            ttft_ms,
            after_headers,
        } => {
            if after_headers && !sleep_interruptible(&state, Some(ttft_ms)).await {
                return;
            }
            let payload = json!({
                "model": req.model,
                "request_id": req_id,
                "ttft_ms": started_at.elapsed().as_millis() as u64,
                "configured_ttft_ms": ttft_ms,
                "status": "completed",
            })
            .to_string();
            guard.note_first_token();
            guard.note_tokens(1);
            if !send_frame(
                &tx,
                emit_delta(
                    req.surface,
                    &req_id,
                    created,
                    &req.model,
                    Some(&payload),
                    Some("assistant"),
                    None,
                    None,
                ),
            )
            .await
            {
                return;
            }
            if !send_frame(
                &tx,
                emit_delta(
                    req.surface,
                    &req_id,
                    created,
                    &req.model,
                    None,
                    None,
                    None,
                    Some("stop"),
                ),
            )
            .await
            {
                return;
            }
            if !send_frame(&tx, SSE_DONE.to_string()).await {
                return;
            }
            guard.finish(HttpOutcome::Completed);
        }

        ResolvedBehavior::SlowSse {
            interval_ms,
            token_count,
        } => {
            for idx in 1..=token_count {
                if idx > 1 {
                    let disconnected = tokio::select! {
                        biased;
                        _ = tx.closed() => true,
                        ok = sleep_interruptible(&state, Some(interval_ms)) => !ok,
                    };
                    if disconnected {
                        return;
                    }
                }
                let piece = format!("tok_{idx:03} ");
                guard.note_first_token();
                guard.note_tokens(1);
                let role = (idx == 1).then_some("assistant");
                if !send_frame(
                    &tx,
                    emit_delta(
                        req.surface,
                        &req_id,
                        created,
                        &req.model,
                        Some(&piece),
                        role,
                        None,
                        None,
                    ),
                )
                .await
                {
                    return;
                }
            }
            if !send_frame(
                &tx,
                emit_delta(
                    req.surface,
                    &req_id,
                    created,
                    &req.model,
                    None,
                    None,
                    None,
                    Some("stop"),
                ),
            )
            .await
            {
                return;
            }
            if req.include_usage
                && req.surface == ApiSurface::ChatCompletions
                && !send_frame(
                    &tx,
                    format_sse_json(&chat_usage_chunk(
                        &req_id,
                        created,
                        &req.model,
                        prompt_tokens,
                        u64::from(token_count),
                    )),
                )
                .await
            {
                return;
            }
            if !send_frame(&tx, SSE_DONE.to_string()).await {
                return;
            }
            guard.finish(HttpOutcome::Completed);
        }

        ResolvedBehavior::MidStreamDrop { chunks } => {
            for idx in 1..=chunks {
                guard.note_first_token();
                guard.note_tokens(1);
                let piece = format!("partial_{idx:02} ");
                let role = (idx == 1).then_some("assistant");
                if !send_frame(
                    &tx,
                    emit_delta(
                        req.surface,
                        &req_id,
                        created,
                        &req.model,
                        Some(&piece),
                        role,
                        None,
                        None,
                    ),
                )
                .await
                {
                    return;
                }
                // Pause briefly so Hyper flushes the SSE chunk onto the TCP wire before
                // the subsequent error frame aborts the chunked transfer.
                tokio::time::sleep(Duration::from_millis(15)).await;
            }
            let _ = tx
                .send(Err(io::Error::new(
                    io::ErrorKind::ConnectionReset,
                    "mesh-dev injected mid-stream drop",
                )))
                .await;
            guard.finish(HttpOutcome::MidStreamDrop);
        }

        ResolvedBehavior::MalformedSse => {
            guard.note_first_token();
            guard.note_tokens(1);
            if !send_frame(
                &tx,
                emit_delta(
                    req.surface,
                    &req_id,
                    created,
                    &req.model,
                    Some("before-corruption "),
                    Some("assistant"),
                    None,
                    None,
                ),
            )
            .await
            {
                return;
            }
            if !send_frame(&tx, "data: {not-valid-json\n\n".to_string()).await {
                return;
            }
            if !send_frame(
                &tx,
                "data: {\"choices\":[{\"delta\":{\"content\":\"cut".to_string(),
            )
            .await
            {
                return;
            }
            guard.finish(HttpOutcome::MalformedSse);
        }

        ResolvedBehavior::ToolCallEdge { raw_tag_mode } => {
            if raw_tag_mode || req.surface != ApiSurface::ChatCompletions {
                let chunks = tool_call_edge::raw_tag_chunks();
                for (idx, chunk) in chunks.iter().enumerate() {
                    guard.note_first_token();
                    guard.note_tokens(1);
                    let role = (idx == 0).then_some("assistant");
                    if !send_frame(
                        &tx,
                        emit_delta(
                            req.surface,
                            &req_id,
                            created,
                            &req.model,
                            Some(chunk),
                            role,
                            None,
                            None,
                        ),
                    )
                    .await
                    {
                        return;
                    }
                }
                if !send_frame(
                    &tx,
                    emit_delta(
                        req.surface,
                        &req_id,
                        created,
                        &req.model,
                        None,
                        None,
                        None,
                        Some("stop"),
                    ),
                )
                .await
                {
                    return;
                }
            } else {
                // 1. Stream split `<think>` reasoning with decoy `</tool_call>`.
                for (idx, frag) in tool_call_edge::THINK_CHUNKS.iter().enumerate() {
                    guard.note_first_token();
                    guard.note_tokens(1);
                    let role = (idx == 0).then_some("assistant");
                    if !send_frame(
                        &tx,
                        emit_delta(
                            req.surface,
                            &req_id,
                            created,
                            &req.model,
                            Some(frag),
                            role,
                            None,
                            None,
                        ),
                    )
                    .await
                    {
                        return;
                    }
                }
                // 2. Stream structured `tool_calls` deltas with tags inside JSON arguments.
                for (idx, arg_frag) in tool_call_edge::TOOL_ARG_CHUNKS.iter().enumerate() {
                    guard.note_tokens(1);
                    let tc = if idx == 0 {
                        json!([{
                            "index": 0,
                            "id": tool_call_edge::TOOL_CALL_ID,
                            "type": "function",
                            "function": {
                                "name": tool_call_edge::TOOL_NAME,
                                "arguments": arg_frag,
                            }
                        }])
                    } else {
                        json!([{
                            "index": 0,
                            "function": {
                                "arguments": arg_frag,
                            }
                        }])
                    };
                    if !send_frame(
                        &tx,
                        emit_delta(
                            req.surface,
                            &req_id,
                            created,
                            &req.model,
                            None,
                            None,
                            Some(tc),
                            None,
                        ),
                    )
                    .await
                    {
                        return;
                    }
                }
                if !send_frame(
                    &tx,
                    emit_delta(
                        req.surface,
                        &req_id,
                        created,
                        &req.model,
                        None,
                        None,
                        None,
                        Some("tool_calls"),
                    ),
                )
                .await
                {
                    return;
                }
            }
            if !send_frame(&tx, SSE_DONE.to_string()).await {
                return;
            }
            guard.finish(HttpOutcome::Completed);
        }

        ResolvedBehavior::MemProbe { hold_ms } => {
            let mode = state.defaults.gpu_probe;
            let before = match initial_mem_probe {
                Some(snap) => snap,
                None => tokio::task::spawn_blocking(move || probe_memory(mode))
                    .await
                    .unwrap_or_else(|_| probe_memory(mode)),
            };
            guard.note_first_token();
            guard.note_tokens(1);
            let first_text = if hold_ms > 0 {
                json!({"phase": "before", "hold_ms": hold_ms, "probe": before}).to_string()
            } else {
                serde_json::to_string(&before).unwrap_or_else(|_| "{}".to_string())
            };
            if !send_frame(
                &tx,
                emit_delta(
                    req.surface,
                    &req_id,
                    created,
                    &req.model,
                    Some(&first_text),
                    Some("assistant"),
                    None,
                    None,
                ),
            )
            .await
            {
                return;
            }
            if hold_ms > 0 {
                let disconnected = tokio::select! {
                    biased;
                    _ = tx.closed() => true,
                    ok = sleep_interruptible(&state, Some(hold_ms)) => !ok,
                };
                if disconnected {
                    return;
                }
                let after = tokio::task::spawn_blocking(move || probe_memory(mode))
                    .await
                    .unwrap_or_else(|_| probe_memory(mode));
                guard.note_tokens(1);
                let second_text = format!("\n{}", json!({"phase": "after", "probe": after}));
                if !send_frame(
                    &tx,
                    emit_delta(
                        req.surface,
                        &req_id,
                        created,
                        &req.model,
                        Some(&second_text),
                        None,
                        None,
                        None,
                    ),
                )
                .await
                {
                    return;
                }
            }
            if !send_frame(
                &tx,
                emit_delta(
                    req.surface,
                    &req_id,
                    created,
                    &req.model,
                    None,
                    None,
                    None,
                    Some("stop"),
                ),
            )
            .await
            {
                return;
            }
            if !send_frame(&tx, SSE_DONE.to_string()).await {
                return;
            }
            guard.finish(HttpOutcome::Completed);
        }

        ResolvedBehavior::EnvProbe => {
            let payload = serde_json::to_string(&probe_env()).unwrap_or_else(|_| "{}".to_string());
            guard.note_first_token();
            guard.note_tokens(1);
            if !send_frame(
                &tx,
                emit_delta(
                    req.surface,
                    &req_id,
                    created,
                    &req.model,
                    Some(&payload),
                    Some("assistant"),
                    None,
                    None,
                ),
            )
            .await
            {
                return;
            }
            if !send_frame(
                &tx,
                emit_delta(
                    req.surface,
                    &req_id,
                    created,
                    &req.model,
                    None,
                    None,
                    None,
                    Some("stop"),
                ),
            )
            .await
            {
                return;
            }
            if !send_frame(&tx, SSE_DONE.to_string()).await {
                return;
            }
            guard.finish(HttpOutcome::Completed);
        }

        ResolvedBehavior::HttpError { .. } => unreachable!("handled before streaming"),
    }
}

fn build_echo_payload(
    state: &AppState,
    req: &NormalizedRequest,
    req_id: &str,
    timestamp_unix_ms: u64,
    started_at: Instant,
    headers: &BTreeMap<String, String>,
) -> Value {
    json!({
        "model": req.model,
        "endpoint": req.surface.path(),
        "stream": req.stream,
        "request_id": req_id,
        "received_at_unix_ms": timestamp_unix_ms,
        "elapsed_ms": started_at.elapsed().as_millis() as u64,
        "prompt": req.prompt_text,
        "message_count": req.message_count,
        "headers": headers,
        "plugin_pid": state.faults.pid(),
        "plugin_instance_id": state.faults.instance_id(),
    })
}

fn error_response(
    status: StatusCode,
    message: &str,
    err_type: &str,
    code: &str,
    retry_after: Option<u64>,
) -> Response {
    let mut resp = (status, Json(openai_error_json(message, err_type, code))).into_response();
    if let Some(secs) = retry_after
        && let Ok(val) = HeaderValue::from_str(&secs.to_string())
    {
        resp.headers_mut().insert(RETRY_AFTER, val);
    }
    resp
}

async fn sleep_interruptible(state: &AppState, duration_ms: Option<u64>) -> bool {
    if state.is_shutting_down() {
        return false;
    }
    match duration_ms {
        Some(0) => true,
        Some(ms) => tokio::select! {
            biased;
            _ = state.shutdown_notified() => false,
            _ = tokio::time::sleep(Duration::from_millis(ms)) => !state.is_shutting_down(),
        },
        None => {
            state.shutdown_notified().await;
            false
        }
    }
}

fn truncate_preview(input: &str) -> String {
    const MAX_CHARS: usize = 160;
    let mut chars = input.chars();
    let preview: String = chars.by_ref().take(MAX_CHARS).collect();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}
