//! OpenAI-compatible request/response wire types and SSE formatting helpers.

use std::collections::BTreeMap;

use axum::http::HeaderMap;
use serde::Deserialize;
use serde_json::{Value, json};

pub const SSE_DONE: &str = "data: [DONE]\n\n";

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: Value,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub prompt: Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub input: Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiSurface {
    ChatCompletions,
    Completions,
    Responses,
}

impl ApiSurface {
    pub fn path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
            Self::Responses => "/v1/responses",
        }
    }
}

/// Extract a human-readable prompt string from chat messages.
pub fn extract_chat_prompt(messages: &[ChatMessage]) -> String {
    let last = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .or_else(|| messages.last());
    match last {
        Some(msg) => content_value_to_string(&msg.content),
        None => String::new(),
    }
}

/// Extract a prompt string from a `/v1/completions` or `/v1/responses` JSON value.
pub fn content_value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                if let Some(s) = item.as_str() {
                    Some(s.to_string())
                } else if let Some(text) = item.get("text").and_then(Value::as_str) {
                    Some(text.to_string())
                } else if let Some(content) = item.get("content") {
                    let s = content_value_to_string(content);
                    (!s.is_empty()).then_some(s)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

const MAX_CAPTURED_HEADERS: usize = 64;
const MAX_CAPTURED_HEADER_VALUE_BYTES: usize = 1024;

/// Capture request headers into a deterministic map, redacting sensitive auth/secret/control
/// headers and bounding header count & value lengths.
pub fn capture_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for (name, value) in headers {
        if map.len() >= MAX_CAPTURED_HEADERS {
            break;
        }
        let key = name.as_str().to_ascii_lowercase();
        let val = if is_sensitive_header(&key) {
            "<redacted>".to_string()
        } else {
            let raw = value.to_str().unwrap_or("<non-utf8>");
            if raw.len() > MAX_CAPTURED_HEADER_VALUE_BYTES {
                let end = raw.floor_char_boundary(MAX_CAPTURED_HEADER_VALUE_BYTES);
                format!("{}...<truncated>", &raw[..end])
            } else {
                raw.to_string()
            }
        };
        map.entry(key)
            .and_modify(|existing: &mut String| {
                if existing != "<redacted>" && existing.len() < MAX_CAPTURED_HEADER_VALUE_BYTES {
                    existing.push_str(", ");
                    existing.push_str(&val);
                }
            })
            .or_insert(val);
    }
    map
}

/// Mirror `mesh_llm_config::safe_exchange_header` sensitivity rules so secrets, credentials,
/// and mesh owner/control headers are never echoed or retained in ring buffers.
pub fn is_sensitive_header(lower: &str) -> bool {
    matches!(
        lower,
        "authorization" | "proxy-authorization" | "cookie" | "set-cookie" | "x-api-key" | "api-key"
    ) || lower.contains("token")
        || lower.contains("secret")
        || lower.contains("key")
        || lower.contains("auth")
        || lower.contains("cookie")
        || lower.contains("session")
        || lower.contains("jwt")
        || lower.starts_with("x-owner-")
        || lower.starts_with("x-mesh-owner-")
        || lower.starts_with("x-mesh-control-")
}

pub fn format_sse_json(payload: &Value) -> String {
    format!("data: {}\n\n", payload)
}

pub fn chat_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Option<&str>,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
    })
}

pub fn chat_usage_chunk(
    id: &str,
    created: u64,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
}

#[derive(Debug, Clone, Copy)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

pub fn chat_completion_json(
    id: &str,
    created: u64,
    model: &str,
    content: &str,
    tool_calls: Option<Value>,
    finish_reason: &str,
    usage: TokenUsage,
) -> Value {
    let mut message = json!({
        "role": "assistant",
        "content": content,
    });
    if let Some(calls) = tool_calls {
        message["tool_calls"] = calls;
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        }
    })
}

pub fn text_completion_chunk(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    finish_reason: Option<&str>,
) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason,
        }]
    })
}

pub fn text_completion_json(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    finish_reason: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
}

pub fn responses_json(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": [{
            "type": "message",
            "id": format!("msg-{id}"),
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": text,
            }]
        }],
        "output_text": text,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
}

pub fn openai_error_json(message: &str, err_type: &str, code: &str) -> Value {
    json!({
        "error": {
            "message": message,
            "type": err_type,
            "code": code,
        }
    })
}

pub fn estimate_tokens(text: &str) -> u64 {
    let words = text.split_whitespace().count() as u64;
    words.max(1)
}
