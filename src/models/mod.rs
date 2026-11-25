//! Programmable fake model catalog for `mesh-dev`.
//!
//! Includes fixed presets advertised in `GET /v1/models` plus dynamic parameterized models:
//! - `dev/slow-ttft-<N>s` / `dev/slow-ttft-<N>ms`
//! - `dev/slow-sse-<N>tps`
//! - `dev/mem-probe-hold-<N>s` / `dev/mem-probe-hold-<N>ms`

pub mod tool_call_edge;

use axum::http::HeaderMap;

use crate::cli::GpuProbeMode;
use crate::faults::{ModelOverride, OverrideBehavior};

pub const MODEL_ECHO: &str = "dev/echo";
pub const MODEL_SLOW_TTFT_15S: &str = "dev/slow-ttft-15s";
pub const MODEL_SLOW_TTFT_45S: &str = "dev/slow-ttft-45s";
pub const MODEL_SLOW_TTFT_75S: &str = "dev/slow-ttft-75s";
pub const MODEL_SLOW_SSE: &str = "dev/slow-sse-1tps";
pub const MODEL_MID_STREAM_DROP: &str = "dev/mid-stream-drop";
pub const MODEL_HTTP_503: &str = "dev/http-503";
pub const MODEL_HTTP_429: &str = "dev/http-429";
pub const MODEL_MALFORMED_SSE: &str = "dev/malformed-sse";
pub const MODEL_TOOL_CALL_EDGE: &str = "dev/tool-call-edge";
pub const MODEL_MEM_PROBE: &str = "dev/mem-probe";
pub const MODEL_MEM_PROBE_HOLD_5S: &str = "dev/mem-probe-hold-5s";
pub const MODEL_ENV_PROBE: &str = "dev/env-probe";

pub const ALL_MODELS: &[&str] = &[
    MODEL_ECHO,
    MODEL_SLOW_TTFT_15S,
    MODEL_SLOW_TTFT_45S,
    MODEL_SLOW_TTFT_75S,
    MODEL_SLOW_SSE,
    MODEL_MID_STREAM_DROP,
    MODEL_HTTP_503,
    MODEL_HTTP_429,
    MODEL_MALFORMED_SSE,
    MODEL_TOOL_CALL_EDGE,
    MODEL_MEM_PROBE,
    MODEL_MEM_PROBE_HOLD_5S,
    MODEL_ENV_PROBE,
];

#[derive(Debug, Clone)]
pub struct ModelDefaults {
    pub slow_ttft_ms: u64,
    pub slow_sse_interval_ms: u64,
    pub token_delay_ms: u64,
    pub mid_stream_chunks: u32,
    pub gpu_probe: GpuProbeMode,
}

impl Default for ModelDefaults {
    fn default() -> Self {
        Self {
            slow_ttft_ms: 15_000,
            slow_sse_interval_ms: 1_000,
            token_delay_ms: 0,
            mid_stream_chunks: 3,
            gpu_probe: GpuProbeMode::Auto,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicModelSpec {
    SlowTtftMs(u64),
    SlowSseIntervalMs(u64),
    MemProbeHoldMs(u64),
}

/// Parse parameterized model IDs such as `dev/slow-ttft-45s`, `dev/slow-ttft-250ms`,
/// `dev/slow-sse-2tps`, `dev/mem-probe-hold-5s`, or `dev/mem-probe-hold-200ms`.
pub fn parse_dynamic_model(model: &str) -> Option<DynamicModelSpec> {
    if let Some(rest) = model.strip_prefix("dev/slow-ttft-") {
        return parse_duration_suffix_ms(rest).map(DynamicModelSpec::SlowTtftMs);
    }
    if let Some(rest) = model.strip_prefix("dev/slow-sse-")
        && let Some(tps_str) = rest.strip_suffix("tps")
        && let Ok(tps) = tps_str.parse::<u64>()
        && tps > 0
    {
        let interval_ms = (1_000 / tps).max(1);
        return Some(DynamicModelSpec::SlowSseIntervalMs(interval_ms));
    }
    if let Some(rest) = model.strip_prefix("dev/mem-probe-hold-") {
        return parse_duration_suffix_ms(rest).map(DynamicModelSpec::MemProbeHoldMs);
    }
    None
}

fn parse_duration_suffix_ms(suffix: &str) -> Option<u64> {
    if let Some(ms_str) = suffix.strip_suffix("ms") {
        return ms_str.parse::<u64>().ok();
    }
    if let Some(secs_str) = suffix.strip_suffix('s') {
        return secs_str
            .parse::<u64>()
            .ok()
            .map(|s| s.saturating_mul(1_000));
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedBehavior {
    Echo {
        delay_ms: u64,
        token_delay_ms: u64,
    },
    SlowTtft {
        ttft_ms: u64,
        after_headers: bool,
    },
    SlowSse {
        interval_ms: u64,
        token_count: u32,
    },
    MidStreamDrop {
        chunks: u32,
    },
    HttpError {
        status: u16,
        err_type: &'static str,
        code: &'static str,
        message: String,
        retry_after: Option<u64>,
    },
    MalformedSse,
    ToolCallEdge {
        raw_tag_mode: bool,
    },
    MemProbe {
        hold_ms: u64,
    },
    EnvProbe,
}

fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

fn header_bool(headers: &HeaderMap, name: &str) -> bool {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|s| matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

/// Resolve the runtime execution behavior for a model ID, applying any active
/// `ModelOverride`, dynamic model suffix parameters, and per-request `x-mesh-dev-*` headers.
pub fn resolve_behavior(
    model: &str,
    prompt_text: &str,
    requested_max_tokens: Option<u32>,
    headers: &HeaderMap,
    defaults: &ModelDefaults,
    active_override: Option<&ModelOverride>,
) -> ResolvedBehavior {
    if let Some(ov) = active_override {
        match ov.behavior {
            OverrideBehavior::Default => {}
            OverrideBehavior::Echo => {
                return ResolvedBehavior::Echo {
                    delay_ms: ov
                        .delay_ms
                        .or_else(|| header_u64(headers, "x-mesh-dev-delay-ms"))
                        .unwrap_or(0),
                    token_delay_ms: header_u64(headers, "x-mesh-dev-token-delay-ms")
                        .unwrap_or(defaults.token_delay_ms),
                };
            }
            OverrideBehavior::Http503 => {
                return ResolvedBehavior::HttpError {
                    status: 503,
                    err_type: "server_error",
                    code: "service_unavailable",
                    message: format!("mesh-dev injected 503 override for {model}"),
                    retry_after: Some(1),
                };
            }
            OverrideBehavior::Http429 => {
                return ResolvedBehavior::HttpError {
                    status: 429,
                    err_type: "rate_limit_error",
                    code: "rate_limit_exceeded",
                    message: format!("mesh-dev injected 429 override for {model}"),
                    retry_after: Some(2),
                };
            }
            OverrideBehavior::MidStreamDrop => {
                return ResolvedBehavior::MidStreamDrop {
                    chunks: ov.chunks.unwrap_or(defaults.mid_stream_chunks).max(1),
                };
            }
            OverrideBehavior::MalformedSse => {
                return ResolvedBehavior::MalformedSse;
            }
            OverrideBehavior::SlowTtft => {
                return ResolvedBehavior::SlowTtft {
                    ttft_ms: ov.delay_ms.unwrap_or(defaults.slow_ttft_ms),
                    after_headers: header_bool(headers, "x-mesh-dev-ttft-after-headers"),
                };
            }
            OverrideBehavior::SlowSse => {
                let token_count = requested_max_tokens
                    .or_else(|| header_u64(headers, "x-mesh-dev-tokens").map(|n| n as u32))
                    .unwrap_or(20)
                    .clamp(1, 256);
                return ResolvedBehavior::SlowSse {
                    interval_ms: ov.delay_ms.unwrap_or(defaults.slow_sse_interval_ms),
                    token_count,
                };
            }
        }
    }

    if let Some(dynamic) = parse_dynamic_model(model) {
        match dynamic {
            DynamicModelSpec::SlowTtftMs(parsed_ms) => {
                let default_ms = if model == MODEL_SLOW_TTFT_15S {
                    defaults.slow_ttft_ms
                } else {
                    parsed_ms
                };
                return ResolvedBehavior::SlowTtft {
                    ttft_ms: header_u64(headers, "x-mesh-dev-ttft-ms").unwrap_or(default_ms),
                    after_headers: header_bool(headers, "x-mesh-dev-ttft-after-headers"),
                };
            }
            DynamicModelSpec::SlowSseIntervalMs(parsed_ms) => {
                let default_ms = if model == MODEL_SLOW_SSE {
                    if defaults.token_delay_ms > 0 {
                        defaults.token_delay_ms
                    } else {
                        defaults.slow_sse_interval_ms
                    }
                } else {
                    parsed_ms
                };
                let token_count = requested_max_tokens
                    .or_else(|| header_u64(headers, "x-mesh-dev-tokens").map(|n| n as u32))
                    .unwrap_or(20)
                    .clamp(1, 256);
                return ResolvedBehavior::SlowSse {
                    interval_ms: header_u64(headers, "x-mesh-dev-interval-ms")
                        .unwrap_or(default_ms),
                    token_count,
                };
            }
            DynamicModelSpec::MemProbeHoldMs(parsed_ms) => {
                return ResolvedBehavior::MemProbe {
                    hold_ms: header_u64(headers, "x-mesh-dev-hold-ms").unwrap_or(parsed_ms),
                };
            }
        }
    }

    match model {
        MODEL_MID_STREAM_DROP => ResolvedBehavior::MidStreamDrop {
            chunks: header_u64(headers, "x-mesh-dev-chunks")
                .map(|n| n as u32)
                .unwrap_or(defaults.mid_stream_chunks)
                .max(1),
        },
        MODEL_HTTP_503 => ResolvedBehavior::HttpError {
            status: 503,
            err_type: "server_error",
            code: "service_unavailable",
            message: "mesh-dev injected 503 Service Unavailable".to_string(),
            retry_after: Some(1),
        },
        MODEL_HTTP_429 => ResolvedBehavior::HttpError {
            status: 429,
            err_type: "rate_limit_error",
            code: "rate_limit_exceeded",
            message: "mesh-dev injected 429 Too Many Requests".to_string(),
            retry_after: Some(2),
        },
        MODEL_MALFORMED_SSE => ResolvedBehavior::MalformedSse,
        MODEL_TOOL_CALL_EDGE => ResolvedBehavior::ToolCallEdge {
            raw_tag_mode: prompt_text.contains("mode=raw-tag")
                || header_bool(headers, "x-mesh-dev-raw-tag"),
        },
        MODEL_MEM_PROBE => ResolvedBehavior::MemProbe {
            hold_ms: header_u64(headers, "x-mesh-dev-hold-ms").unwrap_or(0),
        },
        MODEL_ENV_PROBE => ResolvedBehavior::EnvProbe,
        _ => ResolvedBehavior::Echo {
            delay_ms: header_u64(headers, "x-mesh-dev-delay-ms").unwrap_or(0),
            token_delay_ms: header_u64(headers, "x-mesh-dev-token-delay-ms")
                .unwrap_or(defaults.token_delay_ms),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dynamic_ttft_sse_and_mem_probe_models() {
        assert_eq!(
            parse_dynamic_model("dev/slow-ttft-45s"),
            Some(DynamicModelSpec::SlowTtftMs(45_000))
        );
        assert_eq!(
            parse_dynamic_model("dev/slow-ttft-75s"),
            Some(DynamicModelSpec::SlowTtftMs(75_000))
        );
        assert_eq!(
            parse_dynamic_model("dev/slow-ttft-250ms"),
            Some(DynamicModelSpec::SlowTtftMs(250))
        );
        assert_eq!(
            parse_dynamic_model("dev/slow-sse-2tps"),
            Some(DynamicModelSpec::SlowSseIntervalMs(500))
        );
        assert_eq!(
            parse_dynamic_model("dev/mem-probe-hold-5s"),
            Some(DynamicModelSpec::MemProbeHoldMs(5_000))
        );
        assert_eq!(
            parse_dynamic_model("dev/mem-probe-hold-150ms"),
            Some(DynamicModelSpec::MemProbeHoldMs(150))
        );
        assert_eq!(parse_dynamic_model("dev/slow-sse-0tps"), None);
        assert_eq!(parse_dynamic_model("dev/echo"), None);
    }
}
