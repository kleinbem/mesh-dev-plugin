//! Integration tests for `mesh-dev`'s OpenAI-compatible HTTP server covering all 9 `dev/*` models,
//! SSE framing & cancellation propagation, synchronous `503` backpressure, and transient overrides.

use std::time::{Duration, Instant};

use anyhow::Result;
use mesh_dev::cli::{DevArgs, GpuProbeMode};
use mesh_dev::faults::{HttpFaultMode, ModelOverride, OverrideBehavior};
use mesh_dev::inspector::{HttpOutcome, InspectSource};
use mesh_dev::mesh_adapter::build_app_state;
use mesh_dev::models::ALL_MODELS;
use mesh_dev::server::{AppState, serve_with_state};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout};

struct TestServer {
    port: u16,
    state: AppState,
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.state.request_shutdown();
        self.handle.abort();
    }
}

async fn start_server(args: DevArgs) -> Result<TestServer> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let state = build_app_state(&args);
    let serve_state = state.clone();
    let handle = tokio::spawn(async move {
        let _ = serve_with_state(listener, serve_state).await;
    });
    Ok(TestServer {
        port,
        state,
        handle,
    })
}

async fn send_raw(
    port: u16,
    method: &str,
    path: &str,
    extra_headers: &[(&str, &str)],
    body: Option<&str>,
) -> Result<TcpStream> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).await?;
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n");
    for (k, v) in extra_headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(body) = body {
        req.push_str("Content-Type: application/json\r\n");
        req.push_str(&format!("Content-Length: {}\r\n\r\n{body}", body.len()));
    } else {
        req.push_str("\r\n");
    }
    stream.write_all(req.as_bytes()).await?;
    Ok(stream)
}

async fn read_http(stream: &mut TcpStream, limit: Duration) -> Result<(u16, String, String)> {
    let mut buf = Vec::new();
    timeout(limit, stream.read_to_end(&mut buf)).await??;
    let raw = String::from_utf8_lossy(&buf).into_owned();
    let (head, body_raw) = raw.split_once("\r\n\r\n").unwrap_or((&raw, ""));
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    let body = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        decode_chunked(body_raw)
    } else {
        body_raw.to_string()
    };
    Ok((status, head.to_string(), body))
}

fn decode_chunked(raw: &str) -> String {
    let mut out = String::new();
    let mut rest = raw;
    loop {
        let Some((len_line, after_len)) = rest.split_once("\r\n") else {
            out.push_str(rest);
            break;
        };
        let hex_part = len_line.split(';').next().unwrap_or("").trim();
        let Ok(len) = usize::from_str_radix(hex_part, 16) else {
            out.push_str(rest);
            break;
        };
        if len == 0 {
            break;
        }
        if after_len.len() < len {
            out.push_str(after_len);
            break;
        }
        out.push_str(&after_len[..len]);
        rest = after_len[len..]
            .strip_prefix("\r\n")
            .unwrap_or(&after_len[len..]);
    }
    out
}

async fn read_some(stream: &mut TcpStream, min_bytes: usize, limit: Duration) -> Result<String> {
    let mut buf = Vec::new();
    let deadline = Instant::now() + limit;
    let mut chunk = [0u8; 4096];
    while buf.len() < min_bytes {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        match timeout(left, stream.read(&mut chunk)).await {
            Ok(Ok(0)) => break,
            Ok(Ok(n)) => buf.extend_from_slice(&chunk[..n]),
            _ => break,
        }
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

#[tokio::test]
async fn models_endpoint_lists_all_nine_models_and_honors_http_faults() -> Result<()> {
    let srv = start_server(DevArgs::default()).await?;

    let mut s = send_raw(srv.port, "GET", "/v1/models", &[], None).await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    let parsed: Value = serde_json::from_str(&body)?;
    let ids: Vec<&str> = parsed["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert_eq!(ids.len(), ALL_MODELS.len());
    for expected in ALL_MODELS {
        assert!(ids.contains(expected), "missing model {expected}");
    }

    srv.state.faults.set_http_fault(HttpFaultMode::ModelsEmpty);
    let mut s = send_raw(srv.port, "GET", "/v1/models", &[], None).await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    let parsed: Value = serde_json::from_str(&body)?;
    assert_eq!(parsed["data"].as_array().unwrap().len(), 0);

    srv.state.faults.set_http_fault(HttpFaultMode::Models503 {
        detail: "probe down".into(),
    });
    let mut s = send_raw(srv.port, "GET", "/v1/models", &[], None).await?;
    let (status, _, _) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 503);

    srv.state.faults.reset_all();
    let mut s = send_raw(srv.port, "GET", "/v1/models", &[], None).await?;
    let (status, _, _) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    Ok(())
}

#[tokio::test]
async fn dev_echo_echoes_prompt_headers_and_redacts_auth() -> Result<()> {
    let srv = start_server(DevArgs::default()).await?;
    let payload = json!({
        "model": "dev/echo",
        "messages": [{"role": "user", "content": "hello mesh"}],
        "stream": false
    })
    .to_string();

    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[
            ("x-mesh-target", "peer-1234"),
            ("x-mesh-dev-nonce", "nonce-abc"),
            ("authorization", "Bearer super-secret"),
        ],
        Some(&payload),
    )
    .await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    let resp: Value = serde_json::from_str(&body)?;
    let content = resp["choices"][0]["message"]["content"].as_str().unwrap();
    let echoed: Value = serde_json::from_str(content)?;
    assert_eq!(echoed["model"], "dev/echo");
    assert_eq!(echoed["prompt"], "hello mesh");
    assert_eq!(echoed["headers"]["x-mesh-target"], "peer-1234");
    assert_eq!(echoed["headers"]["x-mesh-dev-nonce"], "nonce-abc");
    assert_eq!(echoed["headers"]["authorization"], "<redacted>");
    assert_eq!(echoed["plugin_instance_id"], srv.state.faults.instance_id());

    // Also verify streaming + /v1/completions + /v1/responses.
    let stream_payload = json!({
        "model": "dev/echo",
        "messages": [{"role": "user", "content": "stream-echo"}],
        "stream": true,
        "stream_options": {"include_usage": true}
    })
    .to_string();
    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(&stream_payload),
    )
    .await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(body.contains("stream-echo"));
    assert!(body.contains("\"usage\""));
    assert!(body.trim_end().ends_with("data: [DONE]"));

    let comp_payload = json!({"model": "dev/echo", "prompt": "legacy prompt"}).to_string();
    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/completions",
        &[],
        Some(&comp_payload),
    )
    .await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(body.contains("legacy prompt"));

    let resp_payload = json!({"model": "dev/echo", "input": "responses input"}).to_string();
    let mut s = send_raw(srv.port, "POST", "/v1/responses", &[], Some(&resp_payload)).await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(body.contains("responses input"));
    Ok(())
}

#[tokio::test]
async fn dev_slow_ttft_delays_and_cancels_on_client_disconnect() -> Result<()> {
    let srv = start_server(DevArgs::default()).await?;

    // Fast completion with header override.
    let body = json!({
        "model": "dev/slow-ttft-15s",
        "messages": [{"role": "user", "content": "wait"}],
        "stream": true
    })
    .to_string();
    let t0 = Instant::now();
    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[("x-mesh-dev-ttft-ms", "120")],
        Some(&body),
    )
    .await?;
    let (status, _, sse) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(t0.elapsed() >= Duration::from_millis(100));
    assert!(sse.trim_end().ends_with("data: [DONE]"));

    // Client disconnect during long TTFT records ClientDisconnected promptly.
    let s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[("x-mesh-dev-ttft-ms", "10000")],
        Some(&body),
    )
    .await?;
    sleep(Duration::from_millis(80)).await;
    drop(s);

    let mut disconnected = false;
    for _ in 0..30 {
        let stats = srv.state.inspector.stats();
        if stats.http.client_disconnected >= 1 {
            disconnected = true;
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }
    assert!(
        disconnected,
        "disconnect during TTFT must record ClientDisconnected promptly"
    );
    Ok(())
}

#[tokio::test]
async fn dev_slow_sse_streams_and_cancels_on_early_client_drop() -> Result<()> {
    let srv = start_server(DevArgs {
        slow_sse_interval_ms: 80,
        ..DevArgs::default()
    })
    .await?;
    let body = json!({
        "model": "dev/slow-sse-1tps",
        "messages": [{"role": "user", "content": "pace"}],
        "max_tokens": 50,
        "stream": true
    })
    .to_string();

    let mut s = send_raw(srv.port, "POST", "/v1/chat/completions", &[], Some(&body)).await?;
    let first = read_some(&mut s, 120, Duration::from_secs(3)).await?;
    assert!(
        first.contains("tok_001"),
        "expected first token in: {first}"
    );
    drop(s);

    let mut seen_disconnect = false;
    for _ in 0..40 {
        let recent =
            srv.state
                .inspector
                .inspect_recent(InspectSource::Http, 5, Some("dev/slow-sse-1tps"));
        if let Some(records) = recent.http
            && let Some(rec) = records.first()
            && rec.outcome == HttpOutcome::ClientDisconnected
        {
            assert!(rec.tokens_emitted >= 1);
            assert!(rec.tokens_emitted < 50);
            seen_disconnect = true;
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }
    assert!(
        seen_disconnect,
        "early SSE client drop must cancel stream and record ClientDisconnected"
    );
    assert_eq!(srv.state.inspector.stats().http.in_flight, 0);
    Ok(())
}

#[tokio::test]
async fn dev_mid_stream_drop_flushes_chunks_then_aborts_without_done() -> Result<()> {
    let srv = start_server(DevArgs {
        mid_stream_chunks: 3,
        ..DevArgs::default()
    })
    .await?;
    let body = json!({
        "model": "dev/mid-stream-drop",
        "messages": [{"role": "user", "content": "drop"}],
        "stream": true
    })
    .to_string();

    let mut s = send_raw(srv.port, "POST", "/v1/chat/completions", &[], Some(&body)).await?;
    let mut raw = Vec::new();
    // ConnectionReset or unexpected EOF is expected; ignore read error after partial data.
    let _ = timeout(Duration::from_secs(5), s.read_to_end(&mut raw)).await;
    let text = String::from_utf8_lossy(&raw);
    assert!(text.starts_with("HTTP/1.1 200"), "got: {text}");
    assert!(text.contains("partial_01"), "missing chunk 1: {text}");
    assert!(text.contains("partial_03"), "missing chunk 3: {text}");
    assert!(
        !text.contains("[DONE]"),
        "mid-stream-drop must never emit [DONE]: {text}"
    );
    assert!(
        !text.ends_with("0\r\n\r\n"),
        "mid-stream-drop must abort chunked stream without terminal 0 chunk"
    );

    sleep(Duration::from_millis(50)).await;
    assert_eq!(srv.state.inspector.stats().http.mid_stream_drops, 1);
    Ok(())
}

#[tokio::test]
async fn dev_http_503_429_and_malformed_sse_behave_deterministically() -> Result<()> {
    let srv = start_server(DevArgs::default()).await?;

    let req503 = json!({"model": "dev/http-503", "messages": [{"role": "user", "content": "x"}]})
        .to_string();
    let mut s = send_raw(srv.port, "POST", "/v1/chat/completions", &[], Some(&req503)).await?;
    let (status, head, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 503);
    assert!(head.to_ascii_lowercase().contains("retry-after: 1"));
    assert!(body.contains("service_unavailable"));

    let req429 = json!({"model": "dev/http-429", "messages": [{"role": "user", "content": "x"}]})
        .to_string();
    let mut s = send_raw(srv.port, "POST", "/v1/chat/completions", &[], Some(&req429)).await?;
    let (status, head, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 429);
    assert!(head.to_ascii_lowercase().contains("retry-after: 2"));
    assert!(body.contains("rate_limit_exceeded"));

    let req_bad = json!({
        "model": "dev/malformed-sse",
        "messages": [{"role": "user", "content": "x"}],
        "stream": true
    })
    .to_string();
    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(&req_bad),
    )
    .await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(body.contains("before-corruption"));
    assert!(body.contains("data: {not-valid-json"));
    assert!(!body.contains("[DONE]"));
    Ok(())
}

#[tokio::test]
async fn dev_tool_call_edge_streams_split_think_and_tool_args_with_embedded_tags() -> Result<()> {
    let srv = start_server(DevArgs::default()).await?;

    // Structured streaming mode.
    let body = json!({
        "model": "dev/tool-call-edge",
        "messages": [{"role": "user", "content": "call tool"}],
        "stream": true
    })
    .to_string();
    let mut s = send_raw(srv.port, "POST", "/v1/chat/completions", &[], Some(&body)).await?;
    let (status, _, sse) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(sse.trim_end().ends_with("data: [DONE]"));

    let mut content = String::new();
    let mut tool_args = String::new();
    let mut saw_finish_tool_calls = false;
    for line in sse.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            continue;
        }
        let chunk: Value = serde_json::from_str(data)?;
        let choice = &chunk["choices"][0];
        if let Some(c) = choice["delta"]["content"].as_str() {
            content.push_str(c);
        }
        if let Some(arr) = choice["delta"]["tool_calls"].as_array() {
            for tc in arr {
                if let Some(a) = tc["function"]["arguments"].as_str() {
                    tool_args.push_str(a);
                }
            }
        }
        if choice["finish_reason"].as_str() == Some("tool_calls") {
            saw_finish_tool_calls = true;
        }
    }
    assert!(saw_finish_tool_calls);
    assert!(content.starts_with("<think>"));
    assert!(content.contains("</tool_call>"));
    assert!(content.ends_with("</think>"));
    let parsed_args: Value = serde_json::from_str(&tool_args)?;
    assert!(parsed_args["note"].as_str().unwrap().contains("<think>"));
    assert!(
        parsed_args["note"]
            .as_str()
            .unwrap()
            .contains("</tool_call>")
    );

    // Raw-tag mode.
    let raw_body = json!({
        "model": "dev/tool-call-edge",
        "messages": [{"role": "user", "content": "mode=raw-tag"}],
        "stream": false
    })
    .to_string();
    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(&raw_body),
    )
    .await?;
    let (status, _, resp_body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    let parsed: Value = serde_json::from_str(&resp_body)?;
    let text = parsed["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(text.contains("</think><tool_call>"));
    assert!(text.ends_with("</tool_call>"));
    Ok(())
}

#[tokio::test]
async fn dev_mem_probe_returns_live_meminfo_rss_and_supports_hold() -> Result<()> {
    let srv = start_server(DevArgs {
        gpu_probe: GpuProbeMode::Off,
        ..DevArgs::default()
    })
    .await?;

    let body = json!({
        "model": "dev/mem-probe",
        "messages": [{"role": "user", "content": "probe"}],
        "stream": false
    })
    .to_string();
    let mut s = send_raw(srv.port, "POST", "/v1/chat/completions", &[], Some(&body)).await?;
    let (status, head, resp_body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    let parsed: Value = serde_json::from_str(&resp_body)?;
    let probe: Value =
        serde_json::from_str(parsed["choices"][0]["message"]["content"].as_str().unwrap())?;
    #[cfg(target_os = "linux")]
    {
        assert!(
            head.to_ascii_lowercase()
                .contains("x-mesh-dev-mem-available-kb:")
        );
        assert!(probe["mem_available_kb"].as_u64().unwrap() > 0);
        assert!(probe["self_rss_kb"].as_u64().unwrap() > 0);
    }

    // Hold mode in streaming emits before + after snapshots.
    let stream_body = json!({
        "model": "dev/mem-probe",
        "messages": [{"role": "user", "content": "hold"}],
        "stream": true
    })
    .to_string();
    let t0 = Instant::now();
    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[("x-mesh-dev-hold-ms", "80")],
        Some(&stream_body),
    )
    .await?;
    let (status, _, sse) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(t0.elapsed() >= Duration::from_millis(60));
    assert!(sse.contains("\"phase\":\"before\"") || sse.contains(r#"\"phase\":\"before\""#));
    assert!(sse.contains("\"phase\":\"after\"") || sse.contains(r#"\"phase\":\"after\""#));
    assert!(sse.trim_end().ends_with("data: [DONE]"));
    Ok(())
}

#[tokio::test]
async fn synchronous_backpressure_returns_503_before_sse_200_ok() -> Result<()> {
    // Default `max_concurrency` is 1.
    let srv = start_server(DevArgs {
        slow_sse_interval_ms: 200,
        ..DevArgs::default()
    })
    .await?;

    let slow = json!({
        "model": "dev/slow-sse-1tps",
        "messages": [{"role": "user", "content": "hold slot"}],
        "max_tokens": 20,
        "stream": true
    })
    .to_string();
    let mut holder = send_raw(srv.port, "POST", "/v1/chat/completions", &[], Some(&slow)).await?;
    let first = read_some(&mut holder, 100, Duration::from_secs(3)).await?;
    assert!(first.starts_with("HTTP/1.1 200"));

    // Verify live in-flight counter is exposed via inspect_recent.
    let mid_inspect = srv
        .state
        .inspector
        .inspect_recent(InspectSource::Http, 5, None);
    assert_eq!(mid_inspect.in_flight, 1);

    let second_body = json!({
        "model": "dev/echo",
        "messages": [{"role": "user", "content": "overflow"}],
        "stream": true
    })
    .to_string();
    let mut rejected = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(&second_body),
    )
    .await?;
    let (status, _, body) = read_http(&mut rejected, Duration::from_secs(5)).await?;
    assert_eq!(
        status, 503,
        "queue-full streaming request must return 503, never 200 + [DONE]"
    );
    assert!(body.contains("queue_full"));
    assert!(!body.contains("[DONE]"));

    drop(holder);
    assert_eq!(srv.state.inspector.stats().http.backpressure_503, 1);
    Ok(())
}

#[tokio::test]
async fn startup_delay_holds_models_in_loading_state_then_becomes_ready() -> Result<()> {
    let srv = start_server(DevArgs {
        startup_delay_ms: 180,
        ..DevArgs::default()
    })
    .await?;

    let mut s = send_raw(srv.port, "GET", "/v1/models", &[], None).await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 503);
    assert!(body.contains("\"loading\""), "got: {body}");

    let mut h = send_raw(srv.port, "GET", "/health", &[], None).await?;
    let (h_status, _, h_body) = read_http(&mut h, Duration::from_secs(5)).await?;
    assert_eq!(h_status, 200);
    let health_json: Value = serde_json::from_str(&h_body)?;
    assert_eq!(health_json["loading"], true);
    assert_eq!(health_json["status"], "loading");

    sleep(Duration::from_millis(220)).await;

    let mut s2 = send_raw(srv.port, "GET", "/v1/models", &[], None).await?;
    let (status2, _, body2) = read_http(&mut s2, Duration::from_secs(5)).await?;
    assert_eq!(status2, 200);
    assert!(body2.contains("dev/echo"));
    Ok(())
}

#[tokio::test]
async fn dynamic_parameterized_models_and_env_probe_work_end_to_end() -> Result<()> {
    let srv = start_server(DevArgs {
        gpu_probe: GpuProbeMode::Off,
        max_concurrency: 4,
        ..DevArgs::default()
    })
    .await?;

    // Dynamic TTFT model (non-streaming JSON).
    let t0 = Instant::now();
    let ttft_req = json!({
        "model": "dev/slow-ttft-100ms",
        "messages": [{"role": "user", "content": "wait"}],
        "stream": false
    })
    .to_string();
    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(&ttft_req),
    )
    .await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(t0.elapsed() >= Duration::from_millis(80));
    assert!(body.contains("dev/slow-ttft-100ms"));

    // Dynamic mem-probe-hold-<N>ms model.
    let t1 = Instant::now();
    let hold_req = json!({
        "model": "dev/mem-probe-hold-80ms",
        "messages": [{"role": "user", "content": "hold"}],
        "stream": false
    })
    .to_string();
    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(&hold_req),
    )
    .await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(t1.elapsed() >= Duration::from_millis(60));
    assert!(body.contains("before"));
    assert!(body.contains("after"));

    // Dynamic slow-sse-<N>tps model.
    let sse_req = json!({
        "model": "dev/slow-sse-20tps",
        "messages": [{"role": "user", "content": "fast"}],
        "max_tokens": 3,
        "stream": true
    })
    .to_string();
    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(&sse_req),
    )
    .await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(body.contains("tok_003"));
    assert!(body.trim_end().ends_with("data: [DONE]"));

    // dev/env-probe model.
    let env_req = json!({
        "model": "dev/env-probe",
        "messages": [{"role": "user", "content": "env"}],
        "stream": false
    })
    .to_string();
    let mut s = send_raw(
        srv.port,
        "POST",
        "/v1/chat/completions",
        &[],
        Some(&env_req),
    )
    .await?;
    let (status, _, body) = read_http(&mut s, Duration::from_secs(5)).await?;
    assert_eq!(status, 200);
    assert!(body.contains("allowlisted_vars"));

    // Verify stats exposes cancelled_count alongside client_disconnected.
    let stats = srv.state.inspector.stats();
    assert_eq!(stats.http.cancelled_count, stats.http.client_disconnected);
    Ok(())
}

#[tokio::test]
async fn counted_model_override_fails_then_auto_recovers() -> Result<()> {
    let srv = start_server(DevArgs::default()).await?;
    srv.state.faults.set_model_override(
        "dev/echo".into(),
        Some(ModelOverride {
            behavior: OverrideBehavior::Http503,
            remaining: Some(2),
            delay_ms: None,
            chunks: None,
        }),
    );
    let body = json!({
        "model": "dev/echo",
        "messages": [{"role": "user", "content": "failover"}],
        "stream": false
    })
    .to_string();

    for expected_status in [503, 503, 200] {
        let mut s = send_raw(srv.port, "POST", "/v1/chat/completions", &[], Some(&body)).await?;
        let (status, _, _) = read_http(&mut s, Duration::from_secs(5)).await?;
        assert_eq!(status, expected_status);
    }
    Ok(())
}
