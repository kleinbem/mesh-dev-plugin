//! Lifecycle, fault-injection crash, observation/exchange inspector, host EOF, and SIGTERM
//! integration tests against the real `mesh-dev` binary.

#![cfg(unix)]

use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use mesh_llm_plugin::{
    LocalStream, PROTOCOL_VERSION, host_capabilities, proto, read_envelope, write_envelope,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UnixListener};
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

fn plugin_command(socket: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mesh-dev"));
    cmd.env("MESH_LLM_PLUGIN_ENDPOINT", socket)
        .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
        .env("MESH_LLM_PLUGIN_NAME", "mesh-dev")
        .env("RUST_LOG", "warn")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    cmd
}

async fn send_env(
    stream: &mut LocalStream,
    id: u64,
    payload: proto::envelope::Payload,
) -> Result<()> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "mesh-dev".into(),
            request_id: id,
            payload: Some(payload),
        },
    )
    .await
}

async fn exchange(
    stream: &mut LocalStream,
    id: u64,
    payload: proto::envelope::Payload,
) -> Result<proto::envelope::Payload> {
    send_env(stream, id, payload).await?;
    let response = timeout(Duration::from_secs(10), read_envelope(stream)).await??;
    assert_eq!(response.request_id, id);
    response.payload.context("missing response payload")
}

async fn call_op(stream: &mut LocalStream, id: u64, op: &str, input: Value) -> Result<Value> {
    let payload = exchange(
        stream,
        id,
        proto::envelope::Payload::InvokeServiceRequest(proto::InvokeServiceRequest {
            kind: proto::ServiceKind::Operation as i32,
            service_name: op.into(),
            input_json: input.to_string(),
            invocation_id: None,
        }),
    )
    .await?;
    let proto::envelope::Payload::InvokeServiceResponse(resp) = payload else {
        bail!("expected InvokeServiceResponse for {op}, got {payload:?}");
    };
    if resp.is_error {
        bail!("operation {op} returned error: {}", resp.output_json);
    }
    let parsed: Value = serde_json::from_str(&resp.output_json)?;
    Ok(parsed.get("structuredContent").cloned().unwrap_or(parsed))
}

struct RunningPlugin {
    child: Child,
    ipc: Option<LocalStream>,
    port: u16,
    _dir: tempfile::TempDir,
}

async fn start_plugin(extra_args: &[&str], mut host_caps: Vec<String>) -> Result<RunningPlugin> {
    if !host_caps
        .iter()
        .any(|c| c == host_capabilities::LOCAL_CHANNELS)
    {
        host_caps.push(host_capabilities::LOCAL_CHANNELS.into());
    }
    let dir = tempfile::tempdir()?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;
    let child = plugin_command(&socket).args(extra_args).spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let init = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "mesh-dev-lifecycle-test".into(),
            host_capabilities: host_caps,
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::InitializeResponse(resp) = init else {
        bail!("unexpected initialize response: {init:?}");
    };
    let url = resp.manifest.context("manifest")?.endpoints[0]
        .address
        .clone()
        .context("endpoint address")?;
    let port: u16 = url
        .trim_start_matches("http://127.0.0.1:")
        .trim_end_matches("/v1")
        .parse()?;
    Ok(RunningPlugin {
        child,
        ipc: Some(stream),
        port,
        _dir: dir,
    })
}

async fn post_http(port: u16, path: &str, body: &str) -> Result<TcpStream> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await?;
    Ok(s)
}

#[tokio::test]
async fn inject_crash_acks_then_exits_with_requested_code() -> Result<()> {
    let mut p = start_plugin(&[], Vec::new()).await?;
    let ipc = p.ipc.as_mut().unwrap();
    let ack = call_op(
        ipc,
        2,
        "inject_crash",
        json!({"exit_code": 42, "delay_ms": 50}),
    )
    .await?;
    assert_eq!(ack["scheduled"], true);
    assert_eq!(ack["exit_code"], 42);

    let status = timeout(Duration::from_secs(5), p.child.wait()).await??;
    assert_eq!(status.code(), Some(42));
    Ok(())
}

#[tokio::test]
async fn observation_feed_exchange_hook_and_publish_status_flow_end_to_end() -> Result<()> {
    let mut p = start_plugin(
        &[],
        vec![
            host_capabilities::LOCAL_INFERENCE_OBSERVATION.into(),
            host_capabilities::OPENAI_EXCHANGE.into(),
            host_capabilities::PLUGIN_STATUS.into(),
        ],
    )
    .await?;
    let ipc = p.ipc.as_mut().unwrap();

    // 1. Plugin proactively sends SubscribeInferenceObservationRequest after on_initialized.
    let sub_env = timeout(Duration::from_secs(5), read_envelope(ipc)).await??;
    assert!(matches!(
        sub_env.payload,
        Some(proto::envelope::Payload::SubscribeInferenceObservationRequest(_))
    ));
    send_env(
        ipc,
        sub_env.request_id,
        proto::envelope::Payload::SubscribeInferenceObservationResponse(
            proto::SubscribeInferenceObservationResponse {
                snapshot: Some(proto::InferenceObservationSnapshot {
                    host_generation: 3,
                    next_sequence: 1,
                    taken_at_unix_ns: 1_000_000,
                    ready_models: vec!["dev/echo".into()],
                    active: Vec::new(),
                }),
            },
        ),
    )
    .await?;

    // 2. Host pushes an observation batch.
    send_env(
        ipc,
        0,
        proto::envelope::Payload::InferenceObservationBatch(proto::InferenceObservationBatch {
            host_generation: 3,
            watermark_unix_ns: 20_000_000,
            events: vec![
                proto::InferenceObservationEvent {
                    sequence: 1,
                    at_unix_ns: 5_000_000,
                    execution_id: 99,
                    kind: Some(proto::inference_observation_event::Kind::Started(
                        proto::InferenceExecutionStarted {
                            model: "dev/echo".into(),
                            operation: proto::InferenceOperation::ChatCompletion as i32,
                            streaming: true,
                        },
                    )),
                },
                proto::InferenceObservationEvent {
                    sequence: 2,
                    at_unix_ns: 12_000_000,
                    execution_id: 99,
                    kind: Some(proto::inference_observation_event::Kind::FirstOutput(
                        proto::Empty {},
                    )),
                },
                proto::InferenceObservationEvent {
                    sequence: 3,
                    at_unix_ns: 18_000_000,
                    execution_id: 99,
                    kind: Some(proto::inference_observation_event::Kind::Usage(
                        proto::InferenceOutputUsage {
                            prompt_tokens: 7,
                            cached_prompt_tokens: None,
                            completion_tokens: 11,
                        },
                    )),
                },
                proto::InferenceObservationEvent {
                    sequence: 4,
                    at_unix_ns: 20_000_000,
                    execution_id: 99,
                    kind: Some(proto::inference_observation_event::Kind::Ended(
                        proto::InferenceExecutionEnded {
                            outcome: proto::InferenceExecutionOutcome::Completed as i32,
                        },
                    )),
                },
            ],
            loss: None,
            resync: None,
        }),
    )
    .await?;

    // 3. Host invokes OpenAiExchange lifecycle hook for request_received + exchange_finished.
    let ex1 = exchange(
        ipc,
        10,
        proto::envelope::Payload::InvokeServiceRequest(proto::InvokeServiceRequest {
            kind: proto::ServiceKind::OpenaiExchange as i32,
            service_name: "observe".into(),
            input_json: json!({
                "exchange_id": "ex-live-1",
                "phase": "request_received",
                "endpoint": "chat_completions",
                "observation_point": "gateway_ingress",
                "method": "POST",
                "path": "/v1/chat/completions",
                "model": "dev/echo",
                "headers": {"x-mesh-target": "peer-xyz"}
            })
            .to_string(),
            invocation_id: None,
        }),
    )
    .await?;
    assert!(matches!(
        ex1,
        proto::envelope::Payload::InvokeServiceResponse(r) if !r.is_error
    ));

    let ex2 = exchange(
        ipc,
        11,
        proto::envelope::Payload::InvokeServiceRequest(proto::InvokeServiceRequest {
            kind: proto::ServiceKind::OpenaiExchange as i32,
            service_name: "observe".into(),
            input_json: json!({
                "exchange_id": "ex-live-1",
                "phase": "exchange_finished",
                "endpoint": "chat_completions",
                "observation_point": "gateway_ingress",
                "method": "POST",
                "path": "/v1/chat/completions",
                "model": "dev/echo",
                "headers": {},
                "execution_outcome": "completed",
                "evidence_complete": true
            })
            .to_string(),
            invocation_id: None,
        }),
    )
    .await?;
    assert!(matches!(
        ex2,
        proto::envelope::Payload::InvokeServiceResponse(r) if !r.is_error
    ));

    // 4. Send a real HTTP request to `dev/echo` too.
    let mut http = post_http(
        p.port,
        "/v1/chat/completions",
        r#"{"model":"dev/echo","messages":[{"role":"user","content":"inspector check"}]}"#,
    )
    .await?;
    let mut http_buf = Vec::new();
    timeout(Duration::from_secs(5), http.read_to_end(&mut http_buf)).await??;
    assert!(String::from_utf8_lossy(&http_buf).starts_with("HTTP/1.1 200"));

    // Allow observation batch task a moment to apply.
    sleep(Duration::from_millis(50)).await;

    // 5. Inspect all 3 layers via `inspect_recent` and `stats`.
    let recent = call_op(ipc, 12, "inspect_recent", json!({"source": "all"})).await?;
    assert_eq!(recent["http"].as_array().unwrap().len(), 1);
    assert_eq!(
        recent["observation"]["recent_finished"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(recent["observation"]["recent_finished"][0]["ttft_ms"], 7);
    assert_eq!(
        recent["observation"]["recent_finished"][0]["completion_tokens"],
        11
    );
    assert_eq!(recent["exchange"].as_array().unwrap().len(), 1);
    assert_eq!(
        recent["exchange"][0]["headers"]["x-mesh-target"],
        "peer-xyz"
    );
    assert_eq!(recent["exchange"][0]["execution_outcome"], "completed");

    let stats = call_op(ipc, 13, "stats", json!({})).await?;
    assert_eq!(stats["inspector"]["http"]["completed"], 1);
    assert_eq!(stats["inspector"]["observation"]["finished_executions"], 1);
    assert_eq!(stats["inspector"]["exchange"]["distinct_exchanges"], 1);

    // 6. Test `publish_status` round-trip with mock host answering PublishPluginStatusRequest.
    send_env(
        ipc,
        14,
        proto::envelope::Payload::InvokeServiceRequest(proto::InvokeServiceRequest {
            kind: proto::ServiceKind::Operation as i32,
            service_name: "publish_status".into(),
            input_json: json!({"payload": {"healthy": true}, "fresh_for_ms": 5000}).to_string(),
            invocation_id: None,
        }),
    )
    .await?;
    let pub_req = timeout(Duration::from_secs(5), read_envelope(ipc)).await??;
    assert!(matches!(
        pub_req.payload,
        Some(proto::envelope::Payload::PublishPluginStatusRequest(_))
    ));
    send_env(
        ipc,
        pub_req.request_id,
        proto::envelope::Payload::PublishPluginStatusResponse(proto::PublishPluginStatusResponse {
            received_at_unix_ms: 1000,
            fresh_until_unix_ms: 6000,
        }),
    )
    .await?;
    let pub_op_resp = timeout(Duration::from_secs(5), read_envelope(ipc)).await??;
    assert_eq!(pub_op_resp.request_id, 14);
    assert!(matches!(
        pub_op_resp.payload,
        Some(proto::envelope::Payload::InvokeServiceResponse(r)) if !r.is_error
    ));

    // 7. Clear inspector buffers and confirm counts reset.
    let _ = call_op(ipc, 15, "clear", json!({"source": "all"})).await?;
    let stats_after = call_op(ipc, 16, "stats", json!({})).await?;
    assert_eq!(stats_after["inspector"]["http"]["total_requests"], 0);
    assert_eq!(
        stats_after["inspector"]["exchange"]["distinct_exchanges"],
        0
    );

    drop(p.ipc.take());
    let _ = timeout(Duration::from_secs(5), p.child.wait()).await;
    Ok(())
}

#[tokio::test]
async fn host_eof_during_slow_stream_exits_promptly() -> Result<()> {
    let mut p = start_plugin(&["--slow-sse-interval-ms", "500"], Vec::new()).await?;
    let _req = post_http(
        p.port,
        "/v1/chat/completions",
        r#"{"model":"dev/slow-sse-1tps","messages":[{"role":"user","content":"long"}],"max_tokens":100,"stream":true}"#,
    )
    .await?;
    sleep(Duration::from_millis(100)).await;

    let t0 = Instant::now();
    drop(p.ipc.take());
    let status = timeout(Duration::from_secs(5), p.child.wait()).await?;
    assert!(status.is_ok());
    assert!(
        t0.elapsed() < Duration::from_secs(3),
        "plugin lingered {:.2}s after host EOF",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[tokio::test]
async fn sigterm_during_slow_stream_exits_promptly() -> Result<()> {
    let mut p = start_plugin(&["--slow-sse-interval-ms", "500"], Vec::new()).await?;
    let _req = post_http(
        p.port,
        "/v1/chat/completions",
        r#"{"model":"dev/slow-sse-1tps","messages":[{"role":"user","content":"long"}],"max_tokens":100,"stream":true}"#,
    )
    .await?;
    sleep(Duration::from_millis(100)).await;

    let pid = p.child.id().context("pid")?;
    let t0 = Instant::now();
    std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()?;
    let status = timeout(Duration::from_secs(5), p.child.wait()).await?;
    assert!(status.is_ok());
    assert!(
        t0.elapsed() < Duration::from_secs(3),
        "plugin lingered {:.2}s after SIGTERM",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
