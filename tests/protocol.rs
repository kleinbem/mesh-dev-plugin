//! End-to-end Mesh-LLM v3 IPC protocol tests against the real `mesh-dev` binary.

#![cfg(unix)]

use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use mesh_dev::models::ALL_MODELS;
use mesh_llm_plugin::{LocalStream, PROTOCOL_VERSION, proto, read_envelope, write_envelope};
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::time::timeout;

fn plugin_command(socket: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mesh-dev"));
    cmd.env("MESH_LLM_PLUGIN_ENDPOINT", socket)
        .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
        .env("MESH_LLM_PLUGIN_NAME", "mesh-dev")
        .env("RUST_LOG", "warn")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    cmd
}

async fn exchange(
    stream: &mut LocalStream,
    id: u64,
    payload: proto::envelope::Payload,
) -> Result<proto::envelope::Payload> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "mesh-dev".into(),
            request_id: id,
            payload: Some(payload),
        },
    )
    .await?;
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

#[tokio::test]
async fn executable_handshake_manifest_health_zero_stdout_and_clean_shutdown() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;

    let mut child = plugin_command(&socket).spawn()?;
    let mut child_stdout = child.stdout.take().context("piped stdout")?;

    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    // 1. Initialize handshake.
    let init = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "mesh-dev-protocol-test".into(),
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::InitializeResponse(init_resp) = init else {
        bail!("unexpected initialize response: {init:?}");
    };
    assert_eq!(init_resp.plugin_protocol_version, PROTOCOL_VERSION);
    let manifest = init_resp.manifest.context("missing manifest")?;
    assert_eq!(manifest.endpoints.len(), 1);
    assert!(manifest.endpoints[0].managed_by_plugin);
    assert_eq!(
        manifest.endpoints[0].protocol.as_deref(),
        Some("openai_compatible")
    );
    let endpoint_url = manifest.endpoints[0]
        .address
        .clone()
        .context("missing endpoint address")?;
    assert!(endpoint_url.starts_with("http://127.0.0.1:"));
    assert!(endpoint_url.ends_with("/v1"));
    assert!(manifest.inference_observation.is_some());
    assert!(manifest.openai_exchange_hook.is_some());
    assert!(manifest.plugin_status.is_some());

    // 2. Health check succeeds.
    let health = exchange(
        &mut stream,
        2,
        proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
    )
    .await?;
    let proto::envelope::Payload::HealthResponse(health_resp) = health else {
        bail!("unexpected health response: {health:?}");
    };
    assert_eq!(
        health_resp.status,
        proto::health_response::Status::Ok as i32
    );
    assert!(
        health_resp
            .detail
            .contains(&format!("models={}", ALL_MODELS.len()))
    );
    assert!(health_resp.detail.contains("state=ready"));

    // 3. `status` operation works.
    let status = call_op(&mut stream, 3, "status", json!({})).await?;
    assert_eq!(status["plugin_name"], "mesh-dev");
    assert_eq!(status["models"].as_array().unwrap().len(), ALL_MODELS.len());
    assert_eq!(status["max_concurrency"], 1);

    // 4. Clean shutdown tears down promptly with ZERO stdout bytes.
    let shutdown = exchange(
        &mut stream,
        4,
        proto::envelope::Payload::ShutdownRequest(proto::ShutdownRequest {
            reason: "protocol test complete".into(),
        }),
    )
    .await?;
    assert!(matches!(
        shutdown,
        proto::envelope::Payload::ShutdownResponse(_)
    ));
    drop(stream);

    let exit_status = timeout(Duration::from_secs(10), child.wait()).await??;
    assert!(exit_status.success(), "plugin exited with {exit_status}");

    let mut stdout_bytes = Vec::new();
    child_stdout.read_to_end(&mut stdout_bytes).await?;
    assert!(
        stdout_bytes.is_empty(),
        "plugin wrote {} stray bytes to stdout: {:?}",
        stdout_bytes.len(),
        String::from_utf8_lossy(&stdout_bytes)
    );
    Ok(())
}

#[tokio::test]
async fn startup_disabled_flag_returns_startup_disabled() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;

    let mut child = plugin_command(&socket)
        .args(["--startup-disabled", "operator disabled test"])
        .spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let resp = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "mesh-dev-protocol-test".into(),
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::ErrorResponse(err) = resp else {
        bail!("expected ErrorResponse for --startup-disabled, got: {resp:?}");
    };
    assert_eq!(err.code, mesh_llm_plugin::STARTUP_DISABLED_ERROR_CODE);
    assert!(err.message.contains("operator disabled test"));
    drop(stream);
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
    Ok(())
}

#[tokio::test]
async fn incompatible_host_protocol_returns_startup_disabled() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;

    let mut child = plugin_command(&socket).spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let resp = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION + 99,
            host_version: "incompatible-host".into(),
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::ErrorResponse(err) = resp else {
        bail!("expected ErrorResponse for incompatible protocol, got: {resp:?}");
    };
    assert_eq!(err.code, mesh_llm_plugin::STARTUP_DISABLED_ERROR_CODE);
    drop(stream);
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
    Ok(())
}

#[tokio::test]
async fn init_delay_ms_delays_initialize_handshake() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;

    let mut child = plugin_command(&socket)
        .args(["--init-delay-ms", "150"])
        .spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let t0 = Instant::now();
    let resp = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "slow-init-test".into(),
            ..Default::default()
        }),
    )
    .await?;
    assert!(matches!(
        resp,
        proto::envelope::Payload::InitializeResponse(_)
    ));
    assert!(t0.elapsed() >= Duration::from_millis(120));
    drop(stream);
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
    Ok(())
}

#[tokio::test]
async fn inject_health_and_unhealthy_on_start_control_health_responses() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;

    let mut child = plugin_command(&socket)
        .args(["--unhealthy-on-start", "boot fault"])
        .spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let init = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "health-fault-test".into(),
            ..Default::default()
        }),
    )
    .await?;
    assert!(matches!(
        init,
        proto::envelope::Payload::InitializeResponse(_)
    ));

    // Initial health check reflects --unhealthy-on-start (PluginRuntime encodes Err as ErrorResponse).
    let h1 = exchange(
        &mut stream,
        2,
        proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
    )
    .await?;
    let proto::envelope::Payload::ErrorResponse(err1) = h1 else {
        bail!("expected ErrorResponse for unhealthy plugin, got {h1:?}");
    };
    assert!(
        err1.message.contains("boot fault"),
        "unexpected health error: {}",
        err1.message
    );

    // Recover via inject_health(ok).
    let _ = call_op(&mut stream, 3, "inject_health", json!({"mode": "ok"})).await?;
    let h2 = exchange(
        &mut stream,
        4,
        proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
    )
    .await?;
    let proto::envelope::Payload::HealthResponse(hr2) = h2 else {
        bail!("expected HealthResponse, got {h2:?}");
    };
    assert_eq!(hr2.status, proto::health_response::Status::Ok as i32);

    // Inject finite hang and confirm it delays health response.
    let _ = call_op(
        &mut stream,
        5,
        "inject_health",
        json!({"mode": "hang", "hang_ms": 150}),
    )
    .await?;
    let t0 = Instant::now();
    let h3 = exchange(
        &mut stream,
        6,
        proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
    )
    .await?;
    assert!(t0.elapsed() >= Duration::from_millis(120));
    assert!(matches!(h3, proto::envelope::Payload::HealthResponse(_)));

    drop(stream);
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
    Ok(())
}

#[tokio::test]
async fn startup_delay_reports_loading_health_and_env_probe_redacts_secrets() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let socket = dir.path().join("host.sock");
    let listener = UnixListener::bind(&socket)?;

    let mut child = plugin_command(&socket)
        .env("MESH_LLM_MODELS_DIR", "/tmp/shared-mesh-models")
        .env("MESH_LLM_JOIN_TOKEN", "top-secret-never-leak")
        .args(["--startup-delay-ms", "220"])
        .spawn()?;
    let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
    let mut stream = LocalStream::Unix(stream);

    let init = exchange(
        &mut stream,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "startup-delay-test".into(),
            ..Default::default()
        }),
    )
    .await?;
    assert!(matches!(
        init,
        proto::envelope::Payload::InitializeResponse(_)
    ));

    // During startup delay, HealthRequest succeeds with state=loading.
    let h_loading = exchange(
        &mut stream,
        2,
        proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
    )
    .await?;
    let proto::envelope::Payload::HealthResponse(hr_loading) = h_loading else {
        bail!("expected HealthResponse during loading, got {h_loading:?}");
    };
    assert!(
        hr_loading.detail.contains("state=loading"),
        "expected state=loading, got: {}",
        hr_loading.detail
    );

    // `env_probe` exposes allowlisted paths and strictly redacts secret env vars.
    let env_resp = call_op(&mut stream, 3, "env_probe", json!({})).await?;
    assert_eq!(
        env_resp["allowlisted_vars"]["MESH_LLM_MODELS_DIR"],
        "/tmp/shared-mesh-models"
    );
    assert_eq!(
        env_resp["allowlisted_vars"]["MESH_LLM_PLUGIN_NAME"],
        "mesh-dev"
    );
    assert!(
        env_resp["allowlisted_vars"]
            .get("MESH_LLM_JOIN_TOKEN")
            .is_none(),
        "secret token must never appear in env_probe"
    );
    assert!(env_resp["redacted_mesh_llm_var_count"].as_u64().unwrap() >= 1);

    // After startup delay elapses, HealthRequest transitions to state=ready.
    tokio::time::sleep(Duration::from_millis(260)).await;
    let h_ready = exchange(
        &mut stream,
        4,
        proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
    )
    .await?;
    let proto::envelope::Payload::HealthResponse(hr_ready) = h_ready else {
        bail!("expected HealthResponse after loading, got {h_ready:?}");
    };
    assert!(
        hr_ready.detail.contains("state=ready"),
        "expected state=ready, got: {}",
        hr_ready.detail
    );

    drop(stream);
    let _ = timeout(Duration::from_secs(5), child.wait()).await;
    Ok(())
}
