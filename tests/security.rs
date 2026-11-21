//! Security hardening tests for `mesh-dev`:
//! - Oversized HTTP request body rejection (`413 Payload Too Large`, 4 MiB cap)
//! - Expanded sensitive header redaction (matching `mesh_llm_config::safe_exchange_header`) & value truncation
//! - Bounded SEP-2663 `TaskStore` capacity and label length limits
//! - Constant-time side-stream auth token comparison & CSPRNG UUIDv4 formatting
//! - Refusal to initialize on `MeshVisibility::Public` unless `--allow-public-mesh` is explicitly set

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use axum::http::{HeaderMap, HeaderValue};
use mesh_dev::capabilities::{
    MAX_TASK_LABEL_LEN, MAX_TRACKED_TASKS, SharedTaskStore, constant_time_eq, create_tracked_task,
};
use mesh_dev::faults::FaultController;
use mesh_dev::inspector::Inspector;
use mesh_dev::models::{ALL_MODELS, ModelDefaults};
use mesh_dev::server::openai::{capture_headers, is_sensitive_header};
use mesh_dev::server::{AppState, MAX_HTTP_BODY_BYTES, serve_with_state};
use mesh_llm_plugin::{
    LocalStream, PROTOCOL_VERSION, TaskStore, host_capabilities, proto, read_envelope,
    write_envelope,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::time::timeout;

#[test]
fn sensitive_header_redaction_covers_tokens_secrets_keys_and_control_headers() {
    for sensitive in [
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-mesh-join-token",
        "x-my-secret-header",
        "x-session-id",
        "x-jwt-assertion",
        "x-owner-sig",
        "x-mesh-owner-id",
        "x-mesh-control-admin",
        "x-custom-auth",
    ] {
        assert!(
            is_sensitive_header(sensitive),
            "expected {sensitive} to be treated as sensitive"
        );
    }

    for safe in ["x-mesh-dev-nonce", "x-request-id", "content-type", "host"] {
        assert!(
            !is_sensitive_header(safe),
            "expected {safe} to remain non-sensitive"
        );
    }

    let mut map = HeaderMap::new();
    map.insert(
        "authorization",
        HeaderValue::from_static("Bearer top-secret"),
    );
    map.insert("x-mesh-join-token", HeaderValue::from_static("join-secret"));
    map.insert("x-owner-sig", HeaderValue::from_static("owner-sig-123"));
    map.insert("x-mesh-dev-nonce", HeaderValue::from_static("nonce-ok"));
    let huge = "a".repeat(2048);
    map.insert(
        "x-long-header",
        HeaderValue::from_str(&huge).expect("valid header"),
    );

    let captured = capture_headers(&map);
    assert_eq!(
        captured.get("authorization").map(String::as_str),
        Some("<redacted>")
    );
    assert_eq!(
        captured.get("x-mesh-join-token").map(String::as_str),
        Some("<redacted>")
    );
    assert_eq!(
        captured.get("x-owner-sig").map(String::as_str),
        Some("<redacted>")
    );
    assert_eq!(
        captured.get("x-mesh-dev-nonce").map(String::as_str),
        Some("nonce-ok")
    );
    let truncated = captured.get("x-long-header").expect("long header captured");
    assert!(truncated.ends_with("...<truncated>"));
    assert!(truncated.len() < 1100);
}

#[test]
fn constant_time_eq_handles_equal_unequal_and_different_lengths() {
    assert!(constant_time_eq(b"abc123", b"abc123"));
    assert!(!constant_time_eq(b"abc123", b"abc124"));
    assert!(!constant_time_eq(b"abc123", b"abc12"));
    assert!(constant_time_eq(b"", b""));
}

#[tokio::test]
async fn task_store_enforces_label_and_capacity_bounds() {
    let store: SharedTaskStore = Arc::new(Mutex::new(TaskStore::default()));

    let long_label = "x".repeat(MAX_TASK_LABEL_LEN + 1);
    let err = create_tracked_task(&store, long_label)
        .await
        .expect_err("oversized label must be rejected");
    assert!(err.to_string().contains("maximum length"), "{err}");

    for i in 0..MAX_TRACKED_TASKS {
        create_tracked_task(&store, format!("task-{i}"))
            .await
            .expect("task within cap");
    }
    let cap_err = create_tracked_task(&store, "overflow".into())
        .await
        .expect_err("exceeding MAX_TRACKED_TASKS must fail");
    assert!(cap_err.to_string().contains("capacity"), "{cap_err}");
}

#[tokio::test]
async fn oversized_http_body_is_rejected_with_413() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let state = AppState::new(
        ALL_MODELS.iter().map(|s| (*s).to_string()).collect(),
        true,
        ModelDefaults::default(),
        4,
        FaultController::default(),
        Inspector::new(32),
    );
    let shutdown = state.clone();
    let server = tokio::spawn(async move {
        let _ = serve_with_state(listener, state).await;
    });

    // Send a 4.5 MiB payload (> MAX_HTTP_BODY_BYTES = 4 MiB).
    let oversized = format!(
        r#"{{"model":"dev/echo","messages":[{{"role":"user","content":"{}"}}]}}"#,
        "A".repeat(MAX_HTTP_BODY_BYTES + 256 * 1024)
    );
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        oversized.len(),
        oversized
    );
    let mut stream = TcpStream::connect(("127.0.0.1", port)).await?;
    let _ = stream.write_all(req.as_bytes()).await;
    let mut resp = Vec::new();
    let _ = timeout(Duration::from_secs(5), stream.read_to_end(&mut resp)).await;
    let text = String::from_utf8_lossy(&resp);
    assert!(
        text.starts_with("HTTP/1.1 413"),
        "expected HTTP 413 Payload Too Large, got: {}",
        text.lines().next().unwrap_or("<empty>")
    );

    shutdown.request_shutdown();
    let _ = timeout(Duration::from_secs(2), server).await;
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn public_mesh_is_refused_by_default_and_allowed_with_explicit_flag() -> Result<()> {
    use std::process::Stdio;
    use tokio::net::UnixListener;
    use tokio::process::Command;

    async fn try_init_public(extra_args: &[&str]) -> Result<proto::envelope::Payload> {
        let dir = tempfile::tempdir()?;
        let socket = dir.path().join("host.sock");
        let listener = UnixListener::bind(&socket)?;
        let mut child = Command::new(env!("CARGO_BIN_EXE_mesh-dev"))
            .args(extra_args)
            .env("MESH_LLM_PLUGIN_ENDPOINT", &socket)
            .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
            .env("MESH_LLM_PLUGIN_NAME", "mesh-dev")
            .env("RUST_LOG", "warn")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let (stream, _) = timeout(Duration::from_secs(10), listener.accept()).await??;
        let mut stream = LocalStream::Unix(stream);
        write_envelope(
            &mut stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "mesh-dev".into(),
                request_id: 1,
                payload: Some(proto::envelope::Payload::InitializeRequest(
                    proto::InitializeRequest {
                        host_protocol_version: PROTOCOL_VERSION,
                        host_version: "security-test-host".into(),
                        mesh_visibility: proto::MeshVisibility::Public as i32,
                        host_capabilities: vec![host_capabilities::LOCAL_CHANNELS.into()],
                        ..Default::default()
                    },
                )),
            },
        )
        .await?;
        let resp = timeout(Duration::from_secs(10), read_envelope(&mut stream)).await??;
        drop(stream);
        let _ = timeout(Duration::from_secs(3), child.wait()).await;
        resp.payload.context("missing init response payload")
    }

    // Default: refuses public mesh with STARTUP_DISABLED_ERROR_CODE (clean disable, no crash-loop).
    let refused = try_init_public(&[]).await?;
    let proto::envelope::Payload::ErrorResponse(err) = refused else {
        bail!("expected ErrorResponse on public mesh by default, got {refused:?}");
    };
    assert_eq!(err.code, mesh_llm_plugin::STARTUP_DISABLED_ERROR_CODE);
    assert!(err.message.contains("public mesh"), "{}", err.message);

    // Explicit opt-in: succeeds on public mesh.
    let allowed = try_init_public(&["--allow-public-mesh"]).await?;
    assert!(
        matches!(allowed, proto::envelope::Payload::InitializeResponse(_)),
        "expected InitializeResponse with --allow-public-mesh, got {allowed:?}"
    );
    Ok(())
}
