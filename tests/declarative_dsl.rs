//! Conformance tests for `mesh_llm_plugin::plugin!` declarative Rust DSL.
//!
//! Exercises every macro arm exposed by `mesh_llm_plugin 0.80.0` over an in-memory
//! `LocalStream::Memory` duplex pipe and locks down observed SDK contract behaviors
//! and confirmed bugs (such as synthetic `http_get__*` HTTP bindings leaking into
//! `manifest.operations` and `tools/list`).

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use mesh_llm_plugin::{
    LocalStream, PROTOCOL_VERSION, PluginMetadata, PluginRuntime, PluginStartupPolicy, capability,
    complete_result, config_schema, config_setting, config_string, events, get_prompt_result,
    host_capabilities, http, inference, mcp, mesh, plugin, plugin_server_info_full, proto,
    read_envelope, read_resource_result, write_envelope,
};
use rmcp::model::{ContentBlock, PromptMessage, ResourceContents, Role};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::timeout;

#[derive(Debug, Deserialize, JsonSchema)]
struct EchoArgs {
    message: String,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
struct EmptyArgs {}

async fn exchange(
    stream: &mut LocalStream,
    id: u64,
    payload: proto::envelope::Payload,
) -> Result<proto::envelope::Payload> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "dsl-probe".into(),
            request_id: id,
            payload: Some(payload),
        },
    )
    .await?;
    let resp = timeout(Duration::from_secs(5), read_envelope(stream)).await??;
    assert_eq!(resp.request_id, id);
    resp.payload.context("missing response payload")
}

async fn rpc(stream: &mut LocalStream, id: u64, method: &str, params: Value) -> Result<Value> {
    let payload = exchange(
        stream,
        id,
        proto::envelope::Payload::RpcRequest(proto::RpcRequest {
            method: method.into(),
            params_json: params.to_string(),
            invocation_id: None,
        }),
    )
    .await?;
    let proto::envelope::Payload::RpcResponse(resp) = payload else {
        bail!("expected RpcResponse for {method}, got {payload:?}");
    };
    Ok(serde_json::from_str(&resp.result_json)?)
}

fn build_dsl_probe_plugin(
    initialized: Arc<AtomicBool>,
    mesh_events: Arc<AtomicUsize>,
    channel_msgs: Arc<AtomicUsize>,
) -> mesh_llm_plugin::SimplePlugin {
    plugin! {
        metadata: PluginMetadata::new(
            "dsl-probe",
            "0.1.0",
            plugin_server_info_full(
                "dsl-probe",
                "0.1.0",
                "DSL Probe",
                "Declarative plugin! conformance probe",
                None::<String>,
            ),
        ),
        startup_policy: PluginStartupPolicy::Any,
        provides: [
            capability("dsl.probe.v1"),
        ],
        config: [
            config_schema("dsl-probe")
                .setting(
                    config_setting("marker", config_string())
                        .default_value(&json!("ok")),
                ),
        ],
        mesh: [
            mesh::local_channel("dsl-probe/bus"),
        ],
        events: [
            events::peer_up(),
            events::peer_down(),
        ],
        mcp: [
            mcp::tool("echo")
                .description("Echo a message")
                .input::<EchoArgs>()
                .handle(|args, _ctx| Box::pin(async move {
                    Ok(json!({ "echoed": args.message }))
                })),
            mcp::resource("dsl://info")
                .name("info")
                .description("Static DSL info")
                .mime_type("application/json")
                .handle(|req, _ctx| Box::pin(async move {
                    Ok(read_resource_result(vec![ResourceContents::text(
                        json!({ "ok": true }).to_string(),
                        req.uri,
                    )]))
                })),
            mcp::resource_template("dsl://item/{id}")
                .name("item")
                .description("Parameterized DSL item")
                .mime_type("application/json")
                .handle(|req, _ctx| Box::pin(async move {
                    let id = req.uri.strip_prefix("dsl://item/").unwrap_or("unknown");
                    Ok(read_resource_result(vec![ResourceContents::text(
                        json!({ "id": id }).to_string(),
                        req.uri,
                    )]))
                })),
            // Note: McpPromptBuilder in dsl.rs:713 has no `.argument(...)` builder method
            // and hardcodes `arguments: None` in `prompts/list`.
            mcp::prompt("greet")
                .description("Greet a user")
                .handle(|req, _ctx| Box::pin(async move {
                    let name = req
                        .arguments
                        .as_ref()
                        .and_then(|m| m.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or("world");
                    Ok(get_prompt_result(vec![PromptMessage::new(
                        Role::User,
                        ContentBlock::text(format!("hello {name}")),
                    )]))
                })),
            mcp::completion("prompt.greet.name")
                .handle(|req, _ctx| Box::pin(async move {
                    let vals = ["alice", "bob", "charlie"]
                        .into_iter()
                        .filter(|v| v.starts_with(&req.argument.value))
                        .map(String::from)
                        .collect();
                    complete_result(vals)
                })),
        ],
        http: [
            http::get("/ping")
                .description("Stapled GET ping")
                .input::<EmptyArgs>()
                .handle(|_args, _ctx| Box::pin(async move {
                    Ok(json!({ "pong": true }))
                })),
            http::post("/echo")
                .description("Stapled POST echo")
                .input::<EchoArgs>()
                .handle(|args, _ctx| Box::pin(async move {
                    Ok(json!({ "http_echo": args.message }))
                })),
        ],
        inference: [
            inference::provider("dsl-openai", "http://127.0.0.1:19999/v1"),
        ],
        health: |_ctx| Box::pin(async { Ok("ready".to_string()) }),
        on_initialized: move |_ctx| {
            let initialized = initialized.clone();
            Box::pin(async move {
                initialized.store(true, Ordering::SeqCst);
                Ok(())
            })
        },
        on_channel_message: move |_msg, _ctx| {
            let channel_msgs = channel_msgs.clone();
            Box::pin(async move {
                channel_msgs.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        },
        on_mesh_event: move |_ev, _ctx| {
            let mesh_events = mesh_events.clone();
            Box::pin(async move {
                mesh_events.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        },
    }
}

#[tokio::test]
async fn declarative_plugin_macro_exercises_all_arms_and_reproduces_http_tool_leak() -> Result<()> {
    let initialized = Arc::new(AtomicBool::new(false));
    let mesh_events = Arc::new(AtomicUsize::new(0));
    let channel_msgs = Arc::new(AtomicUsize::new(0));

    let plugin = build_dsl_probe_plugin(
        initialized.clone(),
        mesh_events.clone(),
        channel_msgs.clone(),
    );

    let (plugin_half, host_half) = tokio::io::duplex(64 * 1024);
    let runtime = tokio::spawn(PluginRuntime::run_with_stream(
        plugin,
        LocalStream::Memory(plugin_half),
    ));
    let mut host = LocalStream::Memory(host_half);

    // 1. Initialize handshake with local_channels.v1.
    let init = exchange(
        &mut host,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION,
            host_version: "dsl-test-host".into(),
            host_capabilities: vec![host_capabilities::LOCAL_CHANNELS.into()],
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::InitializeResponse(init_resp) = init else {
        bail!("expected InitializeResponse, got {init:?}");
    };
    let manifest = init_resp.manifest.context("missing manifest")?;
    assert_eq!(manifest.capabilities, vec!["dsl.probe.v1"]);
    assert!(manifest.config_schema.is_some());
    assert_eq!(manifest.mesh_channels.len(), 1);
    assert_eq!(manifest.mesh_event_subscriptions.len(), 2);
    assert_eq!(manifest.endpoints.len(), 1);
    assert_eq!(manifest.http_bindings.len(), 2);

    // WORKAROUND(mesh-llm-plugin): CONFIRMED UPSTREAM BUG #1 (tracking: kleinbem/mesh-dev-plugin#4):
    // `dsl.rs::HttpRouteBuilder::handle` creates synthetic `http_get__ping` / `http_post__echo`
    // operations (note: `trim_matches('_')` even strips the leading `__`!) and adds them to
    // BOTH `manifest.operations` AND `OperationRouter`, leaking stapled HTTP routes into MCP
    // `tools/list` and `/api/plugins/<name>/tools`. Flip these assertions once the staged fix
    // lands upstream and `mesh-llm-plugin` is bumped.
    let op_names: Vec<&str> = manifest
        .operations
        .iter()
        .map(|o| o.name.as_str())
        .collect();
    assert!(op_names.contains(&"echo"), "op_names: {op_names:?}");
    assert!(
        op_names.contains(&"http_get__ping") && op_names.contains(&"http_post__echo"),
        "expected dsl.rs synthetic HTTP operations in manifest.operations, got {op_names:?}"
    );

    // Confirm the same leak appears in MCP `tools/list`.
    let tools = rpc(&mut host, 2, "tools/list", json!({})).await?;
    let tool_names: Vec<&str> = tools["tools"]
        .as_array()
        .context("tools array")?
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert!(tool_names.contains(&"echo"));
    assert!(
        tool_names.contains(&"http_get__ping"),
        "expected leaked HTTP binding tool in tools/list, got {tool_names:?}"
    );

    // 2. Call tool, read exact + template resource, get prompt, run completion.
    let echo_out = rpc(
        &mut host,
        3,
        "tools/call",
        json!({ "name": "echo", "arguments": { "message": "hi-dsl" } }),
    )
    .await?;
    assert_eq!(echo_out["structuredContent"]["echoed"], "hi-dsl");

    let res_list = rpc(&mut host, 4, "resources/list", json!({})).await?;
    assert_eq!(res_list["resources"][0]["uri"], "dsl://info");
    let res_read = rpc(
        &mut host,
        5,
        "resources/read",
        json!({ "uri": "dsl://info" }),
    )
    .await?;
    assert!(
        res_read["contents"][0]["text"]
            .as_str()
            .unwrap_or("")
            .contains("\"ok\":true")
    );

    let tpl_list = rpc(&mut host, 6, "resources/templates/list", json!({})).await?;
    assert_eq!(
        tpl_list["resourceTemplates"][0]["uriTemplate"],
        "dsl://item/{id}"
    );
    let tpl_read = rpc(
        &mut host,
        7,
        "resources/read",
        json!({ "uri": "dsl://item/42" }),
    )
    .await?;
    assert!(
        tpl_read["contents"][0]["text"]
            .as_str()
            .unwrap_or("")
            .contains("\"id\":\"42\"")
    );

    let prompt_out = rpc(
        &mut host,
        8,
        "prompts/get",
        json!({ "name": "greet", "arguments": { "name": "martin" } }),
    )
    .await?;
    assert_eq!(prompt_out["messages"][0]["content"]["text"], "hello martin");

    let comp_out = rpc(
        &mut host,
        9,
        "completion/complete",
        json!({
            "ref": { "type": "ref/prompt", "name": "greet" },
            "argument": { "name": "name", "value": "bo" }
        }),
    )
    .await?;
    assert_eq!(comp_out["completion"]["values"], json!(["bob"]));

    // 3. Deliver MeshEvent & ChannelMessage notifications.
    write_envelope(
        &mut host,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "dsl-probe".into(),
            request_id: 10,
            payload: Some(proto::envelope::Payload::MeshEvent(proto::MeshEvent {
                kind: proto::mesh_event::Kind::PeerUp as i32,
                local_peer_id: "peer-1".into(),
                ..Default::default()
            })),
        },
    )
    .await?;
    write_envelope(
        &mut host,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "dsl-probe".into(),
            request_id: 11,
            payload: Some(proto::envelope::Payload::ChannelMessage(
                proto::ChannelMessage {
                    channel: "dsl-probe/bus".into(),
                    source_peer_id: "local".into(),
                    target_peer_id: String::new(),
                    content_type: "application/json".into(),
                    body: br#"{"ping":1}"#.to_vec(),
                    message_kind: "ping".into(),
                    correlation_id: String::new(),
                    metadata_json: String::new(),
                },
            )),
        },
    )
    .await?;

    // Health check acts as a barrier after ordered notifications settle.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let health = exchange(
        &mut host,
        12,
        proto::envelope::Payload::HealthRequest(proto::HealthRequest {}),
    )
    .await?;
    assert!(matches!(
        health,
        proto::envelope::Payload::HealthResponse(_)
    ));
    assert!(initialized.load(Ordering::SeqCst));
    assert_eq!(mesh_events.load(Ordering::SeqCst), 1);
    assert_eq!(channel_msgs.load(Ordering::SeqCst), 1);

    // Clean shutdown.
    let shut = exchange(
        &mut host,
        13,
        proto::envelope::Payload::ShutdownRequest(proto::ShutdownRequest {
            reason: "test complete".into(),
        }),
    )
    .await?;
    assert!(matches!(
        shut,
        proto::envelope::Payload::ShutdownResponse(_)
    ));
    let _ = timeout(Duration::from_secs(3), runtime).await??;
    Ok(())
}

#[tokio::test]
async fn missing_local_channels_capability_returns_invalid_request_before_protocol_check()
-> Result<()> {
    let plugin = build_dsl_probe_plugin(
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let (plugin_half, host_half) = tokio::io::duplex(64 * 1024);
    let runtime = tokio::spawn(PluginRuntime::run_with_stream(
        plugin,
        LocalStream::Memory(plugin_half),
    ));
    let mut host = LocalStream::Memory(host_half);

    // WORKAROUND(mesh-llm-plugin): Even with an incompatible host_protocol_version,
    // PluginRuntime checks `host_capabilities::unsupported_requirement` first and returns
    // -32600 (INVALID_REQUEST) instead of -32010 (STARTUP_DISABLED_ERROR_CODE) on v0.80.0
    // (upstream: Mesh-LLM/mesh-llm#3189). Flip to -32010 once #3189 merges and Cargo.toml bumps.
    let init = exchange(
        &mut host,
        1,
        proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
            host_protocol_version: PROTOCOL_VERSION + 99,
            host_version: "old-host".into(),
            host_capabilities: Vec::new(),
            ..Default::default()
        }),
    )
    .await?;
    let proto::envelope::Payload::ErrorResponse(err) = init else {
        bail!("expected ErrorResponse, got {init:?}");
    };
    assert_eq!(err.code, -32600);
    assert!(err.message.contains("local_channels.v1"), "{}", err.message);
    let _ = timeout(Duration::from_secs(3), runtime).await??;
    Ok(())
}
