use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

use mesh_dev::cli::{Cli, Command};
use mesh_dev::mem_probe::probe_memory;
use mesh_dev::mesh_adapter::{build_app_state, build_manifest_with_options, run_mesh_plugin};
use mesh_dev::server::serve_with_state;
use mesh_dev::suite::{SuiteRunOptions, run_suites};

fn main() -> Result<()> {
    // Log strictly to stderr so stdout stays clean for Mesh-LLM plugin IPC and CLI JSON output.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    if let Some(Command::Manifest { dev }) = cli.command {
        let manifest = build_manifest_with_options("http://127.0.0.1:0/v1", "mesh-dev", &dev);
        println!("{}", mesh_llm_plugin::package_manifest_json(&manifest)?);
        return Ok(());
    }

    if let Some(Command::MemProbe { gpu_probe }) = cli.command {
        let snap = probe_memory(gpu_probe);
        println!("{}", serde_json::to_string_pretty(&snap)?);
        return Ok(());
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .context("build tokio runtime")?;

    let result = rt.block_on(async move {
        match cli.command {
            Some(Command::TestSuite {
                suites,
                mesh_llm,
                archive,
                port,
                console_port,
                bind_port,
                repeat,
                keep_logs,
            }) => {
                let opts = SuiteRunOptions {
                    mesh_llm_bin: mesh_llm,
                    archive,
                    port,
                    console_port,
                    bind_port,
                    repeat,
                    keep_logs,
                };
                run_suites(&suites, &opts).await
            }
            Some(Command::Serve { host, port, dev }) => {
                let state = build_app_state(&dev);
                let addr = format!("{host}:{port}");
                let listener = TcpListener::bind(&addr)
                    .await
                    .with_context(|| format!("bind standalone HTTP listener on {addr}"))?;
                let bound = listener.local_addr()?;
                tracing::info!(
                    addr = %bound,
                    models = state.enabled_models.len(),
                    "starting standalone mesh-dev OpenAI server"
                );
                let shutdown_state = state.clone();
                tokio::spawn(async move {
                    let _ = tokio::signal::ctrl_c().await;
                    shutdown_state.request_shutdown();
                });
                serve_with_state(listener, state).await
            }
            Some(Command::Plugin { dev }) => run_mesh_plugin(dev).await,
            None => run_mesh_plugin(cli.plugin_args).await,
            Some(Command::Manifest { .. } | Command::MemProbe { .. }) => unreachable!(),
        }
    });

    // Bound runtime teardown so host EOF / SIGTERM always exits promptly.
    rt.shutdown_timeout(Duration::from_secs(2));
    result
}
