//! Command-line configuration and startup fault-injection flags for `mesh-dev`.

use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};

use crate::models::ALL_MODELS;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum GpuProbeMode {
    /// Automatically detect Tegra UMA, DRM sysfs, or `nvidia-smi` when available.
    #[default]
    Auto,
    /// Disable GPU memory probing (read `/proc/meminfo` and self RSS only).
    Off,
    /// Query `nvidia-smi` with a short timeout.
    NvidiaSmi,
    /// Read Linux DRM sysfs VRAM counters only.
    Drm,
}

#[derive(Debug, Parser)]
#[command(
    name = "mesh-dev",
    version,
    about = "Zero-GPU developer & test plugin for Mesh-LLM: programmable fake OpenAI models, supervision fault injection, and live observation/exchange inspection."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Plugin runtime arguments used when launched directly by `mesh-llm`.
    #[command(flatten)]
    pub plugin_args: DevArgs,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run as a standalone OpenAI-compatible HTTP server (no `mesh-llm` required).
    Serve {
        /// Listen host address.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Listen TCP port (`0` for ephemeral).
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[command(flatten)]
        dev: DevArgs,
    },
    /// Run as a Mesh-LLM v3 IPC plugin (`MESH_LLM_PLUGIN_ENDPOINT`; default when no subcommand is passed).
    Plugin {
        #[command(flatten)]
        dev: DevArgs,
    },
    /// Print the packaged `PluginManifest` JSON to stdout and exit.
    Manifest {
        #[command(flatten)]
        dev: DevArgs,
    },
    /// Run a one-shot live memory probe and print JSON to stdout.
    MemProbe {
        #[arg(long, value_enum, default_value_t = GpuProbeMode::Auto)]
        gpu_probe: GpuProbeMode,
    },
    /// Execute one or more declarative TOML E2E test suites against a real `mesh-llm` binary.
    TestSuite {
        /// Paths to `.toml` suite files.
        #[arg(required = true)]
        suites: Vec<std::path::PathBuf>,
        /// Path or command name of the `mesh-llm` binary to test against.
        #[arg(long, env = "MESH_LLM_BIN", default_value = "mesh-llm")]
        mesh_llm: String,
        /// Optional path to a packaged `mesh-dev-*.tar.gz` archive (defaults to invoking the current `mesh-dev` binary directly).
        #[arg(long)]
        archive: Option<std::path::PathBuf>,
        /// Isolated OpenAI HTTP port.
        #[arg(long, default_value_t = 19337)]
        port: u16,
        /// Isolated console/control API port.
        #[arg(long, default_value_t = 13131)]
        console_port: u16,
        /// Isolated mesh P2P bind port.
        #[arg(long, default_value_t = 17842)]
        bind_port: u16,
    },
}

#[derive(Debug, Clone, clap::Args)]
pub struct DevArgs {
    /// Comma-separated subset of `dev/*` model IDs to advertise (defaults to all built-in models).
    #[arg(long, env = "MESH_DEV_MODELS", value_delimiter = ',')]
    pub models: Vec<String>,

    /// Maximum simultaneous in-flight HTTP inference requests before returning synchronous `503`
    /// (defaults to `1` for deterministic concurrency/spillover routing A/B tests).
    #[arg(
        long,
        alias = "max-concurrent",
        env = "MESH_DEV_MAX_CONCURRENCY",
        default_value_t = 1
    )]
    pub max_concurrency: usize,

    /// Optional per-token delay in milliseconds applied to streaming `dev/echo` and slow SSE pacing
    /// (e.g. `500` to simulate a 2 tok/s node vs a fast peer in a 2-node mesh).
    #[arg(long, env = "MESH_DEV_TOKEN_DELAY_MS", default_value_t = 0)]
    pub token_delay_ms: u64,

    /// Keep the plugin in a `"loading"` state for `<N>` seconds after initialization before
    /// exposing `/v1/models` (for testing `ENDPOINT_STARTUP_GRACE_SECS = 30s` and health transitions).
    #[arg(long, env = "MESH_DEV_STARTUP_DELAY_SECS", default_value_t = 0)]
    pub startup_delay_secs: u64,

    /// Sub-second loading delay in milliseconds (additive with `--startup-delay-secs`, useful in fast tests).
    #[arg(long, env = "MESH_DEV_STARTUP_DELAY_MS", default_value_t = 0)]
    pub startup_delay_ms: u64,

    /// Ring buffer capacity per inspector source (`http`, `observation`, `exchange`).
    #[arg(long, env = "MESH_DEV_RING_CAPACITY", default_value_t = 256)]
    pub ring_capacity: usize,

    /// Default TTFT delay in milliseconds for `dev/slow-ttft-15s`.
    #[arg(long, env = "MESH_DEV_SLOW_TTFT_MS", default_value_t = 15_000)]
    pub slow_ttft_ms: u64,

    /// Default per-token interval in milliseconds for `dev/slow-sse-1tps`.
    #[arg(long, env = "MESH_DEV_SLOW_SSE_INTERVAL_MS", default_value_t = 1_000)]
    pub slow_sse_interval_ms: u64,

    /// Default number of valid SSE chunks flushed before TCP drop on `dev/mid-stream-drop`.
    #[arg(long, env = "MESH_DEV_MID_STREAM_CHUNKS", default_value_t = 3)]
    pub mid_stream_chunks: u32,

    /// GPU memory probing strategy for `dev/mem-probe`.
    #[arg(long, env = "MESH_DEV_GPU_PROBE", value_enum, default_value_t = GpuProbeMode::Auto)]
    pub gpu_probe: GpuProbeMode,

    /// Return `PluginError::startup_disabled` during `InitializeRequest` (optional custom reason).
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "startup disabled by --startup-disabled"
    )]
    pub startup_disabled: Option<String>,

    /// Artificial delay in milliseconds before responding to `InitializeRequest`.
    #[arg(long, env = "MESH_DEV_INIT_DELAY_MS", default_value_t = 0)]
    pub init_delay_ms: u64,

    /// Exit immediately with the given exit code on startup (for crash-loop backoff testing).
    #[arg(long)]
    pub exit_on_start: Option<i32>,

    /// Boot with `HealthRequest` failing immediately (optional custom error detail).
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "injected unhealthy on start"
    )]
    pub unhealthy_on_start: Option<String>,

    /// Opt in to declaring `request_transform.v1` (off by default because transforms fail closed on all requests).
    #[arg(
        long,
        env = "MESH_DEV_ENABLE_REQUEST_TRANSFORM",
        default_value_t = false
    )]
    pub enable_request_transform: bool,

    /// Disable advertising the built-in `dev/virtual-echo` virtual model.
    #[arg(long, env = "MESH_DEV_NO_VIRTUAL_MODEL", default_value_t = false)]
    pub no_virtual_model: bool,

    /// Explicitly allow running on a public mesh (refused by default to prevent exposing dev fault ops).
    #[arg(long, env = "MESH_DEV_ALLOW_PUBLIC_MESH", default_value_t = false)]
    pub allow_public_mesh: bool,
}

impl Default for DevArgs {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            max_concurrency: 1,
            token_delay_ms: 0,
            startup_delay_secs: 0,
            startup_delay_ms: 0,
            ring_capacity: 256,
            slow_ttft_ms: 15_000,
            slow_sse_interval_ms: 1_000,
            mid_stream_chunks: 3,
            gpu_probe: GpuProbeMode::Auto,
            startup_disabled: None,
            init_delay_ms: 0,
            exit_on_start: None,
            unhealthy_on_start: None,
            enable_request_transform: false,
            no_virtual_model: false,
            allow_public_mesh: false,
        }
    }
}

impl DevArgs {
    /// Resolve the enabled model IDs, defaulting to all built-in models if `--models` is empty.
    pub fn resolved_models(&self) -> Vec<String> {
        if self.models.is_empty() {
            ALL_MODELS.iter().map(|m| (*m).to_string()).collect()
        } else {
            self.models
                .iter()
                .map(|m| m.trim())
                .filter(|m| !m.is_empty())
                .map(ToString::to_string)
                .collect()
        }
    }

    pub fn total_startup_delay(&self) -> Duration {
        Duration::from_secs(self.startup_delay_secs) + Duration::from_millis(self.startup_delay_ms)
    }
}
