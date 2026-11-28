//! Declarative TOML E2E test suite runner (`mesh-dev test-suite <suite.toml>...`).
//!
//! Launches an isolated `mesh-llm` instance (temporary `HOME`, non-conflicting ports),
//! executes declarative HTTP / OpenAI / plugin-op / Streamable-HTTP MCP / restart steps,
//! evaluates assertions, and prints a structured pass/fail summary.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::Command;
use tokio::time::{sleep, timeout};

#[derive(Debug, Clone, Deserialize)]
pub struct SuiteSpec {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub mesh: MeshSuiteConfig,
    #[serde(default, rename = "step")]
    pub steps: Vec<StepSpec>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct MeshSuiteConfig {
    /// Extra CLI arguments passed to `mesh-dev` via `[[plugin]] args = [...]`.
    #[serde(default)]
    pub plugin_args: Vec<String>,
    /// Extra environment variables passed to `mesh-dev` via `[plugin.env]`.
    #[serde(default)]
    pub plugin_env: BTreeMap<String, String>,
    #[serde(default)]
    pub allow_inference_observation: bool,
    #[serde(default)]
    pub allow_peer_blocks: bool,
    /// Optional `[plugin.settings]` table validated against `mesh-dev`'s `config_schema`.
    #[serde(default)]
    pub settings: Option<toml::Table>,
    /// Optional `[plugin.openai_exchange_grant]` table.
    #[serde(default)]
    pub openai_exchange_grant: Option<toml::Table>,
    /// Skip waiting for `dev/echo` on `/v1/models` before step 1 (e.g. for `--startup-delay-secs` tests).
    #[serde(default)]
    pub skip_initial_model_wait: bool,
    /// Set `exclusive_gpu = true` on the `mesh-dev` plugin entry.
    #[serde(default)]
    pub exclusive_gpu: bool,
    /// Start a serving node instead of `--client`, so `[[models]]` load on this host.
    #[serde(default)]
    pub serve: bool,
    /// Raw `[[models]]` entries; `${VAR}` in strings expands from the runner's environment.
    #[serde(default)]
    pub models: Vec<toml::Table>,
    /// Extra top-level tables merged into the generated `config.toml`.
    #[serde(default)]
    pub config: Option<toml::Table>,
    /// Extra environment for the `mesh-llm` process (for example `RUST_LOG`).
    #[serde(default)]
    pub node_env: BTreeMap<String, String>,
    /// Skip the suite instead of failing it when one of these runner variables is unset.
    #[serde(default)]
    pub requires_env: Vec<String>,
    /// How long to wait for the console to come up (native runtimes start slower).
    #[serde(default)]
    pub ready_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StepSpec {
    pub name: String,
    /// Invoke a plugin operation (`POST /api/plugins/mesh-dev/tools/<op>`).
    #[serde(default)]
    pub op: Option<String>,
    /// JSON input for `op` (defaults to `{}`).
    #[serde(default)]
    pub input: Option<toml::Value>,
    /// Call an OpenAI proxy endpoint on the OpenAI port (`--port`).
    #[serde(default)]
    pub openai: Option<HttpCallSpec>,
    /// Call a control/console API endpoint on the console port (`--console`).
    #[serde(default)]
    pub console_http: Option<HttpCallSpec>,
    /// Call an MCP JSON-RPC method over Streamable HTTP `POST /mcp` on the console port.
    #[serde(default)]
    pub mcp: Option<McpCallSpec>,
    /// Sleep for the given number of milliseconds.
    #[serde(default)]
    pub sleep_ms: Option<u64>,
    /// Poll `GET /v1/models` until the specified model appears (in milliseconds, default 30s).
    #[serde(default)]
    pub wait_model: Option<String>,
    /// Poll `dev/echo` until `plugin_instance_id` differs from the saved variable name.
    #[serde(default)]
    pub wait_restart_from: Option<String>,
    /// Optional timeout in milliseconds for wait/HTTP operations (default: 20000).
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Exact expected HTTP status code.
    #[serde(default)]
    pub expect_status: Option<u16>,
    /// Inclusive expected HTTP status range `[min, max]`.
    #[serde(default)]
    pub expect_status_range: Option<[u16; 2]>,
    /// Substrings that MUST appear in the response body.
    #[serde(default)]
    pub expect_contains: Vec<String>,
    /// Substrings that MUST NOT appear in the response body.
    #[serde(default)]
    pub expect_not_contains: Vec<String>,
    /// Expected response headers (case-insensitive header name -> expected substring).
    #[serde(default)]
    pub expect_headers: BTreeMap<String, String>,
    /// Response header names (case-insensitive) that MUST NOT be present.
    #[serde(default)]
    pub expect_absent_headers: Vec<String>,
    /// Dotted JSON path assertions (`"path.to.field" = value` or `{ gte = N }`, `{ ne = "..." }`, `{ exists = true }`, `{ len_gte = N }`, `{ contains = "..." }`).
    #[serde(default)]
    pub expect_json: BTreeMap<String, toml::Value>,
    /// Save dotted JSON paths from the response into variables (`var_name = "path.to.field"`).
    #[serde(default)]
    pub save_json: BTreeMap<String, String>,
    /// Record failures as non-fatal XFAIL/finding notes instead of aborting (for documenting known upstream bugs).
    #[serde(default)]
    pub expect_failure_note: Option<String>,
    /// Read a file (`${NODE_HOME}` is the node's temporary HOME) into a variable.
    #[serde(default)]
    pub read_file: Option<ReadFileSpec>,
    /// Fail unless the named background request is still waiting for its response.
    #[serde(default)]
    pub expect_pending: Option<String>,
    /// Print the response status and body (for writing suites).
    #[serde(default)]
    pub dump: bool,
    /// Join the named background request (`openai.background`) and check its response.
    #[serde(default)]
    pub await_background: Option<String>,
    /// Abort the named background request, closing its socket: a client disconnect.
    #[serde(default)]
    pub drop_background: Option<String>,
    /// Retry the step until its assertions pass or `timeout_ms` runs out.
    #[serde(default)]
    pub poll: bool,
    /// Remember the current end of the mesh-llm log under this name.
    #[serde(default)]
    pub log_mark: Option<String>,
    /// Limit this step's log assertions to lines written after the named mark.
    #[serde(default)]
    pub log_since: Option<String>,
    /// Substrings the mesh-llm log must contain.
    #[serde(default)]
    pub expect_log_contains: Vec<String>,
    /// Substrings the mesh-llm log must not contain.
    #[serde(default)]
    pub expect_log_not_contains: Vec<String>,
    /// Exact number of log lines containing each substring.
    #[serde(default)]
    pub expect_log_count: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReadFileSpec {
    pub path: String,
    /// Variable that receives the trimmed file content.
    pub var: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpCallSpec {
    #[serde(default = "default_post")]
    pub method: String,
    pub path: String,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub body: Option<toml::Value>,
    /// Open a concurrent background holder request against this same spec (for backpressure tests)?
    #[serde(default)]
    pub concurrent_background: bool,
    /// Start the request in the background under this name; `await_background` checks it later.
    #[serde(default)]
    pub background: Option<String>,
    /// Close with a reset (SO_LINGER 0) when dropped instead of an orderly FIN.
    #[serde(default)]
    pub reset_on_drop: bool,
}

fn default_post() -> String {
    "POST".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpCallSpec {
    pub method: String,
    #[serde(default)]
    pub params: Option<toml::Value>,
}

#[derive(Debug, Clone)]
pub struct SuiteRunOptions {
    pub mesh_llm_bin: String,
    pub archive: Option<PathBuf>,
    pub port: u16,
    pub console_port: u16,
    pub bind_port: u16,
    /// Run each suite this many times (races only show up under repetition).
    pub repeat: u32,
    /// Copy the mesh-llm log of a failed run into this directory.
    pub keep_logs: Option<PathBuf>,
}

struct HttpResponse {
    status: u16,
    headers: BTreeMap<String, String>,
    body: String,
}

pub async fn run_suites(paths: &[PathBuf], opts: &SuiteRunOptions) -> Result<()> {
    let mut total_suites = 0usize;
    let mut total_steps = 0usize;
    let mut failures = Vec::new();
    let mut skipped = 0usize;

    for path in paths {
        total_suites += 1;
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read suite file {}", path.display()))?;
        let spec: SuiteSpec =
            toml::from_str(&raw).with_context(|| format!("parse TOML suite {}", path.display()))?;
        println!("=== Suite: {} ({}) ===", spec.name, path.display());
        if !spec.description.is_empty() {
            println!("    {}", spec.description);
        }
        for run in 1..=opts.repeat.max(1) {
            let label = if opts.repeat > 1 {
                format!("{} [run {run}/{}]", spec.name, opts.repeat)
            } else {
                spec.name.clone()
            };
            match run_single_suite(&spec, opts).await {
                Ok(SuiteOutcome::Passed(steps_ran)) => {
                    total_steps += steps_ran;
                    println!("PASS suite '{label}' ({steps_ran} steps)\n");
                }
                Ok(SuiteOutcome::Skipped(reason)) => {
                    println!("SKIP suite '{label}' ({reason})\n");
                    skipped += 1;
                    break;
                }
                Err(err) => {
                    eprintln!("FAIL suite '{label}': {err:#}\n");
                    failures.push(format!("{label}: {err:#}"));
                }
            }
        }
    }

    if !failures.is_empty() {
        bail!(
            "{} of {} suites failed:\n  - {}",
            failures.len(),
            total_suites,
            failures.join("\n  - ")
        );
    }
    println!(
        "ALL {} DECLARATIVE SUITES PASSED ({total_steps} total steps, {skipped} skipped)",
        total_suites - skipped
    );
    Ok(())
}

async fn run_single_suite(spec: &SuiteSpec, opts: &SuiteRunOptions) -> Result<SuiteOutcome> {
    let missing: Vec<&str> = spec
        .mesh
        .requires_env
        .iter()
        .filter(|var| std::env::var_os(var).is_none_or(|v| v.is_empty()))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        return Ok(SuiteOutcome::Skipped(format!(
            "unset: {}",
            missing.join(", ")
        )));
    }

    let tmp_home = create_temp_home(&spec.name)?;
    let _cleanup = TempDirGuard(tmp_home.clone());
    let mesh_dir = tmp_home.join(".mesh-llm");
    std::fs::create_dir_all(&mesh_dir)?;

    let archive_path = match &opts.archive {
        Some(path) => path.clone(),
        None => build_local_archive(&tmp_home).await?,
    };
    let install = Command::new(&opts.mesh_llm_bin)
        .args([
            "plugins",
            "install",
            "--archive",
            &archive_path.display().to_string(),
            "--name",
            "mesh-dev",
        ])
        .env("HOME", &tmp_home)
        .env("XDG_CONFIG_HOME", tmp_home.join(".config"))
        .env("XDG_DATA_HOME", tmp_home.join(".local/share"))
        .env("XDG_CACHE_HOME", tmp_home.join(".cache"))
        .output()
        .await
        .context("install mesh-dev archive")?;
    if !install.status.success() {
        bail!(
            "mesh-llm plugins install failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&install.stdout),
            String::from_utf8_lossy(&install.stderr)
        );
    }

    let config_toml = render_mesh_config(&spec.mesh, None)?;
    let config_path = mesh_dir.join("config.toml");
    std::fs::write(&config_path, &config_toml)?;

    let validate = Command::new(&opts.mesh_llm_bin)
        .args([
            "config",
            "validate",
            "--config-path",
            &config_path.display().to_string(),
        ])
        .env("HOME", &tmp_home)
        .env("XDG_CONFIG_HOME", tmp_home.join(".config"))
        .env("XDG_DATA_HOME", tmp_home.join(".local/share"))
        .env("XDG_CACHE_HOME", tmp_home.join(".cache"))
        .output()
        .await
        .context("validate generated mesh-llm config")?;
    if !validate.status.success() {
        bail!(
            "mesh-llm config validate failed:\nstdout:\n{}\nstderr:\n{}\nConfig was:\n{config_toml}",
            String::from_utf8_lossy(&validate.stdout),
            String::from_utf8_lossy(&validate.stderr)
        );
    }

    let stderr_log = tmp_home.join("mesh-llm.err");
    let stderr_file = std::fs::File::create(&stderr_log)?;
    let mut node_args = vec![
        "--config".to_string(),
        config_path.display().to_string(),
        "--port".to_string(),
        opts.port.to_string(),
        "--console".to_string(),
        opts.console_port.to_string(),
        "--bind-port".to_string(),
        opts.bind_port.to_string(),
    ];
    if !spec.mesh.serve {
        node_args.push("--client".to_string());
    }
    let mut child = Command::new(&opts.mesh_llm_bin)
        .args(&node_args)
        .envs(&spec.mesh.plugin_env)
        .envs(&spec.mesh.node_env)
        .env("HOME", &tmp_home)
        .env("XDG_CONFIG_HOME", tmp_home.join(".config"))
        .env("XDG_DATA_HOME", tmp_home.join(".local/share"))
        .env("XDG_CACHE_HOME", tmp_home.join(".cache"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file))
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn {}", opts.mesh_llm_bin))?;

    let ready_limit = Duration::from_millis(spec.mesh.ready_timeout_ms.unwrap_or(30_000));
    let result = async {
        wait_console_ready(opts.console_port, ready_limit).await?;
        if !spec.mesh.skip_initial_model_wait {
            wait_for_model(opts.port, "dev/echo", ready_limit).await?;
        }

        let mut ctx = RunCtx::new(stderr_log.clone());
        ctx.vars
            .insert("NODE_HOME".into(), tmp_home.display().to_string());
        // Required runner variables are also usable as `${VAR}` in steps.
        for var in &spec.mesh.requires_env {
            if let Ok(value) = std::env::var(var) {
                ctx.vars.insert(var.clone(), value);
            }
        }
        for (idx, step) in spec.steps.iter().enumerate() {
            let step_num = idx + 1;
            let t0 = Instant::now();
            let step_res = execute_step_polled(step, opts, &mut ctx).await;

            match (step_res, &step.expect_failure_note) {
                (Ok(()), None) => {
                    println!(
                        "  [PASS {step_num:02}/{}] {} ({}ms)",
                        spec.steps.len(),
                        step.name,
                        t0.elapsed().as_millis()
                    );
                }
                (Ok(()), Some(note)) => {
                    println!(
                        "  [XPASS {step_num:02}/{}] {} ({}ms) — previously noted: {note}",
                        spec.steps.len(),
                        step.name,
                        t0.elapsed().as_millis()
                    );
                }
                (Err(err), Some(note)) => {
                    println!(
                        "  [XFAIL {step_num:02}/{}] {} ({}ms) — {note} ({err})",
                        spec.steps.len(),
                        step.name,
                        t0.elapsed().as_millis()
                    );
                }
                (Err(err), None) => {
                    ctx.abort_all();
                    let tail = tail_file(&stderr_log, 40);
                    return Err(err.context(format!(
                        "step {step_num} '{}' failed\n--- mesh-llm stderr tail ---\n{tail}",
                        step.name
                    )));
                }
            }
        }

        ctx.abort_all();
        Ok(spec.steps.len())
    }
    .await;

    stop_node(&mut child).await;
    if let (Err(_), Some(dir)) = (&result, &opts.keep_logs) {
        keep_log(dir, &spec.name, &stderr_log);
    }
    result.map(SuiteOutcome::Passed)
}

/// Stop mesh-llm with SIGTERM so it shuts down cleanly (and an instrumented build
/// writes its coverage profile); SIGKILL only if it hasn't exited after 15s.
async fn stop_node(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .await;
        if timeout(Duration::from_secs(15), child.wait()).await.is_ok() {
            return;
        }
    }
    let _ = child.kill().await;
    let _ = timeout(Duration::from_secs(3), child.wait()).await;
}

enum SuiteOutcome {
    Passed(usize),
    Skipped(String),
}

type BackgroundRequest = tokio::task::JoinHandle<Result<HttpResponse>>;

/// Per-run state shared by the steps of one suite.
struct RunCtx {
    vars: BTreeMap<String, String>,
    mcp_session_id: Option<String>,
    mcp_req_id: u64,
    background_handles: Vec<tokio::task::JoinHandle<()>>,
    named_background: BTreeMap<String, BackgroundRequest>,
    log_path: PathBuf,
    log_marks: BTreeMap<String, usize>,
}

impl RunCtx {
    fn new(log_path: PathBuf) -> Self {
        Self {
            vars: BTreeMap::new(),
            mcp_session_id: None,
            mcp_req_id: 1,
            background_handles: Vec::new(),
            named_background: BTreeMap::new(),
            log_path,
            log_marks: BTreeMap::new(),
        }
    }

    fn abort_all(&mut self) {
        for h in self.background_handles.drain(..) {
            h.abort();
        }
        for (_, h) in std::mem::take(&mut self.named_background) {
            h.abort();
        }
    }
}

async fn execute_step_polled(
    step: &StepSpec,
    opts: &SuiteRunOptions,
    ctx: &mut RunCtx,
) -> Result<()> {
    if !step.poll {
        return execute_step(step, opts, ctx).await;
    }
    let deadline = Instant::now() + Duration::from_millis(step.timeout_ms.unwrap_or(20_000));
    loop {
        match execute_step(step, opts, ctx).await {
            Ok(()) => return Ok(()),
            Err(err) if Instant::now() >= deadline => {
                return Err(err.context("condition not reached before timeout_ms"));
            }
            Err(_) => sleep(Duration::from_millis(200)).await,
        }
    }
}

async fn execute_step(step: &StepSpec, opts: &SuiteRunOptions, ctx: &mut RunCtx) -> Result<()> {
    execute_action(step, opts, ctx).await?;
    check_log(step, ctx)?;
    if let Some(mark) = &step.log_mark {
        let len = std::fs::read_to_string(&ctx.log_path)
            .map(|s| s.len())
            .unwrap_or(0);
        ctx.log_marks.insert(mark.clone(), len);
    }
    Ok(())
}

fn check_log(step: &StepSpec, ctx: &RunCtx) -> Result<()> {
    if step.expect_log_contains.is_empty()
        && step.expect_log_not_contains.is_empty()
        && step.expect_log_count.is_empty()
    {
        return Ok(());
    }
    let full = std::fs::read_to_string(&ctx.log_path).unwrap_or_default();
    let start = match &step.log_since {
        Some(mark) => *ctx
            .log_marks
            .get(mark)
            .ok_or_else(|| anyhow!("log mark '{mark}' not set"))?,
        None => 0,
    };
    let log = full.get(start.min(full.len())..).unwrap_or_default();
    for needle in &step.expect_log_contains {
        anyhow::ensure!(
            log.contains(needle.as_str()),
            "mesh-llm log missing {needle:?}"
        );
    }
    for needle in &step.expect_log_not_contains {
        let hit = log.lines().find(|l| l.contains(needle.as_str()));
        anyhow::ensure!(
            hit.is_none(),
            "mesh-llm log unexpectedly contained {needle:?}: {}",
            hit.unwrap_or_default()
        );
    }
    for (needle, expected) in &step.expect_log_count {
        let got = log.lines().filter(|l| l.contains(needle.as_str())).count();
        anyhow::ensure!(
            got == *expected,
            "mesh-llm log has {got} lines containing {needle:?}, expected {expected}"
        );
    }
    Ok(())
}

fn step_has_log_checks(step: &StepSpec) -> bool {
    !step.expect_log_contains.is_empty()
        || !step.expect_log_not_contains.is_empty()
        || !step.expect_log_count.is_empty()
        || step.log_mark.is_some()
}

async fn execute_action(step: &StepSpec, opts: &SuiteRunOptions, ctx: &mut RunCtx) -> Result<()> {
    let step_timeout = Duration::from_millis(step.timeout_ms.unwrap_or(20_000));

    if let Some(ms) = step.sleep_ms {
        sleep(Duration::from_millis(ms)).await;
        return Ok(());
    }

    if let Some(name) = &step.await_background {
        let handle = ctx
            .named_background
            .remove(name)
            .ok_or_else(|| anyhow!("no background request named '{name}'"))?;
        let resp = match timeout(step_timeout, handle).await {
            Ok(joined) => joined.context("background request task panicked")??,
            Err(_) => bail!("background request '{name}' did not finish within {step_timeout:?}"),
        };
        return check_response(step, &resp, &mut ctx.vars);
    }

    if let Some(spec) = &step.read_file {
        let path = substitute_str(&spec.path, &ctx.vars);
        let content = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
        ctx.vars
            .insert(spec.var.clone(), content.trim().to_string());
        return Ok(());
    }

    if let Some(name) = &step.expect_pending {
        let handle = ctx
            .named_background
            .get(name)
            .ok_or_else(|| anyhow!("no background request named '{name}'"))?;
        anyhow::ensure!(
            !handle.is_finished(),
            "background request '{name}' already finished; expected it to be waiting"
        );
        return Ok(());
    }

    if let Some(name) = &step.drop_background {
        let handle = ctx
            .named_background
            .remove(name)
            .ok_or_else(|| anyhow!("no background request named '{name}'"))?;
        anyhow::ensure!(
            !handle.is_finished(),
            "background request '{name}' already finished before the disconnect"
        );
        handle.abort();
        let _ = handle.await;
        return Ok(());
    }

    if let Some(model) = &step.wait_model {
        let model = substitute_str(model, &ctx.vars);
        wait_for_model(opts.port, &model, step_timeout).await?;
        return Ok(());
    }

    if let Some(var_name) = &step.wait_restart_from {
        let old_instance = ctx
            .vars
            .get(var_name)
            .cloned()
            .ok_or_else(|| anyhow!("wait_restart_from variable '{var_name}' not saved"))?;
        let deadline = Instant::now() + step_timeout;
        loop {
            let body = json!({
                "model": "dev/echo",
                "messages": [{"role": "user", "content": "wait-restart"}],
                "stream": false
            })
            .to_string();
            if let Ok(resp) = http_request(
                opts.port,
                "POST",
                "/v1/chat/completions",
                &BTreeMap::new(),
                Some(&body),
                Duration::from_secs(3),
            )
            .await
                && resp.status == 200
                && let Ok(v) = serde_json::from_str::<Value>(&resp.body)
                && let Some(content) = v["choices"][0]["message"]["content"].as_str()
                && let Ok(inner) = serde_json::from_str::<Value>(content)
                && let Some(new_id) = inner["plugin_instance_id"].as_str()
                && new_id != old_instance
            {
                ctx.vars
                    .insert("restarted_instance_id".into(), new_id.to_string());
                return Ok(());
            }
            if Instant::now() >= deadline {
                bail!(
                    "plugin did not restart with new instance_id (still {old_instance}) within {step_timeout:?}"
                );
            }
            sleep(Duration::from_millis(300)).await;
        }
    }

    if let Some(op) = &step.op {
        let input_json = match &step.input {
            Some(v) => substitute_json(toml_to_json(v)?, &ctx.vars).to_string(),
            None => "{}".to_string(),
        };
        let path = format!("/api/plugins/mesh-dev/tools/{op}");
        let resp = http_request(
            opts.console_port,
            "POST",
            &path,
            &BTreeMap::new(),
            Some(&input_json),
            step_timeout,
        )
        .await?;
        return check_response(step, &resp, &mut ctx.vars);
    }

    if let Some(call) = &step.openai {
        return run_http_call(step, call, opts.port, step_timeout, ctx).await;
    }

    if let Some(call) = &step.console_http {
        return run_http_call(step, call, opts.console_port, step_timeout, ctx).await;
    }

    if let Some(mcp) = &step.mcp {
        ensure_mcp_session(
            opts.console_port,
            &mut ctx.mcp_session_id,
            &mut ctx.mcp_req_id,
        )
        .await?;
        let params = match &mcp.params {
            Some(p) => substitute_json(toml_to_json(p)?, &ctx.vars),
            None => json!({}),
        };
        let id = ctx.mcp_req_id;
        ctx.mcp_req_id += 1;
        let rpc = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": mcp.method,
            "params": params,
        })
        .to_string();
        let mut headers = BTreeMap::new();
        headers.insert(
            "Accept".into(),
            "application/json, text/event-stream".into(),
        );
        if let Some(sid) = ctx.mcp_session_id.as_ref() {
            headers.insert("Mcp-Session-Id".into(), sid.clone());
        }
        let resp = http_request(
            opts.console_port,
            "POST",
            "/mcp",
            &headers,
            Some(&rpc),
            step_timeout,
        )
        .await?;
        let unwrapped = unwrap_mcp_response_body(&resp.body);
        let normalized = HttpResponse {
            status: resp.status,
            headers: resp.headers,
            body: unwrapped,
        };
        return check_response(step, &normalized, &mut ctx.vars);
    }

    if step_has_log_checks(step) {
        return Ok(());
    }
    bail!(
        "step '{}' has no action (op, openai, console_http, mcp, sleep_ms, wait_model, wait_restart_from, await_background, drop_background, expect_pending, read_file, or log checks)",
        step.name
    );
}

async fn run_http_call(
    step: &StepSpec,
    call: &HttpCallSpec,
    port: u16,
    step_timeout: Duration,
    ctx: &mut RunCtx,
) -> Result<()> {
    let vars = &ctx.vars;
    let path = substitute_str(&call.path, vars);
    let mut headers = BTreeMap::new();
    for (k, v) in &call.headers {
        headers.insert(k.clone(), substitute_str(v, vars));
    }
    let body = match &call.body {
        Some(v) => Some(substitute_json(toml_to_json(v)?, vars).to_string()),
        None => None,
    };

    if let Some(name) = &call.background {
        anyhow::ensure!(
            !ctx.named_background.contains_key(name),
            "background request '{name}' is already running"
        );
        let method = call.method.clone();
        let reset = call.reset_on_drop;
        let handle = tokio::spawn(async move {
            http_request_with(
                port,
                &method,
                &path,
                &headers,
                body.as_deref(),
                step_timeout,
                reset,
            )
            .await
        });
        ctx.named_background.insert(name.clone(), handle);
        // Give the request time to reach the proxy before the next step runs.
        sleep(Duration::from_millis(150)).await;
        return Ok(());
    }

    if call.concurrent_background {
        let method = call.method.clone();
        let handle = tokio::spawn(async move {
            let _ = http_request(
                port,
                &method,
                &path,
                &headers,
                body.as_deref(),
                step_timeout,
            )
            .await;
        });
        ctx.background_handles.push(handle);
        sleep(Duration::from_millis(150)).await;
        return Ok(());
    }

    let resp = http_request(
        port,
        &call.method,
        &path,
        &headers,
        body.as_deref(),
        step_timeout,
    )
    .await?;
    check_response(step, &resp, &mut ctx.vars)
}

async fn ensure_mcp_session(
    console_port: u16,
    session_id: &mut Option<String>,
    req_id: &mut u64,
) -> Result<()> {
    if session_id.is_some() {
        return Ok(());
    }
    let id = *req_id;
    *req_id += 1;
    let init_body = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "extensions": {
                    "io.modelcontextprotocol/tasks": {}
                }
            },
            "clientInfo": { "name": "mesh-dev-suite", "version": env!("CARGO_PKG_VERSION") }
        }
    })
    .to_string();
    let mut headers = BTreeMap::new();
    headers.insert(
        "Accept".into(),
        "application/json, text/event-stream".into(),
    );
    let resp = http_request(
        console_port,
        "POST",
        "/mcp",
        &headers,
        Some(&init_body),
        Duration::from_secs(10),
    )
    .await?;
    anyhow::ensure!(
        (200..300).contains(&resp.status),
        "MCP initialize failed ({}): {}",
        resp.status,
        resp.body
    );
    let sid = resp.headers.get("mcp-session-id").cloned();
    if let Some(ref s) = sid {
        let mut notify_headers = headers;
        notify_headers.insert("Mcp-Session-Id".into(), s.clone());
        let notif = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        })
        .to_string();
        let _ = http_request(
            console_port,
            "POST",
            "/mcp",
            &notify_headers,
            Some(&notif),
            Duration::from_secs(5),
        )
        .await;
    }
    *session_id = sid.or_else(|| Some(String::new()));
    Ok(())
}

fn unwrap_mcp_response_body(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.starts_with('{') {
        return trimmed.to_string();
    }
    // Streamable HTTP MCP returns `data: {...}` SSE events.
    let mut last_json = None;
    for line in trimmed.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            let candidate = data.trim();
            if candidate.starts_with('{') {
                last_json = Some(candidate.to_string());
            }
        }
    }
    last_json.unwrap_or_else(|| raw.to_string())
}

fn check_response(
    step: &StepSpec,
    resp: &HttpResponse,
    vars: &mut BTreeMap<String, String>,
) -> Result<()> {
    if step.dump {
        println!(
            "    [dump] HTTP {}: {}",
            resp.status,
            truncate(&resp.body, 20_000)
        );
    }
    if let Some(expected) = step.expect_status {
        anyhow::ensure!(
            resp.status == expected,
            "expected HTTP {expected}, got {}: {}",
            resp.status,
            truncate(&resp.body, 400)
        );
    } else if let Some([min, max]) = step.expect_status_range {
        anyhow::ensure!(
            resp.status >= min && resp.status <= max,
            "expected HTTP {min}..={max}, got {}: {}",
            resp.status,
            truncate(&resp.body, 400)
        );
    } else if step.expect_json.is_empty() && step.expect_contains.is_empty() {
        anyhow::ensure!(
            (200..300).contains(&resp.status),
            "expected 2xx HTTP status, got {}: {}",
            resp.status,
            truncate(&resp.body, 400)
        );
    }

    for needle in &step.expect_contains {
        let needle = substitute_str(needle, vars);
        anyhow::ensure!(
            resp.body.contains(&needle),
            "response body missing {needle:?}; body was: {}",
            truncate(&resp.body, 500)
        );
    }
    for forbidden in &step.expect_not_contains {
        let forbidden = substitute_str(forbidden, vars);
        anyhow::ensure!(
            !resp.body.contains(&forbidden),
            "response body unexpectedly contained {forbidden:?}; body was: {}",
            truncate(&resp.body, 500)
        );
    }
    for (name, expected) in &step.expect_headers {
        let lower = name.to_ascii_lowercase();
        let actual = resp.headers.get(&lower).ok_or_else(|| {
            anyhow!(
                "missing expected response header '{name}'; headers: {:?}",
                resp.headers
            )
        })?;
        let expected = substitute_str(expected, vars);
        anyhow::ensure!(
            actual.contains(&expected),
            "header '{name}' was {actual:?}, expected substring {expected:?}"
        );
    }
    for absent in &step.expect_absent_headers {
        let lower = absent.to_ascii_lowercase();
        anyhow::ensure!(
            !resp.headers.contains_key(&lower),
            "forbidden response header '{absent}' was unexpectedly present: {:?}",
            resp.headers.get(&lower)
        );
    }

    if !step.expect_json.is_empty() || !step.save_json.is_empty() {
        let parsed: Value = serde_json::from_str(&resp.body)
            .with_context(|| format!("response body is not JSON: {}", truncate(&resp.body, 400)))?;
        let root = normalize_tool_envelope(&parsed);

        for (path, expected_toml) in &step.expect_json {
            let expected = substitute_json(toml_to_json(expected_toml)?, vars);
            let actual = resolve_json_path(root, path);
            check_json_assertion(path, actual.as_ref(), &expected)?;
        }

        for (var_name, path) in &step.save_json {
            let val = resolve_json_path(root, path).ok_or_else(|| {
                anyhow!(
                    "save_json path '{path}' not found in {}",
                    truncate(&root.to_string(), 400)
                )
            })?;
            let as_str = match val {
                Value::String(s) => s,
                other => other.to_string(),
            };
            vars.insert(var_name.clone(), as_str);
        }
    }

    Ok(())
}

fn normalize_tool_envelope(val: &Value) -> &Value {
    val.get("structuredContent").unwrap_or(val)
}

/// Resolve dotted JSON paths, supporting numeric array indices (`choices.0.message.content`)
/// AND automatic JSON-string parsing (`choices.0.message.content|json.plugin_instance_id`).
fn resolve_json_path(root: &Value, path: &str) -> Option<Value> {
    let mut current = root.clone();
    for segment in path.split('.') {
        if let Some((key, transform)) = segment.split_once('|') {
            current = step_segment(&current, key)?;
            if transform == "json" {
                let s = current.as_str()?;
                current = serde_json::from_str(s).ok()?;
            }
        } else {
            current = step_segment(&current, segment)?;
        }
    }
    Some(current)
}

fn step_segment(val: &Value, seg: &str) -> Option<Value> {
    if let Ok(idx) = seg.parse::<usize>()
        && let Some(arr) = val.as_array()
    {
        return arr.get(idx).cloned();
    }
    val.get(seg).cloned()
}

fn check_json_assertion(path: &str, actual: Option<&Value>, expected: &Value) -> Result<()> {
    if let Some(obj) = expected.as_object() {
        if let Some(exists) = obj.get("exists").and_then(Value::as_bool) {
            anyhow::ensure!(
                actual.is_some() == exists,
                "json path '{path}': expected exists={exists}, got {actual:?}"
            );
            return Ok(());
        }
        if let Some(min) = obj.get("gte").and_then(Value::as_f64) {
            let got = actual.and_then(Value::as_f64).ok_or_else(|| {
                anyhow!("json path '{path}': expected number >= {min}, got {actual:?}")
            })?;
            anyhow::ensure!(got >= min, "json path '{path}': {got} < {min}");
            return Ok(());
        }
        if let Some(min) = obj.get("len_gte").and_then(Value::as_u64) {
            let len = actual
                .and_then(Value::as_array)
                .map(|a| a.len() as u64)
                .ok_or_else(|| {
                    anyhow!("json path '{path}': expected array len >= {min}, got {actual:?}")
                })?;
            anyhow::ensure!(len >= min, "json path '{path}': len {len} < {min}");
            return Ok(());
        }
        if let Some(sub) = obj.get("contains").and_then(Value::as_str) {
            let s = actual.and_then(Value::as_str).ok_or_else(|| {
                anyhow!("json path '{path}': expected string containing {sub:?}, got {actual:?}")
            })?;
            anyhow::ensure!(
                s.contains(sub),
                "json path '{path}': {s:?} does not contain {sub:?}"
            );
            return Ok(());
        }
        if let Some(ne) = obj.get("ne") {
            anyhow::ensure!(
                actual != Some(ne),
                "json path '{path}': expected value != {ne}, got {actual:?}"
            );
            return Ok(());
        }
    }
    let actual =
        actual.ok_or_else(|| anyhow!("json path '{path}' missing (expected {expected})"))?;
    anyhow::ensure!(
        actual == expected,
        "json path '{path}': expected {expected}, got {actual}"
    );
    Ok(())
}

async fn build_local_archive(tmp_home: &Path) -> Result<PathBuf> {
    let stage = tmp_home.join("stage");
    let pkg_dir = stage.join("mesh-dev");
    std::fs::create_dir_all(&pkg_dir)?;

    let exe = std::env::current_exe().context("resolve current mesh-dev binary")?;
    let bin_name = format!("mesh-dev{}", std::env::consts::EXE_SUFFIX);
    std::fs::copy(&exe, pkg_dir.join(&bin_name))
        .context("copy current mesh-dev binary into staged archive")?;

    let plugin_toml = format!(
        "name = \"mesh-dev\"\nversion = \"{}\"\ndescription = \"Lightweight zero-GPU developer & test plugin for Mesh-LLM\"\ncommand = \"mesh-dev\"\n",
        crate::mesh_adapter::VERSION
    );
    std::fs::write(pkg_dir.join("plugin.toml"), plugin_toml)?;

    let manifest = crate::mesh_adapter::build_manifest("http://127.0.0.1:0/v1");
    let pkg_manifest = mesh_llm_plugin::package_manifest_json(&manifest)
        .context("serialize plugin-manifest.json")?;
    std::fs::write(pkg_dir.join("plugin-manifest.json"), pkg_manifest)?;

    let archive = tmp_home.join("mesh-dev-local.tar.gz");
    let tar = Command::new("tar")
        .args([
            "-czf",
            &archive.display().to_string(),
            "-C",
            &stage.display().to_string(),
            "mesh-dev",
        ])
        .output()
        .await
        .context("run tar to create local plugin archive")?;
    anyhow::ensure!(
        tar.status.success(),
        "tar failed: {}",
        String::from_utf8_lossy(&tar.stderr)
    );
    Ok(archive)
}

fn render_mesh_config(cfg: &MeshSuiteConfig, command: Option<&str>) -> Result<String> {
    let mut root = toml::Table::new();
    root.insert("version".into(), toml::Value::Integer(1));

    let mut plugin = toml::Table::new();
    plugin.insert("name".into(), toml::Value::String("mesh-dev".into()));
    if let Some(cmd) = command {
        plugin.insert("command".into(), toml::Value::String(cmd.to_string()));
    }
    if !cfg.plugin_args.is_empty() {
        plugin.insert(
            "args".into(),
            toml::Value::Array(
                cfg.plugin_args
                    .iter()
                    .map(|s| toml::Value::String(s.clone()))
                    .collect(),
            ),
        );
    }
    if cfg.allow_inference_observation {
        plugin.insert(
            "allow_inference_observation".into(),
            toml::Value::Boolean(true),
        );
    }
    if cfg.allow_peer_blocks {
        plugin.insert("allow_peer_blocks".into(), toml::Value::Boolean(true));
    }
    if let Some(settings) = &cfg.settings {
        plugin.insert("settings".into(), toml::Value::Table(settings.clone()));
    }
    if let Some(grant) = &cfg.openai_exchange_grant {
        plugin.insert(
            "openai_exchange_grant".into(),
            toml::Value::Table(grant.clone()),
        );
    }

    if cfg.exclusive_gpu {
        plugin.insert("exclusive_gpu".into(), toml::Value::Boolean(true));
    }

    root.insert(
        "plugin".into(),
        toml::Value::Array(vec![toml::Value::Table(plugin)]),
    );
    if !cfg.models.is_empty() {
        let models = cfg
            .models
            .iter()
            .map(|m| expand_env_toml(toml::Value::Table(m.clone())))
            .collect::<Result<Vec<_>>>()?;
        root.insert("models".into(), toml::Value::Array(models));
    }
    if let Some(extra) = &cfg.config {
        for (key, value) in extra {
            anyhow::ensure!(
                !root.contains_key(key),
                "suite config table '{key}' collides with a generated key"
            );
            root.insert(key.clone(), expand_env_toml(value.clone())?);
        }
    }
    toml::to_string_pretty(&root).context("serialize generated mesh-llm config.toml")
}

/// Expand `${VAR}` in TOML strings from the runner's environment.
fn expand_env_toml(value: toml::Value) -> Result<toml::Value> {
    Ok(match value {
        toml::Value::String(s) => toml::Value::String(expand_env_str(&s)?),
        toml::Value::Array(items) => toml::Value::Array(
            items
                .into_iter()
                .map(expand_env_toml)
                .collect::<Result<Vec<_>>>()?,
        ),
        toml::Value::Table(table) => toml::Value::Table(
            table
                .into_iter()
                .map(|(k, v)| Ok((k, expand_env_toml(v)?)))
                .collect::<Result<toml::Table>>()?,
        ),
        other => other,
    })
}

fn expand_env_str(input: &str) -> Result<String> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let end = after
            .find('}')
            .ok_or_else(|| anyhow!("unterminated ${{ in {input:?}"))?;
        let name = &after[..end];
        let value = std::env::var(name).with_context(|| {
            format!("environment variable {name} (used in suite config) is unset")
        })?;
        out.push_str(&value);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn keep_log(dir: &Path, suite_name: &str, log: &Path) {
    let safe: String = suite_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let _ = std::fs::create_dir_all(dir);
    let dest = dir.join(format!("{safe}-{nanos}.log"));
    if std::fs::copy(log, &dest).is_ok() {
        eprintln!("    mesh-llm log kept at {}", dest.display());
    }
}

async fn wait_console_ready(console_port: u16, limit: Duration) -> Result<()> {
    let deadline = Instant::now() + limit;
    loop {
        if let Ok(resp) = http_request(
            console_port,
            "GET",
            "/api/plugins",
            &BTreeMap::new(),
            None,
            Duration::from_secs(2),
        )
        .await
            && resp.status == 200
            && resp.body.contains("mesh-dev")
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("mesh-llm console port {console_port} did not report mesh-dev within {limit:?}");
        }
        sleep(Duration::from_millis(250)).await;
    }
}

async fn wait_for_model(port: u16, model: &str, limit: Duration) -> Result<()> {
    let deadline = Instant::now() + limit;
    loop {
        if let Ok(resp) = http_request(
            port,
            "GET",
            "/v1/models",
            &BTreeMap::new(),
            None,
            Duration::from_secs(2),
        )
        .await
            && resp.status == 200
            && resp.body.contains(model)
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("model '{model}' not advertised on port {port} within {limit:?}");
        }
        sleep(Duration::from_millis(250)).await;
    }
}

async fn http_request(
    port: u16,
    method: &str,
    path: &str,
    extra_headers: &BTreeMap<String, String>,
    body: Option<&str>,
    limit: Duration,
) -> Result<HttpResponse> {
    http_request_with(port, method, path, extra_headers, body, limit, false).await
}

async fn http_request_with(
    port: u16,
    method: &str,
    path: &str,
    extra_headers: &BTreeMap<String, String>,
    body: Option<&str>,
    limit: Duration,
    reset_on_drop: bool,
) -> Result<HttpResponse> {
    let mut stream = timeout(limit, TcpStream::connect(("127.0.0.1", port)))
        .await
        .context("HTTP connect timeout")??;
    if reset_on_drop {
        stream.set_zero_linger()?;
    }
    let mut req =
        format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
    for (k, v) in extra_headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(body) = body {
        if !extra_headers
            .keys()
            .any(|k| k.eq_ignore_ascii_case("content-type"))
        {
            req.push_str("Content-Type: application/json\r\n");
        }
        req.push_str(&format!("Content-Length: {}\r\n\r\n{body}", body.len()));
    } else {
        req.push_str("\r\n");
    }
    timeout(limit, stream.write_all(req.as_bytes())).await??;

    let mut raw = Vec::new();
    let _ = timeout(limit, stream.read_to_end(&mut raw)).await?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body_raw) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    let mut headers = BTreeMap::new();
    for line in head.lines().skip(1) {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let body = if headers
        .get("transfer-encoding")
        .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"))
    {
        decode_chunked(body_raw)
    } else {
        body_raw.to_string()
    };
    Ok(HttpResponse {
        status,
        headers,
        body,
    })
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

fn toml_to_json(v: &toml::Value) -> Result<Value> {
    serde_json::to_value(v).context("convert TOML value to JSON")
}

fn substitute_str(input: &str, vars: &BTreeMap<String, String>) -> String {
    let mut out = input.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("${{{k}}}"), v);
    }
    out
}

fn substitute_json(val: Value, vars: &BTreeMap<String, String>) -> Value {
    match val {
        Value::String(s) => Value::String(substitute_str(&s, vars)),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|i| substitute_json(i, vars))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, substitute_json(v, vars)))
                .collect(),
        ),
        other => other,
    }
}

fn create_temp_home(suite_name: &str) -> Result<PathBuf> {
    let safe: String = suite_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "mesh-dev-suite-{safe}-{}-{:x}",
        std::process::id(),
        nanos & 0xffff_ffff
    ));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

struct TempDirGuard(PathBuf);
impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tail_file(path: &Path, lines: usize) -> String {
    let Ok(content) = std::fs::read_to_string(path) else {
        return String::new();
    };
    let all: Vec<&str> = content.lines().collect();
    let start = all.len().saturating_sub(lines);
    all[start..].join("\n")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_nested_json_string_paths_and_operators() {
        let payload = json!({
            "choices": [{
                "message": {
                    "content": r#"{"plugin_instance_id":"inst-42","count":5}"#
                }
            }]
        });
        let id = resolve_json_path(
            &payload,
            "choices.0.message.content|json.plugin_instance_id",
        );
        assert_eq!(id, Some(json!("inst-42")));
        let count = resolve_json_path(&payload, "choices.0.message.content|json.count");
        check_json_assertion("count", count.as_ref(), &json!({"gte": 3})).unwrap();
    }

    #[test]
    fn parses_all_repo_suites() {
        let suites_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("suites");
        let mut count = 0usize;
        for entry in std::fs::read_dir(&suites_dir).expect("read suites dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                let raw = std::fs::read_to_string(&path).expect("read suite");
                let parsed: SuiteSpec =
                    toml::from_str(&raw).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                assert!(
                    !parsed.steps.is_empty(),
                    "suite {} has no steps",
                    path.display()
                );
                count += 1;
            }
        }
        assert!(count >= 5, "expected at least 5 suites, found {count}");
    }
}
