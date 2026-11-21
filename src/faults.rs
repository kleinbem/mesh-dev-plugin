//! Controllable supervision, HTTP endpoint, and per-model fault injection state.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

static INSTANCE_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum HealthFaultMode {
    #[default]
    Ok,
    Error {
        #[serde(default = "default_health_error_detail")]
        detail: String,
    },
    Hang {
        /// Optional hang duration in milliseconds (`null` hangs until cancelled/shutdown).
        #[serde(default)]
        hang_ms: Option<u64>,
    },
}

fn default_health_error_detail() -> String {
    "injected health check failure".to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum HttpFaultMode {
    #[default]
    Ok,
    Models503 {
        #[serde(default = "default_models_503_detail")]
        detail: String,
    },
    ModelsEmpty,
    ModelsHang {
        #[serde(default)]
        hang_ms: Option<u64>,
    },
}

fn default_models_503_detail() -> String {
    "injected /v1/models 503 failure".to_string()
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OverrideBehavior {
    #[default]
    Default,
    Echo,
    Http503,
    Http429,
    MidStreamDrop,
    MalformedSse,
    SlowTtft,
    SlowSse,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ModelOverride {
    pub behavior: OverrideBehavior,
    /// Optional remaining hit count; once it reaches 0 the override is removed automatically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remaining: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delay_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunks: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExchangePolicyDecision {
    #[default]
    Abstain,
    Allow,
    Deny,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ExchangePolicy {
    #[serde(default)]
    pub decision: ExchangePolicyDecision,
    /// When set with `decision == Deny`, deny only requests targeting this model ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deny_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default)]
    pub response_headers: Vec<(String, String)>,
    #[serde(default)]
    pub annotations: Vec<(String, String)>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransformMode {
    #[default]
    Unchanged,
    TagSystem,
    Reject,
    CorruptModel,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TransformPolicy {
    #[serde(default)]
    pub mode: TransformMode,
    #[serde(default = "default_transform_tag")]
    pub tag: String,
    #[serde(default = "default_transform_reject_reason")]
    pub reason: String,
}

fn default_transform_tag() -> String {
    "[mesh-dev-transformed]".to_string()
}

fn default_transform_reject_reason() -> String {
    "rejected by mesh-dev request transform".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FaultSnapshot {
    pub health: HealthFaultMode,
    pub http: HttpFaultMode,
    pub loading: bool,
    pub loading_remaining_ms: u64,
    pub model_overrides: BTreeMap<String, ModelOverride>,
    pub exchange_policy: ExchangePolicy,
    pub transform_policy: TransformPolicy,
}

#[derive(Debug, Default)]
struct InnerFaultState {
    health: HealthFaultMode,
    http: HttpFaultMode,
    loading_until: Option<Instant>,
    model_overrides: BTreeMap<String, ModelOverride>,
    exchange_policy: ExchangePolicy,
    transform_policy: TransformPolicy,
    received_settings: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct FaultController {
    instance_id: String,
    pid: u32,
    started_at: Instant,
    inner: Arc<RwLock<InnerFaultState>>,
}

impl Default for FaultController {
    fn default() -> Self {
        Self::new(None, Duration::ZERO)
    }
}

impl FaultController {
    pub fn new(unhealthy_on_start: Option<String>, startup_delay: Duration) -> Self {
        let pid = std::process::id();
        let seq = INSTANCE_SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let instance_id = format!("mesh-dev-{pid}-{:x}-{seq}", nanos & 0xffff_ffff);

        let initial_health = match unhealthy_on_start {
            Some(detail) => HealthFaultMode::Error { detail },
            None => HealthFaultMode::Ok,
        };
        let loading_until = (!startup_delay.is_zero()).then(|| Instant::now() + startup_delay);

        Self {
            instance_id,
            pid,
            started_at: Instant::now(),
            inner: Arc::new(RwLock::new(InnerFaultState {
                health: initial_health,
                http: HttpFaultMode::Ok,
                loading_until,
                model_overrides: BTreeMap::new(),
                exchange_policy: ExchangePolicy::default(),
                transform_policy: TransformPolicy::default(),
                received_settings: None,
            })),
        }
    }

    pub fn exchange_policy(&self) -> ExchangePolicy {
        self.inner
            .read()
            .map(|g| g.exchange_policy.clone())
            .unwrap_or_default()
    }

    pub fn set_exchange_policy(&self, policy: ExchangePolicy) {
        if let Ok(mut g) = self.inner.write() {
            g.exchange_policy = policy;
        }
    }

    pub fn transform_policy(&self) -> TransformPolicy {
        self.inner
            .read()
            .map(|g| g.transform_policy.clone())
            .unwrap_or_default()
    }

    pub fn set_transform_policy(&self, policy: TransformPolicy) {
        if let Ok(mut g) = self.inner.write() {
            g.transform_policy = policy;
        }
    }

    pub fn received_settings(&self) -> Option<serde_json::Value> {
        self.inner
            .read()
            .ok()
            .and_then(|g| g.received_settings.clone())
    }

    pub fn set_received_settings(&self, settings: Option<serde_json::Value>) {
        if let Ok(mut g) = self.inner.write() {
            g.received_settings = settings;
        }
    }

    pub fn loading_remaining_ms(&self) -> u64 {
        let Ok(guard) = self.inner.read() else {
            return 0;
        };
        let Some(until) = guard.loading_until else {
            return 0;
        };
        let now = Instant::now();
        if now >= until {
            0
        } else {
            (until - now).as_millis() as u64
        }
    }

    pub fn is_loading(&self) -> bool {
        self.loading_remaining_ms() > 0
    }

    pub fn set_loading_delay(&self, delay: Duration) {
        if let Ok(mut g) = self.inner.write() {
            g.loading_until = (!delay.is_zero()).then(|| Instant::now() + delay);
        }
    }

    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    pub fn uptime_ms(&self) -> u64 {
        self.started_at.elapsed().as_millis() as u64
    }

    pub fn health_mode(&self) -> HealthFaultMode {
        self.inner
            .read()
            .map(|g| g.health.clone())
            .unwrap_or_default()
    }

    pub fn set_health_mode(&self, mode: HealthFaultMode) {
        if let Ok(mut g) = self.inner.write() {
            g.health = mode;
        }
    }

    pub fn http_fault(&self) -> HttpFaultMode {
        self.inner
            .read()
            .map(|g| g.http.clone())
            .unwrap_or_default()
    }

    pub fn set_http_fault(&self, mode: HttpFaultMode) {
        if let Ok(mut g) = self.inner.write() {
            g.http = mode;
        }
    }

    pub fn set_model_override(&self, model: String, entry: Option<ModelOverride>) {
        if let Ok(mut g) = self.inner.write() {
            match entry {
                None
                | Some(ModelOverride {
                    behavior: OverrideBehavior::Default,
                    ..
                }) => {
                    g.model_overrides.remove(&model);
                }
                Some(override_cfg) => {
                    g.model_overrides.insert(model, override_cfg);
                }
            }
        }
    }

    /// Consume one hit of any active override for `model`.
    /// Decrements `remaining` if bounded, removing the entry when exhausted.
    pub fn take_model_override(&self, model: &str) -> Option<ModelOverride> {
        let mut g = self.inner.write().ok()?;
        let entry = g.model_overrides.get_mut(model)?;
        let snapshot = entry.clone();
        if let Some(rem) = entry.remaining.as_mut() {
            *rem = rem.saturating_sub(1);
            if *rem == 0 {
                g.model_overrides.remove(model);
            }
        }
        Some(snapshot)
    }

    pub fn reset_all(&self) {
        if let Ok(mut g) = self.inner.write() {
            g.health = HealthFaultMode::Ok;
            g.http = HttpFaultMode::Ok;
            g.loading_until = None;
            g.model_overrides.clear();
            g.exchange_policy = ExchangePolicy::default();
            g.transform_policy = TransformPolicy::default();
        }
    }

    pub fn snapshot(&self) -> FaultSnapshot {
        let loading_remaining_ms = self.loading_remaining_ms();
        self.inner
            .read()
            .map(|g| FaultSnapshot {
                health: g.health.clone(),
                http: g.http.clone(),
                loading: loading_remaining_ms > 0,
                loading_remaining_ms,
                model_overrides: g.model_overrides.clone(),
                exchange_policy: g.exchange_policy.clone(),
                transform_policy: g.transform_policy.clone(),
            })
            .unwrap_or(FaultSnapshot {
                health: HealthFaultMode::Ok,
                http: HttpFaultMode::Ok,
                loading: false,
                loading_remaining_ms: 0,
                model_overrides: BTreeMap::new(),
                exchange_policy: ExchangePolicy::default(),
                transform_policy: TransformPolicy::default(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counted_model_override_auto_reverts_after_n_hits() {
        let faults = FaultController::new(None, Duration::ZERO);
        faults.set_model_override(
            "dev/echo".into(),
            Some(ModelOverride {
                behavior: OverrideBehavior::Http503,
                remaining: Some(2),
                delay_ms: None,
                chunks: None,
            }),
        );

        let first = faults.take_model_override("dev/echo").expect("1st hit");
        assert_eq!(first.behavior, OverrideBehavior::Http503);
        let second = faults.take_model_override("dev/echo").expect("2nd hit");
        assert_eq!(second.behavior, OverrideBehavior::Http503);
        assert!(
            faults.take_model_override("dev/echo").is_none(),
            "override must revert after 2 hits"
        );
    }

    #[test]
    fn reset_clears_all_injected_faults() {
        let faults = FaultController::new(Some("boot unhealthy".into()), Duration::from_secs(30));
        assert!(matches!(
            faults.health_mode(),
            HealthFaultMode::Error { .. }
        ));
        assert!(faults.is_loading());
        faults.set_http_fault(HttpFaultMode::ModelsEmpty);
        faults.set_model_override(
            "dev/echo".into(),
            Some(ModelOverride {
                behavior: OverrideBehavior::Http429,
                remaining: None,
                delay_ms: None,
                chunks: None,
            }),
        );

        faults.reset_all();
        assert_eq!(faults.health_mode(), HealthFaultMode::Ok);
        assert_eq!(faults.http_fault(), HttpFaultMode::Ok);
        assert!(!faults.is_loading());
        assert!(faults.snapshot().model_overrides.is_empty());
    }
}
