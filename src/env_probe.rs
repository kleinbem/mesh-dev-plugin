//! Strictly allowlisted non-secret environment probe for verifying shared model-cache
//! and runtime path propagation (`MESH_LLM_*` / cache dirs) into plugins.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Exact allowlist of non-secret environment variables that `env_probe` / `dev/env-probe`
/// may report. Secret-bearing variables (tokens, keys, credentials, auth headers) are
/// strictly excluded and never read or returned.
pub const ALLOWLISTED_ENV_VARS: &[&str] = &[
    "MESH_LLM_PLUGIN_NAME",
    "MESH_LLM_PLUGIN_TRANSPORT",
    "MESH_LLM_MODELS_DIR",
    "MESH_LLM_CACHE_DIR",
    "MESH_LLM_DATA_DIR",
    "MESH_LLM_CONFIG_DIR",
    "MESH_LLM_HOME",
    "MESH_LLM_HF_HOME",
    "MESH_LLM_LOG_DIR",
    "MESH_LLM_PLUGIN_DIR",
    "MESH_LLM_RUNTIME_DIR",
    "MESH_LLM_VERSION",
    "HF_HOME",
    "HF_HUB_CACHE",
    "XDG_CACHE_HOME",
];

const FORBIDDEN_SUBSTRINGS: &[&str] = &[
    "TOKEN", "SECRET", "KEY", "PASS", "AUTH", "CRED", "COOKIE", "JWT", "SIGN", "PRIVATE", "BEARER",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvProbeSnapshot {
    pub allowlisted_vars: BTreeMap<String, String>,
    pub plugin_endpoint_set: bool,
    pub redacted_mesh_llm_var_count: usize,
}

pub fn probe_env() -> EnvProbeSnapshot {
    probe_env_from(std::env::vars())
}

pub fn probe_env_from<I>(vars: I) -> EnvProbeSnapshot
where
    I: IntoIterator<Item = (String, String)>,
{
    let mut allowlisted_vars = BTreeMap::new();
    let mut plugin_endpoint_set = false;
    let mut redacted_mesh_llm_var_count = 0usize;

    for (key, value) in vars {
        if key == "MESH_LLM_PLUGIN_ENDPOINT" {
            plugin_endpoint_set = !value.is_empty();
            continue;
        }
        if is_safe_allowlisted(&key) {
            allowlisted_vars.insert(key, value);
        } else if key.starts_with("MESH_LLM_") {
            redacted_mesh_llm_var_count += 1;
        }
    }

    EnvProbeSnapshot {
        allowlisted_vars,
        plugin_endpoint_set,
        redacted_mesh_llm_var_count,
    }
}

fn is_safe_allowlisted(key: &str) -> bool {
    if !ALLOWLISTED_ENV_VARS.contains(&key) {
        return false;
    }
    let upper = key.to_ascii_uppercase();
    !FORBIDDEN_SUBSTRINGS
        .iter()
        .any(|forbidden| upper.contains(forbidden))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_model_cache_vars_and_never_leaks_secrets() {
        let input = vec![
            ("MESH_LLM_PLUGIN_NAME".into(), "mesh-dev".into()),
            (
                "MESH_LLM_MODELS_DIR".into(),
                "/var/cache/mesh/models".into(),
            ),
            ("MESH_LLM_JOIN_TOKEN".into(), "super-secret-token".into()),
            ("MESH_LLM_API_KEY".into(), "sk-secret".into()),
            ("MESH_LLM_PLUGIN_ENDPOINT".into(), "/tmp/host.sock".into()),
            ("HF_TOKEN".into(), "hf_secret".into()),
            ("HF_HOME".into(), "/var/cache/hf".into()),
        ];

        let snap = probe_env_from(input);
        assert!(snap.plugin_endpoint_set);
        assert_eq!(
            snap.allowlisted_vars
                .get("MESH_LLM_PLUGIN_NAME")
                .map(String::as_str),
            Some("mesh-dev")
        );
        assert_eq!(
            snap.allowlisted_vars
                .get("MESH_LLM_MODELS_DIR")
                .map(String::as_str),
            Some("/var/cache/mesh/models")
        );
        assert_eq!(
            snap.allowlisted_vars.get("HF_HOME").map(String::as_str),
            Some("/var/cache/hf")
        );
        assert!(!snap.allowlisted_vars.contains_key("MESH_LLM_JOIN_TOKEN"));
        assert!(!snap.allowlisted_vars.contains_key("MESH_LLM_API_KEY"));
        assert!(!snap.allowlisted_vars.contains_key("HF_TOKEN"));
        assert_eq!(snap.redacted_mesh_llm_var_count, 2);
    }
}
