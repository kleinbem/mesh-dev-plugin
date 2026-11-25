//! Live memory and optional GPU VRAM probing for `dev/mem-probe`.
//!
//! Reads `/proc/meminfo` and `/proc/self/status` (`VmRSS`) on Linux at request execution time
//! without caching so `exclusive_gpu` VRAM handoff (tracking: Mesh-LLM/mesh-llm#3174) and
//! sub-5 MB plugin RSS can be verified deterministically in <1s.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::cli::GpuProbeMode;

const NVIDIA_SMI_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub timestamp_unix_ms: u64,
    pub platform: String,
    pub mem_total_kb: Option<u64>,
    pub mem_free_kb: Option<u64>,
    pub mem_available_kb: Option<u64>,
    pub swap_total_kb: Option<u64>,
    pub swap_free_kb: Option<u64>,
    pub self_rss_kb: Option<u64>,
    pub unified_memory: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<GpuMemoryInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuMemoryInfo {
    pub source: String,
    pub free_mb: u64,
    pub total_mb: u64,
    pub devices: Vec<GpuDeviceMemory>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuDeviceMemory {
    pub index: usize,
    pub free_mb: u64,
    pub total_mb: u64,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ProcMemInfo {
    pub mem_total_kb: Option<u64>,
    pub mem_free_kb: Option<u64>,
    pub mem_available_kb: Option<u64>,
    pub swap_total_kb: Option<u64>,
    pub swap_free_kb: Option<u64>,
}

/// Capture a fresh memory snapshot at the current instant.
pub fn probe_memory(mode: GpuProbeMode) -> MemorySnapshot {
    let timestamp_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let meminfo = read_proc_meminfo();
    let self_rss_kb = read_self_rss_kb();
    let unified_memory = detect_tegra_uma();

    let gpu = match mode {
        GpuProbeMode::Off => None,
        GpuProbeMode::NvidiaSmi => probe_nvidia_smi(),
        GpuProbeMode::Drm => probe_drm_sysfs(),
        GpuProbeMode::Auto => probe_gpu_auto(&meminfo, unified_memory),
    };

    MemorySnapshot {
        timestamp_unix_ms,
        platform: std::env::consts::OS.to_string(),
        mem_total_kb: meminfo.mem_total_kb,
        mem_free_kb: meminfo.mem_free_kb,
        mem_available_kb: meminfo.mem_available_kb,
        swap_total_kb: meminfo.swap_total_kb,
        swap_free_kb: meminfo.swap_free_kb,
        self_rss_kb,
        unified_memory,
        gpu,
    }
}

/// Read the current process's resident set size (`VmRSS`) in KiB on Linux.
pub fn read_self_rss_kb() -> Option<u64> {
    let content = fs::read_to_string("/proc/self/status").ok()?;
    parse_proc_status_rss_kb(&content)
}

fn read_proc_meminfo() -> ProcMemInfo {
    fs::read_to_string("/proc/meminfo")
        .map(|raw| parse_proc_meminfo(&raw))
        .unwrap_or_default()
}

pub fn parse_proc_meminfo(raw: &str) -> ProcMemInfo {
    let mut info = ProcMemInfo::default();
    for line in raw.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let value = rest
            .split_whitespace()
            .next()
            .and_then(|num| num.parse::<u64>().ok());
        match key.trim() {
            "MemTotal" => info.mem_total_kb = value,
            "MemFree" => info.mem_free_kb = value,
            "MemAvailable" => info.mem_available_kb = value,
            "SwapTotal" => info.swap_total_kb = value,
            "SwapFree" => info.swap_free_kb = value,
            _ => {}
        }
    }
    info
}

pub fn parse_proc_status_rss_kb(raw: &str) -> Option<u64> {
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .split_whitespace()
                .next()
                .and_then(|num| num.parse::<u64>().ok());
        }
    }
    None
}

fn detect_tegra_uma() -> bool {
    if Path::new("/etc/nv_tegra_release").exists() {
        return true;
    }
    if let Ok(family) = fs::read_to_string("/sys/devices/soc0/family")
        && family.to_ascii_lowercase().contains("tegra")
    {
        return true;
    }
    false
}

fn probe_gpu_auto(meminfo: &ProcMemInfo, unified_memory: bool) -> Option<GpuMemoryInfo> {
    if let Some(drm) = probe_drm_sysfs() {
        return Some(drm);
    }
    if unified_memory
        && let (Some(avail_kb), Some(total_kb)) = (meminfo.mem_available_kb, meminfo.mem_total_kb)
    {
        let free_mb = avail_kb / 1024;
        let total_mb = total_kb / 1024;
        return Some(GpuMemoryInfo {
            source: "tegra-uma".to_string(),
            free_mb,
            total_mb,
            devices: vec![GpuDeviceMemory {
                index: 0,
                free_mb,
                total_mb,
            }],
        });
    }
    probe_nvidia_smi()
}

fn probe_drm_sysfs() -> Option<GpuMemoryInfo> {
    let entries = fs::read_dir("/sys/class/drm").ok()?;
    let mut devices = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }
        let dev_dir = entry.path().join("device");
        let total_bytes = fs::read_to_string(dev_dir.join("mem_info_vram_total"))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        let used_bytes = fs::read_to_string(dev_dir.join("mem_info_vram_used"))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        if let (Some(total), Some(used)) = (total_bytes, used_bytes)
            && total > 0
        {
            let free = total.saturating_sub(used);
            devices.push(GpuDeviceMemory {
                index: devices.len(),
                free_mb: free / (1024 * 1024),
                total_mb: total / (1024 * 1024),
            });
        }
    }
    if devices.is_empty() {
        return None;
    }
    let free_mb = devices.iter().map(|d| d.free_mb).sum();
    let total_mb = devices.iter().map(|d| d.total_mb).sum();
    Some(GpuMemoryInfo {
        source: "drm-sysfs".to_string(),
        free_mb,
        total_mb,
        devices,
    })
}

fn probe_nvidia_smi() -> Option<GpuMemoryInfo> {
    let mut child = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    let deadline = Instant::now() + NVIDIA_SMI_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let output = child.wait_with_output().ok()?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                return parse_nvidia_smi_csv(&stdout);
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

pub fn parse_nvidia_smi_csv(raw: &str) -> Option<GpuMemoryInfo> {
    let mut devices = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.split(',').map(str::trim);
        let free_mb = parts.next()?.parse::<u64>().ok()?;
        let total_mb = parts.next()?.parse::<u64>().ok()?;
        devices.push(GpuDeviceMemory {
            index: devices.len(),
            free_mb,
            total_mb,
        });
    }
    if devices.is_empty() {
        return None;
    }
    let free_mb = devices.iter().map(|d| d.free_mb).sum();
    let total_mb = devices.iter().map(|d| d.total_mb).sum();
    Some(GpuMemoryInfo {
        source: "nvidia-smi".to_string(),
        free_mb,
        total_mb,
        devices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_proc_meminfo() {
        let sample = "\
MemTotal:        7801236 kB
MemFree:         1234560 kB
MemAvailable:    4567890 kB
Buffers:          102400 kB
SwapTotal:       2097148 kB
SwapFree:        2097000 kB
";
        let info = parse_proc_meminfo(sample);
        assert_eq!(info.mem_total_kb, Some(7_801_236));
        assert_eq!(info.mem_free_kb, Some(1_234_560));
        assert_eq!(info.mem_available_kb, Some(4_567_890));
        assert_eq!(info.swap_total_kb, Some(2_097_148));
        assert_eq!(info.swap_free_kb, Some(2_097_000));
    }

    #[test]
    fn parses_proc_status_rss() {
        let sample = "\
Name:\tmesh-dev
VmSize:\t   18420 kB
VmRSS:\t    3412 kB
Threads:\t4
";
        assert_eq!(parse_proc_status_rss_kb(sample), Some(3_412));
    }

    #[test]
    fn parses_nvidia_smi_multi_gpu_csv() {
        let sample = "10240, 24576\n8192, 16384\n";
        let gpu = parse_nvidia_smi_csv(sample).expect("parsed gpu info");
        assert_eq!(gpu.source, "nvidia-smi");
        assert_eq!(gpu.free_mb, 18_432);
        assert_eq!(gpu.total_mb, 40_960);
        assert_eq!(gpu.devices.len(), 2);
        assert_eq!(gpu.devices[1].index, 1);
    }

    #[test]
    fn parses_nvidia_smi_rejects_na_values() {
        assert!(parse_nvidia_smi_csv("[N/A], [N/A]\n").is_none());
        assert!(parse_nvidia_smi_csv("").is_none());
    }

    #[test]
    fn live_probe_off_succeeds_on_linux() {
        let snap = probe_memory(GpuProbeMode::Off);
        assert!(snap.gpu.is_none());
        #[cfg(target_os = "linux")]
        {
            assert!(snap.mem_total_kb.unwrap_or(0) > 0);
            assert!(snap.mem_available_kb.unwrap_or(0) > 0);
            assert!(snap.self_rss_kb.unwrap_or(0) > 0);
        }
    }
}
