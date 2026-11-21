//! Bounded ring-buffer inspector covering three observation layers:
//! 1. Local HTTP requests handled by `mesh-dev`'s OpenAI-compatible server.
//! 2. Host local inference observation feed (`allow_inference_observation = true`).
//! 3. Host OpenAI exchange lifecycle hook (`openai_exchange_grant`).

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use mesh_llm_plugin::openai_exchange::OpenAiExchangeEvent;
use mesh_llm_plugin::proto;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InspectSource {
    #[default]
    All,
    Http,
    Observation,
    Exchange,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HttpOutcome {
    Completed,
    ClientDisconnected,
    InjectedError,
    MidStreamDrop,
    MalformedSse,
    Backpressure503,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRequestRecord {
    pub seq: u64,
    pub timestamp_unix_ms: u64,
    pub endpoint: String,
    pub model: String,
    pub stream: bool,
    pub headers: BTreeMap<String, String>,
    pub http_status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    pub duration_ms: u64,
    pub tokens_emitted: u64,
    pub outcome: HttpOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub override_applied: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationExecutionRecord {
    pub execution_id: u64,
    pub host_generation: u64,
    pub model: String,
    pub operation: String,
    pub streaming: bool,
    pub started_at_unix_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_output_at_unix_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_finished_at_unix_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at_unix_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLifecycleEvent {
    pub sequence: u64,
    pub at_unix_ns: u64,
    pub event: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeRecord {
    pub exchange_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation_id: Option<String>,
    pub endpoint: String,
    pub observation_point: String,
    pub method: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub phases_seen: Vec<String>,
    pub first_seen_unix_ms: u64,
    pub last_seen_unix_ms: u64,
    pub latency_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_outcome: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admission_denied: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence_complete: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observer_evidence_complete: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_wire_digest: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_request_wire_digest: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_wire_commitment: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectRecentResponse {
    pub source: InspectSource,
    pub capacity: usize,
    pub in_flight: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http: Option<Vec<HttpRequestRecord>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation: Option<ObservationInspectSlice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exchange: Option<Vec<ExchangeRecord>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationInspectSlice {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub active: Vec<ObservationExecutionRecord>,
    pub recent_finished: Vec<ObservationExecutionRecord>,
    pub recent_model_events: Vec<ModelLifecycleEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpStats {
    pub total_requests: u64,
    pub in_flight: u64,
    pub completed: u64,
    pub client_disconnected: u64,
    /// Alias of `client_disconnected` for A/B timeout/cancellation assertions.
    pub cancelled_count: u64,
    pub injected_errors: u64,
    pub mid_stream_drops: u64,
    pub malformed_sse: u64,
    pub backpressure_503: u64,
    pub buffered_records: usize,
    pub per_model: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_ttft_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_duration_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationStats {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub host_generation: u64,
    pub active_executions: usize,
    pub finished_executions: u64,
    pub ready_models: Vec<String>,
    pub events_received: u64,
    pub lost_events: u64,
    pub feed_overflows: u64,
    pub watermark_unix_ns: u64,
    pub buffered_records: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeStats {
    pub total_events: u64,
    pub distinct_exchanges: u64,
    pub buffered_records: usize,
    pub denied_count: u64,
    pub outcomes: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectorStats {
    pub capacity: usize,
    pub http: HttpStats,
    pub observation: ObservationStats,
    pub exchange: ExchangeStats,
}

struct HttpBucket {
    next_seq: u64,
    in_flight: u64,
    total_requests: u64,
    completed: u64,
    client_disconnected: u64,
    injected_errors: u64,
    mid_stream_drops: u64,
    malformed_sse: u64,
    backpressure_503: u64,
    ttft_sum_ms: u64,
    ttft_count: u64,
    duration_sum_ms: u64,
    per_model: BTreeMap<String, u64>,
    recent: VecDeque<HttpRequestRecord>,
}

impl Default for HttpBucket {
    fn default() -> Self {
        Self {
            next_seq: 1,
            in_flight: 0,
            total_requests: 0,
            completed: 0,
            client_disconnected: 0,
            injected_errors: 0,
            mid_stream_drops: 0,
            malformed_sse: 0,
            backpressure_503: 0,
            ttft_sum_ms: 0,
            ttft_count: 0,
            duration_sum_ms: 0,
            per_model: BTreeMap::new(),
            recent: VecDeque::new(),
        }
    }
}

struct ObservationBucket {
    status: String,
    detail: Option<String>,
    host_generation: u64,
    ready_models: Vec<String>,
    active: BTreeMap<u64, ObservationExecutionRecord>,
    finished: VecDeque<ObservationExecutionRecord>,
    model_events: VecDeque<ModelLifecycleEvent>,
    finished_total: u64,
    events_received: u64,
    lost_events: u64,
    feed_overflows: u64,
    watermark_unix_ns: u64,
}

impl Default for ObservationBucket {
    fn default() -> Self {
        Self {
            status: "uninitialized".to_string(),
            detail: None,
            host_generation: 0,
            ready_models: Vec::new(),
            active: BTreeMap::new(),
            finished: VecDeque::new(),
            model_events: VecDeque::new(),
            finished_total: 0,
            events_received: 0,
            lost_events: 0,
            feed_overflows: 0,
            watermark_unix_ns: 0,
        }
    }
}

#[derive(Default)]
struct ExchangeBucket {
    total_events: u64,
    distinct_exchanges: u64,
    denied_count: u64,
    outcomes: BTreeMap<String, u64>,
    recent: VecDeque<ExchangeRecord>,
}

struct InspectorInner {
    capacity: usize,
    http: HttpBucket,
    observation: ObservationBucket,
    exchange: ExchangeBucket,
}

#[derive(Clone)]
pub struct Inspector {
    inner: Arc<Mutex<InspectorInner>>,
}

impl Default for Inspector {
    fn default() -> Self {
        Self::new(256)
    }
}

impl Inspector {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(InspectorInner {
                capacity: capacity.max(1),
                http: HttpBucket::default(),
                observation: ObservationBucket::default(),
                exchange: ExchangeBucket::default(),
            })),
        }
    }

    pub fn begin_http_request(&self) -> u64 {
        let Ok(mut guard) = self.inner.lock() else {
            return 0;
        };
        let seq = guard.http.next_seq;
        guard.http.next_seq += 1;
        guard.http.in_flight += 1;
        seq
    }

    pub fn finish_http_request(&self, record: HttpRequestRecord) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        let cap = guard.capacity;
        let h = &mut guard.http;
        h.in_flight = h.in_flight.saturating_sub(1);
        h.total_requests += 1;
        h.duration_sum_ms += record.duration_ms;
        if let Some(ttft) = record.ttft_ms {
            h.ttft_sum_ms += ttft;
            h.ttft_count += 1;
        }
        *h.per_model.entry(record.model.clone()).or_insert(0) += 1;
        match record.outcome {
            HttpOutcome::Completed => h.completed += 1,
            HttpOutcome::ClientDisconnected => h.client_disconnected += 1,
            HttpOutcome::InjectedError => h.injected_errors += 1,
            HttpOutcome::MidStreamDrop => h.mid_stream_drops += 1,
            HttpOutcome::MalformedSse => h.malformed_sse += 1,
            HttpOutcome::Backpressure503 => h.backpressure_503 += 1,
        }
        push_bounded(&mut h.recent, record, cap);
    }

    pub fn set_observation_status(&self, status: impl Into<String>, detail: Option<String>) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.observation.status = status.into();
            guard.observation.detail = detail;
        }
    }

    pub fn apply_observation_snapshot(&self, snapshot: &proto::InferenceObservationSnapshot) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        let obs = &mut guard.observation;
        obs.status = "subscribed".to_string();
        obs.detail = None;
        obs.host_generation = snapshot.host_generation;
        obs.ready_models = snapshot.ready_models.clone();
        obs.watermark_unix_ns = obs.watermark_unix_ns.max(snapshot.taken_at_unix_ns);
        obs.active = snapshot
            .active
            .iter()
            .map(|a| {
                (
                    a.execution_id,
                    ObservationExecutionRecord {
                        execution_id: a.execution_id,
                        host_generation: snapshot.host_generation,
                        model: a.model.clone(),
                        operation: inference_op_name(a.operation).to_string(),
                        streaming: a.streaming,
                        started_at_unix_ns: a.started_at_unix_ns,
                        first_output_at_unix_ns: None,
                        compute_finished_at_unix_ns: a
                            .compute_finished
                            .then_some(snapshot.taken_at_unix_ns),
                        ended_at_unix_ns: None,
                        ttft_ms: None,
                        duration_ms: None,
                        prompt_tokens: None,
                        cached_prompt_tokens: None,
                        completion_tokens: None,
                        outcome: None,
                    },
                )
            })
            .collect();
    }

    pub fn record_observation_overflow(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.observation.feed_overflows += 1;
        }
    }

    pub fn apply_observation_batch(&self, batch: proto::InferenceObservationBatch) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        let cap = guard.capacity;
        let obs = &mut guard.observation;
        if let Some(loss) = &batch.loss {
            obs.lost_events += loss.lost_events;
        }
        if let Some(resync) = &batch.resync {
            obs.host_generation = resync.host_generation;
            obs.ready_models = resync.ready_models.clone();
            obs.active = resync
                .active
                .iter()
                .map(|a| {
                    (
                        a.execution_id,
                        ObservationExecutionRecord {
                            execution_id: a.execution_id,
                            host_generation: resync.host_generation,
                            model: a.model.clone(),
                            operation: inference_op_name(a.operation).to_string(),
                            streaming: a.streaming,
                            started_at_unix_ns: a.started_at_unix_ns,
                            first_output_at_unix_ns: None,
                            compute_finished_at_unix_ns: a
                                .compute_finished
                                .then_some(resync.taken_at_unix_ns),
                            ended_at_unix_ns: None,
                            ttft_ms: None,
                            duration_ms: None,
                            prompt_tokens: None,
                            cached_prompt_tokens: None,
                            completion_tokens: None,
                            outcome: None,
                        },
                    )
                })
                .collect();
            obs.watermark_unix_ns = obs.watermark_unix_ns.max(resync.taken_at_unix_ns);
        }

        let host_gen = batch.host_generation;
        for event in batch.events {
            obs.events_received += 1;
            use proto::inference_observation_event::Kind;
            match event.kind {
                Some(Kind::Started(started)) => {
                    obs.active.insert(
                        event.execution_id,
                        ObservationExecutionRecord {
                            execution_id: event.execution_id,
                            host_generation: host_gen,
                            model: started.model,
                            operation: inference_op_name(started.operation).to_string(),
                            streaming: started.streaming,
                            started_at_unix_ns: event.at_unix_ns,
                            first_output_at_unix_ns: None,
                            compute_finished_at_unix_ns: None,
                            ended_at_unix_ns: None,
                            ttft_ms: None,
                            duration_ms: None,
                            prompt_tokens: None,
                            cached_prompt_tokens: None,
                            completion_tokens: None,
                            outcome: None,
                        },
                    );
                }
                Some(Kind::FirstOutput(_)) => {
                    if let Some(rec) = obs.active.get_mut(&event.execution_id) {
                        rec.first_output_at_unix_ns = Some(event.at_unix_ns);
                        rec.ttft_ms = Some(
                            event.at_unix_ns.saturating_sub(rec.started_at_unix_ns) / 1_000_000,
                        );
                    }
                }
                Some(Kind::ComputeFinished(_)) => {
                    if let Some(rec) = obs.active.get_mut(&event.execution_id) {
                        rec.compute_finished_at_unix_ns = Some(event.at_unix_ns);
                    }
                }
                Some(Kind::Usage(usage)) => {
                    if let Some(rec) = obs.active.get_mut(&event.execution_id) {
                        rec.prompt_tokens = Some(usage.prompt_tokens);
                        rec.cached_prompt_tokens = usage.cached_prompt_tokens;
                        rec.completion_tokens = Some(usage.completion_tokens);
                    }
                }
                Some(Kind::Ended(ended)) => {
                    let mut rec = obs.active.remove(&event.execution_id).unwrap_or_else(|| {
                        ObservationExecutionRecord {
                            execution_id: event.execution_id,
                            host_generation: host_gen,
                            model: "unknown".to_string(),
                            operation: "unspecified".to_string(),
                            streaming: false,
                            started_at_unix_ns: event.at_unix_ns,
                            first_output_at_unix_ns: None,
                            compute_finished_at_unix_ns: None,
                            ended_at_unix_ns: None,
                            ttft_ms: None,
                            duration_ms: None,
                            prompt_tokens: None,
                            cached_prompt_tokens: None,
                            completion_tokens: None,
                            outcome: None,
                        }
                    });
                    rec.ended_at_unix_ns = Some(event.at_unix_ns);
                    rec.duration_ms =
                        Some(event.at_unix_ns.saturating_sub(rec.started_at_unix_ns) / 1_000_000);
                    rec.outcome = Some(inference_outcome_name(ended.outcome).to_string());
                    obs.finished_total += 1;
                    push_bounded(&mut obs.finished, rec, cap);
                }
                Some(Kind::ModelReady(model)) => {
                    if !obs.ready_models.contains(&model) {
                        obs.ready_models.push(model.clone());
                    }
                    push_bounded(
                        &mut obs.model_events,
                        ModelLifecycleEvent {
                            sequence: event.sequence,
                            at_unix_ns: event.at_unix_ns,
                            event: "model_ready".to_string(),
                            model,
                        },
                        cap,
                    );
                }
                Some(Kind::ModelUnloaded(model)) => {
                    obs.ready_models.retain(|m| m != &model);
                    push_bounded(
                        &mut obs.model_events,
                        ModelLifecycleEvent {
                            sequence: event.sequence,
                            at_unix_ns: event.at_unix_ns,
                            event: "model_unloaded".to_string(),
                            model,
                        },
                        cap,
                    );
                }
                None => {}
            }
        }
        obs.watermark_unix_ns = obs.watermark_unix_ns.max(batch.watermark_unix_ns);
    }

    pub fn record_exchange_event(&self, event: &OpenAiExchangeEvent) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        let cap = guard.capacity;
        let now_ms = unix_now_ms();
        let ex = &mut guard.exchange;
        ex.total_events += 1;
        if event.admission_denied == Some(true) {
            ex.denied_count += 1;
        }
        if let Some(outcome) = &event.execution_outcome {
            *ex.outcomes.entry(outcome.clone()).or_insert(0) += 1;
        }

        if let Some(existing) = ex
            .recent
            .iter_mut()
            .rev()
            .find(|r| r.exchange_id == event.exchange_id)
        {
            if !existing.phases_seen.contains(&event.phase) {
                existing.phases_seen.push(event.phase.clone());
            }
            existing.last_seen_unix_ms = now_ms;
            existing.latency_ms = now_ms.saturating_sub(existing.first_seen_unix_ms);
            if event.observation_id.is_some() {
                existing.observation_id = event.observation_id.clone();
            }
            if event.model.is_some() {
                existing.model = event.model.clone();
            }
            for (k, v) in &event.headers {
                existing.headers.insert(k.clone(), v.clone());
            }
            if event.execution_outcome.is_some() {
                existing.execution_outcome = event.execution_outcome.clone();
            }
            if event.admission_denied.is_some() {
                existing.admission_denied = event.admission_denied;
            }
            if event.evidence_complete.is_some() {
                existing.evidence_complete = event.evidence_complete;
            }
            if event.observer_evidence_complete.is_some() {
                existing.observer_evidence_complete = event.observer_evidence_complete;
            }
            if event.request_wire_digest.is_some() {
                existing.request_wire_digest = event.request_wire_digest.clone();
            }
            if event.effective_request_wire_digest.is_some() {
                existing.effective_request_wire_digest =
                    event.effective_request_wire_digest.clone();
            }
            if event.response_wire_commitment.is_some() {
                existing.response_wire_commitment = event.response_wire_commitment.clone();
            }
            return;
        }

        ex.distinct_exchanges += 1;
        let record = ExchangeRecord {
            exchange_id: event.exchange_id.clone(),
            observation_id: event.observation_id.clone(),
            endpoint: event.endpoint.clone(),
            observation_point: event.observation_point.clone(),
            method: event.method.clone(),
            path: event.path.clone(),
            model: event.model.clone(),
            headers: event.headers.clone(),
            phases_seen: vec![event.phase.clone()],
            first_seen_unix_ms: now_ms,
            last_seen_unix_ms: now_ms,
            latency_ms: 0,
            execution_outcome: event.execution_outcome.clone(),
            admission_denied: event.admission_denied,
            evidence_complete: event.evidence_complete,
            observer_evidence_complete: event.observer_evidence_complete,
            request_wire_digest: event.request_wire_digest.clone(),
            effective_request_wire_digest: event.effective_request_wire_digest.clone(),
            response_wire_commitment: event.response_wire_commitment.clone(),
        };
        push_bounded(&mut ex.recent, record, cap);
    }

    pub fn inspect_recent(
        &self,
        source: InspectSource,
        limit: usize,
        model_filter: Option<&str>,
    ) -> InspectRecentResponse {
        let Ok(guard) = self.inner.lock() else {
            return InspectRecentResponse {
                source,
                capacity: 0,
                in_flight: 0,
                http: None,
                observation: None,
                exchange: None,
            };
        };
        let limit = limit.clamp(1, guard.capacity);

        let http = matches!(source, InspectSource::All | InspectSource::Http).then(|| {
            guard
                .http
                .recent
                .iter()
                .rev()
                .filter(|r| model_filter.is_none_or(|m| r.model == m))
                .take(limit)
                .cloned()
                .collect()
        });

        let observation =
            matches!(source, InspectSource::All | InspectSource::Observation).then(|| {
                ObservationInspectSlice {
                    status: guard.observation.status.clone(),
                    detail: guard.observation.detail.clone(),
                    active: guard
                        .observation
                        .active
                        .values()
                        .filter(|r| model_filter.is_none_or(|m| r.model == m))
                        .cloned()
                        .collect(),
                    recent_finished: guard
                        .observation
                        .finished
                        .iter()
                        .rev()
                        .filter(|r| model_filter.is_none_or(|m| r.model == m))
                        .take(limit)
                        .cloned()
                        .collect(),
                    recent_model_events: guard
                        .observation
                        .model_events
                        .iter()
                        .rev()
                        .filter(|e| model_filter.is_none_or(|m| e.model == m))
                        .take(limit)
                        .cloned()
                        .collect(),
                }
            });

        let exchange = matches!(source, InspectSource::All | InspectSource::Exchange).then(|| {
            guard
                .exchange
                .recent
                .iter()
                .rev()
                .filter(|r| model_filter.is_none_or(|m| r.model.as_deref() == Some(m)))
                .take(limit)
                .cloned()
                .collect()
        });

        InspectRecentResponse {
            source,
            capacity: guard.capacity,
            in_flight: guard.http.in_flight,
            http,
            observation,
            exchange,
        }
    }

    pub fn stats(&self) -> InspectorStats {
        let Ok(guard) = self.inner.lock() else {
            return InspectorStats {
                capacity: 0,
                http: HttpStats {
                    total_requests: 0,
                    in_flight: 0,
                    completed: 0,
                    client_disconnected: 0,
                    cancelled_count: 0,
                    injected_errors: 0,
                    mid_stream_drops: 0,
                    malformed_sse: 0,
                    backpressure_503: 0,
                    buffered_records: 0,
                    per_model: BTreeMap::new(),
                    avg_ttft_ms: None,
                    avg_duration_ms: None,
                },
                observation: ObservationStats {
                    status: "error".into(),
                    detail: Some("lock poisoned".into()),
                    host_generation: 0,
                    active_executions: 0,
                    finished_executions: 0,
                    ready_models: Vec::new(),
                    events_received: 0,
                    lost_events: 0,
                    feed_overflows: 0,
                    watermark_unix_ns: 0,
                    buffered_records: 0,
                },
                exchange: ExchangeStats {
                    total_events: 0,
                    distinct_exchanges: 0,
                    buffered_records: 0,
                    denied_count: 0,
                    outcomes: BTreeMap::new(),
                },
            };
        };
        let h = &guard.http;
        let o = &guard.observation;
        let e = &guard.exchange;

        InspectorStats {
            capacity: guard.capacity,
            http: HttpStats {
                total_requests: h.total_requests,
                in_flight: h.in_flight,
                completed: h.completed,
                client_disconnected: h.client_disconnected,
                cancelled_count: h.client_disconnected,
                injected_errors: h.injected_errors,
                mid_stream_drops: h.mid_stream_drops,
                malformed_sse: h.malformed_sse,
                backpressure_503: h.backpressure_503,
                buffered_records: h.recent.len(),
                per_model: h.per_model.clone(),
                avg_ttft_ms: (h.ttft_count > 0).then(|| h.ttft_sum_ms as f64 / h.ttft_count as f64),
                avg_duration_ms: (h.total_requests > 0)
                    .then(|| h.duration_sum_ms as f64 / h.total_requests as f64),
            },
            observation: ObservationStats {
                status: o.status.clone(),
                detail: o.detail.clone(),
                host_generation: o.host_generation,
                active_executions: o.active.len(),
                finished_executions: o.finished_total,
                ready_models: o.ready_models.clone(),
                events_received: o.events_received,
                lost_events: o.lost_events,
                feed_overflows: o.feed_overflows,
                watermark_unix_ns: o.watermark_unix_ns,
                buffered_records: o.finished.len(),
            },
            exchange: ExchangeStats {
                total_events: e.total_events,
                distinct_exchanges: e.distinct_exchanges,
                buffered_records: e.recent.len(),
                denied_count: e.denied_count,
                outcomes: e.outcomes.clone(),
            },
        }
    }

    pub fn clear(&self, source: InspectSource) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        if matches!(source, InspectSource::All | InspectSource::Http) {
            let in_flight = guard.http.in_flight;
            let next_seq = guard.http.next_seq;
            guard.http = HttpBucket {
                in_flight,
                next_seq,
                ..Default::default()
            };
        }
        if matches!(source, InspectSource::All | InspectSource::Observation) {
            let status = guard.observation.status.clone();
            let detail = guard.observation.detail.clone();
            let host_generation = guard.observation.host_generation;
            let ready_models = guard.observation.ready_models.clone();
            let active = guard.observation.active.clone();
            guard.observation = ObservationBucket {
                status,
                detail,
                host_generation,
                ready_models,
                active,
                ..Default::default()
            };
        }
        if matches!(source, InspectSource::All | InspectSource::Exchange) {
            guard.exchange = ExchangeBucket::default();
        }
    }
}

fn push_bounded<T>(buf: &mut VecDeque<T>, item: T, capacity: usize) {
    if buf.len() >= capacity {
        buf.pop_front();
    }
    buf.push_back(item);
}

pub fn unix_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn inference_op_name(raw: i32) -> &'static str {
    match proto::InferenceOperation::try_from(raw).ok() {
        Some(proto::InferenceOperation::ChatCompletion) => "chat_completion",
        Some(proto::InferenceOperation::Completion) => "completion",
        Some(proto::InferenceOperation::Responses) => "responses",
        Some(proto::InferenceOperation::Messages) => "messages",
        Some(proto::InferenceOperation::Embeddings) => "embeddings",
        Some(proto::InferenceOperation::Rerank) => "rerank",
        Some(proto::InferenceOperation::AudioSpeech) => "audio_speech",
        Some(proto::InferenceOperation::AudioTranscription) => "audio_transcription",
        Some(proto::InferenceOperation::AudioTranslation) => "audio_translation",
        Some(proto::InferenceOperation::SystemOne) => "system_one",
        _ => "unspecified",
    }
}

fn inference_outcome_name(raw: i32) -> &'static str {
    match proto::InferenceExecutionOutcome::try_from(raw).ok() {
        Some(proto::InferenceExecutionOutcome::Completed) => "completed",
        Some(proto::InferenceExecutionOutcome::Failed) => "failed",
        Some(proto::InferenceExecutionOutcome::Cancelled) => "cancelled",
        Some(proto::InferenceExecutionOutcome::Dropped) => "dropped",
        _ => "unspecified",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn http_ring_buffer_evicts_oldest_and_tracks_stats() {
        let insp = Inspector::new(2);
        for i in 1..=3 {
            let seq = insp.begin_http_request();
            insp.finish_http_request(HttpRequestRecord {
                seq,
                timestamp_unix_ms: 1000 + i,
                endpoint: "/v1/chat/completions".into(),
                model: "dev/echo".into(),
                stream: false,
                headers: BTreeMap::new(),
                http_status: 200,
                ttft_ms: Some(10),
                duration_ms: 20,
                tokens_emitted: 5,
                outcome: HttpOutcome::Completed,
                prompt_preview: Some(format!("req {i}")),
                override_applied: None,
            });
        }
        let recent = insp.inspect_recent(InspectSource::Http, 10, None);
        let records = recent.http.expect("http slice");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, 3);
        assert_eq!(records[1].seq, 2);

        let stats = insp.stats();
        assert_eq!(stats.http.total_requests, 3);
        assert_eq!(stats.http.completed, 3);
        assert_eq!(stats.http.in_flight, 0);
        assert_eq!(stats.http.per_model.get("dev/echo"), Some(&3));
    }

    #[test]
    fn correlates_observation_events_and_exchange_phases() {
        let insp = Inspector::new(16);
        insp.apply_observation_snapshot(&proto::InferenceObservationSnapshot {
            host_generation: 7,
            next_sequence: 1,
            taken_at_unix_ns: 1_000_000,
            ready_models: vec!["dev/echo".into()],
            active: Vec::new(),
        });
        insp.apply_observation_batch(proto::InferenceObservationBatch {
            host_generation: 7,
            watermark_unix_ns: 25_000_000,
            events: vec![
                proto::InferenceObservationEvent {
                    sequence: 1,
                    at_unix_ns: 10_000_000,
                    execution_id: 42,
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
                    at_unix_ns: 15_000_000,
                    execution_id: 42,
                    kind: Some(proto::inference_observation_event::Kind::FirstOutput(
                        proto::Empty {},
                    )),
                },
                proto::InferenceObservationEvent {
                    sequence: 3,
                    at_unix_ns: 22_000_000,
                    execution_id: 42,
                    kind: Some(proto::inference_observation_event::Kind::Usage(
                        proto::InferenceOutputUsage {
                            prompt_tokens: 12,
                            cached_prompt_tokens: Some(4),
                            completion_tokens: 9,
                        },
                    )),
                },
                proto::InferenceObservationEvent {
                    sequence: 4,
                    at_unix_ns: 25_000_000,
                    execution_id: 42,
                    kind: Some(proto::inference_observation_event::Kind::Ended(
                        proto::InferenceExecutionEnded {
                            outcome: proto::InferenceExecutionOutcome::Completed as i32,
                        },
                    )),
                },
            ],
            loss: None,
            resync: None,
        });

        let ev1 = OpenAiExchangeEvent::parse(&json!({
            "exchange_id": "ex-1",
            "phase": "request_received",
            "endpoint": "chat_completions",
            "observation_point": "gateway_ingress",
            "method": "POST",
            "path": "/v1/chat/completions",
            "model": "dev/echo",
            "headers": {"x-mesh-target": "peer-a"}
        }))
        .unwrap();
        let ev2 = OpenAiExchangeEvent::parse(&json!({
            "exchange_id": "ex-1",
            "phase": "exchange_finished",
            "endpoint": "chat_completions",
            "observation_point": "gateway_ingress",
            "method": "POST",
            "path": "/v1/chat/completions",
            "model": "dev/echo",
            "headers": {},
            "execution_outcome": "completed",
            "evidence_complete": true
        }))
        .unwrap();
        insp.record_exchange_event(&ev1);
        insp.record_exchange_event(&ev2);

        let all = insp.inspect_recent(InspectSource::All, 10, Some("dev/echo"));
        let obs = all.observation.expect("obs slice");
        assert_eq!(obs.recent_finished.len(), 1);
        assert_eq!(obs.recent_finished[0].ttft_ms, Some(5));
        assert_eq!(obs.recent_finished[0].duration_ms, Some(15));
        assert_eq!(obs.recent_finished[0].completion_tokens, Some(9));
        assert_eq!(obs.recent_finished[0].outcome.as_deref(), Some("completed"));

        let ex = all.exchange.expect("exchange slice");
        assert_eq!(ex.len(), 1);
        assert_eq!(
            ex[0].phases_seen,
            vec!["request_received", "exchange_finished"]
        );
        assert_eq!(
            ex[0].headers.get("x-mesh-target").map(String::as_str),
            Some("peer-a")
        );
        assert_eq!(ex[0].execution_outcome.as_deref(), Some("completed"));

        insp.clear(InspectSource::Exchange);
        assert_eq!(insp.stats().exchange.distinct_exchanges, 0);
        assert_eq!(insp.stats().observation.finished_executions, 1);
    }
}
