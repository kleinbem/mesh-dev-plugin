//! Embedded OpenAI-compatible HTTP server (`127.0.0.1:0` in plugin mode or configurable in `serve` mode).

pub mod openai;
pub mod routes;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tokio::sync::{Notify, Semaphore};

/// Maximum accepted HTTP request body size (4 MiB) to protect the `< 5 MB` dev plugin from OOM.
pub const MAX_HTTP_BODY_BYTES: usize = 4 * 1024 * 1024;

use crate::faults::FaultController;
use crate::inspector::Inspector;
use crate::models::{ModelDefaults, parse_dynamic_model};

#[derive(Clone)]
pub struct AppState {
    pub enabled_models: Vec<String>,
    pub allow_dynamic_models: bool,
    pub max_concurrency: usize,
    pub defaults: ModelDefaults,
    pub faults: FaultController,
    pub inspector: Inspector,
    pub permits: Arc<Semaphore>,
    shutdown: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
}

impl AppState {
    pub fn new(
        enabled_models: Vec<String>,
        allow_dynamic_models: bool,
        defaults: ModelDefaults,
        max_concurrency: usize,
        faults: FaultController,
        inspector: Inspector,
    ) -> Self {
        let max_concurrency = max_concurrency.max(1);
        Self {
            enabled_models,
            allow_dynamic_models,
            max_concurrency,
            defaults,
            faults,
            inspector,
            permits: Arc::new(Semaphore::new(max_concurrency)),
            shutdown: Arc::new(AtomicBool::new(false)),
            shutdown_notify: Arc::new(Notify::new()),
        }
    }

    pub fn is_model_supported(&self, model: &str) -> bool {
        self.enabled_models.iter().any(|m| m == model)
            || (self.allow_dynamic_models && parse_dynamic_model(model).is_some())
    }

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.shutdown_notify.notify_waiters();
    }

    pub fn is_shutting_down(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    pub async fn shutdown_notified(&self) {
        let notified = self.shutdown_notify.notified();
        if self.is_shutting_down() {
            return;
        }
        notified.await;
    }
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/completions", post(routes::completions))
        .route("/v1/responses", post(routes::responses))
        .layer(DefaultBodyLimit::max(MAX_HTTP_BODY_BYTES))
        .with_state(state)
}

pub async fn serve_with_state(listener: TcpListener, state: AppState) -> Result<()> {
    let shutdown_state = state.clone();
    let app = build_router(state);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_state.shutdown_notified().await;
        })
        .await?;
    Ok(())
}
