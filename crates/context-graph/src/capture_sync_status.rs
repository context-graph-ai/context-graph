use std::path::Path;
use std::time::Duration;

use contextdb_core::Lsn;
use contextdb_engine::sync_types::{ConflictPolicy, SyncDirection};
use serde_json::Value;

use crate::ingest::{
    CaptureReport, CaptureRequest, ConversationExtractor, record_conversation_with_extractor,
};
use crate::{ContextId, Result, Store};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsageReportState {
    Tokens(ProviderUsage),
    Unavailable,
    Zero,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelCallUsage {
    pub chunk_index: usize,
    pub usage: UsageReportState,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptureUsageReceipt {
    pub calls: Vec<ModelCallUsage>,
    pub total_tokens: Option<u64>,
    pub elapsed: Duration,
    pub chunk_elapsed: Vec<Duration>,
    pub dollars: Option<f64>,
}

impl Default for CaptureUsageReceipt {
    fn default() -> Self {
        Self {
            calls: Vec::new(),
            total_tokens: None,
            elapsed: Duration::ZERO,
            chunk_elapsed: Vec::new(),
            dollars: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostedProviderReply {
    pub body: String,
    pub elapsed: Duration,
}

pub trait HostedProviderBoundary {
    fn call_model(&self, prompt: &str) -> Result<HostedProviderReply>;
    fn call_count(&self) -> usize;
}

#[derive(Debug, Clone, PartialEq)]
pub enum CapturePricing {
    FlatSubscription,
    PayPerToken { usd_per_token: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaptureEstimate {
    pub provider_calls: usize,
    pub projected_chunks: usize,
    pub projected_tokens: Option<u64>,
    pub projected_time: Duration,
    pub projected_dollars: Option<f64>,
}

impl Default for CaptureEstimate {
    fn default() -> Self {
        Self {
            provider_calls: 0,
            projected_chunks: 0,
            projected_tokens: None,
            projected_time: Duration::ZERO,
            projected_dollars: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressKind {
    InFlight,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureProgressEvent {
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub elapsed: Duration,
    pub rough_remaining: Duration,
    pub kind: ProgressKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncStatusReport {
    pub remote_url: Option<String>,
    pub tenant_id: Option<String>,
    pub push_watermark: Lsn,
    pub pull_watermark: Lsn,
    pub pending_push_count: usize,
    pub sync_client_constructed: bool,
    pub nats_connection_opened: bool,
}

impl Default for SyncStatusReport {
    fn default() -> Self {
        Self {
            remote_url: None,
            tenant_id: None,
            push_watermark: Lsn(0),
            pull_watermark: Lsn(0),
            pending_push_count: 0,
            sync_client_constructed: false,
            nats_connection_opened: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedUsage {
    pub total_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultContextCaptureResult {
    pub context_id: ContextId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextGraphSyncTablePolicy {
    pub table: &'static str,
    pub direction: SyncDirection,
    pub conflict_policy: ConflictPolicy,
}

pub trait SyncStatusBoundary {
    fn push_watermark(&self) -> Lsn;
    fn pull_watermark(&self) -> Lsn;
    fn has_pending_push_changes(&self) -> Result<bool>;
    fn pending_push_change_count(&self) -> Result<usize>;
    fn is_connected(&self) -> bool;
}

impl From<CapturedUsage> for Value {
    fn from(value: CapturedUsage) -> Self {
        Value::from(value.total_tokens)
    }
}

pub fn capture_with_usage_visibility(
    store: &Store,
    request: CaptureRequest,
    _max_chars_per_chunk: usize,
    extractor: &dyn ConversationExtractor,
    _provider: &dyn HostedProviderBoundary,
    _pricing: CapturePricing,
) -> Result<CaptureUsageReceipt> {
    let _ = record_conversation_with_extractor(store, request, extractor)?;
    Ok(CaptureUsageReceipt::default())
}

pub fn estimate_capture(
    _transcript: &str,
    _max_chars_per_chunk: usize,
    _per_chunk_time: Duration,
    _pricing: CapturePricing,
    _provider: &dyn HostedProviderBoundary,
) -> CaptureEstimate {
    CaptureEstimate::default()
}

pub fn capture_with_progress_observer(
    store: &Store,
    request: CaptureRequest,
    extractor: &dyn ConversationExtractor,
    _chunk_window: usize,
    _observer: impl FnMut(CaptureProgressEvent),
) -> Result<CaptureReport> {
    record_conversation_with_extractor(store, request, extractor)
}

fn default_capture_context_id(
    _tenant_id: Option<&str>,
    _explicit_context: Option<&str>,
    _cwd: &Path,
) -> ContextId {
    ContextId::new_v7()
}

pub fn capture_with_default_context(
    _store: &Store,
    tenant_id: Option<&str>,
    explicit_context: Option<&str>,
    cwd: &Path,
    _transcript: &str,
    _extractor: &dyn ConversationExtractor,
) -> Result<DefaultContextCaptureResult> {
    Ok(DefaultContextCaptureResult {
        context_id: default_capture_context_id(tenant_id, explicit_context, cwd),
    })
}

pub fn sync_status_report(
    _store: &Store,
    _remote_url: Option<&str>,
    _tenant_id: Option<&str>,
    _boundary: Option<&dyn SyncStatusBoundary>,
) -> SyncStatusReport {
    SyncStatusReport::default()
}

pub fn context_graph_sync_table_contract() -> Vec<ContextGraphSyncTablePolicy> {
    Vec::new()
}
