use std::collections::BTreeMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::time::{Duration, Instant};

use chrono::Utc;
use contextdb_core::Lsn;
use contextdb_engine::Database;
use contextdb_engine::sync_types::{ApplyResult, ConflictPolicies, ConflictPolicy, SyncDirection};
use contextdb_engine::work_ledger;
use serde_json::Value;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::ingest::{
    CaptureReport, CaptureRequest, ConversationExtractor, ExtractionRequest,
    RollingCaptureAccumulator, record_conversation_with_extractor,
};
use crate::schema::TABLES;
use crate::{
    CgError, ContextId, CreateEntity, EnforcementLevel, EntityId, EntityType, EvidenceId,
    EvidenceKind, EvidenceProducer, EvidenceRef, ListEntityFilter, ModelCallOutcome,
    ModelCallReceipt, ObservationId, RecordObservation, Result, RetentionStatus, Store,
};

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

/// Local-understanding runtime facts riding the SAME call record as its usage — the operational
/// join test 1.4 (`local_gguf_run_receipt_carries_usage_and_backend_from_same_run`) depends on.
/// Deliberately primitive-only (no `cg-local-llm` / `cg-catalog` types): `context-graph` must not
/// depend on those crates, so the caller (cg-catalog) renders its typed backend report down to
/// these plain strings/numbers before attaching it to a `ModelCallUsage`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalRuntimeReceipt {
    pub binary_capability: String,
    pub configured_backend: String,
    pub active_backend: String,
    pub gpu_layers: Option<i32>,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCallUsage {
    pub chunk_index: usize,
    pub usage: UsageReportState,
    pub elapsed: Duration,
    pub reliability_level: EnforcementLevel,
    pub outcome: ModelCallOutcome,
    /// The originating flow when the record is flow-tagged ("extraction" | "identity-judge" |
    /// "repair"); `None` on the raw cost-receipt path, which carries usage untagged.
    pub flow_tag: Option<String>,
    /// The model id when the record is flow-tagged; `None` on the raw cost-receipt path.
    pub model_id: Option<String>,
    /// Local-understanding backend/runtime facts for THIS call, when it ran through the local
    /// GGUF route. `None` for hosted calls and for local calls that predate this field.
    /// Operational metadata only — never a decision-basis lane.
    pub runtime: Option<LocalRuntimeReceipt>,
}

impl ModelCallUsage {
    #[must_use]
    pub fn model_receipt(&self) -> ModelCallReceipt {
        ModelCallReceipt {
            reliability_level: self.reliability_level,
            outcome: self.outcome,
            structured_output_rejected: false,
        }
    }
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
    pub hub_connection_opened: bool,
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
            hub_connection_opened: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedUsage {
    pub total_tokens: u64,
}

impl CapturedUsage {
    pub fn to_properties_value(&self) -> Value {
        Value::from(self.total_tokens)
    }
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

pub type SyncDrainFuture<'a> = Pin<Box<dyn Future<Output = Result<ApplyResult>> + 'a>>;
pub type SyncProbeFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + 'a>>;

pub trait SyncPushBoundary {
    fn push_watermark(&self) -> Lsn;
    fn has_pending_push_changes(&self) -> Result<bool>;
    fn refresh_push_state(&self) -> SyncProbeFuture<'_> {
        Box::pin(async { Ok(()) })
    }
    fn push_round(&self) -> SyncDrainFuture<'_>;
}

pub trait SyncPullBoundary {
    fn pull_watermark(&self) -> Lsn;
    fn pull_round<'a>(&'a self, policies: &'a ConflictPolicies) -> SyncDrainFuture<'a>;
}

#[derive(Debug, Clone)]
pub struct SyncDrainReport {
    pub rounds: usize,
    pub result: ApplyResult,
}

pub fn capture_with_usage_visibility(
    store: &Store,
    request: CaptureRequest,
    max_chars_per_chunk: usize,
    extractor: &dyn ConversationExtractor,
    _provider: &dyn HostedProviderBoundary,
    pricing: CapturePricing,
) -> Result<CaptureUsageReceipt> {
    let started = Instant::now();
    let context_label = request.context_label.clone();
    // Pass the extractor directly so the per-chunk usage records (chunk_index,
    // flow_tag, model_id) are captured as emitted by the extractor, without
    // additional tagging. Flow-tagging for the model-usage lane is applied by
    // record_conversation_internal for the J-facing single-capture path; the
    // cost-visibility path reads the raw per-chunk records that
    // capture_with_progress_observer collects in its loop.
    let report =
        capture_with_progress_observer(store, request, extractor, max_chars_per_chunk, |_| {})?;
    let calls = report.model_usage;
    let receipt = usage_receipt(calls, pricing, started.elapsed());
    persist_usage_receipt(store, &context_label, &receipt)?;
    Ok(receipt)
}

pub fn estimate_capture(
    transcript: &str,
    max_chars_per_chunk: usize,
    per_chunk_time: Duration,
    pricing: CapturePricing,
    _provider: &dyn HostedProviderBoundary,
) -> CaptureEstimate {
    let projected_chunks = transcript_chunks(transcript, max_chars_per_chunk).len();
    let projected_token_count = projected_chunks as u64 * 1290;
    let projected_tokens = Some(projected_token_count);
    let projected_dollars = match pricing {
        CapturePricing::FlatSubscription => None,
        CapturePricing::PayPerToken { usd_per_token } => {
            Some(projected_token_count as f64 * usd_per_token)
        }
    };
    CaptureEstimate {
        provider_calls: 0,
        projected_chunks,
        projected_tokens,
        projected_time: per_chunk_time * projected_chunks as u32,
        projected_dollars,
    }
}

pub fn capture_with_progress_observer(
    store: &Store,
    request: CaptureRequest,
    extractor: &dyn ConversationExtractor,
    chunk_window: usize,
    mut observer: impl FnMut(CaptureProgressEvent),
) -> Result<CaptureReport> {
    let text = request.transcript.as_str();
    if text.trim().is_empty() {
        return record_conversation_with_extractor(store, request, extractor);
    }
    let chunks = transcript_chunks(text, chunk_window);
    let total_chunks = chunks.len();
    let started = Instant::now();
    if total_chunks == 1 {
        observer(progress_event(
            1,
            total_chunks,
            started.elapsed(),
            ProgressKind::InFlight,
        ));
        let mut report = record_conversation_with_extractor(store, request, extractor)?;
        observer(progress_event(
            1,
            total_chunks,
            started.elapsed(),
            ProgressKind::Completed,
        ));
        report.resolved_chunk_window = Some(chunk_window.max(1));
        return Ok(report);
    }
    let mut rolling = RollingCaptureAccumulator::new(16_000);
    let mut per_chunk_usage: Vec<ModelCallUsage> = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let chunk_index = index + 1;
        let chunk_started = Instant::now();
        observer(progress_event(
            chunk_index,
            total_chunks,
            started.elapsed(),
            ProgressKind::InFlight,
        ));
        let harvest = extractor.extract(ExtractionRequest {
            context_label: &request.context_label,
            transcript: chunk,
            prior_decisions: &[],
        })?;
        let chunk_elapsed = chunk_started.elapsed();
        // Drain the extractor's usage records immediately after each chunk.
        // The extractor's internal chunk_index counter resets on each drain,
        // so we assign the authoritative chunk_index from the loop variable.
        // When the extractor emits no records for a chunk (e.g. a verbatim
        // or deterministic double that never calls a model), synthesize one
        // timing-only record so callers can always rely on at least one entry
        // per chunk for timing and cost accounting.
        let mut drained = extractor.drain_model_usage();
        if drained.is_empty() {
            per_chunk_usage.push(ModelCallUsage {
                chunk_index,
                usage: UsageReportState::Unavailable,
                elapsed: chunk_elapsed,
                reliability_level: EnforcementLevel::Floor,
                outcome: ModelCallOutcome::Captured,
                flow_tag: None,
                model_id: None,
                runtime: None,
            });
        } else {
            for record in &mut drained {
                record.chunk_index = chunk_index;
            }
            per_chunk_usage.extend(drained);
        }
        rolling.push_harvest(harvest);
        observer(progress_event(
            chunk_index,
            total_chunks,
            started.elapsed(),
            ProgressKind::Completed,
        ));
    }
    let mut report = rolling.finish_capture(
        store,
        &request.context_label,
        total_chunks,
        extractor,
        &request.transcript,
    )?;
    // Replace the model_usage that finish_capture produced (from its inner
    // record_conversation call) with the per-chunk records collected above.
    // Those records carry the loop-authoritative chunk_index and the raw
    // flow_tag / model_id values the extractor emitted — no additional
    // tagging is applied here. Cost-visibility callers depend on this shape.
    report.model_usage = per_chunk_usage;
    report.resolved_chunk_window = Some(chunk_window.max(1));
    Ok(report)
}

fn default_capture_context_label(
    tenant_id: Option<&str>,
    explicit_context: Option<&str>,
    _cwd: &Path,
) -> String {
    explicit_context
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .or_else(|| {
            tenant_id
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "default".to_string())
}

pub fn capture_with_default_context(
    store: &Store,
    tenant_id: Option<&str>,
    explicit_context: Option<&str>,
    cwd: &Path,
    transcript: &str,
    extractor: &dyn ConversationExtractor,
) -> Result<DefaultContextCaptureResult> {
    let context_label = default_capture_context_label(tenant_id, explicit_context, cwd);
    let _ = record_conversation_with_extractor(
        store,
        CaptureRequest {
            context_label: context_label.clone(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        extractor,
    )?;
    let context_id = store
        .list_contexts()?
        .into_iter()
        .find(|context| context.name == context_label)
        .map(|context| context.id)
        .ok_or_else(|| CgError::Engine(format!("capture context not found: {context_label}")))?;
    Ok(DefaultContextCaptureResult { context_id })
}

pub fn sync_status_report(
    _store: &Store,
    remote_url: Option<&str>,
    tenant_id: Option<&str>,
    boundary: Option<&dyn SyncStatusBoundary>,
) -> SyncStatusReport {
    let Some(boundary) = boundary else {
        return SyncStatusReport {
            remote_url: remote_url.map(str::to_string),
            tenant_id: tenant_id.map(str::to_string),
            ..SyncStatusReport::default()
        };
    };
    SyncStatusReport {
        remote_url: remote_url.map(str::to_string),
        tenant_id: tenant_id.map(str::to_string),
        push_watermark: boundary.push_watermark(),
        pull_watermark: boundary.pull_watermark(),
        pending_push_count: boundary
            .pending_push_change_count()
            .unwrap_or_else(|_| usize::from(boundary.has_pending_push_changes().unwrap_or(false))),
        sync_client_constructed: true,
        hub_connection_opened: boundary.is_connected(),
    }
}

pub async fn drain_sync_push(boundary: &dyn SyncPushBoundary) -> Result<SyncDrainReport> {
    let mut report = empty_sync_drain_report(boundary.push_watermark());
    for _ in 0..MAX_SYNC_DRAIN_ROUNDS {
        boundary.refresh_push_state().await?;
        let before = boundary.push_watermark();
        let round = boundary.push_round().await?;
        merge_sync_round(&mut report, round, boundary.push_watermark());
        if !boundary.has_pending_push_changes()? {
            return Ok(report);
        }
        if boundary.push_watermark() == before {
            return Err(CgError::Engine(
                "cg sync push made no progress while pending changes remained".to_string(),
            ));
        }
    }
    Err(CgError::Engine(format!(
        "cg sync push did not drain pending changes after {MAX_SYNC_DRAIN_ROUNDS} rounds"
    )))
}

pub async fn drain_sync_pull(
    boundary: &dyn SyncPullBoundary,
    policies: &ConflictPolicies,
) -> Result<SyncDrainReport> {
    let mut report = empty_sync_drain_report(boundary.pull_watermark());
    for _ in 0..MAX_SYNC_DRAIN_ROUNDS {
        let before = boundary.pull_watermark();
        let round = boundary.pull_round(policies).await?;
        let after = boundary.pull_watermark();
        merge_sync_round(&mut report, round, after);
        if after == before {
            return Ok(report);
        }
    }
    Err(CgError::Engine(format!(
        "cg sync pull did not drain remote changes after {MAX_SYNC_DRAIN_ROUNDS} rounds"
    )))
}

pub fn context_graph_sync_table_contract() -> Vec<ContextGraphSyncTablePolicy> {
    TABLES
        .iter()
        .map(|table| {
            let (direction, conflict_policy) = sync_policy_for_table(table.name);
            ContextGraphSyncTablePolicy {
                table: table.name,
                direction,
                conflict_policy,
            }
        })
        .collect()
}

pub fn context_graph_sync_conflict_policies() -> ConflictPolicies {
    let mut per_table = context_graph_sync_table_contract()
        .into_iter()
        .map(|policy| (policy.table.to_string(), policy.conflict_policy))
        .collect::<std::collections::HashMap<_, _>>();
    for table in LOCAL_ONLY_RUNTIME_SYNC_TABLES {
        per_table.insert(table.to_string(), ConflictPolicy::InsertIfNotExists);
    }
    // The seven work-ledger fabric tables carry the shipped ledger's own per-table conflict contract
    // verbatim — the ledger owns that arbitration; cg never re-derives it.
    for (table, policy) in work_ledger::work_ledger_conflict_policy_entries() {
        per_table.insert(table.to_string(), policy);
    }
    ConflictPolicies {
        per_table,
        default: ConflictPolicy::ServerWins,
    }
}

pub trait ContextGraphSyncRegistration {
    fn set_default_conflict_policy(&self, policy: ConflictPolicy);
    fn set_table_direction(&self, table: &str, direction: SyncDirection);
    fn set_conflict_policy(&self, table: &str, policy: ConflictPolicy);
}

pub fn register_context_graph_sync_client(client: &impl ContextGraphSyncRegistration) {
    client.set_default_conflict_policy(ConflictPolicy::ServerWins);
    for policy in context_graph_sync_table_contract() {
        client.set_table_direction(policy.table, policy.direction);
        client.set_conflict_policy(policy.table, policy.conflict_policy);
    }
    for table in LOCAL_ONLY_RUNTIME_SYNC_TABLES {
        client.set_table_direction(table, SyncDirection::None);
        client.set_conflict_policy(table, ConflictPolicy::InsertIfNotExists);
    }
    // The work-ledger fabric tables sync tenant-wide (both directions) under the shipped ledger's own
    // conflict contract, so a submitter's job, a worker's claim/result, and the ephemeral input copy
    // converge across the tenant's machines.
    for (table, policy) in work_ledger::work_ledger_conflict_policy_entries() {
        client.set_table_direction(table, SyncDirection::Both);
        client.set_conflict_policy(table, policy);
    }
}

pub fn context_graph_pending_push_change_count(db: &Database, since: Lsn) -> usize {
    let mut directions = context_graph_sync_table_contract()
        .into_iter()
        .map(|policy| (policy.table.to_string(), policy.direction))
        .collect::<std::collections::HashMap<_, _>>();
    for table in LOCAL_ONLY_RUNTIME_SYNC_TABLES {
        directions.insert(table.to_string(), SyncDirection::None);
    }
    for (table, _) in work_ledger::work_ledger_conflict_policy_entries() {
        directions.insert(table.to_string(), SyncDirection::Both);
    }
    let changes = db
        .changes_since(since)
        .filter_by_direction(&directions, &[SyncDirection::Push, SyncDirection::Both]);
    changes.rows.len() + changes.edges.len() + changes.vectors.len() + changes.ddl.len()
}

const LOCAL_ONLY_RUNTIME_SYNC_TABLES: &[&str] = &["cg_changed_cursors", "cg_runtime_models"];
const MAX_SYNC_DRAIN_ROUNDS: usize = 32;

fn empty_sync_drain_report(lsn: Lsn) -> SyncDrainReport {
    SyncDrainReport {
        rounds: 0,
        result: ApplyResult {
            applied_rows: 0,
            skipped_rows: 0,
            conflicts: Vec::new(),
            new_lsn: lsn,
        },
    }
}

fn merge_sync_round(report: &mut SyncDrainReport, round: ApplyResult, watermark: Lsn) {
    report.rounds += 1;
    report.result.applied_rows += round.applied_rows;
    report.result.skipped_rows += round.skipped_rows;
    report.result.conflicts.extend(round.conflicts);
    report.result.new_lsn = watermark;
}

pub fn provider_usage_from_str(body: &str) -> Option<ProviderUsage> {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| provider_usage_from_value(&value))
}

pub fn provider_usage_from_value(value: &Value) -> Option<ProviderUsage> {
    let usage = value.get("usage")?;
    let prompt_tokens = token_field(usage, &["prompt_tokens", "input_tokens"])?;
    let completion_tokens = token_field(usage, &["completion_tokens", "output_tokens"])?;
    let total_tokens = token_field(usage, &["total_tokens"])
        .unwrap_or_else(|| prompt_tokens.saturating_add(completion_tokens));
    Some(ProviderUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    })
}

fn token_field(value: &Value, names: &[&str]) -> Option<u64> {
    names
        .iter()
        .find_map(|name| value.get(*name).and_then(Value::as_u64))
}

fn usage_receipt(
    calls: Vec<ModelCallUsage>,
    pricing: CapturePricing,
    measured_elapsed: Duration,
) -> CaptureUsageReceipt {
    let chunk_elapsed = calls.iter().map(|call| call.elapsed).collect::<Vec<_>>();
    let summed_elapsed = chunk_elapsed.iter().copied().sum::<Duration>();
    let total_tokens = if calls.is_empty()
        || calls
            .iter()
            .any(|call| matches!(call.usage, UsageReportState::Unavailable))
    {
        None
    } else {
        Some(
            calls
                .iter()
                .map(|call| match &call.usage {
                    UsageReportState::Tokens(usage) => usage.total_tokens,
                    UsageReportState::Zero => 0,
                    UsageReportState::Unavailable => 0,
                })
                .sum::<u64>(),
        )
    };
    let dollars = match (pricing, total_tokens) {
        (CapturePricing::PayPerToken { usd_per_token }, Some(tokens)) => {
            Some(tokens as f64 * usd_per_token)
        }
        _ => None,
    };
    CaptureUsageReceipt {
        calls,
        total_tokens,
        elapsed: measured_elapsed.max(summed_elapsed),
        chunk_elapsed,
        dollars,
    }
}

pub fn usage_receipt_from_calls(
    calls: Vec<ModelCallUsage>,
    pricing: CapturePricing,
    measured_elapsed: Duration,
) -> CaptureUsageReceipt {
    usage_receipt(calls, pricing, measured_elapsed)
}

pub fn usage_receipt_json(receipt: &CaptureUsageReceipt) -> Value {
    json!({
        "calls": receipt.calls.iter().map(model_call_usage_json).collect::<Vec<_>>(),
        "total_tokens": receipt.total_tokens,
        "elapsed_ms": receipt.elapsed.as_millis() as u64,
        "dollars": receipt.dollars,
    })
}

fn persist_usage_receipt(
    store: &Store,
    context_label: &str,
    receipt: &CaptureUsageReceipt,
) -> Result<()> {
    if receipt.calls.is_empty() {
        return Ok(());
    }
    let context_id = store
        .list_contexts()?
        .into_iter()
        .find(|context| context.name == context_label)
        .map(|context| context.id)
        .ok_or_else(|| CgError::Engine(format!("capture context not found: {context_label}")))?;
    let usage = usage_receipt_json(receipt);
    let observed_at = Utc::now();
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let hash = hex_sha256(&usage.to_string());
    store.record_observation(RecordObservation {
        id: observation_id,
        entity_id: provider_usage_entity_id(store, context_id)?,
        context_id,
        observation_type: "provider_usage".to_string(),
        source: "context-graph:capture-usage".to_string(),
        observed_at,
        evidence: vec![EvidenceRef {
            id: evidence_id,
            observation_id,
            context_id,
            kind: EvidenceKind::StructuredSignal,
            source_ref: format!("cas://sha256:{hash}"),
            content_hash: Some(format!("sha256:{hash}")),
            captured_at: Some(observed_at),
            producer: EvidenceProducer {
                system: "context-graph".to_string(),
                pipeline_version: "capture-usage-v1".to_string(),
                ..Default::default()
            },
            retention_status: RetentionStatus::RetainedExternal,
            properties: BTreeMap::from([("usage".to_string(), usage.clone())]),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::from([("usage".to_string(), usage)]),
        embeddings: Vec::new(),
    })?;
    Ok(())
}

fn model_call_usage_json(call: &ModelCallUsage) -> Value {
    let receipt = call.model_receipt();
    let reliability_level = receipt.reliability_level.as_receipt_str();
    let outcome = receipt.outcome.as_receipt_str();
    let mut value = match &call.usage {
        UsageReportState::Tokens(usage) => json!({
            "chunk_index": call.chunk_index,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "generated_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "elapsed_ms": call.elapsed.as_millis() as u64,
            "wall_clock_ms": call.elapsed.as_millis() as u64,
            "reliability_level": reliability_level,
            "outcome": outcome,
        }),
        UsageReportState::Zero => json!({
            "chunk_index": call.chunk_index,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "generated_tokens": 0,
            "total_tokens": 0,
            "elapsed_ms": call.elapsed.as_millis() as u64,
            "wall_clock_ms": call.elapsed.as_millis() as u64,
            "reliability_level": reliability_level,
            "outcome": outcome,
        }),
        UsageReportState::Unavailable => json!({
            "chunk_index": call.chunk_index,
            "usage": null,
            "elapsed_ms": call.elapsed.as_millis() as u64,
            "wall_clock_ms": call.elapsed.as_millis() as u64,
            "reliability_level": reliability_level,
            "outcome": outcome,
        }),
    };
    // Fold local-understanding runtime facts into the SAME call record (test 1.4's "same run"
    // requirement) — additive keys only, never replacing the usage fields above.
    if let Some(runtime) = &call.runtime
        && let Some(obj) = value.as_object_mut()
    {
        obj.insert(
            "binary_capability".to_string(),
            json!(runtime.binary_capability),
        );
        obj.insert(
            "configured_backend".to_string(),
            json!(runtime.configured_backend),
        );
        obj.insert("active_backend".to_string(), json!(runtime.active_backend));
        obj.insert("gpu_layers".to_string(), json!(runtime.gpu_layers));
        obj.insert(
            "fallback_reason".to_string(),
            json!(runtime.fallback_reason),
        );
    }
    value
}

fn progress_event(
    chunk_index: usize,
    total_chunks: usize,
    elapsed: Duration,
    kind: ProgressKind,
) -> CaptureProgressEvent {
    let remaining_chunks = match kind {
        ProgressKind::InFlight => total_chunks.saturating_sub(chunk_index) + 1,
        ProgressKind::Completed => total_chunks.saturating_sub(chunk_index),
    };
    CaptureProgressEvent {
        chunk_index,
        total_chunks,
        elapsed,
        rough_remaining: Duration::from_millis(remaining_chunks as u64 * 100),
        kind,
    }
}

fn transcript_chunks(text: &str, max_chars_per_chunk: usize) -> Vec<&str> {
    let chunk_window = max_chars_per_chunk.max(1);
    if text.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + chunk_window).min(text.len());
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            end = text[start..]
                .char_indices()
                .nth(1)
                .map(|(offset, _)| start + offset)
                .unwrap_or(text.len());
        }
        chunks.push(&text[start..end]);
        start = end;
    }
    chunks
}

fn hex_sha256(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn provider_usage_entity_id(store: &Store, context_id: ContextId) -> Result<EntityId> {
    const NAME: &str = "provider usage model";
    if let Some(entity) = store
        .list_entities(ListEntityFilter {
            entity_type: Some(EntityType::Service),
            name_contains: Some(NAME.to_string()),
            context_id: Some(context_id),
            ..Default::default()
        })?
        .into_iter()
        .find(|entity| entity.name == NAME)
    {
        return Ok(entity.id);
    }
    let entity = store.create_entity(CreateEntity {
        entity_type: EntityType::Service,
        name: NAME.to_string(),
        properties: BTreeMap::from([(
            "usage_role".to_string(),
            Value::String("provider_usage".to_string()),
        )]),
        tags: vec!["provider_usage".to_string()],
        context_id,
    })?;
    Ok(entity.id)
}

fn sync_policy_for_table(table: &str) -> (SyncDirection, ConflictPolicy) {
    let write_once = (SyncDirection::Both, ConflictPolicy::InsertIfNotExists);
    let hub_arbitrated = (SyncDirection::Both, ConflictPolicy::ServerWins);
    match table {
        "audit_log" => (SyncDirection::None, ConflictPolicy::InsertIfNotExists),
        "intentions"
        | "decisions"
        | "entities"
        | "reindex_cursors"
        | "outcomes"
        | "invalidations"
        | "entity_current_snapshots" => hub_arbitrated,
        // Skill-loop tables: explicit per-table sync policies.
        //   cg_learned_skills syncs tenant-wide, hub-arbitrated (ServerWins).
        //   cg_teaching_events syncs tenant-wide as an order-free set union (InsertIfNotExists).
        //   cg_skill_firing_trace is local-only per device (None).
        //   cg_review_cursor is local-only per owner (None).
        "cg_learned_skills" => hub_arbitrated,
        "cg_teaching_events" => write_once,
        "cg_skill_firing_trace" => (SyncDirection::None, ConflictPolicy::InsertIfNotExists),
        "cg_review_cursor" => (SyncDirection::None, ConflictPolicy::InsertIfNotExists),
        // Pending review items: local-only (like the review cursor, never synced).
        "cg_pending_review_items" => (SyncDirection::None, ConflictPolicy::InsertIfNotExists),
        // Overlay corrections + compensation/reversal events: append-only, propagate tenant-wide.
        "cg_overlay_entries" => write_once,
        "cg_compensation_events" => write_once,
        // Tier-0 capture tables (schema v10). Raw source chunks, retrieval touches, and local
        // settings are all machine-local runtime state that NEVER leaves the device — raw is the NDA
        // line, and the touches/settings are per-device. The private understanding queue is retired
        // (the work ledger's computed job state subsumes it). The understanding completion is a
        // write-once LOCAL apply marker: it records that THIS node already applied a result onto its
        // digest, so it stays machine-local (`None`) — the structured graph rows are what sync, not
        // the apply guard.
        "cg_source_chunks" => (SyncDirection::None, ConflictPolicy::InsertIfNotExists),
        "cg_retrieval_touches" => (SyncDirection::None, ConflictPolicy::InsertIfNotExists),
        "cg_local_settings" => (SyncDirection::None, ConflictPolicy::InsertIfNotExists),
        "cg_understanding_completions" => (SyncDirection::None, ConflictPolicy::InsertIfNotExists),
        _ => write_once,
    }
}
