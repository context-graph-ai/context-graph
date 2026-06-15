use std::collections::BTreeMap;

use crate::{CgError, ContextId, DigestId, Result, Store};
use crate::{Embedder, EmbeddingInput, EmbeddingOutput};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HarvestCounts {
    pub goals: u32,
    pub decisions: u32,
    pub observations: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureMode {
    Condensed,
    RawVerbatim,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedMemoryItem {
    pub id: String,
    pub kind: String,
    pub text: String,
    pub status: String,
    pub source_digest_id: Option<DigestId>,
    pub context_label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionProvenance {
    pub item_id: String,
    pub digest_id: DigestId,
    pub relationship: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBasisDiff {
    pub field: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryInvalidationNotice {
    pub affected_decision_id: Option<crate::DecisionId>,
    pub decision_text: String,
    pub basis_line: String,
    pub basis_diff: Vec<MemoryBasisDiff>,
    pub threatened_intention_ids: Vec<crate::IntentionId>,
    pub served_goal: String,
    pub goal_now_unserved: bool,
    pub current_decision_text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureImpactSignal {
    DerivedGraph,
    VerbatimOnlyNoGraph,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupEvent {
    pub entity: String,
    pub prior_value: String,
    pub candidate_value: String,
    pub classification: DedupClassification,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureReport {
    pub mode: CaptureMode,
    pub last_harvest: HarvestCounts,
    pub total_harvest: HarvestCounts,
    pub message: String,
    pub digest_summary: String,
    pub verbatim_text: String,
    pub derived_items: Vec<DerivedMemoryItem>,
    pub provenance_edges: Vec<ExtractionProvenance>,
    pub invalidations: Vec<MemoryInvalidationNotice>,
    pub stable_ids: BTreeMap<String, String>,
    pub wrote_digest: bool,
    pub impact_signal: Option<CaptureImpactSignal>,
    pub dedup_events: Vec<DedupEvent>,
}

impl CaptureReport {
    fn wrong_empty() -> Self {
        Self {
            mode: CaptureMode::RawVerbatim,
            last_harvest: HarvestCounts::default(),
            total_harvest: HarvestCounts::default(),
            message: "verbatim mode: no structured items were captured".to_string(),
            digest_summary: "canned empty digest".to_string(),
            verbatim_text: String::new(),
            derived_items: Vec::new(),
            provenance_edges: Vec::new(),
            invalidations: Vec::new(),
            stable_ids: BTreeMap::new(),
            wrote_digest: false,
            impact_signal: None,
            dedup_events: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureRequest {
    pub context_label: String,
    pub transcript: String,
    pub model_available: bool,
}

pub fn record_conversation(_store: &Store, _request: CaptureRequest) -> Result<CaptureReport> {
    Ok(CaptureReport::wrong_empty())
}

pub fn import_conversation_file(_store: &Store, _path: &std::path::Path) -> Result<CaptureReport> {
    Ok(CaptureReport::wrong_empty())
}

pub fn dry_run_import(_store: &Store, _transcript: &str) -> Result<CaptureReport> {
    Ok(CaptureReport::wrong_empty())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupClassification {
    NewInsertion,
    ExactDuplicate,
    RefinesExisting,
    ConflictsExisting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupCandidate {
    pub entity: String,
    pub prior_value: String,
    pub candidate_value: String,
}

pub fn classify_dedup_candidate(_candidate: DedupCandidate) -> DedupClassification {
    DedupClassification::NewInsertion
}

pub trait ImportParser: Send + Sync {
    fn parse(&self, raw: &[u8]) -> std::result::Result<String, CgError>;
    fn format_name(&self) -> &'static str;
}

#[derive(Debug, Default)]
pub struct ClaudeJsonlParser;

impl ImportParser for ClaudeJsonlParser {
    fn parse(&self, raw: &[u8]) -> std::result::Result<String, CgError> {
        Ok(String::from_utf8_lossy(raw).into_owned())
    }

    fn format_name(&self) -> &'static str {
        "claude-jsonl"
    }
}

pub fn parser_for_format(format: &str) -> Result<Box<dyn ImportParser>> {
    if format == "claude-jsonl" {
        Ok(Box::new(ClaudeJsonlParser))
    } else {
        Err(CgError::UnsupportedImportFormat {
            format: format.to_string(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConnectedModelContext {
    advertised_context_tokens: usize,
}

fn advertised_context_window(model: ConnectedModelContext) -> usize {
    let _ = model.advertised_context_tokens;
    32_000
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeModelInfo {
    pub context_label: String,
    pub model_key: String,
    pub advertised_context_tokens: usize,
    pub default_for_context: bool,
}

pub fn register_runtime_model_info(_store: &Store, _info: RuntimeModelInfo) -> Result<()> {
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CondensationReport {
    pub pass_count: usize,
    pub max_prompt_tokens_seen: usize,
    pub discovered_item_ids: Vec<String>,
    pub stable_item_ids: BTreeMap<String, uuid::Uuid>,
    pub provenance_edges: Vec<ExtractionProvenance>,
    pub wrote_digest: bool,
    pub resolved_chunk_window: Option<usize>,
    pub model_context_tokens: Option<usize>,
}

fn rolling_condense(_transcript: &str, _window_tokens: usize) -> Result<CondensationReport> {
    Ok(CondensationReport {
        pass_count: 1,
        max_prompt_tokens_seen: 999_999,
        discovered_item_ids: vec!["late-tail-only".to_string()],
        stable_item_ids: BTreeMap::new(),
        provenance_edges: Vec::new(),
        wrote_digest: true,
        resolved_chunk_window: Some(4_096),
        model_context_tokens: Some(32_000),
    })
}

pub fn rolling_condense_into_store(
    _store: &Store,
    transcript: &str,
    window_tokens: usize,
) -> Result<CondensationReport> {
    rolling_condense(transcript, window_tokens)
}

#[derive(Debug, Clone, PartialEq)]
pub struct RollingIngestRequest {
    pub context_label: String,
    pub model_key: Option<String>,
    pub transcript: String,
    pub prompt_overhead_tokens: usize,
    pub output_budget_tokens: usize,
    pub running_state_cap_tokens: usize,
    pub explicit_chunk_tokens: Option<usize>,
    pub context_fraction: Option<f32>,
}

pub fn rolling_condense_for_connected_model(
    store: &Store,
    request: RollingIngestRequest,
) -> Result<CondensationReport> {
    let _ = (
        request.context_label,
        request.model_key,
        request.prompt_overhead_tokens,
        request.output_budget_tokens,
        request.running_state_cap_tokens,
        request.explicit_chunk_tokens,
        request.context_fraction,
    );
    let model_context = advertised_context_window(ConnectedModelContext {
        advertised_context_tokens: 32_000,
    });
    let chunk_window = derive_chunk_window(
        model_context,
        request.prompt_overhead_tokens,
        request.output_budget_tokens,
        request.running_state_cap_tokens,
        request.explicit_chunk_tokens,
        request.context_fraction,
    );
    let mut report = rolling_condense_into_store(store, &request.transcript, chunk_window)?;
    report.model_context_tokens = Some(model_context);
    report.resolved_chunk_window = Some(chunk_window);
    Ok(report)
}

fn derive_chunk_window(
    _model_context: usize,
    _prompt_overhead_tokens: usize,
    _output_budget: usize,
    _running_state_cap: usize,
    explicit_chunk_tokens: Option<usize>,
    _context_fraction: Option<f32>,
) -> usize {
    explicit_chunk_tokens.unwrap_or(4_096)
}

pub fn embed_digest_summary_for_connected_embedders(
    full_summary: &str,
    embedders: &[&dyn Embedder],
) -> Result<(String, Vec<EmbeddingOutput>)> {
    let summary = full_summary.to_string();
    let mut outputs = Vec::with_capacity(embedders.len());
    for embedder in embedders {
        outputs.push(embedder.embed(EmbeddingInput::Text(summary.clone()))?);
    }
    Ok((summary, outputs))
}

pub fn changed_since_last_record(
    _store: &Store,
    _context: ContextId,
) -> Result<Vec<MemoryInvalidationNotice>> {
    Ok(Vec::new())
}

pub fn standing_impact(
    _store: &Store,
    _context: ContextId,
) -> Result<Vec<MemoryInvalidationNotice>> {
    Ok(Vec::new())
}
