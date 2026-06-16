use chrono::Utc;
use contextdb_core::Value as DbValue;
use serde::{Deserialize, Deserializer};
use serde_json::{Value as JsonValue, json};
use std::collections::{BTreeMap, HashMap};

use crate::embedding_space::BuiltinSpaceRegistry;
use crate::{
    AgentId, CgError, ContextId, CreateContext, CreateDecision, CreateDigest, CreateEntity,
    CreateIntention, DecisionId, DigestId, Embedder, EmbeddingId, EmbeddingInput, EmbeddingOutput,
    EmbeddingTarget, EntityId, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef,
    IntentionId, IntentionOrigin, IntentionStatus, InvalidationStatus, ObservationId,
    PrecomputedEmbedding, RecordObservation, Result, Store,
};

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
    pub affected_decision_id: Option<DecisionId>,
    pub decision_text: String,
    pub basis_line: String,
    pub basis_diff: Vec<MemoryBasisDiff>,
    pub threatened_intention_ids: Vec<IntentionId>,
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
    fn empty(mode: CaptureMode, message: impl Into<String>) -> Self {
        Self {
            mode,
            last_harvest: HarvestCounts::default(),
            total_harvest: HarvestCounts::default(),
            message: message.into(),
            digest_summary: String::new(),
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

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct StructuredHarvest {
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub goals: Vec<GoalSeed>,
    #[serde(default)]
    pub entities: Vec<EntitySeed>,
    #[serde(default)]
    pub decisions: Vec<DecisionSeed>,
    #[serde(default)]
    pub observations: Vec<ObservationSeed>,
}

impl StructuredHarvest {
    pub fn from_json_str(raw: &str) -> Result<Self> {
        serde_json::from_str(raw).map_err(|err| {
            CgError::Engine(format!(
                "extractor returned invalid structured harvest JSON: {err}"
            ))
        })
    }

    fn is_empty(&self) -> bool {
        self.goals.is_empty()
            && self.entities.is_empty()
            && self.decisions.is_empty()
            && self.observations.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GoalSeed {
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub key: String,
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct EntitySeed {
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub key: String,
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub name: String,
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub path: String,
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub entity_type: ExtractedEntityType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractedEntityType {
    Service,
    Configuration,
    #[default]
    File,
    Pipeline,
    Document,
}

fn deserialize_default_on_null<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

fn deserialize_string_default_on_null<'de, D>(
    deserializer: D,
) -> std::result::Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

impl From<ExtractedEntityType> for EntityType {
    fn from(value: ExtractedEntityType) -> Self {
        match value {
            ExtractedEntityType::Service => EntityType::Service,
            ExtractedEntityType::Configuration => EntityType::Configuration,
            ExtractedEntityType::File => EntityType::File,
            ExtractedEntityType::Pipeline => EntityType::Pipeline,
            ExtractedEntityType::Document => EntityType::Document,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct DecisionSeed {
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub key: String,
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub text: String,
    #[serde(default)]
    pub goal_key: Option<String>,
    #[serde(default)]
    pub entity_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ObservationSeed {
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub key: String,
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub text: String,
    #[serde(default)]
    pub entity_key: Option<String>,
    #[serde(default)]
    pub path_change: Option<String>,
}

pub struct ExtractionRequest<'a> {
    pub context_label: &'a str,
    pub transcript: &'a str,
}

pub trait ConversationExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> Result<StructuredHarvest>;
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
    pub same_entity: bool,
    pub prior_value: String,
    pub candidate_value: String,
}

pub fn classify_dedup_candidate(candidate: DedupCandidate) -> DedupClassification {
    if !candidate.same_entity {
        return DedupClassification::NewInsertion;
    }
    let prior = normalized_tracked_value(&candidate.prior_value);
    let next = normalized_tracked_value(&candidate.candidate_value);

    if prior.is_empty() {
        return DedupClassification::NewInsertion;
    }
    if next.is_empty() {
        return DedupClassification::NewInsertion;
    }
    if prior == next {
        return DedupClassification::ExactDuplicate;
    }
    if is_strict_refinement(&prior, &next) {
        return DedupClassification::RefinesExisting;
    }
    DedupClassification::ConflictsExisting
}

fn normalized_tracked_value(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_strict_refinement(prior: &str, next: &str) -> bool {
    next.len() > prior.len() && next.contains(prior)
}

pub trait ImportParser: Send + Sync {
    fn parse(&self, raw: &[u8]) -> std::result::Result<String, CgError>;
    fn parse_line(&self, line: &str) -> std::result::Result<Option<String>, CgError> {
        let parsed = self.parse(line.as_bytes())?;
        Ok((!parsed.trim().is_empty()).then_some(parsed))
    }
    fn format_name(&self) -> &'static str;
}

#[derive(Debug, Default)]
pub struct ClaudeJsonlParser;

impl ImportParser for ClaudeJsonlParser {
    fn parse(&self, raw: &[u8]) -> std::result::Result<String, CgError> {
        let input = String::from_utf8_lossy(raw);
        let mut out = String::new();
        for line in input.lines() {
            if let Some(parsed) = self.parse_line(line)? {
                out.push_str(&parsed);
            }
        }
        Ok(out)
    }

    fn parse_line(&self, line: &str) -> std::result::Result<Option<String>, CgError> {
        Ok(parse_claude_jsonl_line(line))
    }

    fn format_name(&self) -> &'static str {
        "claude-jsonl"
    }
}

fn parse_claude_jsonl_line(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let Ok(value) = serde_json::from_str::<JsonValue>(line) else {
        return None;
    };
    let message = value.get("message").and_then(JsonValue::as_object)?;
    let role = message
        .get("role")
        .and_then(JsonValue::as_str)
        .or_else(|| {
            value
                .get("type")
                .and_then(JsonValue::as_str)
                .filter(|role| *role == "user" || *role == "assistant")
        })?;
    if role != "user" && role != "assistant" {
        return None;
    }
    let mut text = block_text(message.get("content"));
    text = strip_embedded_noise(&text).trim().to_string();
    if is_noise(&text) {
        return None;
    }
    let label = if role == "user" { "User" } else { "Assistant" };
    Some(format!("{label}: {text}\n"))
}

pub fn parser_for_format(format: &str) -> Result<Box<dyn ImportParser>> {
    match format {
        "claude-jsonl" | "claude" | "jsonl" => Ok(Box::new(ClaudeJsonlParser)),
        other => Err(CgError::UnsupportedImportFormat {
            format: other.to_string(),
        }),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeModelInfo {
    pub context_label: String,
    pub model_key: String,
    pub advertised_context_tokens: usize,
    pub default_for_context: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CondensationReport {
    pub pass_count: usize,
    pub max_prompt_tokens_seen: usize,
    pub discovered_item_ids: Vec<String>,
    pub stable_item_ids: StableItemIds,
    pub provenance_edges: Vec<ExtractionProvenance>,
    pub wrote_digest: bool,
    pub resolved_chunk_window: Option<usize>,
    pub model_context_tokens: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StableItemIds {
    entries: Vec<(String, uuid::Uuid)>,
}

impl StableItemIds {
    pub fn insert(&mut self, key: String, value: uuid::Uuid) {
        if let Some((_, existing)) = self
            .entries
            .iter_mut()
            .find(|(existing, _)| existing == &key)
        {
            *existing = value;
        } else {
            self.entries.push((key, value));
        }
    }

    pub fn get(&self, key: &str) -> Option<&uuid::Uuid> {
        self.entries
            .iter()
            .find_map(|(existing, value)| (existing == key).then_some(value))
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.iter().map(|(key, _)| key)
    }
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

pub fn register_runtime_model_info(store: &Store, info: RuntimeModelInfo) -> Result<()> {
    ensure_runtime_model_table(store)?;
    let db = store.internal_db();
    let tx = db.begin().map_err(map_engine)?;
    db.execute_in_tx(
        tx,
        "INSERT INTO cg_runtime_models (id, context_label, model_key, advertised_context_tokens, default_for_context, created_at) VALUES ($id, $context_label, $model_key, $tokens, $default, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(uuid::Uuid::new_v4())),
            ("context_label".to_string(), DbValue::Text(info.context_label)),
            ("model_key".to_string(), DbValue::Text(info.model_key)),
            ("tokens".to_string(), DbValue::Int64(info.advertised_context_tokens as i64)),
            ("default".to_string(), DbValue::Bool(info.default_for_context)),
            ("created_at".to_string(), DbValue::TxId(tx)),
        ]),
    )
    .map_err(|err| {
        let _ = db.rollback(tx);
        map_engine(err)
    })?;
    db.commit(tx).map_err(map_engine)
}

pub fn record_conversation(store: &Store, request: CaptureRequest) -> Result<CaptureReport> {
    record_conversation_internal(store, request, None)
}

pub fn record_conversation_with_extractor(
    store: &Store,
    request: CaptureRequest,
    extractor: &dyn ConversationExtractor,
) -> Result<CaptureReport> {
    record_conversation_internal(store, request, Some(extractor))
}

fn record_conversation_internal(
    store: &Store,
    request: CaptureRequest,
    extractor: Option<&dyn ConversationExtractor>,
) -> Result<CaptureReport> {
    let raw_text = request.transcript.as_str();
    let text = raw_text.trim();
    if text.is_empty() {
        return Ok(CaptureReport::empty(
            CaptureMode::Condensed,
            "nothing to capture: empty import",
        ));
    }

    let context = context_by_label(store, &request.context_label)?;
    let Some(extractor) = extractor else {
        return capture_raw(store, context.id, &request.context_label, text);
    };
    if !request.model_available {
        return capture_raw(store, context.id, &request.context_label, text);
    }
    let harvest = extractor.extract(ExtractionRequest {
        context_label: &request.context_label,
        transcript: text,
    })?;
    capture_structured_harvest(store, context.id, &request.context_label, text, harvest)
}

pub fn import_conversation_text(store: &Store, text: &str) -> Result<CaptureReport> {
    import_conversation_text_internal(store, text, None)
}

pub fn import_conversation_text_with_extractor(
    store: &Store,
    text: &str,
    extractor: &dyn ConversationExtractor,
) -> Result<CaptureReport> {
    import_conversation_text_internal(store, text, Some(extractor))
}

fn import_conversation_text_internal(
    store: &Store,
    text: &str,
    extractor: Option<&dyn ConversationExtractor>,
) -> Result<CaptureReport> {
    let context_label = text
        .lines()
        .find_map(|line| line.strip_prefix("Context:").map(str::trim))
        .filter(|label| !label.is_empty())
        .unwrap_or("acme-coding-assistant")
        .to_string();
    let transcript = text
        .lines()
        .filter(|line| !line.trim_start().starts_with("Context:"))
        .collect::<Vec<_>>()
        .join("\n");
    record_conversation_internal(
        store,
        CaptureRequest {
            context_label,
            transcript,
            model_available: extractor.is_some(),
        },
        extractor,
    )
}

pub fn dry_run_import(_store: &Store, _transcript: &str) -> Result<CaptureReport> {
    Ok(CaptureReport::empty(
        CaptureMode::Condensed,
        "dry run: no writes",
    ))
}

pub fn rolling_condense_into_store(
    store: &Store,
    transcript: &str,
    window_tokens: usize,
) -> Result<CondensationReport> {
    let context = context_by_label(store, "rolling-ingest")?;
    let digest = create_text_digest(
        store,
        context.id,
        "rolled long-session digest",
        &transcript.chars().take(2048).collect::<String>(),
    )?;
    let mut stable = StableItemIds::default();
    let mut discovered = Vec::new();
    let mut provenance = Vec::new();
    let labeled_items = extract_labeled_rolling_items(transcript);
    let mut first_goal = None::<IntentionId>;
    let mut observation_entity = None::<EntityId>;

    for item in labeled_items {
        let source_id = match item.kind {
            RollingItemKind::Goal => {
                let goal = ensure_intention(store, context.id, &item.text)?;
                first_goal.get_or_insert(goal);
                goal.as_uuid()
            }
            RollingItemKind::Decision => {
                let goal_ids = first_goal.iter().copied().collect::<Vec<_>>();
                ensure_decision(store, context.id, &item.text, &goal_ids, None)?.as_uuid()
            }
            RollingItemKind::Observation => {
                let entity = match observation_entity {
                    Some(entity) => entity,
                    None => {
                        let entity = ensure_entity(
                            store,
                            context.id,
                            "rolling session",
                            "rolling-session",
                            EntityType::Document,
                        )?;
                        observation_entity = Some(entity);
                        entity
                    }
                };
                ensure_observation(store, context.id, entity, &item.text, None)?.as_uuid()
            }
        };
        stable.insert(item.key.clone(), source_id);
        discovered.push(item.key.clone());
        store.insert_extracted_from_edge(source_id, digest.id, context.id)?;
        provenance.push(ExtractionProvenance {
            item_id: item.key,
            digest_id: digest.id,
            relationship: "EXTRACTED_FROM".to_string(),
        });
    }
    let chunk_window = window_tokens.max(1);
    let pass_count = if transcript.len() > chunk_window {
        transcript.len().div_ceil(chunk_window)
    } else {
        1
    };
    Ok(CondensationReport {
        pass_count,
        max_prompt_tokens_seen: chunk_window.min(transcript.len().max(1)),
        discovered_item_ids: discovered,
        stable_item_ids: stable,
        provenance_edges: provenance,
        wrote_digest: true,
        resolved_chunk_window: Some(chunk_window),
        model_context_tokens: None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RollingItemKind {
    Goal,
    Decision,
    Observation,
}

impl RollingItemKind {
    fn label(self) -> &'static str {
        match self {
            Self::Goal => "Goal",
            Self::Decision => "Decision",
            Self::Observation => "Observation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RollingItemSeed {
    kind: RollingItemKind,
    key: String,
    text: String,
}

fn extract_labeled_rolling_items(transcript: &str) -> Vec<RollingItemSeed> {
    let mut markers = Vec::new();
    for (needle, kind) in [
        ("Goal ", RollingItemKind::Goal),
        ("Decision ", RollingItemKind::Decision),
        ("Observation ", RollingItemKind::Observation),
    ] {
        for (index, _) in transcript.match_indices(needle) {
            markers.push((index, needle.len(), kind));
        }
    }
    markers.sort_by_key(|(index, _, _)| *index);

    markers
        .iter()
        .enumerate()
        .filter_map(|(position, (index, marker_len, kind))| {
            let start = index + marker_len;
            let end = markers
                .get(position + 1)
                .map(|(next, _, _)| *next)
                .unwrap_or(transcript.len());
            parse_rolling_item(*kind, &transcript[start..end])
        })
        .collect()
}

fn parse_rolling_item(kind: RollingItemKind, segment: &str) -> Option<RollingItemSeed> {
    let first_line = segment.lines().next()?.trim();
    let item = first_line.trim_end_matches(['.', ';']).trim();
    let (key, body) = item.split_once(':')?;
    let key = key.trim();
    let body = body.trim();
    if key.is_empty() || body.is_empty() || key.chars().any(char::is_whitespace) {
        return None;
    }
    Some(RollingItemSeed {
        kind,
        key: key.to_string(),
        text: format!("{} {}: {}", kind.label(), key, body),
    })
}

pub fn rolling_condense_for_connected_model(
    store: &Store,
    request: RollingIngestRequest,
) -> Result<CondensationReport> {
    let model_context =
        runtime_model_context(store, &request.context_label, request.model_key.as_deref())?;
    let chunk_window = derive_chunk_window(
        model_context,
        request.prompt_overhead_tokens,
        request.output_budget_tokens,
        request.running_state_cap_tokens,
        request.explicit_chunk_tokens,
        request.context_fraction,
    )?;
    let mut report = rolling_condense_into_store(store, &request.transcript, chunk_window)?;
    report.model_context_tokens = Some(model_context);
    report.resolved_chunk_window = Some(chunk_window);
    Ok(report)
}

pub fn embed_digest_summary_for_connected_embedders(
    full_summary: &str,
    embedders: &[&dyn Embedder],
) -> Result<(String, Vec<EmbeddingOutput>)> {
    let bound = embedders
        .iter()
        .map(|embedder| embedder.max_sequence_length())
        .min()
        .unwrap_or(full_summary.len());
    let summary = full_summary.chars().take(bound).collect::<String>();
    let mut outputs = Vec::with_capacity(embedders.len());
    for embedder in embedders {
        outputs.push(embedder.embed(EmbeddingInput::Text(summary.clone()))?);
    }
    Ok((summary, outputs))
}

pub fn changed_since_last_record(
    store: &Store,
    context: ContextId,
) -> Result<Vec<MemoryInvalidationNotice>> {
    standing_impact(store, context)
}

pub fn changed_since_context_cursor(
    store: &Store,
    context: ContextId,
) -> Result<Vec<MemoryInvalidationNotice>> {
    ensure_changed_cursor_table(store)?;
    let since = read_changed_cursor_tx(store, context)?;
    let pending = store.list_invalidations(context, None, Some(InvalidationStatus::Pending))?;
    let changed_rows = pending
        .iter()
        .filter(|row| row.detected_at.0 > since.0)
        .cloned()
        .collect::<Vec<_>>();
    let max_seen = pending
        .iter()
        .map(|row| row.detected_at.0)
        .max()
        .map(crate::TxId)
        .unwrap_or(since);
    write_changed_cursor_tx(store, context, max_seen)?;
    changed_rows
        .into_iter()
        .map(|row| invalidation_notice(store, &row))
        .collect()
}

pub fn standing_impact(store: &Store, context: ContextId) -> Result<Vec<MemoryInvalidationNotice>> {
    let invalidations =
        store.list_invalidations(context, None, Some(InvalidationStatus::Pending))?;
    invalidations
        .into_iter()
        .map(|row| invalidation_notice(store, &row))
        .collect()
}

pub fn digest_source_ids_for_context(store: &Store, context: ContextId) -> Result<Vec<uuid::Uuid>> {
    store.digest_source_ids_for_context(context)
}

pub fn memory_counts(store: &Store, context_id: ContextId) -> Result<HarvestCounts> {
    total_counts(store, context_id)
}

fn capture_raw(
    store: &Store,
    context_id: ContextId,
    _context_label: &str,
    text: &str,
) -> Result<CaptureReport> {
    let digest = create_text_digest(store, context_id, text, text)?;
    Ok(CaptureReport {
        mode: CaptureMode::RawVerbatim,
        last_harvest: HarvestCounts::default(),
        total_harvest: total_counts(store, context_id)?,
        message: "model-free capture: stored verbatim text; no derived graph or impact analysis"
            .to_string(),
        digest_summary: digest.summary,
        verbatim_text: text.to_string(),
        derived_items: Vec::new(),
        provenance_edges: Vec::new(),
        invalidations: Vec::new(),
        stable_ids: BTreeMap::new(),
        wrote_digest: true,
        impact_signal: Some(CaptureImpactSignal::VerbatimOnlyNoGraph),
        dedup_events: Vec::new(),
    })
}

fn capture_structured_harvest(
    store: &Store,
    context_id: ContextId,
    context_label: &str,
    transcript: &str,
    harvest: StructuredHarvest,
) -> Result<CaptureReport> {
    if harvest.is_empty() {
        return capture_zero_harvest_digest(store, context_id, transcript);
    }

    let summary = harvest
        .summary
        .as_deref()
        .map(str::trim)
        .filter(|summary| !summary.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| compact_summary(transcript));
    let digest = create_text_digest(store, context_id, &summary, &summary)?;
    let mut stable = BTreeMap::new();
    let mut source_ids = Vec::<(String, uuid::Uuid)>::new();
    let mut derived_items = Vec::new();
    let mut goals = BTreeMap::<String, IntentionId>::new();
    for seed in &harvest.goals {
        if seed.key.trim().is_empty() || seed.text.trim().is_empty() {
            continue;
        }
        let goal = ensure_intention(store, context_id, &seed.text)?;
        stable_id(&mut stable, &seed.key, goal);
        stable_id_text(&mut stable, &seed.text, goal.as_uuid());
        source_ids.push((seed.key.clone(), goal.as_uuid()));
        derived_items.push(derived(
            &seed.key,
            "intention",
            &seed.text,
            "active",
            digest.id,
            context_label,
        ));
        goals.insert(seed.key.clone(), goal);
    }

    let mut entities = BTreeMap::<String, EntityId>::new();
    let mut entity_paths = BTreeMap::<String, String>::new();
    let mut entity_names = BTreeMap::<String, String>::new();
    for seed in &harvest.entities {
        if seed.key.trim().is_empty() || seed.name.trim().is_empty() {
            continue;
        }
        let path = if seed.path.trim().is_empty() {
            seed.name.clone()
        } else {
            seed.path.clone()
        };
        let entity = ensure_entity(
            store,
            context_id,
            &seed.name,
            &path,
            seed.entity_type.into(),
        )?;
        stable_id(&mut stable, &seed.key, entity);
        stable_id_text(&mut stable, &seed.name, entity.as_uuid());
        stable_id_text(&mut stable, &path, entity.as_uuid());
        source_ids.push((seed.key.clone(), entity.as_uuid()));
        entities.insert(seed.key.clone(), entity);
        entity_paths.insert(seed.key.clone(), path);
        entity_names.insert(seed.key.clone(), seed.name.clone());
    }

    let first_goal = goals.values().next().copied();
    let mut decision_items = Vec::<(String, DecisionId, String)>::new();
    let fallback_entity = entities.values().next().copied();
    let mut invalidation_rows = Vec::new();
    let mut dedup_events = Vec::new();
    let mut observation_items = Vec::new();
    let mut observation_source_ids = Vec::<(String, uuid::Uuid)>::new();
    for seed in &harvest.observations {
        if seed.key.trim().is_empty() || seed.text.trim().is_empty() {
            continue;
        }
        let Some(entity) = seed
            .entity_key
            .as_ref()
            .and_then(|key| entities.get(key).copied())
            .or(fallback_entity)
        else {
            continue;
        };
        let prior_path = seed
            .entity_key
            .as_ref()
            .and_then(|key| entity_paths.get(key))
            .cloned();
        let path_change = seed
            .path_change
            .as_deref()
            .map(normalize_path_change_value)
            .filter(|path| !path.is_empty());
        let write = ensure_observation_with_invalidations(
            store,
            context_id,
            entity,
            &seed.text,
            path_change.as_deref().map(|path| ("path", path)),
        )?;
        stable_id(&mut stable, &seed.key, write.id);
        stable_id_text(&mut stable, &seed.text, write.id.as_uuid());
        observation_source_ids.push((seed.key.clone(), write.id.as_uuid()));
        observation_items.push(derived(
            &seed.key,
            "observation",
            &seed.text,
            "current",
            digest.id,
            context_label,
        ));
        if let (Some(entity_key), Some(prior_value), Some(candidate_value)) =
            (seed.entity_key.as_ref(), prior_path, path_change.as_ref())
        {
            let entity_name = entity_names
                .get(entity_key)
                .cloned()
                .unwrap_or_else(|| entity_key.clone());
            let classification = classify_dedup_candidate(DedupCandidate {
                entity: entity_name.clone(),
                same_entity: true,
                prior_value: prior_value.clone(),
                candidate_value: candidate_value.clone(),
            });
            dedup_events.push(DedupEvent {
                entity: entity_name.clone(),
                prior_value: prior_value.clone(),
                candidate_value: candidate_value.clone(),
                classification,
            });
            if classification == DedupClassification::ConflictsExisting {
                for decision_id in active_decision_ids_for_entity_basis(store, context_id, entity)?
                {
                    store.invalidate_decision(
                        decision_id,
                        write.id,
                        format!(
                            "extracted observation conflicts with decision basis for {entity_name}: {prior_value} -> {candidate_value}"
                        ),
                    )?;
                    invalidation_rows.extend(store.list_invalidations(
                        context_id,
                        Some(decision_id),
                        None,
                    )?);
                }
            }
        }
        invalidation_rows.extend(write.invalidations);
    }

    for seed in &harvest.decisions {
        if seed.key.trim().is_empty() || seed.text.trim().is_empty() {
            continue;
        }
        let Some(goal) = seed
            .goal_key
            .as_ref()
            .and_then(|key| goals.get(key).copied())
            .or(first_goal)
        else {
            continue;
        };
        let entity = seed
            .entity_key
            .as_ref()
            .and_then(|key| entities.get(key).copied());
        let decision = ensure_decision(store, context_id, &seed.text, &[goal], entity)?;
        stable_id(&mut stable, &seed.key, decision);
        stable_id_text(&mut stable, &seed.text, decision.as_uuid());
        source_ids.push((seed.key.clone(), decision.as_uuid()));
        decision_items.push((seed.key.clone(), decision, seed.text.clone()));
    }

    for (key, decision, text) in &decision_items {
        let status = store
            .get_decision(*decision)?
            .map(|row| decision_status_label(row.status))
            .unwrap_or("current");
        derived_items.push(derived(
            key,
            "decision",
            text,
            status,
            digest.id,
            context_label,
        ));
    }
    let invalidations = invalidation_rows
        .iter()
        .map(|row| invalidation_notice(store, row))
        .collect::<Result<Vec<_>>>()?;
    derived_items.extend(observation_items);
    source_ids.extend(observation_source_ids);

    let source_refs = source_ids
        .iter()
        .map(|(key, id)| (key.as_str(), *id))
        .collect::<Vec<_>>();
    let provenance = add_provenance_edges(store, context_id, digest.id, &source_refs)?;
    let last_harvest = HarvestCounts {
        goals: harvest.goals.len() as u32,
        decisions: harvest.decisions.len() as u32,
        observations: harvest.observations.len() as u32,
    };
    Ok(report_with_counts(
        CaptureMode::Condensed,
        last_harvest,
        total_counts(store, context_id)?,
        "captured structured memory",
        summary,
        report_artifacts(
            derived_items,
            provenance,
            stable,
            dedup_events,
            invalidations,
        ),
    ))
}

fn decision_status_label(status: crate::DecisionStatus) -> &'static str {
    match status {
        crate::DecisionStatus::Active => "current",
        crate::DecisionStatus::Invalidated => "invalidated",
        crate::DecisionStatus::Superseded => "superseded",
        crate::DecisionStatus::Archived => "archived",
    }
}

fn capture_zero_harvest_digest(
    store: &Store,
    context_id: ContextId,
    transcript: &str,
) -> Result<CaptureReport> {
    let digest = create_text_digest(store, context_id, transcript, transcript)?;
    let mut report = CaptureReport::empty(
        CaptureMode::Condensed,
        "model completed: zero structured items extracted from this capture",
    );
    report.digest_summary = digest.summary;
    report.wrote_digest = true;
    report.total_harvest = total_counts(store, context_id)?;
    Ok(report)
}

struct ReportArtifacts {
    derived_items: Vec<DerivedMemoryItem>,
    provenance_edges: Vec<ExtractionProvenance>,
    stable_ids: BTreeMap<String, String>,
    dedup_events: Vec<DedupEvent>,
    invalidations: Vec<MemoryInvalidationNotice>,
}

fn report_artifacts(
    derived_items: Vec<DerivedMemoryItem>,
    provenance_edges: Vec<ExtractionProvenance>,
    stable_ids: BTreeMap<String, String>,
    dedup_events: Vec<DedupEvent>,
    invalidations: Vec<MemoryInvalidationNotice>,
) -> ReportArtifacts {
    ReportArtifacts {
        derived_items,
        provenance_edges,
        stable_ids,
        dedup_events,
        invalidations,
    }
}

fn report_with_counts(
    mode: CaptureMode,
    last_harvest: HarvestCounts,
    total_harvest: HarvestCounts,
    message: impl Into<String>,
    digest_summary: String,
    artifacts: ReportArtifacts,
) -> CaptureReport {
    let ReportArtifacts {
        derived_items,
        provenance_edges,
        stable_ids,
        dedup_events,
        invalidations,
    } = artifacts;
    CaptureReport {
        mode,
        last_harvest,
        total_harvest,
        message: message.into(),
        digest_summary,
        verbatim_text: String::new(),
        derived_items,
        provenance_edges,
        invalidations,
        stable_ids,
        wrote_digest: true,
        impact_signal: Some(CaptureImpactSignal::DerivedGraph),
        dedup_events,
    }
}

fn block_text(content: Option<&JsonValue>) -> String {
    match content {
        Some(JsonValue::String(text)) => text.clone(),
        Some(JsonValue::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| {
                (block.get("type").and_then(JsonValue::as_str) == Some("text"))
                    .then(|| block.get("text").and_then(JsonValue::as_str))
                    .flatten()
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn strip_embedded_noise(text: &str) -> String {
    let without_system = strip_tag_blocks(text, "<system-reminder>", "</system-reminder>");
    let without_teammate =
        strip_prefixed_tag_blocks(&without_system, "<teammate-message", "</teammate-message>");
    strip_local_command_blocks(&without_teammate)
}

fn strip_tag_blocks(text: &str, open: &str, close: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        out.push_str(&rest[..start]);
        let after = &rest[start + open.len()..];
        if let Some(end) = after.find(close) {
            rest = &after[end + close.len()..];
        } else {
            rest = "";
        }
    }
    out.push_str(rest);
    out
}

fn strip_prefixed_tag_blocks(text: &str, open_prefix: &str, close: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(open_prefix) {
        out.push_str(&rest[..start]);
        let tag = &rest[start..];
        let Some(open_end) = tag.find('>') else {
            break;
        };
        let after_open = &tag[open_end + 1..];
        if let Some(close_start) = after_open.find(close) {
            rest = &after_open[close_start + close.len()..];
        } else {
            rest = "";
        }
    }
    out.push_str(rest);
    out
}

fn strip_local_command_blocks(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<local-command-") {
        out.push_str(&rest[..start]);
        let tag = &rest[start..];
        let Some(open_end) = tag.find('>') else {
            break;
        };
        let open_tag = &tag[..=open_end];
        let suffix = open_tag
            .trim_start_matches("<local-command-")
            .trim_end_matches('>');
        let close = format!("</local-command-{suffix}>");
        let after_open = &tag[open_end + 1..];
        if let Some(close_start) = after_open.find(&close) {
            rest = &after_open[close_start + close.len()..];
        } else {
            rest = "";
        }
    }
    out.push_str(rest);
    out
}

fn is_noise(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.is_empty()
        || trimmed.starts_with("<local-command-")
        || trimmed.starts_with("<command-name>")
        || trimmed.starts_with("<command-message>")
        || trimmed.starts_with("<command-args>")
        || trimmed.starts_with("<command-stdout>")
        || trimmed.starts_with("<system-reminder>")
        || trimmed.starts_with("Caveat:")
        || (trimmed.lines().count() == 1 && trimmed.starts_with('<') && trimmed.ends_with('>'))
}

fn context_by_label(store: &Store, label: &str) -> Result<crate::Context> {
    if let Some(existing) = store
        .list_contexts()?
        .into_iter()
        .find(|ctx| ctx.name == label)
    {
        return Ok(existing);
    }
    store.create_context(CreateContext {
        name: label.to_string(),
        ..Default::default()
    })
}

fn create_text_digest(
    store: &Store,
    context_id: ContextId,
    summary: &str,
    vector_text: &str,
) -> Result<crate::Digest> {
    let id = DigestId::new_v7();
    let vector = store.embed_digest_text(vector_text)?;
    store.create_digest(CreateDigest {
        id,
        source: "claude-cli".to_string(),
        session_ref: Some("capture".to_string()),
        summary: summary.to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: BuiltinSpaceRegistry::TEXT_DEFAULT_ID.to_string(),
            vector,
            target: EmbeddingTarget::Digest(id),
            produced_by: EvidenceProducer::default(),
        },
        context_id,
        properties: BTreeMap::new(),
    })
}

fn ensure_intention(
    store: &Store,
    context_id: ContextId,
    description: &str,
) -> Result<IntentionId> {
    if let Some(id) = find_id_by_text(store, "intentions", "description", description, context_id)?
    {
        return Ok(IntentionId::from(id));
    }
    if let Some(id) = find_intention_by_normalized_description(store, context_id, description)? {
        return Ok(IntentionId::from(id));
    }
    Ok(store
        .create_intention(CreateIntention {
            description: description.to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            ..Default::default()
        })?
        .id)
}

fn find_intention_by_normalized_description(
    store: &Store,
    context_id: ContextId,
    description: &str,
) -> Result<Option<uuid::Uuid>> {
    let target = normalize_intention_description(description);
    if target.is_empty() {
        return Ok(None);
    }
    let result = store
        .internal_db()
        .execute(
            "SELECT id, description FROM intentions WHERE context_id = $context_id",
            &HashMap::from([(
                "context_id".to_string(),
                DbValue::Uuid(context_id.as_uuid()),
            )]),
        )
        .map_err(map_engine)?;
    Ok(result.rows.iter().find_map(|row| match row.as_slice() {
        [DbValue::Uuid(id), DbValue::Text(existing)]
            if normalize_intention_description(existing) == target =>
        {
            Some(*id)
        }
        _ => None,
    }))
}

fn normalize_intention_description(description: &str) -> String {
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in description.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            push_intention_word(&mut words, &mut current);
        }
    }
    if !current.is_empty() {
        push_intention_word(&mut words, &mut current);
    }
    words.join(" ")
}

fn push_intention_word(words: &mut Vec<String>, current: &mut String) {
    if !matches!(current.as_str(), "a" | "an" | "the") {
        words.push(std::mem::take(current));
    } else {
        current.clear();
    }
}

fn normalize_path_change_value(value: &str) -> String {
    value
        .rsplit_once("->")
        .map(|(_, right)| right)
        .unwrap_or(value)
        .trim()
        .to_string()
}

fn ensure_entity(
    store: &Store,
    context_id: ContextId,
    name: &str,
    path: &str,
    entity_type: EntityType,
) -> Result<EntityId> {
    if let Some(entity) = store
        .list_entities(crate::ListEntityFilter {
            entity_type: Some(entity_type),
            name_contains: Some(name.to_string()),
            include_archived: false,
            context_id: Some(context_id),
            ..Default::default()
        })?
        .into_iter()
        .find(|entity| entity.name == name)
    {
        return Ok(entity.id);
    }
    Ok(store
        .create_entity(CreateEntity {
            entity_type,
            name: name.to_string(),
            properties: BTreeMap::from([("path".to_string(), json!(path))]),
            context_id,
            ..Default::default()
        })?
        .id)
}

fn ensure_decision(
    store: &Store,
    context_id: ContextId,
    description: &str,
    intention_ids: &[IntentionId],
    entity_id: Option<EntityId>,
) -> Result<DecisionId> {
    if let Some(id) = find_id_by_text(store, "decisions", "description", description, context_id)? {
        return Ok(DecisionId::from(id));
    }
    Ok(store
        .create_decision(CreateDecision {
            decision_type: "code_reference".to_string(),
            description: description.to_string(),
            confidence: Some(0.9),
            intention_ids: intention_ids.to_vec(),
            based_on_entity_ids: entity_id.into_iter().collect(),
            basis_fields: entity_id.map(|id| HashMap::from([(id, vec!["path".to_string()])])),
            context_id,
            agent_id: Some(AgentId::from("cg-ingest")),
            ..Default::default()
        })?
        .id)
}

fn ensure_observation(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    text: &str,
    state_delta: Option<(&str, &str)>,
) -> Result<ObservationId> {
    Ok(ensure_observation_with_invalidations(store, context_id, entity_id, text, state_delta)?.id)
}

struct ObservationWrite {
    id: ObservationId,
    invalidations: Vec<crate::Invalidation>,
}

fn ensure_observation_with_invalidations(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    text: &str,
    state_delta: Option<(&str, &str)>,
) -> Result<ObservationWrite> {
    if let Some(id) = find_id_by_text(store, "observations", "observation_type", text, context_id)?
    {
        return Ok(ObservationWrite {
            id: ObservationId::from(id),
            invalidations: Vec::new(),
        });
    }
    let id = ObservationId::new_v7();
    let result = store.record_observation(RecordObservation {
        id,
        entity_id,
        context_id,
        observation_type: text.to_string(),
        source: "agent-session".to_string(),
        observed_at: Utc::now(),
        evidence: vec![text_evidence(id, context_id)],
        observed_properties: BTreeMap::new(),
        state_delta: state_delta
            .map(|(field, value)| BTreeMap::from([(field.to_string(), Some(json!(value)))]))
            .unwrap_or_default(),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    })?;
    Ok(ObservationWrite {
        id: result.observation_id,
        invalidations: result.invalidations,
    })
}

fn text_evidence(observation_id: ObservationId, context_id: ContextId) -> EvidenceRef {
    EvidenceRef {
        id: EvidenceId::new_v7(),
        observation_id,
        context_id,
        kind: EvidenceKind::Text,
        source_ref: format!("cas://sha256:{}", "a".repeat(64)),
        content_hash: Some(format!("sha256:{}", "a".repeat(64))),
        captured_at: Some(Utc::now()),
        producer: EvidenceProducer::default(),
        retention_status: crate::RetentionStatus::RetainedExternal,
        ..Default::default()
    }
}

fn find_id_by_text(
    store: &Store,
    table: &str,
    column: &str,
    text: &str,
    context_id: ContextId,
) -> Result<Option<uuid::Uuid>> {
    let db = store.internal_db();
    let result = db
        .execute(
            &format!(
                "SELECT id FROM {table} WHERE context_id = $context_id AND {column} = $text LIMIT 1"
            ),
            &HashMap::from([
                (
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                ),
                ("text".to_string(), DbValue::Text(text.to_string())),
            ]),
        )
        .map_err(map_engine)?;
    Ok(result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::Uuid(id)) => Some(*id),
        _ => None,
    }))
}

fn add_provenance_edges(
    store: &Store,
    context_id: ContextId,
    digest_id: DigestId,
    sources: &[(&str, uuid::Uuid)],
) -> Result<Vec<ExtractionProvenance>> {
    let mut out = Vec::new();
    for (item_id, source_id) in sources {
        store.insert_extracted_from_edge(*source_id, digest_id, context_id)?;
        out.push(ExtractionProvenance {
            item_id: (*item_id).to_string(),
            digest_id,
            relationship: "EXTRACTED_FROM".to_string(),
        });
    }
    Ok(out)
}

fn active_decision_ids_for_entity_basis(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
) -> Result<Vec<DecisionId>> {
    let result = store
        .internal_db()
        .execute(
            "SELECT DISTINCT decisions.id
             FROM decisions
             INNER JOIN edges ON edges.source_id = decisions.id
             WHERE edges.context_id = $context_id
               AND edges.target_id = $entity_id
               AND edges.edge_type = $edge_type
               AND decisions.context_id = $context_id
               AND decisions.status = $status",
            &HashMap::from([
                (
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                ),
                ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
                (
                    "edge_type".to_string(),
                    DbValue::Text("BASED_ON".to_string()),
                ),
                ("status".to_string(), DbValue::Text("active".to_string())),
            ]),
        )
        .map_err(map_engine)?;
    result
        .rows
        .iter()
        .map(|row| match row.first() {
            Some(DbValue::Uuid(id)) => Ok(DecisionId::from(*id)),
            _ => Err(CgError::Engine(
                "active decision lookup returned non-uuid id".to_string(),
            )),
        })
        .collect()
}

fn derived(
    id: &str,
    kind: &str,
    text: &str,
    status: &str,
    digest_id: DigestId,
    context_label: &str,
) -> DerivedMemoryItem {
    DerivedMemoryItem {
        id: id.to_string(),
        kind: kind.to_string(),
        text: text.to_string(),
        status: status.to_string(),
        source_digest_id: Some(digest_id),
        context_label: context_label.to_string(),
    }
}

fn stable_id<T: Into<uuid::Uuid>>(stable: &mut BTreeMap<String, String>, key: &str, id: T) {
    stable.insert(key.to_string(), id.into().to_string());
}

fn stable_id_text(stable: &mut BTreeMap<String, String>, key: &str, id: uuid::Uuid) {
    stable.insert(key.to_string(), id.to_string());
}

fn total_counts(store: &Store, context_id: ContextId) -> Result<HarvestCounts> {
    let db = store.internal_db();
    let count_table = |table: &str| -> Result<u32> {
        Ok(db
            .execute(
                &format!("SELECT id FROM {table} WHERE context_id = $context_id"),
                &HashMap::from([(
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                )]),
            )
            .map_err(map_engine)?
            .rows
            .len() as u32)
    };
    Ok(HarvestCounts {
        goals: count_table("intentions")?,
        decisions: count_table("decisions")?,
        observations: count_table("observations")?,
    })
}

fn invalidation_notice(
    store: &Store,
    invalidation: &crate::Invalidation,
) -> Result<MemoryInvalidationNotice> {
    let affected_decision_id = invalidation.affected_decision_id;
    let threatened_intention_ids = invalidation.threatened_intention_ids.clone();
    let decision = store
        .get_decision(affected_decision_id)?
        .ok_or(CgError::DecisionNotFound {
            id: affected_decision_id,
        })?;
    let served_goal = threatened_intention_ids
        .first()
        .and_then(|id| store.get_intention(*id).ok().flatten())
        .map(|intention| intention.description)
        .unwrap_or_else(|| "unknown served goal".to_string());
    let current = threatened_intention_ids
        .first()
        .and_then(|id| {
            store
                .by_intention(*id, crate::DiscoveryFilter::default())
                .ok()
                .and_then(|rows| rows.into_iter().find(|row| row.id != affected_decision_id))
        })
        .map(|decision| decision.description);
    let basis_diff = memory_basis_diff(&invalidation.basis_diff);
    let basis_line = if basis_diff.is_empty() {
        invalidation
            .reason
            .clone()
            .unwrap_or_else(|| "basis changed".to_string())
    } else {
        basis_diff
            .iter()
            .map(|diff| {
                let old_value = diff.old_value.as_deref().unwrap_or("null");
                let new_value = diff.new_value.as_deref().unwrap_or("null");
                if diff.field == "path" {
                    format!("{old_value} -> {new_value}")
                } else {
                    format!("{}: {old_value} -> {new_value}", diff.field)
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    Ok(MemoryInvalidationNotice {
        affected_decision_id: Some(affected_decision_id),
        decision_text: decision.description,
        basis_line,
        basis_diff,
        threatened_intention_ids,
        served_goal,
        goal_now_unserved: current.is_none(),
        current_decision_text: current,
    })
}

fn memory_basis_diff(diff: &crate::BasisDiffResult) -> Vec<MemoryBasisDiff> {
    diff.divergences
        .iter()
        .flat_map(|divergence| {
            divergence.value_diffs.iter().map(|value| MemoryBasisDiff {
                field: value.field.clone(),
                old_value: value.basis_value.as_ref().map(render_json_value),
                new_value: value.current_value.as_ref().map(render_json_value),
            })
        })
        .collect()
}

fn render_json_value(value: &JsonValue) -> String {
    value
        .as_str()
        .map(ToString::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn compact_summary(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn map_engine(err: contextdb_core::Error) -> CgError {
    CgError::Engine(format!("unexpected engine error: {err}"))
}

fn ensure_runtime_model_table(store: &Store) -> Result<()> {
    let db = store.internal_db();
    if db.table_meta("cg_runtime_models").is_some() {
        return Ok(());
    }
    db.execute(
        "CREATE TABLE cg_runtime_models (id UUID PRIMARY KEY, context_label TEXT NOT NULL, model_key TEXT NOT NULL, advertised_context_tokens INTEGER NOT NULL, default_for_context BOOLEAN NOT NULL, created_at TXID NOT NULL)",
        &HashMap::new(),
    )
    .map_err(map_engine)?;
    Ok(())
}

fn ensure_changed_cursor_table(store: &Store) -> Result<()> {
    let db = store.internal_db();
    if db.table_meta("cg_changed_cursors").is_some() {
        return Ok(());
    }
    db.execute(
        "CREATE TABLE cg_changed_cursors (context_id UUID PRIMARY KEY, last_seen_tx TXID NOT NULL, updated_at TXID NOT NULL)",
        &HashMap::new(),
    )
    .map_err(map_engine)?;
    Ok(())
}

fn read_changed_cursor_tx(store: &Store, context_id: ContextId) -> Result<crate::TxId> {
    Ok(read_changed_cursor_row(store, context_id)?.unwrap_or(crate::TxId(0)))
}

fn read_changed_cursor_row(store: &Store, context_id: ContextId) -> Result<Option<crate::TxId>> {
    ensure_changed_cursor_table(store)?;
    let result = store
        .internal_db()
        .execute(
            "SELECT last_seen_tx FROM cg_changed_cursors WHERE context_id = $context_id",
            &HashMap::from([(
                "context_id".to_string(),
                DbValue::Uuid(context_id.as_uuid()),
            )]),
        )
        .map_err(map_engine)?;
    Ok(result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::TxId(tx)) => Some(*tx),
        _ => None,
    }))
}

fn write_changed_cursor_tx(
    store: &Store,
    context_id: ContextId,
    last_seen: crate::TxId,
) -> Result<()> {
    ensure_changed_cursor_table(store)?;
    let db = store.internal_db();
    let exists = read_changed_cursor_row(store, context_id)?.is_some();
    let tx = db.begin().map_err(map_engine)?;
    let params = HashMap::from([
        (
            "context_id".to_string(),
            DbValue::Uuid(context_id.as_uuid()),
        ),
        ("last_seen_tx".to_string(), DbValue::TxId(last_seen)),
        ("updated_at".to_string(), DbValue::TxId(tx)),
    ]);
    let sql = if exists {
        "UPDATE cg_changed_cursors SET last_seen_tx = $last_seen_tx, updated_at = $updated_at WHERE context_id = $context_id"
    } else {
        "INSERT INTO cg_changed_cursors (context_id, last_seen_tx, updated_at) VALUES ($context_id, $last_seen_tx, $updated_at)"
    };
    db.execute_in_tx(tx, sql, &params).map_err(|err| {
        let _ = db.rollback(tx);
        map_engine(err)
    })?;
    db.commit(tx).map_err(map_engine)
}

fn runtime_model_context(
    store: &Store,
    context_label: &str,
    model_key: Option<&str>,
) -> Result<usize> {
    ensure_runtime_model_table(store)?;
    let mut sql = "SELECT advertised_context_tokens FROM cg_runtime_models WHERE context_label = $context_label".to_string();
    let mut params = HashMap::from([(
        "context_label".to_string(),
        DbValue::Text(context_label.to_string()),
    )]);
    if let Some(model_key) = model_key {
        sql.push_str(" AND model_key = $model_key");
        params.insert(
            "model_key".to_string(),
            DbValue::Text(model_key.to_string()),
        );
    } else {
        sql.push_str(" AND default_for_context = $default");
        params.insert("default".to_string(), DbValue::Bool(true));
    }
    sql.push_str(" ORDER BY created_at DESC LIMIT 1");
    let result = store
        .internal_db()
        .execute(&sql, &params)
        .map_err(map_engine)?;
    Ok(result
        .rows
        .first()
        .and_then(|row| match row.first() {
            Some(DbValue::Int64(tokens)) => Some(*tokens as usize),
            _ => None,
        })
        .unwrap_or(200_000))
}

fn derive_chunk_window(
    model_context: usize,
    prompt_overhead_tokens: usize,
    output_budget: usize,
    running_state_cap: usize,
    explicit_chunk_tokens: Option<usize>,
    context_fraction: Option<f32>,
) -> Result<usize> {
    if let Some(explicit) = explicit_chunk_tokens {
        return Ok(explicit);
    }
    let required = prompt_overhead_tokens + output_budget + running_state_cap + 1;
    if model_context <= required {
        return Err(CgError::TranscriptTooLargeForModel {
            model_context,
            required,
        });
    }
    let additive = model_context - prompt_overhead_tokens - output_budget - running_state_cap;
    let ceiling = (model_context as f32 * context_fraction.unwrap_or(0.6)).floor() as usize;
    Ok(additive.min(ceiling))
}
