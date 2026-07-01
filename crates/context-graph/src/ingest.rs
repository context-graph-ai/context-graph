use chrono::Utc;
use contextdb_core::{Value as DbValue, VectorIndexRef};
use serde::{Deserialize, Deserializer};
use serde_json::{Value as JsonValue, json};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::Instant;

use crate::capture_sync_status::{ModelCallUsage, UsageReportState};
use crate::embedding_space::BuiltinSpaceRegistry;
use crate::grounding;
use crate::{
    AgentId, AuditId, BasisDiffResult, CgError, ContextId, CreateContext, CreateDecision,
    CreateDigest, Decision, DecisionId, DecisionStatus, DigestId, Embedder, EmbeddingId,
    EmbeddingInput, EmbeddingOutput, EmbeddingTarget, EnforcementLevel, EntityId, EntitySnapshotId,
    EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, IntentionId,
    InvalidationId, InvalidationSeverity, InvalidationStatus, ModelCallOutcome, ObservationId,
    PrecomputedEmbedding, ReasoningStep, Result, Store,
};

const EXTRACTION_PRIOR_DECISION_LIMIT: usize = 64;

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

#[derive(Debug, Clone, PartialEq)]
pub struct CapturedEntityInfo {
    pub id: String,
    pub entity_type: String,
    pub key: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CapturedDecisionInfo {
    pub id: String,
    pub decision_type: Option<String>,
    pub agent_id: Option<String>,
    pub confidence: Option<f32>,
    pub reasoning: Vec<String>,
    pub tags: Vec<String>,
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CapturedObservationInfo {
    pub id: String,
    pub observation_type: String,
    pub source: String,
    pub observed_properties: serde_json::Value,
    pub state_delta: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CapturedOutcomeInfo {
    pub id: String,
    pub of_decision_key: String,
    pub success: bool,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlaggedItem {
    pub key: String,
    pub reason: String,
    /// For a CLOSED-VOCABULARY flag (an entity type or a decision type), the legal value set the
    /// correction must be chosen from — so `cg why @<id>` can resolve the item by its id and present
    /// the named set. `None` for a free-text (open) flag (agent_id / observation / outcome), whose
    /// correction is not constrained to a vocabulary.
    pub legal_vocab: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
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
    pub import_chunks: Option<usize>,
    pub resolved_chunk_window: Option<usize>,
    pub model_context_tokens: Option<usize>,
    pub model_usage: Vec<ModelCallUsage>,
    /// Placeholder: always empty. The real implementation lists entities captured from the harvest.
    pub captured_entities: Vec<CapturedEntityInfo>,
    /// Placeholder: always empty. The real implementation lists decisions captured from the harvest.
    pub captured_decisions: Vec<CapturedDecisionInfo>,
    /// Placeholder: always empty. The real implementation lists observations captured from the harvest.
    pub captured_observations: Vec<CapturedObservationInfo>,
    /// Placeholder: always empty. The real implementation lists outcomes captured from the harvest.
    pub captured_outcomes: Vec<CapturedOutcomeInfo>,
    /// Placeholder: always empty. The real implementation lists items the grounding check flagged.
    pub flagged_items: Vec<FlaggedItem>,
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
            import_chunks: None,
            resolved_chunk_window: None,
            model_context_tokens: None,
            model_usage: Vec::new(),
            captured_entities: Vec::new(),
            captured_decisions: Vec::new(),
            captured_observations: Vec::new(),
            captured_outcomes: Vec::new(),
            flagged_items: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureRequest {
    pub context_label: String,
    pub transcript: String,
    pub model_available: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct StructuredHarvest {
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub goals: Vec<GoalSeed>,
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub entities: Vec<EntitySeed>,
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub decisions: Vec<DecisionSeed>,
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub observations: Vec<ObservationSeed>,
    /// Outcomes extracted from the model — currently ignored by the capture harness.
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub outcomes: Vec<OutcomeSeed>,
}

impl StructuredHarvest {
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let value = serde_json::from_str::<JsonValue>(raw).map_err(|err| {
            CgError::Engine(format!(
                "extractor returned invalid structured harvest JSON: {err}"
            ))
        })?;
        serde_json::from_value(value).map_err(|err| {
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

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct GoalSeed {
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub key: String,
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub text: String,
    /// Untrusted model-emitted context_id. The wrong harness trusts this and stores the row
    /// under the emitted context instead of the capture context (cross-tenant leak). The
    /// correct impl ignores this field and always uses the capture context.
    #[serde(default)]
    pub context_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct EntitySeed {
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub key: String,
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub name: String,
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub path: String,
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub entity_type: ExtractedEntityType,
    /// The RAW model-emitted entity-type STRING, when the model reports the type as free text
    /// (the Stage-1 JSON shape: `"entity_type": "Repository"`). When present, capture resolves it to
    /// the matching `EntityType` variant through the live `entity_type_from_str` conversion (in
    /// `resolve_raw_entity_types`, run before grounding and structural fill): a recognized string
    /// (e.g. `"module"` → `Module`) overrides the typed `entity_type`, and an unrecognized string
    /// (e.g. `"wormhole"`) is flagged out-of-vocab — never silently coerced to a default and never
    /// silently dropped.
    #[serde(default)]
    pub entity_type_raw: Option<String>,
    /// Untrusted model-emitted context_id. The wrong harness trusts this and stores the row
    /// under the emitted context instead of the capture context (cross-tenant leak). The
    /// correct impl ignores this field and always uses the capture context.
    #[serde(default)]
    pub context_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractedEntityType {
    Service,
    Person,
    Device,
    Module,
    Repository,
    Configuration,
    Infrastructure,
    Location,
    #[default]
    File,
    Pipeline,
    Database,
    Team,
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
            ExtractedEntityType::Person => EntityType::Person,
            ExtractedEntityType::Device => EntityType::Device,
            ExtractedEntityType::Module => EntityType::Module,
            ExtractedEntityType::Repository => EntityType::Repository,
            ExtractedEntityType::Configuration => EntityType::Configuration,
            ExtractedEntityType::Infrastructure => EntityType::Infrastructure,
            ExtractedEntityType::Location => EntityType::Location,
            ExtractedEntityType::File => EntityType::File,
            ExtractedEntityType::Pipeline => EntityType::Pipeline,
            ExtractedEntityType::Database => EntityType::Database,
            ExtractedEntityType::Team => EntityType::Team,
            ExtractedEntityType::Document => EntityType::Document,
        }
    }
}

/// A field value emitted by the extraction model (such as a decision's `decision_type`). It carries
/// the model's text but is a distinct type that deliberately does not implement `Into<String>`, so a
/// model-extracted value can never be passed where a basis input (an `EvidenceRef` source) is
/// expected — the deterministic core is never seeded from non-deterministic model output. Read the
/// text out explicitly with [`ModelExtractedField::into_value`] when storing it as a plain value.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct ModelExtractedField(String);

impl ModelExtractedField {
    /// Consume the field and return its text, for storing it as a plain value.
    pub fn into_value(self) -> String {
        self.0
    }

    /// Borrow the field's text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ModelExtractedField {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ModelExtractedField {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct DecisionSeed {
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub key: String,
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub text: String,
    #[serde(default)]
    pub goal_key: Option<String>,
    #[serde(default)]
    pub entity_key: Option<String>,
    #[serde(default)]
    pub relation: Option<DecisionRelationSeed>,
    /// From model extraction — the decision's type (e.g. "technology_choice"). Stored as a plain
    /// string value at capture; typed so it can never be passed in as a basis input.
    #[serde(default)]
    pub decision_type: Option<ModelExtractedField>,
    /// From model extraction — agent responsible for the decision.
    #[serde(default)]
    pub agent_id: Option<String>,
    /// From model extraction — confidence (null at capture per Decision §2).
    #[serde(default)]
    pub confidence: Option<f32>,
    /// From model extraction — reasoning steps.
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub reasoning: Vec<String>,
    /// From model extraction — tags.
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub tags: Vec<String>,
    /// From model extraction — properties.
    #[serde(default)]
    pub properties: serde_json::Value,
    /// Verbatim supporting span justifying the `decision_type` closed-vocab label — a quote from the
    /// chunk. The deterministic grounding check (Stage 1.5) reads it: a `decision_type` whose span is
    /// absent, out-of-vocab, mislabeled, or ambiguous is flagged. When absent, the `decision_type` is
    /// unverified metadata (not grounded).
    #[serde(default)]
    pub decision_type_span: Option<ExtractedSpan>,
    /// Closed-vocab labels the `decision_type` span supports as well as (or instead of) the chosen
    /// label. One competing label that differs from the chosen one means the span supports a
    /// different label (mislabel); two or more means the span is ambiguous. The grounding check is
    /// the verdict-holder — the model only supplies the span and the labels it reads from it.
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub decision_type_competing_labels: Vec<String>,
    /// Verbatim supporting span justifying the `agent_id` — a quote from the chunk naming the agent.
    /// When absent, the `agent_id` is unverified metadata (not grounded); when present and absent
    /// from the chunk, the agent_id is flagged (a fabricated attribution).
    #[serde(default)]
    pub agent_id_span: Option<ExtractedSpan>,
    /// Untrusted model-emitted context_id. The wrong harness trusts this and stores the row
    /// under the emitted context instead of the capture context (cross-tenant leak). The
    /// correct impl ignores this field and always uses the capture context.
    #[serde(default)]
    pub context_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct DecisionRelationSeed {
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub prior_decision_id: String,
    #[serde(default, deserialize_with = "deserialize_default_on_null")]
    pub classification: DecisionRelationClassification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionRelationClassification {
    #[default]
    NewInsertion,
    ExactDuplicate,
    RefinesExisting,
    ConflictsExisting,
    SupersedesExisting,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct ObservationSeed {
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub key: String,
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub text: String,
    #[serde(default)]
    pub entity_key: Option<String>,
    #[serde(default)]
    pub path_change: Option<String>,
    /// From model extraction — observation type (e.g. "benchmark").
    #[serde(default)]
    pub observation_type: Option<String>,
    /// From model extraction — source (e.g. "agent-session").
    #[serde(default)]
    pub source: Option<String>,
    /// From model extraction — observed_properties (e.g. {latency_ms: 42}). Currently ignored.
    #[serde(default)]
    pub observed_properties: serde_json::Value,
    /// From model extraction — state_delta.
    #[serde(default)]
    pub state_delta: serde_json::Value,
    /// Verbatim supporting span justifying this observation's value — a quote from the chunk. The
    /// grounding check reads it: an observation whose claimed span is absent from the chunk is
    /// flagged (a fabricated observation). When absent, the observation value is not span-grounded.
    #[serde(default)]
    pub value_span: Option<ExtractedSpan>,
    /// Untrusted model-emitted context_id. The wrong harness trusts this and stores the row
    /// under the emitted context instead of the capture context (cross-tenant leak). The
    /// correct impl ignores this field and always uses the capture context.
    #[serde(default)]
    pub context_id: Option<String>,
}

/// An outcome extracted from the model — maps to Outcome §5.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct OutcomeSeed {
    #[serde(default, deserialize_with = "deserialize_string_default_on_null")]
    pub key: String,
    /// Key of the decision this outcome belongs to.
    #[serde(default)]
    pub of_decision_key: Option<String>,
    #[serde(default)]
    pub success: bool,
    #[serde(default)]
    pub notes: String,
    /// Verbatim supporting span justifying this outcome's value — a quote from the chunk. The
    /// grounding check reads it: an outcome whose claimed span is absent from the chunk is flagged
    /// (a fabricated outcome). When absent, the outcome value is not span-grounded.
    #[serde(default)]
    pub value_span: Option<ExtractedSpan>,
    /// Untrusted model-emitted context_id. The wrong harness trusts this and stores the row
    /// under the emitted context instead of the capture context (cross-tenant leak). The
    /// correct impl ignores this field and always uses the capture context.
    #[serde(default)]
    pub context_id: Option<String>,
}

/// A verbatim supporting span emitted by the model alongside an extracted value.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct ExtractedSpan {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub start: Option<usize>,
    #[serde(default)]
    pub end: Option<usize>,
}

pub struct ExtractionRequest<'a> {
    pub context_label: &'a str,
    pub transcript: &'a str,
    pub prior_decisions: &'a [ExtractionPriorDecision],
}

pub struct DecisionRelationRequest<'a> {
    pub context_label: &'a str,
    pub candidate_decision: &'a str,
    pub candidate_goal: Option<&'a str>,
    pub candidate_entity_name: Option<&'a str>,
    pub candidate_entity_path: Option<&'a str>,
    pub prior_decisions: &'a [ExtractionPriorDecision],
}

/// The verdict of the entity-identity judge: whether a candidate entity is the same as a prior one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityIdentityVerdict {
    /// The candidate is the same entity as a prior one — reuse the existing entity.
    Same,
    /// The candidate is a distinct entity — keep it separate. A false merge is worse than a miss.
    Separate,
}

pub trait ConversationExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> Result<StructuredHarvest>;

    fn classify_decision_relation(
        &self,
        _request: DecisionRelationRequest<'_>,
    ) -> Result<Option<DecisionRelationSeed>> {
        Ok(None)
    }

    /// Decide whether a candidate entity is the SAME as one of the prior candidates. Identity is a
    /// distinct question from a decision relationship: only an *exact duplicate* means "the same
    /// entity"; a Refines/Conflicts/Supersedes relation — or no relation at all — keeps the
    /// candidates separate, so two genuinely distinct entities are never collapsed into one. The
    /// default derives the answer from the decision-relation judge (so the cost/call accounting is
    /// shared); the connected extractor overrides it with an actual same-entity prompt.
    fn classify_entity_identity(
        &self,
        request: DecisionRelationRequest<'_>,
    ) -> Result<EntityIdentityVerdict> {
        let relation = self.classify_decision_relation(request)?;
        Ok(match relation {
            Some(seed) if seed.classification == DecisionRelationClassification::ExactDuplicate => {
                EntityIdentityVerdict::Same
            }
            _ => EntityIdentityVerdict::Separate,
        })
    }

    fn drain_model_usage(&self) -> Vec<ModelCallUsage> {
        Vec::new()
    }
}

/// Wraps any `ConversationExtractor` to tag each call's usage records with the
/// originating flow ("extraction" | "identity-judge") and a stable per-session
/// model identifier. Records emitted by the inner extractor with `flow_tag` or
/// `model_id` already set are preserved as-is.
///
/// When the inner extractor emits no usage records for a call, one timing-based
/// record is synthesized (with `UsageReportState::Unavailable` for the token
/// count, and the measured wall-clock elapsed) so callers can always rely on at
/// least one record per model invocation. This covers test doubles that return a
/// harvest without making a tracked model call.
pub(crate) struct FlowTaggingWrapper<'a> {
    inner: &'a dyn ConversationExtractor,
    accumulated: RefCell<Vec<ModelCallUsage>>,
    default_model_id: &'static str,
}

impl<'a> FlowTaggingWrapper<'a> {
    pub(crate) fn new(inner: &'a dyn ConversationExtractor) -> Self {
        Self {
            inner,
            accumulated: RefCell::new(Vec::new()),
            default_model_id: "context-graph-capture",
        }
    }

    fn drain_and_tag(&self, records: Vec<ModelCallUsage>, flow: &str) {
        let mut acc = self.accumulated.borrow_mut();
        for mut record in records {
            if record.flow_tag.is_none() {
                record.flow_tag = Some(flow.to_string());
            }
            if record.model_id.is_none() {
                record.model_id = Some(self.default_model_id.to_string());
            }
            acc.push(record);
        }
    }

    fn synthesize_and_tag(&self, flow: &str, elapsed: std::time::Duration) {
        let mut acc = self.accumulated.borrow_mut();
        let chunk_index = acc.len() + 1;
        acc.push(ModelCallUsage {
            chunk_index,
            usage: UsageReportState::Unavailable,
            elapsed,
            reliability_level: EnforcementLevel::Floor,
            outcome: ModelCallOutcome::Captured,
            flow_tag: Some(flow.to_string()),
            model_id: Some(self.default_model_id.to_string()),
        });
    }
}

impl ConversationExtractor for FlowTaggingWrapper<'_> {
    fn extract(&self, request: ExtractionRequest<'_>) -> crate::Result<StructuredHarvest> {
        let started = Instant::now();
        let result = self.inner.extract(request)?;
        let elapsed = started.elapsed();
        let drained = self.inner.drain_model_usage();
        if drained.is_empty() {
            self.synthesize_and_tag("extraction", elapsed);
        } else {
            self.drain_and_tag(drained, "extraction");
        }
        Ok(result)
    }

    fn classify_decision_relation(
        &self,
        request: DecisionRelationRequest<'_>,
    ) -> crate::Result<Option<DecisionRelationSeed>> {
        let started = Instant::now();
        let result = self.inner.classify_decision_relation(request)?;
        let elapsed = started.elapsed();
        let drained = self.inner.drain_model_usage();
        if drained.is_empty() {
            // Only synthesize a usage record when classify_decision_relation
            // actually delegated to a model call. An empty drain after a call
            // that returned Ok(None) without calling the model (no prior
            // decisions) means nothing ran and no record should be created.
            // We detect this by checking whether the result indicates a real
            // judge call was made: if the inner extractor ran and the drain
            // is empty, we synthesize; if there were no prior decisions the
            // harness skips calling classify_decision_relation entirely, so
            // this path only triggers on real judge calls whose doubles
            // don't emit usage.
            self.synthesize_and_tag("identity-judge", elapsed);
        } else {
            self.drain_and_tag(drained, "identity-judge");
        }
        Ok(result)
    }

    fn drain_model_usage(&self) -> Vec<ModelCallUsage> {
        std::mem::take(&mut *self.accumulated.borrow_mut())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionPriorDecision {
    pub id: DecisionId,
    pub text: String,
    pub goals: Vec<String>,
    pub entities: Vec<ExtractionPriorEntity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionPriorEntity {
    pub name: String,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupClassification {
    NewInsertion,
    ExactDuplicate,
    RefinesExisting,
    ConflictsExisting,
}

/// The kind of item an identity-resolution outcome describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedItemKind {
    Entity,
    Intention,
    Decision,
}

/// Whether a captured item was matched to an existing identity, judged distinct, or
/// left grouped-but-uncertain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedIdentityVerdict {
    /// Matched to an existing identity and reused it (entity reuse, or an intention
    /// grouped into an existing catalog entry on strong evidence).
    Same,
    /// No matching identity — a fresh row / catalog entry was minted.
    New,
    /// Candidates existed but none was a strong enough match to merge; a distinct row /
    /// catalog entry was minted and a proposal surfaced (a false merge is worse than a miss).
    Unsure,
}

/// One identity-resolution outcome produced while capturing a harvest. The harness
/// surfaces these on its report; `record_conversation` ignores them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedIdentity {
    /// The harvest seed key this outcome describes.
    pub item_key: String,
    pub kind: ResolvedItemKind,
    pub verdict: ResolvedIdentityVerdict,
    /// For a Decision, the classified relation to a prior decision (when one was found).
    pub decision_relation: Option<DecisionRelationClassification>,
}

/// The result of capturing a structured harvest: the capture report plus the Stage-3
/// identity-resolution outcomes the harness reports on.
pub struct HarvestCaptureResult {
    pub report: CaptureReport,
    pub resolutions: Vec<ResolvedIdentity>,
}

/// One existing intention catalog membership: which grouping handle a stored intention
/// in the current context already belongs to.
#[derive(Debug, Clone)]
struct IntentionCatalogEntry {
    intention_id: IntentionId,
    normalized_description: String,
    catalog_id: String,
}

/// The grouping decision for one intention: which catalog handle to assign and how that
/// was judged.
struct IntentionGroupingOutcome {
    catalog_id: String,
    verdict: ResolvedIdentityVerdict,
}

/// Decide which `blueprint_catalog_id` grouping handle an intention belongs to.
///
/// Conservative get-or-create: if this exact intention row already carries a handle, reuse
/// it (a re-capture is the Same identity). If a distinct intention with an identical
/// normalized description already has a handle, the two are the same identity and the new
/// one links to it on strong evidence (a merge). Otherwise a fresh distinct handle is minted —
/// `New` when no other intentions exist in the context, `Unsure` when candidates exist but
/// none is a strong match (a false merge is worse than a miss, so it stays separate and a
/// grouping proposal is what surfaces, never a silent merge).
fn resolve_intention_grouping(
    normalized_description: &str,
    intention_id: IntentionId,
    existing: &[IntentionCatalogEntry],
) -> IntentionGroupingOutcome {
    // Re-capture of the same intention row — reuse its existing handle.
    if let Some(entry) = existing
        .iter()
        .find(|entry| entry.intention_id == intention_id)
    {
        return IntentionGroupingOutcome {
            catalog_id: entry.catalog_id.clone(),
            verdict: ResolvedIdentityVerdict::Same,
        };
    }
    // Strong evidence: a distinct existing intention with an identical normalized description
    // is the same identity — link the new one to that existing handle.
    if !normalized_description.is_empty()
        && let Some(entry) = existing
            .iter()
            .find(|entry| entry.normalized_description == normalized_description)
    {
        return IntentionGroupingOutcome {
            catalog_id: entry.catalog_id.clone(),
            verdict: ResolvedIdentityVerdict::Same,
        };
    }
    // No strong match. Mint a fresh distinct handle. With no candidates this is simply New;
    // with candidates present it is an Unsure (kept separate, a proposal surfaced).
    let verdict = if existing.is_empty() {
        ResolvedIdentityVerdict::New
    } else {
        ResolvedIdentityVerdict::Unsure
    };
    IntentionGroupingOutcome {
        catalog_id: format!("catalog-{}", intention_id.as_uuid()),
        verdict,
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollingCaptureRequest {
    pub context_label: String,
    pub transcript: String,
    pub chunk_window: usize,
    pub running_state_cap_tokens: usize,
    pub model_context_tokens: Option<usize>,
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
    let prior_decisions = extraction_prior_decisions(store, context.id)?;
    // Wrap the extractor to tag each call's usage records with the originating
    // flow ("extraction" | "identity-judge") and a stable model identifier.
    // When the inner extractor emits no records for a call, one timing-based
    // record is synthesized so the usage lane always has a measurement.
    let tagger = FlowTaggingWrapper::new(extractor);
    let mut harvest = tagger.extract(ExtractionRequest {
        context_label: &request.context_label,
        transcript: text,
        prior_decisions: &prior_decisions,
    })?;
    // Resolve any raw model-emitted entity-type strings to their typed variant before grounding and
    // structural fill; an out-of-vocab raw string is flagged here (never silently coerced).
    let mut flagged_items = resolve_raw_entity_types(&mut harvest);
    prepare_harvest_for_capture(
        store,
        context.id,
        &request.context_label,
        &mut harvest,
        &prior_decisions,
        &tagger,
    )?;
    // Run the deterministic grounding check over every value class before consuming the harvest,
    // plus the Stage-3 empty-candidate-set flag for a decision that needed a candidate and found none.
    flagged_items.extend(grounding_check_harvest(&harvest, text));
    flagged_items.extend(flag_empty_candidate_decisions(&harvest, &prior_decisions));
    let mut report = capture_structured_harvest(
        store,
        context.id,
        &request.context_label,
        text,
        harvest,
        &[],
    )?
    .report;
    report.model_usage = tagger.drain_model_usage();
    // Persist each flagged item as a pending review item so `cg review` surfaces it from store state
    // until it is acted on (it survives intervening confident captures and cursor advances).
    crate::review_surface::register_flagged_items(store, &flagged_items);
    report.flagged_items = flagged_items;
    Ok(report)
}

/// Flag the Stage-3 empty-candidate-set fact: a decision that needed a candidate to resolve against —
/// it carries neither a basis entity in this harvest nor any prior decision in the store — has an
/// empty candidate set, which is a deterministic flag (no model call). A decision anchored by a
/// harvested entity, or with prior decisions available to narrow, is not flagged here.
pub(crate) fn flag_empty_candidate_decisions(
    harvest: &StructuredHarvest,
    prior_decisions: &[ExtractionPriorDecision],
) -> Vec<FlaggedItem> {
    let mut flagged = Vec::new();
    if !prior_decisions.is_empty() || !harvest.entities.is_empty() {
        return flagged;
    }
    for seed in &harvest.decisions {
        if seed.key.trim().is_empty() || seed.text.trim().is_empty() {
            continue;
        }
        flagged.push(FlaggedItem {
            key: format!("{}_resolution", seed.key),
            reason: format!(
                "{:?}: decision '{}' had no candidate to resolve against (no basis entity, no prior decision)",
                grounding::FlagReason::EmptyCandidateSet,
                seed.text
            ),
            legal_vocab: None,
        });
    }
    flagged
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
        .unwrap_or("default")
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

pub fn capture_rolling_conversation_with_extractor(
    store: &Store,
    request: RollingCaptureRequest,
    extractor: &dyn ConversationExtractor,
) -> Result<CaptureReport> {
    let chunk_window = request.chunk_window.max(1);
    let text = request.transcript.trim();
    if text.is_empty() {
        let mut report =
            CaptureReport::empty(CaptureMode::Condensed, "nothing to capture: empty import");
        report.import_chunks = Some(0);
        report.resolved_chunk_window = Some(chunk_window);
        report.model_context_tokens = request.model_context_tokens;
        return Ok(report);
    }
    let context = context_by_label(store, &request.context_label)?;
    let prior_decisions = extraction_prior_decisions(store, context.id)?;
    let mut rolling = RollingCaptureAccumulator::new(request.running_state_cap_tokens);
    let chunks = stream_text_chunks(text, chunk_window, |chunk| {
        let transcript = rolling.chunk_prompt(chunk);
        let harvest = extractor.extract(ExtractionRequest {
            context_label: &request.context_label,
            transcript: &transcript,
            prior_decisions: &prior_decisions,
        })?;
        rolling.push_harvest(harvest);
        Ok(())
    })?;
    let mut report = rolling.finish_capture(store, &request.context_label, chunks, extractor)?;
    report.resolved_chunk_window = Some(chunk_window);
    report.model_context_tokens = request.model_context_tokens;
    Ok(report)
}

pub fn capture_rolling_conversation_for_model_with_extractor(
    store: &Store,
    request: RollingIngestRequest,
    extractor: &dyn ConversationExtractor,
) -> Result<CaptureReport> {
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
    capture_rolling_conversation_with_extractor(
        store,
        RollingCaptureRequest {
            context_label: request.context_label,
            transcript: request.transcript,
            chunk_window,
            running_state_cap_tokens: request.running_state_cap_tokens,
            model_context_tokens: Some(model_context),
        },
        extractor,
    )
}

pub struct RollingCaptureAccumulator {
    harvest: StructuredHarvest,
    summary: String,
    summary_cap_chars: usize,
    goal_keys_by_text: BTreeMap<String, String>,
    entity_keys_by_identity: BTreeMap<String, String>,
    decision_texts: BTreeMap<String, String>,
    observation_texts: BTreeMap<String, String>,
}

impl RollingCaptureAccumulator {
    pub fn new(summary_cap_chars: usize) -> Self {
        Self {
            harvest: StructuredHarvest::default(),
            summary: String::new(),
            summary_cap_chars,
            goal_keys_by_text: BTreeMap::new(),
            entity_keys_by_identity: BTreeMap::new(),
            decision_texts: BTreeMap::new(),
            observation_texts: BTreeMap::new(),
        }
    }

    pub fn chunk_prompt(&self, chunk: &str) -> String {
        if self.summary.trim().is_empty() {
            return chunk.to_string();
        }
        format!(
            "Earlier chunks condensed so far:\n{}\n\nNext session chunk:\n{}",
            self.summary, chunk
        )
    }

    pub fn push_harvest(&mut self, harvest: StructuredHarvest) {
        self.push_summary(harvest.summary.as_deref());
        let prefix = format!(
            "c{}",
            self.harvest.goals.len()
                + self.harvest.entities.len()
                + self.harvest.decisions.len()
                + self.harvest.observations.len()
                + 1
        );
        let mut goal_key_map = BTreeMap::new();
        let mut entity_key_map = BTreeMap::new();

        for goal in harvest.goals {
            let normalized = normalize_rolling_text(&goal.text);
            if normalized.is_empty() {
                continue;
            }
            if let Some(existing) = self.goal_keys_by_text.get(&normalized) {
                goal_key_map.insert(goal.key, existing.clone());
                continue;
            }
            let key = unique_rolling_key(&prefix, &goal.key, self.harvest.goals.len() + 1);
            self.goal_keys_by_text.insert(normalized, key.clone());
            goal_key_map.insert(goal.key.clone(), key.clone());
            self.harvest.goals.push(GoalSeed {
                key,
                text: goal.text,
                context_id: None,
            });
        }

        for entity in harvest.entities {
            let identity = normalize_rolling_entity_identity(&entity);
            if identity.is_empty() {
                continue;
            }
            if let Some(existing) = self.entity_keys_by_identity.get(&identity) {
                entity_key_map.insert(entity.key, existing.clone());
                continue;
            }
            let key = unique_rolling_key(&prefix, &entity.key, self.harvest.entities.len() + 1);
            self.entity_keys_by_identity.insert(identity, key.clone());
            entity_key_map.insert(entity.key.clone(), key.clone());
            self.harvest.entities.push(EntitySeed {
                key,
                name: entity.name,
                path: entity.path,
                entity_type: entity.entity_type,
                entity_type_raw: None,
                context_id: None,
            });
        }

        for mut decision in harvest.decisions {
            let normalized = normalize_rolling_text(&decision.text);
            if normalized.is_empty() || self.decision_texts.contains_key(&normalized) {
                continue;
            }
            let key = unique_rolling_key(&prefix, &decision.key, self.harvest.decisions.len() + 1);
            decision.goal_key = decision
                .goal_key
                .as_ref()
                .and_then(|key| goal_key_map.get(key).cloned())
                .or_else(|| self.harvest.goals.first().map(|goal| goal.key.clone()));
            decision.entity_key = decision
                .entity_key
                .as_ref()
                .and_then(|key| entity_key_map.get(key).cloned());
            self.decision_texts.insert(normalized, key.clone());
            self.harvest
                .decisions
                .push(DecisionSeed { key, ..decision });
        }

        for mut observation in harvest.observations {
            let normalized = normalize_rolling_text(&observation.text);
            if normalized.is_empty() || self.observation_texts.contains_key(&normalized) {
                continue;
            }
            let key = unique_rolling_key(
                &prefix,
                &observation.key,
                self.harvest.observations.len() + 1,
            );
            observation.entity_key = observation
                .entity_key
                .as_ref()
                .and_then(|key| entity_key_map.get(key).cloned());
            self.observation_texts.insert(normalized, key.clone());
            self.harvest
                .observations
                .push(ObservationSeed { key, ..observation });
        }
    }

    /// Consume the accumulator and return the merged harvest accreted across every pushed pass.
    /// The harness path uses this when it has windowed extraction itself and wants the single
    /// merged harvest to drive the rest of its pipeline (grounding, identity resolution, atomic
    /// structural fill), rather than `finish_capture`'s self-contained one-shot store write.
    pub fn into_harvest(mut self) -> StructuredHarvest {
        if self.harvest.summary.as_deref().is_none_or(str::is_empty) {
            self.harvest.summary = Some(if self.summary.trim().is_empty() {
                "condensed capture".to_string()
            } else {
                self.summary
            });
        }
        self.harvest
    }

    pub fn finish_capture(
        self,
        store: &Store,
        context_label: &str,
        chunks: usize,
        relation_classifier: &dyn ConversationExtractor,
    ) -> Result<CaptureReport> {
        if chunks == 0 {
            let mut report =
                CaptureReport::empty(CaptureMode::Condensed, "nothing to capture: empty import");
            report.import_chunks = Some(0);
            return Ok(report);
        }
        let digest_text = self.digest_text(chunks);
        let final_harvest = self.finish_harvest(chunks);
        let one_shot = OneShotRollingExtractor {
            harvest: final_harvest,
            relation_classifier,
        };
        let mut report = record_conversation_with_extractor(
            store,
            CaptureRequest {
                context_label: context_label.to_string(),
                transcript: digest_text,
                model_available: true,
            },
            &one_shot,
        )?;
        if chunks > 1 {
            report.message = format!("captured import in {chunks} chunks as one condensed memory");
            report.digest_summary = format!("rolling import: {chunks} chunks");
        }
        report.import_chunks = Some(chunks);
        Ok(report)
    }

    fn push_summary(&mut self, summary: Option<&str>) {
        let Some(summary) = summary.map(str::trim).filter(|summary| !summary.is_empty()) else {
            return;
        };
        if !self.summary.is_empty() {
            self.summary.push('\n');
        }
        self.summary.push_str(summary);
        let summary_limit = self.summary_cap_chars.max(1);
        if self.summary.len() > summary_limit {
            self.summary = bounded_rolling_summary(&self.summary, summary_limit);
        }
    }

    fn finish_harvest(mut self, chunks: usize) -> StructuredHarvest {
        if self.harvest.summary.as_deref().is_none_or(str::is_empty) {
            self.harvest.summary = Some(if self.summary.trim().is_empty() {
                format!("condensed import from {chunks} streamed chunks")
            } else {
                self.summary
            });
        }
        self.harvest
    }

    fn digest_text(&self, chunks: usize) -> String {
        if self.summary.trim().is_empty() {
            format!("condensed import from {chunks} streamed chunks")
        } else {
            self.summary.clone()
        }
    }
}

struct OneShotRollingExtractor<'a> {
    harvest: StructuredHarvest,
    relation_classifier: &'a dyn ConversationExtractor,
}

impl ConversationExtractor for OneShotRollingExtractor<'_> {
    fn extract(&self, _request: ExtractionRequest<'_>) -> Result<StructuredHarvest> {
        Ok(self.harvest.clone())
    }

    fn classify_decision_relation(
        &self,
        request: DecisionRelationRequest<'_>,
    ) -> Result<Option<DecisionRelationSeed>> {
        self.relation_classifier.classify_decision_relation(request)
    }

    fn drain_model_usage(&self) -> Vec<ModelCallUsage> {
        self.relation_classifier.drain_model_usage()
    }
}

/// Split a transcript into windows of at most `chunk_window` bytes, on char boundaries, dropping
/// any empty windows. The harness (`run_capture`) uses this when `CaptureHarnessConfig.chunk_window`
/// is set so a long transcript drives one Stage-1 extraction call per window — the same windowing
/// `stream_text_chunks` performs for the rolling import path, exposed for the harness path.
pub fn split_transcript_into_windows(text: &str, chunk_window: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let _ = stream_text_chunks(text, chunk_window, |chunk| {
        chunks.push(chunk);
        Ok(())
    });
    chunks
}

fn stream_text_chunks<'a, F>(text: &'a str, chunk_window: usize, mut on_chunk: F) -> Result<usize>
where
    F: FnMut(&'a str) -> Result<()>,
{
    let mut rest = text;
    let mut chunks = 0;
    let window = chunk_window.max(1);
    while !rest.is_empty() {
        let split = rolling_chunk_boundary(rest, window);
        let chunk = rest[..split].trim();
        if !chunk.is_empty() {
            chunks += 1;
            on_chunk(chunk)?;
        }
        rest = &rest[split..];
    }
    Ok(chunks)
}

fn rolling_chunk_boundary(text: &str, max_bytes: usize) -> usize {
    if max_bytes >= text.len() {
        return text.len();
    }
    let split = rolling_char_boundary_at_or_before(text, max_bytes);
    if split > 0 {
        split
    } else {
        text.char_indices()
            .nth(1)
            .map(|(idx, _)| idx)
            .unwrap_or(text.len())
    }
}

fn bounded_rolling_summary(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    if limit <= 5 {
        return text.chars().take(limit).collect();
    }
    let half = (limit - 5) / 2;
    let head_end = rolling_char_boundary_at_or_before(text, half);
    let tail_start = rolling_char_boundary_tail_start(text, limit - 5 - head_end);
    format!("{} ... {}", &text[..head_end], &text[tail_start..])
}

fn rolling_char_boundary_at_or_before(text: &str, max_bytes: usize) -> usize {
    if max_bytes >= text.len() {
        return text.len();
    }
    let mut last = 0;
    for (index, ch) in text.char_indices() {
        let end = index + ch.len_utf8();
        if end > max_bytes {
            break;
        }
        last = end;
    }
    last
}

fn rolling_char_boundary_tail_start(text: &str, max_bytes: usize) -> usize {
    if max_bytes >= text.len() {
        return 0;
    }
    let mut start = text.len() - max_bytes;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    start
}

fn unique_rolling_key(prefix: &str, key: &str, ordinal: usize) -> String {
    let clean = key
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let clean = clean.trim_matches('_');
    if clean.is_empty() {
        format!("{prefix}_{ordinal}")
    } else {
        format!("{prefix}_{clean}")
    }
}

fn normalize_rolling_entity_identity(entity: &EntitySeed) -> String {
    let text = if entity.path.trim().is_empty() {
        entity.name.as_str()
    } else {
        entity.path.as_str()
    };
    normalize_rolling_text(text)
}

fn normalize_rolling_text(text: &str) -> String {
    let mut out = String::new();
    let mut last_space = false;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim().to_string()
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
        import_chunks: None,
        resolved_chunk_window: None,
        model_context_tokens: None,
        model_usage: Vec::new(),
        captured_entities: Vec::new(),
        captured_decisions: Vec::new(),
        captured_observations: Vec::new(),
        captured_outcomes: Vec::new(),
        flagged_items: Vec::new(),
    })
}

fn prepare_harvest_for_capture(
    store: &Store,
    context_id: ContextId,
    context_label: &str,
    harvest: &mut StructuredHarvest,
    prior_decisions: &[ExtractionPriorDecision],
    extractor: &dyn ConversationExtractor,
) -> Result<()> {
    reuse_prior_goals_for_entity_decisions(harvest, prior_decisions);
    ensure_goal_keys_for_decisions(harvest, context_label);
    complete_decision_relations(
        store,
        context_id,
        context_label,
        harvest,
        prior_decisions,
        extractor,
    )?;
    validate_harvest_relations(store, context_id, harvest)
}

fn reuse_prior_goals_for_entity_decisions(
    harvest: &mut StructuredHarvest,
    prior_decisions: &[ExtractionPriorDecision],
) {
    if prior_decisions.is_empty() || harvest.decisions.is_empty() || harvest.entities.is_empty() {
        return;
    }
    let entity_names = harvest
        .entities
        .iter()
        .map(|entity| {
            (
                entity.key.clone(),
                normalize_entity_identity_name(&entity.name),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let known_goal_keys = harvest
        .goals
        .iter()
        .map(|goal| goal.key.clone())
        .collect::<BTreeSet<_>>();
    let mut existing_goal_keys = harvest
        .goals
        .iter()
        .map(|goal| {
            (
                normalize_intention_description(&goal.text),
                goal.key.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut updates = Vec::new();
    for (index, decision) in harvest.decisions.iter().enumerate() {
        if decision
            .goal_key
            .as_ref()
            .is_some_and(|key| known_goal_keys.contains(key))
        {
            continue;
        }
        let Some(entity_name) = decision
            .entity_key
            .as_ref()
            .and_then(|key| entity_names.get(key))
            .filter(|name| !name.is_empty())
        else {
            continue;
        };
        let Some(prior_goal) = prior_decisions
            .iter()
            .find(|prior| {
                prior
                    .entities
                    .iter()
                    .any(|entity| normalize_entity_identity_name(&entity.name) == *entity_name)
            })
            .and_then(|prior| prior.goals.first())
            .filter(|goal| !goal.trim().is_empty())
        else {
            continue;
        };
        let normalized_goal = normalize_intention_description(prior_goal);
        let key = if let Some(existing) = existing_goal_keys.get(&normalized_goal) {
            existing.clone()
        } else {
            let key = format!("G_prior_{}", harvest.goals.len() + 1);
            harvest.goals.push(GoalSeed {
                key: key.clone(),
                text: prior_goal.clone(),
                context_id: None,
            });
            existing_goal_keys.insert(normalized_goal, key.clone());
            key
        };
        updates.push((index, key));
    }
    for (index, key) in updates {
        if let Some(decision) = harvest.decisions.get_mut(index) {
            decision.goal_key = Some(key);
        }
    }
}

fn ensure_goal_keys_for_decisions(harvest: &mut StructuredHarvest, context_label: &str) {
    if harvest.decisions.is_empty() {
        return;
    }
    if harvest.goals.is_empty() {
        harvest.goals.push(GoalSeed {
            key: "G_context".to_string(),
            text: format!("track durable decisions for {context_label}"),
            context_id: None,
        });
    }
    let goal_keys = harvest
        .goals
        .iter()
        .map(|goal| goal.key.as_str())
        .collect::<BTreeSet<_>>();
    let fallback = harvest.goals.first().map(|goal| goal.key.clone());
    for decision in &mut harvest.decisions {
        let missing_or_unknown = decision
            .goal_key
            .as_deref()
            .is_none_or(|key| !goal_keys.contains(key));
        if missing_or_unknown {
            decision.goal_key = fallback.clone();
        }
    }
}

fn complete_decision_relations(
    store: &Store,
    context_id: ContextId,
    context_label: &str,
    harvest: &mut StructuredHarvest,
    prior_decisions: &[ExtractionPriorDecision],
    extractor: &dyn ConversationExtractor,
) -> Result<()> {
    if prior_decisions.is_empty() {
        for decision in &mut harvest.decisions {
            decision.relation = None;
        }
        return Ok(());
    }
    let goal_text = harvest
        .goals
        .iter()
        .map(|goal| (goal.key.as_str(), goal.text.as_str()))
        .collect::<BTreeMap<_, _>>();
    let entity_data = harvest
        .entities
        .iter()
        .map(|entity| {
            (
                entity.key.as_str(),
                (entity.name.as_str(), entity.path.as_str()),
            )
        })
        .collect::<BTreeMap<_, _>>();

    for decision in &mut harvest.decisions {
        if decision.text.trim().is_empty() {
            continue;
        }
        let relation_is_valid = match decision.relation.as_ref() {
            Some(relation)
                if relation.classification == DecisionRelationClassification::NewInsertion
                    || relation.prior_decision_id.trim().is_empty() =>
            {
                false
            }
            Some(relation)
                if relation.classification != DecisionRelationClassification::NewInsertion
                    && !relation.prior_decision_id.trim().is_empty() =>
            {
                match resolved_decision_relation(store, context_id, Some(relation)) {
                    Ok(_) => true,
                    Err(CgError::CrossContextViolation) => {
                        return Err(CgError::CrossContextViolation);
                    }
                    Err(_) => false,
                }
            }
            Some(_) => false,
            None => false,
        };
        if relation_is_valid {
            continue;
        }

        let candidate_goal = decision
            .goal_key
            .as_deref()
            .and_then(|key| goal_text.get(key).copied());
        let (candidate_entity_name, candidate_entity_path) = decision
            .entity_key
            .as_deref()
            .and_then(|key| entity_data.get(key).copied())
            .map(|(name, path)| (Some(name), Some(path)))
            .unwrap_or((None, None));
        if let Some(relation) = extractor.classify_decision_relation(DecisionRelationRequest {
            context_label,
            candidate_decision: &decision.text,
            candidate_goal,
            candidate_entity_name,
            candidate_entity_path,
            prior_decisions,
        })? {
            if relation.classification == DecisionRelationClassification::NewInsertion
                || relation.prior_decision_id.trim().is_empty()
            {
                decision.relation = None;
            } else {
                decision.relation = Some(relation);
            }
        }
        if decision.relation.is_none()
            && let Some(relation) =
                structural_relation_for_decision(&decision.text, candidate_goal, prior_decisions)
        {
            decision.relation = Some(relation);
        }
    }
    Ok(())
}

fn structural_relation_for_decision(
    candidate_decision: &str,
    candidate_goal: Option<&str>,
    prior_decisions: &[ExtractionPriorDecision],
) -> Option<DecisionRelationSeed> {
    if !replacement_choice_language(candidate_decision) {
        return None;
    }
    let candidate_goal = candidate_goal
        .map(normalize_intention_description)
        .unwrap_or_default();
    prior_decisions
        .iter()
        .find(|prior| {
            replacement_choice_language(&prior.text)
                && ((!candidate_goal.is_empty()
                    && prior
                        .goals
                        .iter()
                        .any(|goal| normalize_intention_description(goal) == candidate_goal))
                    || choice_subject_matches(candidate_decision, &prior.text))
        })
        .map(|prior| DecisionRelationSeed {
            prior_decision_id: prior.id.as_uuid().to_string(),
            classification: DecisionRelationClassification::SupersedesExisting,
        })
}

fn choice_subject_matches(candidate: &str, prior: &str) -> bool {
    let candidate = choice_subject_tokens(candidate);
    let prior = choice_subject_tokens(prior);
    if candidate.is_empty() || prior.is_empty() {
        return false;
    }
    let shared = candidate.intersection(&prior).count();
    shared >= 2 && shared * 2 >= candidate.len().min(prior.len())
}

fn choice_subject_tokens(text: &str) -> BTreeSet<String> {
    normalize_intention_description(text)
        .split_whitespace()
        .filter(|word| {
            !choice_subject_stopword(word) && !word.chars().any(|ch| ch.is_ascii_digit())
        })
        .map(str::to_string)
        .collect()
}

fn choice_subject_stopword(word: &str) -> bool {
    matches!(
        word,
        "and"
            | "as"
            | "choose"
            | "chosen"
            | "decision"
            | "decide"
            | "decided"
            | "default"
            | "for"
            | "from"
            | "in"
            | "make"
            | "new"
            | "now"
            | "of"
            | "old"
            | "on"
            | "or"
            | "replace"
            | "replaced"
            | "replacing"
            | "select"
            | "selected"
            | "set"
            | "should"
            | "switch"
            | "switching"
            | "to"
            | "use"
            | "uses"
            | "using"
            | "we"
            | "will"
            | "with"
            | "without"
    )
}

fn replacement_choice_language(text: &str) -> bool {
    normalize_intention_description(text)
        .split_whitespace()
        .any(|word| {
            matches!(
                word,
                "default" | "switch" | "switching" | "replace" | "replacing" | "replaced"
            )
        })
}

fn validate_harvest_relations(
    store: &Store,
    context_id: ContextId,
    harvest: &StructuredHarvest,
) -> Result<()> {
    for decision in &harvest.decisions {
        resolved_decision_relation(store, context_id, decision.relation.as_ref())?;
    }
    Ok(())
}

/// The default closed vocabulary of `decision_type` labels the grounding check measures a model's
/// `decision_type` against. A `decision_type` not in this set is flagged out-of-vocab. This is the
/// `decision_type` analogue of the entity-type vocabulary single source; a later plan can widen it
/// per tenant without touching the grounding logic.
pub(crate) fn default_decision_type_vocabulary() -> &'static [&'static str] {
    &[
        "technology_choice",
        "architecture",
        "api_design",
        "infrastructure",
        "security",
        "security_policy",
        "monitoring_threshold",
        "storage_choice",
        "policy",
        "process",
        "change",
    ]
}

/// Resolve any RAW model-emitted entity-type strings on a harvest's entity seeds to their typed
/// variant through the canonical conversion, in place. An entity whose raw type string is recognized
/// has its typed `entity_type` set to the resolved variant; one whose raw string is out of vocabulary
/// is left at its default and reported here so the caller can flag it (never silently coerced). Runs
/// before grounding and structural fill so the resolved type is what gets stored.
pub(crate) fn resolve_raw_entity_types(harvest: &mut StructuredHarvest) -> Vec<FlaggedItem> {
    let mut flagged = Vec::new();
    for seed in &mut harvest.entities {
        let Some(raw) = seed.entity_type_raw.as_ref().map(|s| s.trim().to_string()) else {
            continue;
        };
        if raw.is_empty() {
            continue;
        }
        match crate::store::convert::entity_type_from_label(&raw) {
            Some(resolved) => {
                seed.entity_type = extracted_entity_type_from(resolved);
            }
            None => {
                flagged.push(FlaggedItem {
                    key: seed.key.clone(),
                    reason: format!(
                        "{:?}: entity '{}' raw type '{}' is not in the closed entity-type vocabulary",
                        grounding::FlagReason::OutOfVocab,
                        seed.name,
                        raw
                    ),
                    legal_vocab: Some(grounding::legal_entity_type_names()),
                });
            }
        }
    }
    flagged
}

/// Map a resolved `EntityType` back to the `ExtractedEntityType` the harvest seed carries (the two
/// enums are 1:1 over the 13 closed variants), so a raw-string-resolved type rides the existing
/// structural-fill path unchanged.
pub(crate) fn extracted_entity_type_from(entity_type: EntityType) -> ExtractedEntityType {
    match entity_type {
        EntityType::Service => ExtractedEntityType::Service,
        EntityType::Person => ExtractedEntityType::Person,
        EntityType::Device => ExtractedEntityType::Device,
        EntityType::Module => ExtractedEntityType::Module,
        EntityType::Repository => ExtractedEntityType::Repository,
        EntityType::Configuration => ExtractedEntityType::Configuration,
        EntityType::Infrastructure => ExtractedEntityType::Infrastructure,
        EntityType::Location => ExtractedEntityType::Location,
        EntityType::File => ExtractedEntityType::File,
        EntityType::Pipeline => ExtractedEntityType::Pipeline,
        EntityType::Database => ExtractedEntityType::Database,
        EntityType::Team => ExtractedEntityType::Team,
        EntityType::Document => ExtractedEntityType::Document,
    }
}

/// Run the deterministic grounding check over EVERY extracted value class in a harvest (entity,
/// decision_type, agent_id, observation, outcome) — pure string comparison, zero model calls. The
/// flag signal is a function of (span, chunk, value, vocab, competing labels); the model's confidence
/// is never consulted. Returns the items that could not be grounded against the transcript.
pub(crate) fn grounding_check_harvest(
    harvest: &StructuredHarvest,
    transcript: &str,
) -> Vec<FlaggedItem> {
    let mut flagged = Vec::new();

    // Entities (name + chosen type), as before.
    for seed in &harvest.entities {
        if seed.key.trim().is_empty() || seed.name.trim().is_empty() {
            continue;
        }
        let entity_type: EntityType = seed.entity_type.into();
        let type_name = entity_type_label(entity_type);
        if let Some(reason) = grounding::check_entity_grounding(&seed.name, type_name, transcript) {
            flagged.push(FlaggedItem {
                key: seed.key.clone(),
                reason: format!("{reason:?}: entity '{}' (type={})", seed.name, type_name),
                legal_vocab: Some(grounding::legal_entity_type_names()),
            });
        }
    }

    let decision_vocab: Vec<String> = default_decision_type_vocabulary()
        .iter()
        .map(|s| s.to_string())
        .collect();

    for seed in &harvest.decisions {
        if seed.key.trim().is_empty() {
            continue;
        }
        // decision_type — a closed-vocab value justified by a verbatim span (the decision phrase).
        if let Some(decision_type) = seed.decision_type.as_ref().map(|f| f.as_str().to_string())
            && !decision_type.is_empty()
        {
            let verdict = grounding::full_grounding_check(&grounding::GroundingInput {
                value: decision_type.clone(),
                cited_span: span_text(seed.decision_type_span.as_ref()),
                chunk: transcript.to_string(),
                model_confidence: 1.0,
                vocab: Some(decision_vocab.clone()),
                label: Some(decision_type.clone()),
                competing_labels: seed.decision_type_competing_labels.clone(),
            });
            if let grounding::GroundingVerdict::Flagged(reason) = verdict {
                flagged.push(FlaggedItem {
                    key: format!("{}_decision_type", seed.key),
                    reason: format!(
                        "{reason:?}: decision_type '{}' (span={:?})",
                        decision_type,
                        seed.decision_type_span
                            .as_ref()
                            .and_then(|s| s.text.clone())
                            .unwrap_or_default()
                    ),
                    legal_vocab: Some(decision_vocab.clone()),
                });
            }
        }
        // agent_id — only span-grounded when the model claims a quote; absent span = metadata.
        if let Some(agent_id) = seed.agent_id.as_ref()
            && !agent_id.is_empty()
            && let Some(span) = span_text(seed.agent_id_span.as_ref())
        {
            let verdict = grounding::full_grounding_check(&grounding::GroundingInput {
                value: agent_id.clone(),
                cited_span: Some(span),
                chunk: transcript.to_string(),
                ..Default::default()
            });
            if let grounding::GroundingVerdict::Flagged(reason) = verdict {
                flagged.push(FlaggedItem {
                    key: format!("{}_agent_id", seed.key),
                    reason: format!("{reason:?}: agent_id '{agent_id}'"),
                    legal_vocab: None,
                });
            }
        }
    }

    // Observations — a value with no supporting span flags MissingSpan (§2: flagged when, and only
    // when, the span is missing or does not occur in the chunk).
    for seed in &harvest.observations {
        if seed.key.trim().is_empty() {
            continue;
        }
        let verdict = grounding::full_grounding_check(&grounding::GroundingInput {
            value: seed.text.clone(),
            cited_span: span_text(seed.value_span.as_ref()),
            chunk: transcript.to_string(),
            ..Default::default()
        });
        if let grounding::GroundingVerdict::Flagged(reason) = verdict {
            flagged.push(FlaggedItem {
                key: format!("{}_observation", seed.key),
                reason: format!("{reason:?}: observation '{}'", seed.text),
                legal_vocab: None,
            });
        }
    }

    // Outcomes — a value with no supporting span flags MissingSpan (§2, as for observations).
    for seed in &harvest.outcomes {
        if seed.key.trim().is_empty() {
            continue;
        }
        let verdict = grounding::full_grounding_check(&grounding::GroundingInput {
            value: seed.notes.clone(),
            cited_span: span_text(seed.value_span.as_ref()),
            chunk: transcript.to_string(),
            ..Default::default()
        });
        if let grounding::GroundingVerdict::Flagged(reason) = verdict {
            flagged.push(FlaggedItem {
                key: format!("{}_outcome", seed.key),
                reason: format!("{reason:?}: outcome '{}'", seed.notes),
                legal_vocab: None,
            });
        }
    }

    flagged
}

/// Extract the verbatim span text from an optional `ExtractedSpan`, treating an empty quote as no
/// span (the grounding check then flags it MissingSpan).
fn span_text(span: Option<&ExtractedSpan>) -> Option<String> {
    span.and_then(|s| s.text.clone())
        .filter(|text| !text.is_empty())
}

pub(crate) fn capture_structured_harvest(
    store: &Store,
    context_id: ContextId,
    context_label: &str,
    transcript: &str,
    harvest: StructuredHarvest,
    skill_firing_records: &[crate::skill_store::SkillFiringRecord],
) -> Result<HarvestCaptureResult> {
    if harvest.is_empty() {
        return Ok(HarvestCaptureResult {
            report: capture_zero_harvest_digest(store, context_id, transcript)?,
            resolutions: Vec::new(),
        });
    }

    // Identity-resolution outcomes recorded as we resolve entities, intentions, and
    // decisions against existing rows. The harness surfaces these on its report.
    let mut resolutions: Vec<ResolvedIdentity> = Vec::new();
    let mut tx = CaptureTxn::begin(store)?;
    let summary = harvest
        .summary
        .as_deref()
        .map(str::trim)
        .filter(|summary| !summary.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| compact_summary(transcript));
    let digest = tx.create_text_digest(context_id, &summary, &summary)?;
    let mut stable = BTreeMap::new();
    let mut source_ids = Vec::<(String, uuid::Uuid)>::new();
    let mut derived_items = Vec::new();
    let mut goals = BTreeMap::<String, IntentionId>::new();
    let mut goal_ids_by_normalized_text = BTreeMap::<String, IntentionId>::new();
    // Stage-3 intention grouping reads the catalog entries already present in this context
    // (each captured intention carries a `blueprint_catalog_id` grouping handle). The grouping
    // judgement is conservative: an exact existing intention reuses its entry (a merge), an
    // unrelated one mints a fresh distinct entry, and a candidate that is similar but not a
    // strong match mints a fresh entry plus a proposal rather than merging silently.
    let mut existing_catalog = tx.intention_catalog_entries(context_id)?;
    for seed in &harvest.goals {
        if seed.key.trim().is_empty() || seed.text.trim().is_empty() {
            continue;
        }
        let normalized = normalize_intention_description(&seed.text);
        if let Some(goal) = goal_ids_by_normalized_text.get(&normalized).copied() {
            stable_id(&mut stable, &seed.key, goal);
            stable_id_text(&mut stable, &seed.text, goal.as_uuid());
            goals.insert(seed.key.clone(), goal);
            continue;
        }
        let goal = tx.ensure_intention(context_id, &seed.text)?;
        goal_ids_by_normalized_text.insert(normalized.clone(), goal);
        // Assign or reuse the catalog grouping handle.
        let grouping = resolve_intention_grouping(&normalized, goal, &existing_catalog);
        tx.set_intention_catalog_id(context_id, goal, &grouping.catalog_id)?;
        existing_catalog.push(IntentionCatalogEntry {
            intention_id: goal,
            normalized_description: normalized,
            catalog_id: grouping.catalog_id.clone(),
        });
        resolutions.push(ResolvedIdentity {
            item_key: seed.key.clone(),
            kind: ResolvedItemKind::Intention,
            verdict: grouping.verdict,
            decision_relation: None,
        });
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
    let mut entity_prior_paths = BTreeMap::<String, Option<String>>::new();
    let mut entity_names = BTreeMap::<String, String>::new();
    let mut entity_writes_by_identity = BTreeMap::<String, EntityWrite>::new();
    for seed in &harvest.entities {
        if seed.key.trim().is_empty() || seed.name.trim().is_empty() {
            continue;
        }
        let path = if seed.path.trim().is_empty() {
            seed.name.clone()
        } else {
            seed.path.clone()
        };
        let entity_type = seed.entity_type.into();
        let entity_dedup_key = format!(
            "{}:{}:{}",
            entity_type_label(entity_type),
            normalize_entity_identity_name(&seed.name),
            path.trim().to_ascii_lowercase()
        );
        if let Some(entity) = entity_writes_by_identity.get(&entity_dedup_key).cloned() {
            stable_id(&mut stable, &seed.key, entity.id);
            stable_id_text(&mut stable, &seed.name, entity.id.as_uuid());
            stable_id_text(&mut stable, &path, entity.id.as_uuid());
            entities.insert(seed.key.clone(), entity.id);
            entity_paths.insert(seed.key.clone(), path);
            entity_prior_paths.insert(seed.key.clone(), entity.prior_path);
            entity_names.insert(seed.key.clone(), seed.name.clone());
            continue;
        }
        // Entity reuse (Stage-3): an entity matched to an existing row by name+context is a
        // Same verdict; a freshly minted one is New.
        let entity_existed =
            find_existing_entity_for_seed(store, context_id, &seed.name, &path, entity_type)?
                .is_some();
        let entity =
            tx.ensure_entity_with_prior_path(context_id, &seed.name, &path, entity_type)?;
        resolutions.push(ResolvedIdentity {
            item_key: seed.key.clone(),
            kind: ResolvedItemKind::Entity,
            verdict: if entity_existed {
                ResolvedIdentityVerdict::Same
            } else {
                ResolvedIdentityVerdict::New
            },
            decision_relation: None,
        });
        entity_writes_by_identity.insert(entity_dedup_key, entity.clone());
        stable_id(&mut stable, &seed.key, entity.id);
        stable_id_text(&mut stable, &seed.name, entity.id.as_uuid());
        stable_id_text(&mut stable, &path, entity.id.as_uuid());
        source_ids.push((seed.key.clone(), entity.id.as_uuid()));
        entities.insert(seed.key.clone(), entity.id);
        entity_paths.insert(seed.key.clone(), path);
        entity_prior_paths.insert(seed.key.clone(), entity.prior_path);
        entity_names.insert(seed.key.clone(), seed.name.clone());
    }

    let first_goal = goals.values().next().copied();
    let mut decision_items = Vec::<(String, DecisionId, String)>::new();
    let fallback_entity = entities.values().next().copied();
    let mut invalidation_rows = Vec::new();
    let mut dedup_events = Vec::new();
    let mut observation_items = Vec::new();
    let mut observation_source_ids = Vec::<(String, uuid::Uuid)>::new();
    let mut observation_ids_by_normalized_text = BTreeMap::<String, ObservationId>::new();
    let explicit_path_change_entities = harvest
        .observations
        .iter()
        .filter_map(|seed| seed.path_change.as_ref().and(seed.entity_key.as_ref()))
        .cloned()
        .collect::<BTreeSet<_>>();
    let captured_entity_path_changes = entities
        .iter()
        .filter_map(|(entity_key, entity)| {
            if explicit_path_change_entities.contains(entity_key) {
                return None;
            }
            let prior = entity_prior_paths.get(entity_key)?.as_ref()?;
            let candidate = entity_paths.get(entity_key)?;
            (prior != candidate).then(|| {
                (
                    entity_key.clone(),
                    *entity,
                    prior.clone(),
                    candidate.clone(),
                    entity_names
                        .get(entity_key)
                        .cloned()
                        .unwrap_or_else(|| entity_key.clone()),
                )
            })
        })
        .collect::<Vec<_>>();
    for (entity_key, entity, prior_value, candidate_value, entity_name) in
        captured_entity_path_changes
    {
        let observation_key = format!("{entity_key}_path");
        let observation_text = format!("{entity_name} observed at {candidate_value}");
        let write = tx.ensure_observation_with_invalidations(
            context_id,
            entity,
            &observation_text,
            Some(("path", candidate_value.as_str())),
        )?;
        stable_id(&mut stable, &observation_key, write.id);
        stable_id_text(&mut stable, &observation_text, write.id.as_uuid());
        observation_source_ids.push((observation_key.clone(), write.id.as_uuid()));
        observation_items.push(derived(
            &observation_key,
            "observation",
            &observation_text,
            "current",
            digest.id,
            context_label,
        ));
        dedup_events.push(DedupEvent {
            entity: entity_name.clone(),
            prior_value: prior_value.clone(),
            candidate_value: candidate_value.clone(),
            classification: DedupClassification::ConflictsExisting,
        });
        invalidation_rows.extend(write.invalidations);
    }
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
            .and_then(|key| entity_prior_paths.get(key))
            .and_then(|path| path.as_ref())
            .cloned();
        let path_change = seed
            .path_change
            .as_deref()
            .map(normalize_path_change_value)
            .filter(|path| !path.is_empty());
        let observation_dedup_key = format!(
            "{}:{}:{}",
            entity.as_uuid(),
            normalize_intention_description(&seed.text),
            path_change.as_deref().unwrap_or_default()
        );
        if let Some(observation) = observation_ids_by_normalized_text
            .get(&observation_dedup_key)
            .copied()
        {
            stable_id(&mut stable, &seed.key, observation);
            stable_id_text(&mut stable, &seed.text, observation.as_uuid());
            continue;
        }
        let write = tx.ensure_observation_from_seed(context_id, entity, seed)?;
        observation_ids_by_normalized_text.insert(observation_dedup_key, write.id);
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
            dedup_events.push(DedupEvent {
                entity: entity_name.clone(),
                prior_value: prior_value.clone(),
                candidate_value: candidate_value.clone(),
                classification: DedupClassification::ConflictsExisting,
            });
        }
        invalidation_rows.extend(write.invalidations);
    }

    let mut decision_ids_by_normalized_text = BTreeMap::<String, DecisionId>::new();
    let mut decision_ids_by_key: HashMap<String, DecisionId> = HashMap::new();
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
        let decision_dedup_key = normalize_intention_description(&seed.text);
        if let Some(decision) = decision_ids_by_normalized_text
            .get(&decision_dedup_key)
            .copied()
        {
            stable_id(&mut stable, &seed.key, decision);
            stable_id_text(&mut stable, &seed.text, decision.as_uuid());
            continue;
        }
        let relation = resolved_decision_relation(store, context_id, seed.relation.as_ref())?;
        // Record the classified relation for the harness report (None prior → New insertion).
        resolutions.push(ResolvedIdentity {
            item_key: seed.key.clone(),
            kind: ResolvedItemKind::Decision,
            verdict: match relation.as_ref().map(|(classification, _)| *classification) {
                Some(DecisionRelationClassification::ExactDuplicate) => {
                    ResolvedIdentityVerdict::Same
                }
                _ => ResolvedIdentityVerdict::New,
            },
            decision_relation: Some(
                relation
                    .as_ref()
                    .map(|(classification, _)| *classification)
                    .unwrap_or(DecisionRelationClassification::NewInsertion),
            ),
        });
        let decision = match relation.as_ref().map(|(classification, _)| *classification) {
            Some(DecisionRelationClassification::ExactDuplicate) => {
                let prior_row = &relation.as_ref().expect("relation is present").1;
                let prior = prior_row.id;
                dedup_events.push(DedupEvent {
                    entity: relation_entity_label(seed, &entity_names),
                    prior_value: relation
                        .as_ref()
                        .expect("relation is present")
                        .1
                        .description
                        .clone(),
                    candidate_value: seed.text.clone(),
                    classification: DedupClassification::ExactDuplicate,
                });
                let prior_status = tx.decision_status(prior)?.unwrap_or(prior_row.status);
                if prior_status == DecisionStatus::Active && !tx.will_invalidate_decision(prior) {
                    prior
                } else {
                    tx.create_decision(decision_input(
                        context_id,
                        seed,
                        &[goal],
                        entity,
                        Vec::new(),
                    ))?
                    .id
                }
            }
            Some(DecisionRelationClassification::RefinesExisting) => {
                let prior = relation.as_ref().expect("relation is present").1.id;
                let precedent_ids = tx.precedent_ids_if_prior_stays_current(prior)?;
                dedup_events.push(DedupEvent {
                    entity: relation_entity_label(seed, &entity_names),
                    prior_value: relation
                        .as_ref()
                        .expect("relation is present")
                        .1
                        .description
                        .clone(),
                    candidate_value: seed.text.clone(),
                    classification: DedupClassification::RefinesExisting,
                });
                let refined = tx
                    .create_decision(decision_input(
                        context_id,
                        seed,
                        &[goal],
                        entity,
                        precedent_ids,
                    ))?
                    .id;
                // A REFINES edge records the refinement relation; the prior stays Active.
                tx.ensure_decision_relation_edge(refined, prior, "REFINES", context_id)?;
                refined
            }
            Some(DecisionRelationClassification::ConflictsExisting) => {
                let prior = relation.as_ref().expect("relation is present").1.id;
                let precedent_ids = tx.precedent_ids_if_prior_stays_current(prior)?;
                let prior_text = relation
                    .as_ref()
                    .expect("relation is present")
                    .1
                    .description
                    .clone();
                dedup_events.push(DedupEvent {
                    entity: relation_entity_label(seed, &entity_names),
                    prior_value: prior_text.clone(),
                    candidate_value: seed.text.clone(),
                    classification: DedupClassification::ConflictsExisting,
                });
                let decision = tx
                    .create_decision(decision_input(
                        context_id,
                        seed,
                        &[goal],
                        entity,
                        precedent_ids,
                    ))?
                    .id;
                let trigger = tx.ensure_semantic_conflict_observation(
                    context_id,
                    entity,
                    prior,
                    &prior_text,
                    &seed.text,
                )?;
                tx.invalidate_decision(
                    prior,
                    trigger,
                    format!(
                        "model classified extracted decision as conflicting with prior decision: {} -> {}",
                        prior_text, seed.text
                    ),
                )?;
                invalidation_rows.extend(tx.list_invalidations(context_id, Some(prior))?);
                decision
            }
            Some(DecisionRelationClassification::SupersedesExisting) => {
                let prior = relation.as_ref().expect("relation is present").1.id;
                tx.supersede_decision(
                    prior,
                    decision_input(context_id, seed, &[goal], entity, Vec::new()),
                )?
                .id
            }
            Some(DecisionRelationClassification::NewInsertion) | None => {
                // find-or-create using full seed metadata (decision_type, agent_id, reasoning…)
                if let Some(id) =
                    find_id_by_text(store, "decisions", "description", &seed.text, context_id)?
                {
                    DecisionId::from(id)
                } else {
                    tx.create_decision(decision_input(
                        context_id,
                        seed,
                        &[goal],
                        entity,
                        Vec::new(),
                    ))?
                    .id
                }
            }
        };
        decision_ids_by_normalized_text.insert(decision_dedup_key, decision);
        decision_ids_by_key.insert(seed.key.clone(), decision);
        stable_id(&mut stable, &seed.key, decision);
        stable_id_text(&mut stable, &seed.text, decision.as_uuid());
        source_ids.push((seed.key.clone(), decision.as_uuid()));
        decision_items.push((seed.key.clone(), decision, seed.text.clone()));
    }

    for (key, decision, text) in &decision_items {
        let status = tx
            .decision_status(*decision)?
            .map(decision_status_label)
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
    derived_items.extend(observation_items);
    source_ids.extend(observation_source_ids);

    let source_refs = source_ids
        .iter()
        .map(|(key, id)| (key.as_str(), *id))
        .collect::<Vec<_>>();
    let provenance = tx.add_provenance_edges(context_id, digest.id, &source_refs)?;
    tx.record_skill_firings(skill_firing_records, &provenance)?;
    let mut invalidated_decision_ids = invalidation_rows
        .iter()
        .map(|row| row.affected_decision_id)
        .chain(tx.pending_invalidated_decisions.iter().copied())
        .collect::<Vec<_>>();
    invalidated_decision_ids.sort();
    invalidated_decision_ids.dedup();
    let last_harvest = HarvestCounts {
        goals: harvest.goals.len() as u32,
        decisions: harvest.decisions.len() as u32,
        observations: harvest.observations.len() as u32,
    };
    // Extracted outcomes commit inside the same capture transaction as everything else,
    // so a mid-capture failure strands nothing (one atomic capture). Each replaces any
    // active Outcome on its decision (single active — Invariant 9); if the harvest names
    // the same decision more than once, the last outcome wins.
    let mut outcome_by_decision: BTreeMap<DecisionId, (bool, String)> = BTreeMap::new();
    for seed in &harvest.outcomes {
        if seed.key.trim().is_empty() {
            continue;
        }
        if let Some(decision_id) = seed
            .of_decision_key
            .as_ref()
            .and_then(|k| decision_ids_by_key.get(k).copied())
        {
            outcome_by_decision.insert(decision_id, (seed.success, seed.notes.clone()));
        }
    }
    for (decision_id, (success, notes)) in outcome_by_decision {
        tx.record_outcome(context_id, decision_id, success, Some(notes))?;
    }

    let total_harvest = tx.total_counts(context_id)?;
    tx.commit()?;

    let mut committed_invalidation_rows = Vec::new();
    for decision_id in invalidated_decision_ids {
        committed_invalidation_rows.extend(store.list_invalidations(
            context_id,
            Some(decision_id),
            None,
        )?);
    }
    committed_invalidation_rows.sort_by_key(|row| row.id);
    committed_invalidation_rows.dedup_by_key(|row| row.id);
    let invalidations = committed_invalidation_rows
        .iter()
        .map(|row| invalidation_notice(store, row))
        .collect::<Result<Vec<_>>>()?;
    let report = report_with_counts(
        CaptureMode::Condensed,
        last_harvest,
        total_harvest,
        "captured structured memory",
        summary,
        report_artifacts(
            derived_items,
            provenance,
            stable,
            dedup_events,
            invalidations,
        ),
    );
    Ok(HarvestCaptureResult {
        report,
        resolutions,
    })
}

struct CaptureTxn<'a> {
    store: &'a Store,
    tx: crate::TxId,
    active: bool,
    pending_invalidated_decisions: HashSet<DecisionId>,
    current_snapshot_overrides: HashMap<EntityId, EntitySnapshotId>,
}

impl<'a> CaptureTxn<'a> {
    fn begin(store: &'a Store) -> Result<Self> {
        let tx = store.internal_db().begin().map_err(map_engine)?;
        Ok(Self {
            store,
            tx,
            active: true,
            pending_invalidated_decisions: HashSet::new(),
            current_snapshot_overrides: HashMap::new(),
        })
    }

    fn commit(mut self) -> Result<()> {
        self.store
            .internal_db()
            .commit(self.tx)
            .map_err(map_engine)?;
        self.active = false;
        Ok(())
    }

    /// Record an extracted Outcome inside the open capture transaction, so it commits
    /// atomically with the rest of the capture (no orphan rows on mid-capture failure).
    /// Replaces any active Outcome on the decision (single active — Invariant 9).
    fn record_outcome(
        &self,
        context_id: ContextId,
        decision_id: DecisionId,
        success: bool,
        notes: Option<String>,
    ) -> Result<()> {
        self.store
            .record_outcome_in_tx(self.tx, context_id, decision_id, success, notes)?;
        Ok(())
    }

    fn create_text_digest(
        &mut self,
        context_id: ContextId,
        summary: &str,
        vector_text: &str,
    ) -> Result<crate::Digest> {
        let id = DigestId::new_v7();
        let embedding_id = EmbeddingId::new_v7();
        let vector = self.store.embed_digest_text(vector_text)?;
        let row_id = self
            .store
            .internal_db()
            .insert_row(
                self.tx,
                "digests",
                HashMap::from([
                    ("id".to_string(), DbValue::Uuid(id.as_uuid())),
                    (
                        "source".to_string(),
                        DbValue::Text("claude-cli".to_string()),
                    ),
                    (
                        "session_ref".to_string(),
                        DbValue::Text("capture".to_string()),
                    ),
                    ("summary".to_string(), DbValue::Text(summary.to_string())),
                    ("kind".to_string(), DbValue::Text("text".to_string())),
                    ("vector_text".to_string(), DbValue::Vector(vector.clone())),
                    (
                        "dimension".to_string(),
                        DbValue::Int64(crate::schema::DIGEST_VECTOR_DIMENSION as i64),
                    ),
                    (
                        "context_id".to_string(),
                        DbValue::Uuid(context_id.as_uuid()),
                    ),
                    ("properties".to_string(), DbValue::Json(json!({}))),
                    ("created_at".to_string(), DbValue::TxId(self.tx)),
                ]),
            )
            .map_err(map_engine)?;
        self.execute(
            "INSERT INTO digest_embedding_handles (id, context_id, embedding_space_id, target_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $embedding_space_id, $target_id, $produced_by, $producer_system, $created_at)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(embedding_id.as_uuid())),
                (
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                ),
                (
                    "embedding_space_id".to_string(),
                    DbValue::Text(BuiltinSpaceRegistry::TEXT_DEFAULT_ID.to_string()),
                ),
                ("target_id".to_string(), DbValue::Uuid(id.as_uuid())),
                (
                    "produced_by".to_string(),
                    DbValue::Json(serde_json::to_value(EvidenceProducer::default()).map_err(
                        |err| CgError::Engine(format!("producer serialize: {err}")),
                    )?),
                ),
                ("producer_system".to_string(), DbValue::Text(String::new())),
                ("created_at".to_string(), DbValue::TxId(self.tx)),
            ]),
        )?;
        self.store
            .internal_db()
            .insert_vector(
                self.tx,
                VectorIndexRef::new("digests", "vector_text"),
                row_id,
                vector.clone(),
            )
            .map_err(map_engine)?;
        Ok(crate::Digest {
            id,
            source: "claude-cli".to_string(),
            summary: summary.to_string(),
            vector_text: Some(vector),
            dimension: crate::schema::DIGEST_VECTOR_DIMENSION,
            context_id,
            session_ref: Some("capture".to_string()),
            properties: BTreeMap::new(),
            created_at: self.tx,
        })
    }

    fn ensure_intention(
        &mut self,
        context_id: ContextId,
        description: &str,
    ) -> Result<IntentionId> {
        if let Some(id) = find_id_by_text(
            self.store,
            "intentions",
            "description",
            description,
            context_id,
        )? {
            return Ok(IntentionId::from(id));
        }
        if let Some(id) =
            find_intention_by_normalized_description(self.store, context_id, description)?
        {
            return Ok(IntentionId::from(id));
        }
        let id = IntentionId::new_v7();
        self.execute(
            "INSERT INTO intentions (id, description, status, context_id, origin, origin_ref, blueprint_id, bindings, blueprint_proposal, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $origin_ref, $blueprint_id, $bindings, $blueprint_proposal, $properties, $tags, $created_at)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(id.as_uuid())),
                ("description".to_string(), DbValue::Text(description.to_string())),
                ("status".to_string(), DbValue::Text("active".to_string())),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("origin".to_string(), DbValue::Text("human".to_string())),
                ("origin_ref".to_string(), DbValue::Null),
                ("blueprint_id".to_string(), DbValue::Null),
                ("bindings".to_string(), DbValue::Null),
                ("blueprint_proposal".to_string(), DbValue::Null),
                ("properties".to_string(), DbValue::Json(json!({}))),
                ("tags".to_string(), DbValue::Json(json!([]))),
                ("created_at".to_string(), DbValue::TxId(self.tx)),
            ]),
        )?;
        self.execute(
            "INSERT INTO intention_snapshots (id, context_id, intention_id, description, status, origin, properties, tags, valid_from, valid_to) VALUES ($id, $context_id, $intention_id, $description, $status, $origin, $properties, $tags, $valid_from, $valid_to)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(uuid::Uuid::now_v7())),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("intention_id".to_string(), DbValue::Uuid(id.as_uuid())),
                ("description".to_string(), DbValue::Text(description.to_string())),
                ("status".to_string(), DbValue::Text("active".to_string())),
                ("origin".to_string(), DbValue::Text("human".to_string())),
                ("properties".to_string(), DbValue::Json(json!({}))),
                ("tags".to_string(), DbValue::Json(json!([]))),
                ("valid_from".to_string(), DbValue::TxId(self.tx)),
                ("valid_to".to_string(), DbValue::Null),
            ]),
        )?;
        self.insert_audit("create", "intention", id.as_uuid(), context_id)?;
        Ok(id)
    }

    fn ensure_entity_with_prior_path(
        &mut self,
        context_id: ContextId,
        name: &str,
        path: &str,
        entity_type: EntityType,
    ) -> Result<EntityWrite> {
        if let Some(entity) =
            find_existing_entity_for_seed(self.store, context_id, name, path, entity_type)?
        {
            let prior_path = current_entity_path(self.store, entity.id)?
                .or_else(|| entity_path_from_properties(&entity.properties));
            return Ok(EntityWrite {
                id: entity.id,
                prior_path,
            });
        }
        let id = EntityId::new_v7();
        let properties = BTreeMap::from([("path".to_string(), json!(path))]);
        let properties_json = properties_to_json_value(&properties);
        self.execute(
            "INSERT INTO entities (id, entity_type, name, status, context_id, properties, tags, created_at) VALUES ($id, $entity_type, $name, $status, $context_id, $properties, $tags, $created_at)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(id.as_uuid())),
                ("entity_type".to_string(), DbValue::Text(entity_type_label(entity_type).to_string())),
                ("name".to_string(), DbValue::Text(name.to_string())),
                ("status".to_string(), DbValue::Text("active".to_string())),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("properties".to_string(), DbValue::Json(properties_json.clone())),
                ("tags".to_string(), DbValue::Json(json!([]))),
                ("created_at".to_string(), DbValue::TxId(self.tx)),
            ]),
        )?;
        let snapshot_id = EntitySnapshotId::new_v7();
        self.insert_entity_snapshot(EntitySnapshotInsert {
            context_id,
            entity_id: id,
            snapshot_id,
            name,
            entity_type,
            properties_json: &properties_json,
            valid_to: None,
        })?;
        self.set_current_entity_snapshot(context_id, id, snapshot_id)?;
        self.insert_audit("create", "entity", id.as_uuid(), context_id)?;
        Ok(EntityWrite {
            id,
            prior_path: None,
        })
    }

    fn create_decision(&mut self, input: CreateDecision) -> Result<Decision> {
        let id = DecisionId::new_v7();
        let based_on_snapshots = self.insert_decision_row(id, &input, DecisionStatus::Active)?;
        let precedent_ids = input.precedent_ids.clone();
        Ok(Decision {
            id,
            decision_type: input.decision_type,
            description: input.description,
            reasoning: input.reasoning,
            confidence: input.confidence,
            status: DecisionStatus::Active,
            agent_id: input.agent_id,
            context_id: input.context_id,
            tags: input.tags,
            properties: input.properties,
            created_at: self.tx,
            intention_ids: input.intention_ids,
            based_on_snapshots,
            outcome: None,
            precedent_ids,
        })
    }

    fn ensure_observation_with_invalidations(
        &mut self,
        context_id: ContextId,
        entity_id: EntityId,
        text: &str,
        state_delta: Option<(&str, &str)>,
    ) -> Result<ObservationWrite> {
        if let Some(id) = find_id_by_text(
            self.store,
            "observations",
            "observation_type",
            text,
            context_id,
        )? {
            return Ok(ObservationWrite {
                id: ObservationId::from(id),
                invalidations: Vec::new(),
            });
        }

        let id = ObservationId::new_v7();
        let state_delta = state_delta
            .map(|(field, value)| BTreeMap::from([(field.to_string(), Some(json!(value)))]))
            .unwrap_or_default();
        self.execute(
            "INSERT INTO observations (id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at) VALUES ($id, $observation_type, $observed_properties, $state_delta, $properties, $source, $entity_id, $context_id, $timestamp, $created_at)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(id.as_uuid())),
                ("observation_type".to_string(), DbValue::Text(text.to_string())),
                ("observed_properties".to_string(), DbValue::Json(json!({}))),
                ("state_delta".to_string(), DbValue::Json(state_delta_to_json(&state_delta))),
                ("properties".to_string(), DbValue::Json(json!({}))),
                ("source".to_string(), DbValue::Text("agent-session".to_string())),
                ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("timestamp".to_string(), DbValue::Timestamp(Utc::now().timestamp_millis())),
                ("created_at".to_string(), DbValue::TxId(self.tx)),
            ]),
        )?;
        let evidence = text_evidence(id, context_id);
        self.insert_evidence(&evidence, 0)?;
        self.mark_decisions_invalidated_by_delta(context_id, entity_id, &state_delta)?;
        self.merge_state_delta(context_id, entity_id, id, &state_delta)?;
        self.insert_audit(
            "record_observation",
            "observation",
            id.as_uuid(),
            context_id,
        )?;
        let invalidations = self.list_invalidations_for_observation(context_id, id)?;
        Ok(ObservationWrite { id, invalidations })
    }

    fn ensure_observation_from_seed(
        &mut self,
        context_id: ContextId,
        entity_id: EntityId,
        seed: &ObservationSeed,
    ) -> Result<ObservationWrite> {
        // Dedup by observation_type text (use seed.observation_type if set, else seed.text)
        let obs_type = seed
            .observation_type
            .as_deref()
            .unwrap_or(seed.text.as_str());
        if let Some(id) = find_id_by_text(
            self.store,
            "observations",
            "observation_type",
            obs_type,
            context_id,
        )? {
            return Ok(ObservationWrite {
                id: ObservationId::from(id),
                invalidations: Vec::new(),
            });
        }

        let id = ObservationId::new_v7();

        // Convert observed_properties from serde_json::Value to a storable form
        let observed_properties_json = match &seed.observed_properties {
            serde_json::Value::Object(_) => seed.observed_properties.clone(),
            _ => serde_json::json!({}),
        };

        // Build state_delta_map: prefer seed.state_delta; fall back to seed.path_change
        // (the arrow-value field that the old ensure_observation_with_invalidations handled).
        let state_delta_map: BTreeMap<String, Option<JsonValue>> = match &seed.state_delta {
            serde_json::Value::Object(map) if !map.contains_key("format") && !map.is_empty() => map
                .iter()
                .map(|(k, v)| (k.clone(), Some(v.clone())))
                .collect(),
            serde_json::Value::Null | serde_json::Value::Object(_) => {
                // No structured state_delta — check path_change fallback
                if let Some(path_value) = seed
                    .path_change
                    .as_deref()
                    .map(normalize_path_change_value)
                    .filter(|p| !p.is_empty())
                {
                    BTreeMap::from([("path".to_string(), Some(json!(path_value)))])
                } else {
                    BTreeMap::new()
                }
            }
            _ => BTreeMap::new(),
        };
        let state_delta_json = match &seed.state_delta {
            serde_json::Value::Object(map) if map.contains_key("format") => {
                seed.state_delta.clone()
            }
            _ => state_delta_to_json(&state_delta_map),
        };

        let source = seed
            .source
            .as_deref()
            .unwrap_or("agent-session")
            .to_string();

        self.execute(
            "INSERT INTO observations (id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at) VALUES ($id, $observation_type, $observed_properties, $state_delta, $properties, $source, $entity_id, $context_id, $timestamp, $created_at)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(id.as_uuid())),
                (
                    "observation_type".to_string(),
                    DbValue::Text(obs_type.to_string()),
                ),
                (
                    "observed_properties".to_string(),
                    DbValue::Json(observed_properties_json),
                ),
                (
                    "state_delta".to_string(),
                    DbValue::Json(state_delta_json),
                ),
                ("properties".to_string(), DbValue::Json(json!({}))),
                ("source".to_string(), DbValue::Text(source)),
                ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
                (
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                ),
                (
                    "timestamp".to_string(),
                    DbValue::Timestamp(Utc::now().timestamp_millis()),
                ),
                ("created_at".to_string(), DbValue::TxId(self.tx)),
            ]),
        )?;
        let evidence = text_evidence(id, context_id);
        self.insert_evidence(&evidence, 0)?;
        self.mark_decisions_invalidated_by_delta(context_id, entity_id, &state_delta_map)?;
        self.merge_state_delta(context_id, entity_id, id, &state_delta_map)?;
        self.insert_audit(
            "record_observation",
            "observation",
            id.as_uuid(),
            context_id,
        )?;
        let invalidations = self.list_invalidations_for_observation(context_id, id)?;
        Ok(ObservationWrite { id, invalidations })
    }

    fn mark_decisions_invalidated_by_delta(
        &mut self,
        context_id: ContextId,
        entity_id: EntityId,
        state_delta: &BTreeMap<String, Option<JsonValue>>,
    ) -> Result<()> {
        if state_delta.is_empty() {
            return Ok(());
        }
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT decision_id, snapshot_id FROM decision_basis_snapshots WHERE context_id = $context_id AND entity_id = $entity_id",
                &HashMap::from([
                    ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                    ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
                ]),
            )
            .map_err(map_engine)?;
        let mut candidates = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            let decision_id =
                DecisionId::from(required_uuid_column(row, &result.columns, "decision_id")?);
            let snapshot_id =
                EntitySnapshotId::from(required_uuid_column(row, &result.columns, "snapshot_id")?);
            candidates.push((decision_id, snapshot_id));
        }
        let decision_ids = candidates
            .iter()
            .map(|(decision_id, _)| *decision_id)
            .collect::<Vec<_>>();
        let snapshot_ids = candidates
            .iter()
            .map(|(_, snapshot_id)| *snapshot_id)
            .collect::<Vec<_>>();
        let statuses = self.decision_statuses(context_id, &decision_ids)?;
        let snapshots = self.entity_snapshot_data_batch(context_id, &snapshot_ids)?;
        let edge_properties =
            self.based_on_edge_properties_batch(context_id, entity_id, &decision_ids)?;
        for (decision_id, snapshot_id) in candidates {
            if statuses.get(&decision_id) != Some(&DecisionStatus::Active) {
                continue;
            }
            let Some(snapshot_data) = snapshots.get(&snapshot_id) else {
                continue;
            };
            let edge_properties = edge_properties
                .get(&decision_id)
                .cloned()
                .unwrap_or_else(|| json!({}));
            if basis_snapshot_differs_from_delta(snapshot_data, &edge_properties, state_delta) {
                self.pending_invalidated_decisions.insert(decision_id);
            }
        }
        Ok(())
    }

    fn will_invalidate_decision(&self, decision_id: DecisionId) -> bool {
        self.pending_invalidated_decisions.contains(&decision_id)
    }

    fn precedent_ids_if_prior_stays_current(&self, prior: DecisionId) -> Result<Vec<DecisionId>> {
        if self.decision_status(prior)? == Some(DecisionStatus::Active)
            && !self.will_invalidate_decision(prior)
        {
            Ok(vec![prior])
        } else {
            Ok(Vec::new())
        }
    }

    fn ensure_semantic_conflict_observation(
        &mut self,
        context_id: ContextId,
        entity_id: Option<EntityId>,
        prior_decision_id: DecisionId,
        prior_text: &str,
        candidate_text: &str,
    ) -> Result<ObservationId> {
        let entity = match entity_id {
            Some(entity) => entity,
            None => {
                self.ensure_entity_with_prior_path(
                    context_id,
                    "decision conflict",
                    &format!("decision:{}", prior_decision_id.as_uuid()),
                    EntityType::Document,
                )?
                .id
            }
        };
        let text = format!("model classified decision conflict: {prior_text} -> {candidate_text}");
        Ok(self
            .ensure_observation_with_invalidations(context_id, entity, &text, None)?
            .id)
    }

    fn invalidate_decision(
        &mut self,
        decision_id: DecisionId,
        trigger_observation_id: ObservationId,
        reason: String,
    ) -> Result<()> {
        let status = self
            .decision_status(decision_id)?
            .ok_or(CgError::DecisionNotFound { id: decision_id })?;
        if status != DecisionStatus::Active {
            return Ok(());
        }
        let context_id = self
            .decision_context_id(decision_id)?
            .ok_or(CgError::DecisionNotFound { id: decision_id })?;
        let threatened_intention_ids = self.decision_intention_ids(context_id, decision_id)?;
        let replacement_exists = threatened_intention_ids.iter().any(|intention_id| {
            self.active_decision_for_intention(*intention_id, decision_id)
                .is_ok_and(|row| row.is_some())
        });
        self.execute(
            "UPDATE decisions SET status = $new_status WHERE id = $id AND status = $old_status",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(decision_id.as_uuid())),
                (
                    "new_status".to_string(),
                    DbValue::Text("invalidated".to_string()),
                ),
                (
                    "old_status".to_string(),
                    DbValue::Text("active".to_string()),
                ),
            ]),
        )?;
        let diff = empty_basis_diff(self.tx);
        self.insert_invalidation_row(InvalidationInsert {
            context_id,
            decision_id,
            trigger_observation_id,
            reason: Some(reason),
            basis_diff: diff,
            cascade_from: None,
            severity: InvalidationSeverity::Warning,
            threatened_intention_ids,
            intention_now_un_served: !replacement_exists,
        })?;
        Ok(())
    }

    fn list_invalidations(
        &self,
        context_id: ContextId,
        decision_id: Option<DecisionId>,
    ) -> Result<Vec<crate::Invalidation>> {
        let mut sql = "SELECT id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id FROM invalidations WHERE context_id = $context_id".to_string();
        let mut params = HashMap::from([(
            "context_id".to_string(),
            DbValue::Uuid(context_id.as_uuid()),
        )]);
        if let Some(decision_id) = decision_id {
            sql.push_str(" AND affected_decision_id = $affected_decision_id");
            params.insert(
                "affected_decision_id".to_string(),
                DbValue::Uuid(decision_id.as_uuid()),
            );
        }
        let result = self
            .store
            .internal_db()
            .execute_in_tx(self.tx, &sql, &params)
            .map_err(map_engine)?;
        decode_invalidation_rows(&result.columns, &result.rows)
    }

    fn list_invalidations_for_observation(
        &self,
        context_id: ContextId,
        observation_id: ObservationId,
    ) -> Result<Vec<crate::Invalidation>> {
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id FROM invalidations WHERE context_id = $context_id AND trigger_observation_id = $trigger_observation_id",
                &HashMap::from([
                    ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                    (
                        "trigger_observation_id".to_string(),
                        DbValue::Uuid(observation_id.as_uuid()),
                    ),
                ]),
            )
            .map_err(map_engine)?;
        decode_invalidation_rows(&result.columns, &result.rows)
    }

    fn supersede_decision(&mut self, old: DecisionId, mut new: CreateDecision) -> Result<Decision> {
        let existing_status = self
            .decision_status(old)?
            .ok_or(CgError::DecisionNotFound { id: old })?;
        if existing_status != DecisionStatus::Active {
            return Err(CgError::IllegalTransition {
                from: existing_status,
                to: DecisionStatus::Superseded,
            });
        }
        self.execute(
            "UPDATE decisions SET status = $status WHERE id = $id",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(old.as_uuid())),
                (
                    "status".to_string(),
                    DbValue::Text("superseded".to_string()),
                ),
            ]),
        )?;
        if !new.precedent_ids.contains(&old) {
            new.precedent_ids.push(old);
        }
        let decision = self.create_decision(new)?;
        self.insert_edge(
            decision.id.as_uuid(),
            old.as_uuid(),
            "SUPERSEDES",
            decision.context_id,
            None,
        )?;
        self.insert_audit("supersede", "decision", old.as_uuid(), decision.context_id)?;
        Ok(decision)
    }

    /// Read the catalog grouping memberships of every intention already stored in this
    /// context (those that carry a `blueprint_catalog_id`), so Stage-3 grouping can get-or-create.
    fn intention_catalog_entries(
        &self,
        context_id: ContextId,
    ) -> Result<Vec<IntentionCatalogEntry>> {
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT id, description, blueprint_catalog_id FROM intentions WHERE context_id = $context_id AND blueprint_catalog_id IS NOT NULL",
                &HashMap::from([(
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                )]),
            )
            .map_err(map_engine)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            let intention_id = IntentionId::from(required_uuid_column(row, &result.columns, "id")?);
            let description = required_text_column(row, &result.columns, "description")?;
            let catalog_id = required_text_column(row, &result.columns, "blueprint_catalog_id")?;
            out.push(IntentionCatalogEntry {
                intention_id,
                normalized_description: normalize_intention_description(&description),
                catalog_id,
            });
        }
        Ok(out)
    }

    /// Assign the catalog grouping handle to an intention (an additive grouping link, not an
    /// edit of the captured description/status — Invariant 3).
    fn set_intention_catalog_id(
        &self,
        context_id: ContextId,
        intention_id: IntentionId,
        catalog_id: &str,
    ) -> Result<()> {
        self.execute(
            "UPDATE intentions SET blueprint_catalog_id = $catalog_id WHERE id = $id AND context_id = $context_id",
            HashMap::from([
                ("catalog_id".to_string(), DbValue::Text(catalog_id.to_string())),
                ("id".to_string(), DbValue::Uuid(intention_id.as_uuid())),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
            ]),
        )?;
        Ok(())
    }

    /// Insert a decision-to-decision relation edge, pre-checking the UNIQUE
    /// (source, target, type) constraint so a re-classified relation does not raise the
    /// hard-error backstop that cg's raw `insert_edge` relies on.
    fn ensure_decision_relation_edge(
        &mut self,
        from: DecisionId,
        to: DecisionId,
        edge_type: &str,
        context_id: ContextId,
    ) -> Result<()> {
        let existing = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT id FROM edges WHERE source_id = $source_id AND target_id = $target_id AND edge_type = $edge_type",
                &HashMap::from([
                    ("source_id".to_string(), DbValue::Uuid(from.as_uuid())),
                    ("target_id".to_string(), DbValue::Uuid(to.as_uuid())),
                    ("edge_type".to_string(), DbValue::Text(edge_type.to_string())),
                ]),
            )
            .map_err(map_engine)?;
        if existing.rows.is_empty() {
            self.insert_edge(from.as_uuid(), to.as_uuid(), edge_type, context_id, None)?;
        }
        Ok(())
    }

    fn decision_status(&self, decision_id: DecisionId) -> Result<Option<DecisionStatus>> {
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT status FROM decisions WHERE id = $id",
                &HashMap::from([("id".to_string(), DbValue::Uuid(decision_id.as_uuid()))]),
            )
            .map_err(map_engine)?;
        result
            .rows
            .first()
            .map(|row| match row.first() {
                Some(DbValue::Text(status)) => decision_status_from_label(status).map(Some),
                other => Err(CgError::Engine(format!(
                    "decision status query returned {other:?}"
                ))),
            })
            .unwrap_or(Ok(None))
    }

    fn decision_statuses(
        &self,
        context_id: ContextId,
        decision_ids: &[DecisionId],
    ) -> Result<HashMap<DecisionId, DecisionStatus>> {
        if decision_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let mut params = HashMap::from([(
            "context_id".to_string(),
            DbValue::Uuid(context_id.as_uuid()),
        )]);
        let id_list = decision_id_list("decision", decision_ids, &mut params);
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                &format!(
                    "SELECT id, status FROM decisions WHERE context_id = $context_id AND id IN ({id_list})"
                ),
                &params,
            )
            .map_err(map_engine)?;
        let mut out = HashMap::new();
        for row in &result.rows {
            out.insert(
                DecisionId::from(required_uuid_column(row, &result.columns, "id")?),
                decision_status_from_label(&required_text_column(row, &result.columns, "status")?)?,
            );
        }
        Ok(out)
    }

    fn entity_snapshot_data_batch(
        &self,
        context_id: ContextId,
        snapshot_ids: &[EntitySnapshotId],
    ) -> Result<HashMap<EntitySnapshotId, JsonValue>> {
        if snapshot_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let mut params = HashMap::from([(
            "context_id".to_string(),
            DbValue::Uuid(context_id.as_uuid()),
        )]);
        let id_list = snapshot_id_list("snapshot", snapshot_ids, &mut params);
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                &format!(
                    "SELECT id, data FROM entity_snapshots WHERE context_id = $context_id AND id IN ({id_list})"
                ),
                &params,
            )
            .map_err(map_engine)?;
        let mut out = HashMap::new();
        for row in &result.rows {
            out.insert(
                EntitySnapshotId::from(required_uuid_column(row, &result.columns, "id")?),
                required_json_column(row, &result.columns, "data")?,
            );
        }
        Ok(out)
    }

    fn based_on_edge_properties_batch(
        &self,
        context_id: ContextId,
        entity_id: EntityId,
        decision_ids: &[DecisionId],
    ) -> Result<HashMap<DecisionId, JsonValue>> {
        if decision_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let mut params = HashMap::from([
            (
                "context_id".to_string(),
                DbValue::Uuid(context_id.as_uuid()),
            ),
            ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
            (
                "edge_type".to_string(),
                DbValue::Text("BASED_ON".to_string()),
            ),
        ]);
        let id_list = decision_id_list("decision", decision_ids, &mut params);
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                &format!(
                    "SELECT source_id, properties FROM edges WHERE context_id = $context_id AND target_id = $entity_id AND edge_type = $edge_type AND source_id IN ({id_list})"
                ),
                &params,
            )
            .map_err(map_engine)?;
        let mut out = HashMap::new();
        for row in &result.rows {
            out.entry(DecisionId::from(required_uuid_column(
                row,
                &result.columns,
                "source_id",
            )?))
            .or_insert(required_json_column(row, &result.columns, "properties")?);
        }
        Ok(out)
    }

    fn total_counts(&self, context_id: ContextId) -> Result<HarvestCounts> {
        let count_table = |table: &str| -> Result<u32> {
            let result = self
                .store
                .internal_db()
                .execute_in_tx(
                    self.tx,
                    &format!("SELECT COUNT(*) AS n FROM {table} WHERE context_id = $context_id"),
                    &HashMap::from([(
                        "context_id".to_string(),
                        DbValue::Uuid(context_id.as_uuid()),
                    )]),
                )
                .map_err(map_engine)?;
            count_from_result(&result.columns, &result.rows)
        };
        Ok(HarvestCounts {
            goals: count_table("intentions")?,
            decisions: count_table("decisions")?,
            observations: count_table("observations")?,
        })
    }

    fn add_provenance_edges(
        &mut self,
        context_id: ContextId,
        digest_id: DigestId,
        sources: &[(&str, uuid::Uuid)],
    ) -> Result<Vec<ExtractionProvenance>> {
        let mut out = Vec::new();
        for (item_id, source_id) in sources {
            self.insert_edge(
                *source_id,
                digest_id.as_uuid(),
                "EXTRACTED_FROM",
                context_id,
                None,
            )?;
            out.push(ExtractionProvenance {
                item_id: (*item_id).to_string(),
                digest_id,
                relationship: "EXTRACTED_FROM".to_string(),
            });
        }
        Ok(out)
    }

    fn record_skill_firings(
        &mut self,
        skill_firing_records: &[crate::skill_store::SkillFiringRecord],
        provenance_edges: &[ExtractionProvenance],
    ) -> Result<()> {
        if skill_firing_records.is_empty() {
            return Ok(());
        }
        let mut capture_unit_ids = provenance_edges
            .iter()
            .map(|edge| edge.digest_id.as_uuid().to_string())
            .collect::<Vec<_>>();
        capture_unit_ids.sort();
        capture_unit_ids.dedup();

        for firing in skill_firing_records {
            for capture_unit_id in &capture_unit_ids {
                self.insert_skill_firing_trace(firing, capture_unit_id)?;
            }
        }
        Ok(())
    }

    fn insert_skill_firing_trace(
        &self,
        firing: &crate::skill_store::SkillFiringRecord,
        capture_id: &str,
    ) -> Result<()> {
        let (effect_entity_key, effect_entity_name, effect_from_type, effect_to_type) =
            match &firing.effect {
                Some(effect) => (
                    DbValue::Text(effect.entity_key.clone()),
                    DbValue::Text(effect.entity_name.clone()),
                    DbValue::Text(effect.from_type.clone()),
                    DbValue::Text(effect.to_type.clone()),
                ),
                None => (DbValue::Null, DbValue::Null, DbValue::Null, DbValue::Null),
            };
        self.execute(
            "INSERT INTO cg_skill_firing_trace (id, skill_id, capture_id, correct, effect_entity_key, effect_entity_name, effect_from_type, effect_to_type, created_at) VALUES ($id, $skill_id, $capture_id, $correct, $effect_entity_key, $effect_entity_name, $effect_from_type, $effect_to_type, $created_at)",
            HashMap::from([
                (
                    "id".to_string(),
                    DbValue::Text(uuid::Uuid::new_v4().to_string()),
                ),
                ("skill_id".to_string(), DbValue::Text(firing.skill_id.clone())),
                ("capture_id".to_string(), DbValue::Text(capture_id.to_string())),
                ("correct".to_string(), DbValue::Null),
                ("effect_entity_key".to_string(), effect_entity_key),
                ("effect_entity_name".to_string(), effect_entity_name),
                ("effect_from_type".to_string(), effect_from_type),
                ("effect_to_type".to_string(), effect_to_type),
                ("created_at".to_string(), DbValue::TxId(self.tx)),
            ]),
        )?;
        Ok(())
    }

    fn insert_decision_row(
        &mut self,
        id: DecisionId,
        input: &CreateDecision,
        status: DecisionStatus,
    ) -> Result<Vec<EntitySnapshotId>> {
        if input.intention_ids.is_empty() {
            return Err(CgError::MissingIntention);
        }
        let mut basis_snapshot_ids = input.based_on_snapshots.clone();
        for entity_id in &input.based_on_entity_ids {
            let snapshot_id = if let Some(snapshot_id) =
                self.current_snapshot_overrides.get(entity_id).copied()
            {
                snapshot_id
            } else {
                self.current_entity_snapshot_id(input.context_id, *entity_id)?
                    .ok_or(CgError::NotFound {
                        id: Some(entity_id.as_uuid().to_string()),
                        missing: Some("entity_snapshot"),
                    })?
            };
            if !basis_snapshot_ids.contains(&snapshot_id) {
                basis_snapshot_ids.push(snapshot_id);
            }
        }
        let reasoning_json = serde_json::to_value(&input.reasoning)
            .map_err(|err| CgError::Engine(format!("decision reasoning serialize: {err}")))?;
        let tags_json = JsonValue::Array(
            input
                .tags
                .iter()
                .map(|tag| JsonValue::String(tag.clone()))
                .collect(),
        );
        self.execute(
            "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(id.as_uuid())),
                ("decision_type".to_string(), DbValue::Text(input.decision_type.clone())),
                ("description".to_string(), DbValue::Text(input.description.clone())),
                ("reasoning".to_string(), DbValue::Json(reasoning_json)),
                (
                    "confidence".to_string(),
                    input
                        .confidence
                        .map(|value| DbValue::Float64(value as f64))
                        .unwrap_or(DbValue::Null),
                ),
                (
                    "status".to_string(),
                    DbValue::Text(decision_status_to_label(status).to_string()),
                ),
                (
                    "agent_id".to_string(),
                    input
                        .agent_id
                        .clone()
                        .map(DbValue::Text)
                        .unwrap_or(DbValue::Null),
                ),
                ("context_id".to_string(), DbValue::Uuid(input.context_id.as_uuid())),
                (
                    "properties".to_string(),
                    DbValue::Json(properties_to_json_value(&input.properties)),
                ),
                ("tags".to_string(), DbValue::Json(tags_json)),
                ("created_at".to_string(), DbValue::TxId(self.tx)),
            ]),
        )?;
        let mut basis_edge_entities = HashSet::new();
        for (basis_order, snapshot_id) in basis_snapshot_ids.iter().enumerate() {
            let snapshot = self.entity_snapshot_by_id(input.context_id, *snapshot_id)?;
            self.execute(
                "INSERT INTO decision_basis_snapshots (id, context_id, decision_id, entity_id, snapshot_id, basis_order, created_at) VALUES ($id, $context_id, $decision_id, $entity_id, $snapshot_id, $basis_order, $created_at)",
                HashMap::from([
                    ("id".to_string(), DbValue::Uuid(decision_basis_row_id(id, basis_order))),
                    ("context_id".to_string(), DbValue::Uuid(input.context_id.as_uuid())),
                    ("decision_id".to_string(), DbValue::Uuid(id.as_uuid())),
                    ("entity_id".to_string(), DbValue::Uuid(snapshot.entity_id.as_uuid())),
                    ("snapshot_id".to_string(), DbValue::Uuid(snapshot.id.as_uuid())),
                    ("basis_order".to_string(), DbValue::Int64(basis_order as i64)),
                    ("created_at".to_string(), DbValue::TxId(self.tx)),
                ]),
            )?;
            if basis_edge_entities.insert(snapshot.entity_id) {
                let basis_fields = input
                    .basis_fields
                    .as_ref()
                    .and_then(|fields| fields.get(&snapshot.entity_id))
                    .map(Vec::as_slice);
                self.insert_edge(
                    id.as_uuid(),
                    snapshot.entity_id.as_uuid(),
                    "BASED_ON",
                    input.context_id,
                    Some(basis_snapshot_edge_properties(&snapshot, basis_fields)),
                )?;
            }
        }
        for intention_id in &input.intention_ids {
            self.insert_edge(
                id.as_uuid(),
                intention_id.as_uuid(),
                "SERVES",
                input.context_id,
                None,
            )?;
        }
        for precedent_id in &input.precedent_ids {
            self.insert_edge(
                id.as_uuid(),
                precedent_id.as_uuid(),
                "CITES",
                input.context_id,
                None,
            )?;
        }
        self.insert_audit("create", "decision", id.as_uuid(), input.context_id)?;
        Ok(basis_snapshot_ids)
    }

    fn insert_evidence(&self, evidence: &EvidenceRef, supply_order: i64) -> Result<()> {
        let captured_at = evidence.captured_at.unwrap_or_else(Utc::now);
        self.execute(
            "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, mime_type, content_hash, captured_at, captured_at_nanos, byte_range, time_range_ms, frame_index, region, producer, text_projection, derived_from, retention_status, media_retention_until, media_retention_until_nanos, properties, supply_order, vector_text, vector_vision) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $mime_type, $content_hash, $captured_at, $captured_at_nanos, $byte_range, $time_range_ms, $frame_index, $region, $producer, $text_projection, $derived_from, $retention_status, $media_retention_until, $media_retention_until_nanos, $properties, $supply_order, $vector_text, $vector_vision)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(evidence.id.as_uuid())),
                (
                    "observation_id".to_string(),
                    DbValue::Uuid(evidence.observation_id.as_uuid()),
                ),
                ("context_id".to_string(), DbValue::Uuid(evidence.context_id.as_uuid())),
                ("kind".to_string(), DbValue::Text(evidence_kind_label(evidence.kind).to_string())),
                ("source_ref".to_string(), DbValue::Text(evidence.source_ref.clone())),
                (
                    "mime_type".to_string(),
                    evidence.mime_type.clone().map(DbValue::Text).unwrap_or(DbValue::Null),
                ),
                (
                    "content_hash".to_string(),
                    evidence
                        .content_hash
                        .clone()
                        .map(DbValue::Text)
                        .unwrap_or(DbValue::Null),
                ),
                ("captured_at".to_string(), DbValue::Timestamp(captured_at.timestamp_millis())),
                (
                    "captured_at_nanos".to_string(),
                    DbValue::Int64(captured_at.timestamp_subsec_nanos() as i64),
                ),
                ("byte_range".to_string(), optional_pair_json(&evidence.byte_range)),
                (
                    "time_range_ms".to_string(),
                    optional_pair_json(&evidence.time_range_ms),
                ),
                (
                    "frame_index".to_string(),
                    evidence
                        .frame_index
                        .map(|value| DbValue::Int64(value as i64))
                        .unwrap_or(DbValue::Null),
                ),
                (
                    "region".to_string(),
                    evidence
                        .region
                        .clone()
                        .map(DbValue::Json)
                        .unwrap_or(DbValue::Null),
                ),
                (
                    "producer".to_string(),
                    DbValue::Json(serde_json::to_value(&evidence.producer).map_err(|err| {
                        CgError::Engine(format!("evidence producer serialize: {err}"))
                    })?),
                ),
                (
                    "text_projection".to_string(),
                    evidence
                        .text_projection
                        .clone()
                        .map(DbValue::Text)
                        .unwrap_or(DbValue::Null),
                ),
                (
                    "derived_from".to_string(),
                    evidence
                        .derived_from
                        .map(|id| DbValue::Uuid(id.as_uuid()))
                        .unwrap_or(DbValue::Null),
                ),
                (
                    "retention_status".to_string(),
                    DbValue::Text(retention_status_label(evidence.retention_status).to_string()),
                ),
                (
                    "media_retention_until".to_string(),
                    evidence
                        .media_retention_until
                        .map(|value| DbValue::Timestamp(value.timestamp_millis()))
                        .unwrap_or(DbValue::Null),
                ),
                (
                    "media_retention_until_nanos".to_string(),
                    evidence
                        .media_retention_until
                        .map(|value| DbValue::Int64(value.timestamp_subsec_nanos() as i64))
                        .unwrap_or(DbValue::Null),
                ),
                (
                    "properties".to_string(),
                    DbValue::Json(properties_to_json_value(&evidence.properties)),
                ),
                ("supply_order".to_string(), DbValue::Int64(supply_order)),
                ("vector_text".to_string(), DbValue::Null),
                ("vector_vision".to_string(), DbValue::Null),
            ]),
        )?;
        Ok(())
    }

    fn merge_state_delta(
        &mut self,
        context_id: ContextId,
        entity_id: EntityId,
        observation_id: ObservationId,
        state_delta: &BTreeMap<String, Option<JsonValue>>,
    ) -> Result<Option<EntitySnapshotId>> {
        if state_delta.is_empty() {
            return Ok(None);
        }
        let current = self.current_entity_snapshot(context_id, entity_id)?;
        let mut new_properties = current
            .as_ref()
            .map(|snapshot| snapshot.properties.clone())
            .unwrap_or_default();
        for (field, value) in state_delta {
            if let Some(value) = value {
                new_properties.insert(field.clone(), value.clone());
            } else {
                new_properties.remove(field);
            }
        }
        if current
            .as_ref()
            .is_some_and(|snapshot| snapshot.properties == new_properties)
        {
            return Ok(None);
        }
        let (entity_type, entity_name) =
            self.entity_identity(context_id, entity_id, current.as_ref())?;
        if let Some(current) = &current {
            self.execute(
                "UPDATE entity_snapshots SET valid_to = $valid_to WHERE id = $id AND valid_to IS NULL",
                HashMap::from([
                    ("id".to_string(), DbValue::Uuid(current.id.as_uuid())),
                    ("valid_to".to_string(), DbValue::TxId(self.tx)),
                ]),
            )?;
        }
        let snapshot_id = EntitySnapshotId::new_v7();
        let properties_json = properties_to_json_value(&new_properties);
        self.insert_entity_snapshot(EntitySnapshotInsert {
            context_id,
            entity_id,
            snapshot_id,
            name: &entity_name,
            entity_type,
            properties_json: &properties_json,
            valid_to: None,
        })?;
        self.execute(
            "INSERT INTO entity_snapshot_observations (id, context_id, entity_id, snapshot_id, observation_id, observation_order, created_at) VALUES ($id, $context_id, $entity_id, $snapshot_id, $observation_id, $observation_order, $created_at)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(uuid::Uuid::now_v7())),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
                ("snapshot_id".to_string(), DbValue::Uuid(snapshot_id.as_uuid())),
                (
                    "observation_id".to_string(),
                    DbValue::Uuid(observation_id.as_uuid()),
                ),
                ("observation_order".to_string(), DbValue::Int64(0)),
                ("created_at".to_string(), DbValue::TxId(self.tx)),
            ]),
        )?;
        if let Some(current) = current {
            self.execute(
                "INSERT INTO entity_snapshot_links (id, context_id, entity_id, predecessor_snapshot_id, successor_snapshot_id, created_at) VALUES ($id, $context_id, $entity_id, $predecessor_snapshot_id, $successor_snapshot_id, $created_at)",
                HashMap::from([
                    ("id".to_string(), DbValue::Uuid(uuid::Uuid::now_v7())),
                    ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                    ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
                    (
                        "predecessor_snapshot_id".to_string(),
                        DbValue::Uuid(current.id.as_uuid()),
                    ),
                    (
                        "successor_snapshot_id".to_string(),
                        DbValue::Uuid(snapshot_id.as_uuid()),
                    ),
                    ("created_at".to_string(), DbValue::TxId(self.tx)),
                ]),
            )?;
        }
        self.set_current_entity_snapshot(context_id, entity_id, snapshot_id)?;
        self.current_snapshot_overrides
            .insert(entity_id, snapshot_id);
        Ok(Some(snapshot_id))
    }

    fn insert_entity_snapshot(&self, input: EntitySnapshotInsert<'_>) -> Result<()> {
        let snapshot_data = json!({
            "entity_id": input.entity_id.as_uuid().to_string(),
            "entity_type": entity_type_label(input.entity_type),
            "name": input.name,
            "properties": input.properties_json,
        });
        self.execute(
            "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(input.snapshot_id.as_uuid())),
                ("context_id".to_string(), DbValue::Uuid(input.context_id.as_uuid())),
                ("entity_id".to_string(), DbValue::Uuid(input.entity_id.as_uuid())),
                ("data".to_string(), DbValue::Json(snapshot_data)),
                ("valid_from".to_string(), DbValue::TxId(self.tx)),
                (
                    "valid_to".to_string(),
                    input.valid_to.map(DbValue::TxId).unwrap_or(DbValue::Null),
                ),
            ]),
        )?;
        Ok(())
    }

    fn set_current_entity_snapshot(
        &self,
        context_id: ContextId,
        entity_id: EntityId,
        snapshot_id: EntitySnapshotId,
    ) -> Result<()> {
        let updated = self.execute(
            "UPDATE entity_current_snapshots SET snapshot_id = $snapshot_id, updated_at = $updated_at WHERE context_id = $context_id AND entity_id = $entity_id",
            HashMap::from([
                ("snapshot_id".to_string(), DbValue::Uuid(snapshot_id.as_uuid())),
                ("updated_at".to_string(), DbValue::TxId(self.tx)),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
            ]),
        )?;
        if updated == 0 {
            self.execute(
                "INSERT INTO entity_current_snapshots (context_id, entity_id, snapshot_id, updated_at) VALUES ($context_id, $entity_id, $snapshot_id, $updated_at)",
                HashMap::from([
                    ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                    ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
                    ("snapshot_id".to_string(), DbValue::Uuid(snapshot_id.as_uuid())),
                    ("updated_at".to_string(), DbValue::TxId(self.tx)),
                ]),
            )?;
        }
        Ok(())
    }

    fn current_entity_snapshot_id(
        &self,
        context_id: ContextId,
        entity_id: EntityId,
    ) -> Result<Option<EntitySnapshotId>> {
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT snapshot_id FROM entity_current_snapshots WHERE context_id = $context_id AND entity_id = $entity_id",
                &HashMap::from([
                    ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                    ("entity_id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
                ]),
            )
            .map_err(map_engine)?;
        Ok(result.rows.first().and_then(|row| match row.first() {
            Some(DbValue::Uuid(id)) => Some(EntitySnapshotId::from(*id)),
            _ => None,
        }))
    }

    fn current_entity_snapshot(
        &self,
        context_id: ContextId,
        entity_id: EntityId,
    ) -> Result<Option<TxEntitySnapshot>> {
        let Some(snapshot_id) = self.current_entity_snapshot_id(context_id, entity_id)? else {
            return Ok(None);
        };
        self.entity_snapshot_by_id_optional(context_id, snapshot_id)
    }

    fn entity_snapshot_by_id(
        &self,
        context_id: ContextId,
        snapshot_id: EntitySnapshotId,
    ) -> Result<TxEntitySnapshot> {
        self.entity_snapshot_by_id_optional(context_id, snapshot_id)?
            .ok_or(CgError::NotFound {
                id: Some(snapshot_id.as_uuid().to_string()),
                missing: Some("entity_snapshot"),
            })
    }

    fn entity_snapshot_by_id_optional(
        &self,
        context_id: ContextId,
        snapshot_id: EntitySnapshotId,
    ) -> Result<Option<TxEntitySnapshot>> {
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT id, entity_id, data, valid_from, valid_to FROM entity_snapshots WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                    ("id".to_string(), DbValue::Uuid(snapshot_id.as_uuid())),
                ]),
            )
            .map_err(map_engine)?;
        result
            .rows
            .first()
            .map(|row| decode_tx_entity_snapshot(&result.columns, row))
            .transpose()
    }

    fn entity_identity(
        &self,
        context_id: ContextId,
        entity_id: EntityId,
        current: Option<&TxEntitySnapshot>,
    ) -> Result<(EntityType, String)> {
        if let Some(current) = current {
            return Ok((current.entity_type, current.name.clone()));
        }
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT entity_type, name FROM entities WHERE context_id = $context_id AND id = $id",
                &HashMap::from([
                    ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                    ("id".to_string(), DbValue::Uuid(entity_id.as_uuid())),
                ]),
            )
            .map_err(map_engine)?;
        let row = result.rows.first().ok_or(CgError::NotFound {
            id: Some(entity_id.as_uuid().to_string()),
            missing: Some("entity"),
        })?;
        match row.as_slice() {
            [DbValue::Text(entity_type), DbValue::Text(name)] => {
                Ok((entity_type_from_label(entity_type)?, name.clone()))
            }
            other => Err(CgError::Engine(format!(
                "entity identity query returned {other:?}"
            ))),
        }
    }

    fn insert_invalidation_row(&self, input: InvalidationInsert) -> Result<()> {
        self.execute(
            "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $reason, $basis_diff, $cascade_from, $severity, $threatened_intention_ids, $intention_now_un_served, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(InvalidationId::from(input.decision_id.as_uuid()).as_uuid())),
                ("status".to_string(), DbValue::Text("pending".to_string())),
                ("context_id".to_string(), DbValue::Uuid(input.context_id.as_uuid())),
                (
                    "affected_decision_id".to_string(),
                    DbValue::Uuid(input.decision_id.as_uuid()),
                ),
                (
                    "trigger_observation_id".to_string(),
                    DbValue::Uuid(input.trigger_observation_id.as_uuid()),
                ),
                (
                    "reason".to_string(),
                    input.reason.map(DbValue::Text).unwrap_or(DbValue::Null),
                ),
                (
                    "basis_diff".to_string(),
                    DbValue::Json(serde_json::to_value(input.basis_diff).map_err(|err| {
                        CgError::Engine(format!("basis diff serialize: {err}"))
                    })?),
                ),
                (
                    "cascade_from".to_string(),
                    input
                        .cascade_from
                        .map(|id| DbValue::Uuid(id.as_uuid()))
                        .unwrap_or(DbValue::Null),
                ),
                (
                    "severity".to_string(),
                    DbValue::Text(invalidation_severity_label(input.severity).to_string()),
                ),
                (
                    "threatened_intention_ids".to_string(),
                    DbValue::Json(serde_json::to_value(input.threatened_intention_ids).map_err(
                        |err| CgError::Engine(format!("threatened intentions serialize: {err}")),
                    )?),
                ),
                (
                    "intention_now_un_served".to_string(),
                    DbValue::Bool(input.intention_now_un_served),
                ),
                ("detected_at".to_string(), DbValue::TxId(self.tx)),
                ("resolved_at".to_string(), DbValue::Null),
                ("resolution_action".to_string(), DbValue::Null),
                ("resolution_decision_id".to_string(), DbValue::Null),
            ]),
        )?;
        Ok(())
    }

    fn decision_context_id(&self, decision_id: DecisionId) -> Result<Option<ContextId>> {
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT context_id FROM decisions WHERE id = $id",
                &HashMap::from([("id".to_string(), DbValue::Uuid(decision_id.as_uuid()))]),
            )
            .map_err(map_engine)?;
        Ok(result.rows.first().and_then(|row| match row.first() {
            Some(DbValue::Uuid(id)) => Some(ContextId::from(*id)),
            _ => None,
        }))
    }

    fn decision_intention_ids(
        &self,
        context_id: ContextId,
        decision_id: DecisionId,
    ) -> Result<Vec<IntentionId>> {
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT target_id FROM edges WHERE context_id = $context_id AND source_id = $decision_id AND edge_type = $edge_type",
                &HashMap::from([
                    ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                    ("decision_id".to_string(), DbValue::Uuid(decision_id.as_uuid())),
                    ("edge_type".to_string(), DbValue::Text("SERVES".to_string())),
                ]),
            )
            .map_err(map_engine)?;
        Ok(result
            .rows
            .iter()
            .filter_map(|row| match row.first() {
                Some(DbValue::Uuid(id)) => Some(IntentionId::from(*id)),
                _ => None,
            })
            .collect())
    }

    fn active_decision_for_intention(
        &self,
        intention_id: IntentionId,
        excluded_decision_id: DecisionId,
    ) -> Result<Option<String>> {
        let result = self
            .store
            .internal_db()
            .execute_in_tx(
                self.tx,
                "SELECT decisions.description FROM edges INNER JOIN decisions ON decisions.id = edges.source_id WHERE edges.target_id = $intention_id AND edges.edge_type = $edge_type AND decisions.status = $status AND decisions.id <> $excluded_decision_id ORDER BY decisions.created_at DESC LIMIT 1",
                &HashMap::from([
                    ("intention_id".to_string(), DbValue::Uuid(intention_id.as_uuid())),
                    ("edge_type".to_string(), DbValue::Text("SERVES".to_string())),
                    ("status".to_string(), DbValue::Text("active".to_string())),
                    (
                        "excluded_decision_id".to_string(),
                        DbValue::Uuid(excluded_decision_id.as_uuid()),
                    ),
                ]),
            )
            .map_err(map_engine)?;
        Ok(result.rows.first().and_then(|row| match row.first() {
            Some(DbValue::Text(text)) => Some(text.clone()),
            _ => None,
        }))
    }

    fn insert_audit(
        &self,
        operation: &str,
        target_kind: &str,
        target_id: uuid::Uuid,
        context_id: ContextId,
    ) -> Result<()> {
        self.execute(
            "INSERT INTO audit_log (id, txid, operation, actor, reason, target_kind, target_id, context_id, payload) VALUES ($id, $txid, $operation, $actor, $reason, $target_kind, $target_id, $context_id, $payload)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(AuditId::new_v7().as_uuid())),
                ("txid".to_string(), DbValue::TxId(self.tx)),
                ("operation".to_string(), DbValue::Text(operation.to_string())),
                ("actor".to_string(), DbValue::Text("system".to_string())),
                ("reason".to_string(), DbValue::Null),
                ("target_kind".to_string(), DbValue::Text(target_kind.to_string())),
                ("target_id".to_string(), DbValue::Uuid(target_id)),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("payload".to_string(), DbValue::Null),
            ]),
        )?;
        Ok(())
    }

    fn execute(&self, sql: &str, params: HashMap<String, DbValue>) -> Result<u64> {
        Ok(self
            .store
            .internal_db()
            .execute_in_tx(self.tx, sql, &params)
            .map_err(map_engine)?
            .rows_affected)
    }

    fn insert_edge(
        &mut self,
        source_id: uuid::Uuid,
        target_id: uuid::Uuid,
        edge_type: &str,
        context_id: ContextId,
        properties: Option<JsonValue>,
    ) -> Result<()> {
        self.execute(
            "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(uuid::Uuid::now_v7())),
                ("source_id".to_string(), DbValue::Uuid(source_id)),
                ("target_id".to_string(), DbValue::Uuid(target_id)),
                ("edge_type".to_string(), DbValue::Text(edge_type.to_string())),
                ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
                ("valid_from".to_string(), DbValue::TxId(self.tx)),
                ("valid_to".to_string(), DbValue::Null),
                (
                    "properties".to_string(),
                    DbValue::Json(properties.unwrap_or_else(|| json!({}))),
                ),
            ]),
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct TxEntitySnapshot {
    id: EntitySnapshotId,
    entity_id: EntityId,
    entity_type: EntityType,
    name: String,
    properties: BTreeMap<String, JsonValue>,
    valid_from: crate::TxId,
    valid_to: Option<crate::TxId>,
}

struct EntitySnapshotInsert<'a> {
    context_id: ContextId,
    entity_id: EntityId,
    snapshot_id: EntitySnapshotId,
    name: &'a str,
    entity_type: EntityType,
    properties_json: &'a JsonValue,
    valid_to: Option<crate::TxId>,
}

struct InvalidationInsert {
    context_id: ContextId,
    decision_id: DecisionId,
    trigger_observation_id: ObservationId,
    reason: Option<String>,
    basis_diff: BasisDiffResult,
    cascade_from: Option<DecisionId>,
    severity: InvalidationSeverity,
    threatened_intention_ids: Vec<IntentionId>,
    intention_now_un_served: bool,
}

fn properties_to_json_value(properties: &BTreeMap<String, JsonValue>) -> JsonValue {
    JsonValue::Object(
        properties
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

fn state_delta_to_json(delta: &BTreeMap<String, Option<JsonValue>>) -> JsonValue {
    let mut ops = serde_json::Map::new();
    for (key, value) in delta {
        let mut op = serde_json::Map::new();
        match value {
            Some(value) => {
                op.insert("op".to_string(), JsonValue::String("set".to_string()));
                op.insert("value".to_string(), value.clone());
            }
            None => {
                op.insert("op".to_string(), JsonValue::String("delete".to_string()));
            }
        }
        ops.insert(key.clone(), JsonValue::Object(op));
    }
    json!({
        "format": "cg_state_delta_v1",
        "ops": ops,
    })
}

fn optional_pair_json(pair: &Option<(u64, u64)>) -> DbValue {
    pair.map(|(start, end)| json!([start, end]))
        .map(DbValue::Json)
        .unwrap_or(DbValue::Null)
}

fn decision_basis_row_id(decision_id: DecisionId, basis_order: usize) -> uuid::Uuid {
    let mut bytes = *decision_id.as_uuid().as_bytes();
    let order = (basis_order as u64).to_be_bytes();
    for (idx, byte) in order.iter().enumerate() {
        bytes[8 + idx] ^= byte;
    }
    uuid::Uuid::from_bytes(bytes)
}

fn basis_snapshot_edge_properties(
    snapshot: &TxEntitySnapshot,
    basis_fields: Option<&[String]>,
) -> JsonValue {
    let mut properties = json!({
        "basis_snapshot": {
            "id": snapshot.id.as_uuid().to_string(),
            "entity_id": snapshot.entity_id.as_uuid().to_string(),
            "entity_type": entity_type_label(snapshot.entity_type),
            "name": snapshot.name,
            "properties": properties_to_json_value(&snapshot.properties),
            "evidenced_by": [],
            "captured_at": snapshot.valid_from.0,
            "valid_from": snapshot.valid_from.0,
            "valid_to": snapshot.valid_to.map(|tx| tx.0),
        }
    });
    if let Some(fields) = basis_fields
        && let Some(object) = properties.as_object_mut()
    {
        object.insert("basis_fields".to_string(), json!(fields));
    }
    properties
}

fn basis_snapshot_differs_from_delta(
    snapshot_data: &JsonValue,
    edge_properties: &JsonValue,
    state_delta: &BTreeMap<String, Option<JsonValue>>,
) -> bool {
    let Some(properties) = snapshot_data
        .get("properties")
        .and_then(JsonValue::as_object)
    else {
        return false;
    };
    let mut fields = edge_properties
        .get("basis_fields")
        .and_then(JsonValue::as_array)
        .map(|fields| {
            fields
                .iter()
                .filter_map(JsonValue::as_str)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| state_delta.keys().cloned().collect());
    fields.sort();
    fields.dedup();
    fields.into_iter().any(|field| {
        state_delta
            .get(&field)
            .is_some_and(|new_value| properties.get(&field) != new_value.as_ref())
    })
}

fn empty_basis_diff(tx: crate::TxId) -> BasisDiffResult {
    BasisDiffResult {
        is_invalidated_candidate: false,
        divergences: Vec::new(),
        read_txid: tx,
        current_snapshot_read_txid: tx,
    }
}

fn decode_tx_entity_snapshot(columns: &[String], row: &[DbValue]) -> Result<TxEntitySnapshot> {
    let id = EntitySnapshotId::from(required_uuid_column(row, columns, "id")?);
    let entity_id = EntityId::from(required_uuid_column(row, columns, "entity_id")?);
    let data = required_json_column(row, columns, "data")?;
    let entity_type = data
        .get("entity_type")
        .and_then(JsonValue::as_str)
        .map(entity_type_from_label)
        .transpose()?
        .unwrap_or(EntityType::Document);
    let name = data
        .get("name")
        .and_then(JsonValue::as_str)
        .unwrap_or("entity")
        .to_string();
    let properties = data
        .get("properties")
        .and_then(JsonValue::as_object)
        .map(|map| {
            map.iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default();
    Ok(TxEntitySnapshot {
        id,
        entity_id,
        entity_type,
        name,
        properties,
        valid_from: required_txid_column(row, columns, "valid_from")?,
        valid_to: optional_txid_column(row, columns, "valid_to")?,
    })
}

fn decode_invalidation_rows(
    columns: &[String],
    rows: &[Vec<DbValue>],
) -> Result<Vec<crate::Invalidation>> {
    let mut invalidations = rows
        .iter()
        .map(|row| decode_invalidation_row(columns, row))
        .collect::<Result<Vec<_>>>()?;
    invalidations.sort_by_key(|row| row.detected_at);
    Ok(invalidations)
}

fn decode_invalidation_row(columns: &[String], row: &[DbValue]) -> Result<crate::Invalidation> {
    Ok(crate::Invalidation {
        id: InvalidationId::from(required_uuid_column(row, columns, "id")?),
        status: invalidation_status_from_label(&required_text_column(row, columns, "status")?)?,
        context_id: ContextId::from(required_uuid_column(row, columns, "context_id")?),
        affected_decision_id: DecisionId::from(required_uuid_column(
            row,
            columns,
            "affected_decision_id",
        )?),
        trigger_observation_id: ObservationId::from(required_uuid_column(
            row,
            columns,
            "trigger_observation_id",
        )?),
        reason: optional_text_column(row, columns, "reason")?,
        basis_diff: basis_diff_from_json(required_json_column(row, columns, "basis_diff")?)?,
        cascade_from: optional_uuid_column(row, columns, "cascade_from")?.map(DecisionId::from),
        severity: invalidation_severity_from_label(&required_text_column(
            row, columns, "severity",
        )?)?,
        threatened_intention_ids: intention_ids_from_json(required_json_column(
            row,
            columns,
            "threatened_intention_ids",
        )?)?,
        intention_now_un_served: required_bool_column(row, columns, "intention_now_un_served")?,
        detected_at: required_txid_column(row, columns, "detected_at")?,
        resolved_at: optional_txid_column(row, columns, "resolved_at")?,
        resolution_action: optional_text_column(row, columns, "resolution_action")?,
        resolution_decision_id: optional_uuid_column(row, columns, "resolution_decision_id")?
            .map(DecisionId::from),
    })
}

fn basis_diff_from_json(value: JsonValue) -> Result<BasisDiffResult> {
    serde_json::from_value(value)
        .map_err(|err| CgError::Engine(format!("invalid basis_diff JSON: {err}")))
}

fn intention_ids_from_json(value: JsonValue) -> Result<Vec<IntentionId>> {
    serde_json::from_value(value)
        .map_err(|err| CgError::Engine(format!("invalid threatened_intention_ids JSON: {err}")))
}

fn column_index(columns: &[String], name: &str) -> Result<usize> {
    columns
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| CgError::Engine(format!("missing column {name}")))
}

fn required_uuid_column(row: &[DbValue], columns: &[String], name: &str) -> Result<uuid::Uuid> {
    match row.get(column_index(columns, name)?) {
        Some(DbValue::Uuid(id)) => Ok(*id),
        other => Err(CgError::Engine(format!(
            "column {name}: expected uuid, got {other:?}"
        ))),
    }
}

fn optional_uuid_column(
    row: &[DbValue],
    columns: &[String],
    name: &str,
) -> Result<Option<uuid::Uuid>> {
    match row.get(column_index(columns, name)?) {
        Some(DbValue::Uuid(id)) => Ok(Some(*id)),
        Some(DbValue::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected uuid|null, got {other:?}"
        ))),
    }
}

fn required_text_column(row: &[DbValue], columns: &[String], name: &str) -> Result<String> {
    match row.get(column_index(columns, name)?) {
        Some(DbValue::Text(text)) => Ok(text.clone()),
        other => Err(CgError::Engine(format!(
            "column {name}: expected text, got {other:?}"
        ))),
    }
}

fn optional_text_column(row: &[DbValue], columns: &[String], name: &str) -> Result<Option<String>> {
    match row.get(column_index(columns, name)?) {
        Some(DbValue::Text(text)) => Ok(Some(text.clone())),
        Some(DbValue::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected text|null, got {other:?}"
        ))),
    }
}

fn required_json_column(row: &[DbValue], columns: &[String], name: &str) -> Result<JsonValue> {
    match row.get(column_index(columns, name)?) {
        Some(DbValue::Json(value)) => Ok(value.clone()),
        Some(DbValue::Text(text)) => serde_json::from_str(text)
            .map_err(|err| CgError::Engine(format!("column {name}: invalid json: {err}"))),
        other => Err(CgError::Engine(format!(
            "column {name}: expected json, got {other:?}"
        ))),
    }
}

fn count_from_result(columns: &[String], rows: &[Vec<DbValue>]) -> Result<u32> {
    let row = rows
        .first()
        .ok_or_else(|| CgError::Engine("count query returned no rows".to_string()))?;
    match row.get(column_index(columns, "n")?) {
        Some(DbValue::Int64(value)) if *value >= 0 && *value <= u32::MAX as i64 => {
            Ok(*value as u32)
        }
        other => Err(CgError::Engine(format!(
            "count query returned invalid count {other:?}"
        ))),
    }
}

fn required_bool_column(row: &[DbValue], columns: &[String], name: &str) -> Result<bool> {
    match row.get(column_index(columns, name)?) {
        Some(DbValue::Bool(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected bool, got {other:?}"
        ))),
    }
}

fn required_txid_column(row: &[DbValue], columns: &[String], name: &str) -> Result<crate::TxId> {
    match row.get(column_index(columns, name)?) {
        Some(DbValue::TxId(tx)) => Ok(*tx),
        other => Err(CgError::Engine(format!(
            "column {name}: expected txid, got {other:?}"
        ))),
    }
}

fn optional_txid_column(
    row: &[DbValue],
    columns: &[String],
    name: &str,
) -> Result<Option<crate::TxId>> {
    match row.get(column_index(columns, name)?) {
        Some(DbValue::TxId(tx)) => Ok(Some(*tx)),
        Some(DbValue::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected txid|null, got {other:?}"
        ))),
    }
}

fn decision_status_to_label(status: DecisionStatus) -> &'static str {
    match status {
        DecisionStatus::Active => "active",
        DecisionStatus::Superseded => "superseded",
        DecisionStatus::Archived => "archived",
        DecisionStatus::Invalidated => "invalidated",
    }
}

fn decision_status_from_label(label: &str) -> Result<DecisionStatus> {
    match label {
        "active" => Ok(DecisionStatus::Active),
        "superseded" => Ok(DecisionStatus::Superseded),
        "archived" => Ok(DecisionStatus::Archived),
        "invalidated" => Ok(DecisionStatus::Invalidated),
        other => Err(CgError::Engine(format!("unknown decision status: {other}"))),
    }
}

fn entity_type_label(entity_type: EntityType) -> &'static str {
    match entity_type {
        EntityType::Service => "SERVICE",
        EntityType::Person => "PERSON",
        EntityType::Device => "DEVICE",
        EntityType::Module => "MODULE",
        EntityType::Repository => "REPOSITORY",
        EntityType::Configuration => "CONFIGURATION",
        EntityType::Infrastructure => "INFRASTRUCTURE",
        EntityType::Location => "LOCATION",
        EntityType::File => "FILE",
        EntityType::Pipeline => "PIPELINE",
        EntityType::Database => "DATABASE",
        EntityType::Team => "TEAM",
        EntityType::Document => "DOCUMENT",
    }
}

fn entity_type_from_label(label: &str) -> Result<EntityType> {
    match label {
        "SERVICE" => Ok(EntityType::Service),
        "PERSON" => Ok(EntityType::Person),
        "DEVICE" => Ok(EntityType::Device),
        "MODULE" => Ok(EntityType::Module),
        "REPOSITORY" => Ok(EntityType::Repository),
        "CONFIGURATION" => Ok(EntityType::Configuration),
        "INFRASTRUCTURE" => Ok(EntityType::Infrastructure),
        "LOCATION" => Ok(EntityType::Location),
        "FILE" => Ok(EntityType::File),
        "PIPELINE" => Ok(EntityType::Pipeline),
        "DATABASE" => Ok(EntityType::Database),
        "TEAM" => Ok(EntityType::Team),
        "DOCUMENT" => Ok(EntityType::Document),
        other => Err(CgError::Engine(format!("unknown entity type: {other}"))),
    }
}

fn evidence_kind_label(kind: EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::Text => "text",
        EvidenceKind::ImageFrame => "image_frame",
        EvidenceKind::AudioSegment => "audio_segment",
        EvidenceKind::VideoSegment => "video_segment",
        EvidenceKind::StructuredSignal => "structured_signal",
    }
}

fn retention_status_label(status: crate::RetentionStatus) -> &'static str {
    match status {
        crate::RetentionStatus::RetainedExternal => "retained_external",
        crate::RetentionStatus::NotStored => "not_stored",
        crate::RetentionStatus::Ephemeral => "ephemeral",
        crate::RetentionStatus::Expired => "expired",
        crate::RetentionStatus::Unavailable => "unavailable",
    }
}

fn invalidation_status_from_label(label: &str) -> Result<InvalidationStatus> {
    match label {
        "pending" => Ok(InvalidationStatus::Pending),
        "acknowledged" => Ok(InvalidationStatus::Acknowledged),
        "resolved" => Ok(InvalidationStatus::Resolved),
        "dismissed" => Ok(InvalidationStatus::Dismissed),
        other => Err(CgError::Engine(format!(
            "unknown invalidation status: {other}"
        ))),
    }
}

fn invalidation_severity_label(severity: InvalidationSeverity) -> &'static str {
    match severity {
        InvalidationSeverity::Info => "info",
        InvalidationSeverity::Warning => "warning",
        InvalidationSeverity::Critical => "critical",
    }
}

fn invalidation_severity_from_label(label: &str) -> Result<InvalidationSeverity> {
    match label {
        "info" => Ok(InvalidationSeverity::Info),
        "warning" => Ok(InvalidationSeverity::Warning),
        "critical" => Ok(InvalidationSeverity::Critical),
        other => Err(CgError::Engine(format!(
            "unknown invalidation severity: {other}"
        ))),
    }
}

impl Drop for CaptureTxn<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.store.internal_db().rollback(self.tx);
        }
    }
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
        import_chunks: None,
        resolved_chunk_window: None,
        model_context_tokens: None,
        model_usage: Vec::new(),
        captured_entities: Vec::new(),
        captured_decisions: Vec::new(),
        captured_observations: Vec::new(),
        captured_outcomes: Vec::new(),
        flagged_items: Vec::new(),
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

/// Narrow a candidate set of prior decisions to the near few most relevant to a query
/// decision (its text plus its serving goal), bounded by `limit`.
///
/// Relevance is a deterministic lexical overlap of distinguishing tokens between the query
/// and each prior decision (its description and the goals it serves). The highest-overlap
/// priors come first; ties break by the original order (most-recent-first). This is the
/// retrieval primitive that feeds the identity judge only the near neighbours, never the
/// whole store. When every prior shares nothing distinctive, the most-recent few are kept so
/// the set is the relevant few, not collapsed to zero.
pub(crate) fn narrow_prior_decisions(
    query_decision: &str,
    query_goal: Option<&str>,
    priors: &[ExtractionPriorDecision],
    limit: usize,
) -> Vec<ExtractionPriorDecision> {
    if priors.is_empty() || limit == 0 {
        return Vec::new();
    }
    let mut query_tokens = relevance_tokens(query_decision);
    if let Some(goal) = query_goal {
        query_tokens.extend(relevance_tokens(goal));
    }
    let mut scored: Vec<(usize, usize, &ExtractionPriorDecision)> = priors
        .iter()
        .enumerate()
        .map(|(order, prior)| {
            let mut prior_tokens = relevance_tokens(&prior.text);
            for goal in &prior.goals {
                prior_tokens.extend(relevance_tokens(goal));
            }
            let overlap = query_tokens.intersection(&prior_tokens).count();
            (overlap, order, prior)
        })
        // A candidate set is the NEAR neighbours, not the whole store: a prior with zero
        // distinguishing-token overlap is a far/irrelevant row and is excluded outright (a positive
        // exclusion), so the judge never sees the whole store.
        .filter(|(overlap, _, _)| *overlap > 0)
        .collect();
    // Highest overlap first; tie-break by original order (priors arrive most-recent-first).
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, _, prior)| prior.clone())
        .collect()
}

/// Distinguishing lexical tokens for relevance overlap: alphanumeric words, lowercased, with
/// the common decision/choice filler words dropped so generic phrasing ("use", "for") does not
/// drown out the distinctive subject.
fn relevance_tokens(text: &str) -> BTreeSet<String> {
    normalize_intention_description(text)
        .split_whitespace()
        .filter(|word| !choice_subject_stopword(word))
        .map(str::to_string)
        .collect()
}

pub(crate) fn extraction_prior_decisions(
    store: &Store,
    context_id: ContextId,
) -> Result<Vec<ExtractionPriorDecision>> {
    let result = store
        .internal_db()
        .execute(
            &format!(
                "SELECT id, description
             FROM decisions
             WHERE context_id = $context_id
               AND status = $status
             ORDER BY created_at DESC, id ASC
             LIMIT {EXTRACTION_PRIOR_DECISION_LIMIT}"
            ),
            &HashMap::from([
                (
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                ),
                ("status".to_string(), DbValue::Text("active".to_string())),
            ]),
        )
        .map_err(map_engine)?;
    let mut rows = result
        .rows
        .iter()
        .map(|row| {
            let id = match row.first() {
                Some(DbValue::Uuid(id)) => DecisionId::from(*id),
                _ => {
                    return Err(CgError::Engine(
                        "prior decision recall returned non-uuid id".to_string(),
                    ));
                }
            };
            let text = match row.get(1) {
                Some(DbValue::Text(text)) => text.clone(),
                _ => {
                    return Err(CgError::Engine(
                        "prior decision recall returned non-text description".to_string(),
                    ));
                }
            };
            Ok((id, text))
        })
        .collect::<Result<Vec<_>>>()?;
    rows.reverse();
    let ids = rows.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    let goals = prior_decision_goals_batch(store, context_id, &ids)?;
    let entities = prior_decision_entities_batch(store, context_id, &ids)?;
    Ok(rows
        .into_iter()
        .map(|(id, text)| ExtractionPriorDecision {
            id,
            text,
            goals: goals.get(&id).cloned().unwrap_or_default(),
            entities: entities.get(&id).cloned().unwrap_or_default(),
        })
        .collect())
}

fn prior_decision_goals_batch(
    store: &Store,
    context_id: ContextId,
    decision_ids: &[DecisionId],
) -> Result<HashMap<DecisionId, Vec<String>>> {
    if decision_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut params = HashMap::from([
        (
            "context_id".to_string(),
            DbValue::Uuid(context_id.as_uuid()),
        ),
        ("edge_type".to_string(), DbValue::Text("SERVES".to_string())),
    ]);
    let id_list = decision_id_list("decision", decision_ids, &mut params);
    let result = store
        .internal_db()
        .execute(
            &format!(
                "SELECT edges.source_id, intentions.description
             FROM edges
             INNER JOIN intentions ON intentions.id = edges.target_id
             WHERE edges.context_id = $context_id
               AND edges.source_id IN ({id_list})
               AND edges.edge_type = $edge_type
               AND intentions.context_id = $context_id
             ORDER BY intentions.created_at ASC"
            ),
            &params,
        )
        .map_err(map_engine)?;
    let mut out = decision_ids
        .iter()
        .map(|id| (*id, Vec::new()))
        .collect::<HashMap<_, _>>();
    for row in &result.rows {
        let decision_id =
            DecisionId::from(required_uuid_column(row, &result.columns, "source_id")?);
        let text = required_text_column(row, &result.columns, "description")?;
        out.entry(decision_id).or_default().push(text);
    }
    Ok(out)
}

fn prior_decision_entities_batch(
    store: &Store,
    context_id: ContextId,
    decision_ids: &[DecisionId],
) -> Result<HashMap<DecisionId, Vec<ExtractionPriorEntity>>> {
    if decision_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut params = HashMap::from([
        (
            "context_id".to_string(),
            DbValue::Uuid(context_id.as_uuid()),
        ),
        (
            "edge_type".to_string(),
            DbValue::Text("BASED_ON".to_string()),
        ),
    ]);
    let id_list = decision_id_list("decision", decision_ids, &mut params);
    let result = store
        .internal_db()
        .execute(
            &format!(
                "SELECT edges.source_id, entities.id, entities.name
             FROM edges
             INNER JOIN entities ON entities.id = edges.target_id
             WHERE edges.context_id = $context_id
               AND edges.source_id IN ({id_list})
               AND edges.edge_type = $edge_type
               AND entities.context_id = $context_id
             ORDER BY entities.created_at ASC"
            ),
            &params,
        )
        .map_err(map_engine)?;
    let mut out = decision_ids
        .iter()
        .map(|id| (*id, Vec::new()))
        .collect::<HashMap<_, _>>();
    for row in &result.rows {
        let decision_id =
            DecisionId::from(required_uuid_column(row, &result.columns, "source_id")?);
        let entity_id = EntityId::from(required_uuid_column(row, &result.columns, "id")?);
        let name = required_text_column(row, &result.columns, "name")?;
        out.entry(decision_id)
            .or_default()
            .push(ExtractionPriorEntity {
                name,
                path: current_entity_path(store, entity_id)?,
            });
    }
    Ok(out)
}

fn decision_id_list(
    prefix: &str,
    decision_ids: &[DecisionId],
    params: &mut HashMap<String, DbValue>,
) -> String {
    decision_ids
        .iter()
        .enumerate()
        .map(|(idx, decision_id)| {
            let key = format!("{prefix}_{idx}");
            params.insert(key.clone(), DbValue::Uuid(decision_id.as_uuid()));
            format!("${key}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn snapshot_id_list(
    prefix: &str,
    snapshot_ids: &[EntitySnapshotId],
    params: &mut HashMap<String, DbValue>,
) -> String {
    snapshot_ids
        .iter()
        .enumerate()
        .map(|(idx, snapshot_id)| {
            let key = format!("{prefix}_{idx}");
            params.insert(key.clone(), DbValue::Uuid(snapshot_id.as_uuid()));
            format!("${key}")
        })
        .collect::<Vec<_>>()
        .join(", ")
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

#[derive(Clone)]
struct EntityWrite {
    id: EntityId,
    prior_path: Option<String>,
}

fn find_existing_entity_for_seed(
    store: &Store,
    context_id: ContextId,
    name: &str,
    path: &str,
    entity_type: EntityType,
) -> Result<Option<crate::Entity>> {
    let candidates = store.list_entities(crate::ListEntityFilter {
        entity_type: Some(entity_type),
        include_archived: false,
        context_id: Some(context_id),
        ..Default::default()
    })?;
    if let Some(exact) = candidates.iter().find(|entity| entity.name == name) {
        return Ok(Some(exact.clone()));
    }
    let target = normalize_entity_identity_name(name);
    if !target.is_empty()
        && let Some(named) = candidates
            .iter()
            .find(|entity| normalize_entity_identity_name(&entity.name) == target)
    {
        return Ok(Some(named.clone()));
    }
    let path = normalize_entity_identity_path(path);
    if path.is_empty() {
        return Ok(None);
    }
    for entity in candidates {
        let candidate_path = current_entity_path(store, entity.id)?
            .or_else(|| entity_path_from_properties(&entity.properties))
            .map(|path| normalize_entity_identity_path(&path))
            .unwrap_or_default();
        if candidate_path == path {
            return Ok(Some(entity));
        }
    }
    Ok(None)
}

fn normalize_entity_identity_name(name: &str) -> String {
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            push_entity_identity_word(&mut words, &mut current);
        }
    }
    if !current.is_empty() {
        push_entity_identity_word(&mut words, &mut current);
    }
    words.join(" ")
}

fn push_entity_identity_word(words: &mut Vec<String>, current: &mut String) {
    let word = match current.as_str() {
        "a" | "an" | "the" => {
            current.clear();
            return;
        }
        "configuration" | "configurations" | "configs" => "config",
        "services" => "service",
        "modules" => "module",
        "files" => "file",
        other => other,
    };
    words.push(word.to_string());
    current.clear();
}

fn normalize_entity_identity_path(path: &str) -> String {
    path.trim().replace('\\', "/")
}

fn current_entity_path(store: &Store, entity_id: EntityId) -> Result<Option<String>> {
    Ok(store
        .current_entity_snapshot(entity_id)?
        .and_then(|snapshot| entity_path_from_properties(&snapshot.properties)))
}

fn entity_path_from_properties(properties: &BTreeMap<String, JsonValue>) -> Option<String> {
    properties
        .get("path")
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(ToOwned::to_owned)
}

fn resolved_decision_relation(
    store: &Store,
    context_id: ContextId,
    relation: Option<&DecisionRelationSeed>,
) -> Result<Option<(DecisionRelationClassification, Decision)>> {
    let Some(relation) = relation else {
        return Ok(None);
    };
    let id = relation.prior_decision_id.trim();
    if id.is_empty() || relation.classification == DecisionRelationClassification::NewInsertion {
        return Ok(None);
    }
    let prior_id = uuid::Uuid::parse_str(id)
        .map(DecisionId::from)
        .map_err(|err| {
            CgError::Engine(format!(
                "extractor returned invalid prior_decision_id: {err}"
            ))
        })?;
    let prior = store.get_decision(prior_id)?.ok_or_else(|| {
        CgError::Engine("extractor returned unknown prior_decision_id".to_string())
    })?;
    if prior.context_id != context_id {
        return Err(CgError::CrossContextViolation);
    }
    Ok(Some((relation.classification, prior)))
}

fn relation_entity_label(seed: &DecisionSeed, entity_names: &BTreeMap<String, String>) -> String {
    seed.entity_key
        .as_ref()
        .and_then(|key| entity_names.get(key))
        .cloned()
        .unwrap_or_else(|| "decision".to_string())
}

fn decision_input(
    context_id: ContextId,
    seed: &DecisionSeed,
    intention_ids: &[IntentionId],
    entity_id: Option<EntityId>,
    precedent_ids: Vec<DecisionId>,
) -> CreateDecision {
    // Convert Vec<String> reasoning into Vec<ReasoningStep>
    let reasoning: Vec<ReasoningStep> = seed
        .reasoning
        .iter()
        .map(|text| ReasoningStep {
            operation: "observed".to_string(),
            details: text.clone(),
            result: text.clone(),
            agent_id: None,
            timestamp: chrono::Utc::now(),
        })
        .collect();

    // Convert serde_json::Value properties to BTreeMap<String, serde_json::Value>
    let properties: std::collections::BTreeMap<String, serde_json::Value> =
        if let serde_json::Value::Object(map) = &seed.properties {
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        } else {
            std::collections::BTreeMap::new()
        };

    CreateDecision {
        decision_type: seed
            .decision_type
            .clone()
            .map(ModelExtractedField::into_value)
            .unwrap_or_else(|| "decision".to_string()),
        description: seed.text.clone(),
        confidence: None, // Always None at capture — invariant
        reasoning,
        tags: seed.tags.clone(),
        properties,
        intention_ids: intention_ids.to_vec(),
        based_on_entity_ids: entity_id.into_iter().collect(),
        basis_fields: entity_id
            .map(|id| std::collections::HashMap::from([(id, vec!["path".to_string()])])),
        context_id,
        precedent_ids,
        agent_id: seed.agent_id.as_deref().map(AgentId::from),
        ..Default::default()
    }
}

struct ObservationWrite {
    id: ObservationId,
    invalidations: Vec<crate::Invalidation>,
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
        let result = db
            .execute(
                &format!("SELECT COUNT(*) AS n FROM {table} WHERE context_id = $context_id"),
                &HashMap::from([(
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                )]),
            )
            .map_err(map_engine)?;
        count_from_result(&result.columns, &result.rows)
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::embedder::mock::LexicalMockEmbedder;
    use crate::{
        CreateIntention, EmbedderRegistration, IntentionOrigin, IntentionStatus, StoreConfig,
    };

    #[derive(Debug)]
    struct StaticExtractor(StructuredHarvest);

    impl ConversationExtractor for StaticExtractor {
        fn extract(&self, _request: ExtractionRequest<'_>) -> Result<StructuredHarvest> {
            Ok(self.0.clone())
        }
    }

    #[derive(Debug)]
    struct PriorBoundExtractor {
        prior: DecisionId,
    }

    impl ConversationExtractor for PriorBoundExtractor {
        fn extract(&self, request: ExtractionRequest<'_>) -> Result<StructuredHarvest> {
            assert!(
                request.prior_decisions.len() <= EXTRACTION_PRIOR_DECISION_LIMIT,
                "capture prompt prior recall must stay bounded"
            );
            assert!(
                request
                    .prior_decisions
                    .iter()
                    .any(|prior| prior.id == self.prior),
                "bounded prior recall must keep the newest relevant prior"
            );
            Ok(StructuredHarvest {
                summary: Some("bounded prior recall conflict".to_string()),
                goals: vec![GoalSeed {
                    key: "G1".to_string(),
                    text: "keep bounded prior recall useful".to_string(),
                    context_id: None,
                }],
                entities: Vec::new(),
                decisions: vec![DecisionSeed {
                    key: "D1".to_string(),
                    text: "replace the newest bounded prior decision".to_string(),
                    goal_key: Some("G1".to_string()),
                    entity_key: None,
                    relation: None,
                    ..Default::default()
                }],
                observations: Vec::new(),
                outcomes: Vec::new(),
            })
        }

        fn classify_decision_relation(
            &self,
            request: DecisionRelationRequest<'_>,
        ) -> Result<Option<DecisionRelationSeed>> {
            assert!(
                request.prior_decisions.len() <= EXTRACTION_PRIOR_DECISION_LIMIT,
                "relation prompt prior recall must stay bounded"
            );
            Ok(Some(DecisionRelationSeed {
                prior_decision_id: self.prior.as_uuid().to_string(),
                classification: DecisionRelationClassification::ConflictsExisting,
            }))
        }
    }

    #[test]
    fn structured_capture_rolls_back_digest_row_when_embedding_handle_insert_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StoreConfig {
            db_path: dir.path().join("cg.db"),
            default_text_embedder: None,
            ..Default::default()
        };
        let embedder = Arc::new(LexicalMockEmbedder::default());
        let store = Store::open_with_embedder_registrations(
            config,
            vec![EmbedderRegistration {
                space: embedder.embedding_space(),
                embedder,
            }],
        )
        .expect("open store");
        store
            .internal_db()
            .execute(
                "DELETE FROM embedding_space_bindings WHERE embedding_space_id = $embedding_space_id",
                &HashMap::from([(
                    "embedding_space_id".to_string(),
                    DbValue::Text(BuiltinSpaceRegistry::TEXT_DEFAULT_ID.to_string()),
                )]),
            )
            .expect("remove embedding space binding");
        store
            .internal_db()
            .execute(
                "DELETE FROM embedding_spaces WHERE embedding_space_id = $embedding_space_id",
                &HashMap::from([(
                    "embedding_space_id".to_string(),
                    DbValue::Text(BuiltinSpaceRegistry::TEXT_DEFAULT_ID.to_string()),
                )]),
            )
            .expect("remove embedding space to force handle FK failure");
        let extractor = StaticExtractor(StructuredHarvest {
            summary: Some("rollback capture".to_string()),
            goals: vec![GoalSeed {
                key: "G1".to_string(),
                text: "keep rollback test reliable".to_string(),
                context_id: None,
            }],
            entities: vec![EntitySeed {
                key: "E1".to_string(),
                name: "rollback config".to_string(),
                path: "config/rollback.toml".to_string(),
                entity_type: ExtractedEntityType::Configuration,
                entity_type_raw: None,
                context_id: None,
            }],
            decisions: vec![DecisionSeed {
                key: "D1".to_string(),
                text: "use config/rollback.toml for rollback config".to_string(),
                goal_key: Some("G1".to_string()),
                entity_key: Some("E1".to_string()),
                relation: None,
                ..Default::default()
            }],
            observations: Vec::new(),
            outcomes: Vec::new(),
        });

        let result = record_conversation_with_extractor(
            &store,
            CaptureRequest {
                context_label: "rollback-capture".to_string(),
                transcript: "capture should roll back after a partial write".to_string(),
                model_available: true,
            },
            &extractor,
        );

        assert!(result.is_err());
        let context = context_by_label(&store, "rollback-capture").expect("context");
        assert_eq!(
            total_counts(&store, context.id).expect("counts"),
            HarvestCounts::default()
        );
        for table in ["digests", "entities", "edges"] {
            assert_eq!(
                count_context_rows(&store, table, context.id),
                0,
                "{table} rows must roll back with the failed capture"
            );
        }
    }

    #[test]
    fn extraction_prior_recall_is_bounded_and_still_invalidates_recalled_prior() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StoreConfig {
            db_path: dir.path().join("cg.db"),
            default_text_embedder: None,
            ..Default::default()
        };
        let embedder = Arc::new(LexicalMockEmbedder::default());
        let store = Store::open_with_embedder_registrations(
            config,
            vec![EmbedderRegistration {
                space: embedder.embedding_space(),
                embedder,
            }],
        )
        .expect("open store");
        let context = context_by_label(&store, "bounded-prior-recall").expect("context");
        let goal = store
            .create_intention(CreateIntention {
                description: "keep bounded prior recall useful".to_string(),
                status: IntentionStatus::Active,
                origin: IntentionOrigin::Human,
                context_id: context.id,
                ..Default::default()
            })
            .expect("goal");
        let mut latest = None;
        for idx in 0..(EXTRACTION_PRIOR_DECISION_LIMIT + 12) {
            let decision = store
                .create_decision(CreateDecision {
                    decision_type: "prior".to_string(),
                    description: format!("prior decision {idx}"),
                    intention_ids: vec![goal.id],
                    context_id: context.id,
                    ..Default::default()
                })
                .expect("prior decision");
            latest = Some(decision.id);
        }
        let latest = latest.expect("latest prior");

        let report = record_conversation_with_extractor(
            &store,
            CaptureRequest {
                context_label: "bounded-prior-recall".to_string(),
                transcript: "new decision conflicts with the newest bounded prior".to_string(),
                model_available: true,
            },
            &PriorBoundExtractor { prior: latest },
        )
        .expect("bounded capture");
        let latest_row = store
            .get_decision(latest)
            .expect("latest decision lookup")
            .expect("latest decision row");

        assert_eq!(latest_row.status, DecisionStatus::Invalidated);
        assert_eq!(report.invalidations.len(), 1);
    }

    fn count_context_rows(store: &Store, table: &str, context_id: ContextId) -> usize {
        store
            .internal_db()
            .execute(
                &format!("SELECT id FROM {table} WHERE context_id = $context_id"),
                &HashMap::from([(
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                )]),
            )
            .expect("count query")
            .rows
            .len()
    }
}
