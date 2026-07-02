use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use contextdb_core::Lsn;

use crate::embedder::EmbedderConfig;
use crate::{AgentId, TxId};

macro_rules! id_type {
    ($name:ident) => {
        id_type!($name, { Self::new_v7() });
    };
    ($name:ident, nil_default) => {
        id_type!($name, {
            Self {
                raw: Uuid::nil(),
                valid: true,
            }
        });
    };
    ($name:ident, $default:block) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        pub struct $name {
            raw: Uuid,
            valid: bool,
        }

        impl $name {
            pub fn new_v7() -> Self {
                Self {
                    raw: Uuid::now_v7(),
                    valid: true,
                }
            }

            pub fn new_v4() -> Self {
                Self {
                    raw: Uuid::new_v4(),
                    valid: true,
                }
            }

            pub fn from_raw_bytes(_raw: &str) -> Self {
                Self {
                    raw: Uuid::nil(),
                    valid: false,
                }
            }

            pub fn as_uuid(&self) -> Uuid {
                self.raw
            }

            pub fn is_valid(&self) -> bool {
                self.valid
            }
        }

        impl Default for $name {
            fn default() -> Self $default
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.raw.fmt(f)
            }
        }

        impl From<uuid::Uuid> for $name {
            fn from(uuid: uuid::Uuid) -> Self {
                Self {
                    raw: uuid,
                    valid: true,
                }
            }
        }

        impl From<$name> for uuid::Uuid {
            fn from(id: $name) -> Self {
                id.raw
            }
        }
    };
}

id_type!(ContextId);
id_type!(IntentionId);
id_type!(DecisionId);
id_type!(EntityId);
id_type!(ObservationId, nil_default);
id_type!(OutcomeId);
id_type!(DigestId);
id_type!(AuditId);
id_type!(InvalidationId);
id_type!(EvidenceId, nil_default);
id_type!(EmbeddingId, nil_default);
id_type!(EntitySnapshotId, nil_default);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IntentionStatus {
    #[default]
    Active,
    Paused,
    Superseded,
    Archived,
}

impl IntentionStatus {
    pub const ALL: [Self; 4] = [Self::Active, Self::Paused, Self::Superseded, Self::Archived];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IntentionOrigin {
    #[default]
    Human,
    Agent,
    Signature,
    Blueprint,
}

impl IntentionOrigin {
    pub const ALL: [Self; 4] = [Self::Human, Self::Agent, Self::Signature, Self::Blueprint];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStatus {
    #[default]
    Active,
    Superseded,
    Archived,
    Invalidated,
}

impl DecisionStatus {
    pub const ALL: [Self; 4] = [
        Self::Active,
        Self::Superseded,
        Self::Archived,
        Self::Invalidated,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EntityType {
    #[default]
    Service,
    Person,
    Device,
    Module,
    Repository,
    Configuration,
    Infrastructure,
    Location,
    File,
    Pipeline,
    Database,
    Team,
    Document,
}

impl EntityType {
    pub const ALL: [Self; 13] = [
        Self::Service,
        Self::Person,
        Self::Device,
        Self::Module,
        Self::Repository,
        Self::Configuration,
        Self::Infrastructure,
        Self::Location,
        Self::File,
        Self::Pipeline,
        Self::Database,
        Self::Team,
        Self::Document,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EntityStatus {
    #[default]
    Active,
    Archived,
}

impl EntityStatus {
    pub const ALL: [Self; 2] = [Self::Active, Self::Archived];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EntityMatchPolicy {
    #[default]
    Any,
    All,
}

impl EntityMatchPolicy {
    pub const ALL: [Self; 2] = [Self::Any, Self::All];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    Serves,
    BasedOn,
    Supersedes,
    HasOutcome,
    DependsOn,
    Cites,
    About,
    RelatesTo,
    ObservedOn,
    Invalidates,
    TriggeredBy,
    ExtractedFrom,
    MatchedBy,
    Generated,
    Instantiates,
    Refines,
}

impl RelationshipType {
    pub const ALL: [Self; 16] = [
        Self::Serves,
        Self::BasedOn,
        Self::Supersedes,
        Self::HasOutcome,
        Self::DependsOn,
        Self::Cites,
        Self::About,
        Self::RelatesTo,
        Self::ObservedOn,
        Self::Invalidates,
        Self::TriggeredBy,
        Self::ExtractedFrom,
        Self::MatchedBy,
        Self::Generated,
        Self::Instantiates,
        Self::Refines,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditOperation {
    Create,
    Update,
    Archive,
    Supersede,
    RecordOutcome,
    RecordObservation,
    AcknowledgeInvalidation,
    ResolveInvalidation,
    DismissInvalidation,
}

impl AuditOperation {
    pub const ALL: [Self; 9] = [
        Self::Create,
        Self::Update,
        Self::Archive,
        Self::Supersede,
        Self::RecordOutcome,
        Self::RecordObservation,
        Self::AcknowledgeInvalidation,
        Self::ResolveInvalidation,
        Self::DismissInvalidation,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditTarget {
    Context(ContextId),
    Decision(DecisionId),
    Intention(IntentionId),
    Entity(EntityId),
    Observation(ObservationId),
    Outcome(DecisionId),
    Invalidation(InvalidationId),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Principal {
    System,
    Agent(AgentId),
    Human(String),
}

impl Principal {
    pub fn encode(&self) -> String {
        match self {
            Principal::System => "system".to_string(),
            Principal::Agent(a) => format!("agent:{a}"),
            Principal::Human(h) => format!("human:{h}"),
        }
    }

    pub fn decode(raw: &str) -> Self {
        if raw == "system" || raw == "System" {
            return Principal::System;
        }
        if let Some(agent) = raw.strip_prefix("agent:") {
            return Principal::Agent(agent.to_string());
        }
        if let Some(human) = raw.strip_prefix("human:") {
            return Principal::Human(human.to_string());
        }
        if raw.starts_with("claude-code-instance-") {
            return Principal::Agent(raw.to_string());
        }
        Principal::Human(raw.to_string())
    }
}

pub type Region = Value;

#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvidenceKind {
    #[default]
    Text,
    ImageFrame,
    AudioSegment,
    VideoSegment,
    StructuredSignal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionStatus {
    NotStored,
    Ephemeral,
    RetainedExternal,
    Expired,
    Unavailable,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceProducer {
    pub system: String,
    pub model_name: String,
    pub model_version: String,
    pub pipeline_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub id: EvidenceId,
    pub observation_id: ObservationId,
    pub context_id: ContextId,
    pub kind: EvidenceKind,
    pub source_ref: String,
    pub mime_type: Option<String>,
    pub content_hash: Option<String>,
    pub captured_at: Option<DateTime<Utc>>,
    pub byte_range: Option<(u64, u64)>,
    pub time_range_ms: Option<(u64, u64)>,
    pub frame_index: Option<u64>,
    pub region: Option<Region>,
    pub producer: EvidenceProducer,
    pub text_projection: Option<String>,
    pub derived_from: Option<EvidenceId>,
    pub retention_status: RetentionStatus,
    pub media_retention_until: Option<DateTime<Utc>>,
    pub properties: BTreeMap<String, Value>,
}

impl Default for EvidenceRef {
    fn default() -> Self {
        Self {
            id: EvidenceId::default(),
            observation_id: ObservationId::default(),
            context_id: ContextId::default(),
            kind: EvidenceKind::Text,
            source_ref: String::new(),
            mime_type: None,
            content_hash: None,
            captured_at: None,
            byte_range: None,
            time_range_ms: None,
            frame_index: None,
            region: None,
            producer: EvidenceProducer::default(),
            text_projection: None,
            derived_from: None,
            retention_status: RetentionStatus::NotStored,
            media_retention_until: None,
            properties: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSpaceBinding {
    pub embedding_space_id: String,
    pub table_name: String,
    pub vector_column: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSpace {
    pub embedding_space_id: String,
    pub dimension: usize,
    pub metric: DistanceMetric,
    pub model_family: String,
    pub model_name: String,
    pub model_version: String,
    pub supported_evidence_kinds: Vec<EvidenceKind>,
    pub default_for_evidence_kind: Option<EvidenceKind>,
    pub bindings: Vec<EmbeddingSpaceBinding>,
}

impl Default for EmbeddingSpace {
    fn default() -> Self {
        Self {
            embedding_space_id: String::new(),
            dimension: 0,
            metric: DistanceMetric::Cosine,
            model_family: String::new(),
            model_name: String::new(),
            model_version: String::new(),
            supported_evidence_kinds: Vec::new(),
            default_for_evidence_kind: None,
            bindings: Vec::new(),
        }
    }
}

pub type EmbeddingSpaceDecl = EmbeddingSpace;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingTarget {
    Evidence(EvidenceId),
    EntityReference(EntityId, Option<String>),
    Digest(DigestId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredEmbedding {
    pub id: EmbeddingId,
    pub context_id: ContextId,
    pub embedding_space_id: String,
    pub vector: Vec<f32>,
    pub dimension: usize,
    pub produced_by: EvidenceProducer,
    pub target: EmbeddingTarget,
    pub created_at: TxId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecomputedEmbedding {
    pub id: EmbeddingId,
    pub embedding_space_id: String,
    pub vector: Vec<f32>,
    pub target: EmbeddingTarget,
    pub produced_by: EvidenceProducer,
}

impl Default for PrecomputedEmbedding {
    fn default() -> Self {
        Self {
            id: EmbeddingId::default(),
            embedding_space_id: String::new(),
            vector: Vec::new(),
            target: EmbeddingTarget::Evidence(EvidenceId::default()),
            produced_by: EvidenceProducer::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueDiff {
    pub field: String,
    pub basis_value: Option<Value>,
    pub current_value: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityDivergence {
    pub basis_snapshot_id: EntitySnapshotId,
    pub entity_id: EntityId,
    pub value_diffs: Vec<ValueDiff>,
    pub read_txid: TxId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BasisDiffResult {
    pub is_invalidated_candidate: bool,
    pub divergences: Vec<EntityDivergence>,
    pub read_txid: TxId,
    pub current_snapshot_read_txid: TxId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionInvalidationResult {
    pub previous_status: DecisionStatus,
    pub new_status: DecisionStatus,
    pub reason: Option<String>,
    pub invalidation_id: InvalidationId,
    pub transitioned_at: TxId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceHandle {
    pub evidence_id: EvidenceId,
    pub source_ref: String,
    pub byte_range: Option<(u64, u64)>,
    pub time_range_ms: Option<(u64, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceAvailabilityLogRow {
    pub evidence_id: EvidenceId,
    pub status: RetentionStatus,
    pub recorded_at_txid: TxId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub operation: String,
    pub details: String,
    pub result: String,
    pub agent_id: Option<AgentId>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Context {
    pub id: ContextId,
    pub name: String,
    pub labels: Vec<String>,
    pub properties: BTreeMap<String, Value>,
    pub created_at: TxId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Intention {
    pub id: IntentionId,
    pub description: String,
    pub status: IntentionStatus,
    pub origin: IntentionOrigin,
    pub context_id: ContextId,
    pub properties: BTreeMap<String, Value>,
    pub tags: Vec<String>,
    pub created_at: TxId,
    /// Which catalog group this intention belongs to. None means unassigned (no grouping
    /// resolution has run). The correct impl sets this during Stage-3 grouping; the wrong
    /// stub never sets it, so captures always store None here.
    pub blueprint_catalog_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntentionSnapshot {
    pub intention_id: IntentionId,
    pub description: String,
    pub status: IntentionStatus,
    pub origin: IntentionOrigin,
    pub properties: BTreeMap<String, Value>,
    pub tags: Vec<String>,
    pub valid_from: TxId,
    pub valid_to: Option<TxId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub id: DecisionId,
    pub decision_type: String,
    pub description: String,
    pub reasoning: Vec<ReasoningStep>,
    pub confidence: Option<f32>,
    pub status: DecisionStatus,
    pub agent_id: Option<AgentId>,
    pub context_id: ContextId,
    pub tags: Vec<String>,
    pub properties: BTreeMap<String, Value>,
    pub created_at: TxId,
    pub intention_ids: Vec<IntentionId>,
    pub based_on_snapshots: Vec<EntitySnapshotId>,
    pub outcome: Option<Outcome>,
    pub precedent_ids: Vec<DecisionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub entity_type: EntityType,
    pub name: String,
    pub properties: BTreeMap<String, Value>,
    pub tags: Vec<String>,
    pub context_id: ContextId,
    pub status: EntityStatus,
    pub created_at: TxId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntitySnapshot {
    pub id: EntitySnapshotId,
    pub entity_id: EntityId,
    pub entity_type: EntityType,
    pub name: String,
    pub properties: BTreeMap<String, Value>,
    pub evidenced_by: Vec<ObservationId>,
    pub captured_at: TxId,
    pub valid_from: TxId,
    pub valid_to: Option<TxId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub id: ObservationId,
    pub observation_type: String,
    pub evidence: Vec<EvidenceRef>,
    pub observed_properties: BTreeMap<String, Value>,
    pub state_delta: BTreeMap<String, Option<Value>>,
    pub properties: BTreeMap<String, Value>,
    pub source: String,
    pub entity_id: EntityId,
    pub context_id: ContextId,
    pub observed_at: DateTime<Utc>,
    pub created_at: TxId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordObservationResult {
    pub observation_id: ObservationId,
    pub evidence_ids: Vec<EvidenceId>,
    pub embedding_ids: Vec<EmbeddingId>,
    pub snapshot_id: Option<EntitySnapshotId>,
    pub invalidations: Vec<Invalidation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub id: OutcomeId,
    pub decision_id: DecisionId,
    pub success: bool,
    pub notes: Option<String>,
    pub recorded_at: TxId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: AuditId,
    pub txid: TxId,
    pub operation: AuditOperation,
    pub actor: Principal,
    pub reason: Option<String>,
    pub target: AuditTarget,
    pub context_id: ContextId,
    pub outcome: Option<Outcome>,
    pub payload: Option<AuditPayload>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Digest {
    pub id: DigestId,
    pub source: String,
    pub summary: String,
    pub vector_text: Option<Vec<f32>>,
    pub dimension: usize,
    pub context_id: ContextId,
    pub session_ref: Option<String>,
    pub properties: BTreeMap<String, Value>,
    pub created_at: TxId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawRow {
    pub tombstone: bool,
    pub data: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawEdge {
    pub from: EntityId,
    pub to: EntityId,
    pub rel: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageKey {
    Decision(DecisionId),
    EdgeTable { from: EntityId, rel: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CreateContext {
    pub name: String,
    pub labels: Vec<String>,
    pub properties: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CreateIntention {
    pub id: Option<IntentionId>,
    pub description: String,
    pub status: IntentionStatus,
    pub origin: IntentionOrigin,
    pub context_id: ContextId,
    pub properties: BTreeMap<String, Value>,
    pub tags: Vec<String>,
    /// Catalog group to assign at creation. None means unassigned.
    pub blueprint_catalog_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct IntentionPatch {
    pub description: Option<String>,
    pub status: Option<IntentionStatus>,
    pub properties: Option<BTreeMap<String, Value>>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CreateDecision {
    pub decision_type: String,
    pub description: String,
    pub reasoning: Vec<ReasoningStep>,
    pub confidence: Option<f32>,
    pub intention_ids: Vec<IntentionId>,
    pub based_on_entity_ids: Vec<EntityId>,
    pub basis_fields: Option<HashMap<EntityId, Vec<String>>>,
    pub based_on_snapshots: Vec<EntitySnapshotId>,
    pub tags: Vec<String>,
    pub properties: BTreeMap<String, Value>,
    pub context_id: ContextId,
    pub precedent_ids: Vec<DecisionId>,
    pub agent_id: Option<AgentId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldChange {
    pub field: String,
    pub old_value: Option<Value>,
    pub new_value: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityDiffEntry {
    pub entity_id: EntityId,
    pub entity_name: String,
    pub field_changes: Vec<FieldChange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalDiff {
    pub target_id: EntityId,
    pub from: TxId,
    pub to: TxId,
    pub changes: Vec<EntityDiffEntry>,
    pub unchanged: Vec<EntityDiffEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputDiff {
    pub entity_id: EntityId,
    pub entity_name: String,
    pub changed: bool,
    pub field_changes: Vec<FieldChange>,
    pub changed_at: Option<TxId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayEvidenceLineage {
    pub basis_snapshot_id: EntitySnapshotId,
    pub observation_id: Option<ObservationId>,
    pub evidence: Vec<EvidenceRef>,
    pub read_txid: TxId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayResult {
    pub decision_id: DecisionId,
    pub decision_status: DecisionStatus,
    pub original_outcome: Option<bool>,
    pub inputs_changed: bool,
    pub changed_input_count: u32,
    pub unchanged_input_count: u32,
    pub read_txid: TxId,
    pub input_diffs: Vec<InputDiff>,
    pub evidence_lineage: Vec<ReplayEvidenceLineage>,
    pub invalidations_at_read: Vec<Invalidation>,
    pub cited_intention_snapshots: Vec<IntentionSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DecisionPatch {
    pub properties: Option<BTreeMap<String, Value>>,
    pub tags: Option<Vec<String>>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CreateEntity {
    pub entity_type: EntityType,
    pub name: String,
    pub properties: BTreeMap<String, Value>,
    pub tags: Vec<String>,
    pub context_id: ContextId,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EntityPatch {
    pub name: Option<String>,
    pub properties: Option<BTreeMap<String, Value>>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ListEntityFilter {
    pub entity_type: Option<EntityType>,
    pub name_contains: Option<String>,
    pub tags: Vec<String>,
    pub include_archived: bool,
    pub limit: Option<u32>,
    pub after: Option<EntityId>,
    pub context_id: Option<ContextId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordObservation {
    pub id: ObservationId,
    pub entity_id: EntityId,
    pub context_id: ContextId,
    pub observation_type: String,
    pub source: String,
    pub observed_at: DateTime<Utc>,
    pub evidence: Vec<EvidenceRef>,
    pub observed_properties: BTreeMap<String, Value>,
    pub state_delta: BTreeMap<String, Option<Value>>,
    pub properties: BTreeMap<String, Value>,
    pub embeddings: Vec<PrecomputedEmbedding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DiscoveryFilter {
    pub include_decision_statuses: Option<Vec<DecisionStatus>>,
    pub include_intention_statuses: Option<Vec<IntentionStatus>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TraverseFilter {
    pub depth: u8,
    pub rel: Option<RelationshipType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TraversalResult {
    pub nodes: Vec<TraversalNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraversalNode {
    pub entity_id: EntityId,
    pub hop: u8,
    pub relationship: Option<RelationshipType>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentionTree {
    pub intention: Intention,
    pub decisions: Vec<Decision>,
    pub entities: Vec<Entity>,
    pub outcomes: Vec<Outcome>,
    pub invalidations: Vec<Invalidation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Neighborhood {
    pub entities: Vec<Entity>,
    pub decisions: Vec<Decision>,
    pub intentions: Vec<Intention>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvalidationStatus {
    Pending,
    Acknowledged,
    Resolved,
    Dismissed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvalidationSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, strum::EnumIter)]
pub enum ResolveAction {
    NewDecisionCreated,
    BasisStillValid,
}

impl ResolveAction {
    pub const ALL: [Self; 2] = [Self::NewDecisionCreated, Self::BasisStillValid];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionPayload {
    pub action: ResolveAction,
    pub resolution_decision_id: Option<DecisionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DismissalPayload {
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditPayload {
    Resolution(ResolutionPayload),
    Dismissal(DismissalPayload),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Invalidation {
    pub id: InvalidationId,
    pub status: InvalidationStatus,
    pub context_id: ContextId,
    pub affected_decision_id: DecisionId,
    pub trigger_observation_id: ObservationId,
    pub reason: Option<String>,
    pub basis_diff: BasisDiffResult,
    pub cascade_from: Option<DecisionId>,
    pub severity: InvalidationSeverity,
    pub threatened_intention_ids: Vec<IntentionId>,
    pub intention_now_un_served: bool,
    pub detected_at: TxId,
    pub resolved_at: Option<TxId>,
    pub resolution_action: Option<String>,
    pub resolution_decision_id: Option<DecisionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvalidationDetectedPayload {
    pub invalidation: Invalidation,
    pub decision_description: String,
    pub entity_name: String,
    pub threatened_intention_descriptions: BTreeMap<IntentionId, String>,
    pub cascade_from: Option<DecisionId>,
    pub at_lsn: Lsn,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewDecisionPayload {
    pub decision: Decision,
    pub served_intention_ids: Vec<IntentionId>,
    pub served_intention_descriptions: BTreeMap<IntentionId, String>,
    pub at_lsn: Lsn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConflictDetectedPayload {
    pub context_id: ContextId,
    pub older_decision_id: DecisionId,
    pub newer_decision_id: DecisionId,
    pub threatened_intention_ids: Vec<IntentionId>,
    pub intention_in_tension: bool,
    pub threatened_intention_descriptions: BTreeMap<IntentionId, String>,
    pub at_lsn: Lsn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecedentMatchPayload {
    pub context_id: ContextId,
    pub new_decision_id: DecisionId,
    pub precedent_decision_id: DecisionId,
    pub served_intention_ids: Vec<IntentionId>,
    pub served_intention_descriptions: BTreeMap<IntentionId, String>,
    pub matched_precedent_outcome_success: bool,
    pub at_lsn: Lsn,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SubscriptionFilter {
    pub context_ids: Option<Vec<ContextId>>,
    pub entity_id: Option<EntityId>,
    pub decision_id: Option<DecisionId>,
    pub intention_id: Option<IntentionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SubscriptionId {
    sink_name: String,
    route_name: String,
}

impl SubscriptionId {
    pub fn new(sink_name: impl Into<String>, route_name: impl Into<String>) -> Self {
        Self {
            sink_name: sink_name.into(),
            route_name: route_name.into(),
        }
    }

    pub fn sink_name(&self) -> &str {
        &self.sink_name
    }

    pub(crate) fn route_name(&self) -> &str {
        &self.route_name
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SubscriptionMetrics {
    pub delivered: u64,
    pub queued: u64,
    pub retried: u64,
    pub permanent_failures: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubscriptionError {
    UnknownSubscription(SubscriptionId),
    SinkDdlMissing(String),
    ContextNotAuthorized(ContextId),
    HandlerFailed(String),
    Engine(String),
}

impl std::fmt::Display for SubscriptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SubscriptionError {}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AuditFilter {
    pub decision_id: Option<DecisionId>,
    pub intention_id: Option<IntentionId>,
    pub entity_id: Option<EntityId>,
    pub since_txid: Option<TxId>,
    pub until_txid: Option<TxId>,
    pub target: Option<AuditTarget>,
    pub context_id: Option<ContextId>,
    pub actor: Option<Principal>,
    // Reserved for governance approval queries joining audit_log with approvals.
    pub approver: Option<String>,
    // Reserved until approval levels flow through audit_log.
    pub level: Option<String>,
    // Reserved until audit rows carry tags.
    pub tags: Vec<String>,
    pub limit: Option<u32>,
    pub after: Option<AuditId>,
}

pub const CG_SCHEMA_VERSION: u32 = 9;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreConfig {
    pub db_path: std::path::PathBuf,
    pub https_allowed_hosts: Vec<String>,
    pub file_uri_allowed: bool,
    pub extra_embedding_spaces: Vec<EmbeddingSpaceDecl>,
    pub default_text_embedder: Option<EmbedderConfig>,
    pub expected_schema_version: u32,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            db_path: std::env::temp_dir().join("context-graph.db"),
            https_allowed_hosts: Vec::new(),
            file_uri_allowed: false,
            extra_embedding_spaces: Vec::new(),
            default_text_embedder: Some(EmbedderConfig::default_nomic()),
            expected_schema_version: CG_SCHEMA_VERSION,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacySchemaFixture {
    LegacyDigest384,
    NoVersionRowButTablesExist,
    MultipleVersionRows,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CreateDigest {
    pub id: DigestId,
    pub source: String,
    pub session_ref: Option<String>,
    pub summary: String,
    pub kind: EvidenceKind,
    pub embedding: PrecomputedEmbedding,
    pub context_id: ContextId,
    pub properties: BTreeMap<String, Value>,
}
