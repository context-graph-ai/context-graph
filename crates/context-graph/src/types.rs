use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::{AgentId, TxId};

macro_rules! id_type {
    ($name:ident) => {
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
            fn default() -> Self {
                Self::new_v7()
            }
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
id_type!(ObservationId);
id_type!(OutcomeId);
id_type!(DigestId);
id_type!(AuditId);
id_type!(InvalidationId);

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
pub enum DecisionStatus {
    #[default]
    Active,
    Superseded,
    Archived,
}

impl DecisionStatus {
    pub const ALL: [Self; 3] = [Self::Active, Self::Superseded, Self::Archived];
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
}

impl AuditOperation {
    pub const ALL: [Self; 6] = [
        Self::Create,
        Self::Update,
        Self::Archive,
        Self::Supersede,
        Self::RecordOutcome,
        Self::RecordObservation,
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
            Principal::System => "System".to_string(),
            Principal::Agent(a) => a.clone(),
            Principal::Human(h) => h.clone(),
        }
    }

    pub fn decode(raw: &str) -> Self {
        if raw == "System" {
            return Principal::System;
        }
        if raw.starts_with("agent:") || raw.starts_with("claude-code-instance-") {
            return Principal::Agent(raw.to_string());
        }
        Principal::Human(raw.to_string())
    }
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
    pub based_on_snapshots: BTreeMap<EntityId, EntitySnapshot>,
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
    pub entity_id: EntityId,
    pub entity_type: EntityType,
    pub name: String,
    pub properties: BTreeMap<String, Value>,
    pub captured_at: TxId,
    pub valid_from: TxId,
    pub valid_to: Option<TxId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub id: ObservationId,
    pub entity_id: EntityId,
    pub context_id: ContextId,
    pub observation_type: String,
    pub source: String,
    pub data: BTreeMap<String, Value>,
    pub observed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordObservationResult {
    pub observation_id: ObservationId,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Digest {
    pub id: DigestId,
    pub source: String,
    pub summary: String,
    pub embedding: Vec<f32>,
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
pub struct ReplayResult {
    pub decision_id: DecisionId,
    pub original_outcome: Option<bool>,
    pub inputs_changed: bool,
    pub changed_input_count: u32,
    pub unchanged_input_count: u32,
    pub input_diffs: Vec<InputDiff>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordObservation {
    pub entity_id: EntityId,
    pub context_id: ContextId,
    pub observation_type: String,
    pub source: String,
    pub data: BTreeMap<String, Value>,
    pub observed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ListDecisionFilter {
    pub intention_id: Option<IntentionId>,
    pub tags: Vec<String>,
    pub based_on_entity_ids: Vec<EntityId>,
    pub entity_match_policy: EntityMatchPolicy,
    pub status: Option<DecisionStatus>,
    pub context_id: Option<ContextId>,
    pub limit: Option<u32>,
    pub after: Option<DecisionId>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Invalidation {
    pub id: InvalidationId,
    pub status: InvalidationStatus,
    pub affected_decision_id: DecisionId,
    pub trigger_observation_id: ObservationId,
    pub basis_diff: BTreeMap<String, Value>,
    pub severity: InvalidationSeverity,
    pub detected_at: TxId,
    pub resolved_at: Option<TxId>,
    pub resolution_action: Option<String>,
    pub resolution_decision_id: Option<DecisionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AuditFilter {
    pub decision_id: Option<DecisionId>,
    pub intention_id: Option<IntentionId>,
    pub entity_id: Option<EntityId>,
    pub since_txid: Option<TxId>,
    pub until_txid: Option<TxId>,
    pub target: Option<AuditTarget>,
    pub context_id: Option<ContextId>,
    // consumed by plan 9 (cg-governance) joining audit_log with approvals; ignored here.
    pub approver: Option<String>,
    // consumed by plan 9 (cg-governance); ignored here until approval levels flow through audit_log.
    pub level: Option<String>,
    // consumed by plan 9 (cg-governance); ignored here until audit rows carry tags.
    pub tags: Vec<String>,
    pub limit: Option<u32>,
    pub after: Option<AuditId>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CreateDigest {
    pub source: String,
    pub summary: String,
    pub embedding: Vec<f32>,
    pub context_id: ContextId,
    pub session_ref: Option<String>,
    pub properties: BTreeMap<String, Value>,
}
