use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub mod error;

pub use error::CgError;

pub type Result<T> = std::result::Result<T, CgError>;
pub type TxId = u64;
pub type AgentId = String;

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
    };
}

id_type!(ContextId);
id_type!(IntentionId);
id_type!(DecisionId);
id_type!(EntityId);
id_type!(ObservationId);
id_type!(OutcomeId);
id_type!(DigestId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IntentionStatus {
    #[default]
    Active,
    Paused,
    Superseded,
    Archived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IntentionOrigin {
    #[default]
    Human,
    Agent,
    Signature,
    Blueprint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DecisionStatus {
    #[default]
    Active,
    Superseded,
    Archived,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EntityStatus {
    #[default]
    Active,
    Archived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EntityMatchPolicy {
    #[default]
    Any,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    Serves,
    BasedOn,
    Supersedes,
    HasOutcome,
    DependsOn,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditTarget {
    Decision(DecisionId),
    Intention(IntentionId),
    Entity(EntityId),
    Observation(ObservationId),
    Outcome(DecisionId),
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
    pub confidence: f32,
    pub status: DecisionStatus,
    pub agent_id: Option<AgentId>,
    pub tags: Vec<String>,
    pub properties: BTreeMap<String, Value>,
    pub created_at: TxId,
    pub intention_ids: Vec<IntentionId>,
    pub based_on_snapshots: BTreeMap<EntityId, EntitySnapshot>,
    pub outcome: Option<Outcome>,
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
    pub txid: TxId,
    pub operation: AuditOperation,
    pub actor: AgentId,
    pub reason: Option<String>,
    pub target: AuditTarget,
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
    pub limit: Option<u32>,
    pub after: Option<DecisionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TraverseFilter {
    pub depth: u32,
    pub rel: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AuditFilter {
    pub decision_id: Option<DecisionId>,
    pub intention_id: Option<IntentionId>,
    pub entity_id: Option<EntityId>,
    pub since_txid: Option<TxId>,
    pub until_txid: Option<TxId>,
    pub target: Option<AuditTarget>,
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

#[derive(Debug, Clone)]
pub struct Store {
    db_path: PathBuf,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Self {
        Self {
            db_path: path.as_ref().to_path_buf(),
        }
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn disable_app_decision_scan_for_tests(&self) {}

    pub fn create_context(&self, _input: CreateContext) -> Result<Context> {
        Err(CgError::NotImplemented {
            symbol: "create_context",
        })
    }

    pub fn list_contexts(&self) -> Result<Vec<Context>> {
        Err(CgError::NotImplemented {
            symbol: "list_contexts",
        })
    }

    pub fn get_context(&self, _id: ContextId) -> Result<Option<Context>> {
        Err(CgError::NotImplemented {
            symbol: "get_context",
        })
    }

    pub fn create_intention(&self, _input: CreateIntention) -> Result<Intention> {
        Err(CgError::NotImplemented {
            symbol: "create_intention",
        })
    }

    pub fn get_intention(&self, _id: IntentionId) -> Result<Option<Intention>> {
        Err(CgError::NotImplemented {
            symbol: "get_intention",
        })
    }

    pub fn update_intention(&self, _id: IntentionId, _patch: IntentionPatch) -> Result<Intention> {
        Err(CgError::NotImplemented {
            symbol: "update_intention",
        })
    }

    pub fn update_intentions_batch(
        &self,
        _patches: Vec<(IntentionId, IntentionPatch)>,
    ) -> Result<Vec<Intention>> {
        Err(CgError::NotImplemented {
            symbol: "update_intentions_batch",
        })
    }

    pub fn archive_intention(&self, _id: IntentionId, _reason: String) -> Result<Intention> {
        Err(CgError::NotImplemented {
            symbol: "archive_intention",
        })
    }

    pub fn history(&self, _id: IntentionId) -> Result<Vec<IntentionSnapshot>> {
        Err(CgError::NotImplemented { symbol: "history" })
    }

    pub fn admin_raw_insert_intention(&self, _input: Intention) -> Result<()> {
        Err(CgError::NotImplemented {
            symbol: "admin_raw_insert_intention",
        })
    }

    pub fn admin_raw_scan_intention_rows(&self, _id: IntentionId) -> Result<Vec<Intention>> {
        Err(CgError::NotImplemented {
            symbol: "admin_raw_scan_intention_rows",
        })
    }

    pub fn create_decision(&self, _input: CreateDecision) -> Result<Decision> {
        Err(CgError::NotImplemented {
            symbol: "create_decision",
        })
    }

    pub fn get_decision(&self, _id: DecisionId) -> Result<Option<Decision>> {
        Err(CgError::NotImplemented {
            symbol: "get_decision",
        })
    }

    pub fn list_decisions(&self, _filter: ListDecisionFilter) -> Result<Vec<Decision>> {
        Err(CgError::NotImplemented {
            symbol: "list_decisions",
        })
    }

    pub fn update_decision(&self, _id: DecisionId, _patch: DecisionPatch) -> Result<Decision> {
        Err(CgError::NotImplemented {
            symbol: "update_decision",
        })
    }

    pub fn admin_raw_update_decision_immutable_fields(
        &self,
        _id: DecisionId,
        _decision_type: Option<String>,
        _description: Option<String>,
    ) -> Result<Decision> {
        Err(CgError::NotImplemented {
            symbol: "admin_raw_update_decision_immutable_fields",
        })
    }

    pub fn supersede_decision(&self, _old: DecisionId, _new: CreateDecision) -> Result<Decision> {
        Err(CgError::NotImplemented {
            symbol: "supersede_decision",
        })
    }

    pub fn precedent_chain(&self, _id: DecisionId, _depth: u32) -> Result<Vec<Decision>> {
        Err(CgError::NotImplemented {
            symbol: "precedent_chain",
        })
    }

    pub fn archive_decision(&self, _id: DecisionId, _reason: String) -> Result<Decision> {
        Err(CgError::NotImplemented {
            symbol: "archive_decision",
        })
    }

    pub fn create_entity(&self, _input: CreateEntity) -> Result<Entity> {
        Err(CgError::NotImplemented {
            symbol: "create_entity",
        })
    }

    pub fn get_entity(&self, _id: EntityId) -> Result<Option<Entity>> {
        Err(CgError::NotImplemented {
            symbol: "get_entity",
        })
    }

    pub fn list_entities(&self, _filter: ListEntityFilter) -> Result<Vec<Entity>> {
        Err(CgError::NotImplemented {
            symbol: "list_entities",
        })
    }

    pub fn snapshot_at(&self, _id: EntityId, _at: DateTime<Utc>) -> Result<Option<EntitySnapshot>> {
        Err(CgError::NotImplemented {
            symbol: "snapshot_at",
        })
    }

    pub fn state_at(&self, _id: EntityId, _txid: TxId) -> Result<Option<EntitySnapshot>> {
        Err(CgError::NotImplemented { symbol: "state_at" })
    }

    pub fn update_entity(&self, _id: EntityId, _patch: EntityPatch) -> Result<Entity> {
        Err(CgError::NotImplemented {
            symbol: "update_entity",
        })
    }

    pub fn entity_history(&self, _id: EntityId) -> Result<Vec<EntitySnapshot>> {
        Err(CgError::NotImplemented {
            symbol: "entity_history",
        })
    }

    pub fn archive_entity(&self, _id: EntityId) -> Result<Entity> {
        Err(CgError::NotImplemented {
            symbol: "archive_entity",
        })
    }

    pub fn record_observation(&self, _input: RecordObservation) -> Result<RecordObservationResult> {
        Err(CgError::NotImplemented {
            symbol: "record_observation",
        })
    }

    pub fn get_observation(&self, _id: ObservationId) -> Result<Option<Observation>> {
        Err(CgError::NotImplemented {
            symbol: "get_observation",
        })
    }

    pub fn create_digest(&self, _input: CreateDigest) -> Result<Digest> {
        Err(CgError::NotImplemented {
            symbol: "create_digest",
        })
    }

    pub fn get_digest(&self, _id: DigestId) -> Result<Option<Digest>> {
        Err(CgError::NotImplemented {
            symbol: "get_digest",
        })
    }

    pub fn admin_overwrite_digest(&self, _id: DigestId, _summary: String) -> Result<()> {
        Err(CgError::NotImplemented {
            symbol: "admin_overwrite_digest",
        })
    }

    pub fn admin_overwrite_observation(
        &self,
        _id: ObservationId,
        _data: BTreeMap<String, Value>,
    ) -> Result<()> {
        Err(CgError::NotImplemented {
            symbol: "admin_overwrite_observation",
        })
    }

    pub fn record_outcome(
        &self,
        _decision_id: DecisionId,
        _success: bool,
        _notes: Option<String>,
    ) -> Result<Outcome> {
        Err(CgError::NotImplemented {
            symbol: "record_outcome",
        })
    }

    pub fn audit_query(&self, _filter: AuditFilter) -> Result<Vec<AuditEntry>> {
        Err(CgError::NotImplemented {
            symbol: "audit_query",
        })
    }

    pub fn admin_raw_has_outcome_edge_count(&self, _decision_id: DecisionId) -> Result<u64> {
        Err(CgError::NotImplemented {
            symbol: "admin_raw_has_outcome_edge_count",
        })
    }

    pub fn admin_raw_outcomes_for_decision(
        &self,
        _decision_id: DecisionId,
    ) -> Result<Vec<Outcome>> {
        Err(CgError::NotImplemented {
            symbol: "admin_raw_outcomes_for_decision",
        })
    }

    pub fn add_relationship(&self, _from: EntityId, _to: EntityId, _rel: String) -> Result<()> {
        Err(CgError::NotImplemented {
            symbol: "add_relationship",
        })
    }

    pub fn traverse(&self, _from: EntityId, _filter: TraverseFilter) -> Result<Vec<EntityId>> {
        Err(CgError::NotImplemented { symbol: "traverse" })
    }

    pub fn admin_raw_edge_count(
        &self,
        _from: EntityId,
        _rel: String,
        _to: EntityId,
    ) -> Result<u64> {
        Err(CgError::NotImplemented {
            symbol: "admin_raw_edge_count",
        })
    }

    pub fn admin_raw_edge_scan(&self, _rel: String) -> Result<Vec<RawEdge>> {
        Err(CgError::NotImplemented {
            symbol: "admin_raw_edge_scan",
        })
    }

    pub fn admin_raw_scan(&self, _key: StorageKey) -> Result<Option<RawRow>> {
        Err(CgError::NotImplemented {
            symbol: "admin_raw_scan",
        })
    }

    pub fn edge_exists(
        &self,
        _from: DecisionId,
        _rel: RelationshipType,
        _to: DecisionId,
    ) -> Result<bool> {
        Err(CgError::NotImplemented {
            symbol: "edge_exists",
        })
    }

    pub fn decision_targets_intention(
        &self,
        _decision_id: DecisionId,
        _rel: RelationshipType,
        _intention_id: IntentionId,
    ) -> Result<bool> {
        Err(CgError::NotImplemented {
            symbol: "decision_targets_intention",
        })
    }

    pub fn decision_targets_entity(
        &self,
        _decision_id: DecisionId,
        _rel: RelationshipType,
        _entity_id: EntityId,
    ) -> Result<bool> {
        Err(CgError::NotImplemented {
            symbol: "decision_targets_entity",
        })
    }

    pub fn incoming_supersedes(&self, _decision_id: DecisionId) -> Result<Vec<DecisionId>> {
        Err(CgError::NotImplemented {
            symbol: "incoming_supersedes",
        })
    }

    pub fn admin_raw_observations_for_entity_in_context(
        &self,
        _entity_id: EntityId,
        _context_id: ContextId,
    ) -> Result<Vec<ObservationId>> {
        Err(CgError::NotImplemented {
            symbol: "admin_raw_observations_for_entity_in_context",
        })
    }

    pub fn last_query_trace(&self) -> Result<String> {
        Err(CgError::NotImplemented {
            symbol: "last_query_trace",
        })
    }
}
