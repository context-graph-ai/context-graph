use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::error::CgError;
use crate::types::*;
use crate::{Result, TxId};

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
