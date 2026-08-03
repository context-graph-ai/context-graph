use std::collections::{BTreeSet, HashMap, HashSet};

use contextdb_core::{Error as EngineError, Value};
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::error::CgError;
use crate::types::*;
use crate::{Result, TxId};

use super::Store;
use super::convert::{
    intention_origin_from_str, intention_origin_str, intention_status_from_str,
    intention_status_str, map_engine_err,
};
use super::encode::{
    basis_diff_from_json, decode_outcome, get_bool, get_json, get_optional_text, get_optional_txid,
    get_text, get_txid, get_uuid, intention_ids_from_json, properties_from_json,
    properties_to_json, tags_from_json,
};
use super::relationship::uuid_or_clause;

impl Store {
    pub fn create_intention(&self, input: CreateIntention) -> Result<Intention> {
        if input.id.is_some() {
            return Err(CgError::InvalidIdentity {
                context: "IntentionId",
            });
        }
        self.require_context(input.context_id)?;
        let id = IntentionId::new_v7();
        let properties_json = properties_to_json(&input.properties);
        let tags_json = JsonValue::Array(
            input
                .tags
                .iter()
                .map(|t| JsonValue::String(t.clone()))
                .collect(),
        );

        let tx = self.begin_tx()?;
        let blueprint_catalog_id_value = input
            .blueprint_catalog_id
            .as_deref()
            .map(|s| Value::Text(s.to_string()))
            .unwrap_or(Value::Null);
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            (
                "description".to_string(),
                Value::Text(input.description.clone()),
            ),
            (
                "status".to_string(),
                Value::Text(intention_status_str(input.status).to_string()),
            ),
            (
                "context_id".to_string(),
                Value::Uuid(input.context_id.as_uuid()),
            ),
            (
                "origin".to_string(),
                Value::Text(intention_origin_str(input.origin).to_string()),
            ),
            ("origin_ref".to_string(), Value::Null),
            ("blueprint_id".to_string(), Value::Null),
            ("bindings".to_string(), Value::Null),
            ("blueprint_proposal".to_string(), Value::Null),
            (
                "blueprint_catalog_id".to_string(),
                blueprint_catalog_id_value,
            ),
            (
                "properties".to_string(),
                Value::Json(properties_json.clone()),
            ),
            ("tags".to_string(), Value::Json(tags_json.clone())),
            ("created_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO intentions (id, description, status, context_id, origin, origin_ref, blueprint_id, bindings, blueprint_proposal, blueprint_catalog_id, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $origin_ref, $blueprint_id, $bindings, $blueprint_proposal, $blueprint_catalog_id, $properties, $tags, $created_at)",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        let snapshot_id = match self.insert_intention_snapshot_in_tx(
            tx,
            input.context_id,
            id,
            &input.description,
            input.status,
            input.origin,
            &properties_json,
            &tags_json,
            tx,
        ) {
            Ok(snapshot_id) => snapshot_id,
            Err(e) => {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        };
        if let Err(e) =
            self.set_current_intention_snapshot_in_tx(tx, input.context_id, id, snapshot_id)
        {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Create,
            Principal::System,
            None,
            AuditTarget::Intention(id),
            input.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        Ok(Intention {
            id,
            description: input.description,
            status: input.status,
            origin: input.origin,
            context_id: input.context_id,
            properties: input.properties,
            tags: input.tags,
            created_at: tx,
            blueprint_catalog_id: input.blueprint_catalog_id,
        })
    }

    pub fn get_intention(&self, id: IntentionId) -> Result<Option<Intention>> {
        self.observe_read("get_intention");
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "IntentionId",
            });
        }
        let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT id, description, status, origin, context_id, blueprint_catalog_id, properties, tags, created_at FROM intentions WHERE id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        Ok(Some(decode_intention(&result.columns, &result.rows[0])?))
    }

    pub fn intention_tree(&self, intention_id: IntentionId) -> Result<IntentionTree> {
        let intention = self.get_intention(intention_id)?.ok_or(CgError::NotFound {
            id: Some(intention_id.as_uuid().to_string()),
            missing: Some("intention"),
        })?;
        let mut decisions = self.decisions_serving_intention(intention.id, intention.context_id)?;
        decisions = self.sort_decisions_topologically(decisions)?;

        let mut entity_ids = BTreeSet::new();
        for decision in &decisions {
            for entity_id in self.edge_targets(
                intention.context_id,
                decision.id.as_uuid(),
                RelationshipType::BasedOn,
            )? {
                entity_ids.insert(EntityId::from(entity_id));
            }
        }
        let mut entities = Vec::new();
        for entity_id in entity_ids {
            if let Some(entity) = self.get_entity(entity_id)?
                && entity.context_id == intention.context_id
            {
                entities.push(entity);
            }
        }

        let decision_ids: Vec<DecisionId> = decisions.iter().map(|d| d.id).collect();
        let mut outcomes = self.outcomes_for_decisions(&decision_ids)?;
        outcomes.sort_by(|a, b| {
            a.recorded_at
                .0
                .cmp(&b.recorded_at.0)
                .then_with(|| a.id.as_uuid().cmp(&b.id.as_uuid()))
        });
        let mut invalidations = self.pending_invalidations_for_decisions(&decision_ids)?;
        invalidations.sort_by(|a, b| {
            severity_rank(b.severity)
                .cmp(&severity_rank(a.severity))
                .then_with(|| a.detected_at.0.cmp(&b.detected_at.0))
                .then_with(|| a.id.as_uuid().cmp(&b.id.as_uuid()))
        });

        Ok(IntentionTree {
            intention,
            decisions,
            entities,
            outcomes,
            invalidations,
        })
    }

    pub(super) fn decisions_serving_intention(
        &self,
        intention_id: IntentionId,
        context_id: ContextId,
    ) -> Result<Vec<Decision>> {
        let params: HashMap<String, Value> = [
            ("target_id".to_string(), Value::Uuid(intention_id.as_uuid())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("edge_type".to_string(), Value::Text("SERVES".to_string())),
        ]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT source_id FROM edges WHERE target_id = $target_id AND context_id = $context_id AND edge_type = $edge_type",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            if let Some(Value::Uuid(u)) = row.first() {
                let id = DecisionId::from(*u);
                if let Some(decision) = self.get_decision(id)?
                    && decision.context_id == context_id
                {
                    out.push(decision);
                }
            }
        }
        out.sort_by(|a, b| {
            a.created_at
                .0
                .cmp(&b.created_at.0)
                .then_with(|| a.id.as_uuid().cmp(&b.id.as_uuid()))
        });
        Ok(out)
    }

    pub(super) fn sort_decisions_topologically(
        &self,
        decisions: Vec<Decision>,
    ) -> Result<Vec<Decision>> {
        let by_id: HashMap<DecisionId, Decision> =
            decisions.into_iter().map(|d| (d.id, d)).collect();
        let ids: HashSet<DecisionId> = by_id.keys().copied().collect();
        let mut remaining_precedents: HashMap<DecisionId, HashSet<DecisionId>> = HashMap::new();
        for id in &ids {
            let decision = by_id.get(id).expect("id from map keys");
            let precedents = self
                .precedents_for_decision(*id, decision.context_id)?
                .into_iter()
                .filter(|pid| ids.contains(pid))
                .collect();
            remaining_precedents.insert(*id, precedents);
        }

        let mut out = Vec::with_capacity(by_id.len());
        while out.len() < by_id.len() {
            let mut ready: Vec<DecisionId> = remaining_precedents
                .iter()
                .filter_map(|(id, precedents)| precedents.is_empty().then_some(*id))
                .collect();
            if ready.is_empty() {
                ready = remaining_precedents.keys().copied().collect();
            }
            ready.sort_by(|a, b| {
                let da = by_id.get(a).expect("ready id");
                let db = by_id.get(b).expect("ready id");
                da.created_at
                    .0
                    .cmp(&db.created_at.0)
                    .then_with(|| da.id.as_uuid().cmp(&db.id.as_uuid()))
            });
            let next = ready[0];
            let decision = by_id.get(&next).expect("next id").clone();
            remaining_precedents.remove(&next);
            for precedents in remaining_precedents.values_mut() {
                precedents.remove(&next);
            }
            out.push(decision);
        }
        Ok(out)
    }

    fn outcomes_for_decisions(&self, decision_ids: &[DecisionId]) -> Result<Vec<Outcome>> {
        if decision_ids.is_empty() {
            return Ok(Vec::new());
        }
        let uuids: Vec<Uuid> = decision_ids.iter().map(|id| id.as_uuid()).collect();
        let mut params = HashMap::new();
        let clause = uuid_or_clause("decision_id", "decision", &uuids, &mut params);
        let sql = format!(
            "SELECT id, decision_id, success, notes, recorded_at FROM outcomes WHERE {clause}"
        );
        let result = self.db.execute(&sql, &params).map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            out.push(decode_outcome(&result.columns, row)?);
        }
        Ok(out)
    }

    fn pending_invalidations_for_decisions(
        &self,
        decision_ids: &[DecisionId],
    ) -> Result<Vec<Invalidation>> {
        if decision_ids.is_empty() {
            return Ok(Vec::new());
        }
        let uuids: Vec<Uuid> = decision_ids.iter().map(|id| id.as_uuid()).collect();
        let mut params: HashMap<String, Value> =
            [("status".to_string(), Value::Text("pending".to_string()))]
                .into_iter()
                .collect();
        let clause = uuid_or_clause("affected_decision_id", "decision", &uuids, &mut params);
        let sql = format!(
            "SELECT id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id FROM invalidations WHERE status = $status AND ({clause})"
        );
        let result = self.db.execute(&sql, &params).map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            out.push(decode_invalidation(&result.columns, row)?);
        }
        Ok(out)
    }

    pub fn update_intention(&self, id: IntentionId, patch: IntentionPatch) -> Result<Intention> {
        let results = self.update_intentions_batch(vec![(id, patch)])?;
        results.into_iter().next().ok_or(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("intention"),
        })
    }

    pub fn update_intentions_batch(
        &self,
        patches: Vec<(IntentionId, IntentionPatch)>,
    ) -> Result<Vec<Intention>> {
        let tx = self.begin_tx()?;
        let mut currents: Vec<Intention> = Vec::with_capacity(patches.len());
        for (id, _) in &patches {
            match self.get_intention_in_tx(tx, *id) {
                Ok(Some(i)) => currents.push(i),
                Err(error) => {
                    let _ = self.db.rollback(tx);
                    return Err(error);
                }
                Ok(None) => {
                    let _ = self.db.rollback(tx);
                    return Err(CgError::NotFound {
                        id: Some(id.as_uuid().to_string()),
                        missing: Some("intention"),
                    });
                }
            }
        }

        let mut ok = true;
        let mut err: Option<CgError> = None;
        for (current, patch) in currents.iter().zip(patches.iter().map(|(_, p)| p)) {
            if let Err(e) = self.apply_intention_patch_in_tx(tx, current, patch) {
                err = Some(e);
                ok = false;
                break;
            }
        }
        if !ok {
            let _ = self.db.rollback(tx);
            return Err(err.expect("error recorded"));
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        let mut out = Vec::with_capacity(patches.len());
        for (id, _) in &patches {
            out.push(self.get_intention(*id)?.ok_or_else(|| CgError::NotFound {
                id: Some(id.as_uuid().to_string()),
                missing: Some("intention"),
            })?);
        }
        Ok(out)
    }

    fn apply_intention_patch_in_tx(
        &self,
        tx: TxId,
        current: &Intention,
        patch: &IntentionPatch,
    ) -> Result<()> {
        let has_any = patch.description.is_some()
            || patch.properties.is_some()
            || patch.tags.is_some()
            || patch.status.is_some();
        if !has_any {
            return Ok(());
        }
        let new_description = patch
            .description
            .clone()
            .unwrap_or_else(|| current.description.clone());
        let new_properties = patch
            .properties
            .clone()
            .unwrap_or_else(|| current.properties.clone());
        let new_tags = patch.tags.clone().unwrap_or_else(|| current.tags.clone());
        let new_status = patch.status.unwrap_or(current.status);

        if let Some(desc) = &patch.description {
            let params: HashMap<String, Value> = [
                ("id".to_string(), Value::Uuid(current.id.as_uuid())),
                ("description".to_string(), Value::Text(desc.clone())),
            ]
            .into_iter()
            .collect();
            self.db
                .execute_in_tx(
                    tx,
                    "UPDATE intentions SET description = $description WHERE id = $id",
                    &params,
                )
                .map_err(map_engine_err)?;
        }
        if let Some(properties) = &patch.properties {
            let params: HashMap<String, Value> = [
                ("id".to_string(), Value::Uuid(current.id.as_uuid())),
                (
                    "properties".to_string(),
                    Value::Json(properties_to_json(properties)),
                ),
            ]
            .into_iter()
            .collect();
            self.db
                .execute_in_tx(
                    tx,
                    "UPDATE intentions SET properties = $properties WHERE id = $id",
                    &params,
                )
                .map_err(map_engine_err)?;
        }
        if let Some(tags) = &patch.tags {
            let tags_json =
                JsonValue::Array(tags.iter().map(|t| JsonValue::String(t.clone())).collect());
            let params: HashMap<String, Value> = [
                ("id".to_string(), Value::Uuid(current.id.as_uuid())),
                ("tags".to_string(), Value::Json(tags_json)),
            ]
            .into_iter()
            .collect();
            self.db
                .execute_in_tx(
                    tx,
                    "UPDATE intentions SET tags = $tags WHERE id = $id",
                    &params,
                )
                .map_err(map_engine_err)?;
        }
        if let Some(status) = patch.status {
            let params: HashMap<String, Value> = [
                ("id".to_string(), Value::Uuid(current.id.as_uuid())),
                (
                    "status".to_string(),
                    Value::Text(intention_status_str(status).to_string()),
                ),
            ]
            .into_iter()
            .collect();
            self.db
                .execute_in_tx(
                    tx,
                    "UPDATE intentions SET status = $status WHERE id = $id",
                    &params,
                )
                .map_err(|e| match e {
                    EngineError::InvalidStateTransition(_)
                    | EngineError::PropagationAborted { .. } => {
                        CgError::IntentionIllegalTransition {
                            from: current.status,
                            to: status,
                        }
                    }
                    other => map_engine_err(other),
                })?;
        }

        // Snapshots are write-once.  A concurrent/offline update branches from the exact
        // predecessor it observed; it never rewrites that predecessor's validity interval.
        let predecessor_snapshot_id = self
            .current_intention_snapshot_id_in_tx(tx, current.id)?
            .ok_or_else(|| {
                CgError::Engine(format!(
                    "data-integrity: intention {} has no current snapshot pointer",
                    current.id
                ))
            })?;
        let new_properties_json = properties_to_json(&new_properties);
        let new_tags_json = JsonValue::Array(
            new_tags
                .iter()
                .map(|t| JsonValue::String(t.clone()))
                .collect(),
        );
        let successor_snapshot_id = self.insert_intention_snapshot_in_tx(
            tx,
            current.context_id,
            current.id,
            &new_description,
            new_status,
            current.origin,
            &new_properties_json,
            &new_tags_json,
            tx,
        )?;
        self.insert_intention_snapshot_link_in_tx(
            tx,
            current.context_id,
            current.id,
            predecessor_snapshot_id,
            successor_snapshot_id,
        )?;
        self.set_current_intention_snapshot_in_tx(
            tx,
            current.context_id,
            current.id,
            successor_snapshot_id,
        )?;
        self.record_audit_in_tx(
            tx,
            AuditOperation::Update,
            Principal::System,
            None,
            AuditTarget::Intention(current.id),
            current.context_id,
            None,
        )?;
        Ok(())
    }

    /// Create a successor from one explicitly named immutable snapshot.
    ///
    /// Offline writers use this seam after they have both observed the same predecessor. Each
    /// writer's values and predecessor link are read and written in one transaction, so a later
    /// current-pointer winner cannot silently change what the other branch meant.
    pub fn update_intention_on_branch(
        &self,
        id: IntentionId,
        predecessor_snapshot_id: IntentionSnapshotId,
        patch: IntentionPatch,
    ) -> Result<IntentionSnapshot> {
        if !id.is_valid() || !predecessor_snapshot_id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "IntentionSnapshotId",
            });
        }
        let tx = self.begin_tx()?;
        let successor =
            match self.update_intention_on_branch_in_tx(tx, id, predecessor_snapshot_id, &patch) {
                Ok(successor) => successor,
                Err(error) => {
                    let _ = self.db.rollback(tx);
                    return Err(error);
                }
            };
        self.db.commit(tx).map_err(map_engine_err)?;
        self.history(id)?
            .into_iter()
            .find(|snapshot| snapshot.id == successor.id)
            .ok_or_else(|| {
                CgError::Engine(format!(
                    "data-integrity: committed intention snapshot {} is not readable",
                    successor.id
                ))
            })
    }

    fn update_intention_on_branch_in_tx(
        &self,
        tx: TxId,
        id: IntentionId,
        predecessor_snapshot_id: IntentionSnapshotId,
        patch: &IntentionPatch,
    ) -> Result<IntentionSnapshot> {
        let current = self
            .get_intention_in_tx(tx, id)?
            .ok_or(CgError::IntentionNotFound { id })?;
        let predecessor = self
            .intention_snapshot_in_tx(tx, current.context_id, id, predecessor_snapshot_id)?
            .ok_or(CgError::NotFound {
                id: Some(predecessor_snapshot_id.as_uuid().to_string()),
                missing: Some("intention_snapshot"),
            })?;

        let description = patch
            .description
            .clone()
            .unwrap_or_else(|| predecessor.description.clone());
        let properties = patch
            .properties
            .clone()
            .unwrap_or_else(|| predecessor.properties.clone());
        let tags = patch
            .tags
            .clone()
            .unwrap_or_else(|| predecessor.tags.clone());
        let status = patch.status.unwrap_or(predecessor.status);
        let properties_json = properties_to_json(&properties);
        let tags_json = JsonValue::Array(tags.iter().cloned().map(JsonValue::String).collect());

        self.db
            .execute_in_tx(
                tx,
                "UPDATE intentions SET description = $description, properties = $properties, tags = $tags WHERE id = $id AND context_id = $context_id",
                &HashMap::from([
                    ("id".to_string(), Value::Uuid(id.as_uuid())),
                    ("context_id".to_string(), Value::Uuid(current.context_id.as_uuid())),
                    ("description".to_string(), Value::Text(description.clone())),
                    ("properties".to_string(), Value::Json(properties_json.clone())),
                    ("tags".to_string(), Value::Json(tags_json.clone())),
                ]),
            )
            .map_err(map_engine_err)?;
        if patch.status.is_some() {
            self.db
                .execute_in_tx(
                    tx,
                    "UPDATE intentions SET status = $status WHERE id = $id AND context_id = $context_id",
                    &HashMap::from([
                        ("id".to_string(), Value::Uuid(id.as_uuid())),
                        ("context_id".to_string(), Value::Uuid(current.context_id.as_uuid())),
                        (
                            "status".to_string(),
                            Value::Text(intention_status_str(status).to_string()),
                        ),
                    ]),
                )
                .map_err(|error| match error {
                    EngineError::InvalidStateTransition(_)
                    | EngineError::PropagationAborted { .. } => {
                        CgError::IntentionIllegalTransition {
                            from: current.status,
                            to: status,
                        }
                    }
                    other => map_engine_err(other),
                })?;
        }

        let successor_snapshot_id = self.insert_intention_snapshot_in_tx(
            tx,
            current.context_id,
            id,
            &description,
            status,
            predecessor.origin,
            &properties_json,
            &tags_json,
            tx,
        )?;
        self.insert_intention_snapshot_link_in_tx(
            tx,
            current.context_id,
            id,
            predecessor_snapshot_id,
            successor_snapshot_id,
        )?;
        self.set_current_intention_snapshot_in_tx(
            tx,
            current.context_id,
            id,
            successor_snapshot_id,
        )?;
        self.record_audit_in_tx(
            tx,
            AuditOperation::Update,
            Principal::System,
            None,
            AuditTarget::Intention(id),
            current.context_id,
            None,
        )?;
        Ok(IntentionSnapshot {
            id: successor_snapshot_id,
            intention_id: id,
            description,
            status,
            origin: predecessor.origin,
            properties,
            tags,
            valid_from: tx,
            valid_to: None,
        })
    }

    pub fn archive_intention(&self, id: IntentionId, reason: String) -> Result<Intention> {
        let tx = self.begin_tx()?;
        let current = match self.get_intention_in_tx(tx, id) {
            Ok(Some(current)) => current,
            Err(error) => {
                let _ = self.db.rollback(tx);
                return Err(error);
            }
            Ok(None) => {
                let _ = self.db.rollback(tx);
                return Err(CgError::NotFound {
                    id: Some(id.as_uuid().to_string()),
                    missing: Some("intention"),
                });
            }
        };
        let active_serving_result = match self
            .db
            .execute_in_tx(
                tx,
                "WITH serving_decisions AS (SELECT b_id FROM GRAPH_TABLE(edges MATCH (a)<-[:SERVES]-(b) WHERE a.id = $intention_id COLUMNS (b.id AS b_id))) SELECT d.id FROM decisions d INNER JOIN serving_decisions sd ON d.id = sd.b_id WHERE d.context_id = $context_id AND d.status = $status",
                &HashMap::from([(
                    "intention_id".to_string(),
                    Value::Uuid(id.as_uuid()),
                ), (
                    "context_id".to_string(),
                    Value::Uuid(current.context_id.as_uuid()),
                ), (
                    "status".to_string(),
                    Value::Text("active".to_string()),
                )]),
            ) {
            Ok(result) => result,
            Err(error) => {
                let _ = self.db.rollback(tx);
                return Err(map_engine_err(error));
            }
        };
        let active_serving: Vec<DecisionId> = active_serving_result
            .rows
            .iter()
            .filter_map(|row| match row.first() {
                Some(Value::Uuid(source)) => Some(DecisionId::from(*source)),
                _ => None,
            })
            .collect();
        if !active_serving.is_empty() {
            let _ = self.db.rollback(tx);
            return Err(CgError::ActiveDecisionsReference {
                intention_id: id,
                decision_ids: active_serving
                    .iter()
                    .map(|d| d.as_uuid().to_string())
                    .collect(),
            });
        }

        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("status".to_string(), Value::Text("archived".to_string())),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "UPDATE intentions SET status = $status WHERE id = $id",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        let predecessor_snapshot_id = match self.current_intention_snapshot_id_in_tx(tx, id) {
            Ok(Some(snapshot_id)) => snapshot_id,
            Ok(None) => {
                let _ = self.db.rollback(tx);
                return Err(CgError::Engine(format!(
                    "data-integrity: intention {} has no current snapshot pointer",
                    id
                )));
            }
            Err(error) => {
                let _ = self.db.rollback(tx);
                return Err(error);
            }
        };
        let properties_json = properties_to_json(&current.properties);
        let tags_json = JsonValue::Array(
            current
                .tags
                .iter()
                .cloned()
                .map(JsonValue::String)
                .collect(),
        );
        let archived_snapshot_id = match self.insert_intention_snapshot_in_tx(
            tx,
            current.context_id,
            id,
            &current.description,
            IntentionStatus::Archived,
            current.origin,
            &properties_json,
            &tags_json,
            tx,
        ) {
            Ok(snapshot_id) => snapshot_id,
            Err(e) => {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        };
        if let Err(e) = self.insert_intention_snapshot_link_in_tx(
            tx,
            current.context_id,
            id,
            predecessor_snapshot_id,
            archived_snapshot_id,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        if let Err(e) = self.set_current_intention_snapshot_in_tx(
            tx,
            current.context_id,
            id,
            archived_snapshot_id,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Archive,
            Principal::System,
            Some(reason),
            AuditTarget::Intention(id),
            current.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        Ok(Intention {
            status: IntentionStatus::Archived,
            ..current
        })
    }

    pub fn history(&self, id: IntentionId) -> Result<Vec<IntentionSnapshot>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "IntentionId",
            });
        }
        let params: HashMap<String, Value> =
            [("intention_id".to_string(), Value::Uuid(id.as_uuid()))]
                .into_iter()
                .collect();
        let result = self
            .db
            .execute(
                "SELECT id, intention_id, description, status, origin, properties, tags, valid_from, valid_to FROM intention_snapshots WHERE intention_id = $intention_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut snaps: Vec<IntentionSnapshot> = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            snaps.push(decode_intention_snapshot(&result.columns, row)?);
        }
        snaps.sort_by_key(|s| (s.valid_from.0, s.id.as_uuid()));
        Ok(snaps)
    }

    pub fn intention_at(&self, id: IntentionId, at: TxId) -> Result<Option<IntentionSnapshot>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "IntentionId",
            });
        }
        let history = self.history(id)?;
        let eligible: HashSet<IntentionSnapshotId> = history
            .iter()
            .filter(|snapshot| snapshot.valid_from <= at)
            .map(|snapshot| snapshot.id)
            .collect();
        if eligible.is_empty() {
            return Ok(None);
        }
        let linked_predecessors: HashSet<IntentionSnapshotId> = self
            .intention_snapshot_link_pairs(id)?
            .into_iter()
            .filter_map(|(predecessor, successor)| {
                eligible.contains(&successor).then_some(predecessor)
            })
            .collect();
        let mut heads: Vec<IntentionSnapshotId> = eligible
            .iter()
            .copied()
            .filter(|snapshot_id| !linked_predecessors.contains(snapshot_id))
            .collect();
        heads.sort_by_key(|snapshot_id| snapshot_id.as_uuid());
        match heads.as_slice() {
            [] => Err(CgError::Engine(format!(
                "data-integrity: intention {} has no history head at {}",
                id, at.0
            ))),
            [head] => self.intention_at_on_branch(id, at, *head),
            _ => Err(CgError::AmbiguousIntentionHistory {
                intention_id: id,
                at,
                branch_head_ids: heads,
            }),
        }
    }

    /// Read one named immutable history branch.  This is the only branch-aware temporal read:
    /// callers must choose when offline edits produced two legitimate successor meanings.
    pub fn intention_at_on_branch(
        &self,
        id: IntentionId,
        at: TxId,
        branch_head_id: IntentionSnapshotId,
    ) -> Result<Option<IntentionSnapshot>> {
        if !id.is_valid() || !branch_head_id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "IntentionSnapshotId",
            });
        }
        let history = self.history(id)?;
        let by_id: HashMap<IntentionSnapshotId, IntentionSnapshot> = history
            .into_iter()
            .map(|snapshot| (snapshot.id, snapshot))
            .collect();
        if !by_id.contains_key(&branch_head_id) {
            return Err(CgError::NotFound {
                id: Some(branch_head_id.as_uuid().to_string()),
                missing: Some("intention_snapshot"),
            });
        }
        let predecessors: HashMap<IntentionSnapshotId, IntentionSnapshotId> = self
            .intention_snapshot_link_pairs(id)?
            .into_iter()
            .map(|(predecessor, successor)| (successor, predecessor))
            .collect();
        let mut cursor = branch_head_id;
        let mut lineage = HashSet::new();
        loop {
            if !lineage.insert(cursor) {
                return Err(CgError::Engine(format!(
                    "data-integrity: cyclic intention history for {}",
                    id
                )));
            }
            let Some(predecessor) = predecessors.get(&cursor).copied() else {
                break;
            };
            cursor = predecessor;
        }
        Ok(lineage
            .into_iter()
            .filter_map(|snapshot_id| by_id.get(&snapshot_id).cloned())
            .filter(|snapshot| snapshot.valid_from <= at)
            .max_by_key(|snapshot| (snapshot.valid_from.0, snapshot.id.as_uuid())))
    }

    pub fn admin_raw_insert_intention(&self, input: Intention) -> Result<()> {
        let existing: HashMap<String, Value> =
            [("id".to_string(), Value::Uuid(input.id.as_uuid()))]
                .into_iter()
                .collect();
        let result = self
            .db
            .execute("SELECT id FROM intentions WHERE id = $id", &existing)
            .map_err(map_engine_err)?;
        if !result.rows.is_empty() {
            return Err(CgError::UniqueViolation {
                entity: "intention",
                id: Some(input.id.as_uuid().to_string()),
            });
        }
        let tags_json = JsonValue::Array(
            input
                .tags
                .iter()
                .map(|t| JsonValue::String(t.clone()))
                .collect(),
        );
        let tx = self.begin_tx()?;
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(input.id.as_uuid())),
            (
                "description".to_string(),
                Value::Text(input.description.clone()),
            ),
            (
                "status".to_string(),
                Value::Text(intention_status_str(input.status).to_string()),
            ),
            (
                "context_id".to_string(),
                Value::Uuid(input.context_id.as_uuid()),
            ),
            (
                "origin".to_string(),
                Value::Text(intention_origin_str(input.origin).to_string()),
            ),
            ("origin_ref".to_string(), Value::Null),
            ("blueprint_id".to_string(), Value::Null),
            ("bindings".to_string(), Value::Null),
            ("blueprint_proposal".to_string(), Value::Null),
            (
                "blueprint_catalog_id".to_string(),
                input
                    .blueprint_catalog_id
                    .as_deref()
                    .map(|s| Value::Text(s.to_string()))
                    .unwrap_or(Value::Null),
            ),
            (
                "properties".to_string(),
                Value::Json(properties_to_json(&input.properties)),
            ),
            ("tags".to_string(), Value::Json(tags_json)),
            ("created_at".to_string(), Value::TxId(input.created_at)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO intentions (id, description, status, context_id, origin, origin_ref, blueprint_id, bindings, blueprint_proposal, blueprint_catalog_id, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $origin_ref, $blueprint_id, $bindings, $blueprint_proposal, $blueprint_catalog_id, $properties, $tags, $created_at)",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        self.db.commit(tx).map_err(map_engine_err)?;
        Ok(())
    }

    /// Returns all intentions stored under the given context, ordered by creation time.
    /// A read-only helper for verifying context-scoped capture output in tests.
    pub fn list_intentions_in_context(&self, context_id: ContextId) -> Result<Vec<Intention>> {
        let params: HashMap<String, Value> =
            [("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]
                .into_iter()
                .collect();
        let result = self
            .db
            .execute(
                "SELECT id, description, status, origin, context_id, blueprint_catalog_id, properties, tags, created_at FROM intentions WHERE context_id = $context_id ORDER BY created_at ASC",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            let mut intention = decode_intention(&result.columns, row)?;
            // Two-layer read merge: a confirmed grouping links the intention to a canonical catalog
            // entry as an additive overlay. Override the read-view `blueprint_catalog_id` with the
            // grouped catalog while leaving the raw `intentions` row untouched;
            // `admin_raw_scan_intention_rows` still returns the raw ungrouped row.
            let overlay = crate::overlay::list_overlay_entries_for_item(
                self,
                &intention.id.as_uuid().to_string(),
            );
            if let Some(catalog) = crate::overlay::grouped_catalog_for_item(&overlay) {
                intention.blueprint_catalog_id = Some(catalog);
            }
            out.push(intention);
        }
        Ok(out)
    }

    pub fn admin_raw_scan_intention_rows(&self, id: IntentionId) -> Result<Vec<Intention>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "IntentionId",
            });
        }
        let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT id, description, status, origin, context_id, blueprint_catalog_id, properties, tags, created_at FROM intentions WHERE id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            out.push(decode_intention(&result.columns, row)?);
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn insert_intention_snapshot_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        intention_id: IntentionId,
        description: &str,
        status: IntentionStatus,
        origin: IntentionOrigin,
        properties_json: &JsonValue,
        tags_json: &JsonValue,
        valid_from: TxId,
    ) -> Result<IntentionSnapshotId> {
        let snapshot_id = IntentionSnapshotId::new_v7();
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(snapshot_id.as_uuid())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            (
                "intention_id".to_string(),
                Value::Uuid(intention_id.as_uuid()),
            ),
            (
                "description".to_string(),
                Value::Text(description.to_string()),
            ),
            (
                "status".to_string(),
                Value::Text(intention_status_str(status).to_string()),
            ),
            (
                "origin".to_string(),
                Value::Text(intention_origin_str(origin).to_string()),
            ),
            (
                "properties".to_string(),
                Value::Json(properties_json.clone()),
            ),
            ("tags".to_string(), Value::Json(tags_json.clone())),
            ("valid_from".to_string(), Value::TxId(valid_from)),
            ("valid_to".to_string(), Value::Null),
        ]
        .into_iter()
        .collect();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO intention_snapshots (id, context_id, intention_id, description, status, origin, properties, tags, valid_from, valid_to) VALUES ($id, $context_id, $intention_id, $description, $status, $origin, $properties, $tags, $valid_from, $valid_to)",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(snapshot_id)
    }

    pub(super) fn set_current_intention_snapshot_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        intention_id: IntentionId,
        snapshot_id: IntentionSnapshotId,
    ) -> Result<()> {
        let params: HashMap<String, Value> = [
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            (
                "intention_id".to_string(),
                Value::Uuid(intention_id.as_uuid()),
            ),
            (
                "snapshot_id".to_string(),
                Value::Uuid(snapshot_id.as_uuid()),
            ),
            ("updated_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        let updated = self.db
            .execute_in_tx(
                tx,
                "UPDATE intention_current_snapshots SET snapshot_id = $snapshot_id, updated_at = $updated_at WHERE context_id = $context_id AND intention_id = $intention_id",
                &params,
            )
            .map_err(map_engine_err)?
            .rows_affected;
        if updated == 1 {
            return Ok(());
        }
        self.db.execute_in_tx(
            tx,
            "INSERT INTO intention_current_snapshots (context_id, intention_id, snapshot_id, updated_at) VALUES ($context_id, $intention_id, $snapshot_id, $updated_at)",
            &params,
        ).map_err(map_engine_err)?;
        Ok(())
    }

    pub(super) fn current_intention_snapshot_id_in_tx(
        &self,
        tx: TxId,
        intention_id: IntentionId,
    ) -> Result<Option<IntentionSnapshotId>> {
        let result = self
            .db
            .execute_in_tx(
                tx,
                "SELECT snapshot_id FROM intention_current_snapshots WHERE intention_id = $intention_id",
                &HashMap::from([(
                    "intention_id".to_string(),
                    Value::Uuid(intention_id.as_uuid()),
                )]),
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| get_uuid(row, &result.columns, "snapshot_id").map(IntentionSnapshotId::from))
            .transpose()
    }

    fn get_intention_in_tx(&self, tx: TxId, id: IntentionId) -> Result<Option<Intention>> {
        let result = self
            .db
            .execute_in_tx(
                tx,
                "SELECT id, description, status, origin, context_id, blueprint_catalog_id, properties, tags, created_at FROM intentions WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| decode_intention(&result.columns, row))
            .transpose()
    }

    fn intention_snapshot_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        intention_id: IntentionId,
        snapshot_id: IntentionSnapshotId,
    ) -> Result<Option<IntentionSnapshot>> {
        let result = self
            .db
            .execute_in_tx(
                tx,
                "SELECT id, intention_id, description, status, origin, properties, tags, valid_from, valid_to FROM intention_snapshots WHERE context_id = $context_id AND intention_id = $intention_id AND id = $snapshot_id",
                &HashMap::from([
                    ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                    ("intention_id".to_string(), Value::Uuid(intention_id.as_uuid())),
                    ("snapshot_id".to_string(), Value::Uuid(snapshot_id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| decode_intention_snapshot(&result.columns, row))
            .transpose()
    }

    fn insert_intention_snapshot_link_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        intention_id: IntentionId,
        predecessor_snapshot_id: IntentionSnapshotId,
        successor_snapshot_id: IntentionSnapshotId,
    ) -> Result<()> {
        let params = HashMap::from([
            ("id".to_string(), Value::Uuid(Uuid::now_v7())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            (
                "intention_id".to_string(),
                Value::Uuid(intention_id.as_uuid()),
            ),
            (
                "predecessor_snapshot_id".to_string(),
                Value::Uuid(predecessor_snapshot_id.as_uuid()),
            ),
            (
                "successor_snapshot_id".to_string(),
                Value::Uuid(successor_snapshot_id.as_uuid()),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        self.db.execute_in_tx(
            tx,
            "INSERT INTO intention_snapshot_links (id, context_id, intention_id, predecessor_snapshot_id, successor_snapshot_id, created_at) VALUES ($id, $context_id, $intention_id, $predecessor_snapshot_id, $successor_snapshot_id, $created_at)",
            &params,
        ).map_err(map_engine_err)?;
        Ok(())
    }

    /// Return the immutable history edges so a caller can show the user exactly which branch it
    /// is asking `intention_at_on_branch` to follow.
    pub fn intention_snapshot_links(
        &self,
        intention_id: IntentionId,
    ) -> Result<Vec<IntentionSnapshotLink>> {
        if !intention_id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "IntentionId",
            });
        }
        let result = self
            .db
            .execute(
                "SELECT id, intention_id, predecessor_snapshot_id, successor_snapshot_id, created_at FROM intention_snapshot_links WHERE intention_id = $intention_id",
                &HashMap::from([(
                    "intention_id".to_string(),
                    Value::Uuid(intention_id.as_uuid()),
                )]),
            )
            .map_err(map_engine_err)?;
        let mut links = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            links.push(IntentionSnapshotLink {
                id: IntentionSnapshotLinkId::from(get_uuid(row, &result.columns, "id")?),
                intention_id: IntentionId::from(get_uuid(row, &result.columns, "intention_id")?),
                predecessor_snapshot_id: IntentionSnapshotId::from(get_uuid(
                    row,
                    &result.columns,
                    "predecessor_snapshot_id",
                )?),
                successor_snapshot_id: IntentionSnapshotId::from(get_uuid(
                    row,
                    &result.columns,
                    "successor_snapshot_id",
                )?),
                created_at: get_txid(row, &result.columns, "created_at")?,
            });
        }
        links.sort_by_key(|link| (link.created_at.0, link.id.as_uuid()));
        Ok(links)
    }

    fn intention_snapshot_link_pairs(
        &self,
        intention_id: IntentionId,
    ) -> Result<Vec<(IntentionSnapshotId, IntentionSnapshotId)>> {
        let result = self.db.execute(
            "SELECT predecessor_snapshot_id, successor_snapshot_id FROM intention_snapshot_links WHERE intention_id = $intention_id",
            &HashMap::from([("intention_id".to_string(), Value::Uuid(intention_id.as_uuid()))]),
        ).map_err(map_engine_err)?;
        result
            .rows
            .iter()
            .map(|row| {
                Ok((
                    IntentionSnapshotId::from(get_uuid(
                        row,
                        &result.columns,
                        "predecessor_snapshot_id",
                    )?),
                    IntentionSnapshotId::from(get_uuid(
                        row,
                        &result.columns,
                        "successor_snapshot_id",
                    )?),
                ))
            })
            .collect()
    }
}

pub(super) fn decode_intention(columns: &[String], row: &[Value]) -> Result<Intention> {
    Ok(Intention {
        id: IntentionId::from(get_uuid(row, columns, "id")?),
        description: get_text(row, columns, "description")?,
        status: intention_status_from_str(&get_text(row, columns, "status")?)?,
        origin: intention_origin_from_str(&get_text(row, columns, "origin")?)?,
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        // Summary-projection queries (e.g. discovery's by_entity_intentions) intentionally omit
        // this column; absence means the row carries no catalog grouping in this projection.
        blueprint_catalog_id: if columns.iter().any(|c| c == "blueprint_catalog_id") {
            get_optional_text(row, columns, "blueprint_catalog_id")?
        } else {
            None
        },
        properties: properties_from_json(&get_json(row, columns, "properties")?),
        tags: tags_from_json(&get_json(row, columns, "tags")?),
        created_at: get_txid(row, columns, "created_at")?,
    })
}

fn decode_intention_snapshot(columns: &[String], row: &[Value]) -> Result<IntentionSnapshot> {
    let intention_id = IntentionId::from(get_uuid(row, columns, "intention_id")?);
    let description = get_text(row, columns, "description")?;
    let status = intention_status_from_str(&get_text(row, columns, "status")?)?;
    let origin = intention_origin_from_str(&get_text(row, columns, "origin")?)?;
    let properties = properties_from_json(&get_json(row, columns, "properties")?);
    let tags = tags_from_json(&get_json(row, columns, "tags")?);
    let valid_from = get_txid(row, columns, "valid_from")?;
    let valid_to = get_optional_txid(row, columns, "valid_to")?;
    Ok(IntentionSnapshot {
        id: IntentionSnapshotId::from(get_uuid(row, columns, "id")?),
        intention_id,
        description,
        status,
        origin,
        properties,
        tags,
        valid_from,
        valid_to,
    })
}

fn decode_invalidation(columns: &[String], row: &[Value]) -> Result<Invalidation> {
    let resolution_decision_id =
        match super::encode::column_index(columns, "resolution_decision_id")
            .ok()
            .and_then(|idx| row.get(idx))
        {
            Some(Value::Uuid(u)) => Some(DecisionId::from(*u)),
            _ => None,
        };
    Ok(Invalidation {
        id: InvalidationId::from(get_uuid(row, columns, "id")?),
        status: invalidation_status_from_str(&get_text(row, columns, "status")?)?,
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        affected_decision_id: DecisionId::from(get_uuid(row, columns, "affected_decision_id")?),
        trigger_observation_id: ObservationId::from(get_uuid(
            row,
            columns,
            "trigger_observation_id",
        )?),
        reason: get_optional_text(row, columns, "reason")?,
        basis_diff: basis_diff_from_json(&get_json(row, columns, "basis_diff")?)?,
        cascade_from: optional_uuid(row, columns, "cascade_from")?.map(DecisionId::from),
        severity: invalidation_severity_from_str(&get_text(row, columns, "severity")?)?,
        threatened_intention_ids: intention_ids_from_json(&get_json(
            row,
            columns,
            "threatened_intention_ids",
        )?)?,
        intention_now_un_served: get_bool(row, columns, "intention_now_un_served")?,
        detected_at: get_txid(row, columns, "detected_at")?,
        resolved_at: get_optional_txid(row, columns, "resolved_at")?,
        resolution_action: get_optional_text(row, columns, "resolution_action")?,
        resolution_decision_id,
    })
}

fn optional_uuid(row: &[Value], columns: &[String], name: &str) -> Result<Option<Uuid>> {
    let Some(idx) = columns.iter().position(|column| column == name) else {
        return Ok(None);
    };
    match row.get(idx) {
        Some(Value::Uuid(u)) => Ok(Some(*u)),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected uuid|null, got {other:?}"
        ))),
    }
}

fn invalidation_status_from_str(s: &str) -> Result<InvalidationStatus> {
    match s {
        "pending" => Ok(InvalidationStatus::Pending),
        "acknowledged" => Ok(InvalidationStatus::Acknowledged),
        "resolved" => Ok(InvalidationStatus::Resolved),
        "dismissed" => Ok(InvalidationStatus::Dismissed),
        other => Err(CgError::Engine(format!(
            "unknown invalidation status: {other}"
        ))),
    }
}

fn invalidation_severity_from_str(s: &str) -> Result<InvalidationSeverity> {
    match s {
        "info" => Ok(InvalidationSeverity::Info),
        "warning" => Ok(InvalidationSeverity::Warning),
        "critical" => Ok(InvalidationSeverity::Critical),
        other => Err(CgError::Engine(format!(
            "unknown invalidation severity: {other}"
        ))),
    }
}

fn severity_rank(severity: InvalidationSeverity) -> u8 {
    match severity {
        InvalidationSeverity::Info => 0,
        InvalidationSeverity::Warning => 1,
        InvalidationSeverity::Critical => 2,
    }
}
