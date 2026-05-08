use std::collections::{BTreeSet, HashMap, HashSet};

use contextdb_core::{Error as EngineError, SnapshotId, Value};
use serde_json::Value as JsonValue;

use crate::error::CgError;
use crate::types::*;
use crate::{Result, TxId};

use super::Store;
use super::convert::{
    decision_status_from_str, intention_origin_from_str, intention_status_from_str, map_engine_err,
    relationship_type_str,
};
use super::encode::{
    basis_diff_from_json, decode_outcome, get_bool, get_json, get_optional_float,
    get_optional_text, get_optional_txid, get_text, get_txid, get_uuid, intention_ids_from_json,
    properties_from_json, properties_to_json, tags_from_json,
};
use super::evidence::decode_evidence;

impl Store {
    pub fn create_decision(&self, input: CreateDecision) -> Result<Decision> {
        if input.intention_ids.is_empty() {
            return Err(CgError::MissingIntention);
        }
        self.require_context(input.context_id)?;
        for iid in &input.intention_ids {
            if !iid.is_valid() {
                return Err(CgError::InvalidIdentity {
                    context: "IntentionId",
                });
            }
        }
        let intention_contexts = self.intention_context_ids(&input.intention_ids)?;
        for iid in &input.intention_ids {
            match intention_contexts.get(iid) {
                Some(ictx) => {
                    if *ictx != input.context_id {
                        return Err(CgError::CrossContextViolation);
                    }
                }
                None => return Err(CgError::IntentionNotFound { id: *iid }),
            }
        }

        for eid in &input.based_on_entity_ids {
            if !eid.is_valid() {
                return Err(CgError::InvalidIdentity {
                    context: "EntityId",
                });
            }
            let entity = self.get_entity(*eid)?.ok_or(CgError::NotFound {
                id: Some(eid.as_uuid().to_string()),
                missing: Some("entity"),
            })?;
            if entity.context_id != input.context_id {
                return Err(CgError::CrossContextViolation);
            }
        }
        for pid in &input.precedent_ids {
            if !pid.is_valid() {
                return Err(CgError::InvalidIdentity {
                    context: "DecisionId",
                });
            }
            let precedent_context = self.decision_context_id(*pid)?.ok_or(CgError::NotFound {
                id: Some(pid.as_uuid().to_string()),
                missing: Some("decision"),
            })?;
            if precedent_context != input.context_id {
                return Err(CgError::CrossContextRelationship);
            }
        }

        let id = DecisionId::new_v7();
        let reasoning_json = serde_json::to_value(&input.reasoning)
            .map_err(|e| CgError::Engine(format!("reasoning serialize: {e}")))?;
        let tags_json = JsonValue::Array(
            input
                .tags
                .iter()
                .map(|t| JsonValue::String(t.clone()))
                .collect(),
        );
        let properties_json = properties_to_json(&input.properties);
        let mut basis_snapshot_ids = input.based_on_snapshots.clone();
        for eid in &input.based_on_entity_ids {
            let basis_snapshot = match self.current_entity_snapshot(*eid) {
                Ok(Some(snapshot)) => snapshot,
                Ok(None) => {
                    return Err(CgError::NotFound {
                        id: Some(eid.as_uuid().to_string()),
                        missing: Some("entity_snapshot"),
                    });
                }
                Err(e) => return Err(e),
            };
            if !basis_snapshot_ids.contains(&basis_snapshot.id) {
                basis_snapshot_ids.push(basis_snapshot.id);
            }
        }
        self.validate_basis_snapshots_for_context(input.context_id, &basis_snapshot_ids)?;
        let basis_snapshots = self.basis_snapshots_for_basis(&basis_snapshot_ids)?;

        let tx = self.begin_tx()?;
        let confidence_value = match input.confidence {
            Some(c) => Value::Float64(c as f64),
            None => Value::Null,
        };
        let agent_id_value = match &input.agent_id {
            Some(a) => Value::Text(a.clone()),
            None => Value::Null,
        };
        let decision_params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            (
                "decision_type".to_string(),
                Value::Text(input.decision_type.clone()),
            ),
            (
                "description".to_string(),
                Value::Text(input.description.clone()),
            ),
            ("reasoning".to_string(), Value::Json(reasoning_json)),
            ("confidence".to_string(), confidence_value),
            ("status".to_string(), Value::Text("active".to_string())),
            ("agent_id".to_string(), agent_id_value),
            (
                "context_id".to_string(),
                Value::Uuid(input.context_id.as_uuid()),
            ),
            ("properties".to_string(), Value::Json(properties_json)),
            ("tags".to_string(), Value::Json(tags_json)),
            ("created_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
            &decision_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) =
            self.insert_decision_basis_snapshots_in_tx(tx, input.context_id, id, &basis_snapshots)
        {
            let _ = self.db.rollback(tx);
            return Err(e);
        }

        for iid in &input.intention_ids {
            if let Err(e) = self.insert_edge_in_tx(
                tx,
                id.as_uuid(),
                iid.as_uuid(),
                "SERVES",
                input.context_id,
                None,
            ) {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        }
        let mut basis_edge_entities = BTreeSet::new();
        for basis_snapshot in &basis_snapshots {
            if !basis_edge_entities.insert(basis_snapshot.entity_id) {
                continue;
            }
            if let Err(e) = self.insert_edge_in_tx(
                tx,
                id.as_uuid(),
                basis_snapshot.entity_id.as_uuid(),
                "BASED_ON",
                input.context_id,
                Some(basis_snapshot_edge_properties(basis_snapshot)),
            ) {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        }
        for pid in &input.precedent_ids {
            if let Err(e) = self.insert_edge_in_tx(
                tx,
                id.as_uuid(),
                pid.as_uuid(),
                "CITES",
                input.context_id,
                None,
            ) {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Create,
            Principal::System,
            None,
            AuditTarget::Decision(id),
            input.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }

        self.db.commit(tx).map_err(map_engine_err)?;

        let precedent_ids = self.precedents_for_decision(id, input.context_id)?;
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
            created_at: tx,
            intention_ids: input.intention_ids,
            based_on_snapshots: basis_snapshot_ids,
            outcome: None,
            precedent_ids,
        })
    }

    pub fn get_decision(&self, id: DecisionId) -> Result<Option<Decision>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "DecisionId",
            });
        }
        let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at FROM decisions WHERE id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        let mut decision = decode_decision(&result.columns, &result.rows[0])?;
        decision.intention_ids = self.intentions_for_decision(id)?;
        decision.based_on_snapshots = self.basis_snapshots_for_decision(id)?;
        decision.outcome = self.outcome_for_decision(id)?;
        decision.precedent_ids = self.precedents_for_decision(id, decision.context_id)?;
        Ok(Some(decision))
    }

    fn intentions_for_decision(&self, id: DecisionId) -> Result<Vec<IntentionId>> {
        let params: HashMap<String, Value> = [("source_id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT b_id FROM GRAPH_TABLE(edges MATCH (a)-[:SERVES]->(b) WHERE a.id = $source_id COLUMNS (b.id AS b_id))",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            if let Some(Value::Uuid(u)) = row.first() {
                out.push(IntentionId::from(*u));
            }
        }
        Ok(out)
    }

    fn basis_snapshots_for_decision(&self, id: DecisionId) -> Result<Vec<EntitySnapshotId>> {
        let result = self
            .db
            .execute(
                "SELECT snapshot_id, basis_order FROM decision_basis_snapshots WHERE decision_id = $decision_id",
                &HashMap::from([("decision_id".to_string(), Value::Uuid(id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        ordered_snapshot_ids(&result.columns, &result.rows, "basis_order", "snapshot_id")
    }

    pub(super) fn basis_snapshots_for_basis(
        &self,
        basis_snapshot_ids: &[EntitySnapshotId],
    ) -> Result<Vec<EntitySnapshot>> {
        let snapshot = self.db.snapshot();
        let mut snapshots = Vec::with_capacity(basis_snapshot_ids.len());
        for snapshot_id in basis_snapshot_ids {
            let Some(row) = self.snapshot_by_id_at_read(*snapshot_id, snapshot)? else {
                return Err(CgError::NotFound {
                    id: Some(snapshot_id.as_uuid().to_string()),
                    missing: Some("entity_snapshot"),
                });
            };
            snapshots.push(row);
        }
        Ok(snapshots)
    }

    pub(super) fn basis_snapshot_ids_for_decision_at_read(
        &self,
        decision_id: DecisionId,
        snapshot: SnapshotId,
    ) -> Result<Vec<EntitySnapshotId>> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT snapshot_id, basis_order FROM decision_basis_snapshots WHERE decision_id = $decision_id",
                &HashMap::from([(
                    "decision_id".to_string(),
                    Value::Uuid(decision_id.as_uuid()),
                )]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        ordered_snapshot_ids(&result.columns, &result.rows, "basis_order", "snapshot_id")
    }

    pub(super) fn insert_decision_basis_snapshots_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        decision_id: DecisionId,
        basis_snapshots: &[EntitySnapshot],
    ) -> Result<()> {
        for (basis_order, snapshot) in basis_snapshots.iter().enumerate() {
            let params = HashMap::from([
                (
                    "id".to_string(),
                    Value::Uuid(decision_basis_row_id(decision_id, basis_order)),
                ),
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                (
                    "decision_id".to_string(),
                    Value::Uuid(decision_id.as_uuid()),
                ),
                (
                    "entity_id".to_string(),
                    Value::Uuid(snapshot.entity_id.as_uuid()),
                ),
                (
                    "snapshot_id".to_string(),
                    Value::Uuid(snapshot.id.as_uuid()),
                ),
                ("basis_order".to_string(), Value::Int64(basis_order as i64)),
                ("created_at".to_string(), Value::TxId(tx)),
            ]);
            self.db
                .execute_in_tx(
                    tx,
                    "INSERT INTO decision_basis_snapshots (id, context_id, decision_id, entity_id, snapshot_id, basis_order, created_at) VALUES ($id, $context_id, $decision_id, $entity_id, $snapshot_id, $basis_order, $created_at)",
                    &params,
                )
                .map_err(map_engine_err)?;
        }
        Ok(())
    }

    pub(super) fn replace_decision_basis_snapshots_in_tx(
        &self,
        tx: TxId,
        context_id: ContextId,
        decision_id: DecisionId,
        basis_snapshots: &[EntitySnapshot],
    ) -> Result<()> {
        self.db
            .execute_in_tx(
                tx,
                "DELETE FROM decision_basis_snapshots WHERE decision_id = $decision_id",
                &HashMap::from([(
                    "decision_id".to_string(),
                    Value::Uuid(decision_id.as_uuid()),
                )]),
            )
            .map_err(map_engine_err)?;
        for (basis_order, snapshot) in basis_snapshots.iter().enumerate() {
            let params = HashMap::from([
                (
                    "id".to_string(),
                    Value::Uuid(decision_basis_row_id(decision_id, basis_order)),
                ),
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                (
                    "decision_id".to_string(),
                    Value::Uuid(decision_id.as_uuid()),
                ),
                (
                    "entity_id".to_string(),
                    Value::Uuid(snapshot.entity_id.as_uuid()),
                ),
                (
                    "snapshot_id".to_string(),
                    Value::Uuid(snapshot.id.as_uuid()),
                ),
                ("basis_order".to_string(), Value::Int64(basis_order as i64)),
                ("created_at".to_string(), Value::TxId(tx)),
            ]);
            self.db
                .execute_in_tx(
                    tx,
                    "INSERT INTO decision_basis_snapshots (id, context_id, decision_id, entity_id, snapshot_id, basis_order, created_at) VALUES ($id, $context_id, $decision_id, $entity_id, $snapshot_id, $basis_order, $created_at)",
                    &params,
                )
                .map_err(map_engine_err)?;
        }
        Ok(())
    }

    pub(super) fn decision_context_id(&self, id: DecisionId) -> Result<Option<ContextId>> {
        let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute("SELECT context_id FROM decisions WHERE id = $id", &params)
            .map_err(map_engine_err)?;
        match result.rows.first().and_then(|r| r.first()) {
            Some(Value::Uuid(u)) => Ok(Some(ContextId::from(*u))),
            Some(_) => Err(CgError::Engine("decisions.context_id non-uuid".to_string())),
            None => Ok(None),
        }
    }

    pub(super) fn outcome_for_decision(&self, id: DecisionId) -> Result<Option<Outcome>> {
        let params: HashMap<String, Value> =
            [("decision_id".to_string(), Value::Uuid(id.as_uuid()))]
                .into_iter()
                .collect();
        let result = self
            .db
            .execute(
                "SELECT id, decision_id, success, notes, recorded_at FROM outcomes WHERE decision_id = $decision_id",
                &params,
            )
            .map_err(map_engine_err)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        Some(decode_outcome(&result.columns, &result.rows[0])).transpose()
    }

    pub(super) fn precedents_for_decision(
        &self,
        id: DecisionId,
        context_id: ContextId,
    ) -> Result<Vec<DecisionId>> {
        let targets = self.edge_targets(context_id, id.as_uuid(), RelationshipType::Cites)?;
        Ok(targets.into_iter().map(DecisionId::from).collect())
    }

    pub fn list_decisions(&self, filter: ListDecisionFilter) -> Result<Vec<Decision>> {
        let mut trace_parts: Vec<String> = Vec::new();

        let mut candidate_ids: Option<HashSet<DecisionId>> = None;

        if let Some(iid) = filter.intention_id {
            trace_parts.push("edge_type=SERVES".to_string());
            let params: HashMap<String, Value> =
                [("target_id".to_string(), Value::Uuid(iid.as_uuid()))]
                    .into_iter()
                    .collect();
            let result = self
                .db
                .execute(
                    "SELECT b_id FROM GRAPH_TABLE(edges MATCH (a)<-[:SERVES]-(b) WHERE a.id = $target_id COLUMNS (b.id AS b_id))",
                    &params,
                )
                .map_err(map_engine_err)?;
            let ids: HashSet<DecisionId> = result
                .rows
                .iter()
                .filter_map(|r| match r.first() {
                    Some(Value::Uuid(u)) => Some(DecisionId::from(*u)),
                    _ => None,
                })
                .collect();
            candidate_ids = Some(ids);
        }

        if !filter.based_on_entity_ids.is_empty() {
            let mut per_entity: Vec<HashSet<DecisionId>> = Vec::new();
            for eid in &filter.based_on_entity_ids {
                let params: HashMap<String, Value> =
                    [("target_id".to_string(), Value::Uuid(eid.as_uuid()))]
                        .into_iter()
                        .collect();
                let result = self
                    .db
                    .execute(
                        "SELECT b_id FROM GRAPH_TABLE(edges MATCH (a)<-[:BASED_ON]-(b) WHERE a.id = $target_id COLUMNS (b.id AS b_id))",
                        &params,
                    )
                    .map_err(map_engine_err)?;
                let ids: HashSet<DecisionId> = result
                    .rows
                    .iter()
                    .filter_map(|r| match r.first() {
                        Some(Value::Uuid(u)) => Some(DecisionId::from(*u)),
                        _ => None,
                    })
                    .collect();
                per_entity.push(ids);
            }
            let combined: HashSet<DecisionId> = match filter.entity_match_policy {
                EntityMatchPolicy::Any => per_entity.into_iter().flatten().collect(),
                EntityMatchPolicy::All => {
                    if let Some(first) = per_entity.first().cloned() {
                        per_entity
                            .iter()
                            .skip(1)
                            .fold(first, |acc, s| acc.intersection(s).copied().collect())
                    } else {
                        HashSet::new()
                    }
                }
            };
            candidate_ids = Some(match candidate_ids {
                Some(existing) => existing.intersection(&combined).copied().collect(),
                None => combined,
            });
        }

        let mut result_rows: Vec<Decision> = Vec::new();
        if let Some(ids) = candidate_ids {
            for id in ids {
                if let Some(d) = self.get_decision(id)? {
                    result_rows.push(d);
                }
            }
        } else {
            let mut clauses: Vec<String> = Vec::new();
            let mut params: HashMap<String, Value> = HashMap::new();
            if let Some(ctx) = filter.context_id {
                clauses.push("context_id = $context_id".to_string());
                params.insert("context_id".to_string(), Value::Uuid(ctx.as_uuid()));
            }
            let mut sql = String::from(
                "SELECT id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at FROM decisions",
            );
            if !clauses.is_empty() {
                sql.push_str(" WHERE ");
                sql.push_str(&clauses.join(" AND "));
            }
            let result = self.db.execute(&sql, &params).map_err(map_engine_err)?;
            for row in &result.rows {
                let mut d = decode_decision(&result.columns, row)?;
                d.intention_ids = self.intentions_for_decision(d.id)?;
                d.based_on_snapshots = self.basis_snapshots_for_decision(d.id)?;
                d.outcome = self.outcome_for_decision(d.id)?;
                result_rows.push(d);
            }
        }

        if let Some(ctx) = filter.context_id {
            result_rows.retain(|d| d.context_id == ctx);
        }
        if !filter.tags.is_empty() {
            result_rows.retain(|d| filter.tags.iter().all(|t| d.tags.iter().any(|dt| dt == t)));
        }
        if let Some(status) = filter.status {
            result_rows.retain(|d| d.status == status);
        }

        result_rows.sort_by(|a, b| {
            b.created_at
                .0
                .cmp(&a.created_at.0)
                .then_with(|| b.id.as_uuid().cmp(&a.id.as_uuid()))
        });

        if let Some(limit) = filter.limit {
            result_rows.truncate(limit as usize);
        }

        self.set_trace(trace_parts.join(";"));
        Ok(result_rows)
    }

    pub fn update_decision(&self, id: DecisionId, patch: DecisionPatch) -> Result<Decision> {
        let existing = self.get_decision(id)?.ok_or(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("decision"),
        })?;
        let mut set_clauses: Vec<&str> = Vec::new();
        let mut params: HashMap<String, Value> =
            [("id".to_string(), Value::Uuid(id.as_uuid()))].into();
        if let Some(properties) = &patch.properties {
            set_clauses.push("properties = $properties");
            params.insert(
                "properties".to_string(),
                Value::Json(properties_to_json(properties)),
            );
        }
        if let Some(tags) = &patch.tags {
            let tags_json =
                JsonValue::Array(tags.iter().map(|t| JsonValue::String(t.clone())).collect());
            set_clauses.push("tags = $tags");
            params.insert("tags".to_string(), Value::Json(tags_json));
        }
        if let Some(confidence) = patch.confidence {
            set_clauses.push("confidence = $confidence");
            params.insert("confidence".to_string(), Value::Float64(confidence as f64));
        }
        let tx = self.begin_tx()?;
        if !set_clauses.is_empty() {
            let sql = format!(
                "UPDATE decisions SET {} WHERE id = $id",
                set_clauses.join(", ")
            );
            if let Err(e) = self.db.execute_in_tx(tx, &sql, &params) {
                let _ = self.db.rollback(tx);
                return Err(map_engine_err(e));
            }
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Update,
            Principal::System,
            None,
            AuditTarget::Decision(id),
            existing.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        self.get_decision(id)?.ok_or(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("decision"),
        })
    }

    pub fn admin_raw_update_decision_immutable_fields(
        &self,
        id: DecisionId,
        decision_type: Option<String>,
        description: Option<String>,
    ) -> Result<Decision> {
        if let Some(dt) = &decision_type {
            let params: HashMap<String, Value> = [
                ("id".to_string(), Value::Uuid(id.as_uuid())),
                ("decision_type".to_string(), Value::Text(dt.clone())),
            ]
            .into_iter()
            .collect();
            return Err(
                match self.db.execute(
                    "UPDATE decisions SET decision_type = $decision_type WHERE id = $id",
                    &params,
                ) {
                    Ok(_) => CgError::Engine(
                        "engine accepted UPDATE on IMMUTABLE column decisions.decision_type"
                            .to_string(),
                    ),
                    Err(EngineError::ImmutableColumn { column, .. }) => {
                        CgError::ImmutableField { field: column }
                    }
                    Err(other) => map_engine_err(other),
                },
            );
        }
        if let Some(desc) = &description {
            let params: HashMap<String, Value> = [
                ("id".to_string(), Value::Uuid(id.as_uuid())),
                ("description".to_string(), Value::Text(desc.clone())),
            ]
            .into_iter()
            .collect();
            return Err(
                match self.db.execute(
                    "UPDATE decisions SET description = $description WHERE id = $id",
                    &params,
                ) {
                    Ok(_) => CgError::Engine(
                        "engine accepted UPDATE on IMMUTABLE column decisions.description"
                            .to_string(),
                    ),
                    Err(EngineError::ImmutableColumn { column, .. }) => {
                        CgError::ImmutableField { field: column }
                    }
                    Err(other) => map_engine_err(other),
                },
            );
        }
        self.get_decision(id)?.ok_or(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("decision"),
        })
    }

    pub fn supersede_decision(&self, old: DecisionId, new: CreateDecision) -> Result<Decision> {
        let existing = self.get_decision(old)?.ok_or(CgError::NotFound {
            id: Some(old.as_uuid().to_string()),
            missing: Some("decision"),
        })?;
        if existing.status != DecisionStatus::Active {
            return Err(CgError::IllegalTransition {
                from: existing.status,
                to: DecisionStatus::Superseded,
            });
        }
        // Validate intentions and basis entities before opening the tx.
        if new.intention_ids.is_empty() {
            return Err(CgError::MissingIntention);
        }
        self.require_context(new.context_id)?;
        for iid in &new.intention_ids {
            if !iid.is_valid() {
                return Err(CgError::InvalidIdentity {
                    context: "IntentionId",
                });
            }
        }
        let intention_contexts = self.intention_context_ids(&new.intention_ids)?;
        for iid in &new.intention_ids {
            match intention_contexts.get(iid) {
                Some(ictx) => {
                    if *ictx != new.context_id {
                        return Err(CgError::CrossContextViolation);
                    }
                }
                None => return Err(CgError::IntentionNotFound { id: *iid }),
            }
        }
        for eid in &new.based_on_entity_ids {
            if !eid.is_valid() {
                return Err(CgError::InvalidIdentity {
                    context: "EntityId",
                });
            }
            let entity = self.get_entity(*eid)?.ok_or(CgError::NotFound {
                id: Some(eid.as_uuid().to_string()),
                missing: Some("entity"),
            })?;
            if entity.context_id != new.context_id {
                return Err(CgError::CrossContextViolation);
            }
        }
        for pid in &new.precedent_ids {
            if !pid.is_valid() {
                return Err(CgError::InvalidIdentity {
                    context: "DecisionId",
                });
            }
            let precedent_context = self.decision_context_id(*pid)?.ok_or(CgError::NotFound {
                id: Some(pid.as_uuid().to_string()),
                missing: Some("decision"),
            })?;
            if precedent_context != new.context_id {
                return Err(CgError::CrossContextRelationship);
            }
        }

        let new_id = DecisionId::new_v7();
        let reasoning_json = serde_json::to_value(&new.reasoning)
            .map_err(|e| CgError::Engine(format!("reasoning serialize: {e}")))?;
        let tags_json = JsonValue::Array(
            new.tags
                .iter()
                .map(|t| JsonValue::String(t.clone()))
                .collect(),
        );
        let properties_json = properties_to_json(&new.properties);
        let mut basis_snapshot_ids = new.based_on_snapshots.clone();
        for eid in &new.based_on_entity_ids {
            let basis_snapshot = match self.current_entity_snapshot(*eid) {
                Ok(Some(snapshot)) => snapshot,
                Ok(None) => {
                    return Err(CgError::NotFound {
                        id: Some(eid.as_uuid().to_string()),
                        missing: Some("entity_snapshot"),
                    });
                }
                Err(e) => return Err(e),
            };
            if !basis_snapshot_ids.contains(&basis_snapshot.id) {
                basis_snapshot_ids.push(basis_snapshot.id);
            }
        }
        self.validate_basis_snapshots_for_context(new.context_id, &basis_snapshot_ids)?;
        let basis_snapshots = self.basis_snapshots_for_basis(&basis_snapshot_ids)?;

        let tx = self.begin_tx()?;
        // Transition old decision to superseded.
        let transition_params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(old.as_uuid())),
            ("status".to_string(), Value::Text("superseded".to_string())),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "UPDATE decisions SET status = $status WHERE id = $id",
            &transition_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(match e {
                EngineError::InvalidStateTransition(_) | EngineError::PropagationAborted { .. } => {
                    CgError::IllegalTransition {
                        from: existing.status,
                        to: DecisionStatus::Superseded,
                    }
                }
                other => map_engine_err(other),
            });
        }
        // Insert new decision.
        let confidence_value = match new.confidence {
            Some(c) => Value::Float64(c as f64),
            None => Value::Null,
        };
        let agent_id_value = match &new.agent_id {
            Some(a) => Value::Text(a.clone()),
            None => Value::Null,
        };
        let decision_params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(new_id.as_uuid())),
            (
                "decision_type".to_string(),
                Value::Text(new.decision_type.clone()),
            ),
            (
                "description".to_string(),
                Value::Text(new.description.clone()),
            ),
            ("reasoning".to_string(), Value::Json(reasoning_json)),
            ("confidence".to_string(), confidence_value),
            ("status".to_string(), Value::Text("active".to_string())),
            ("agent_id".to_string(), agent_id_value),
            (
                "context_id".to_string(),
                Value::Uuid(new.context_id.as_uuid()),
            ),
            ("properties".to_string(), Value::Json(properties_json)),
            ("tags".to_string(), Value::Json(tags_json)),
            ("created_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
            &decision_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) =
            self.insert_decision_basis_snapshots_in_tx(tx, new.context_id, new_id, &basis_snapshots)
        {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        for iid in &new.intention_ids {
            if let Err(e) = self.insert_edge_in_tx(
                tx,
                new_id.as_uuid(),
                iid.as_uuid(),
                "SERVES",
                new.context_id,
                None,
            ) {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        }
        let mut basis_edge_entities = BTreeSet::new();
        for basis_snapshot in &basis_snapshots {
            if !basis_edge_entities.insert(basis_snapshot.entity_id) {
                continue;
            }
            if let Err(e) = self.insert_edge_in_tx(
                tx,
                new_id.as_uuid(),
                basis_snapshot.entity_id.as_uuid(),
                "BASED_ON",
                new.context_id,
                Some(basis_snapshot_edge_properties(basis_snapshot)),
            ) {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        }
        for pid in &new.precedent_ids {
            if let Err(e) = self.insert_edge_in_tx(
                tx,
                new_id.as_uuid(),
                pid.as_uuid(),
                "CITES",
                new.context_id,
                None,
            ) {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
        }
        if let Err(e) = self.insert_edge_in_tx(
            tx,
            new_id.as_uuid(),
            old.as_uuid(),
            "SUPERSEDES",
            new.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Supersede,
            Principal::System,
            None,
            AuditTarget::Decision(old),
            existing.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Create,
            Principal::System,
            None,
            AuditTarget::Decision(new_id),
            new.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        let precedent_ids = self.precedents_for_decision(new_id, new.context_id)?;
        Ok(Decision {
            id: new_id,
            decision_type: new.decision_type,
            description: new.description,
            reasoning: new.reasoning,
            confidence: new.confidence,
            status: DecisionStatus::Active,
            agent_id: new.agent_id,
            context_id: new.context_id,
            tags: new.tags,
            properties: new.properties,
            created_at: tx,
            intention_ids: new.intention_ids,
            based_on_snapshots: basis_snapshot_ids,
            outcome: None,
            precedent_ids,
        })
    }

    pub fn supersede_history(&self, id: DecisionId, depth: u32) -> Result<Vec<Decision>> {
        let mut out: Vec<Decision> = Vec::new();
        let mut cursor = id;
        for _ in 0..=depth {
            let d = match self.get_decision(cursor)? {
                Some(d) => d,
                None => break,
            };
            out.push(d);
            let params: HashMap<String, Value> = [
                ("source_id".to_string(), Value::Uuid(cursor.as_uuid())),
                (
                    "edge_type".to_string(),
                    Value::Text("SUPERSEDES".to_string()),
                ),
            ]
            .into_iter()
            .collect();
            let result = self
                .db
                .execute(
                    "SELECT target_id FROM edges WHERE source_id = $source_id AND edge_type = $edge_type",
                    &params,
                )
                .map_err(map_engine_err)?;
            match result.rows.first().and_then(|r| r.first()) {
                Some(Value::Uuid(u)) => cursor = DecisionId::from(*u),
                _ => break,
            }
        }
        Ok(out)
    }

    pub fn precedent_chain(&self, decision_id: DecisionId, max_depth: u8) -> Result<Vec<Decision>> {
        self.validate_graph_depth(max_depth)?;
        let start = self.get_decision(decision_id)?.ok_or(CgError::NotFound {
            id: Some(decision_id.as_uuid().to_string()),
            missing: Some("decision"),
        })?;
        let context_id = start.context_id;
        let mut out = vec![start];
        if max_depth == 0 {
            return Ok(out);
        }

        let mut visited: HashSet<DecisionId> = [decision_id].into_iter().collect();
        let mut frontier = vec![decision_id.as_uuid()];
        for _hop in 1..=max_depth {
            let edges =
                self.edges_from_sources(context_id, &frontier, Some(RelationshipType::Cites))?;
            let mut candidates = Vec::new();
            for (_source, target, _) in edges {
                let target_id = DecisionId::from(target);
                if visited.insert(target_id)
                    && let Some(decision) = self.get_decision(target_id)?
                    && decision.context_id == context_id
                {
                    candidates.push(decision);
                }
            }
            if candidates.is_empty() {
                break;
            }
            candidates.sort_by(|a, b| {
                a.created_at
                    .0
                    .cmp(&b.created_at.0)
                    .then_with(|| a.id.as_uuid().cmp(&b.id.as_uuid()))
            });
            frontier = candidates.iter().map(|d| d.id.as_uuid()).collect();
            out.extend(candidates);
        }
        Ok(out)
    }

    pub fn admin_raw_insert_cites(&self, from: DecisionId, to: DecisionId) -> Result<()> {
        let from_context = self.decision_context_id(from)?.ok_or(CgError::NotFound {
            id: Some(from.as_uuid().to_string()),
            missing: Some("from"),
        })?;
        let to_context = self.decision_context_id(to)?.ok_or(CgError::NotFound {
            id: Some(to.as_uuid().to_string()),
            missing: Some("to"),
        })?;
        if from_context != to_context {
            return Err(CgError::CrossContextRelationship);
        }
        let tx = self.begin_tx()?;
        if let Err(e) = self.insert_edge_in_tx(
            tx,
            from.as_uuid(),
            to.as_uuid(),
            "CITES",
            from_context,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;
        Ok(())
    }

    pub fn archive_decision(&self, id: DecisionId, reason: String) -> Result<Decision> {
        let existing = self.get_decision(id)?.ok_or(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("decision"),
        })?;
        let tx = self.begin_tx()?;
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("status".to_string(), Value::Text("archived".to_string())),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "UPDATE decisions SET status = $status WHERE id = $id",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(match e {
                EngineError::InvalidStateTransition(_) | EngineError::PropagationAborted { .. } => {
                    CgError::IllegalTransition {
                        from: existing.status,
                        to: DecisionStatus::Archived,
                    }
                }
                other => map_engine_err(other),
            });
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Archive,
            Principal::System,
            Some(reason),
            AuditTarget::Decision(id),
            existing.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        self.get_decision(id)?.ok_or(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("decision"),
        })
    }

    pub fn edge_exists(
        &self,
        from: DecisionId,
        rel: RelationshipType,
        to: DecisionId,
    ) -> Result<bool> {
        let params: HashMap<String, Value> = [
            ("source_id".to_string(), Value::Uuid(from.as_uuid())),
            ("target_id".to_string(), Value::Uuid(to.as_uuid())),
            (
                "edge_type".to_string(),
                Value::Text(relationship_type_str(rel).to_string()),
            ),
        ]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT id FROM edges WHERE source_id = $source_id AND target_id = $target_id AND edge_type = $edge_type",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(!result.rows.is_empty())
    }

    pub fn decision_targets_intention(
        &self,
        decision_id: DecisionId,
        rel: RelationshipType,
        intention_id: IntentionId,
    ) -> Result<bool> {
        let params: HashMap<String, Value> = [
            ("source_id".to_string(), Value::Uuid(decision_id.as_uuid())),
            ("target_id".to_string(), Value::Uuid(intention_id.as_uuid())),
            (
                "edge_type".to_string(),
                Value::Text(relationship_type_str(rel).to_string()),
            ),
        ]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT id FROM edges WHERE source_id = $source_id AND target_id = $target_id AND edge_type = $edge_type",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(!result.rows.is_empty())
    }

    pub fn decision_targets_entity(
        &self,
        decision_id: DecisionId,
        rel: RelationshipType,
        entity_id: EntityId,
    ) -> Result<bool> {
        let params: HashMap<String, Value> = [
            ("source_id".to_string(), Value::Uuid(decision_id.as_uuid())),
            ("target_id".to_string(), Value::Uuid(entity_id.as_uuid())),
            (
                "edge_type".to_string(),
                Value::Text(relationship_type_str(rel).to_string()),
            ),
        ]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT id FROM edges WHERE source_id = $source_id AND target_id = $target_id AND edge_type = $edge_type",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(!result.rows.is_empty())
    }

    pub fn incoming_supersedes(&self, decision_id: DecisionId) -> Result<Vec<DecisionId>> {
        let params: HashMap<String, Value> = [
            ("target_id".to_string(), Value::Uuid(decision_id.as_uuid())),
            (
                "edge_type".to_string(),
                Value::Text("SUPERSEDES".to_string()),
            ),
        ]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT source_id FROM edges WHERE target_id = $target_id AND edge_type = $edge_type",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            if let Some(Value::Uuid(u)) = row.first() {
                out.push(DecisionId::from(*u));
            }
        }
        Ok(out)
    }

    pub fn replay(&self, id: DecisionId) -> Result<ReplayResult> {
        let snapshot = self.db.snapshot();
        let read_txid = crate::TxId::from_snapshot(snapshot);
        let params = HashMap::from([("id".to_string(), Value::Uuid(id.as_uuid()))]);
        let decision_rows = self
            .db
            .execute_at_snapshot(
                "SELECT id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at FROM decisions WHERE id = $id",
                &params,
                snapshot,
            )
            .map_err(map_engine_err)?;
        let Some(row) = decision_rows.rows.first() else {
            return Err(CgError::NotFound {
                id: Some(id.as_uuid().to_string()),
                missing: Some("decision"),
            });
        };
        let mut decision = decode_decision(&decision_rows.columns, row)?;
        decision.based_on_snapshots = self.basis_snapshot_ids_for_decision_at_read(id, snapshot)?;
        decision.intention_ids =
            self.intentions_for_decision_at_read(id, decision.context_id, snapshot)?;
        decision.outcome = self.outcome_for_decision_at_read(id, snapshot)?;

        let basis_snapshots = self.basis_snapshots_for_decision_at_read(&decision, snapshot)?;
        let mut input_diffs = Vec::with_capacity(basis_snapshots.len());
        let mut evidence_lineage = Vec::new();
        for basis_snapshot in basis_snapshots {
            let entity_id = basis_snapshot.entity_id;
            if basis_snapshot.evidenced_by.is_empty() {
                evidence_lineage.push(ReplayEvidenceLineage {
                    basis_snapshot_id: basis_snapshot.id,
                    observation_id: None,
                    evidence: Vec::new(),
                    read_txid,
                });
            } else {
                for observation_id in &basis_snapshot.evidenced_by {
                    evidence_lineage.push(ReplayEvidenceLineage {
                        basis_snapshot_id: basis_snapshot.id,
                        observation_id: Some(*observation_id),
                        evidence: self
                            .evidence_for_observation_at_read(*observation_id, snapshot)?,
                        read_txid,
                    });
                }
            }
            let current_snapshot = self.current_snapshot_for_entity_at_read(entity_id, snapshot)?;
            let field_changes =
                replay_field_changes(Some(&basis_snapshot), current_snapshot.as_ref());
            let changed = !field_changes.is_empty();
            let changed_at = if changed {
                current_snapshot
                    .as_ref()
                    .map(|snapshot| snapshot.valid_from)
                    .or(Some(basis_snapshot.valid_from))
            } else {
                None
            };
            input_diffs.push(InputDiff {
                entity_id,
                entity_name: current_snapshot
                    .as_ref()
                    .map(|snapshot| snapshot.name.clone())
                    .unwrap_or_else(|| basis_snapshot.name.clone()),
                changed,
                field_changes,
                changed_at,
            });
        }
        input_diffs.sort_by(|a, b| a.entity_id.cmp(&b.entity_id));
        let changed_input_count = input_diffs.iter().filter(|diff| diff.changed).count() as u32;

        let mut cited_intention_snapshots = Vec::with_capacity(decision.intention_ids.len());
        for intention_id in &decision.intention_ids {
            let snapshot_row =
                self.intention_snapshot_for_decision_at_read(*intention_id, &decision, snapshot)?;
            cited_intention_snapshots.push(snapshot_row);
        }
        cited_intention_snapshots.sort_by(|a, b| a.intention_id.cmp(&b.intention_id));

        let unchanged_input_count = input_diffs.len() as u32 - changed_input_count;
        let invalidations_at_read = self.invalidations_for_decision_at_read(id, snapshot)?;
        Ok(ReplayResult {
            decision_id: id,
            decision_status: decision.status,
            original_outcome: decision.outcome.as_ref().map(|outcome| outcome.success),
            inputs_changed: changed_input_count > 0,
            changed_input_count,
            unchanged_input_count,
            read_txid,
            input_diffs,
            evidence_lineage,
            invalidations_at_read,
            cited_intention_snapshots,
        })
    }

    fn intentions_for_decision_at_read(
        &self,
        id: DecisionId,
        context_id: ContextId,
        snapshot: SnapshotId,
    ) -> Result<Vec<IntentionId>> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT target_id FROM edges WHERE source_id = $source_id AND edge_type = $edge_type AND context_id = $context_id",
                &HashMap::from([
                    ("source_id".to_string(), Value::Uuid(id.as_uuid())),
                    ("edge_type".to_string(), Value::Text("SERVES".to_string())),
                    ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                ]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        let mut ids = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            if let Some(Value::Uuid(id)) = row.first() {
                ids.push(IntentionId::from(*id));
            }
        }
        ids.sort();
        Ok(ids)
    }

    fn outcome_for_decision_at_read(
        &self,
        id: DecisionId,
        snapshot: SnapshotId,
    ) -> Result<Option<Outcome>> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT id, decision_id, success, notes, recorded_at FROM outcomes WHERE decision_id = $decision_id",
                &HashMap::from([("decision_id".to_string(), Value::Uuid(id.as_uuid()))]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| decode_outcome(&result.columns, row))
            .transpose()
    }

    fn basis_snapshots_for_decision_at_read(
        &self,
        decision: &Decision,
        snapshot: SnapshotId,
    ) -> Result<Vec<EntitySnapshot>> {
        let mut out = Vec::with_capacity(decision.based_on_snapshots.len());
        for snapshot_id in &decision.based_on_snapshots {
            let Some(snapshot_row) = self.snapshot_by_id_at_read(*snapshot_id, snapshot)? else {
                return Err(CgError::Engine(format!(
                    "data-integrity: decision {} references missing basis snapshot {}",
                    decision.id, snapshot_id
                )));
            };
            let snapshot_context = self
                .entity_context_at_read(snapshot_row.entity_id, snapshot)?
                .ok_or_else(|| {
                    CgError::Engine(format!(
                        "data-integrity: basis snapshot {} references missing entity {}",
                        snapshot_row.id, snapshot_row.entity_id
                    ))
                })?;
            if snapshot_context != decision.context_id {
                return Err(CgError::CrossContextViolation);
            }
            out.push(snapshot_row);
        }
        Ok(out)
    }

    fn evidence_for_observation_at_read(
        &self,
        observation_id: ObservationId,
        snapshot: SnapshotId,
    ) -> Result<Vec<EvidenceRef>> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT id, observation_id, context_id, kind, source_ref, mime_type, content_hash, captured_at, captured_at_nanos, byte_range, time_range_ms, frame_index, region, producer, text_projection, derived_from, retention_status, media_retention_until, media_retention_until_nanos, properties, supply_order FROM evidence WHERE observation_id = $observation_id",
                &HashMap::from([(
                    "observation_id".to_string(),
                    Value::Uuid(observation_id.as_uuid()),
                )]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        let supply_order_idx = result
            .columns
            .iter()
            .position(|column| column == "supply_order")
            .unwrap_or(usize::MAX);
        let mut rows = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            let mut evidence = decode_evidence(&result.columns, row)?;
            if let Some(status) = self.evidence_status_at_snapshot(evidence.id, snapshot)? {
                evidence.retention_status = status;
            }
            let supply_order = match row.get(supply_order_idx) {
                Some(Value::Int64(value)) => *value,
                _ => i64::MAX,
            };
            rows.push((supply_order, evidence));
        }
        rows.sort_by_key(|row| row.0);
        Ok(rows.into_iter().map(|(_, evidence)| evidence).collect())
    }

    fn intention_snapshot_for_decision_at_read(
        &self,
        intention_id: IntentionId,
        decision: &Decision,
        snapshot: SnapshotId,
    ) -> Result<IntentionSnapshot> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT intention_id, description, status, origin, properties, tags, valid_from, valid_to FROM intention_snapshots WHERE intention_id = $intention_id AND valid_from <= $decision_tx",
                &HashMap::from([
                    (
                        "intention_id".to_string(),
                        Value::Uuid(intention_id.as_uuid()),
                    ),
                    ("decision_tx".to_string(), Value::TxId(decision.created_at)),
                ]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        let mut snapshots = Vec::new();
        for row in &result.rows {
            let snapshot_row = decode_intention_snapshot_at_read(&result.columns, row)?;
            if snapshot_row.valid_from <= decision.created_at
                && snapshot_row
                    .valid_to
                    .map(|valid_to| decision.created_at < valid_to)
                    .unwrap_or(true)
            {
                snapshots.push(snapshot_row);
            }
        }
        snapshots
            .into_iter()
            .max_by_key(|snapshot| snapshot.valid_from)
            .ok_or_else(|| {
                CgError::Engine(format!(
                    "data-integrity: SERVES edge to intention {} has no snapshot covering decision {}.created_at {}",
                    intention_id,
                    decision.id,
                    decision.created_at.0
                ))
            })
    }

    fn invalidations_for_decision_at_read(
        &self,
        decision_id: DecisionId,
        snapshot: SnapshotId,
    ) -> Result<Vec<Invalidation>> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id FROM invalidations WHERE affected_decision_id = $affected_decision_id",
                &HashMap::from([(
                    "affected_decision_id".to_string(),
                    Value::Uuid(decision_id.as_uuid()),
                )]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        let mut rows = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            rows.push(decode_invalidation_at_read(&result.columns, row)?);
        }
        rows.sort_by_key(|row| row.detected_at);
        Ok(rows)
    }
}

fn decode_intention_snapshot_at_read(
    columns: &[String],
    row: &[Value],
) -> Result<IntentionSnapshot> {
    Ok(IntentionSnapshot {
        intention_id: IntentionId::from(get_uuid(row, columns, "intention_id")?),
        description: get_text(row, columns, "description")?,
        status: intention_status_from_str(&get_text(row, columns, "status")?)?,
        origin: intention_origin_from_str(&get_text(row, columns, "origin")?)?,
        properties: properties_from_json(&get_json(row, columns, "properties")?),
        tags: tags_from_json(&get_json(row, columns, "tags")?),
        valid_from: get_txid(row, columns, "valid_from")?,
        valid_to: get_optional_txid(row, columns, "valid_to")?,
    })
}

fn decode_invalidation_at_read(columns: &[String], row: &[Value]) -> Result<Invalidation> {
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
        basis_diff: basis_diff_from_json(&get_json(row, columns, "basis_diff")?),
        cascade_from: optional_uuid(row, columns, "cascade_from")?.map(DecisionId::from),
        severity: invalidation_severity_from_str(&get_text(row, columns, "severity")?)?,
        threatened_intention_ids: intention_ids_from_json(&get_json(
            row,
            columns,
            "threatened_intention_ids",
        )?),
        intention_now_un_served: get_bool(row, columns, "intention_now_un_served")?,
        detected_at: get_txid(row, columns, "detected_at")?,
        resolved_at: get_optional_txid(row, columns, "resolved_at")?,
        resolution_action: get_optional_text(row, columns, "resolution_action")?,
        resolution_decision_id: optional_uuid(row, columns, "resolution_decision_id")?
            .map(DecisionId::from),
    })
}

fn optional_uuid(row: &[Value], columns: &[String], name: &str) -> Result<Option<uuid::Uuid>> {
    let Some(idx) = columns.iter().position(|column| column == name) else {
        return Ok(None);
    };
    match row.get(idx) {
        Some(Value::Uuid(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected uuid|null, got {other:?}"
        ))),
    }
}

fn invalidation_status_from_str(raw: &str) -> Result<InvalidationStatus> {
    match raw {
        "pending" => Ok(InvalidationStatus::Pending),
        "acknowledged" => Ok(InvalidationStatus::Acknowledged),
        "resolved" => Ok(InvalidationStatus::Resolved),
        "dismissed" => Ok(InvalidationStatus::Dismissed),
        other => Err(CgError::Engine(format!(
            "unknown invalidation status: {other}"
        ))),
    }
}

fn invalidation_severity_from_str(raw: &str) -> Result<InvalidationSeverity> {
    match raw {
        "info" => Ok(InvalidationSeverity::Info),
        "warning" => Ok(InvalidationSeverity::Warning),
        "critical" => Ok(InvalidationSeverity::Critical),
        other => Err(CgError::Engine(format!(
            "unknown invalidation severity: {other}"
        ))),
    }
}

fn basis_snapshot_edge_properties(snapshot: &EntitySnapshot) -> JsonValue {
    serde_json::json!({ "basis_snapshot": snapshot })
}

fn decision_basis_row_id(decision_id: DecisionId, basis_order: usize) -> uuid::Uuid {
    let mut bytes = *decision_id.as_uuid().as_bytes();
    let order = (basis_order as u64).to_be_bytes();
    for (idx, byte) in order.iter().enumerate() {
        bytes[8 + idx] ^= byte;
    }
    uuid::Uuid::from_bytes(bytes)
}

fn replay_field_changes(
    basis: Option<&EntitySnapshot>,
    current: Option<&EntitySnapshot>,
) -> Vec<FieldChange> {
    let mut fields = BTreeSet::new();
    if let Some(snapshot) = basis {
        fields.extend(snapshot.properties.keys().cloned());
    }
    if let Some(snapshot) = current {
        fields.extend(snapshot.properties.keys().cloned());
    }

    let mut changes = Vec::new();
    for field in fields {
        let old_value = basis.and_then(|snapshot| snapshot.properties.get(&field).cloned());
        let new_value = current.and_then(|snapshot| snapshot.properties.get(&field).cloned());
        if old_value != new_value {
            changes.push(FieldChange {
                field,
                old_value,
                new_value,
            });
        }
    }
    changes
}

fn decode_decision(columns: &[String], row: &[Value]) -> Result<Decision> {
    let reasoning_json = get_json(row, columns, "reasoning")?;
    let reasoning: Vec<ReasoningStep> = match reasoning_json {
        JsonValue::Null => Vec::new(),
        other => serde_json::from_value(other).unwrap_or_default(),
    };
    let confidence = get_optional_float(row, columns, "confidence")?.map(|f| f as f32);
    Ok(Decision {
        id: DecisionId::from(get_uuid(row, columns, "id")?),
        decision_type: get_text(row, columns, "decision_type")?,
        description: get_text(row, columns, "description")?,
        reasoning,
        confidence,
        status: decision_status_from_str(&get_text(row, columns, "status")?)?,
        agent_id: get_optional_text(row, columns, "agent_id")?,
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        tags: tags_from_json(&get_json(row, columns, "tags")?),
        properties: properties_from_json(&get_json(row, columns, "properties")?),
        created_at: get_txid(row, columns, "created_at")?,
        intention_ids: Vec::new(),
        based_on_snapshots: Vec::new(),
        outcome: None,
        precedent_ids: Vec::new(),
    })
}

fn ordered_snapshot_ids(
    columns: &[String],
    rows: &[Vec<Value>],
    order_column: &str,
    snapshot_column: &str,
) -> Result<Vec<EntitySnapshotId>> {
    let order_idx = columns
        .iter()
        .position(|column| column == order_column)
        .ok_or_else(|| CgError::Engine(format!("missing column {order_column}")))?;
    let snapshot_idx = columns
        .iter()
        .position(|column| column == snapshot_column)
        .ok_or_else(|| CgError::Engine(format!("missing column {snapshot_column}")))?;
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let order = match row.get(order_idx) {
            Some(Value::Int64(value)) => *value,
            other => {
                return Err(CgError::Engine(format!(
                    "column {order_column}: expected int, got {other:?}"
                )));
            }
        };
        let snapshot_id = match row.get(snapshot_idx) {
            Some(Value::Uuid(value)) => EntitySnapshotId::from(*value),
            other => {
                return Err(CgError::Engine(format!(
                    "column {snapshot_column}: expected uuid, got {other:?}"
                )));
            }
        };
        values.push((order, snapshot_id));
    }
    values.sort_by_key(|(order, snapshot_id)| (*order, snapshot_id.as_uuid()));
    Ok(values
        .into_iter()
        .map(|(_, snapshot_id)| snapshot_id)
        .collect())
}
