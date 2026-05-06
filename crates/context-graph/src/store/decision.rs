use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use contextdb_core::{Error as EngineError, Value};
use serde_json::Value as JsonValue;

use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::{decision_status_from_str, map_engine_err, relationship_type_str};
use super::encode::{
    decode_outcome, get_json, get_optional_float, get_optional_text, get_text, get_txid, get_uuid,
    properties_from_json, properties_to_json, tags_from_json,
};

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
        let mut basis_snapshots = Vec::with_capacity(input.based_on_entity_ids.len());
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
            basis_snapshots.push((*eid, basis_snapshot));
        }

        let tx = self.db.begin();
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
            (
                "based_on_snapshots".to_string(),
                Value::Json(serde_json::to_value(&basis_snapshot_ids).unwrap_or_default()),
            ),
            ("properties".to_string(), Value::Json(properties_json)),
            ("tags".to_string(), Value::Json(tags_json)),
            ("created_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, based_on_snapshots, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $based_on_snapshots, $properties, $tags, $created_at)",
            &decision_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
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
        for (eid, basis_snapshot) in &basis_snapshots {
            if let Err(e) = self.insert_edge_in_tx(
                tx,
                id.as_uuid(),
                eid.as_uuid(),
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
                "SELECT id, decision_type, description, reasoning, confidence, status, agent_id, context_id, based_on_snapshots, properties, tags, created_at FROM decisions WHERE id = $id",
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
        let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT based_on_snapshots FROM decisions WHERE id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        if let Some(Value::Json(v)) = result.rows.first().and_then(|row| row.first()) {
            return Ok(serde_json::from_value(v.clone()).unwrap_or_default());
        }
        Ok(Vec::new())
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
                "SELECT id, decision_type, description, reasoning, confidence, status, agent_id, context_id, based_on_snapshots, properties, tags, created_at FROM decisions",
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
        let tx = self.db.begin();
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
        let mut basis_snapshots = Vec::with_capacity(new.based_on_entity_ids.len());
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
            basis_snapshots.push((*eid, basis_snapshot));
        }

        let tx = self.db.begin();
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
            (
                "based_on_snapshots".to_string(),
                Value::Json(serde_json::to_value(&basis_snapshot_ids).unwrap_or_default()),
            ),
            ("properties".to_string(), Value::Json(properties_json)),
            ("tags".to_string(), Value::Json(tags_json)),
            ("created_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, based_on_snapshots, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $based_on_snapshots, $properties, $tags, $created_at)",
            &decision_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
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
        for (eid, basis_snapshot) in &basis_snapshots {
            if let Err(e) = self.insert_edge_in_tx(
                tx,
                new_id.as_uuid(),
                eid.as_uuid(),
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
        let tx = self.db.begin();
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
        let tx = self.db.begin();
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
        let decision = self.get_decision(id)?.ok_or(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("decision"),
        })?;
        let read_txid = crate::TxId::from_snapshot(self.db.snapshot());

        let basis_snapshots = self.basis_snapshot_entities_for_decision(id, decision.context_id)?;
        let mut input_diffs = Vec::with_capacity(basis_snapshots.len());
        let mut evidence_lineage = Vec::new();
        for (entity_id, basis_snapshot) in basis_snapshots {
            if basis_snapshot.evidenced_by.is_empty() {
                evidence_lineage.push(ReplayEvidenceLineage {
                    basis_snapshot_id: basis_snapshot.id,
                    observation_id: None,
                    evidence: Vec::new(),
                    read_txid,
                });
            } else {
                evidence_lineage.extend(basis_snapshot.evidenced_by.iter().map(|observation_id| {
                    ReplayEvidenceLineage {
                        basis_snapshot_id: basis_snapshot.id,
                        observation_id: Some(*observation_id),
                        evidence: Vec::new(),
                        read_txid,
                    }
                }));
            }
            let current_snapshot = self.latest_entity_snapshot(entity_id)?;
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
            if let Some(intention) = self.get_intention(*intention_id)?
                && intention.context_id != decision.context_id
            {
                return Err(CgError::CrossContextViolation);
            }
            let snapshot = self
                .intention_at(*intention_id, decision.created_at)?
                .ok_or_else(|| {
                    CgError::Engine(format!(
                        "data-integrity: SERVES edge to intention {} has no snapshot covering decision {}.created_at {}",
                        intention_id,
                        decision.id,
                        decision.created_at.0
                    ))
                })?;
            cited_intention_snapshots.push(snapshot);
        }
        cited_intention_snapshots.sort_by(|a, b| a.intention_id.cmp(&b.intention_id));

        let unchanged_input_count = input_diffs.len() as u32 - changed_input_count;
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
            invalidations_at_read: Vec::new(),
            cited_intention_snapshots,
        })
    }

    fn basis_snapshot_entities_for_decision(
        &self,
        id: DecisionId,
        context_id: ContextId,
    ) -> Result<BTreeMap<EntityId, EntitySnapshot>> {
        let params: HashMap<String, Value> = [
            ("source_id".to_string(), Value::Uuid(id.as_uuid())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("edge_type".to_string(), Value::Text("BASED_ON".to_string())),
        ]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT target_id, properties FROM edges WHERE source_id = $source_id AND context_id = $context_id AND edge_type = $edge_type",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut out = BTreeMap::new();
        for row in &result.rows {
            let entity_id = EntityId::from(get_uuid(row, &result.columns, "target_id")?);
            let properties = get_json(row, &result.columns, "properties")?;
            let snapshot = match properties.get("basis_snapshot") {
                Some(value) => serde_json::from_value(value.clone())
                    .map_err(|e| CgError::Engine(format!("basis snapshot deserialize: {e}")))?,
                None => continue,
            };
            out.insert(entity_id, snapshot);
        }
        Ok(out)
    }

    fn latest_entity_snapshot(&self, entity_id: EntityId) -> Result<Option<EntitySnapshot>> {
        Ok(self.entity_history(entity_id)?.into_iter().last())
    }
}

fn basis_snapshot_edge_properties(snapshot: &EntitySnapshot) -> JsonValue {
    serde_json::json!({ "basis_snapshot": snapshot })
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
        based_on_snapshots: serde_json::from_value(
            get_json(row, columns, "based_on_snapshots").unwrap_or(JsonValue::Array(Vec::new())),
        )
        .unwrap_or_default(),
        outcome: None,
        precedent_ids: Vec::new(),
    })
}
