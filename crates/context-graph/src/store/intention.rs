use std::collections::HashMap;

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
    get_json, get_optional_txid, get_text, get_txid, get_uuid, properties_from_json,
    properties_to_json, tags_from_json,
};

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

        let tx = self.db.begin();
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
            "INSERT INTO intentions (id, description, status, context_id, origin, origin_ref, blueprint_id, bindings, blueprint_proposal, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $origin_ref, $blueprint_id, $bindings, $blueprint_proposal, $properties, $tags, $created_at)",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) = self.insert_intention_snapshot_in_tx(
            tx,
            id,
            &input.description,
            input.status,
            input.origin,
            &properties_json,
            &tags_json,
            tx,
        ) {
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
        })
    }

    pub fn get_intention(&self, id: IntentionId) -> Result<Option<Intention>> {
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
                "SELECT id, description, status, origin, context_id, properties, tags, created_at FROM intentions WHERE id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        Ok(Some(decode_intention(&result.columns, &result.rows[0])?))
    }

    pub fn intention_tree(&self, intention_id: IntentionId) -> Result<IntentionTree> {
        // Wrong: return the intention with empty collections. Drives every
        // intention_tree_* assertion (decisions, entities, outcomes, invalidations
        // all have concrete non-empty expected shapes).
        let intention = self.get_intention(intention_id)?.ok_or(CgError::NotFound {
            id: Some(intention_id.as_uuid().to_string()),
            missing: Some("intention"),
        })?;
        Ok(IntentionTree {
            intention,
            decisions: Vec::new(),
            entities: Vec::new(),
            outcomes: Vec::new(),
            invalidations: Vec::new(),
        })
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
        let mut currents: Vec<Intention> = Vec::with_capacity(patches.len());
        for (id, _) in &patches {
            match self.get_intention(*id)? {
                Some(i) => currents.push(i),
                None => {
                    return Err(CgError::NotFound {
                        id: Some(id.as_uuid().to_string()),
                        missing: Some("intention"),
                    });
                }
            }
        }

        let tx = self.db.begin();
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

        // Close prior snapshot, open a new one in the same tx.
        self.close_open_intention_snapshot_in_tx(tx, current.id, tx)?;
        let new_properties_json = properties_to_json(&new_properties);
        let new_tags_json = JsonValue::Array(
            new_tags
                .iter()
                .map(|t| JsonValue::String(t.clone()))
                .collect(),
        );
        self.insert_intention_snapshot_in_tx(
            tx,
            current.id,
            &new_description,
            new_status,
            current.origin,
            &new_properties_json,
            &new_tags_json,
            tx,
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

    pub fn archive_intention(&self, id: IntentionId, reason: String) -> Result<Intention> {
        let current = self.get_intention(id)?.ok_or_else(|| CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("intention"),
        })?;
        let active_serving = self.decisions_actively_serving_intention(id)?;
        if !active_serving.is_empty() {
            return Err(CgError::ActiveDecisionsReference {
                intention_id: id,
                decision_ids: active_serving
                    .iter()
                    .map(|d| d.as_uuid().to_string())
                    .collect(),
            });
        }

        let tx = self.db.begin();
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
        if let Err(e) = self.close_open_intention_snapshot_in_tx(tx, id, tx) {
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

    pub(super) fn intention_context_ids(
        &self,
        ids: &[IntentionId],
    ) -> Result<HashMap<IntentionId, ContextId>> {
        let mut out: HashMap<IntentionId, ContextId> = HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        for iid in ids {
            let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(iid.as_uuid()))]
                .into_iter()
                .collect();
            let r = self
                .db
                .execute(
                    "SELECT id, context_id FROM intentions WHERE id = $id",
                    &params,
                )
                .map_err(map_engine_err)?;
            if r.rows.is_empty() {
                continue;
            }
            let ictx = match r.rows[0].get(1) {
                Some(Value::Uuid(u)) => ContextId::from(*u),
                _ => {
                    return Err(CgError::Engine(
                        "intention row missing context_id".to_string(),
                    ));
                }
            };
            out.insert(*iid, ictx);
        }
        Ok(out)
    }

    fn decisions_actively_serving_intention(
        &self,
        intention_id: IntentionId,
    ) -> Result<Vec<DecisionId>> {
        let params: HashMap<String, Value> = [(
            "intention_id".to_string(),
            Value::Uuid(intention_id.as_uuid()),
        )]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT b_id FROM GRAPH_TABLE(edges MATCH (a)<-[:SERVES]-(b) WHERE a.id = $intention_id COLUMNS (b.id AS b_id))",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut active = Vec::new();
        for row in &result.rows {
            if let Some(Value::Uuid(source)) = row.first() {
                let decision_id = DecisionId::from(*source);
                let dparams: HashMap<String, Value> = [("id".to_string(), Value::Uuid(*source))]
                    .into_iter()
                    .collect();
                let dresult = self
                    .db
                    .execute("SELECT status FROM decisions WHERE id = $id", &dparams)
                    .map_err(map_engine_err)?;
                if let Some(row) = dresult.rows.first()
                    && let Some(Value::Text(status)) = row.first()
                    && status == "active"
                {
                    active.push(decision_id);
                }
            }
        }
        Ok(active)
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
        snaps.sort_by_key(|s| s.valid_from.0);
        Ok(snaps)
    }

    pub fn intention_at(&self, id: IntentionId, at: TxId) -> Result<Option<IntentionSnapshot>> {
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
        for row in &result.rows {
            let snap = decode_intention_snapshot(&result.columns, row)?;
            let in_range =
                snap.valid_from.0 <= at.0 && snap.valid_to.map(|t| t.0 > at.0).unwrap_or(true);
            if in_range {
                return Ok(Some(snap));
            }
        }
        Ok(None)
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
        let tx = self.db.begin();
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
            "INSERT INTO intentions (id, description, status, context_id, origin, origin_ref, blueprint_id, bindings, blueprint_proposal, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $origin_ref, $blueprint_id, $bindings, $blueprint_proposal, $properties, $tags, $created_at)",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        self.db.commit(tx).map_err(map_engine_err)?;
        Ok(())
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
                "SELECT id, description, status, origin, context_id, properties, tags, created_at FROM intentions WHERE id = $id",
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
        intention_id: IntentionId,
        description: &str,
        status: IntentionStatus,
        origin: IntentionOrigin,
        properties_json: &JsonValue,
        tags_json: &JsonValue,
        valid_from: TxId,
    ) -> Result<()> {
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(Uuid::now_v7())),
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
                "INSERT INTO intention_snapshots (id, intention_id, description, status, origin, properties, tags, valid_from, valid_to) VALUES ($id, $intention_id, $description, $status, $origin, $properties, $tags, $valid_from, $valid_to)",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(())
    }

    pub(super) fn close_open_intention_snapshot_in_tx(
        &self,
        tx: TxId,
        intention_id: IntentionId,
        valid_to: TxId,
    ) -> Result<()> {
        let params: HashMap<String, Value> = [
            (
                "intention_id".to_string(),
                Value::Uuid(intention_id.as_uuid()),
            ),
            ("valid_to".to_string(), Value::TxId(valid_to)),
        ]
        .into_iter()
        .collect();
        self.db
            .execute_in_tx(
                tx,
                "UPDATE intention_snapshots SET valid_to = $valid_to WHERE intention_id = $intention_id AND valid_to IS NULL",
                &params,
            )
            .map_err(map_engine_err)?;
        Ok(())
    }
}

fn decode_intention(columns: &[String], row: &[Value]) -> Result<Intention> {
    Ok(Intention {
        id: IntentionId::from(get_uuid(row, columns, "id")?),
        description: get_text(row, columns, "description")?,
        status: intention_status_from_str(&get_text(row, columns, "status")?)?,
        origin: intention_origin_from_str(&get_text(row, columns, "origin")?)?,
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
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
