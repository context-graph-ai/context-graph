use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use contextdb_core::Value;
use serde_json::Value as JsonValue;

use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::map_engine_err;
use super::encode::{
    get_json, get_text, get_timestamp, get_txid, get_uuid, properties_from_json, properties_to_json,
};

impl Store {
    pub fn record_observation(&self, input: RecordObservation) -> Result<RecordObservationResult> {
        let entity = self.get_entity(input.entity_id)?;
        if let Some(entity) = entity {
            if entity.context_id != input.context_id {
                return Err(CgError::CrossContextObservation);
            }
        } else {
            self.ensure_context_and_entity(input.context_id, input.entity_id)?;
        }
        let id = input.id;
        let observed_at = input.observed_at;
        let observed_properties_json = properties_to_json(&input.observed_properties);
        let state_delta_json =
            serde_json::to_value(&input.state_delta).unwrap_or_else(|_| serde_json::json!({}));
        let properties_json = properties_to_json(&input.properties);

        let tx = self.db.begin();
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            (
                "observation_type".to_string(),
                Value::Text(input.observation_type.clone()),
            ),
            (
                "observed_properties".to_string(),
                Value::Json(observed_properties_json),
            ),
            ("state_delta".to_string(), Value::Json(state_delta_json)),
            ("properties".to_string(), Value::Json(properties_json)),
            ("source".to_string(), Value::Text(input.source.clone())),
            (
                "entity_id".to_string(),
                Value::Uuid(input.entity_id.as_uuid()),
            ),
            (
                "context_id".to_string(),
                Value::Uuid(input.context_id.as_uuid()),
            ),
            (
                "timestamp".to_string(),
                Value::Timestamp(observed_at.timestamp_millis()),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO observations (id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at) VALUES ($id, $observation_type, $observed_properties, $state_delta, $properties, $source, $entity_id, $context_id, $timestamp, $created_at)",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::RecordObservation,
            Principal::System,
            None,
            AuditTarget::Observation(id),
            input.context_id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        Ok(RecordObservationResult {
            observation_id: id,
            evidence_ids: Vec::new(),
            embedding_ids: Vec::new(),
            snapshot_id: None,
        })
    }

    pub fn get_observation(&self, id: ObservationId) -> Result<Observation> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "ObservationId",
            });
        }
        let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at FROM observations WHERE id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        if result.rows.is_empty() {
            return Err(CgError::NotFound {
                id: Some(id.as_uuid().to_string()),
                missing: Some("observation"),
            });
        }
        decode_observation(&result.columns, &result.rows[0])
    }

    pub fn list_observations(&self, context_id: Option<ContextId>) -> Result<Vec<Observation>> {
        let (sql, params) = if let Some(context_id) = context_id {
            (
                "SELECT id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at FROM observations WHERE context_id = $context_id",
                HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]),
            )
        } else {
            (
                "SELECT id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at FROM observations",
                HashMap::new(),
            )
        };
        let result = self.db.execute(sql, &params).map_err(map_engine_err)?;
        result
            .rows
            .iter()
            .map(|row| decode_observation(&result.columns, row))
            .collect()
    }

    pub fn admin_overwrite_observation(
        &self,
        _id: ObservationId,
        _observed_properties: BTreeMap<String, JsonValue>,
    ) -> Result<()> {
        Err(CgError::ImmutableEntity {
            kind: "observation",
        })
    }

    pub fn admin_raw_observations_for_entity_in_context(
        &self,
        entity_id: EntityId,
        context_id: ContextId,
    ) -> Result<Vec<ObservationId>> {
        let params: HashMap<String, Value> = [
            ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
        ]
        .into_iter()
        .collect();
        let result = self
            .db
            .execute(
                "SELECT id FROM observations WHERE entity_id = $entity_id AND context_id = $context_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            if let Some(Value::Uuid(u)) = row.first() {
                out.push(ObservationId::from(*u));
            }
        }
        Ok(out)
    }

    fn ensure_context_and_entity(&self, context_id: ContextId, entity_id: EntityId) -> Result<()> {
        let tx = self.db.begin();
        let context_params = HashMap::from([
            ("id".to_string(), Value::Uuid(context_id.as_uuid())),
            (
                "name".to_string(),
                Value::Text(format!("context-{context_id}")),
            ),
            ("labels".to_string(), Value::Json(serde_json::json!([]))),
            ("properties".to_string(), Value::Json(serde_json::json!({}))),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        let _ = self.db.execute_in_tx(
            tx,
            "INSERT INTO contexts (id, name, labels, properties, created_at) VALUES ($id, $name, $labels, $properties, $created_at)",
            &context_params,
        );
        let entity_params = HashMap::from([
            ("id".to_string(), Value::Uuid(entity_id.as_uuid())),
            (
                "entity_type".to_string(),
                Value::Text("SERVICE".to_string()),
            ),
            (
                "name".to_string(),
                Value::Text(format!("entity-{entity_id}")),
            ),
            ("status".to_string(), Value::Text("active".to_string())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("properties".to_string(), Value::Json(serde_json::json!({}))),
            ("tags".to_string(), Value::Json(serde_json::json!([]))),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        if let Err(err) = self.db.execute_in_tx(
            tx,
            "INSERT INTO entities (id, entity_type, name, status, context_id, properties, tags, created_at) VALUES ($id, $entity_type, $name, $status, $context_id, $properties, $tags, $created_at)",
            &entity_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(err));
        }
        self.db.commit(tx).map_err(map_engine_err)
    }
}

fn decode_observation(columns: &[String], row: &[Value]) -> Result<Observation> {
    let ts = get_timestamp(row, columns, "timestamp")?;
    let observed_at = DateTime::<Utc>::from_timestamp_millis(ts).unwrap_or_else(Utc::now);
    Ok(Observation {
        id: ObservationId::from(get_uuid(row, columns, "id")?),
        observation_type: get_text(row, columns, "observation_type")?,
        evidence: Vec::new(),
        observed_properties: properties_from_json(&get_json(row, columns, "observed_properties")?),
        state_delta: serde_json::from_value(get_json(row, columns, "state_delta")?)
            .unwrap_or_default(),
        properties: properties_from_json(&get_json(row, columns, "properties")?),
        source: get_text(row, columns, "source")?,
        entity_id: EntityId::from(get_uuid(row, columns, "entity_id")?),
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        observed_at,
        created_at: get_txid(row, columns, "created_at")?,
    })
}
