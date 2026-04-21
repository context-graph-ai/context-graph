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
    get_json, get_text, get_timestamp, get_uuid, properties_from_json, properties_to_json,
};

impl Store {
    pub fn record_observation(&self, input: RecordObservation) -> Result<RecordObservationResult> {
        let entity = self.get_entity(input.entity_id)?.ok_or(CgError::NotFound {
            id: Some(input.entity_id.as_uuid().to_string()),
            missing: Some("entity"),
        })?;
        if entity.context_id != input.context_id {
            return Err(CgError::CrossContextObservation);
        }
        let id = ObservationId::new_v7();
        let observed_at = input.observed_at.unwrap_or_else(Utc::now);
        let data_json = properties_to_json(&input.data);

        let tx = self.db.begin();
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            (
                "observation_type".to_string(),
                Value::Text(input.observation_type.clone()),
            ),
            ("data".to_string(), Value::Json(data_json)),
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
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO observations (id, observation_type, data, source, entity_id, context_id, timestamp) VALUES ($id, $observation_type, $data, $source, $entity_id, $context_id, $timestamp)",
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

        Ok(RecordObservationResult { observation_id: id })
    }

    pub fn get_observation(&self, id: ObservationId) -> Result<Option<Observation>> {
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
                "SELECT id, observation_type, data, source, entity_id, context_id, timestamp FROM observations WHERE id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        Ok(Some(decode_observation(&result.columns, &result.rows[0])?))
    }

    pub fn admin_overwrite_observation(
        &self,
        _id: ObservationId,
        _data: BTreeMap<String, JsonValue>,
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
}

fn decode_observation(columns: &[String], row: &[Value]) -> Result<Observation> {
    let ts = get_timestamp(row, columns, "timestamp")?;
    let observed_at = DateTime::<Utc>::from_timestamp_millis(ts).unwrap_or_else(Utc::now);
    Ok(Observation {
        id: ObservationId::from(get_uuid(row, columns, "id")?),
        entity_id: EntityId::from(get_uuid(row, columns, "entity_id")?),
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        observation_type: get_text(row, columns, "observation_type")?,
        source: get_text(row, columns, "source")?,
        data: properties_from_json(&get_json(row, columns, "data")?),
        observed_at,
    })
}
