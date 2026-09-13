use std::collections::HashMap;

use contextdb_core::Value;
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::map_engine_err;
use super::encode::{
    get_json, get_text, get_txid, get_uuid, properties_from_json, properties_to_json,
    tags_from_json,
};
use super::graph_rows::GraphRows;

impl Store {
    pub fn create_context(&self, input: CreateContext) -> Result<Context> {
        let placement = self.event_placement_state()?;
        self.require_event_write_ready()?;
        let id = deterministic_context_id(&input.name);

        let existing = self
            .db
            .execute(
                "SELECT id FROM contexts WHERE name = $name",
                &[("name".to_string(), Value::Text(input.name.clone()))]
                    .into_iter()
                    .collect::<HashMap<_, _>>(),
            )
            .map_err(map_engine_err)?;
        if !existing.rows.is_empty() {
            return Err(CgError::UniqueViolation {
                entity: "context",
                id: None,
            });
        }

        let tx = self.begin_tx()?;
        if let Err(error) = self.guard_event_placement_in_tx(tx, placement) {
            let _ = self.db.rollback(tx);
            return Err(error);
        }
        let context = match self.insert_context_in_tx(tx, &input, id) {
            Ok(context) => context,
            Err(error) => {
                let _ = self.db.rollback(tx);
                return Err(error);
            }
        };
        self.db.commit(tx).map_err(map_engine_err)?;
        Ok(context)
    }

    pub fn list_contexts(&self) -> Result<Vec<Context>> {
        list_contexts(self.as_graph_rows())
    }

    pub fn get_context(&self, id: ContextId) -> Result<Option<Context>> {
        self.observe_read("get_context");
        get_context(self.as_graph_rows(), id)
    }

    pub(super) fn insert_context_in_tx(
        &self,
        tx: crate::TxId,
        input: &CreateContext,
        id: ContextId,
    ) -> Result<Context> {
        let labels_json = JsonValue::Array(
            input
                .labels
                .iter()
                .map(|label| JsonValue::String(label.clone()))
                .collect(),
        );
        let properties_json = properties_to_json(&input.properties);
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("name".to_string(), Value::Text(input.name.clone())),
            ("labels".to_string(), Value::Json(labels_json)),
            ("properties".to_string(), Value::Json(properties_json)),
            ("created_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO contexts (id, name, labels, properties, created_at) VALUES ($id, $name, $labels, $properties, $created_at)",
                &params,
            )
            .map_err(map_engine_err)?;
        self.record_audit_in_tx(
            tx,
            AuditOperation::Create,
            Principal::System,
            None,
            AuditTarget::Context(id),
            id,
            None,
        )?;
        Ok(Context {
            id,
            name: input.name.clone(),
            labels: input.labels.clone(),
            properties: input.properties.clone(),
            created_at: tx,
        })
    }
}

/// Every context in the store, as both doors answer it.
///
/// Reads `contexts`.
pub(crate) fn list_contexts(rows: &dyn GraphRows) -> Result<Vec<Context>> {
    let result = rows.query_rows(
        "SELECT id, name, labels, properties, created_at FROM contexts",
        &HashMap::new(),
    )?;
    let mut out = Vec::with_capacity(result.rows.len());
    for row in &result.rows {
        out.push(decode_context(&result.columns, row)?);
    }
    Ok(out)
}

/// One context by id, or the answered absence when the store holds no such row. An id that is not
/// a well-formed identity is refused before the store is asked, on both doors alike.
///
/// Reads `contexts`.
pub(crate) fn get_context(rows: &dyn GraphRows, id: ContextId) -> Result<Option<Context>> {
    if !id.is_valid() {
        return Err(CgError::InvalidIdentity {
            context: "ContextId",
        });
    }
    let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
        .into_iter()
        .collect();
    let result = rows.query_rows(
        "SELECT id, name, labels, properties, created_at FROM contexts WHERE id = $id",
        &params,
    )?;
    if result.rows.is_empty() {
        return Ok(None);
    }
    Ok(Some(decode_context(&result.columns, &result.rows[0])?))
}

pub(super) fn deterministic_context_id(name: &str) -> ContextId {
    let mut hasher = Sha256::new();
    hasher.update(b"context-graph/context/");
    hasher.update(name.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    ContextId::from(Uuid::from_bytes(bytes))
}

fn decode_context(columns: &[String], row: &[Value]) -> Result<Context> {
    Ok(Context {
        id: ContextId::from(get_uuid(row, columns, "id")?),
        name: get_text(row, columns, "name")?,
        labels: tags_from_json(&get_json(row, columns, "labels")?),
        properties: properties_from_json(&get_json(row, columns, "properties")?),
        created_at: get_txid(row, columns, "created_at")?,
    })
}
