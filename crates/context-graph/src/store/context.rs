use std::collections::HashMap;

use contextdb_core::Value;
use serde_json::Value as JsonValue;

use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::map_engine_err;
use super::encode::{
    get_json, get_text, get_txid, get_uuid, properties_from_json, properties_to_json,
    tags_from_json,
};

impl Store {
    pub fn create_context(&self, input: CreateContext) -> Result<Context> {
        let id = ContextId::new_v7();
        let labels_json = JsonValue::Array(
            input
                .labels
                .iter()
                .map(|l| JsonValue::String(l.clone()))
                .collect(),
        );
        let properties_json = properties_to_json(&input.properties);

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
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("name".to_string(), Value::Text(input.name.clone())),
            ("labels".to_string(), Value::Json(labels_json)),
            ("properties".to_string(), Value::Json(properties_json)),
            ("created_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO contexts (id, name, labels, properties, created_at) VALUES ($id, $name, $labels, $properties, $created_at)",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) = self.record_audit_in_tx(
            tx,
            AuditOperation::Create,
            Principal::System,
            None,
            AuditTarget::Context(id),
            id,
            None,
        ) {
            let _ = self.db.rollback(tx);
            return Err(e);
        }
        self.db.commit(tx).map_err(map_engine_err)?;

        Ok(Context {
            id,
            name: input.name,
            labels: input.labels,
            properties: input.properties,
            created_at: tx,
        })
    }

    pub fn list_contexts(&self) -> Result<Vec<Context>> {
        let result = self
            .db
            .execute(
                "SELECT id, name, labels, properties, created_at FROM contexts",
                &HashMap::new(),
            )
            .map_err(map_engine_err)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            out.push(decode_context(&result.columns, row)?);
        }
        Ok(out)
    }

    pub fn get_context(&self, id: ContextId) -> Result<Option<Context>> {
        self.observe_read("get_context");
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "ContextId",
            });
        }
        let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT id, name, labels, properties, created_at FROM contexts WHERE id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        Ok(Some(decode_context(&result.columns, &result.rows[0])?))
    }
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
