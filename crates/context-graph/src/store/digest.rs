use std::collections::HashMap;

use contextdb_core::Value;

use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::map_engine_err;
use super::encode::{
    get_json, get_optional_text, get_text, get_txid, get_uuid, get_vector, properties_from_json,
    properties_to_json,
};

impl Store {
    pub fn create_digest(&self, input: CreateDigest) -> Result<Digest> {
        self.require_context(input.context_id)?;
        if input.embedding.len() != crate::schema::DIGEST_VECTOR_DIMENSION {
            return Err(CgError::VectorDimensionMismatch {
                expected: crate::schema::DIGEST_VECTOR_DIMENSION,
                got: input.embedding.len(),
            });
        }
        let id = DigestId::new_v7();
        let properties_json = properties_to_json(&input.properties);
        let session_ref_value = match &input.session_ref {
            Some(s) => Value::Text(s.clone()),
            None => Value::Null,
        };

        let tx = self.db.begin();
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("source".to_string(), Value::Text(input.source.clone())),
            ("session_ref".to_string(), session_ref_value),
            ("summary".to_string(), Value::Text(input.summary.clone())),
            (
                "embedding".to_string(),
                Value::Vector(input.embedding.clone()),
            ),
            (
                "context_id".to_string(),
                Value::Uuid(input.context_id.as_uuid()),
            ),
            ("properties".to_string(), Value::Json(properties_json)),
            ("created_at".to_string(), Value::TxId(tx)),
        ]
        .into_iter()
        .collect();
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO digests (id, source, session_ref, summary, embedding, context_id, properties, created_at) VALUES ($id, $source, $session_ref, $summary, $embedding, $context_id, $properties, $created_at)",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        // digests use a dedicated audit target kind within plan 16 scope. For now
        // context_id scoping covers provenance; no audit_log row is needed here.
        self.db.commit(tx).map_err(map_engine_err)?;

        Ok(Digest {
            id,
            source: input.source,
            summary: input.summary,
            embedding: input.embedding,
            context_id: input.context_id,
            session_ref: input.session_ref,
            properties: input.properties,
            created_at: tx,
        })
    }

    pub fn get_digest(&self, id: DigestId) -> Result<Option<Digest>> {
        if !id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "DigestId",
            });
        }
        let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
            .into_iter()
            .collect();
        let result = self
            .db
            .execute(
                "SELECT id, source, session_ref, summary, embedding, context_id, properties, created_at FROM digests WHERE id = $id",
                &params,
            )
            .map_err(map_engine_err)?;
        if result.rows.is_empty() {
            return Ok(None);
        }
        Ok(Some(decode_digest(&result.columns, &result.rows[0])?))
    }

    pub fn admin_overwrite_digest(&self, _id: DigestId, _summary: String) -> Result<()> {
        Err(CgError::ImmutableEntity { kind: "digest" })
    }
}

fn decode_digest(columns: &[String], row: &[Value]) -> Result<Digest> {
    Ok(Digest {
        id: DigestId::from(get_uuid(row, columns, "id")?),
        source: get_text(row, columns, "source")?,
        summary: get_text(row, columns, "summary")?,
        embedding: get_vector(row, columns, "embedding")?,
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        session_ref: get_optional_text(row, columns, "session_ref")?,
        properties: properties_from_json(&get_json(row, columns, "properties")?),
        created_at: get_txid(row, columns, "created_at")?,
    })
}
