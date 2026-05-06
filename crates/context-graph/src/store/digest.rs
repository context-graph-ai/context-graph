use std::collections::HashMap;

use contextdb_core::{Value, VectorIndexRef};

use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::map_engine_err;
use super::encode::{
    get_json, get_optional_text, get_text, get_txid, get_uuid, get_vector, properties_from_json,
    properties_to_json,
};
use super::evidence::{
    embedding_target_kind, embedding_target_uuid, producer_value, validate_space_kind,
    vector_binding_for_target,
};
use super::evidence_kind_str;

impl Store {
    pub fn create_digest(&self, input: CreateDigest) -> Result<Digest> {
        if input.kind != EvidenceKind::Text {
            return Err(CgError::UnsupportedEvidenceKind { kind: input.kind });
        }
        if input.embedding.target != EmbeddingTarget::Digest(input.id) {
            return Err(CgError::EmbeddingSpaceMismatch {
                conflicting_field: "target",
            });
        }
        let space = self.embedding_space_for_id(&input.embedding.embedding_space_id)?;
        if input.embedding.vector.len() != space.dimension {
            return Err(CgError::EmbeddingSpaceMismatch {
                conflicting_field: "dimension",
            });
        }
        validate_space_kind(&space, input.kind)?;
        let binding = vector_binding_for_target(&space, "digest")?;
        let id = input.id;
        let properties_json = properties_to_json(&input.properties);
        let session_ref_value = match &input.session_ref {
            Some(s) => Value::Text(s.clone()),
            None => Value::Null,
        };

        let tx = self.db.begin();
        let context_params = HashMap::from([
            ("id".to_string(), Value::Uuid(input.context_id.as_uuid())),
            (
                "name".to_string(),
                Value::Text(format!("context-{}", input.context_id)),
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
        let params: HashMap<String, Value> = [
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("source".to_string(), Value::Text(input.source.clone())),
            ("session_ref".to_string(), session_ref_value),
            ("summary".to_string(), Value::Text(input.summary.clone())),
            (
                "kind".to_string(),
                Value::Text(evidence_kind_str(input.kind).to_string()),
            ),
            (
                "vector_text".to_string(),
                Value::Vector(input.embedding.vector.clone()),
            ),
            (
                "dimension".to_string(),
                Value::Int64(space.dimension as i64),
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
        let row_id = match self.db.insert_row(tx, "digests", params) {
            Ok(row_id) => row_id,
            Err(e) => {
                let _ = self.db.rollback(tx);
                return Err(map_engine_err(e));
            }
        };
        let handle_params = HashMap::from([
            ("id".to_string(), Value::Uuid(input.embedding.id.as_uuid())),
            (
                "context_id".to_string(),
                Value::Uuid(input.context_id.as_uuid()),
            ),
            (
                "embedding_space_id".to_string(),
                Value::Text(input.embedding.embedding_space_id.clone()),
            ),
            (
                "target_kind".to_string(),
                Value::Text(embedding_target_kind(&input.embedding.target).to_string()),
            ),
            (
                "target_id".to_string(),
                Value::Uuid(embedding_target_uuid(&input.embedding.target)),
            ),
            (
                "dimension".to_string(),
                Value::Int64(space.dimension as i64),
            ),
            (
                "produced_by".to_string(),
                producer_value(&input.embedding.produced_by),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO embedding_handles (id, context_id, embedding_space_id, target_kind, target_id, dimension, produced_by, created_at) VALUES ($id, $context_id, $embedding_space_id, $target_kind, $target_id, $dimension, $produced_by, $created_at)",
            &handle_params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        if let Err(e) = self.db.insert_vector(
            tx,
            VectorIndexRef::new(&binding.table_name, &binding.vector_column),
            row_id,
            input.embedding.vector.clone(),
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(e));
        }
        // Digests use a dedicated audit target kind downstream. For now context_id
        // scoping covers provenance; no audit_log row is needed here.
        self.db.commit(tx).map_err(map_engine_err)?;

        Ok(Digest {
            id,
            source: input.source,
            summary: input.summary,
            vector_text: Some(input.embedding.vector.clone()),
            dimension: space.dimension,
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
                "SELECT id, source, session_ref, summary, vector_text, dimension, context_id, properties, created_at FROM digests WHERE id = $id",
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
        vector_text: get_vector(row, columns, "vector_text").ok(),
        dimension: match row.get(
            columns
                .iter()
                .position(|c| c == "dimension")
                .unwrap_or(usize::MAX),
        ) {
            Some(Value::Int64(n)) => *n as usize,
            _ => 0,
        },
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        session_ref: get_optional_text(row, columns, "session_ref")?,
        properties: properties_from_json(&get_json(row, columns, "properties")?),
        created_at: get_txid(row, columns, "created_at")?,
    })
}
