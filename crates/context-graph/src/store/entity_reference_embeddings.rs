use serde::{Deserialize, Serialize};

use contextdb_core::{Error as EngineError, Value, VectorIndexRef};

use crate::schema::ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX;
use crate::{CgError, ContextId, EmbeddingId, EntityId, EntityStatus, EvidenceProducer, Result};

use super::Store;
use super::convert::map_engine_err;
use super::evidence::producer_value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityReferenceEmbeddingRow {
    pub id: EmbeddingId,
    pub entity_id: EntityId,
    pub embedding_space_id: String,
    pub label: Option<String>,
    pub vector: Vec<f32>,
    pub produced_by: EvidenceProducer,
    pub created_at: crate::TxId,
}

/// One ranked open-set match: an enrolled entity whose reference vector is close
/// to the caller's probe, with the best-matching reference's label and the cosine
/// similarity score. The lean public result for consumers that already hold a
/// vector — internal search machinery stays private.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityMatch {
    pub entity_id: EntityId,
    pub label: Option<String>,
    pub score: f32,
}

/// Options for [`Store::match_entity_reference`]. `context_id` scopes the match to
/// one context (a site); `top_k` bounds the number of entities returned (default
/// 10); `min_score` drops matches below a cosine-similarity floor so an unenrolled
/// subject honestly matches nothing.
#[derive(Debug, Clone, Default)]
pub struct EntityReferenceMatchOptions {
    pub context_id: Option<ContextId>,
    pub top_k: Option<usize>,
    pub min_score: Option<f32>,
}

impl Store {
    /// Open-set entity-reference match: given a probe vector in a registered
    /// embedding space, return the enrolled entities ranked by cosine similarity,
    /// deduplicated to the best reference per entity. The caller already holds a
    /// vector — this surface accepts no text.
    pub fn match_entity_reference(
        &self,
        embedding_space_id: &str,
        query_vector: &[f32],
        options: EntityReferenceMatchOptions,
    ) -> Result<Vec<EntityMatch>> {
        let space = self
            .embedding_space_for_id(embedding_space_id)
            .map_err(|_| CgError::UnknownEmbeddingSpace {
                id: embedding_space_id.to_string(),
            })?;
        if query_vector.len() != space.dimension {
            return Err(CgError::EmbeddingDimensionMismatch {
                space_id: embedding_space_id.to_string(),
                expected: space.dimension as u32,
                found: query_vector.len() as u32,
            });
        }
        let table = format!(
            "{}{}",
            ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX, space.dimension
        );
        let top_k = options.top_k.unwrap_or(10).max(1);
        // Over-fetch reference rows so several references of one entity cannot
        // crowd other entities out of the deduplicated result. The cap bounds
        // the over-fetch, never the caller's top_k: fetch at least top_k rows,
        // at most max(256, top_k).
        let fetch_limit = top_k.saturating_mul(8).min(256.max(top_k));
        let mut params = std::collections::HashMap::from([
            ("query".to_string(), Value::Vector(query_vector.to_vec())),
            (
                "embedding_space_id".to_string(),
                Value::Text(embedding_space_id.to_string()),
            ),
        ]);
        let sql = match options.context_id {
            Some(context_id) => {
                params.insert("context_id".to_string(), Value::Uuid(context_id.as_uuid()));
                format!(
                    "SELECT entity_id, label, score FROM {table} WHERE context_id = $context_id AND embedding_space_id = $embedding_space_id ORDER BY vector <=> $query LIMIT {fetch_limit}"
                )
            }
            None => format!(
                "SELECT entity_id, label, score FROM {table} WHERE embedding_space_id = $embedding_space_id ORDER BY vector <=> $query LIMIT {fetch_limit}"
            ),
        };
        let result = self.read_query(&sql, &params)?;
        let column = |name: &str| result.columns.iter().position(|c| c == name);
        let (Some(entity_column), Some(label_column), Some(score_column)) =
            (column("entity_id"), column("label"), column("score"))
        else {
            return Err(CgError::Engine(
                "entity reference match row lookup missing a column".to_string(),
            ));
        };
        let mut scored: Vec<EntityMatch> = Vec::new();
        for row in &result.rows {
            let Some(Value::Uuid(entity_id)) = row.get(entity_column) else {
                continue;
            };
            let label = match row.get(label_column) {
                Some(Value::Text(label)) => Some(label.clone()),
                _ => None,
            };
            let score = match row.get(score_column) {
                Some(Value::Float64(score)) => *score as f32,
                _ => continue,
            };
            scored.push(EntityMatch {
                entity_id: EntityId::from(*entity_id),
                label,
                score,
            });
        }
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seen = std::collections::HashSet::new();
        let mut matches: Vec<EntityMatch> = Vec::new();
        for candidate in scored {
            if let Some(min_score) = options.min_score
                && candidate.score < min_score
            {
                continue;
            }
            if seen.insert(candidate.entity_id) {
                matches.push(candidate);
            }
            if matches.len() == top_k {
                break;
            }
        }
        Ok(matches)
    }

    pub fn enroll_entity_reference(
        &self,
        entity_id: EntityId,
        embedding_space_id: &str,
        label: Option<String>,
        vector: Vec<f32>,
        produced_by: EvidenceProducer,
    ) -> Result<EmbeddingId> {
        let entity = self.get_entity(entity_id)?;
        if entity
            .as_ref()
            .is_some_and(|entity| entity.status == EntityStatus::Archived)
        {
            return Err(CgError::EntityArchived {
                entity_id,
                archived_at: self.archived_entity_txid(entity_id)?,
            });
        }
        let context_id = match entity.as_ref() {
            Some(entity) => entity.context_id,
            None => self
                .first_context_id_for_entity_reference_enrollment()?
                .unwrap_or_else(ContextId::new_v7),
        };
        let space = self
            .embedding_space_for_id(embedding_space_id)
            .map_err(|_| CgError::UnknownEmbeddingSpace {
                id: embedding_space_id.to_string(),
            })?;
        if vector.len() != space.dimension {
            return Err(CgError::EmbeddingDimensionMismatch {
                space_id: embedding_space_id.to_string(),
                expected: space.dimension as u32,
                found: vector.len() as u32,
            });
        }
        let table = format!(
            "{}{}",
            ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX, space.dimension
        );
        let tx = self.begin_tx()?;
        let existing = self.find_entity_reference_row(&table, entity_id, label.as_deref())?;
        let (id, row_id) = if let Some((id, row_id)) = existing {
            let updated = self.db.execute_in_tx(
                tx,
                &format!(
                    "UPDATE {table} SET vector = $vector, embedding_space_id = $space, produced_by = $producer WHERE id = $id"
                ),
                &std::collections::HashMap::from([
                    ("id".to_string(), Value::Uuid(id.as_uuid())),
                    ("vector".to_string(), Value::Vector(vector.clone())),
                    ("space".to_string(), Value::Text(embedding_space_id.to_string())),
                    ("producer".to_string(), producer_value(&produced_by)),
                ]),
            );
            if let Err(err) = updated {
                let _ = self.db.rollback(tx);
                return Err(map_entity_reference_enrollment_engine_err(err, entity_id));
            }
            if let Err(err) = self.insert_entity_reference_handle_in_tx(
                tx,
                EmbeddingId::new_v7(),
                context_id,
                entity_id,
                embedding_space_id,
                &produced_by,
            ) {
                let _ = self.db.rollback(tx);
                return Err(map_entity_reference_enrollment_engine_err(err, entity_id));
            }
            (id, row_id)
        } else {
            let id = EmbeddingId::new_v7();
            let row_id = self.db.insert_row(
                tx,
                &table,
                std::collections::HashMap::from([
                    ("id".to_string(), Value::Uuid(id.as_uuid())),
                    ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                    ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
                    (
                        "label".to_string(),
                        label.clone().map(Value::Text).unwrap_or(Value::Null),
                    ),
                    ("vector".to_string(), Value::Vector(vector.clone())),
                    (
                        "embedding_space_id".to_string(),
                        Value::Text(embedding_space_id.to_string()),
                    ),
                    ("produced_by".to_string(), producer_value(&produced_by)),
                    ("created_at".to_string(), Value::TxId(tx)),
                ]),
            );
            let row_id = match row_id {
                Ok(row_id) => row_id,
                Err(err) => {
                    let _ = self.db.rollback(tx);
                    return Err(map_entity_reference_enrollment_engine_err(err, entity_id));
                }
            };
            if let Err(err) = self.insert_entity_reference_handle_in_tx(
                tx,
                id,
                context_id,
                entity_id,
                embedding_space_id,
                &produced_by,
            ) {
                let _ = self.db.rollback(tx);
                return Err(map_entity_reference_enrollment_engine_err(err, entity_id));
            }
            (id, row_id)
        };
        if let Err(err) =
            self.db
                .insert_vector(tx, VectorIndexRef::new(&table, "vector"), row_id, vector)
        {
            let _ = self.db.rollback(tx);
            return Err(map_entity_reference_enrollment_engine_err(err, entity_id));
        }
        self.db
            .commit(tx)
            .map_err(|err| map_entity_reference_enrollment_engine_err(err, entity_id))?;
        Ok(id)
    }

    fn insert_entity_reference_handle_in_tx(
        &self,
        tx: crate::TxId,
        handle_id: EmbeddingId,
        context_id: ContextId,
        entity_id: EntityId,
        embedding_space_id: &str,
        produced_by: &EvidenceProducer,
    ) -> std::result::Result<(), EngineError> {
        self.db.execute_in_tx(
            tx,
            "INSERT INTO entity_reference_embedding_handles (id, context_id, target_id, embedding_space_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $target_id, $space, $producer, $producer_system, $created_at)",
            &std::collections::HashMap::from([
                ("id".to_string(), Value::Uuid(handle_id.as_uuid())),
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                ("target_id".to_string(), Value::Uuid(entity_id.as_uuid())),
                ("space".to_string(), Value::Text(embedding_space_id.to_string())),
                ("producer".to_string(), producer_value(produced_by)),
                (
                    "producer_system".to_string(),
                    Value::Text(produced_by.system.clone()),
                ),
                ("created_at".to_string(), Value::TxId(tx)),
            ]),
        )?;
        Ok(())
    }

    /// Remove ALL of an entity's enrolled reference vectors (every space, every
    /// per-dimension table), including their vector-index entries, in one
    /// transaction. Returns the number of references removed; idempotent. This
    /// is the deletion half of enrollment: a removed entity stops matching
    /// immediately, and its biometric vectors are gone from the store. The
    /// enrollment HANDLE rows are deliberately kept — they are the immutable
    /// provenance layer (ids and producer only, no vector) recording that an
    /// enrollment happened.
    pub fn remove_entity_references(&self, entity_id: EntityId) -> Result<usize> {
        let snapshot = self.db.snapshot();
        let mut reference_rows: Vec<(String, contextdb_core::RowId)> = Vec::new();
        for table in self
            .db
            .table_names()
            .into_iter()
            .filter(|table| table.starts_with(ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX))
        {
            for row in self.db.scan(&table, snapshot).map_err(map_engine_err)? {
                if row.values.get("entity_id") == Some(&Value::Uuid(entity_id.as_uuid())) {
                    reference_rows.push((table.clone(), row.row_id));
                }
            }
        }
        if reference_rows.is_empty() {
            return Ok(0);
        }
        let tx = self.begin_tx()?;
        let rollback_on = |err: EngineError| {
            let _ = self.db.rollback(tx);
            map_engine_err(err)
        };
        for (table, row_id) in &reference_rows {
            self.db
                .delete_vector(tx, VectorIndexRef::new(table, "vector"), *row_id)
                .map_err(rollback_on)?;
            self.db
                .delete_row(tx, table, *row_id)
                .map_err(rollback_on)?;
        }
        self.db.commit(tx).map_err(map_engine_err)?;
        Ok(reference_rows.len())
    }

    pub fn list_entity_reference_embeddings_for_entity(
        &self,
        entity_id: EntityId,
    ) -> Result<Vec<EntityReferenceEmbeddingRow>> {
        let mut rows = Vec::new();
        for table in self
            .db
            .table_names()
            .into_iter()
            .filter(|table| table.starts_with(ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX))
        {
            for row in self
                .db
                .scan(&table, self.db.snapshot())
                .map_err(map_engine_err)?
            {
                if row.values.get("entity_id") != Some(&Value::Uuid(entity_id.as_uuid())) {
                    continue;
                }
                if let Some(decoded) = decode_reference_row(&row.values)? {
                    rows.push(decoded);
                }
            }
        }
        rows.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(rows)
    }

    fn find_entity_reference_row(
        &self,
        table: &str,
        entity_id: EntityId,
        label: Option<&str>,
    ) -> Result<Option<(EmbeddingId, contextdb_core::RowId)>> {
        let rows = self
            .db
            .scan(table, self.db.snapshot())
            .map_err(map_engine_err)?;
        Ok(rows.into_iter().find_map(|row| {
            if row.values.get("entity_id") != Some(&Value::Uuid(entity_id.as_uuid())) {
                return None;
            }
            let row_label = match row.values.get("label") {
                Some(Value::Text(value)) => Some(value.as_str()),
                _ => None,
            };
            if row_label != label {
                return None;
            }
            match row.values.get("id") {
                Some(Value::Uuid(id)) => Some((EmbeddingId::from(*id), row.row_id)),
                _ => None,
            }
        }))
    }

    fn first_context_id_for_entity_reference_enrollment(&self) -> Result<Option<ContextId>> {
        let result = self
            .db
            .execute(
                "SELECT id FROM contexts LIMIT 1",
                &std::collections::HashMap::new(),
            )
            .map_err(map_engine_err)?;
        Ok(result.rows.first().and_then(|row| match row.first() {
            Some(Value::Uuid(id)) => Some(ContextId::from(*id)),
            _ => None,
        }))
    }

    fn archived_entity_txid(&self, entity_id: EntityId) -> Result<crate::TxId> {
        let graph = self.entity_history_graph(entity_id)?;
        let archived_heads = graph.archived_heads;
        graph
            .snapshots
            .into_iter()
            .filter(|snapshot| archived_heads.contains(&snapshot.id))
            .map(|snapshot| snapshot.valid_from)
            .max_by_key(|txid| txid.0)
            .ok_or(CgError::EntityNotFound { entity_id })
    }
}

fn map_entity_reference_enrollment_engine_err(err: EngineError, entity_id: EntityId) -> CgError {
    match err {
        EngineError::ForeignKeyViolation { .. } => CgError::EntityNotFound { entity_id },
        other => map_engine_err(other),
    }
}

#[cfg(context_graph_internal_test_support)]
pub fn entity_reference_cosine_match(
    store: &Store,
    live_space: &str,
    reference_embedding_id: EmbeddingId,
) -> Result<()> {
    for table in store
        .db
        .table_names()
        .into_iter()
        .filter(|table| table.starts_with(ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX))
    {
        for row in store
            .db
            .scan(&table, store.db.snapshot())
            .map_err(map_engine_err)?
        {
            if row.values.get("id") == Some(&Value::Uuid(reference_embedding_id.as_uuid()))
                && let Some(Value::Text(reference_space)) = row.values.get("embedding_space_id")
            {
                if reference_space != live_space {
                    return Err(CgError::CrossSpaceCosineRefused {
                        live_space: live_space.to_string(),
                        reference_space: reference_space.clone(),
                    });
                }
                return Ok(());
            }
        }
    }
    Err(CgError::NotFound {
        id: Some(reference_embedding_id.to_string()),
        missing: Some("entity_reference_embedding"),
    })
}

fn decode_reference_row(
    values: &std::collections::HashMap<String, Value>,
) -> Result<Option<EntityReferenceEmbeddingRow>> {
    let (
        Some(Value::Uuid(id)),
        Some(Value::Uuid(entity_id)),
        Some(Value::Text(space)),
        Some(Value::Vector(vector)),
        Some(Value::Json(producer)),
        Some(Value::TxId(created_at)),
    ) = (
        values.get("id"),
        values.get("entity_id"),
        values.get("embedding_space_id"),
        values.get("vector"),
        values.get("produced_by"),
        values.get("created_at"),
    )
    else {
        return Ok(None);
    };
    let produced_by = serde_json::from_value(producer.clone())
        .map_err(|err| CgError::Engine(format!("invalid entity reference producer json: {err}")))?;
    let label = match values.get("label") {
        Some(Value::Text(label)) => Some(label.clone()),
        _ => None,
    };
    Ok(Some(EntityReferenceEmbeddingRow {
        id: EmbeddingId::from(*id),
        entity_id: EntityId::from(*entity_id),
        embedding_space_id: space.clone(),
        label,
        vector: vector.clone(),
        produced_by,
        created_at: *created_at,
    }))
}
