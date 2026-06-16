use std::collections::HashMap;

use contextdb_core::{Error as EngineError, Value, VectorIndexRef};

use crate::embedder::EmbeddingInput;
use crate::embedding_space::BuiltinSpaceRegistry;
use crate::error::CgError;
use crate::types::*;
use crate::{Result, TxId};

use super::Store;
use super::convert::map_engine_err;
use super::encode::{
    get_json, get_optional_text, get_text, get_txid, get_uuid, get_vector, properties_from_json,
    properties_to_json,
};
use super::evidence::{
    embedding_target_uuid, producer_value, validate_precomputed_embedding, validate_space_kind,
    vector_binding_for_target,
};
use super::evidence_kind_str;
use super::observation::state_delta_from_json;

impl Store {
    pub fn create_digest(&self, input: CreateDigest) -> Result<Digest> {
        if input.kind != EvidenceKind::Text {
            return Err(CgError::UnsupportedEvidenceKind { kind: input.kind });
        }
        validate_precomputed_embedding(&input.embedding, false)?;
        if input.embedding.target != EmbeddingTarget::Digest(input.id) {
            return Err(CgError::EmbeddingSpaceConfigMismatch {
                field: "target",
                expected: format!("Digest({})", input.id),
                actual: format!("{:?}", input.embedding.target),
            });
        }
        let space = self
            .embedding_space_for_id(&input.embedding.embedding_space_id)
            .map_err(|_| CgError::UnknownEmbeddingSpace {
                id: input.embedding.embedding_space_id.clone(),
            })?;
        if input.embedding.vector.len() != space.dimension {
            return Err(CgError::EmbeddingDimensionMismatch {
                space_id: input.embedding.embedding_space_id.clone(),
                expected: space.dimension as u32,
                found: input.embedding.vector.len() as u32,
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

        let tx = self.begin_tx()?;
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
        match self.db.execute_in_tx(
            tx,
            "INSERT INTO contexts (id, name, labels, properties, created_at) VALUES ($id, $name, $labels, $properties, $created_at)",
            &context_params,
        ) {
            Ok(_) => {}
            Err(EngineError::UniqueViolation { table, column })
                if table == "contexts"
                    && column == "id"
                    && digest_context_id_exists(self, input.context_id)? => {}
            Err(err) => {
                let _ = self.db.rollback(tx);
                return Err(map_engine_err(err));
            }
        }
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
                "target_id".to_string(),
                Value::Uuid(embedding_target_uuid(&input.embedding.target)),
            ),
            (
                "produced_by".to_string(),
                producer_value(&input.embedding.produced_by),
            ),
            (
                "producer_system".to_string(),
                Value::Text(input.embedding.produced_by.system.clone()),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        if let Err(e) = self.db.execute_in_tx(
            tx,
            "INSERT INTO digest_embedding_handles (id, context_id, embedding_space_id, target_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $embedding_space_id, $target_id, $produced_by, $producer_system, $created_at)",
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

    pub(crate) fn digest_source_ids_for_context(
        &self,
        context_id: ContextId,
    ) -> Result<Vec<uuid::Uuid>> {
        let result = self
            .db
            .execute(
                "SELECT id FROM digests WHERE context_id = $context_id",
                &HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        let mut source_ids = Vec::new();
        for row in &result.rows {
            let digest_id = DigestId::from(get_uuid(row, &result.columns, "id")?);
            source_ids.extend(self.extracted_from_source_ids(context_id, digest_id)?.0);
        }
        source_ids.sort();
        source_ids.dedup();
        Ok(source_ids)
    }

    pub fn admin_overwrite_digest(&self, _id: DigestId, _summary: String) -> Result<()> {
        Err(CgError::ImmutableEntity { kind: "digest" })
    }

    pub fn find_via_digests(
        &self,
        query: &str,
        filter: DigestSearchFilter,
    ) -> Result<Vec<DigestSearchHit>> {
        if query.trim().is_empty() {
            self.set_trace("digests.vector_text empty-query");
            return Ok(Vec::new());
        }
        let query_vector = self.digest_query_vector(query)?;
        let contexts = if let Some(cross) = filter.cross_context.clone() {
            Some(cross)
        } else {
            filter.context_id.map(|id| vec![id])
        };
        let context_labels = self
            .list_contexts()?
            .into_iter()
            .map(|context| (context.id, context.name))
            .collect::<HashMap<_, _>>();
        let mut params = HashMap::from([("query".to_string(), Value::Vector(query_vector))]);
        let mut where_parts = Vec::new();
        if let Some(contexts) = contexts.as_ref() {
            if contexts.is_empty() {
                return Ok(Vec::new());
            }
            let mut parts = Vec::new();
            for (idx, context) in contexts.iter().enumerate() {
                let key = format!("context_{idx}");
                params.insert(key.clone(), Value::Uuid(context.as_uuid()));
                parts.push(format!("context_id = ${key}"));
            }
            where_parts.push(format!("({})", parts.join(" OR ")));
        }
        if let Some(start) = filter.created_after {
            params.insert("created_after".to_string(), Value::TxId(start));
            where_parts.push("created_at >= $created_after".to_string());
        }
        if let Some(end) = filter.created_before {
            params.insert("created_before".to_string(), Value::TxId(end));
            where_parts.push("created_at <= $created_before".to_string());
        }
        let mut sql = "SELECT id, context_id, summary, created_at, score FROM digests".to_string();
        if !where_parts.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&where_parts.join(" AND "));
        }
        sql.push_str(" ORDER BY vector_text <=> $query LIMIT 10000");
        let result = self.db.execute(&sql, &params).map_err(map_engine_err)?;
        let vector_trace = format!("{:?}", result.trace);
        let mut hits = Vec::new();
        for row in result.rows {
            let digest_id = DigestId::from(value_uuid(&row, &result.columns, "id")?);
            let context_id = ContextId::from(value_uuid(&row, &result.columns, "context_id")?);
            let similarity = value_score(&row, &result.columns, "score")?;
            if similarity <= 0.0 {
                continue;
            }
            if filter
                .min_similarity
                .is_some_and(|threshold| similarity < threshold)
            {
                continue;
            }
            let summary = value_text(&row, &result.columns, "summary")?;
            let created_at = value_txid(&row, &result.columns, "created_at")?;
            let (edge_sources, graph_trace) =
                self.extracted_from_source_ids(context_id, digest_id)?;
            let mut hit = DigestSearchHit {
                digest_id,
                context_id,
                context_label: context_labels
                    .get(&context_id)
                    .cloned()
                    .unwrap_or_else(|| format!("context-{context_id}")),
                created_at,
                similarity,
                summary,
                intention_ids: Vec::new(),
                intention_texts: Vec::new(),
                decision_ids: Vec::new(),
                decision_statuses: Vec::new(),
                decision_texts: Vec::new(),
                observation_ids: Vec::new(),
                observation_texts: Vec::new(),
                extracted_from_edges: Vec::new(),
                graph_walk: DigestGraphWalkBoundary {
                    target_digest_id: digest_id,
                    source_ids: Vec::new(),
                    relationship: "EXTRACTED_FROM".to_string(),
                    engine_plan: graph_trace,
                },
            };
            let mut included_source_ids = Vec::new();
            let mut saw_decision_source = false;
            let mut included_decision_source = false;
            for source_id in edge_sources {
                if let Some((id, text)) = self.intention_text(source_id)? {
                    hit.intention_ids.push(id);
                    hit.intention_texts.push(text);
                    included_source_ids.push(source_id);
                    continue;
                }
                if let Some((id, status, text)) = self.decision_text(source_id)? {
                    saw_decision_source = true;
                    if filter.include_inactive || status == DecisionStatus::Active {
                        hit.decision_ids.push(id);
                        hit.decision_statuses.push(status);
                        hit.decision_texts.push(text);
                        included_source_ids.push(source_id);
                        included_decision_source = true;
                    }
                    continue;
                }
                if let Some((id, text)) = self.observation_text(source_id)? {
                    hit.observation_ids.push(id);
                    hit.observation_texts.push(text);
                    included_source_ids.push(source_id);
                    continue;
                }
                if self.entity_name(source_id)?.is_some() {
                    included_source_ids.push(source_id);
                }
            }
            if !filter.include_inactive && saw_decision_source && !included_decision_source {
                continue;
            }
            included_source_ids.sort();
            hit.extracted_from_edges = included_source_ids
                .iter()
                .map(|source_id| DigestExtractedFromEdge {
                    source_id: *source_id,
                    target_digest_id: digest_id,
                    relationship: "EXTRACTED_FROM".to_string(),
                })
                .collect();
            hit.graph_walk.source_ids = included_source_ids;
            hits.push(hit);
        }
        if let Some(limit) = filter.limit {
            hits.truncate(limit);
        }
        self.set_trace(format!("digests.vector_text {vector_trace}"));
        Ok(hits)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DigestSearchFilter {
    pub context_id: Option<ContextId>,
    pub cross_context: Option<Vec<ContextId>>,
    pub min_similarity: Option<f32>,
    pub limit: Option<usize>,
    pub created_after: Option<TxId>,
    pub created_before: Option<TxId>,
    pub include_inactive: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DigestSearchHit {
    pub digest_id: DigestId,
    pub context_id: ContextId,
    pub context_label: String,
    pub created_at: TxId,
    pub similarity: f32,
    pub summary: String,
    pub intention_ids: Vec<IntentionId>,
    pub intention_texts: Vec<String>,
    pub decision_ids: Vec<DecisionId>,
    pub decision_statuses: Vec<DecisionStatus>,
    pub decision_texts: Vec<String>,
    pub observation_ids: Vec<ObservationId>,
    pub observation_texts: Vec<String>,
    pub extracted_from_edges: Vec<DigestExtractedFromEdge>,
    pub graph_walk: DigestGraphWalkBoundary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestExtractedFromEdge {
    pub source_id: uuid::Uuid,
    pub target_digest_id: DigestId,
    pub relationship: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestGraphWalkBoundary {
    pub target_digest_id: DigestId,
    pub source_ids: Vec<uuid::Uuid>,
    pub relationship: String,
    pub engine_plan: String,
}

impl Store {
    pub(crate) fn embed_digest_text(&self, text: &str) -> Result<Vec<f32>> {
        let space_id = BuiltinSpaceRegistry::TEXT_DEFAULT_ID;
        if let Some(embedder) = self.embedders.get(space_id) {
            return Ok(embedder
                .embed(EmbeddingInput::Text(text.to_string()))?
                .vector);
        }
        Err(CgError::TextEmbedderDisabled)
    }

    fn digest_query_vector(&self, query: &str) -> Result<Vec<f32>> {
        self.embed_digest_text(query)
    }

    fn extracted_from_source_ids(
        &self,
        context_id: ContextId,
        digest_id: DigestId,
    ) -> Result<(Vec<uuid::Uuid>, String)> {
        let params = HashMap::from([
            ("digest_id".to_string(), Value::Uuid(digest_id.as_uuid())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
        ]);
        let result = self
            .db
            .execute(
                "SELECT source_id FROM GRAPH_TABLE(
                    edges MATCH (d)<-[:EXTRACTED_FROM]-(s)
                    WHERE d.id = $digest_id
                    COLUMNS (s.id AS source_id)
                )
                WHERE source_id IN (SELECT source_id FROM edges WHERE context_id = $context_id)",
                &params,
            )
            .map_err(map_engine_err)?;
        let graph_trace = format!("{:?}", result.trace);
        let mut graph_ids = result
            .rows
            .iter()
            .filter_map(|row| match row.first() {
                Some(Value::Uuid(id)) => Some(*id),
                _ => None,
            })
            .collect::<Vec<_>>();
        graph_ids.sort();
        Ok((graph_ids, graph_trace))
    }

    fn intention_text(&self, id: uuid::Uuid) -> Result<Option<(IntentionId, String)>> {
        let result = self
            .db
            .execute(
                "SELECT id, description FROM intentions WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(id))]),
            )
            .map_err(map_engine_err)?;
        Ok(result.rows.first().and_then(|row| {
            Some((
                IntentionId::from(value_uuid(row, &result.columns, "id").ok()?),
                value_text(row, &result.columns, "description").ok()?,
            ))
        }))
    }

    fn decision_text(
        &self,
        id: uuid::Uuid,
    ) -> Result<Option<(DecisionId, DecisionStatus, String)>> {
        let result = self
            .db
            .execute(
                "SELECT id, description, status FROM decisions WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(id))]),
            )
            .map_err(map_engine_err)?;
        Ok(result.rows.first().and_then(|row| {
            let status = match value_text(row, &result.columns, "status").ok()?.as_str() {
                "Active" | "active" => DecisionStatus::Active,
                "Superseded" | "superseded" => DecisionStatus::Superseded,
                "Archived" | "archived" => DecisionStatus::Archived,
                "Invalidated" | "invalidated" => DecisionStatus::Invalidated,
                _ => return None,
            };
            Some((
                DecisionId::from(value_uuid(row, &result.columns, "id").ok()?),
                status,
                value_text(row, &result.columns, "description").ok()?,
            ))
        }))
    }

    fn observation_text(&self, id: uuid::Uuid) -> Result<Option<(ObservationId, String)>> {
        let result = self
            .db
            .execute(
                "SELECT id, observation_type, state_delta, entity_id FROM observations WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(id))]),
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| {
                let id = ObservationId::from(value_uuid(row, &result.columns, "id")?);
                let observation_type = value_text(row, &result.columns, "observation_type")?;
                let entity_id = value_uuid(row, &result.columns, "entity_id")?;
                let entity_name = self
                    .entity_name(entity_id)?
                    .unwrap_or_else(|| "entity".to_string());
                let state_delta =
                    state_delta_from_json(get_json(row, &result.columns, "state_delta")?)?;
                let text = state_delta
                    .get("path")
                    .and_then(|value| value.as_ref())
                    .and_then(|value| value.as_str())
                    .map(|path| format!("{entity_name} observed at {path}"))
                    .unwrap_or(observation_type);
                Ok((id, text))
            })
            .transpose()
    }

    fn entity_name(&self, id: uuid::Uuid) -> Result<Option<String>> {
        let result = self
            .db
            .execute(
                "SELECT name FROM entities WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(id))]),
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| value_text(row, &result.columns, "name"))
            .transpose()
    }
}

fn value_index(columns: &[String], name: &str) -> Result<usize> {
    columns
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| CgError::Engine(format!("missing column {name}")))
}

fn value_uuid(row: &[Value], columns: &[String], name: &str) -> Result<uuid::Uuid> {
    match row.get(value_index(columns, name)?) {
        Some(Value::Uuid(id)) => Ok(*id),
        _ => Err(CgError::Engine(format!("column {name} was not uuid"))),
    }
}

fn value_text(row: &[Value], columns: &[String], name: &str) -> Result<String> {
    match row.get(value_index(columns, name)?) {
        Some(Value::Text(value)) => Ok(value.clone()),
        _ => Err(CgError::Engine(format!("column {name} was not text"))),
    }
}

fn value_txid(row: &[Value], columns: &[String], name: &str) -> Result<TxId> {
    match row.get(value_index(columns, name)?) {
        Some(Value::TxId(value)) => Ok(*value),
        _ => Err(CgError::Engine(format!("column {name} was not txid"))),
    }
}

fn value_score(row: &[Value], columns: &[String], name: &str) -> Result<f32> {
    match row.get(value_index(columns, name)?) {
        Some(Value::Float64(value)) => Ok(*value as f32),
        _ => Err(CgError::Engine(format!("column {name} was not score"))),
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

fn digest_context_id_exists(store: &Store, context_id: ContextId) -> Result<bool> {
    let result = store
        .db
        .execute(
            "SELECT id FROM contexts WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    Ok(!result.rows.is_empty())
}
