use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, Utc};
use contextdb_core::{Error as EngineError, RowId, Value, VectorIndexRef};
use serde_json::Value as JsonValue;

use crate::Result;
use crate::embedder::EmbeddingInput;
use crate::error::{CgError, EngineError as CgEngineError};
use crate::messaging::OverrideFailureReason;
use crate::schema::ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX;
use crate::types::*;

use super::Store;
use super::convert::{entity_type_from_str, entity_type_str, map_engine_err};
use super::encode::{
    get_json, get_optional_float, get_text, get_timestamp, get_txid, get_uuid,
    properties_from_json, properties_to_json,
};
use super::evidence::{
    embedding_target_uuid, list_evidence_for_observation, producer_value, properties_value,
    validate_precomputed_embedding, validate_space_kind, value_for_optional_evidence_id,
    value_for_optional_json, value_for_optional_pair, value_for_optional_text,
    value_for_optional_timestamp, value_for_optional_timestamp_nanos, value_for_optional_u64,
    vector_binding_for_target,
};
use super::graph_rows::GraphRows;
use super::impact_analysis::decision_invalidation_context;
use super::{evidence_kind_str, retention_status_str};

const ENTITY_REFERENCE_MATCH_THRESHOLD: f32 = 0.8;

trait RecordObservationHooks {
    fn after_state_delta_guards(&mut self) {}

    fn conditional_update_conflict(&mut self) {}
}

struct NoopRecordObservationHooks;

impl RecordObservationHooks for NoopRecordObservationHooks {}

impl Store {
    pub fn record_observation(&self, input: RecordObservation) -> Result<RecordObservationResult> {
        self.record_observation_with_hooks(input, &mut NoopRecordObservationHooks)
    }

    // The hook is private and the public entry point always supplies the no-op implementation.
    // It gives the unit test one exact point after strict snapshot guards are staged and before
    // the no-op decision, without changing the production operation.
    fn record_observation_with_hooks<H: RecordObservationHooks>(
        &self,
        mut input: RecordObservation,
        hooks: &mut H,
    ) -> Result<RecordObservationResult> {
        for evidence in &mut input.evidence {
            evidence.observation_id = input.id;
            evidence.context_id = input.context_id;
        }
        self.validate_record_observation_input(&input)?;
        let caller_supplied_embeddings = !input.embeddings.is_empty();
        self.generate_default_embeddings(&mut input)?;
        let embedding_routes =
            self.validate_embeddings_for_observation(&input, !caller_supplied_embeddings)?;
        let evidence_vectors = evidence_vectors_for_insert(&embedding_routes)?;
        let evidence_ids = input
            .evidence
            .iter()
            .map(|evidence| evidence.id)
            .collect::<HashSet<_>>();
        let auto_create_entity = if let Some(entity) = self.get_entity(input.entity_id)? {
            if entity.context_id != input.context_id {
                return Err(CgError::CrossContextObservation);
            }
            false
        } else {
            true
        };
        let id = input.id;
        let observed_at = input.observed_at;
        let observed_properties_json = properties_to_json(&input.observed_properties);
        let state_delta_json = state_delta_to_json(&input.state_delta);
        let properties_json = properties_to_json(&input.properties);
        for _ in 0..16 {
            let tx = self.begin_tx()?;
            if auto_create_entity
                && let Err(e) =
                    self.ensure_context_and_entity_in_tx(tx, input.context_id, input.entity_id)
            {
                let _ = self.db.rollback(tx);
                return Err(e);
            }
            let params: HashMap<String, Value> = [
                ("id".to_string(), Value::Uuid(id.as_uuid())),
                (
                    "observation_type".to_string(),
                    Value::Text(input.observation_type.clone()),
                ),
                (
                    "observed_properties".to_string(),
                    Value::Json(observed_properties_json.clone()),
                ),
                (
                    "state_delta".to_string(),
                    Value::Json(state_delta_json.clone()),
                ),
                (
                    "properties".to_string(),
                    Value::Json(properties_json.clone()),
                ),
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
            for (supply_order, evidence) in input.evidence.iter().enumerate() {
                if let Err(err) = self.insert_evidence_in_tx(
                    tx,
                    evidence,
                    supply_order as i64,
                    evidence_vectors.get(&evidence.id),
                ) {
                    let _ = self.db.rollback(tx);
                    return Err(err);
                }
            }
            for route in &embedding_routes {
                if let EmbeddingTarget::Evidence(id) = route.target {
                    if let Err(err) = self.insert_embedding_handle_in_tx(tx, route) {
                        let _ = self.db.rollback(tx);
                        return Err(self.map_embedding_route_engine_err(err, route));
                    }
                    if !evidence_ids.contains(&id)
                        && self.evidence_exists_in_context(id, input.context_id)?
                    {
                        let _ = self.db.rollback(tx);
                        return Err(CgError::EmbeddingTargetNotFound { id });
                    }
                }
            }
            let snapshot_id = match self.merge_state_delta_in_tx(tx, &input, hooks) {
                Ok(snapshot_id) => snapshot_id,
                Err(StateDeltaMergeError::Retry) => {
                    let _ = self.db.rollback(tx);
                    std::thread::yield_now();
                    continue;
                }
                Err(StateDeltaMergeError::Error(err)) => {
                    let _ = self.db.rollback(tx);
                    return Err(err);
                }
            };
            for route in &embedding_routes {
                match route.target {
                    EmbeddingTarget::Evidence(_) => {}
                    EmbeddingTarget::EntityReference(_, _) => {
                        let row_id =
                            match self.insert_entity_reference_embedding_row_in_tx(tx, route) {
                                Ok(row_id) => row_id,
                                Err(err) => {
                                    let _ = self.db.rollback(tx);
                                    return Err(self.map_embedding_route_engine_err(err, route));
                                }
                            };
                        if let Err(err) = self.insert_embedding_handle_in_tx(tx, route) {
                            let _ = self.db.rollback(tx);
                            return Err(self.map_embedding_route_engine_err(err, route));
                        }
                        if let Err(err) = self.insert_embedding_vector_in_tx(tx, route, row_id) {
                            let _ = self.db.rollback(tx);
                            return Err(err);
                        }
                    }
                    EmbeddingTarget::Digest(id) => {
                        if let Err(err) = self.insert_embedding_handle_in_tx(tx, route) {
                            let _ = self.db.rollback(tx);
                            return Err(self.map_embedding_route_engine_err(err, route));
                        }
                        let row_id = match self.digest_row_id(id, input.context_id) {
                            Ok(Some(row_id)) => row_id,
                            Ok(None) => match self.digest_context(id) {
                                Ok(Some(target_context)) if target_context == input.context_id => {
                                    let _ = self.db.rollback(tx);
                                    return Err(CgError::EmbeddingTargetNotFound {
                                        id: EvidenceId::default(),
                                    });
                                }
                                Ok(_) => continue,
                                Err(err) => {
                                    let _ = self.db.rollback(tx);
                                    return Err(err);
                                }
                            },
                            Err(err) => {
                                let _ = self.db.rollback(tx);
                                return Err(err);
                            }
                        };
                        if let Err(err) = self.insert_embedding_vector_in_tx(tx, route, row_id) {
                            let _ = self.db.rollback(tx);
                            return Err(err);
                        }
                    }
                    EmbeddingTarget::EventSearchPassage(passage_id) => {
                        let _ = self.db.rollback(tx);
                        return Err(CgError::EventSearchPassageTargetRequiresProjection {
                            passage_id,
                        });
                    }
                }
            }
            if let Err(err) = self.insert_entity_reference_no_match_invalidations_in_tx(
                tx,
                &input,
                &embedding_routes,
            ) {
                let _ = self.db.rollback(tx);
                return Err(err);
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
            if let Err(err) = self.db.commit(tx) {
                match err {
                    EngineError::ConditionalUpdateConflict { .. } => {
                        hooks.conditional_update_conflict();
                        std::thread::yield_now();
                        continue;
                    }
                    EngineError::UniqueViolation { table, column }
                        if record_observation_snapshot_retry_unique(&table, &column) =>
                    {
                        std::thread::yield_now();
                        continue;
                    }
                    EngineError::UniqueViolation { table, column }
                        if auto_create_entity
                            && record_observation_auto_create_retry_unique(&table, &column) =>
                    {
                        std::thread::yield_now();
                        continue;
                    }
                    EngineError::UniqueViolation { table, column }
                        if table == "evidence" && column == "id" =>
                    {
                        return Err(self.duplicate_evidence_commit_error(&input)?);
                    }
                    EngineError::UniqueViolation { table, column } => {
                        if is_embedding_handle_table(&table) {
                            return Err(CgError::EngineFault {
                                source: CgEngineError::UniqueViolation { table, column },
                            });
                        }
                        return Err(map_engine_err(EngineError::UniqueViolation {
                            table,
                            column,
                        }));
                    }
                    err @ EngineError::ForeignKeyViolation { .. } => {
                        return Err(self.map_embedding_commit_fk_err(err, &embedding_routes));
                    }
                    err => return Err(map_engine_err(err)),
                }
            }

            let invalidations = self.invalidations_for_observation(input.context_id, id)?;
            std::thread::yield_now();
            return Ok(RecordObservationResult {
                observation_id: id,
                evidence_ids: input.evidence.iter().map(|evidence| evidence.id).collect(),
                embedding_ids: input
                    .embeddings
                    .iter()
                    .map(|embedding| embedding.id)
                    .collect(),
                snapshot_id,
                invalidations,
            });
        }

        Err(CgError::Engine(
            "record_observation snapshot contention retry exhausted".to_string(),
        ))
    }

    pub fn get_observation(&self, id: ObservationId) -> Result<Observation> {
        if self.read_observer.is_some() {
            self.observe_read_target("get_observation", Some(id.to_string()));
        }
        get_observation(self.as_graph_rows(), id)
    }

    pub fn list_observations(&self, context_id: Option<ContextId>) -> Result<Vec<Observation>> {
        self.observe_read("list_observations");
        list_observations(self.as_graph_rows(), context_id)
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

    /// Count the NON-invalidating Observations on an entity: those whose `state_delta` carries no
    /// state change (so they record no snapshot and cannot invalidate a decision). Each such
    /// observation is a "survived scrutiny" signal feeding the evidence-derived confidence of a
    /// decision based on this entity.
    pub(crate) fn non_invalidating_observation_count(&self, entity_id: EntityId) -> Result<usize> {
        let params: HashMap<String, Value> =
            [("entity_id".to_string(), Value::Uuid(entity_id.as_uuid()))]
                .into_iter()
                .collect();
        let result = self
            .db
            .execute(
                "SELECT state_delta FROM observations WHERE entity_id = $entity_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut count = 0usize;
        for row in &result.rows {
            let is_non_invalidating = match row.first() {
                Some(Value::Json(json)) => state_delta_has_no_ops(json),
                Some(Value::Text(text)) => serde_json::from_str::<JsonValue>(text)
                    .map(|json| state_delta_has_no_ops(&json))
                    .unwrap_or(false),
                Some(Value::Null) | None => true,
                _ => false,
            };
            if is_non_invalidating {
                count += 1;
            }
        }
        Ok(count)
    }

    fn ensure_context_and_entity_in_tx(
        &self,
        tx: crate::TxId,
        context_id: ContextId,
        entity_id: EntityId,
    ) -> Result<()> {
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
        match self.db.execute_in_tx(
            tx,
            "INSERT INTO contexts (id, name, labels, properties, created_at) VALUES ($id, $name, $labels, $properties, $created_at)",
            &context_params,
        ) {
            Ok(_) => {}
            Err(EngineError::UniqueViolation { .. }) if self.context_id_exists(context_id)? => {}
            Err(err) => return Err(map_engine_err(err)),
        }
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
        match self.db.execute_in_tx(
            tx,
            "INSERT INTO entities (id, entity_type, name, status, context_id, properties, tags, created_at) VALUES ($id, $entity_type, $name, $status, $context_id, $properties, $tags, $created_at)",
            &entity_params,
        ) {
            Ok(_) => {}
            Err(EngineError::UniqueViolation { .. })
                if self.entity_context(entity_id)? == Some(context_id) => {}
            Err(err) => return Err(map_engine_err(err)),
        }
        Ok(())
    }

    fn context_id_exists(&self, context_id: ContextId) -> Result<bool> {
        let result = self
            .db
            .execute(
                "SELECT id FROM contexts WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(context_id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        Ok(!result.rows.is_empty())
    }

    fn validate_record_observation_input(&self, input: &RecordObservation) -> Result<()> {
        if !input.id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "ObservationId",
            });
        }
        if input.evidence.is_empty() {
            return Err(CgError::MissingEvidenceRef);
        }
        if contains_raw_media_payload(&input.properties)
            || contains_raw_media_payload(&input.observed_properties)
            || contains_raw_media_state_delta(&input.state_delta)
        {
            return Err(CgError::RawMediaNotStored);
        }
        let mut ids = HashSet::new();
        for evidence in &input.evidence {
            if !ids.insert(evidence.id) {
                return Err(CgError::DuplicateEvidenceId { id: evidence.id });
            }
            validate_source_ref(
                &evidence.source_ref,
                &self.https_allowed_hosts,
                self.file_uri_allowed,
            )?;
            validate_content_hash(evidence.content_hash.as_deref())?;
            validate_mime(evidence.kind, evidence.mime_type.as_deref())?;
            validate_region(evidence.region.as_ref())?;
            if contains_raw_media_payload(&evidence.properties) {
                return Err(CgError::RawMediaNotStored);
            }
            if self.evidence_id_exists(evidence.id)? {
                return Err(CgError::DuplicateEvidenceId { id: evidence.id });
            }
        }
        self.validate_derived_from(input)
    }

    fn validate_derived_from(&self, input: &RecordObservation) -> Result<()> {
        let by_id = input
            .evidence
            .iter()
            .map(|evidence| (evidence.id, evidence.derived_from))
            .collect::<HashMap<_, _>>();
        for evidence in &input.evidence {
            let Some(parent) = evidence.derived_from else {
                continue;
            };
            if parent == evidence.id {
                return Err(CgError::InvalidDerivedFromReference);
            }
            if !by_id.contains_key(&parent)
                && !self.evidence_exists_in_context(parent, input.context_id)?
            {
                return Err(CgError::InvalidDerivedFromReference);
            }
            let mut seen = HashSet::new();
            let mut cursor = evidence.id;
            while let Some(Some(parent)) = by_id.get(&cursor) {
                if !seen.insert(cursor) || *parent == evidence.id {
                    return Err(CgError::InvalidDerivedFromReference);
                }
                if !by_id.contains_key(parent) {
                    break;
                }
                cursor = *parent;
            }
        }
        Ok(())
    }

    fn validate_embeddings_for_observation(
        &self,
        input: &RecordObservation,
        allow_reserved_producer: bool,
    ) -> Result<Vec<EmbeddingRoute>> {
        let evidence_by_id = input
            .evidence
            .iter()
            .map(|evidence| (evidence.id, evidence))
            .collect::<HashMap<_, _>>();
        let mut routes = Vec::with_capacity(input.embeddings.len());
        for embedding in &input.embeddings {
            validate_precomputed_embedding(embedding, allow_reserved_producer)?;
            routes.push(self.validate_embedding_route(input, embedding, &evidence_by_id)?);
        }
        Ok(routes)
    }

    fn validate_embedding_route(
        &self,
        input: &RecordObservation,
        embedding: &PrecomputedEmbedding,
        evidence_by_id: &HashMap<EvidenceId, &EvidenceRef>,
    ) -> Result<EmbeddingRoute> {
        match embedding.target {
            EmbeddingTarget::Evidence(target_id) => {
                let space = self
                    .embedding_space_for_id(&embedding.embedding_space_id)
                    .map_err(|_| CgError::UnknownEmbeddingSpace {
                        id: embedding.embedding_space_id.clone(),
                    })?;
                if embedding.vector.len() != space.dimension {
                    return Err(CgError::EmbeddingDimensionMismatch {
                        space_id: embedding.embedding_space_id.clone(),
                        expected: space.dimension as u32,
                        found: embedding.vector.len() as u32,
                    });
                }
                if let Some(evidence) = evidence_by_id.get(&target_id)
                    && !(space.supported_evidence_kinds.contains(&evidence.kind)
                        || (evidence.text_projection.is_some()
                            && space.supported_evidence_kinds.contains(&EvidenceKind::Text)))
                {
                    validate_space_kind(&space, evidence.kind)?;
                }
                let binding = vector_binding_for_target(&space, "evidence")?;
                Ok(EmbeddingRoute {
                    id: embedding.id,
                    context_id: input.context_id,
                    embedding_space_id: embedding.embedding_space_id.clone(),
                    vector: embedding.vector.clone(),
                    target: embedding.target.clone(),
                    produced_by: embedding.produced_by.clone(),
                    table_name: binding.table_name,
                    vector_column: binding.vector_column,
                })
            }
            EmbeddingTarget::EntityReference(_, _) => {
                let space = self
                    .embedding_space_for_id(&embedding.embedding_space_id)
                    .map_err(|_| CgError::UnknownEmbeddingSpace {
                        id: embedding.embedding_space_id.clone(),
                    })?;
                if embedding.vector.len() != space.dimension {
                    return Err(CgError::EmbeddingDimensionMismatch {
                        space_id: embedding.embedding_space_id.clone(),
                        expected: space.dimension as u32,
                        found: embedding.vector.len() as u32,
                    });
                }
                Ok(EmbeddingRoute {
                    id: embedding.id,
                    context_id: input.context_id,
                    embedding_space_id: embedding.embedding_space_id.clone(),
                    vector: embedding.vector.clone(),
                    target: embedding.target.clone(),
                    produced_by: embedding.produced_by.clone(),
                    table_name: format!(
                        "{}{}",
                        ENTITY_REFERENCE_EMBEDDING_TABLE_PREFIX, space.dimension
                    ),
                    vector_column: "vector".to_string(),
                })
            }
            EmbeddingTarget::Digest(_) => {
                let space = self
                    .embedding_space_for_id(&embedding.embedding_space_id)
                    .map_err(|_| CgError::UnknownEmbeddingSpace {
                        id: embedding.embedding_space_id.clone(),
                    })?;
                if embedding.vector.len() != space.dimension {
                    return Err(CgError::EmbeddingDimensionMismatch {
                        space_id: embedding.embedding_space_id.clone(),
                        expected: space.dimension as u32,
                        found: embedding.vector.len() as u32,
                    });
                }
                validate_space_kind(&space, EvidenceKind::Text)?;
                let binding = vector_binding_for_target(&space, "digest")?;
                Ok(EmbeddingRoute {
                    id: embedding.id,
                    context_id: input.context_id,
                    embedding_space_id: embedding.embedding_space_id.clone(),
                    vector: embedding.vector.clone(),
                    target: embedding.target.clone(),
                    produced_by: embedding.produced_by.clone(),
                    table_name: binding.table_name,
                    vector_column: binding.vector_column,
                })
            }
            EmbeddingTarget::EventSearchPassage(passage_id) => {
                Err(CgError::EventSearchPassageTargetRequiresProjection { passage_id })
            }
        }
    }

    fn generate_default_embeddings(&self, input: &mut RecordObservation) -> Result<()> {
        if !input.embeddings.is_empty() {
            return Ok(());
        }
        for evidence in &input.evidence {
            let text_projection = match evidence.kind {
                EvidenceKind::Text => match evidence.text_projection.clone() {
                    Some(text) => Some(text),
                    None => continue,
                },
                _ => None,
            };
            let Some(space_id) = self.default_embedding_spaces.get(&evidence.kind) else {
                continue;
            };
            let Some(embedder) = self.embedders.get(space_id) else {
                if evidence.kind == EvidenceKind::Text && self.default_text_embedder_disabled {
                    return Err(CgError::TextEmbedderDisabled);
                }
                continue;
            };
            let embed_input = match evidence.kind {
                EvidenceKind::Text => {
                    let Some(text) = text_projection else {
                        continue;
                    };
                    EmbeddingInput::Text(text)
                }
                EvidenceKind::ImageFrame => EmbeddingInput::ImageFrame {
                    uri: evidence.source_ref.clone(),
                },
                EvidenceKind::AudioSegment => EmbeddingInput::AudioSegment {
                    uri: evidence.source_ref.clone(),
                },
                EvidenceKind::VideoSegment => EmbeddingInput::VideoSegment {
                    uri: evidence.source_ref.clone(),
                },
                EvidenceKind::StructuredSignal => {
                    EmbeddingInput::StructuredSignal(serde_json::json!({
                        "source_ref": evidence.source_ref,
                        "properties": evidence.properties,
                    }))
                }
            };
            let output = embedder.embed(embed_input)?;
            if output.embedding_space_id != *space_id {
                return Err(CgError::EmbeddingSpaceIdMismatch {
                    embedder_declared: output.embedding_space_id,
                    registry_declared: space_id.clone(),
                });
            }
            input.embeddings.push(PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: output.embedding_space_id,
                vector: output.vector,
                target: EmbeddingTarget::Evidence(evidence.id),
                produced_by: output.producer,
            });
        }
        Ok(())
    }

    fn evidence_id_exists(&self, id: EvidenceId) -> Result<bool> {
        let result = self
            .db
            .execute(
                "SELECT id FROM evidence WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        Ok(!result.rows.is_empty())
    }

    fn evidence_context(&self, id: EvidenceId) -> Result<Option<ContextId>> {
        let result = self
            .db
            .execute(
                "SELECT context_id FROM evidence WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| get_uuid(row, &result.columns, "context_id").map(ContextId::from))
            .transpose()
    }

    fn duplicate_evidence_commit_error(&self, input: &RecordObservation) -> Result<CgError> {
        for evidence in &input.evidence {
            if self.evidence_id_exists(evidence.id)? {
                return Ok(CgError::DuplicateEvidenceId { id: evidence.id });
            }
        }
        Ok(CgError::DuplicateEvidenceId {
            id: input
                .evidence
                .first()
                .map(|evidence| evidence.id)
                .unwrap_or_default(),
        })
    }

    fn evidence_exists_in_context(&self, id: EvidenceId, context_id: ContextId) -> Result<bool> {
        let result = self
            .db
            .execute(
                "SELECT id FROM evidence WHERE id = $id AND context_id = $context_id",
                &HashMap::from([
                    ("id".to_string(), Value::Uuid(id.as_uuid())),
                    ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(!result.rows.is_empty())
    }

    fn digest_row_id(&self, id: DigestId, context_id: ContextId) -> Result<Option<RowId>> {
        let rows = self
            .db
            .scan("digests", self.db.snapshot())
            .map_err(map_engine_err)?;
        Ok(rows
            .into_iter()
            .find(|row| {
                row.values.get("id") == Some(&Value::Uuid(id.as_uuid()))
                    && row.values.get("context_id") == Some(&Value::Uuid(context_id.as_uuid()))
            })
            .map(|row| row.row_id))
    }

    fn digest_context(&self, id: DigestId) -> Result<Option<ContextId>> {
        let rows = self
            .db
            .scan("digests", self.db.snapshot())
            .map_err(map_engine_err)?;
        rows.into_iter()
            .find(|row| row.values.get("id") == Some(&Value::Uuid(id.as_uuid())))
            .map(|row| match row.values.get("context_id") {
                Some(Value::Uuid(context_id)) => Ok(ContextId::from(*context_id)),
                _ => Err(CgError::Engine("digest row missing context_id".to_string())),
            })
            .transpose()
    }

    fn map_embedding_route_engine_err(&self, err: EngineError, route: &EmbeddingRoute) -> CgError {
        if matches!(err, EngineError::ForeignKeyViolation { .. }) {
            return self
                .embedding_route_fk_error_candidate(route)
                .unwrap_or_else(|| map_engine_err(err));
        }
        map_engine_err(err)
    }

    fn map_embedding_commit_fk_err(&self, err: EngineError, routes: &[EmbeddingRoute]) -> CgError {
        routes
            .iter()
            .find_map(|route| self.embedding_route_fk_error_candidate(route))
            .unwrap_or_else(|| map_engine_err(err))
    }

    fn embedding_route_fk_error_candidate(&self, route: &EmbeddingRoute) -> Option<CgError> {
        match route.target {
            EmbeddingTarget::Evidence(id) => match self.evidence_context(id) {
                Ok(Some(target_context)) if target_context != route.context_id => {
                    Some(CgError::CrossContextEmbeddingTargetRejected {
                        handle_context: route.context_id,
                        target_context,
                    })
                }
                Ok(None) => Some(CgError::EmbeddingTargetNotFound { id }),
                Ok(Some(_)) => None,
                Err(err) => Some(err),
            },
            EmbeddingTarget::EntityReference(entity_id, _) => {
                match self.entity_context(entity_id) {
                    Ok(Some(target_context)) if target_context != route.context_id => {
                        Some(CgError::CrossContextEmbeddingTargetRejected {
                            handle_context: route.context_id,
                            target_context,
                        })
                    }
                    Ok(None) => Some(CgError::EntityNotFound { entity_id }),
                    Ok(Some(_)) => None,
                    Err(err) => Some(err),
                }
            }
            EmbeddingTarget::Digest(id) => match self.digest_context(id) {
                Ok(Some(target_context)) if target_context != route.context_id => {
                    Some(CgError::CrossContextEmbeddingTargetRejected {
                        handle_context: route.context_id,
                        target_context,
                    })
                }
                Ok(None) => Some(CgError::EmbeddingTargetNotFound {
                    id: EvidenceId::default(),
                }),
                Ok(Some(_)) => None,
                Err(err) => Some(err),
            },
            EmbeddingTarget::EventSearchPassage(passage_id) => {
                Some(CgError::EventSearchPassageTargetRequiresProjection { passage_id })
            }
        }
    }

    fn insert_evidence_in_tx(
        &self,
        tx: crate::TxId,
        evidence: &EvidenceRef,
        supply_order: i64,
        vectors: Option<&HashMap<String, Vec<f32>>>,
    ) -> Result<()> {
        let vector_text = vectors
            .and_then(|columns| columns.get("vector_text"))
            .cloned()
            .map(Value::Vector)
            .unwrap_or(Value::Null);
        let vector_vision = vectors
            .and_then(|columns| columns.get("vector_vision"))
            .cloned()
            .map(Value::Vector)
            .unwrap_or(Value::Null);
        let params = HashMap::from([
            ("id".to_string(), Value::Uuid(evidence.id.as_uuid())),
            (
                "observation_id".to_string(),
                Value::Uuid(evidence.observation_id.as_uuid()),
            ),
            (
                "context_id".to_string(),
                Value::Uuid(evidence.context_id.as_uuid()),
            ),
            (
                "kind".to_string(),
                Value::Text(evidence_kind_str(evidence.kind).to_string()),
            ),
            (
                "source_ref".to_string(),
                Value::Text(evidence.source_ref.clone()),
            ),
            (
                "mime_type".to_string(),
                value_for_optional_text(&evidence.mime_type),
            ),
            (
                "content_hash".to_string(),
                value_for_optional_text(&evidence.content_hash),
            ),
            (
                "captured_at".to_string(),
                value_for_optional_timestamp(&evidence.captured_at),
            ),
            (
                "captured_at_nanos".to_string(),
                value_for_optional_timestamp_nanos(&evidence.captured_at),
            ),
            (
                "byte_range".to_string(),
                value_for_optional_pair(evidence.byte_range),
            ),
            (
                "time_range_ms".to_string(),
                value_for_optional_pair(evidence.time_range_ms),
            ),
            (
                "frame_index".to_string(),
                value_for_optional_u64(evidence.frame_index),
            ),
            (
                "region".to_string(),
                value_for_optional_json(&evidence.region),
            ),
            ("producer".to_string(), producer_value(&evidence.producer)),
            (
                "text_projection".to_string(),
                value_for_optional_text(&evidence.text_projection),
            ),
            (
                "derived_from".to_string(),
                value_for_optional_evidence_id(evidence.derived_from),
            ),
            (
                "retention_status".to_string(),
                Value::Text(retention_status_str(evidence.retention_status).to_string()),
            ),
            (
                "media_retention_until".to_string(),
                value_for_optional_timestamp(&evidence.media_retention_until),
            ),
            (
                "media_retention_until_nanos".to_string(),
                value_for_optional_timestamp_nanos(&evidence.media_retention_until),
            ),
            (
                "properties".to_string(),
                properties_value(&evidence.properties),
            ),
            ("supply_order".to_string(), Value::Int64(supply_order)),
            ("vector_text".to_string(), vector_text),
            ("vector_vision".to_string(), vector_vision),
        ]);
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, mime_type, content_hash, captured_at, captured_at_nanos, byte_range, time_range_ms, frame_index, region, producer, text_projection, derived_from, retention_status, media_retention_until, media_retention_until_nanos, properties, supply_order, vector_text, vector_vision) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $mime_type, $content_hash, $captured_at, $captured_at_nanos, $byte_range, $time_range_ms, $frame_index, $region, $producer, $text_projection, $derived_from, $retention_status, $media_retention_until, $media_retention_until_nanos, $properties, $supply_order, $vector_text, $vector_vision)",
                &params,
            )
            .map(|_| ())
            .map_err(|err| {
                if matches!(
                    err,
                    EngineError::UniqueViolation {
                        ref table,
                        ref column
                    } if table == "evidence" && column == "id"
                ) {
                    CgError::DuplicateEvidenceId { id: evidence.id }
                } else {
                    map_engine_err(err)
                }
            })
    }

    fn merge_state_delta_in_tx<H: RecordObservationHooks>(
        &self,
        tx: crate::TxId,
        input: &RecordObservation,
        hooks: &mut H,
    ) -> std::result::Result<Option<EntitySnapshotId>, StateDeltaMergeError> {
        if input.state_delta.is_empty() {
            return Ok(None);
        }
        let current = self
            .open_entity_snapshot_for_merge(input.entity_id)
            .map_err(StateDeltaMergeError::Error)?;
        if let Some(current) = current.as_ref() {
            let still_open = self
                .db
                .guard_row_conditions_in_tx(
                    tx,
                    "entity_snapshots",
                    "id",
                    &Value::Uuid(current.id.as_uuid()),
                    &[("valid_to".to_string(), Value::Null)],
                )
                .map_err(map_engine_err)
                .map_err(StateDeltaMergeError::Error)?;
            let still_current = self
                .db
                .guard_row_conditions_in_tx(
                    tx,
                    "entity_current_snapshots",
                    "entity_id",
                    &Value::Uuid(input.entity_id.as_uuid()),
                    &[("snapshot_id".to_string(), Value::Uuid(current.id.as_uuid()))],
                )
                .map_err(map_engine_err)
                .map_err(StateDeltaMergeError::Error)?;
            if !still_open || !still_current {
                return Err(StateDeltaMergeError::Retry);
            }
            hooks.after_state_delta_guards();
        }
        let mut new_properties = current
            .as_ref()
            .map(|snapshot| snapshot.properties.clone())
            .unwrap_or_else(BTreeMap::new);
        for (key, value) in &input.state_delta {
            match value {
                Some(value) => {
                    new_properties.insert(key.clone(), value.clone());
                }
                None => {
                    new_properties.remove(key);
                }
            }
        }
        if current
            .as_ref()
            .is_some_and(|snapshot| snapshot.properties == new_properties)
        {
            return Ok(None);
        }

        let (entity_type, entity_name) = self
            .entity_snapshot_identity_in_tx(tx, input.entity_id)
            .map_err(StateDeltaMergeError::Error)?;
        let snapshot_id = EntitySnapshotId::new_v7();
        let snapshot_data = serde_json::json!({
            "entity_id": input.entity_id.as_uuid().to_string(),
            "entity_type": entity_type,
            "name": entity_name,
            "properties": properties_to_json(&new_properties),
        });
        let snap_params = HashMap::from([
            ("id".to_string(), Value::Uuid(snapshot_id.as_uuid())),
            (
                "context_id".to_string(),
                Value::Uuid(input.context_id.as_uuid()),
            ),
            (
                "entity_id".to_string(),
                Value::Uuid(input.entity_id.as_uuid()),
            ),
            ("data".to_string(), Value::Json(snapshot_data)),
            ("valid_from".to_string(), Value::TxId(tx)),
            ("valid_to".to_string(), Value::Null),
        ]);
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
                &snap_params,
            )
            .map_err(map_engine_err)
            .map_err(StateDeltaMergeError::Error)?;
        self.insert_entity_snapshot_observation_in_tx(
            tx,
            input.context_id,
            input.entity_id,
            snapshot_id,
            input.id,
            0,
        )
        .map_err(StateDeltaMergeError::Error)?;
        if let Some(current) = current {
            let link_params = HashMap::from([
                ("id".to_string(), Value::Uuid(uuid::Uuid::now_v7())),
                (
                    "context_id".to_string(),
                    Value::Uuid(input.context_id.as_uuid()),
                ),
                (
                    "entity_id".to_string(),
                    Value::Uuid(input.entity_id.as_uuid()),
                ),
                (
                    "predecessor_snapshot_id".to_string(),
                    Value::Uuid(current.id.as_uuid()),
                ),
                (
                    "successor_snapshot_id".to_string(),
                    Value::Uuid(snapshot_id.as_uuid()),
                ),
                ("created_at".to_string(), Value::TxId(tx)),
            ]);
            let inserted_link = self
                .db
                .execute_in_tx(
                    tx,
                    "INSERT INTO entity_snapshot_links (id, context_id, entity_id, predecessor_snapshot_id, successor_snapshot_id, created_at) VALUES ($id, $context_id, $entity_id, $predecessor_snapshot_id, $successor_snapshot_id, $created_at)",
                    &link_params,
                )
                .map_err(|err| {
                    if matches!(err, EngineError::UniqueViolation { .. }) {
                        StateDeltaMergeError::Retry
                    } else {
                        StateDeltaMergeError::Error(map_engine_err(err))
                    }
                })?
                .rows_affected;
            if inserted_link != 1 {
                return Err(StateDeltaMergeError::Retry);
            }
        }
        self.set_current_entity_snapshot_in_tx(tx, input.context_id, input.entity_id, snapshot_id)
            .map_err(StateDeltaMergeError::Error)?;
        Ok(Some(snapshot_id))
    }

    fn insert_embedding_handle_in_tx(
        &self,
        tx: crate::TxId,
        route: &EmbeddingRoute,
    ) -> std::result::Result<(), EngineError> {
        let params = HashMap::from([
            ("id".to_string(), Value::Uuid(route.id.as_uuid())),
            (
                "context_id".to_string(),
                Value::Uuid(route.context_id.as_uuid()),
            ),
            (
                "embedding_space_id".to_string(),
                Value::Text(route.embedding_space_id.clone()),
            ),
            (
                "target_id".to_string(),
                Value::Uuid(embedding_target_uuid(&route.target)),
            ),
            (
                "produced_by".to_string(),
                producer_value(&route.produced_by),
            ),
            (
                "producer_system".to_string(),
                Value::Text(route.produced_by.system.clone()),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        let handle_table = match route.target {
            EmbeddingTarget::Evidence(_) => "evidence_embedding_handles",
            EmbeddingTarget::EntityReference(_, _) => "entity_reference_embedding_handles",
            EmbeddingTarget::Digest(_) => "digest_embedding_handles",
            EmbeddingTarget::EventSearchPassage(_) => "event_search_embedding_handles",
        };
        self.db
            .execute_in_tx(
                tx,
                &format!("INSERT INTO {handle_table} (id, context_id, embedding_space_id, target_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $embedding_space_id, $target_id, $produced_by, $producer_system, $created_at)"),
                &params,
            )?;
        Ok(())
    }

    fn insert_embedding_vector_in_tx(
        &self,
        tx: crate::TxId,
        route: &EmbeddingRoute,
        row_id: RowId,
    ) -> Result<()> {
        self.db
            .insert_vector(
                tx,
                VectorIndexRef::new(&route.table_name, &route.vector_column),
                row_id,
                route.vector.clone(),
            )
            .map_err(map_engine_err)
    }

    fn insert_entity_reference_embedding_row_in_tx(
        &self,
        tx: crate::TxId,
        route: &EmbeddingRoute,
    ) -> std::result::Result<RowId, EngineError> {
        let EmbeddingTarget::EntityReference(entity_id, ref label) = route.target else {
            unreachable!("entity reference route required");
        };
        let params = HashMap::from([
            ("id".to_string(), Value::Uuid(route.id.as_uuid())),
            (
                "context_id".to_string(),
                Value::Uuid(route.context_id.as_uuid()),
            ),
            ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
            (
                "label".to_string(),
                label.clone().map(Value::Text).unwrap_or(Value::Null),
            ),
            ("vector".to_string(), Value::Vector(route.vector.clone())),
            (
                "embedding_space_id".to_string(),
                Value::Text(route.embedding_space_id.clone()),
            ),
            (
                "produced_by".to_string(),
                producer_value(&route.produced_by),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        self.db.insert_row(tx, &route.table_name, params)
    }

    fn insert_entity_reference_no_match_invalidations_in_tx(
        &self,
        tx: crate::TxId,
        input: &RecordObservation,
        routes: &[EmbeddingRoute],
    ) -> Result<()> {
        let evidence_by_id = input
            .evidence
            .iter()
            .map(|evidence| (evidence.id, evidence))
            .collect::<HashMap<_, _>>();
        let has_image_frame_evidence_route = routes.iter().any(|route| {
            let EmbeddingTarget::Evidence(evidence_id) = route.target else {
                return false;
            };
            evidence_by_id
                .get(&evidence_id)
                .is_some_and(|evidence| evidence.kind == EvidenceKind::ImageFrame)
        });
        if !has_image_frame_evidence_route {
            return Ok(());
        }
        let candidates =
            self.active_entity_basis_decisions_in_tx(tx, input.context_id, input.entity_id)?;
        if candidates.is_empty() {
            return Ok(());
        }

        let mut invalidated = HashSet::new();
        for route in routes {
            let EmbeddingTarget::Evidence(evidence_id) = route.target else {
                continue;
            };
            let Some(evidence) = evidence_by_id.get(&evidence_id) else {
                continue;
            };
            if evidence.kind != EvidenceKind::ImageFrame {
                continue;
            }

            for candidate in &candidates {
                if invalidated.contains(&candidate.decision_id) {
                    continue;
                }
                let best_reference_score = self.best_entity_reference_score_in_tx(
                    tx,
                    candidate.entity_id,
                    input.context_id,
                    &route.embedding_space_id,
                    &route.vector,
                )?;
                let Some(best_reference_score) = best_reference_score else {
                    continue;
                };
                if best_reference_score >= ENTITY_REFERENCE_MATCH_THRESHOLD {
                    continue;
                }

                let (_, entity_name) =
                    self.entity_snapshot_identity_in_tx(tx, candidate.entity_id)?;
                let snapshot_id = self.decision_basis_snapshot_for_entity_in_tx(
                    tx,
                    input.context_id,
                    candidate.decision_id,
                    candidate.entity_id,
                )?;
                self.insert_entity_no_match_invalidation_in_tx(
                    tx,
                    EntityNoMatchInvalidation {
                        context_id: input.context_id,
                        observation_id: input.id,
                        decision_id: candidate.decision_id,
                        entity_id: candidate.entity_id,
                        entity_name: &entity_name,
                        basis_snapshot_id: snapshot_id,
                        confidence: candidate.confidence,
                        embedding_space_id: &route.embedding_space_id,
                    },
                )?;
                invalidated.insert(candidate.decision_id);
            }
        }

        Ok(())
    }

    fn active_entity_basis_decisions_in_tx(
        &self,
        tx: crate::TxId,
        context_id: ContextId,
        live_entity_id: EntityId,
    ) -> Result<Vec<EntityBasisDecisionCandidate>> {
        let result = self
            .db
            .execute_in_tx(
                tx,
                "SELECT edges.source_id, edges.target_id, decisions.confidence FROM edges INNER JOIN decisions ON edges.source_id = decisions.id WHERE edges.context_id = $context_id AND edges.edge_type = $edge_type AND decisions.context_id = $context_id AND decisions.status = $status",
                &HashMap::from([
                    ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                    ("edge_type".to_string(), Value::Text("BASED_ON".to_string())),
                    ("status".to_string(), Value::Text("active".to_string())),
                ]),
            )
            .map_err(map_engine_err)?;
        let mut candidates = Vec::new();
        for row in &result.rows {
            let entity_id = EntityId::from(get_uuid(row, &result.columns, "target_id")?);
            if entity_id == live_entity_id {
                continue;
            }
            candidates.push(EntityBasisDecisionCandidate {
                decision_id: DecisionId::from(get_uuid(row, &result.columns, "source_id")?),
                entity_id,
                confidence: get_optional_float(row, &result.columns, "confidence")?,
            });
        }
        candidates.sort_by_key(|candidate| (candidate.decision_id, candidate.entity_id));
        candidates.dedup_by_key(|candidate| (candidate.decision_id, candidate.entity_id));
        Ok(candidates)
    }

    fn best_entity_reference_score_in_tx(
        &self,
        tx: crate::TxId,
        entity_id: EntityId,
        context_id: ContextId,
        embedding_space_id: &str,
        query_vector: &[f32],
    ) -> Result<Option<f32>> {
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
        let result = self
            .db
            .execute_in_tx(
                tx,
                &format!(
                    "SELECT id, score FROM {table} WHERE context_id = $context_id AND entity_id = $entity_id AND embedding_space_id = $embedding_space_id ORDER BY vector <=> $query LIMIT 1"
                ),
                &HashMap::from([
                    ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                    ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
                    (
                        "embedding_space_id".to_string(),
                        Value::Text(embedding_space_id.to_string()),
                    ),
                    ("query".to_string(), Value::Vector(query_vector.to_vec())),
                ]),
            )
            .map_err(map_engine_err)?;
        let Some(row) = result.rows.first() else {
            return Ok(None);
        };
        let score = get_optional_float(row, &result.columns, "score")?.ok_or_else(|| {
            CgError::Engine("entity reference vector search missing score".to_string())
        })?;
        Ok(Some(score as f32))
    }

    fn decision_basis_snapshot_for_entity_in_tx(
        &self,
        tx: crate::TxId,
        context_id: ContextId,
        decision_id: DecisionId,
        entity_id: EntityId,
    ) -> Result<EntitySnapshotId> {
        let result = self
            .db
            .execute_in_tx(
                tx,
                "SELECT snapshot_id FROM decision_basis_snapshots WHERE context_id = $context_id AND decision_id = $decision_id AND entity_id = $entity_id",
                &HashMap::from([
                    ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                    ("decision_id".to_string(), Value::Uuid(decision_id.as_uuid())),
                    ("entity_id".to_string(), Value::Uuid(entity_id.as_uuid())),
                ]),
            )
            .map_err(map_engine_err)?;
        let Some(row) = result.rows.first() else {
            return Err(CgError::NotFound {
                id: Some(decision_id.as_uuid().to_string()),
                missing: Some("decision_basis_snapshot"),
            });
        };
        Ok(EntitySnapshotId::from(get_uuid(
            row,
            &result.columns,
            "snapshot_id",
        )?))
    }

    fn insert_entity_no_match_invalidation_in_tx(
        &self,
        tx: crate::TxId,
        invalidation: EntityNoMatchInvalidation<'_>,
    ) -> Result<()> {
        let intention_context = decision_invalidation_context(
            &self.db,
            tx,
            invalidation.context_id.as_uuid(),
            invalidation.decision_id.as_uuid(),
            invalidation.confidence,
        )
        .map_err(map_engine_err)?;
        let override_failure = serde_json::to_value(OverrideFailureReason::EntityAwareNoMatch {
            reference_space: invalidation.embedding_space_id.to_string(),
            live_space: invalidation.embedding_space_id.to_string(),
        })
        .map_err(|err| CgError::Engine(format!("override failure serialize: {err}")))?;
        let basis_diff = BasisDiffResult {
            is_invalidated_candidate: true,
            divergences: vec![EntityDivergence {
                basis_snapshot_id: invalidation.basis_snapshot_id,
                entity_id: invalidation.entity_id,
                value_diffs: vec![
                    ValueDiff {
                        field: "entity_name".to_string(),
                        basis_value: None,
                        current_value: Some(JsonValue::String(
                            invalidation.entity_name.to_string(),
                        )),
                    },
                    ValueDiff {
                        field: "entity_id".to_string(),
                        basis_value: None,
                        current_value: Some(serde_json::json!(invalidation.entity_id.as_uuid())),
                    },
                    ValueDiff {
                        field: "override_failure_reason".to_string(),
                        basis_value: None,
                        current_value: Some(override_failure),
                    },
                ],
                read_txid: tx,
            }],
            read_txid: tx,
            current_snapshot_read_txid: tx,
        };
        let basis_diff_json = serde_json::to_value(&basis_diff)
            .map_err(|err| CgError::Engine(format!("basis diff serialize: {err}")))?;
        let threatened_ids = intention_context
            .threatened_intention_ids
            .iter()
            .copied()
            .map(IntentionId::from)
            .collect::<Vec<_>>();
        let threatened_json = serde_json::to_value(threatened_ids)
            .map_err(|err| CgError::Engine(format!("threatened intentions serialize: {err}")))?;

        let updated = self
            .db
            .execute_in_tx(
                tx,
                    "UPDATE decisions SET status = $new_status WHERE id = $id AND context_id = $context_id AND status = $old_status",
                &HashMap::from([
                    ("id".to_string(), Value::Uuid(invalidation.decision_id.as_uuid())),
                    (
                        "context_id".to_string(),
                        Value::Uuid(invalidation.context_id.as_uuid()),
                    ),
                    (
                        "new_status".to_string(),
                        Value::Text("invalidated".to_string()),
                    ),
                    ("old_status".to_string(), Value::Text("active".to_string())),
                ]),
            )
            .map_err(map_engine_err)?;
        if updated.rows_affected == 0 {
            return Ok(());
        }
        if updated.rows_affected != 1 {
            return Err(CgError::Engine(format!(
                "entity no-match invalidated {} decisions, expected 1",
                updated.rows_affected
            )));
        }

        let invalidation_id = InvalidationId::from(invalidation.decision_id.as_uuid());
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $reason, $basis_diff, $cascade_from, $severity, $threatened_intention_ids, $intention_now_un_served, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
                &HashMap::from([
                    ("id".to_string(), Value::Uuid(invalidation_id.as_uuid())),
                    ("status".to_string(), Value::Text("pending".to_string())),
                    (
                        "context_id".to_string(),
                        Value::Uuid(invalidation.context_id.as_uuid()),
                    ),
                    (
                        "affected_decision_id".to_string(),
                        Value::Uuid(invalidation.decision_id.as_uuid()),
                    ),
                    (
                        "trigger_observation_id".to_string(),
                        Value::Uuid(invalidation.observation_id.as_uuid()),
                    ),
                    (
                        "reason".to_string(),
                        Value::Text("entity-reference-no-match".to_string()),
                    ),
                    ("basis_diff".to_string(), Value::Json(basis_diff_json)),
                    ("cascade_from".to_string(), Value::Null),
                    (
                        "severity".to_string(),
                        Value::Text(invalidation_severity_text(intention_context.severity).to_string()),
                    ),
                    (
                        "threatened_intention_ids".to_string(),
                        Value::Json(threatened_json),
                    ),
                    (
                        "intention_now_un_served".to_string(),
                        Value::Bool(intention_context.intention_now_un_served),
                    ),
                    ("detected_at".to_string(), Value::TxId(tx)),
                    ("resolved_at".to_string(), Value::Null),
                    ("resolution_action".to_string(), Value::Null),
                    ("resolution_decision_id".to_string(), Value::Null),
                ]),
            )
            .map_err(map_engine_err)?;
        Ok(())
    }

    fn open_entity_snapshot_for_merge(
        &self,
        entity_id: EntityId,
    ) -> Result<Option<EntitySnapshotForMerge>> {
        let pointer = self
            .db
            .execute(
                "SELECT snapshot_id FROM entity_current_snapshots WHERE entity_id = $entity_id",
                &HashMap::from([("entity_id".to_string(), Value::Uuid(entity_id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        let Some(row) = pointer.rows.first() else {
            return Ok(None);
        };
        let snapshot_id = EntitySnapshotId::from(get_uuid(row, &pointer.columns, "snapshot_id")?);
        let result = self
            .db
            .execute(
                "SELECT id, data FROM entity_snapshots WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(snapshot_id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| decode_snapshot_for_merge(&result.columns, row))
            .transpose()
    }

    fn entity_snapshot_identity_in_tx(
        &self,
        tx: crate::TxId,
        entity_id: EntityId,
    ) -> Result<(String, String)> {
        let result = self
            .db
            .execute_in_tx(
                tx,
                "SELECT entity_type, name FROM entities WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(entity_id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        let Some(row) = result.rows.first() else {
            return Err(CgError::NotFound {
                id: Some(entity_id.as_uuid().to_string()),
                missing: Some("entity"),
            });
        };
        let entity_type = entity_type_from_str(&get_text(row, &result.columns, "entity_type")?)?;
        let name = get_text(row, &result.columns, "name")?;
        Ok((entity_type_str(entity_type).to_string(), name))
    }
}

fn record_observation_snapshot_retry_unique(table: &str, _column: &str) -> bool {
    matches!(table, "entity_snapshot_links" | "entity_current_snapshots")
}

fn record_observation_auto_create_retry_unique(table: &str, _column: &str) -> bool {
    matches!(table, "contexts" | "entities")
}

fn is_embedding_handle_table(table: &str) -> bool {
    matches!(
        table,
        "evidence_embedding_handles"
            | "entity_reference_embedding_handles"
            | "digest_embedding_handles"
    )
}

fn evidence_vectors_for_insert(
    routes: &[EmbeddingRoute],
) -> Result<HashMap<EvidenceId, HashMap<String, Vec<f32>>>> {
    let mut by_evidence: HashMap<EvidenceId, HashMap<String, Vec<f32>>> = HashMap::new();
    for route in routes {
        let EmbeddingTarget::Evidence(evidence_id) = route.target else {
            continue;
        };
        if route.table_name != "evidence" {
            return Err(CgError::Engine(format!(
                "evidence embedding route points at unexpected table {}",
                route.table_name
            )));
        }
        let columns = by_evidence.entry(evidence_id).or_default();
        if columns
            .insert(route.vector_column.clone(), route.vector.clone())
            .is_some()
        {
            return Err(CgError::Engine(format!(
                "multiple embeddings target evidence {} column {}",
                evidence_id.as_uuid(),
                route.vector_column
            )));
        }
    }
    Ok(by_evidence)
}

#[derive(Debug, Clone)]
pub(super) struct EmbeddingRoute {
    pub id: EmbeddingId,
    pub context_id: ContextId,
    pub embedding_space_id: String,
    pub vector: Vec<f32>,
    pub target: EmbeddingTarget,
    pub produced_by: EvidenceProducer,
    pub table_name: String,
    pub vector_column: String,
}

#[derive(Debug, Clone, Copy)]
struct EntityBasisDecisionCandidate {
    decision_id: DecisionId,
    entity_id: EntityId,
    confidence: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct EntityNoMatchInvalidation<'a> {
    context_id: ContextId,
    observation_id: ObservationId,
    decision_id: DecisionId,
    entity_id: EntityId,
    entity_name: &'a str,
    basis_snapshot_id: EntitySnapshotId,
    confidence: Option<f64>,
    embedding_space_id: &'a str,
}

#[derive(Debug, Clone)]
struct EntitySnapshotForMerge {
    id: EntitySnapshotId,
    properties: BTreeMap<String, JsonValue>,
}

enum StateDeltaMergeError {
    Retry,
    Error(CgError),
}

fn decode_snapshot_for_merge(columns: &[String], row: &[Value]) -> Result<EntitySnapshotForMerge> {
    let id = EntitySnapshotId::from(get_uuid(row, columns, "id")?);
    let data = get_json(row, columns, "data")?;
    let properties = match data.get("properties") {
        Some(JsonValue::Object(map)) => map
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        _ => BTreeMap::new(),
    };
    Ok(EntitySnapshotForMerge { id, properties })
}

fn invalidation_severity_text(severity: InvalidationSeverity) -> &'static str {
    match severity {
        InvalidationSeverity::Info => "info",
        InvalidationSeverity::Warning => "warning",
        InvalidationSeverity::Critical => "critical",
    }
}

/// One observation by id, with the evidence it carries, as both doors answer it. An observation
/// that is not in the store is a typed refusal rather than an absence: an observation is what a
/// caller was pointed at, so "it is not there" is a fact about the pointer.
///
/// Reads `observations`, and through the evidence read `evidence` and
/// `evidence_availability_log`.
pub(crate) fn get_observation(rows: &dyn GraphRows, id: ObservationId) -> Result<Observation> {
    if !id.is_valid() {
        return Err(CgError::InvalidIdentity {
            context: "ObservationId",
        });
    }
    let params: HashMap<String, Value> = [("id".to_string(), Value::Uuid(id.as_uuid()))]
        .into_iter()
        .collect();
    let result = rows.query_rows(
        "SELECT id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at FROM observations WHERE id = $id",
        &params,
    )?;
    if result.rows.is_empty() {
        return Err(CgError::NotFound {
            id: Some(id.as_uuid().to_string()),
            missing: Some("observation"),
        });
    }
    let mut observation = decode_observation(&result.columns, &result.rows[0])?;
    observation.evidence = list_evidence_for_observation(rows, id)?;
    Ok(observation)
}

/// The observations in one context, or every observation in the store when no context is named,
/// each with the evidence it carries.
///
/// Reads `observations`, and through the evidence read `evidence` and
/// `evidence_availability_log`.
pub(crate) fn list_observations(
    rows: &dyn GraphRows,
    context_id: Option<ContextId>,
) -> Result<Vec<Observation>> {
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
    let result = rows.query_rows(sql, &params)?;
    let mut observations = result
        .rows
        .iter()
        .map(|row| decode_observation(&result.columns, row))
        .collect::<Result<Vec<_>>>()?;
    for observation in &mut observations {
        observation.evidence = list_evidence_for_observation(rows, observation.id)?;
    }
    Ok(observations)
}

fn decode_observation(columns: &[String], row: &[Value]) -> Result<Observation> {
    let ts = get_timestamp(row, columns, "timestamp")?;
    let observed_at = DateTime::<Utc>::from_timestamp_millis(ts).unwrap_or_else(Utc::now);
    Ok(Observation {
        id: ObservationId::from(get_uuid(row, columns, "id")?),
        observation_type: get_text(row, columns, "observation_type")?,
        evidence: Vec::new(),
        observed_properties: properties_from_json(&get_json(row, columns, "observed_properties")?),
        state_delta: state_delta_from_json(get_json(row, columns, "state_delta")?)?,
        properties: properties_from_json(&get_json(row, columns, "properties")?),
        source: get_text(row, columns, "source")?,
        entity_id: EntityId::from(get_uuid(row, columns, "entity_id")?),
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        observed_at,
        created_at: get_txid(row, columns, "created_at")?,
    })
}

fn contains_raw_media_payload(properties: &BTreeMap<String, JsonValue>) -> bool {
    properties
        .iter()
        .any(|(key, value)| is_raw_media_key(key) || contains_raw_media_value(value))
}

fn contains_raw_media_state_delta(delta: &BTreeMap<String, Option<JsonValue>>) -> bool {
    delta.iter().any(|(key, value)| {
        is_raw_media_key(key) || value.as_ref().is_some_and(contains_raw_media_value)
    })
}

/// Whether a persisted `state_delta` carries no state-change ops — i.e. it is a non-invalidating
/// observation (records nothing that could change an entity snapshot). Tolerates the canonical
/// `{"format":"cg_state_delta_v1","ops":{}}` shape and a bare empty object.
fn state_delta_has_no_ops(json: &JsonValue) -> bool {
    match json {
        JsonValue::Object(object) => match object.get("ops") {
            Some(JsonValue::Object(ops)) => ops.is_empty(),
            Some(_) => false,
            // No "ops" key at all (a bare object) is empty only when the object itself is empty.
            None => object.is_empty(),
        },
        JsonValue::Null => true,
        _ => false,
    }
}

fn state_delta_to_json(delta: &BTreeMap<String, Option<JsonValue>>) -> JsonValue {
    let mut ops = serde_json::Map::new();
    for (key, value) in delta {
        let mut op = serde_json::Map::new();
        match value {
            Some(value) => {
                op.insert("op".to_string(), JsonValue::String("set".to_string()));
                op.insert("value".to_string(), value.clone());
            }
            None => {
                op.insert("op".to_string(), JsonValue::String("delete".to_string()));
            }
        }
        ops.insert(key.clone(), JsonValue::Object(op));
    }
    serde_json::json!({
        "format": "cg_state_delta_v1",
        "ops": ops,
    })
}

pub(super) fn state_delta_from_json(
    value: JsonValue,
) -> Result<BTreeMap<String, Option<JsonValue>>> {
    let JsonValue::Object(mut object) = value else {
        return Err(CgError::Engine(
            "observations.state_delta must be a JSON object".to_string(),
        ));
    };
    if matches!(
        object.get("format").and_then(JsonValue::as_str),
        Some("cg_state_delta_v1")
    ) {
        let ops = object
            .remove("ops")
            .ok_or_else(|| CgError::Engine("state_delta missing ops".to_string()))?;
        let JsonValue::Object(ops) = ops else {
            return Err(CgError::Engine(
                "state_delta ops must be a JSON object".to_string(),
            ));
        };
        let mut out = BTreeMap::new();
        for (key, op) in ops {
            let JsonValue::Object(mut op) = op else {
                return Err(CgError::Engine(
                    "state_delta op must be a JSON object".to_string(),
                ));
            };
            match op.get("op").and_then(JsonValue::as_str) {
                Some("set") => {
                    let value = op.remove("value").ok_or_else(|| {
                        CgError::Engine("state_delta set op missing value".to_string())
                    })?;
                    out.insert(key, Some(value));
                }
                Some("delete") => {
                    out.insert(key, None);
                }
                other => {
                    return Err(CgError::Engine(format!(
                        "state_delta has unknown op {other:?}"
                    )));
                }
            }
        }
        return Ok(out);
    }
    Ok(object
        .into_iter()
        .map(|(key, value)| {
            let value = if value.is_null() { None } else { Some(value) };
            (key, value)
        })
        .collect())
}

fn contains_raw_media_value(value: &JsonValue) -> bool {
    match value {
        JsonValue::String(s) => s.trim_start().to_ascii_lowercase().starts_with("data:"),
        JsonValue::Array(values) => values.iter().any(contains_raw_media_value),
        JsonValue::Object(map) => map
            .iter()
            .any(|(key, value)| is_raw_media_key(key) || contains_raw_media_value(value)),
        _ => false,
    }
}

fn is_raw_media_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    if matches!(
        key.as_str(),
        "expected_bytes" | "byte_size" | "total_byte_budget"
    ) {
        return false;
    }
    key.contains("raw_media")
        || key.contains("rawmedia")
        || key.contains("media_payload")
        || key.contains("base64_media")
        || key.contains("base64")
        || key.contains("bytes")
}

fn validate_source_ref(
    source_ref: &str,
    https_allowed_hosts: &[String],
    file_uri_allowed: bool,
) -> Result<()> {
    let Some((scheme, rest)) = source_ref.split_once(':') else {
        return Err(CgError::InvalidSourceRefScheme {
            scheme: String::new(),
            host: None,
        });
    };
    let scheme = scheme.to_ascii_lowercase();
    if scheme == "data" {
        return Err(CgError::RawMediaNotStored);
    }
    if matches!(
        scheme.as_str(),
        "blob" | "javascript" | "vbscript" | "about"
    ) {
        return Err(CgError::InvalidSourceRefScheme { scheme, host: None });
    }
    if matches!(
        scheme.as_str(),
        "vigil-edge" | "s3" | "gs" | "azure" | "cas" | "webhook"
    ) {
        return Ok(());
    }
    if scheme == "file" {
        if file_uri_allowed {
            return Ok(());
        }
        return Err(CgError::InvalidSourceRefScheme { scheme, host: None });
    }
    if scheme == "https" {
        let host = rest
            .strip_prefix("//")
            .and_then(|rest| rest.split('/').next())
            .and_then(|authority| authority.split('@').next_back())
            .map(|authority| authority.split(':').next().unwrap_or(authority))
            .unwrap_or_default()
            .to_ascii_lowercase();
        if https_allowed_hosts.iter().any(|allowed| allowed == &host) {
            return Ok(());
        }
        return Err(CgError::InvalidSourceRefScheme {
            scheme,
            host: Some(host),
        });
    }
    Err(CgError::InvalidSourceRefScheme { scheme, host: None })
}

fn validate_content_hash(hash: Option<&str>) -> Result<()> {
    let Some(hash) = hash else {
        return Ok(());
    };
    let valid = hash.len() == 71
        && hash.starts_with("sha256:")
        && hash[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
    if valid {
        Ok(())
    } else {
        Err(CgError::InvalidEvidenceHash {
            hash: hash.to_string(),
        })
    }
}

fn validate_mime(kind: EvidenceKind, mime: Option<&str>) -> Result<()> {
    if mime == Some("application/octet-stream") {
        return Err(CgError::InvalidMimeForEvidenceKind);
    }
    let accepted = match kind {
        EvidenceKind::Text => {
            mime.is_none()
                || mime == Some("application/json")
                || mime.is_some_and(|m| m.starts_with("text/"))
        }
        EvidenceKind::StructuredSignal => mime.is_none() || mime == Some("application/json"),
        EvidenceKind::ImageFrame => mime.is_some_and(|mime| mime.starts_with("image/")),
        EvidenceKind::AudioSegment => mime.is_some_and(|mime| mime.starts_with("audio/")),
        EvidenceKind::VideoSegment => mime.is_some_and(|mime| mime.starts_with("video/")),
    };
    if accepted {
        Ok(())
    } else {
        Err(CgError::InvalidMimeForEvidenceKind)
    }
}

fn validate_region(region: Option<&JsonValue>) -> Result<()> {
    let Some(JsonValue::Object(region)) = region else {
        return Ok(());
    };
    if !(region.contains_key("x")
        || region.contains_key("y")
        || region.contains_key("w")
        || region.contains_key("h"))
    {
        if region.contains_key("line_start") || region.contains_key("line_end") {
            let line_start = region_integer(region, "line_start")?;
            let line_end = region_integer(region, "line_end")?;
            if line_start < 0 {
                return Err(CgError::InvalidEvidenceRegionBounds {
                    field: "line_start",
                });
            }
            if line_end < line_start {
                return Err(CgError::InvalidEvidenceRegionBounds { field: "line_end" });
            }
        }
        return Ok(());
    }
    let x = region_number(region, "x")?;
    let y = region_number(region, "y")?;
    let w = region_number(region, "w")?;
    let h = region_number(region, "h")?;
    if x < 0.0 {
        return Err(CgError::InvalidEvidenceRegionBounds { field: "x" });
    }
    if y < 0.0 {
        return Err(CgError::InvalidEvidenceRegionBounds { field: "y" });
    }
    if w <= 0.0 {
        return Err(CgError::InvalidEvidenceRegionBounds { field: "w" });
    }
    if h <= 0.0 {
        return Err(CgError::InvalidEvidenceRegionBounds { field: "h" });
    }
    if x + w > 1.0 {
        return Err(CgError::InvalidEvidenceRegionBounds { field: "x" });
    }
    if y + h > 1.0 {
        return Err(CgError::InvalidEvidenceRegionBounds { field: "y" });
    }
    Ok(())
}

fn region_integer(region: &serde_json::Map<String, JsonValue>, field: &'static str) -> Result<i64> {
    region
        .get(field)
        .and_then(JsonValue::as_i64)
        .ok_or(CgError::InvalidEvidenceRegionBounds { field })
}

fn region_number(region: &serde_json::Map<String, JsonValue>, field: &'static str) -> Result<f64> {
    region
        .get(field)
        .and_then(JsonValue::as_f64)
        .filter(|value| value.is_finite())
        .ok_or(CgError::InvalidEvidenceRegionBounds { field })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::mpsc;

    use chrono::Utc;

    use super::*;
    use crate::{CreateContext, CreateEntity, EntityType, StoreConfig};

    fn observation_input(
        entity_id: EntityId,
        context_id: ContextId,
        region: &str,
    ) -> RecordObservation {
        let observation_id = ObservationId::new_v7();
        RecordObservation {
            id: observation_id,
            entity_id,
            context_id,
            observation_type: "snapshot-guard-race".to_string(),
            source: "unit-test".to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: EvidenceId::new_v7(),
                observation_id,
                context_id,
                kind: EvidenceKind::Text,
                source_ref:
                    "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .to_string(),
                mime_type: Some("text/plain".to_string()),
                content_hash: None,
                captured_at: Some(Utc::now()),
                byte_range: None,
                time_range_ms: None,
                frame_index: None,
                region: None,
                producer: EvidenceProducer {
                    system: "unit-test".to_string(),
                    model_name: "fixture".to_string(),
                    model_version: "1".to_string(),
                    pipeline_version: "1".to_string(),
                },
                text_projection: None,
                derived_from: None,
                retention_status: RetentionStatus::RetainedExternal,
                media_retention_until: None,
                properties: BTreeMap::new(),
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::from([("region".to_string(), Some(serde_json::json!(region)))]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        }
    }

    struct SnapshotGuardRaceHook {
        paused_once: bool,
        guards_registered: mpsc::Sender<()>,
        release: Option<mpsc::Receiver<()>>,
        conditional_conflicts: usize,
    }

    impl RecordObservationHooks for SnapshotGuardRaceHook {
        fn after_state_delta_guards(&mut self) {
            if !self.paused_once {
                self.paused_once = true;
                self.guards_registered
                    .send(())
                    .expect("strict snapshot guards announced");
                self.release
                    .take()
                    .expect("one-shot release receiver")
                    .recv()
                    .expect("stale no-op writer released");
            }
        }

        fn conditional_update_conflict(&mut self) {
            self.conditional_conflicts += 1;
        }
    }

    #[test]
    fn stale_no_op_snapshot_merge_retries_after_strict_guard_conflict() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Store::open(StoreConfig {
            db_path: dir.path().join("snapshot-guard-race.db"),
            default_text_embedder: None,
            ..Default::default()
        })
        .expect("open store");
        let context_id = store
            .create_context(CreateContext {
                name: "snapshot-guard-race".to_string(),
                labels: Vec::new(),
                properties: BTreeMap::new(),
            })
            .expect("context")
            .id;
        let entity_id = store
            .create_entity(CreateEntity {
                entity_type: EntityType::Service,
                name: "stateful service".to_string(),
                properties: BTreeMap::from([("region".to_string(), serde_json::json!("stable"))]),
                tags: Vec::new(),
                context_id,
            })
            .expect("entity")
            .id;
        let (guards_registered_tx, guards_registered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let stale_writer = store.clone();
        let stale = std::thread::spawn(move || {
            let mut hook = SnapshotGuardRaceHook {
                paused_once: false,
                guards_registered: guards_registered_tx,
                release: Some(release_rx),
                conditional_conflicts: 0,
            };
            let result = stale_writer.record_observation_with_hooks(
                observation_input(entity_id, context_id, "stable"),
                &mut hook,
            );
            (result, hook.conditional_conflicts)
        });

        guards_registered_rx
            .recv()
            .expect("the no-op writer registered strict guards before its decision");
        store
            .record_observation(observation_input(entity_id, context_id, "draining"))
            .expect("competing state change commits while stale writer is paused");
        release_tx.send(()).expect("release stale no-op writer");
        let (restored, conditional_conflicts) = stale.join().expect("stale writer joined");
        let restored = restored.expect("conditional conflict retries");

        assert_eq!(
            conditional_conflicts, 1,
            "the stale no-op commit must observe ConditionalUpdateConflict exactly once"
        );
        assert!(
            restored.snapshot_id.is_some(),
            "retry turns the former no-op into the snapshot that restores the requested state"
        );
        let history = store.entity_history(entity_id).expect("history");
        assert_eq!(
            history.len(),
            3,
            "seed, competing, and restoring snapshots remain"
        );
        assert_eq!(
            history
                .last()
                .expect("restoring snapshot")
                .properties
                .get("region"),
            Some(&serde_json::json!("stable")),
            "the retry restores the no-op writer's intended state"
        );
    }
}
