use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, Utc};
use contextdb_core::{Error as EngineError, RowId, Value, VectorIndexRef};
use serde_json::Value as JsonValue;

use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::{entity_type_from_str, entity_type_str, map_engine_err};
use super::encode::{
    get_json, get_text, get_timestamp, get_txid, get_uuid, properties_from_json, properties_to_json,
};
use super::evidence::{
    embedding_target_kind, embedding_target_uuid, producer_value, properties_value,
    validate_space_kind, value_for_optional_evidence_id, value_for_optional_json,
    value_for_optional_pair, value_for_optional_text, value_for_optional_timestamp,
    value_for_optional_timestamp_nanos, value_for_optional_u64, vector_binding_for_target,
};
use super::{evidence_kind_str, retention_status_str};

impl Store {
    pub fn record_observation(
        &self,
        mut input: RecordObservation,
    ) -> Result<RecordObservationResult> {
        for evidence in &mut input.evidence {
            evidence.observation_id = input.id;
            evidence.context_id = input.context_id;
        }
        self.validate_record_observation_input(&input)?;
        let embedding_routes = self.validate_embeddings_for_observation(&input)?;
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
            let mut evidence_row_ids: HashMap<EvidenceId, RowId> = HashMap::new();
            for (supply_order, evidence) in input.evidence.iter().enumerate() {
                match self.insert_evidence_in_tx(tx, evidence, supply_order as i64) {
                    Ok(row_id) => {
                        evidence_row_ids.insert(evidence.id, row_id);
                    }
                    Err(err) => {
                        let _ = self.db.rollback(tx);
                        return Err(err);
                    }
                }
            }
            let snapshot_id = match self.merge_state_delta_in_tx(tx, &input) {
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
                let row_id = match route.target {
                    EmbeddingTarget::Evidence(id) => evidence_row_ids.get(&id).copied(),
                    EmbeddingTarget::Digest(_) => None,
                };
                let Some(row_id) = row_id else {
                    let _ = self.db.rollback(tx);
                    return Err(CgError::EmbeddingTargetNotFound {
                        id: match route.target {
                            EmbeddingTarget::Evidence(id) => id,
                            EmbeddingTarget::Digest(_) => EvidenceId::default(),
                        },
                    });
                };
                if let Err(err) = self.insert_embedding_handle_and_vector_in_tx(tx, route, row_id) {
                    let _ = self.db.rollback(tx);
                    return Err(err);
                }
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
                        return Err(map_engine_err(EngineError::UniqueViolation {
                            table,
                            column,
                        }));
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
        let mut observation = decode_observation(&result.columns, &result.rows[0])?;
        observation.evidence = self.list_evidence_for_observation(id)?;
        Ok(observation)
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
        let mut observations = result
            .rows
            .iter()
            .map(|row| decode_observation(&result.columns, row))
            .collect::<Result<Vec<_>>>()?;
        for observation in &mut observations {
            observation.evidence = self.list_evidence_for_observation(observation.id)?;
        }
        Ok(observations)
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
    ) -> Result<Vec<EmbeddingRoute>> {
        let evidence_by_id = input
            .evidence
            .iter()
            .map(|evidence| (evidence.id, evidence))
            .collect::<HashMap<_, _>>();
        let mut routes = Vec::with_capacity(input.embeddings.len());
        for embedding in &input.embeddings {
            let target_id = match embedding.target {
                EmbeddingTarget::Evidence(id) => id,
                EmbeddingTarget::Digest(_) => {
                    return Err(CgError::EmbeddingTargetNotFound {
                        id: EvidenceId::default(),
                    });
                }
            };
            let Some(evidence) = evidence_by_id.get(&target_id) else {
                return Err(CgError::EmbeddingTargetNotFound { id: target_id });
            };
            let space = self.embedding_space_for_id(&embedding.embedding_space_id)?;
            if embedding.vector.len() != space.dimension {
                return Err(CgError::EmbeddingSpaceMismatch {
                    conflicting_field: "dimension",
                });
            }
            validate_space_kind(&space, evidence.kind)?;
            let binding = vector_binding_for_target(&space, "evidence")?;
            routes.push(EmbeddingRoute {
                id: embedding.id,
                context_id: input.context_id,
                embedding_space_id: embedding.embedding_space_id.clone(),
                vector: embedding.vector.clone(),
                target: embedding.target.clone(),
                produced_by: embedding.produced_by.clone(),
                dimension: space.dimension,
                table_name: binding.table_name,
                vector_column: binding.vector_column,
            });
        }
        Ok(routes)
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

    fn insert_evidence_in_tx(
        &self,
        tx: crate::TxId,
        evidence: &EvidenceRef,
        supply_order: i64,
    ) -> Result<RowId> {
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
        ]);
        self.db.insert_row(tx, "evidence", params).map_err(|err| {
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

    fn merge_state_delta_in_tx(
        &self,
        tx: crate::TxId,
        input: &RecordObservation,
    ) -> std::result::Result<Option<EntitySnapshotId>, StateDeltaMergeError> {
        if input.state_delta.is_empty() {
            return Ok(None);
        }
        let current = self
            .open_entity_snapshot_for_merge(input.entity_id)
            .map_err(StateDeltaMergeError::Error)?;
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
        if let Some(current) = current.as_ref() {
            let close_params = HashMap::from([
                ("id".to_string(), Value::Uuid(current.id.as_uuid())),
                ("new_valid_to".to_string(), Value::TxId(tx)),
            ]);
            let closed = self
                .db
                .execute_in_tx(
                    tx,
                    "UPDATE entity_snapshots SET valid_to = $new_valid_to WHERE id = $id AND valid_to IS NULL",
                    &close_params,
                )
                .map_err(map_engine_err)
                .map_err(StateDeltaMergeError::Error)?
                .rows_affected;
            if closed != 1 {
                return Err(StateDeltaMergeError::Retry);
            }
        }

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

    fn insert_embedding_handle_and_vector_in_tx(
        &self,
        tx: crate::TxId,
        route: &EmbeddingRoute,
        row_id: RowId,
    ) -> Result<()> {
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
                "target_kind".to_string(),
                Value::Text(embedding_target_kind(&route.target).to_string()),
            ),
            (
                "target_id".to_string(),
                Value::Uuid(embedding_target_uuid(&route.target)),
            ),
            (
                "dimension".to_string(),
                Value::Int64(route.dimension as i64),
            ),
            (
                "produced_by".to_string(),
                producer_value(&route.produced_by),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        self.db
            .execute_in_tx(
                tx,
                "INSERT INTO embedding_handles (id, context_id, embedding_space_id, target_kind, target_id, dimension, produced_by, created_at) VALUES ($id, $context_id, $embedding_space_id, $target_kind, $target_id, $dimension, $produced_by, $created_at)",
                &params,
            )
            .map_err(map_engine_err)?;
        self.db
            .insert_vector(
                tx,
                VectorIndexRef::new(&route.table_name, &route.vector_column),
                row_id,
                route.vector.clone(),
            )
            .map_err(map_engine_err)
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

#[derive(Debug, Clone)]
pub(super) struct EmbeddingRoute {
    pub id: EmbeddingId,
    pub context_id: ContextId,
    pub embedding_space_id: String,
    pub vector: Vec<f32>,
    pub target: EmbeddingTarget,
    pub produced_by: EvidenceProducer,
    pub dimension: usize,
    pub table_name: String,
    pub vector_column: String,
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
