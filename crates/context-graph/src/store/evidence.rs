use std::collections::{BTreeMap, BTreeSet, HashMap};

use chrono::{DateTime, Utc};
use contextdb_core::{Error as EngineError, SnapshotId, Value};
use serde_json::Value as JsonValue;

use crate::Result;
use crate::error::CgError;
use crate::types::*;

use super::Store;
use super::convert::{decision_status_from_str, map_engine_err};
use super::encode::{
    column_index, get_json, get_optional_text, get_optional_txid, get_text, get_txid, get_uuid,
    properties_from_json, properties_to_json,
};
use super::{
    distance_metric_from_str, evidence_kind_from_str, retention_status_from_str,
    retention_status_str,
};

impl Store {
    pub fn list_evidence_for_observation(
        &self,
        observation_id: ObservationId,
    ) -> Result<Vec<EvidenceRef>> {
        let params = HashMap::from([(
            "observation_id".to_string(),
            Value::Uuid(observation_id.as_uuid()),
        )]);
        let result = self
            .db
            .execute(
                evidence_select_sql("WHERE observation_id = $observation_id"),
                &params,
            )
            .map_err(map_engine_err)?;
        let mut rows = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            rows.push(self.decode_evidence_with_latest_status(&result.columns, row)?);
        }
        rows.sort_by_key(|row| supply_order_for_evidence_id(&result, row.id).unwrap_or(i64::MAX));
        Ok(rows)
    }

    pub fn get_evidence(&self, id: EvidenceId) -> Result<EvidenceRef> {
        let params = HashMap::from([("id".to_string(), Value::Uuid(id.as_uuid()))]);
        let result = self
            .db
            .execute(evidence_select_sql("WHERE id = $id"), &params)
            .map_err(map_engine_err)?;
        let Some(row) = result.rows.first() else {
            return Err(CgError::NotFound {
                id: Some(id.as_uuid().to_string()),
                missing: Some("evidence"),
            });
        };
        self.decode_evidence_with_latest_status(&result.columns, row)
    }

    pub fn update_evidence_retention_status(
        &self,
        evidence_id: EvidenceId,
        status: RetentionStatus,
    ) -> Result<()> {
        let _ = self.get_evidence(evidence_id)?;
        let tx = self.db.begin();
        let params = HashMap::from([
            ("id".to_string(), Value::Uuid(uuid::Uuid::now_v7())),
            (
                "evidence_id".to_string(),
                Value::Uuid(evidence_id.as_uuid()),
            ),
            (
                "status".to_string(),
                Value::Text(retention_status_str(status).to_string()),
            ),
            ("recorded_at_txid".to_string(), Value::TxId(tx)),
            (
                "recorded_at".to_string(),
                Value::Timestamp(Utc::now().timestamp_millis()),
            ),
        ]);
        if let Err(err) = self.db.execute_in_tx(
            tx,
            "INSERT INTO evidence_availability_log (id, evidence_id, status, recorded_at_txid, recorded_at) VALUES ($id, $evidence_id, $status, $recorded_at_txid, $recorded_at)",
            &params,
        ) {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(err));
        }
        self.db.commit(tx).map_err(map_engine_err)
    }

    pub fn evidence_availability_log(
        &self,
        evidence_id: EvidenceId,
    ) -> Result<Vec<EvidenceAvailabilityLogRow>> {
        let params = HashMap::from([(
            "evidence_id".to_string(),
            Value::Uuid(evidence_id.as_uuid()),
        )]);
        let result = self
            .db
            .execute(
                "SELECT evidence_id, status, recorded_at_txid FROM evidence_availability_log WHERE evidence_id = $evidence_id",
                &params,
            )
            .map_err(map_engine_err)?;
        let mut rows = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            rows.push(EvidenceAvailabilityLogRow {
                evidence_id: EvidenceId::from(get_uuid(row, &result.columns, "evidence_id")?),
                status: retention_status_from_str(&get_text(row, &result.columns, "status")?)?,
                recorded_at_txid: get_txid(row, &result.columns, "recorded_at_txid")?,
            });
        }
        rows.sort_by_key(|row| row.recorded_at_txid);
        Ok(rows)
    }

    pub fn dereference_evidence(&self, evidence_id: EvidenceId) -> Result<EvidenceHandle> {
        let evidence = self.get_evidence(evidence_id)?;
        if evidence.retention_status == RetentionStatus::Unavailable {
            return Err(CgError::EvidenceUnavailable);
        }
        Ok(EvidenceHandle {
            evidence_id,
            source_ref: evidence.source_ref,
            byte_range: evidence.byte_range,
            time_range_ms: evidence.time_range_ms,
        })
    }

    pub fn list_embedding_spaces(&self) -> Result<Vec<EmbeddingSpace>> {
        let result = self
            .db
            .execute(
                "SELECT embedding_space_id, dimension, metric, model_family, model_name, model_version, supported_evidence_kinds, default_for_evidence_kind FROM embedding_spaces",
                &HashMap::new(),
            )
            .map_err(map_engine_err)?;
        let mut spaces = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            let embedding_space_id = get_text(row, &result.columns, "embedding_space_id")?;
            spaces.push(EmbeddingSpace {
                embedding_space_id: embedding_space_id.clone(),
                dimension: get_int(row, &result.columns, "dimension")? as usize,
                metric: distance_metric_from_str(&get_text(row, &result.columns, "metric")?)?,
                model_family: get_optional_text(row, &result.columns, "model_family")?
                    .unwrap_or_default(),
                model_name: get_optional_text(row, &result.columns, "model_name")?
                    .unwrap_or_default(),
                model_version: get_optional_text(row, &result.columns, "model_version")?
                    .unwrap_or_default(),
                supported_evidence_kinds: evidence_kind_vec_from_json(&get_json(
                    row,
                    &result.columns,
                    "supported_evidence_kinds",
                )?)?,
                default_for_evidence_kind: get_optional_text(
                    row,
                    &result.columns,
                    "default_for_evidence_kind",
                )?
                .map(|raw| evidence_kind_from_str(&raw))
                .transpose()?,
                bindings: self.bindings_for_space(&embedding_space_id)?,
            });
        }
        spaces.sort_by(|a, b| a.embedding_space_id.cmp(&b.embedding_space_id));
        Ok(spaces)
    }

    pub fn basis_diff(&self, decision_id: DecisionId) -> Result<BasisDiffResult> {
        let snapshot = self.db.snapshot();
        let read_txid = crate::TxId::from_snapshot(snapshot);
        let params = HashMap::from([("id".to_string(), Value::Uuid(decision_id.as_uuid()))]);
        let decision_rows = self
            .db
            .execute_at_snapshot(
                "SELECT id, context_id FROM decisions WHERE id = $id",
                &params,
                snapshot,
            )
            .map_err(map_engine_err)?;
        let Some(decision_row) = decision_rows.rows.first() else {
            return Err(CgError::NotFound {
                id: Some(decision_id.as_uuid().to_string()),
                missing: Some("decision"),
            });
        };
        let decision_context = ContextId::from(get_uuid(
            decision_row,
            &decision_rows.columns,
            "context_id",
        )?);
        let basis_ids = self.basis_snapshot_ids_for_decision_at_read(decision_id, snapshot)?;
        let mut divergences = Vec::new();
        for basis_id in basis_ids {
            let Some(basis) = self.snapshot_by_id_at_read(basis_id, snapshot)? else {
                return Err(CgError::Engine(format!(
                    "data-integrity: decision {} references missing basis snapshot {}",
                    decision_id, basis_id
                )));
            };
            let basis_context = self
                .entity_context_at_read(basis.entity_id, snapshot)?
                .ok_or_else(|| {
                    CgError::Engine(format!(
                        "data-integrity: basis snapshot {} references missing entity {}",
                        basis.id, basis.entity_id
                    ))
                })?;
            if basis_context != decision_context {
                return Err(CgError::CrossContextViolation);
            }
            let current = self.current_snapshot_for_entity_at_read(basis.entity_id, snapshot)?;
            let value_diffs = value_diffs(
                Some(&basis.properties),
                current.as_ref().map(|snapshot| &snapshot.properties),
            );
            if value_diffs.is_empty() {
                continue;
            }
            divergences.push(EntityDivergence {
                basis_snapshot_id: basis.id,
                entity_id: basis.entity_id,
                value_diffs,
                read_txid,
            });
        }
        divergences.sort_by(|a, b| {
            a.entity_id
                .cmp(&b.entity_id)
                .then_with(|| a.basis_snapshot_id.cmp(&b.basis_snapshot_id))
        });
        Ok(BasisDiffResult {
            is_invalidated_candidate: !divergences.is_empty(),
            divergences,
            read_txid,
            current_snapshot_read_txid: read_txid,
        })
    }

    pub fn invalidate_decision(
        &self,
        decision_id: DecisionId,
        trigger_observation_id: ObservationId,
        reason: impl Into<String>,
    ) -> Result<DecisionInvalidationResult> {
        let reason = reason.into();
        let decision = self.decision_status_and_context(decision_id)?;
        if decision.status != DecisionStatus::Active {
            return Err(CgError::IllegalTransition {
                from: decision.status,
                to: DecisionStatus::Invalidated,
            });
        }
        let trigger_context = self.observation_context(trigger_observation_id)?;
        if trigger_context != decision.context_id {
            return Err(CgError::CrossContextViolation);
        }
        let diff = self.basis_diff(decision_id)?;
        let basis_diff_json = serde_json::to_value(&diff).unwrap_or_else(|_| serde_json::json!({}));

        let tx = self.db.begin();
        let invalidation_id = InvalidationId::from(decision_id.as_uuid());
        let update = self.db.execute_in_tx(
            tx,
            "UPDATE decisions SET status = $new_status WHERE id = $id AND status = $old_status",
            &HashMap::from([
                ("id".to_string(), Value::Uuid(decision_id.as_uuid())),
                (
                    "new_status".to_string(),
                    Value::Text("invalidated".to_string()),
                ),
                ("old_status".to_string(), Value::Text("active".to_string())),
            ]),
        );
        let rows_affected = match update {
            Ok(result) => result.rows_affected,
            Err(err) => {
                let _ = self.db.rollback(tx);
                if matches!(err, EngineError::UniqueViolation { .. }) {
                    return Err(CgError::IllegalTransition {
                        from: DecisionStatus::Invalidated,
                        to: DecisionStatus::Invalidated,
                    });
                }
                return Err(map_engine_err(err));
            }
        };
        if rows_affected != 1 {
            let _ = self.db.rollback(tx);
            let current = self.decision_status_and_context(decision_id)?.status;
            return Err(CgError::IllegalTransition {
                from: current,
                to: DecisionStatus::Invalidated,
            });
        }

        let params = HashMap::from([
            ("id".to_string(), Value::Uuid(invalidation_id.as_uuid())),
            ("status".to_string(), Value::Text("pending".to_string())),
            (
                "context_id".to_string(),
                Value::Uuid(decision.context_id.as_uuid()),
            ),
            (
                "affected_decision_id".to_string(),
                Value::Uuid(decision_id.as_uuid()),
            ),
            (
                "trigger_observation_id".to_string(),
                Value::Uuid(trigger_observation_id.as_uuid()),
            ),
            ("reason".to_string(), Value::Text(reason.clone())),
            ("basis_diff".to_string(), Value::Json(basis_diff_json)),
            ("severity".to_string(), Value::Text("warning".to_string())),
            ("detected_at".to_string(), Value::TxId(tx)),
            ("resolved_at".to_string(), Value::Null),
            ("resolution_action".to_string(), Value::Null),
            ("resolution_decision_id".to_string(), Value::Null),
        ]);
        let inserted = match self.db.execute_in_tx(
            tx,
            "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $reason, $basis_diff, $severity, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
            &params,
        ) {
            Ok(result) => result.rows_affected,
            Err(err) => {
                let _ = self.db.rollback(tx);
                if matches!(err, EngineError::UniqueViolation { .. }) {
                    return Err(CgError::IllegalTransition {
                        from: DecisionStatus::Invalidated,
                        to: DecisionStatus::Invalidated,
                    });
                }
                return Err(map_engine_err(err));
            }
        };
        if inserted != 1 {
            let _ = self.db.rollback(tx);
            return Err(CgError::IllegalTransition {
                from: DecisionStatus::Invalidated,
                to: DecisionStatus::Invalidated,
            });
        }
        if let Err(err) = self.db.commit(tx) {
            if matches!(err, EngineError::UniqueViolation { .. }) {
                return Err(CgError::IllegalTransition {
                    from: DecisionStatus::Invalidated,
                    to: DecisionStatus::Invalidated,
                });
            }
            return Err(map_engine_err(err));
        }
        Ok(DecisionInvalidationResult {
            previous_status: DecisionStatus::Active,
            new_status: DecisionStatus::Invalidated,
            reason: Some(reason),
            invalidation_id,
            transitioned_at: tx,
        })
    }

    pub fn list_invalidations(
        &self,
        context_id: ContextId,
        decision_id: Option<DecisionId>,
        status: Option<InvalidationStatus>,
    ) -> Result<Vec<Invalidation>> {
        let mut clauses = vec!["context_id = $context_id"];
        let mut params =
            HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]);
        if let Some(decision_id) = decision_id {
            match self.decision_context_id(decision_id)? {
                Some(decision_context) if decision_context == context_id => {}
                Some(_) | None => return Ok(Vec::new()),
            }
            clauses.push("affected_decision_id = $affected_decision_id");
            params.insert(
                "affected_decision_id".to_string(),
                Value::Uuid(decision_id.as_uuid()),
            );
        }
        if let Some(status) = status {
            clauses.push("status = $status");
            params.insert(
                "status".to_string(),
                Value::Text(invalidation_status_str(status).to_string()),
            );
        }
        let where_clause = if clauses.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", clauses.join(" AND "))
        };
        let sql = format!(
            "SELECT id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id FROM invalidations{where_clause}"
        );
        let result = self.db.execute(&sql, &params).map_err(map_engine_err)?;
        let mut rows = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            rows.push(decode_invalidation(&result.columns, row)?);
        }
        rows.sort_by_key(|row| row.detected_at);
        Ok(rows)
    }

    pub fn update_decision_basis(
        &self,
        decision_id: DecisionId,
        basis: Vec<EntitySnapshotId>,
    ) -> Result<()> {
        let decision = self.decision_status_and_context(decision_id)?;
        self.validate_basis_snapshots_for_context(decision.context_id, &basis)?;
        let basis_snapshots = self.basis_snapshots_for_basis(&basis)?;
        for attempt in 0..8 {
            let tx = self.db.begin();
            if let Err(err) = self.replace_decision_basis_snapshots_in_tx(
                tx,
                decision.context_id,
                decision_id,
                &basis_snapshots,
            ) {
                let _ = self.db.rollback(tx);
                if decision_basis_retry_unique(&err) && attempt < 7 {
                    std::thread::yield_now();
                    continue;
                }
                return Err(err);
            }
            match self.db.commit(tx).map_err(map_engine_err) {
                Ok(()) => return Ok(()),
                Err(err) if decision_basis_retry_unique(&err) && attempt < 7 => {
                    std::thread::yield_now();
                    continue;
                }
                Err(err) => return Err(err),
            }
        }
        unreachable!("bounded update_decision_basis retry loop always returns")
    }

    fn bindings_for_space(&self, embedding_space_id: &str) -> Result<Vec<EmbeddingSpaceBinding>> {
        let result = self
            .db
            .execute(
                "SELECT embedding_space_id, table_name, vector_column FROM embedding_space_bindings WHERE embedding_space_id = $embedding_space_id",
                &HashMap::from([(
                    "embedding_space_id".to_string(),
                    Value::Text(embedding_space_id.to_string()),
                )]),
            )
            .map_err(map_engine_err)?;
        let mut bindings = Vec::with_capacity(result.rows.len());
        for row in &result.rows {
            bindings.push(EmbeddingSpaceBinding {
                embedding_space_id: get_text(row, &result.columns, "embedding_space_id")?,
                table_name: get_text(row, &result.columns, "table_name")?,
                vector_column: get_text(row, &result.columns, "vector_column")?,
            });
        }
        bindings.sort_by(|a, b| {
            a.table_name
                .cmp(&b.table_name)
                .then_with(|| a.vector_column.cmp(&b.vector_column))
        });
        Ok(bindings)
    }

    fn decode_evidence_with_latest_status(
        &self,
        columns: &[String],
        row: &[Value],
    ) -> Result<EvidenceRef> {
        let mut evidence = decode_evidence(columns, row)?;
        if let Some(status) = self.latest_evidence_status(evidence.id)? {
            evidence.retention_status = status;
        }
        Ok(evidence)
    }

    fn latest_evidence_status(&self, evidence_id: EvidenceId) -> Result<Option<RetentionStatus>> {
        let rows = self.evidence_availability_log(evidence_id)?;
        Ok(rows.last().map(|row| row.status))
    }

    pub(super) fn snapshot_by_id_at_read(
        &self,
        snapshot_id: EntitySnapshotId,
        snapshot: SnapshotId,
    ) -> Result<Option<EntitySnapshot>> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT id, entity_id, data, valid_from, valid_to FROM entity_snapshots WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(snapshot_id.as_uuid()))]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        let decoded = result
            .rows
            .first()
            .map(|row| decode_snapshot_at_read(&result.columns, row))
            .transpose()?;
        match decoded {
            Some(mut row) => {
                row.evidenced_by =
                    self.snapshot_observations_at_read(row.entity_id, row.id, snapshot)?;
                Ok(Some(row))
            }
            None => Ok(None),
        }
    }

    pub(super) fn current_snapshot_for_entity_at_read(
        &self,
        entity_id: EntityId,
        snapshot: SnapshotId,
    ) -> Result<Option<EntitySnapshot>> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT snapshot_id FROM entity_current_snapshots WHERE entity_id = $entity_id",
                &HashMap::from([("entity_id".to_string(), Value::Uuid(entity_id.as_uuid()))]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        let Some(row) = result.rows.first() else {
            return Ok(None);
        };
        let snapshot_id = EntitySnapshotId::from(get_uuid(row, &result.columns, "snapshot_id")?);
        self.snapshot_by_id_at_read(snapshot_id, snapshot)
    }

    pub(super) fn entity_context_at_read(
        &self,
        entity_id: EntityId,
        snapshot: SnapshotId,
    ) -> Result<Option<ContextId>> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT context_id FROM entities WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(entity_id.as_uuid()))]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| get_uuid(row, &result.columns, "context_id").map(ContextId::from))
            .transpose()
    }

    pub(super) fn validate_basis_snapshots_for_context(
        &self,
        context_id: ContextId,
        basis: &[EntitySnapshotId],
    ) -> Result<()> {
        for snapshot_id in basis {
            let result = self
                .db
                .execute(
                    "SELECT entity_id FROM entity_snapshots WHERE id = $id",
                    &HashMap::from([("id".to_string(), Value::Uuid(snapshot_id.as_uuid()))]),
                )
                .map_err(map_engine_err)?;
            let Some(row) = result.rows.first() else {
                return Err(CgError::NotFound {
                    id: Some(snapshot_id.as_uuid().to_string()),
                    missing: Some("entity_snapshot"),
                });
            };
            let entity_id = EntityId::from(get_uuid(row, &result.columns, "entity_id")?);
            let entity_context = self.entity_context(entity_id)?.ok_or(CgError::NotFound {
                id: Some(entity_id.as_uuid().to_string()),
                missing: Some("entity"),
            })?;
            if entity_context != context_id {
                return Err(CgError::CrossContextViolation);
            }
        }
        Ok(())
    }

    pub(super) fn entity_context(&self, entity_id: EntityId) -> Result<Option<ContextId>> {
        let result = self
            .db
            .execute(
                "SELECT context_id FROM entities WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(entity_id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        result
            .rows
            .first()
            .map(|row| get_uuid(row, &result.columns, "context_id").map(ContextId::from))
            .transpose()
    }

    fn decision_status_and_context(
        &self,
        decision_id: DecisionId,
    ) -> Result<DecisionStatusAndContext> {
        let result = self
            .db
            .execute(
                "SELECT status, context_id FROM decisions WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(decision_id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        let Some(row) = result.rows.first() else {
            return Err(CgError::NotFound {
                id: Some(decision_id.as_uuid().to_string()),
                missing: Some("decision"),
            });
        };
        Ok(DecisionStatusAndContext {
            status: decision_status_from_str(&get_text(row, &result.columns, "status")?)?,
            context_id: ContextId::from(get_uuid(row, &result.columns, "context_id")?),
        })
    }

    fn observation_context(&self, observation_id: ObservationId) -> Result<ContextId> {
        let result = self
            .db
            .execute(
                "SELECT context_id FROM observations WHERE id = $id",
                &HashMap::from([("id".to_string(), Value::Uuid(observation_id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        let Some(row) = result.rows.first() else {
            return Err(CgError::NotFound {
                id: Some(observation_id.as_uuid().to_string()),
                missing: Some("observation"),
            });
        };
        Ok(ContextId::from(get_uuid(
            row,
            &result.columns,
            "context_id",
        )?))
    }

    pub(super) fn embedding_space_for_id(&self, id: &str) -> Result<EmbeddingSpace> {
        self.list_embedding_spaces()?
            .into_iter()
            .find(|space| space.embedding_space_id == id)
            .ok_or(CgError::EmbeddingSpaceMismatch {
                conflicting_field: "embedding_space_id",
            })
    }

    pub(super) fn evidence_status_at_snapshot(
        &self,
        evidence_id: EvidenceId,
        snapshot: SnapshotId,
    ) -> Result<Option<RetentionStatus>> {
        let result = self
            .db
            .execute_at_snapshot(
                "SELECT status, recorded_at_txid FROM evidence_availability_log WHERE evidence_id = $evidence_id",
                &HashMap::from([("evidence_id".to_string(), Value::Uuid(evidence_id.as_uuid()))]),
                snapshot,
            )
            .map_err(map_engine_err)?;
        let mut rows = Vec::new();
        for row in &result.rows {
            rows.push((
                get_txid(row, &result.columns, "recorded_at_txid")?,
                retention_status_from_str(&get_text(row, &result.columns, "status")?)?,
            ));
        }
        rows.sort_by_key(|row| row.0);
        Ok(rows.last().map(|row| row.1))
    }
}

struct DecisionStatusAndContext {
    status: DecisionStatus,
    context_id: ContextId,
}

fn evidence_select_sql(where_clause: &str) -> &'static str {
    match where_clause {
        "WHERE id = $id" => {
            "SELECT id, observation_id, context_id, kind, source_ref, mime_type, content_hash, captured_at, captured_at_nanos, byte_range, time_range_ms, frame_index, region, producer, text_projection, derived_from, retention_status, media_retention_until, media_retention_until_nanos, properties, supply_order FROM evidence WHERE id = $id"
        }
        "WHERE observation_id = $observation_id" => {
            "SELECT id, observation_id, context_id, kind, source_ref, mime_type, content_hash, captured_at, captured_at_nanos, byte_range, time_range_ms, frame_index, region, producer, text_projection, derived_from, retention_status, media_retention_until, media_retention_until_nanos, properties, supply_order FROM evidence WHERE observation_id = $observation_id"
        }
        _ => unreachable!("unsupported evidence where clause"),
    }
}

pub(super) fn decode_evidence(columns: &[String], row: &[Value]) -> Result<EvidenceRef> {
    Ok(EvidenceRef {
        id: EvidenceId::from(get_uuid(row, columns, "id")?),
        observation_id: ObservationId::from(get_uuid(row, columns, "observation_id")?),
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        kind: evidence_kind_from_str(&get_text(row, columns, "kind")?)?,
        source_ref: get_text(row, columns, "source_ref")?,
        mime_type: get_optional_text(row, columns, "mime_type")?,
        content_hash: get_optional_text(row, columns, "content_hash")?,
        captured_at: optional_datetime(row, columns, "captured_at", "captured_at_nanos")?,
        byte_range: optional_pair(row, columns, "byte_range")?,
        time_range_ms: optional_pair(row, columns, "time_range_ms")?,
        frame_index: optional_u64(row, columns, "frame_index")?,
        region: optional_json(row, columns, "region")?,
        producer: serde_json::from_value(get_json(row, columns, "producer")?).unwrap_or_default(),
        text_projection: get_optional_text(row, columns, "text_projection")?,
        derived_from: optional_uuid(row, columns, "derived_from")?.map(EvidenceId::from),
        retention_status: retention_status_from_str(&get_text(row, columns, "retention_status")?)?,
        media_retention_until: optional_datetime(
            row,
            columns,
            "media_retention_until",
            "media_retention_until_nanos",
        )?,
        properties: properties_from_json(&get_json(row, columns, "properties")?),
    })
}

fn supply_order_for_evidence_id(
    result: &contextdb_engine::QueryResult,
    evidence_id: EvidenceId,
) -> Option<i64> {
    let id_idx = result.columns.iter().position(|column| column == "id")?;
    let order_idx = result
        .columns
        .iter()
        .position(|column| column == "supply_order")?;
    result
        .rows
        .iter()
        .find_map(|row| match (row.get(id_idx), row.get(order_idx)) {
            (Some(Value::Uuid(id)), Some(Value::Int64(order))) if *id == evidence_id.as_uuid() => {
                Some(*order)
            }
            _ => None,
        })
}

fn optional_uuid(row: &[Value], columns: &[String], name: &str) -> Result<Option<uuid::Uuid>> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Uuid(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected uuid|null, got {other:?}"
        ))),
    }
}

fn optional_timestamp(
    row: &[Value],
    columns: &[String],
    name: &str,
) -> Result<Option<DateTime<Utc>>> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Timestamp(value)) => Ok(DateTime::<Utc>::from_timestamp_millis(*value)),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected timestamp|null, got {other:?}"
        ))),
    }
}

fn optional_datetime(
    row: &[Value],
    columns: &[String],
    millis_name: &str,
    nanos_name: &str,
) -> Result<Option<DateTime<Utc>>> {
    if let Some(idx) = columns.iter().position(|column| column == nanos_name) {
        match row.get(idx) {
            Some(Value::Int64(nanos)) => {
                let seconds = nanos.div_euclid(1_000_000_000);
                let sub_nanos = nanos.rem_euclid(1_000_000_000) as u32;
                return DateTime::<Utc>::from_timestamp(seconds, sub_nanos)
                    .map(Some)
                    .ok_or_else(|| {
                        CgError::Engine(format!("column {nanos_name}: invalid nanos {nanos}"))
                    });
            }
            Some(Value::Null) | None => {}
            other => {
                return Err(CgError::Engine(format!(
                    "column {nanos_name}: expected int|null, got {other:?}"
                )));
            }
        }
    }
    optional_timestamp(row, columns, millis_name)
}

fn optional_u64(row: &[Value], columns: &[String], name: &str) -> Result<Option<u64>> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Int64(value)) => Ok(Some(*value as u64)),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected int|null, got {other:?}"
        ))),
    }
}

fn optional_pair(row: &[Value], columns: &[String], name: &str) -> Result<Option<(u64, u64)>> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Json(JsonValue::Array(values))) if values.len() == 2 => {
            let start = values[0].as_u64().unwrap_or_default();
            let end = values[1].as_u64().unwrap_or_default();
            Ok(Some((start, end)))
        }
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected pair json|null, got {other:?}"
        ))),
    }
}

fn optional_json(row: &[Value], columns: &[String], name: &str) -> Result<Option<JsonValue>> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Json(value)) => Ok(Some(value.clone())),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected json|null, got {other:?}"
        ))),
    }
}

fn get_int(row: &[Value], columns: &[String], name: &str) -> Result<i64> {
    let idx = column_index(columns, name)?;
    match row.get(idx) {
        Some(Value::Int64(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected int, got {other:?}"
        ))),
    }
}

fn evidence_kind_vec_from_json(value: &JsonValue) -> Result<Vec<EvidenceKind>> {
    match value {
        JsonValue::Array(items) => items
            .iter()
            .filter_map(JsonValue::as_str)
            .map(evidence_kind_from_str)
            .collect(),
        JsonValue::Null => Ok(Vec::new()),
        other => Err(CgError::Engine(format!(
            "expected evidence kind array, got {other:?}"
        ))),
    }
}

fn decode_snapshot_at_read(columns: &[String], row: &[Value]) -> Result<EntitySnapshot> {
    let data = get_json(row, columns, "data")?;
    let entity_type = data
        .get("entity_type")
        .and_then(JsonValue::as_str)
        .map(super::convert::entity_type_from_str)
        .transpose()?
        .unwrap_or(EntityType::Service);
    let valid_from = get_txid(row, columns, "valid_from")?;
    Ok(EntitySnapshot {
        id: EntitySnapshotId::from(get_uuid(row, columns, "id")?),
        entity_id: EntityId::from(get_uuid(row, columns, "entity_id")?),
        entity_type,
        name: data
            .get("name")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .to_string(),
        properties: data
            .get("properties")
            .map(properties_from_json)
            .unwrap_or_default(),
        evidenced_by: Vec::new(),
        captured_at: valid_from,
        valid_from,
        valid_to: get_optional_txid(row, columns, "valid_to")?,
    })
}

fn value_diffs(
    basis: Option<&BTreeMap<String, JsonValue>>,
    current: Option<&BTreeMap<String, JsonValue>>,
) -> Vec<ValueDiff> {
    let mut fields = BTreeSet::new();
    if let Some(values) = basis {
        fields.extend(values.keys().cloned());
    }
    if let Some(values) = current {
        fields.extend(values.keys().cloned());
    }
    let mut diffs = Vec::new();
    for field in fields {
        let basis_value = basis.and_then(|values| values.get(&field).cloned());
        let current_value = current.and_then(|values| values.get(&field).cloned());
        if basis_value != current_value {
            diffs.push(ValueDiff {
                field,
                basis_value,
                current_value,
            });
        }
    }
    diffs
}

fn decision_basis_retry_unique(err: &CgError) -> bool {
    matches!(
        err,
        CgError::UniqueViolation {
            entity: "decision_basis_snapshot",
            ..
        }
    )
}

fn decode_invalidation(columns: &[String], row: &[Value]) -> Result<Invalidation> {
    Ok(Invalidation {
        id: InvalidationId::from(get_uuid(row, columns, "id")?),
        status: invalidation_status_from_str(&get_text(row, columns, "status")?)?,
        context_id: ContextId::from(get_uuid(row, columns, "context_id")?),
        affected_decision_id: DecisionId::from(get_uuid(row, columns, "affected_decision_id")?),
        trigger_observation_id: ObservationId::from(get_uuid(
            row,
            columns,
            "trigger_observation_id",
        )?),
        reason: get_optional_text(row, columns, "reason")?,
        basis_diff: properties_from_json(&get_json(row, columns, "basis_diff")?),
        severity: invalidation_severity_from_str(&get_text(row, columns, "severity")?)?,
        detected_at: get_txid(row, columns, "detected_at")?,
        resolved_at: get_optional_txid(row, columns, "resolved_at")?,
        resolution_action: get_optional_text(row, columns, "resolution_action")?,
        resolution_decision_id: optional_uuid(row, columns, "resolution_decision_id")?
            .map(DecisionId::from),
    })
}

fn invalidation_status_str(status: InvalidationStatus) -> &'static str {
    match status {
        InvalidationStatus::Pending => "pending",
        InvalidationStatus::Acknowledged => "acknowledged",
        InvalidationStatus::Resolved => "resolved",
        InvalidationStatus::Dismissed => "dismissed",
    }
}

fn invalidation_status_from_str(raw: &str) -> Result<InvalidationStatus> {
    match raw {
        "pending" => Ok(InvalidationStatus::Pending),
        "acknowledged" => Ok(InvalidationStatus::Acknowledged),
        "resolved" => Ok(InvalidationStatus::Resolved),
        "dismissed" => Ok(InvalidationStatus::Dismissed),
        other => Err(CgError::Engine(format!(
            "unknown invalidation status: {other}"
        ))),
    }
}

fn invalidation_severity_from_str(raw: &str) -> Result<InvalidationSeverity> {
    match raw {
        "info" => Ok(InvalidationSeverity::Info),
        "warning" => Ok(InvalidationSeverity::Warning),
        "critical" => Ok(InvalidationSeverity::Critical),
        other => Err(CgError::Engine(format!(
            "unknown invalidation severity: {other}"
        ))),
    }
}

pub(super) fn value_for_optional_text(value: &Option<String>) -> Value {
    value
        .as_ref()
        .map(|value| Value::Text(value.clone()))
        .unwrap_or(Value::Null)
}

pub(super) fn value_for_optional_timestamp(value: &Option<DateTime<Utc>>) -> Value {
    value
        .map(|value| Value::Timestamp(value.timestamp_millis()))
        .unwrap_or(Value::Null)
}

pub(super) fn value_for_optional_timestamp_nanos(value: &Option<DateTime<Utc>>) -> Value {
    value
        .and_then(|value| value.timestamp_nanos_opt())
        .map(Value::Int64)
        .unwrap_or(Value::Null)
}

pub(super) fn value_for_optional_pair(value: Option<(u64, u64)>) -> Value {
    value
        .map(|(start, end)| Value::Json(serde_json::json!([start, end])))
        .unwrap_or(Value::Null)
}

pub(super) fn value_for_optional_u64(value: Option<u64>) -> Value {
    value
        .map(|value| Value::Int64(value as i64))
        .unwrap_or(Value::Null)
}

pub(super) fn value_for_optional_json(value: &Option<JsonValue>) -> Value {
    value
        .as_ref()
        .map(|value| Value::Json(value.clone()))
        .unwrap_or(Value::Null)
}

pub(super) fn value_for_optional_evidence_id(value: Option<EvidenceId>) -> Value {
    value
        .map(|value| Value::Uuid(value.as_uuid()))
        .unwrap_or(Value::Null)
}

pub(super) fn embedding_target_kind(target: &EmbeddingTarget) -> &'static str {
    match target {
        EmbeddingTarget::Evidence(_) => "evidence",
        EmbeddingTarget::Digest(_) => "digest",
    }
}

pub(super) fn embedding_target_uuid(target: &EmbeddingTarget) -> uuid::Uuid {
    match target {
        EmbeddingTarget::Evidence(id) => id.as_uuid(),
        EmbeddingTarget::Digest(id) => id.as_uuid(),
    }
}

pub(super) fn vector_binding_for_target(
    space: &EmbeddingSpace,
    target_kind: &str,
) -> Result<EmbeddingSpaceBinding> {
    let table = match target_kind {
        "evidence" => "evidence",
        "digest" => "digests",
        _ => "",
    };
    space
        .bindings
        .iter()
        .find(|binding| binding.table_name == table)
        .cloned()
        .ok_or(CgError::EmbeddingSpaceMismatch {
            conflicting_field: "bindings",
        })
}

pub(super) fn validate_space_kind(space: &EmbeddingSpace, kind: EvidenceKind) -> Result<()> {
    if !space.supported_evidence_kinds.contains(&kind) {
        return Err(CgError::UnsupportedEvidenceKind { kind });
    }
    Ok(())
}

pub(super) fn producer_value(producer: &EvidenceProducer) -> Value {
    Value::Json(serde_json::to_value(producer).unwrap_or_else(|_| serde_json::json!({})))
}

pub(super) fn properties_value(properties: &BTreeMap<String, JsonValue>) -> Value {
    Value::Json(properties_to_json(properties))
}
