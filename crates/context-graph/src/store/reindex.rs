use std::collections::{HashMap, HashSet};

use contextdb_core::{RowId, TxId, Value, VectorIndexRef, VersionedRow};
use contextdb_engine::QueryResult;
use serde::{Deserialize, Serialize};

use crate::embedder::EmbeddingInput;
use crate::{CgError, ContextId, EmbeddingId, EvidenceKind, EvidenceProducer, Result};

use super::convert::map_engine_err;
use super::evidence::producer_value;
use super::{Store, evidence_kind_from_str};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReindexOptions {
    pub interrupt_after: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReindexEvent {
    Completed {
        embedding_space_id: String,
        rows_processed: usize,
        from_producer: Option<String>,
        to_producer: Option<String>,
    },
    Paused {
        embedding_space_id: String,
        last_committed_cursor: usize,
    },
    Resumed {
        embedding_space_id: String,
        resumed_at_cursor: usize,
    },
    TriggerSuggested {
        from_space: String,
        to_space: String,
    },
}

struct ReindexCommit<'a> {
    embedding_space_id: &'a str,
    candidate: &'a ReindexCandidate,
    producer: &'a EvidenceProducer,
    vector: Vec<f32>,
    committed_cursor: usize,
    from_producer: &'a EvidenceProducer,
    to_producer: &'a EvidenceProducer,
    status: ReindexCursorStatus,
}

impl Store {
    pub fn reindex_space(&self, embedding_space_id: &str, options: ReindexOptions) -> Result<()> {
        let space = self
            .embedding_space_for_write(embedding_space_id)
            .map_err(|_| CgError::UnknownEmbeddingSpace {
                id: embedding_space_id.to_string(),
            })?;
        let lease = self.acquire_reindex_lease(embedding_space_id)?;
        let candidates = self.reindex_candidates(&space.embedding_space_id, lease.started_at)?;

        if candidates.is_empty() {
            self.finish_empty_reindex(&space.embedding_space_id, &lease)?;
            return Ok(());
        }

        let from_producer = lease
            .from_producer
            .clone()
            .unwrap_or_else(|| candidates[0].produced_by.clone());
        if lease.resumed_at_cursor > 0 {
            self.push_reindex_event(ReindexEvent::Resumed {
                embedding_space_id: space.embedding_space_id.clone(),
                resumed_at_cursor: lease.resumed_at_cursor,
            });
        }

        let mut to_producer = lease.to_producer.clone();
        let start = lease.resumed_at_cursor.min(candidates.len());
        let mut last_committed_cursor = lease.resumed_at_cursor;
        for (index, candidate) in candidates.iter().enumerate().skip(start) {
            let embedder = match self.embedders.get(&space.embedding_space_id) {
                Some(embedder) => embedder,
                None => {
                    let err = CgError::UnsupportedEvidenceKind {
                        kind: candidate.kind,
                    };
                    self.pause_reindex_after_failure(
                        &space.embedding_space_id,
                        last_committed_cursor,
                        &from_producer,
                        to_producer.as_ref(),
                    )?;
                    return Err(err);
                }
            };
            let output = match embedder.embed(candidate.input.clone()) {
                Ok(output) => output,
                Err(err) => {
                    self.pause_reindex_after_failure(
                        &space.embedding_space_id,
                        last_committed_cursor,
                        &from_producer,
                        to_producer.as_ref(),
                    )?;
                    return Err(err);
                }
            };
            if output.embedding_space_id != space.embedding_space_id {
                let err = CgError::EmbeddingSpaceIdMismatch {
                    embedder_declared: output.embedding_space_id,
                    registry_declared: space.embedding_space_id.clone(),
                };
                self.pause_reindex_after_failure(
                    &space.embedding_space_id,
                    last_committed_cursor,
                    &from_producer,
                    to_producer.as_ref(),
                )?;
                return Err(err);
            }
            if output.dimension != space.dimension || output.vector.len() != space.dimension {
                let err = CgError::EmbeddingDimensionMismatch {
                    space_id: space.embedding_space_id.clone(),
                    expected: space.dimension as u32,
                    found: output.vector.len() as u32,
                };
                self.pause_reindex_after_failure(
                    &space.embedding_space_id,
                    last_committed_cursor,
                    &from_producer,
                    to_producer.as_ref(),
                )?;
                return Err(err);
            }

            if to_producer.is_none() {
                to_producer = Some(output.producer.clone());
            }
            let committed_cursor = index + 1;
            let final_status = if committed_cursor == candidates.len() {
                ReindexCursorStatus::Completed
            } else if options.interrupt_after == Some(committed_cursor) {
                ReindexCursorStatus::Paused
            } else {
                ReindexCursorStatus::InProgress
            };
            self.commit_reindexed_candidate(ReindexCommit {
                embedding_space_id: &space.embedding_space_id,
                candidate,
                producer: &output.producer,
                vector: output.vector,
                committed_cursor,
                from_producer: &from_producer,
                to_producer: to_producer.as_ref().unwrap_or(&output.producer),
                status: final_status,
            })
            .inspect_err(|_| {
                let _ = self.pause_reindex_after_failure(
                    &space.embedding_space_id,
                    last_committed_cursor,
                    &from_producer,
                    to_producer.as_ref(),
                );
            })?;
            last_committed_cursor = committed_cursor;
            if matches!(final_status, ReindexCursorStatus::Paused) {
                self.push_reindex_event(ReindexEvent::Paused {
                    embedding_space_id: space.embedding_space_id.clone(),
                    last_committed_cursor: committed_cursor,
                });
                return Err(CgError::ReindexPaused {
                    embedding_space_id: space.embedding_space_id.clone(),
                    last_committed_cursor: committed_cursor,
                });
            }
        }

        self.push_reindex_event(ReindexEvent::Completed {
            embedding_space_id: space.embedding_space_id.clone(),
            rows_processed: candidates.len(),
            from_producer: Some(producer_json(&from_producer)),
            to_producer: to_producer.as_ref().map(producer_json),
        });
        Ok(())
    }

    pub fn drain_reindex_events(&self) -> Vec<ReindexEvent> {
        self.with_reindex_events(std::mem::take)
    }

    pub(super) fn push_reindex_event(&self, event: ReindexEvent) {
        self.with_reindex_events(|events| events.push(event));
    }

    fn with_reindex_events<R>(&self, f: impl FnOnce(&mut Vec<ReindexEvent>) -> R) -> R {
        let mut events = loop {
            match self.reindex_events.try_lock() {
                Ok(events) => break events,
                Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                    break poisoned.into_inner();
                }
                Err(std::sync::TryLockError::WouldBlock) => std::thread::yield_now(),
            }
        };
        f(&mut events)
    }

    fn acquire_reindex_lease(&self, embedding_space_id: &str) -> Result<ReindexLease> {
        let tx = self.db.begin().map_err(map_engine_err)?;
        let existing = self.reindex_cursor_in_tx(tx, embedding_space_id)?;
        if matches!(
            existing.as_ref().map(|cursor| cursor.status.as_str()),
            Some("in_progress")
        ) {
            let _ = self.db.rollback(tx);
            return Err(CgError::ReindexAlreadyInProgress {
                embedding_space_id: embedding_space_id.to_string(),
            });
        }

        let owner_run_id = format!("reindex-run-{}", uuid::Uuid::now_v7());
        let paused_cursor = existing.as_ref().filter(|cursor| cursor.status == "paused");
        let resumed_at_cursor = paused_cursor
            .map(|cursor| cursor.last_committed_cursor)
            .unwrap_or(0);
        let started_at = paused_cursor.map(|cursor| cursor.started_at).unwrap_or(tx);
        let from_producer = paused_cursor.and_then(|cursor| cursor.from_producer.clone());
        let to_producer = paused_cursor.and_then(|cursor| cursor.to_producer.clone());
        let next_lease_epoch = existing
            .as_ref()
            .map(|cursor| cursor.lease_epoch.saturating_add(1))
            .unwrap_or(1);
        let params = HashMap::from([
            (
                "space".to_string(),
                Value::Text(embedding_space_id.to_string()),
            ),
            ("status".to_string(), Value::Text("in_progress".to_string())),
            (
                "blocked_status".to_string(),
                Value::Text("in_progress".to_string()),
            ),
            ("owner".to_string(), Value::Text(owner_run_id)),
            (
                "lease_epoch".to_string(),
                Value::Int64(next_lease_epoch as i64),
            ),
            ("cursor".to_string(), Value::Int64(resumed_at_cursor as i64)),
            (
                "from_producer".to_string(),
                from_producer
                    .as_ref()
                    .map(producer_value)
                    .unwrap_or(Value::Null),
            ),
            (
                "to_producer".to_string(),
                to_producer
                    .as_ref()
                    .map(producer_value)
                    .unwrap_or(Value::Null),
            ),
            ("started_at".to_string(), Value::TxId(started_at)),
            ("advanced_at".to_string(), Value::TxId(tx)),
        ]);
        let result = if existing.is_some() {
            self.db.execute_in_tx(
                tx,
                "UPDATE reindex_cursors SET status = $status, owner_run_id = $owner, lease_epoch = $lease_epoch, last_committed_cursor = $cursor, from_producer = $from_producer, to_producer = $to_producer, started_at = $started_at, last_advanced_at = $advanced_at WHERE embedding_space_id = $space AND status != $blocked_status",
                &params,
            )
        } else {
            self.db.execute_in_tx(
                tx,
                "INSERT INTO reindex_cursors (embedding_space_id, status, owner_run_id, lease_epoch, last_committed_cursor, from_producer, to_producer, started_at, last_advanced_at) VALUES ($space, $status, $owner, $lease_epoch, $cursor, $from_producer, $to_producer, $started_at, $advanced_at)",
                &params,
            )
        };
        let result = match result {
            Ok(result) => result,
            Err(err) => {
                let _ = self.db.rollback(tx);
                if matches!(err, contextdb_core::Error::UniqueViolation { .. }) {
                    return Err(CgError::ReindexAlreadyInProgress {
                        embedding_space_id: embedding_space_id.to_string(),
                    });
                }
                return Err(map_engine_err(err));
            }
        };
        if result.rows_affected != 1 {
            let _ = self.db.rollback(tx);
            return Err(CgError::ReindexAlreadyInProgress {
                embedding_space_id: embedding_space_id.to_string(),
            });
        }
        self.db.commit(tx).map_err(map_engine_err)?;
        Ok(ReindexLease {
            started_at,
            resumed_at_cursor,
            from_producer,
            to_producer,
        })
    }

    fn finish_empty_reindex(&self, embedding_space_id: &str, lease: &ReindexLease) -> Result<()> {
        let tx = self.db.begin().map_err(map_engine_err)?;
        let params = HashMap::from([
            (
                "space".to_string(),
                Value::Text(embedding_space_id.to_string()),
            ),
            ("status".to_string(), Value::Text("completed".to_string())),
            ("cursor".to_string(), Value::Int64(0)),
            ("advanced_at".to_string(), Value::TxId(tx)),
        ]);
        let result = self.db.execute_in_tx(
            tx,
            "UPDATE reindex_cursors SET status = $status, last_committed_cursor = $cursor, last_advanced_at = $advanced_at WHERE embedding_space_id = $space",
            &params,
        );
        if let Err(err) = result {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(err));
        }
        self.db.commit(tx).map_err(map_engine_err)?;
        if lease.resumed_at_cursor > 0 {
            self.push_reindex_event(ReindexEvent::Resumed {
                embedding_space_id: embedding_space_id.to_string(),
                resumed_at_cursor: lease.resumed_at_cursor,
            });
        }
        self.push_reindex_event(ReindexEvent::Completed {
            embedding_space_id: embedding_space_id.to_string(),
            rows_processed: 0,
            from_producer: None,
            to_producer: None,
        });
        Ok(())
    }

    fn commit_reindexed_candidate(&self, commit: ReindexCommit<'_>) -> Result<()> {
        let tx = self.db.begin().map_err(map_engine_err)?;
        let handle_id = EmbeddingId::new_v7();
        let handle_params = HashMap::from([
            ("id".to_string(), Value::Uuid(handle_id.as_uuid())),
            (
                "context_id".to_string(),
                Value::Uuid(commit.candidate.context_id.as_uuid()),
            ),
            (
                "target_id".to_string(),
                Value::Uuid(commit.candidate.target_id),
            ),
            (
                "embedding_space_id".to_string(),
                Value::Text(commit.embedding_space_id.to_string()),
            ),
            ("produced_by".to_string(), producer_value(commit.producer)),
            (
                "producer_system".to_string(),
                Value::Text(commit.producer.system.clone()),
            ),
            ("created_at".to_string(), Value::TxId(tx)),
        ]);
        let result = self
            .db
            .insert_vector(
                tx,
                VectorIndexRef::new(
                    &commit.candidate.table_name,
                    &commit.candidate.vector_column,
                ),
                commit.candidate.row_id,
                commit.vector,
            )
            .and_then(|_| {
                self.db.execute_in_tx(
                    tx,
                    &format!(
                        "INSERT INTO {} (id, context_id, target_id, embedding_space_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $target_id, $embedding_space_id, $produced_by, $producer_system, $created_at)",
                        commit.candidate.handle_table
                    ),
                    &handle_params,
                )
            })
            .and_then(|_| {
                let cursor_params = HashMap::from([
                    (
                        "space".to_string(),
                        Value::Text(commit.embedding_space_id.to_string()),
                    ),
                    (
                        "status".to_string(),
                        Value::Text(commit.status.as_str().to_string()),
                    ),
                    (
                        "cursor".to_string(),
                        Value::Int64(commit.committed_cursor as i64),
                    ),
                    (
                        "from_producer".to_string(),
                        producer_value(commit.from_producer),
                    ),
                    ("to_producer".to_string(), producer_value(commit.to_producer)),
                    ("advanced_at".to_string(), Value::TxId(tx)),
                ]);
                self.db.execute_in_tx(
                    tx,
                    "UPDATE reindex_cursors SET status = $status, last_committed_cursor = $cursor, from_producer = $from_producer, to_producer = $to_producer, last_advanced_at = $advanced_at WHERE embedding_space_id = $space",
                    &cursor_params,
                )
            });
        if let Err(err) = result {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(err));
        }
        self.db.commit(tx).map_err(map_engine_err)
    }

    fn pause_reindex_after_failure(
        &self,
        embedding_space_id: &str,
        last_committed_cursor: usize,
        from_producer: &EvidenceProducer,
        to_producer: Option<&EvidenceProducer>,
    ) -> Result<()> {
        let tx = self.db.begin().map_err(map_engine_err)?;
        let params = HashMap::from([
            (
                "space".to_string(),
                Value::Text(embedding_space_id.to_string()),
            ),
            ("status".to_string(), Value::Text("paused".to_string())),
            (
                "cursor".to_string(),
                Value::Int64(last_committed_cursor as i64),
            ),
            ("from_producer".to_string(), producer_value(from_producer)),
            (
                "to_producer".to_string(),
                to_producer.map(producer_value).unwrap_or(Value::Null),
            ),
            ("advanced_at".to_string(), Value::TxId(tx)),
        ]);
        let result = self.db.execute_in_tx(
            tx,
            "UPDATE reindex_cursors SET status = $status, last_committed_cursor = $cursor, from_producer = $from_producer, to_producer = $to_producer, last_advanced_at = $advanced_at WHERE embedding_space_id = $space",
            &params,
        );
        if let Err(err) = result {
            let _ = self.db.rollback(tx);
            return Err(map_engine_err(err));
        }
        self.db.commit(tx).map_err(map_engine_err)?;
        self.push_reindex_event(ReindexEvent::Paused {
            embedding_space_id: embedding_space_id.to_string(),
            last_committed_cursor,
        });
        Ok(())
    }

    fn reindex_cursor_in_tx(
        &self,
        tx: TxId,
        embedding_space_id: &str,
    ) -> Result<Option<ReindexCursor>> {
        let result = self
            .db
            .execute_in_tx(
                tx,
                "SELECT status, lease_epoch, last_committed_cursor, from_producer, to_producer, started_at FROM reindex_cursors WHERE embedding_space_id = $space",
                &HashMap::from([(
                    "space".to_string(),
                    Value::Text(embedding_space_id.to_string()),
                )]),
            )
            .map_err(map_engine_err)?;
        decode_reindex_cursor(&result)
    }

    fn reindex_candidates(
        &self,
        embedding_space_id: &str,
        started_at: TxId,
    ) -> Result<Vec<ReindexCandidate>> {
        let space = self.embedding_space_for_write(embedding_space_id)?;
        let bindings = space
            .bindings
            .iter()
            .map(|binding| (binding.table_name.clone(), binding.vector_column.clone()))
            .collect::<HashSet<_>>();
        if bindings.is_empty() {
            return Ok(Vec::new());
        }
        let snapshot = self.db.snapshot();
        let mut candidates = Vec::new();
        if bindings.iter().any(|(table, _)| table == "evidence") {
            candidates.extend(self.reindex_evidence_candidates(
                embedding_space_id,
                started_at,
                snapshot,
                &bindings,
            )?);
        }
        if bindings.iter().any(|(table, _)| table == "digests") {
            candidates.extend(self.reindex_digest_candidates(
                embedding_space_id,
                started_at,
                snapshot,
                &bindings,
            )?);
        }
        candidates.sort_by_key(|candidate| (candidate.created_at, candidate.target_id));
        Ok(candidates)
    }

    fn reindex_evidence_candidates(
        &self,
        embedding_space_id: &str,
        started_at: TxId,
        snapshot: contextdb_core::SnapshotId,
        bindings: &HashSet<(String, String)>,
    ) -> Result<Vec<ReindexCandidate>> {
        let evidence_rows = self.db.scan("evidence", snapshot).map_err(map_engine_err)?;
        let evidence_by_id = evidence_rows
            .into_iter()
            .filter_map(|row| {
                let id = row.values.get("id").and_then(value_uuid)?;
                Some((id, row))
            })
            .collect::<HashMap<_, _>>();
        let handle_rows = self
            .db
            .scan("evidence_embedding_handles", snapshot)
            .map_err(map_engine_err)?;
        let mut latest_handles = HashMap::<uuid::Uuid, VersionedRow>::new();
        for handle in handle_rows {
            let Some(space) = handle.values.get("embedding_space_id").and_then(value_str) else {
                continue;
            };
            if space != embedding_space_id {
                continue;
            }
            let Some(created_at) = handle.values.get("created_at").and_then(value_txid_raw) else {
                continue;
            };
            if created_at > started_at {
                continue;
            }
            let Some(target_id) = handle.values.get("target_id").and_then(value_uuid) else {
                continue;
            };
            let replace = latest_handles
                .get(&target_id)
                .and_then(|row| row.values.get("created_at").and_then(value_txid_raw))
                .is_none_or(|existing_created_at| created_at > existing_created_at);
            if replace {
                latest_handles.insert(target_id, handle);
            }
        }
        let mut candidates = Vec::new();
        for handle in latest_handles.into_values() {
            let Some(created_at) = handle.values.get("created_at").and_then(value_txid_raw) else {
                continue;
            };
            let Some(target_id) = handle.values.get("target_id").and_then(value_uuid) else {
                continue;
            };
            let Some(evidence) = evidence_by_id.get(&target_id) else {
                continue;
            };
            let kind = evidence
                .values
                .get("kind")
                .and_then(value_str)
                .map(evidence_kind_from_str)
                .transpose()?
                .unwrap_or(EvidenceKind::Text);
            let vector_column = match kind {
                EvidenceKind::ImageFrame => "vector_vision",
                _ => "vector_text",
            }
            .to_string();
            if !bindings.contains(&("evidence".to_string(), vector_column.clone())) {
                continue;
            }
            let input = embedding_input_for_evidence(kind, evidence)?;
            candidates.push(ReindexCandidate {
                handle_table: "evidence_embedding_handles",
                table_name: "evidence".to_string(),
                vector_column,
                row_id: evidence.row_id,
                target_id,
                context_id: ContextId::from(
                    handle.values.get("context_id").and_then(value_uuid).ok_or(
                        CgError::NotFound {
                            id: Some(target_id.to_string()),
                            missing: Some("context_id"),
                        },
                    )?,
                ),
                kind,
                input,
                produced_by: required_producer(handle.values.get("produced_by"), "produced_by")?,
                created_at,
            });
        }
        Ok(candidates)
    }

    fn reindex_digest_candidates(
        &self,
        embedding_space_id: &str,
        started_at: TxId,
        snapshot: contextdb_core::SnapshotId,
        bindings: &HashSet<(String, String)>,
    ) -> Result<Vec<ReindexCandidate>> {
        if !bindings.contains(&("digests".to_string(), "vector_text".to_string())) {
            return Ok(Vec::new());
        }
        let digest_rows = self.db.scan("digests", snapshot).map_err(map_engine_err)?;
        let digest_by_id = digest_rows
            .into_iter()
            .filter_map(|row| {
                let id = row.values.get("id").and_then(value_uuid)?;
                Some((id, row))
            })
            .collect::<HashMap<_, _>>();
        let handle_rows = self
            .db
            .scan("digest_embedding_handles", snapshot)
            .map_err(map_engine_err)?;
        let mut latest_handles = HashMap::<uuid::Uuid, VersionedRow>::new();
        for handle in handle_rows {
            let Some(space) = handle.values.get("embedding_space_id").and_then(value_str) else {
                continue;
            };
            if space != embedding_space_id {
                continue;
            }
            let Some(created_at) = handle.values.get("created_at").and_then(value_txid_raw) else {
                continue;
            };
            if created_at > started_at {
                continue;
            }
            let Some(target_id) = handle.values.get("target_id").and_then(value_uuid) else {
                continue;
            };
            let replace = latest_handles
                .get(&target_id)
                .and_then(|row| row.values.get("created_at").and_then(value_txid_raw))
                .is_none_or(|existing_created_at| created_at > existing_created_at);
            if replace {
                latest_handles.insert(target_id, handle);
            }
        }
        let mut candidates = Vec::new();
        for handle in latest_handles.into_values() {
            let Some(created_at) = handle.values.get("created_at").and_then(value_txid_raw) else {
                continue;
            };
            let Some(target_id) = handle.values.get("target_id").and_then(value_uuid) else {
                continue;
            };
            let Some(digest) = digest_by_id.get(&target_id) else {
                continue;
            };
            let text = digest
                .values
                .get("summary")
                .and_then(value_str)
                .unwrap_or_default()
                .to_string();
            candidates.push(ReindexCandidate {
                handle_table: "digest_embedding_handles",
                table_name: "digests".to_string(),
                vector_column: "vector_text".to_string(),
                row_id: digest.row_id,
                target_id,
                context_id: ContextId::from(
                    handle.values.get("context_id").and_then(value_uuid).ok_or(
                        CgError::NotFound {
                            id: Some(target_id.to_string()),
                            missing: Some("context_id"),
                        },
                    )?,
                ),
                kind: EvidenceKind::Text,
                input: EmbeddingInput::Text(text),
                produced_by: required_producer(handle.values.get("produced_by"), "produced_by")?,
                created_at,
            });
        }
        Ok(candidates)
    }
}

#[derive(Debug, Clone)]
struct ReindexLease {
    started_at: TxId,
    resumed_at_cursor: usize,
    from_producer: Option<EvidenceProducer>,
    to_producer: Option<EvidenceProducer>,
}

#[derive(Debug, Clone)]
struct ReindexCursor {
    status: String,
    lease_epoch: usize,
    last_committed_cursor: usize,
    from_producer: Option<EvidenceProducer>,
    to_producer: Option<EvidenceProducer>,
    started_at: TxId,
}

#[derive(Debug, Clone)]
struct ReindexCandidate {
    handle_table: &'static str,
    table_name: String,
    vector_column: String,
    row_id: RowId,
    target_id: uuid::Uuid,
    context_id: ContextId,
    kind: EvidenceKind,
    input: EmbeddingInput,
    produced_by: EvidenceProducer,
    created_at: TxId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReindexCursorStatus {
    InProgress,
    Paused,
    Completed,
}

impl ReindexCursorStatus {
    fn as_str(self) -> &'static str {
        match self {
            ReindexCursorStatus::InProgress => "in_progress",
            ReindexCursorStatus::Paused => "paused",
            ReindexCursorStatus::Completed => "completed",
        }
    }
}

fn embedding_input_for_evidence(kind: EvidenceKind, row: &VersionedRow) -> Result<EmbeddingInput> {
    match kind {
        EvidenceKind::Text => row
            .values
            .get("text_projection")
            .and_then(value_str)
            .map(|text| EmbeddingInput::Text(text.to_string()))
            .ok_or(CgError::EvidenceUnavailable),
        EvidenceKind::ImageFrame => Ok(EmbeddingInput::ImageFrame {
            uri: row
                .values
                .get("source_ref")
                .and_then(value_str)
                .unwrap_or_default()
                .to_string(),
        }),
        EvidenceKind::AudioSegment => Ok(EmbeddingInput::AudioSegment {
            uri: row
                .values
                .get("source_ref")
                .and_then(value_str)
                .unwrap_or_default()
                .to_string(),
        }),
        EvidenceKind::VideoSegment => Ok(EmbeddingInput::VideoSegment {
            uri: row
                .values
                .get("source_ref")
                .and_then(value_str)
                .unwrap_or_default()
                .to_string(),
        }),
        EvidenceKind::StructuredSignal => Ok(EmbeddingInput::StructuredSignal(serde_json::json!({
            "source_ref": row.values.get("source_ref").and_then(value_str).unwrap_or_default(),
        }))),
    }
}

fn producer_json(producer: &EvidenceProducer) -> String {
    serde_json::to_string(producer).unwrap_or_else(|_| "{}".to_string())
}

fn value_text(row: &[Value], columns: &[String], name: &str) -> Result<String> {
    let index = column_index(columns, name)?;
    match row.get(index) {
        Some(Value::Text(value)) => Ok(value.clone()),
        value => Err(CgError::Engine(format!(
            "expected text column {name}, got {value:?}"
        ))),
    }
}

fn value_i64(row: &[Value], columns: &[String], name: &str) -> Result<i64> {
    let index = column_index(columns, name)?;
    match row.get(index) {
        Some(Value::Int64(value)) => Ok(*value),
        value => Err(CgError::Engine(format!(
            "expected integer column {name}, got {value:?}"
        ))),
    }
}

fn value_txid(row: &[Value], columns: &[String], name: &str) -> Result<TxId> {
    let index = column_index(columns, name)?;
    match row.get(index) {
        Some(Value::TxId(value)) => Ok(*value),
        value => Err(CgError::Engine(format!(
            "expected txid column {name}, got {value:?}"
        ))),
    }
}

fn decode_reindex_cursor(result: &QueryResult) -> Result<Option<ReindexCursor>> {
    let Some(row) = result.rows.first() else {
        return Ok(None);
    };
    Ok(Some(ReindexCursor {
        status: value_text(row, &result.columns, "status")?,
        lease_epoch: value_i64(row, &result.columns, "lease_epoch")? as usize,
        last_committed_cursor: value_i64(row, &result.columns, "last_committed_cursor")? as usize,
        from_producer: value_producer(row, &result.columns, "from_producer")?,
        to_producer: value_producer(row, &result.columns, "to_producer")?,
        started_at: value_txid(row, &result.columns, "started_at")?,
    }))
}

fn value_producer(
    row: &[Value],
    columns: &[String],
    name: &str,
) -> Result<Option<EvidenceProducer>> {
    let index = column_index(columns, name)?;
    match row.get(index) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => required_producer(Some(value), name).map(Some),
    }
}

fn column_index(columns: &[String], name: &str) -> Result<usize> {
    columns
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| CgError::Engine(format!("missing column {name}")))
}

fn value_str(value: &Value) -> Option<&str> {
    match value {
        Value::Text(value) => Some(value.as_str()),
        _ => None,
    }
}

fn value_uuid(value: &Value) -> Option<uuid::Uuid> {
    match value {
        Value::Uuid(value) => Some(*value),
        _ => None,
    }
}

fn value_txid_raw(value: &Value) -> Option<TxId> {
    match value {
        Value::TxId(value) => Some(*value),
        _ => None,
    }
}

fn required_producer(value: Option<&Value>, name: &str) -> Result<EvidenceProducer> {
    match value {
        Some(Value::Json(value)) => serde_json::from_value(value.clone())
            .map_err(|err| CgError::Engine(format!("invalid {name} producer json: {err}"))),
        other => Err(CgError::Engine(format!(
            "expected producer json in {name}, got {other:?}"
        ))),
    }
}
