#[cfg(context_graph_internal_test_support)]
use std::collections::BTreeSet;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
#[cfg(context_graph_internal_test_support)]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(context_graph_internal_test_support)]
use std::sync::{Arc, Mutex};

use chrono::Utc;
#[cfg(context_graph_internal_test_support)]
use context_graph::error::EngineError;
use context_graph::{
    CgError, CreateContext, CreateEntity, EmbeddingId, EmbeddingTarget, EntityType, EvidenceId,
    EvidenceKind, EvidenceProducer, EvidenceRef, PrecomputedEmbedding, RecordObservation,
    RetentionStatus, Store, StoreConfig,
};
use contextdb_core::{RowId, Value, VectorIndexRef};
use contextdb_engine::Database;
#[cfg(context_graph_internal_test_support)]
use contextdb_engine::plugin::{CommitSource, DatabasePlugin};
#[cfg(context_graph_internal_test_support)]
use contextdb_tx::WriteSet;

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    store: Store,
    context_id: context_graph::ContextId,
    entity_id: context_graph::EntityId,
}

#[cfg(context_graph_internal_test_support)]
#[derive(Default)]
struct StagedFailureProbe {
    armed: AtomicBool,
    seen_matching_write_set: AtomicBool,
    observed_pre_commit_calls: AtomicUsize,
    observed_vector_inserts: AtomicUsize,
    observed_staged_tx_count: AtomicUsize,
    target: Mutex<Option<StagedFailureTarget>>,
}

#[cfg(context_graph_internal_test_support)]
#[derive(Clone)]
struct StagedFailureTarget {
    good_evidence: EvidenceId,
    bad_evidence: EvidenceId,
    good_embedding: EmbeddingId,
    bad_embedding: EmbeddingId,
    good_vector: Vec<f32>,
    bad_vector: Vec<f32>,
}

#[cfg(context_graph_internal_test_support)]
impl StagedFailureProbe {
    fn arm(&self, target: StagedFailureTarget) {
        *self.target.lock().expect("target") = Some(target);
        self.seen_matching_write_set.store(false, Ordering::SeqCst);
        self.observed_pre_commit_calls.store(0, Ordering::SeqCst);
        self.observed_vector_inserts.store(0, Ordering::SeqCst);
        self.observed_staged_tx_count.store(0, Ordering::SeqCst);
        self.armed.store(true, Ordering::SeqCst);
    }

    fn disarm(&self) {
        self.armed.store(false, Ordering::SeqCst);
    }

    fn saw_matching_write_set(&self) -> bool {
        self.seen_matching_write_set.load(Ordering::SeqCst)
    }

    fn pre_commit_calls(&self) -> usize {
        self.observed_pre_commit_calls.load(Ordering::SeqCst)
    }

    fn vector_insert_count(&self) -> usize {
        self.observed_vector_inserts.load(Ordering::SeqCst)
    }

    fn staged_tx_count(&self) -> usize {
        self.observed_staged_tx_count.load(Ordering::SeqCst)
    }
}

#[cfg(context_graph_internal_test_support)]
impl DatabasePlugin for StagedFailureProbe {
    fn pre_commit(&self, ws: &WriteSet, _source: CommitSource) -> contextdb_core::Result<()> {
        if !self.armed.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.observed_pre_commit_calls
            .fetch_add(1, Ordering::SeqCst);
        let Some(target) = self.target.lock().expect("target").clone() else {
            return Ok(());
        };
        let good_row = write_set_row_for_uuid(ws, "evidence", "id", target.good_evidence.as_uuid());
        let bad_row = write_set_row_for_uuid(ws, "evidence", "id", target.bad_evidence.as_uuid());
        let good_handle = write_set_row_for_uuid(
            ws,
            "evidence_embedding_handles",
            "id",
            target.good_embedding.as_uuid(),
        );
        let bad_handle = write_set_row_for_uuid(
            ws,
            "evidence_embedding_handles",
            "id",
            target.bad_embedding.as_uuid(),
        );
        let vector_inserts = ws
            .vector_inserts
            .iter()
            .filter(|entry| entry.index.table == "evidence" && entry.index.column == "vector_text")
            .count();
        self.observed_vector_inserts
            .store(vector_inserts, Ordering::SeqCst);
        let good_vector_tx = good_row.and_then(|(row_id, _)| {
            write_set_vector_tx(ws, row_id, "evidence", "vector_text", &target.good_vector)
        });
        let bad_vector_tx = bad_row.and_then(|(row_id, _)| {
            write_set_vector_tx(ws, row_id, "evidence", "vector_text", &target.bad_vector)
        });
        let txs = [
            good_row.map(|(_, tx)| tx),
            bad_row.map(|(_, tx)| tx),
            good_handle.map(|(_, tx)| tx),
            bad_handle.map(|(_, tx)| tx),
            good_vector_tx,
            bad_vector_tx,
        ]
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>();
        self.observed_staged_tx_count
            .store(txs.len(), Ordering::SeqCst);
        if good_row.is_some()
            && bad_row.is_some()
            && good_handle.is_some()
            && bad_handle.is_some()
            && vector_inserts == 2
            && good_vector_tx.is_some()
            && bad_vector_tx.is_some()
            && txs.len() == 1
        {
            self.seen_matching_write_set.store(true, Ordering::SeqCst);
            return Err(contextdb_core::Error::UniqueViolation {
                table: "evidence_embedding_handles".to_string(),
                column: "id".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(context_graph_internal_test_support)]
fn write_set_row_for_uuid(
    ws: &WriteSet,
    table: &str,
    column: &str,
    value: uuid::Uuid,
) -> Option<(RowId, u64)> {
    ws.relational_inserts
        .iter()
        .filter(|(table_name, _)| table_name == table)
        .find_map(|(_, row)| {
            (row.values.get(column) == Some(&Value::Uuid(value)))
                .then_some((row.row_id, row.created_tx.0))
        })
}

#[cfg(context_graph_internal_test_support)]
fn write_set_vector_tx(
    ws: &WriteSet,
    row_id: RowId,
    table: &str,
    column: &str,
    vector: &[f32],
) -> Option<u64> {
    ws.vector_inserts.iter().find_map(|entry| {
        (entry.row_id == row_id
            && entry.index.table == table
            && entry.index.column == column
            && entry.vector == vector)
            .then_some(entry.created_tx.0)
    })
}

fn producer(system: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: system.to_string(),
        model_name: "external-encoder".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("precomputed.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let context = store
        .create_context(CreateContext {
            name: format!("precomputed-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "camera-event".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    Fixture {
        _dir: dir,
        db_path,
        store,
        context_id: context.id,
        entity_id: entity.id,
    }
}

fn text_vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[axis] = 1.0;
    vector
}

fn observation(
    fixture: &Fixture,
    evidence_id: EvidenceId,
    embedding_id: EmbeddingId,
    vector: Vec<f32>,
    space: &str,
) -> RecordObservation {
    RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: fixture.entity_id,
        context_id: fixture.context_id,
        observation_type: "camera-frame".to_string(),
        source: "edge".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref:
                "cas://sha256:1111111111111111111111111111111111111111111111111111111111111111"
                    .to_string(),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer("capture-pipeline"),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: embedding_id,
            embedding_space_id: space.to_string(),
            vector,
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer("external"),
        }],
    }
}

fn raw_db(path: &Path, store: Store) -> Database {
    drop(store);
    Database::open(path).expect("raw database open")
}

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn count(db: &Database, sql: &str, params: &HashMap<String, Value>) -> i64 {
    let result = db.execute(sql, params).expect("count query");
    assert_eq!(result.rows.len(), 1);
    assert!(
        matches!(result.rows[0].first(), Some(Value::Int64(_))),
        "expected count integer, got {:?}",
        result.rows[0].first()
    );
    match result.rows[0][0] {
        Value::Int64(value) => value,
        _ => 0,
    }
}

fn hit_contains_id(db: &Database, table: &str, hits: &[(RowId, f32)], id: uuid::Uuid) -> bool {
    let rows = db.scan(table, db.snapshot()).expect("scan table");
    hits.iter().any(|(row_id, _)| {
        rows.iter()
            .find(|row| row.row_id == *row_id)
            .is_some_and(|row| row.values.get("id") == Some(&Value::Uuid(id)))
    })
}

#[test]
fn ingest_observation_with_precomputed_embedding() {
    let fixture = fixture();
    let evidence_id = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let vector = text_vector(0);
    let result = fixture.store.record_observation(observation(
        &fixture,
        evidence_id,
        embedding_id,
        vector.clone(),
        TEXT_SPACE,
    ));
    let db = raw_db(&fixture.db_path, fixture.store);

    let handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND target_id = $target_id AND embedding_space_id = $space",
        &params(vec![
            ("id", Value::Uuid(embedding_id.as_uuid())),
            ("target_id", Value::Uuid(evidence_id.as_uuid())),
            ("space", Value::Text(TEXT_SPACE.to_string())),
        ]),
    );
    let producer_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND produced_by = $producer",
        &params(vec![
            ("id", Value::Uuid(embedding_id.as_uuid())),
            (
                "producer",
                Value::Json(serde_json::to_value(producer("external")).expect("producer json")),
            ),
        ]),
    );
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &vector,
            1,
            None,
            db.snapshot(),
        )
        .expect("vector query");

    assert!(
        result
            .as_ref()
            .is_ok_and(|output| output.embedding_ids == vec![embedding_id])
            && handle_rows == 1
            && producer_rows == 1
            && hit_contains_id(&db, "evidence", &hits, evidence_id.as_uuid()),
        "expected precomputed evidence embedding to write a variant handle, producer JSON, and queryable vector; result={result:?}, handle_rows={handle_rows}, producer_rows={producer_rows}, hits={hits:?}"
    );
}

#[test]
fn ingest_observation_rejects_dimension_mismatch_for_known_space() {
    let fixture = fixture();
    let evidence_id = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let result = fixture.store.record_observation(observation(
        &fixture,
        evidence_id,
        embedding_id,
        vec![1.0; 767],
        TEXT_SPACE,
    ));
    let db = raw_db(&fixture.db_path, fixture.store);
    let evidence_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $id",
        &params(vec![("id", Value::Uuid(evidence_id.as_uuid()))]),
    );

    assert!(
        matches!(
            result,
            Err(CgError::EmbeddingDimensionMismatch {
                ref space_id,
                expected: 768,
                found: 767,
            }) if space_id == TEXT_SPACE
        ) && evidence_rows == 0,
        "expected typed dimension mismatch before evidence write, result={result:?}, evidence_rows={evidence_rows}"
    );
}

#[test]
fn precomputed_embedding_dimension_validated_at_record_observation_before_db_write() {
    let fixture = fixture();
    let before_observations = 0;

    let evidence_id = EvidenceId::new_v7();
    let result = fixture.store.record_observation(observation(
        &fixture,
        evidence_id,
        EmbeddingId::new_v7(),
        vec![1.0; 767],
        TEXT_SPACE,
    ));
    let db = raw_db(&fixture.db_path, fixture.store);
    let after_observations = count(
        &db,
        "SELECT COUNT(*) AS n FROM observations",
        &HashMap::new(),
    );
    let leaked_vectors = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &text_vector(0),
            10,
            None,
            db.snapshot(),
        )
        .expect("vector query");

    assert!(
        matches!(result, Err(CgError::EmbeddingDimensionMismatch { .. }))
            && after_observations == before_observations
            && leaked_vectors.is_empty(),
        "dimension validation must happen before observation/evidence/vector writes, result={result:?}, before={before_observations}, after={after_observations}, leaked_vectors={leaked_vectors:?}"
    );
}

#[test]
fn precomputed_embedding_unknown_space_rejected_at_record_observation() {
    let fixture = fixture();
    let result = fixture.store.record_observation(observation(
        &fixture,
        EvidenceId::new_v7(),
        EmbeddingId::new_v7(),
        vec![1.0; 32],
        "unknown_space",
    ));

    assert!(
        matches!(result, Err(CgError::UnknownEmbeddingSpace { ref id }) if id == "unknown_space"),
        "expected unknown-space typed error, got {result:?}"
    );
}

#[cfg(context_graph_internal_test_support)]
#[test]
fn precomputed_embedding_atomic_with_observation_partial_failure_rolls_back_all() {
    let fixture = fixture();
    let Fixture {
        _dir,
        db_path,
        store,
        context_id,
        entity_id,
    } = fixture;
    drop(store);
    let failure_probe = Arc::new(StagedFailureProbe::default());
    let before_db = Database::open(&db_path).expect("raw database open before failure");
    let before_watermark = before_db.committed_watermark();
    drop(before_db);
    let store = Store::open_with_database_plugin(
        StoreConfig {
            db_path: db_path.clone(),
            ..Default::default()
        },
        failure_probe.clone(),
    )
    .expect("reopen for failing call");
    let fixture = Fixture {
        _dir,
        db_path,
        store,
        context_id,
        entity_id,
    };

    let good_evidence = EvidenceId::new_v7();
    let bad_evidence = EvidenceId::new_v7();
    let good_embedding = EmbeddingId::new_v7();
    let bad_embedding = EmbeddingId::new_v7();
    let good_vector = text_vector(1);
    let bad_vector = text_vector(2);
    failure_probe.arm(StagedFailureTarget {
        good_evidence,
        bad_evidence,
        good_embedding,
        bad_embedding,
        good_vector: good_vector.clone(),
        bad_vector: bad_vector.clone(),
    });
    let mut input = observation(
        &fixture,
        good_evidence,
        good_embedding,
        good_vector.clone(),
        TEXT_SPACE,
    );
    input.evidence.push(EvidenceRef {
        id: bad_evidence,
        kind: EvidenceKind::Text,
        source_ref: "cas://sha256:2222222222222222222222222222222222222222222222222222222222222222"
            .to_string(),
        retention_status: RetentionStatus::RetainedExternal,
        ..Default::default()
    });
    input.embeddings.push(PrecomputedEmbedding {
        id: bad_embedding,
        embedding_space_id: TEXT_SPACE.to_string(),
        vector: bad_vector.clone(),
        target: EmbeddingTarget::Evidence(bad_evidence),
        produced_by: producer("external"),
    });

    let result = fixture.store.record_observation(input);
    failure_probe.disarm();
    let db = raw_db(&fixture.db_path, fixture.store);
    let after_watermark = db.committed_watermark();
    let evidence_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $good OR id = $bad",
        &params(vec![
            ("good", Value::Uuid(good_evidence.as_uuid())),
            ("bad", Value::Uuid(bad_evidence.as_uuid())),
        ]),
    );
    let handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $good OR target_id = $bad",
        &params(vec![
            ("good", Value::Uuid(good_evidence.as_uuid())),
            ("bad", Value::Uuid(bad_evidence.as_uuid())),
        ]),
    );
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &good_vector,
            10,
            None,
            db.snapshot(),
        )
        .expect("vector query");
    let leaked_good_vector = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &good_vector,
            10,
            None,
            db.snapshot(),
        )
        .expect("good vector query")
        .into_iter()
        .any(|(_, score)| score > 0.99);
    let leaked_bad_vector = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &bad_vector,
            10,
            None,
            db.snapshot(),
        )
        .expect("bad vector query")
        .into_iter()
        .any(|(_, score)| score > 0.99);

    assert!(
        matches!(
            result,
            Err(CgError::EngineFault {
                source: EngineError::UniqueViolation {
                    ref table,
                    ref column,
                }
            }) if table == "evidence_embedding_handles" && column == "id"
        ) && evidence_rows == 0
            && handle_rows == 0
            && failure_probe.pre_commit_calls() == 1
            && failure_probe.saw_matching_write_set()
            && failure_probe.vector_insert_count() == 2
            && failure_probe.staged_tx_count() == 1
            && !hit_contains_id(&db, "evidence", &hits, good_evidence.as_uuid())
            && !hit_contains_id(&db, "evidence", &hits, bad_evidence.as_uuid())
            && !leaked_good_vector
            && !leaked_bad_vector
            && after_watermark == before_watermark,
        "partial embedding failure must stage the full write set in exactly one tx and roll it back, leaving no evidence, handle, or vector writes; result={result:?}, evidence_rows={evidence_rows}, handle_rows={handle_rows}, hits={hits:?}, pre_commit_calls={}, saw_matching_write_set={}, vector_insert_count={}, staged_tx_count={}, leaked_good_vector={leaked_good_vector}, leaked_bad_vector={leaked_bad_vector}, before_watermark={before_watermark:?}, after_watermark={after_watermark:?}",
        failure_probe.pre_commit_calls(),
        failure_probe.saw_matching_write_set(),
        failure_probe.vector_insert_count(),
        failure_probe.staged_tx_count()
    );
}
