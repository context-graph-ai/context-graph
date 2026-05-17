use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use chrono::Utc;
use context_graph::test_support::embedding::{VectorSearchFilters, vector_search};
use context_graph::{
    ContextId, CreateContext, CreateEntity, DistanceMetric, EmbeddingId, EmbeddingSpaceBinding,
    EmbeddingSpaceDecl, EmbeddingTarget, EntityId, EntityType, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, PrecomputedEmbedding, RecordObservation, RetentionStatus, Store,
    StoreConfig,
};
use contextdb_core::{Error as EngineError, RowId, TxId, Value, VectorIndexRef};
use contextdb_engine::Database;
use contextdb_vector::test_seam::PauseWindow;
use serde_json::json;

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";
const VISION_SPACE: &str = "vigil_face_reid_v1";
const FACE_REFERENCE_SPACE: &str = "face_reference_v1";
const ENGINE_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const HNSW_PADDING_ROWS: usize = 1000;

struct ProbeFixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    config: StoreConfig,
    store: Store,
    context_id: ContextId,
    entity_id: EntityId,
}

fn fixture(name: &str) -> ProbeFixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let config = StoreConfig {
        db_path: db_path.clone(),
        extra_embedding_spaces: vec![vision_space(), face_reference_space()],
        ..Default::default()
    };
    let store = Store::open(config.clone()).expect("open");
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: "probe entity".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    ProbeFixture {
        _dir: dir,
        db_path,
        config,
        store,
        context_id: context.id,
        entity_id: entity.id,
    }
}

fn vision_space() -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: VISION_SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "vigil-face".to_string(),
        model_version: "1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: VISION_SPACE.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_vision".to_string(),
        }],
    }
}

fn face_reference_space() -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: FACE_REFERENCE_SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "face-reference".to_string(),
        model_version: "1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: None,
        bindings: Vec::new(),
    }
}

fn producer(label: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: format!("probe-{label}"),
        model_name: "probe-model".to_string(),
        model_version: label.to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn vector(axis: usize) -> Vec<f32> {
    let mut value = vec![0.0; 512];
    value[axis % 512] = 1.0;
    value
}

fn text_vector(axis: usize) -> Vec<f32> {
    let mut value = vec![0.0; 768];
    value[axis % 768] = 1.0;
    value
}

fn text_index() -> VectorIndexRef {
    VectorIndexRef::new("evidence", "vector_text")
}

fn vision_index() -> VectorIndexRef {
    VectorIndexRef::new("evidence", "vector_vision")
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

fn register_counting_impact_trigger(db: &Database) -> Arc<AtomicUsize> {
    let counter = Arc::new(AtomicUsize::new(0));
    let registered = db.registered_trigger_callbacks();
    for trigger in db.list_triggers() {
        if registered.iter().any(|name| name == &trigger.name) {
            continue;
        }
        if trigger.name == "impact_analysis_trigger" {
            let counter = Arc::clone(&counter);
            db.register_trigger_callback(&trigger.name, move |_db, _ctx| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .expect("register counting impact trigger");
        } else {
            db.register_trigger_callback(&trigger.name, |_db, _ctx| Ok(()))
                .expect("register noop trigger");
        }
    }
    if !db.list_triggers().is_empty() {
        db.complete_initialization()
            .expect("complete trigger initialization");
    }
    counter
}

fn raw_duplicate_evidence_insert(
    db: &Database,
    evidence_id: EvidenceId,
    observation_id: uuid::Uuid,
    context_id: ContextId,
    supply_order: i64,
) -> contextdb_core::Result<contextdb_engine::QueryResult> {
    db.execute(
        "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, retention_status, supply_order) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $retention_status, $supply_order)",
        &params(vec![
            ("id", Value::Uuid(evidence_id.as_uuid())),
            ("observation_id", Value::Uuid(observation_id)),
            ("context_id", Value::Uuid(context_id.as_uuid())),
            ("kind", Value::Text("text".to_string())),
            (
                "source_ref",
                Value::Text(format!("cas://sha256:{supply_order:064x}")),
            ),
            (
                "retention_status",
                Value::Text("retained_external".to_string()),
            ),
            ("supply_order", Value::Int64(supply_order)),
        ]),
    )
}

fn raw_observation_insert(
    db: &Database,
    context_id: ContextId,
    entity_id: EntityId,
) -> contextdb_core::Result<contextdb_engine::QueryResult> {
    db.execute(
        "INSERT INTO observations (id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at) VALUES ($id, $observation_type, $observed_properties, $state_delta, $properties, $source, $entity_id, $context_id, $timestamp, $created_at)",
        &params(vec![
            ("id", Value::Uuid(uuid::Uuid::now_v7())),
            (
                "observation_type",
                Value::Text("trigger-positive-control".to_string()),
            ),
            ("observed_properties", Value::Json(json!({}))),
            ("state_delta", Value::Json(json!({"trigger_probe": true}))),
            ("properties", Value::Json(json!({}))),
            ("source", Value::Text("engine-probe".to_string())),
            ("entity_id", Value::Uuid(entity_id.as_uuid())),
            ("context_id", Value::Uuid(context_id.as_uuid())),
            ("timestamp", Value::Timestamp(1_700_000_001)),
            ("created_at", Value::TxId(TxId(1))),
        ]),
    )
}

#[derive(Clone)]
struct EvidenceWrite {
    evidence_id: EvidenceId,
    embedding_id: EmbeddingId,
    kind: EvidenceKind,
    embedding_space_id: &'static str,
    vector: Vec<f32>,
}

fn record_evidence_concurrently_with_reader(
    fixture: &ProbeFixture,
    writes: Vec<EvidenceWrite>,
) -> (
    Vec<context_graph::Result<context_graph::RecordObservationResult>>,
    std::time::Duration,
    Vec<usize>,
    Vec<usize>,
) {
    let barrier = Arc::new(Barrier::new(writes.len() + 2));
    let done = Arc::new(AtomicBool::new(false));
    let observed_counts = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_vector_counts = Arc::new(std::sync::Mutex::new(Vec::new()));
    let reader_writes = writes.clone();
    let final_writes = writes.clone();
    let reader_store = fixture.store.clone();
    let reader_barrier = Arc::clone(&barrier);
    let reader_done = Arc::clone(&done);
    let reader_counts = Arc::clone(&observed_counts);
    let reader_vector_counts = Arc::clone(&observed_vector_counts);
    let reader = thread::spawn(move || {
        reader_barrier.wait();
        for _ in 0..256 {
            let count = reader_writes
                .iter()
                .filter(|write| reader_store.get_evidence(write.evidence_id).is_ok())
                .count();
            let vector_count = matching_vector_hit_count(&reader_store, &reader_writes);
            reader_counts.lock().expect("counts").push(count);
            reader_vector_counts
                .lock()
                .expect("vector counts")
                .push(vector_count);
            if reader_done.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });
    let handles = writes
        .into_iter()
        .map(|write| {
            let writer = fixture.store.clone();
            let barrier = Arc::clone(&barrier);
            let context_id = fixture.context_id;
            let entity_id = fixture.entity_id;
            thread::spawn(move || {
                barrier.wait();
                record_embedding(writer, context_id, entity_id, write)
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    let started = Instant::now();
    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer thread"))
        .collect();
    let duration = started.elapsed();
    done.store(true, Ordering::SeqCst);
    reader.join().expect("reader thread");
    let final_count = final_writes
        .iter()
        .filter(|write| fixture.store.get_evidence(write.evidence_id).is_ok())
        .count();
    let final_vector_count = matching_vector_hit_count(&fixture.store, &final_writes);
    let mut counts = observed_counts.lock().expect("counts").clone();
    counts.push(final_count);
    let mut vector_counts = observed_vector_counts
        .lock()
        .expect("vector counts")
        .clone();
    vector_counts.push(final_vector_count);
    (results, duration, counts, vector_counts)
}

fn record_embedding(
    store: Store,
    context_id: ContextId,
    entity_id: EntityId,
    write: EvidenceWrite,
) -> context_graph::Result<context_graph::RecordObservationResult> {
    store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "probe-observation".to_string(),
        source: "engine-probe".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: write.evidence_id,
            kind: write.kind,
            source_ref: format!(
                "cas://sha256:{:064x}",
                write.evidence_id.as_uuid().as_u128()
            ),
            mime_type: Some(
                match write.kind {
                    EvidenceKind::Text => "text/plain",
                    EvidenceKind::StructuredSignal => "application/json",
                    EvidenceKind::ImageFrame => "image/jpeg",
                    EvidenceKind::AudioSegment => "audio/wav",
                    EvidenceKind::VideoSegment => "video/mp4",
                }
                .to_string(),
            ),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer("record"),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: write.embedding_id,
            embedding_space_id: write.embedding_space_id.to_string(),
            vector: write.vector,
            target: EmbeddingTarget::Evidence(write.evidence_id),
            produced_by: producer("record"),
        }],
    })
}

fn record_text_embedding(
    store: Store,
    context_id: ContextId,
    entity_id: EntityId,
    evidence_id: EvidenceId,
    embedding_id: EmbeddingId,
    axis: usize,
) -> context_graph::Result<context_graph::RecordObservationResult> {
    record_embedding(
        store,
        context_id,
        entity_id,
        EvidenceWrite {
            evidence_id,
            embedding_id,
            kind: EvidenceKind::Text,
            embedding_space_id: TEXT_SPACE,
            vector: text_vector(axis),
        },
    )
}

fn matching_vector_hit_count(store: &Store, writes: &[EvidenceWrite]) -> usize {
    writes
        .iter()
        .filter(|write| {
            vector_search(
                store,
                &write.vector,
                write.embedding_space_id,
                VectorSearchFilters {
                    top_k: Some(1),
                    ..Default::default()
                },
            )
            .is_ok_and(|hits| {
                matches!(
                    hits.first().map(|hit| &hit.target),
                    Some(EmbeddingTarget::Evidence(id)) if *id == write.evidence_id
                )
            })
        })
        .count()
}

fn first_hit_id(db: &Database, index: VectorIndexRef, query: Vec<f32>) -> Option<uuid::Uuid> {
    let hits = db
        .query_vector(index, &query, 1, None, db.snapshot())
        .expect("vector query");
    let (row_id, _) = *hits.first()?;
    let rows = db.scan("evidence", db.snapshot()).expect("scan evidence");
    rows.into_iter().find_map(|row| {
        (row.row_id == row_id)
            .then(|| row.values.get("id").cloned())
            .flatten()
            .and_then(|value| match value {
                Value::Uuid(id) => Some(id),
                _ => None,
            })
    })
}

fn evidence_row_id(db: &Database, evidence_id: EvidenceId) -> Option<RowId> {
    db.scan("evidence", db.snapshot())
        .expect("scan evidence")
        .into_iter()
        .find_map(|row| {
            let id = row.values.get("id").and_then(|value| match value {
                Value::Uuid(id) => Some(*id),
                _ => None,
            });
            (id == Some(evidence_id.as_uuid())).then_some(row.row_id)
        })
}

fn replace_vector(
    db: Arc<Database>,
    index: VectorIndexRef,
    row_id: RowId,
    vector: Vec<f32>,
) -> contextdb_core::Result<()> {
    let tx = db.begin()?;
    if let Err(err) = db.insert_vector(tx, index, row_id, vector) {
        let _ = db.rollback(tx);
        return Err(err);
    }
    db.commit(tx)
}

fn seed_hnsw_padding(
    db: &Database,
    context_id: ContextId,
    observation_id: uuid::Uuid,
) -> Result<(), String> {
    for i in 0..HNSW_PADDING_ROWS {
        db.execute(
            "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, retention_status, supply_order, vector_text, vector_vision) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $retention_status, $supply_order, $text, $vision)",
            &params(vec![
                ("id", Value::Uuid(uuid::Uuid::now_v7())),
                ("observation_id", Value::Uuid(observation_id)),
                ("context_id", Value::Uuid(context_id.as_uuid())),
                ("kind", Value::Text("text".to_string())),
                ("source_ref", Value::Text(format!("cas://sha256:{i:064x}"))),
                ("retention_status", Value::Text("retained_external".to_string())),
                ("supply_order", Value::Int64(10_000 + i as i64)),
                ("text", Value::Vector(text_vector(100))),
                ("vision", Value::Vector(vector(100))),
            ]),
        )
        .map_err(|err| format!("{err:?}"))?;
    }
    Ok(())
}

#[test]
fn engine_probe_named_vector_index_concurrent_writes_distinct_index_refs_serialize_per_index() {
    let fixture = fixture("distinct-indexes");
    let text_evidence = EvidenceId::new_v7();
    let vision_evidence = EvidenceId::new_v7();
    let text_seed = record_embedding(
        fixture.store.clone(),
        fixture.context_id,
        fixture.entity_id,
        EvidenceWrite {
            evidence_id: text_evidence,
            embedding_id: EmbeddingId::new_v7(),
            kind: EvidenceKind::Text,
            embedding_space_id: TEXT_SPACE,
            vector: text_vector(7),
        },
    );
    let vision_seed = record_embedding(
        fixture.store.clone(),
        fixture.context_id,
        fixture.entity_id,
        EvidenceWrite {
            evidence_id: vision_evidence,
            embedding_id: EmbeddingId::new_v7(),
            kind: EvidenceKind::ImageFrame,
            embedding_space_id: VISION_SPACE,
            vector: vector(3),
        },
    );
    let db_path = fixture.db_path.clone();
    let context_id = fixture.context_id;
    drop(fixture.store);
    let db = Arc::new(Database::open(&db_path).expect("open raw contextdb"));
    let vision_row_id = evidence_row_id(&db, vision_evidence);
    let padding_seed = match text_seed.as_ref() {
        Ok(result) => seed_hnsw_padding(&db, context_id, result.observation_id.as_uuid()),
        Err(err) => Err(format!("text seed failed: {err:?}")),
    };

    let vector_store = db.vector_store_for_test();
    let text_pause = vector_store.arm_maintenance_pause_for_test(&text_index(), PauseWindow::Build);
    let vision_pause =
        vector_store.arm_maintenance_pause_for_test(&vision_index(), PauseWindow::Build);

    let (text_done_tx, text_done_rx) = mpsc::channel();
    let text_db = Arc::clone(&db);
    thread::spawn(move || {
        let result = first_hit_id(&text_db, text_index(), text_vector(7))
            .ok_or_else(|| "missing text hit".to_string());
        text_done_tx.send(result).expect("send text result");
    });
    let text_reached = text_pause.wait_until_reached(ENGINE_PROBE_TIMEOUT);

    let (vision_done_tx, vision_done_rx) = mpsc::channel();
    let vision_db = Arc::clone(&db);
    let vision_row_for_thread = vision_row_id;
    thread::spawn(move || {
        let result = vision_row_for_thread
            .ok_or_else(|| "missing vision evidence row".to_string())
            .and_then(|row_id| {
                replace_vector(Arc::clone(&vision_db), vision_index(), row_id, vector(4))
                    .map_err(|err| format!("{err:?}"))
            })
            .and_then(|()| {
                first_hit_id(&vision_db, vision_index(), vector(4))
                    .ok_or_else(|| "missing vision hit".to_string())
            });
        vision_done_tx.send(result).expect("send vision result");
    });
    let vision_reached_before_text_release = vision_pause.wait_until_reached(ENGINE_PROBE_TIMEOUT);
    let text_done_before_release = text_done_rx.try_recv();
    if vision_reached_before_text_release {
        vision_pause.release();
    }
    let vision_done_before_release = vision_done_rx.recv_timeout(ENGINE_PROBE_TIMEOUT);
    let text_done_before_release_debug = format!("{text_done_before_release:?}");
    let vision_done_before_release_debug = format!("{vision_done_before_release:?}");
    let text_blocked_before_release = matches!(text_done_before_release, Err(TryRecvError::Empty));
    let vision_completed_before_text_release = vision_done_before_release.is_ok();

    if !vision_reached_before_text_release {
        text_pause.release();
        let _ = vision_pause.wait_until_reached(ENGINE_PROBE_TIMEOUT);
        vision_pause.release();
    } else {
        text_pause.release();
    }

    let text_result = match text_done_before_release {
        Ok(result) => Some(result),
        Err(TryRecvError::Empty) => text_done_rx.recv_timeout(ENGINE_PROBE_TIMEOUT).ok(),
        Err(TryRecvError::Disconnected) => None,
    };
    let vision_result = match vision_done_before_release {
        Ok(result) => Some(result),
        Err(_) => vision_done_rx.recv_timeout(ENGINE_PROBE_TIMEOUT).ok(),
    };

    let text_hit = first_hit_id(&db, text_index(), text_vector(7));
    let vision_hit = first_hit_id(&db, vision_index(), vector(4));
    let text_rejects_vision_dim = db.query_vector(text_index(), &vector(3), 1, None, db.snapshot());
    let vision_rejects_text_dim =
        db.query_vector(vision_index(), &text_vector(7), 1, None, db.snapshot());

    assert!(
        text_seed.is_ok()
            && vision_seed.is_ok()
            && padding_seed.is_ok()
            && text_reached
            && vision_reached_before_text_release
            && text_blocked_before_release
            && vision_completed_before_text_release
            && matches!(text_result.as_ref(), Some(Ok(id)) if *id == text_evidence.as_uuid())
            && matches!(vision_result.as_ref(), Some(Ok(id)) if *id == vision_evidence.as_uuid())
            && text_hit == Some(text_evidence.as_uuid())
            && vision_hit == Some(vision_evidence.as_uuid())
            && text_rejects_vision_dim.is_err()
            && vision_rejects_text_dim.is_err(),
        "distinct VectorIndexRef maintenance must let vision update/search complete while text build is paused, and routing must remain disjoint, text_seed={text_seed:?}, vision_seed={vision_seed:?}, padding_seed={padding_seed:?}, vision_row_id={vision_row_id:?}, text_reached={text_reached}, vision_reached_before_text_release={vision_reached_before_text_release}, text_done_before_release={text_done_before_release_debug}, vision_done_before_release={vision_done_before_release_debug}, text_result={text_result:?}, vision_result={vision_result:?}, text_hit={text_hit:?}, vision_hit={vision_hit:?}, text_rejects_vision_dim={text_rejects_vision_dim:?}, vision_rejects_text_dim={vision_rejects_text_dim:?}"
    );
}

#[test]
fn engine_probe_named_vector_index_concurrent_writes_same_index_ref_serialize_no_torn_index() {
    let fixture = fixture("same-index");
    let first_evidence = EvidenceId::new_v7();
    let second_evidence = EvidenceId::new_v7();
    let (results, duration, observed_counts, observed_vector_counts) =
        record_evidence_concurrently_with_reader(
            &fixture,
            vec![
                EvidenceWrite {
                    evidence_id: first_evidence,
                    embedding_id: EmbeddingId::new_v7(),
                    kind: EvidenceKind::Text,
                    embedding_space_id: TEXT_SPACE,
                    vector: text_vector(0),
                },
                EvidenceWrite {
                    evidence_id: second_evidence,
                    embedding_id: EmbeddingId::new_v7(),
                    kind: EvidenceKind::Text,
                    embedding_space_id: TEXT_SPACE,
                    vector: text_vector(1),
                },
            ],
        );
    let db_path = fixture.db_path.clone();
    drop(fixture.store);
    let db = Database::open(&db_path).expect("open raw contextdb");
    let first_hit = first_hit_id(&db, text_index(), text_vector(0));
    let second_hit = first_hit_id(&db, text_index(), text_vector(1));
    let final_count = [first_hit, second_hit].into_iter().flatten().count();
    let evidence_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $id",
        &params(vec![("id", Value::Uuid(first_evidence.as_uuid()))]),
    ) + count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $id",
        &params(vec![("id", Value::Uuid(second_evidence.as_uuid()))]),
    );
    let handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id",
        &params(vec![("target_id", Value::Uuid(first_evidence.as_uuid()))]),
    ) + count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id",
        &params(vec![("target_id", Value::Uuid(second_evidence.as_uuid()))]),
    );
    let monotonic_counts = observed_counts
        .windows(2)
        .all(|pair| pair[0] <= pair[1] && pair[1] <= 2);
    let monotonic_vector_counts = observed_vector_counts
        .windows(2)
        .all(|pair| pair[0] <= pair[1] && pair[1] <= 2);
    let row_and_vector_snapshots_move_together = observed_counts
        .iter()
        .zip(observed_vector_counts.iter())
        .all(|(row_count, vector_count)| row_count == vector_count);

    assert!(
        results.iter().all(Result::is_ok)
            && first_hit == Some(first_evidence.as_uuid())
            && second_hit == Some(second_evidence.as_uuid())
            && final_count == 2
            && evidence_rows == 2
            && handle_rows == 2
            && monotonic_counts
            && monotonic_vector_counts
            && row_and_vector_snapshots_move_together
            && observed_counts.contains(&1)
            && observed_vector_counts.contains(&1)
            && observed_counts.last() == Some(&2)
            && observed_vector_counts.last() == Some(&2)
            && duration.as_millis() >= 1,
        "same VectorIndexRef writes must serialize without torn index state, persist both handle rows, and expose one-then-both MVCC progression through both row reads and vector search during overlap, results={results:?}, first_hit={first_hit:?}, second_hit={second_hit:?}, evidence_rows={evidence_rows}, handle_rows={handle_rows}, observed_counts={observed_counts:?}, observed_vector_counts={observed_vector_counts:?}, monotonic_counts={monotonic_counts}, monotonic_vector_counts={monotonic_vector_counts}, row_and_vector_snapshots_move_together={row_and_vector_snapshots_move_together}, duration={duration:?}"
    );
}

#[test]
fn engine_probe_immutable_embedding_handle_id_rejects_update_under_contention() {
    let fixture = fixture("immutable-handle");
    let evidence_id = EvidenceId::new_v7();
    let handle_id = EmbeddingId::new_v7();
    let seed = record_text_embedding(
        fixture.store.clone(),
        fixture.context_id,
        fixture.entity_id,
        evidence_id,
        handle_id,
        0,
    );
    let db_path = fixture.db_path.clone();
    let entity_id = fixture.entity_id;
    drop(fixture.store);
    let db = Arc::new(Database::open(&db_path).expect("open raw contextdb"));
    let barrier = Arc::new(Barrier::new(4));
    let update_a = {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            db.execute(
                "UPDATE evidence_embedding_handles SET id = $new_id WHERE id = $id",
                &params(vec![
                    ("new_id", Value::Uuid(EmbeddingId::new_v7().as_uuid())),
                    ("id", Value::Uuid(handle_id.as_uuid())),
                ]),
            )
        })
    };
    let update_b = {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            db.execute(
                "UPDATE evidence_embedding_handles SET id = $new_id WHERE id = $id",
                &params(vec![
                    ("new_id", Value::Uuid(EmbeddingId::new_v7().as_uuid())),
                    ("id", Value::Uuid(handle_id.as_uuid())),
                ]),
            )
        })
    };
    let mutable_control = {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            db.execute(
                "UPDATE entities SET properties = $properties WHERE id = $id",
                &params(vec![
                    ("properties", Value::Json(json!({"control": "updated"}))),
                    ("id", Value::Uuid(entity_id.as_uuid())),
                ]),
            )
        })
    };
    barrier.wait();
    let update_a = update_a.join().expect("update a joined");
    let update_b = update_b.join().expect("update b joined");
    let mutable_control = mutable_control.join().expect("control joined");
    let handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id",
        &params(vec![("id", Value::Uuid(handle_id.as_uuid()))]),
    );

    assert!(
        seed.is_ok()
            && matches!(update_a, Err(EngineError::ImmutableTable(ref table)) if table == "evidence_embedding_handles")
            && matches!(update_b, Err(EngineError::ImmutableTable(ref table)) if table == "evidence_embedding_handles")
            && mutable_control
                .as_ref()
                .is_ok_and(|result| result.rows_affected == 1)
            && handle_rows == 1,
        "contended handle-id updates must be rejected as ImmutableTable while mutable sibling update succeeds, seed={seed:?}, update_a={update_a:?}, update_b={update_b:?}, mutable_control={mutable_control:?}, handle_rows={handle_rows}"
    );
}

#[test]
fn engine_probe_composite_fk_same_context_rejects_cross_context_embedding_handle_write() {
    let fixture = fixture("cross-context");
    let target_evidence = EvidenceId::new_v7();
    let target_embedding = EmbeddingId::new_v7();
    let seed = record_text_embedding(
        fixture.store.clone(),
        fixture.context_id,
        fixture.entity_id,
        target_evidence,
        target_embedding,
        0,
    );
    let handle_only_context = fixture
        .store
        .create_context(CreateContext {
            name: "handle-only-context".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("handle context");
    let db_path = fixture.db_path.clone();
    let valid_context = fixture.context_id;
    drop(fixture.store);
    let db = Arc::new(Database::open(&db_path).expect("raw db"));
    let barrier = Arc::new(Barrier::new(3));
    let valid_handle = EmbeddingId::new_v7();
    let invalid_handle = EmbeddingId::new_v7();
    let insert_handle = |context_id: ContextId, handle_id: EmbeddingId, txid: TxId| {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            db.execute(
                "INSERT INTO evidence_embedding_handles (id, context_id, target_id, embedding_space_id, produced_by, created_at) VALUES ($id, $context_id, $target_id, $space, $producer, $created_at)",
                &params(vec![
                    ("id", Value::Uuid(handle_id.as_uuid())),
                    ("context_id", Value::Uuid(context_id.as_uuid())),
                    ("target_id", Value::Uuid(target_evidence.as_uuid())),
                    ("space", Value::Text(TEXT_SPACE.to_string())),
                    (
                        "producer",
                        Value::Json(serde_json::to_value(producer("cross-context")).expect("json")),
                    ),
                    ("created_at", Value::TxId(txid)),
                ]),
            )
        })
    };
    let invalid_writer = insert_handle(handle_only_context.id, invalid_handle, TxId(10));
    let valid_writer = insert_handle(valid_context, valid_handle, TxId(11));
    barrier.wait();
    let invalid = invalid_writer.join().expect("invalid writer joined");
    let valid = valid_writer.join().expect("valid writer joined");
    let valid_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id",
        &params(vec![("id", Value::Uuid(valid_handle.as_uuid()))]),
    );
    let invalid_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id",
        &params(vec![("id", Value::Uuid(invalid_handle.as_uuid()))]),
    );

    assert!(
        seed.is_ok()
            && valid.as_ref().is_ok_and(|result| result.rows_affected == 1)
            && matches!(
                invalid,
                Err(EngineError::ForeignKeyViolation {
                    ref child_table,
                    ref child_columns,
                    ref parent_table,
                    ..
                }) if child_table == "evidence_embedding_handles"
                    && *child_columns == vec!["context_id".to_string(), "target_id".to_string()]
                    && parent_table == "evidence"
            )
            && valid_rows == 1
            && invalid_rows == 0,
        "composite FK must reject only the cross-context handle while a contending same-context insert commits, seed={seed:?}, valid={valid:?}, invalid={invalid:?}, valid_rows={valid_rows}, invalid_rows={invalid_rows}"
    );
}

#[test]
fn engine_probe_atomic_multi_row_write_observation_evidence_snapshot_handle_rolls_back_on_partial_failure_under_contention()
 {
    let fixture = fixture("atomic-rollback");
    let embedding_id = EmbeddingId::new_v7();
    let first_evidence = EvidenceId::new_v7();
    let second_evidence = EvidenceId::new_v7();
    let barrier = Arc::new(Barrier::new(3));
    let writes = [(first_evidence, 0), (second_evidence, 1)];
    let handles = writes
        .into_iter()
        .map(|(evidence_id, axis)| {
            let writer = fixture.store.clone();
            let barrier = Arc::clone(&barrier);
            let context_id = fixture.context_id;
            let entity_id = fixture.entity_id;
            thread::spawn(move || {
                barrier.wait();
                record_text_embedding(
                    writer,
                    context_id,
                    entity_id,
                    evidence_id,
                    embedding_id,
                    axis,
                )
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer joined"))
        .collect::<Vec<_>>();
    let db_path = fixture.db_path.clone();
    drop(fixture.store);
    let db = Database::open(&db_path).expect("open raw contextdb");
    let variant_handles = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id",
        &params(vec![("id", Value::Uuid(embedding_id.as_uuid()))]),
    );
    let first_evidence_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $id",
        &params(vec![("id", Value::Uuid(first_evidence.as_uuid()))]),
    );
    let second_evidence_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $id",
        &params(vec![("id", Value::Uuid(second_evidence.as_uuid()))]),
    );
    let first_handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id",
        &params(vec![("target_id", Value::Uuid(first_evidence.as_uuid()))]),
    );
    let second_handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id",
        &params(vec![("target_id", Value::Uuid(second_evidence.as_uuid()))]),
    );
    let successful_evidence_id = results
        .iter()
        .find_map(|result| result.as_ref().ok())
        .and_then(|result| result.evidence_ids.first().copied());
    let loser_evidence_id = match successful_evidence_id {
        Some(id) if id == first_evidence => Some(second_evidence),
        Some(id) if id == second_evidence => Some(first_evidence),
        _ => None,
    };
    let first_hit = first_hit_id(&db, text_index(), text_vector(0));
    let second_hit = first_hit_id(&db, text_index(), text_vector(1));
    let successful_reachable = successful_evidence_id.is_some_and(|winner| {
        first_hit == Some(winner.as_uuid()) || second_hit == Some(winner.as_uuid())
    });
    let loser_leaked_to_vector_index = loser_evidence_id.is_some_and(|loser| {
        first_hit == Some(loser.as_uuid()) || second_hit == Some(loser.as_uuid())
    });

    assert!(
        results.iter().filter(|result| result.is_ok()).count() == 1
            && results.iter().any(|result| matches!(
                result,
                Err(context_graph::CgError::EngineFault {
                    source: context_graph::error::EngineError::UniqueViolation { .. }
                })
            ))
            && variant_handles == 1
            && first_evidence_rows + second_evidence_rows == 1
            && first_handle_rows + second_handle_rows == 1
            && successful_evidence_id.is_some()
            && loser_evidence_id.is_some()
            && successful_reachable
            && !loser_leaked_to_vector_index,
        "duplicate embedding handle under contention must roll back exactly one full observation/evidence/snapshot/handle/vector write, results={results:?}, variant_handles={variant_handles}, first_evidence_rows={first_evidence_rows}, second_evidence_rows={second_evidence_rows}, first_handle_rows={first_handle_rows}, second_handle_rows={second_handle_rows}, first_hit={first_hit:?}, second_hit={second_hit:?}, successful_evidence_id={successful_evidence_id:?}, loser_evidence_id={loser_evidence_id:?}, successful_reachable={successful_reachable}, loser_leaked_to_vector_index={loser_leaked_to_vector_index}"
    );
}

#[test]
fn engine_probe_evidence_unique_context_id_id_constraint_rejects_duplicate_insert() {
    let fixture = fixture("duplicate-evidence");
    let evidence_id = EvidenceId::new_v7();
    let seed = record_text_embedding(
        fixture.store.clone(),
        fixture.context_id,
        fixture.entity_id,
        evidence_id,
        EmbeddingId::new_v7(),
        0,
    );
    let observation_id = seed
        .as_ref()
        .ok()
        .map(|result| result.observation_id.as_uuid());
    let db_path = fixture.db_path.clone();
    let config = fixture.config.clone();
    let context_id = fixture.context_id;
    let entity_id = fixture.entity_id;
    drop(fixture.store);

    let db = Arc::new(Database::open(&db_path).expect("open raw contextdb"));
    let barrier = Arc::new(Barrier::new(3));
    let handles = [11_i64, 12_i64]
        .into_iter()
        .map(|supply_order| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                observation_id.map_or_else(
                    || {
                        Err(EngineError::NotFound(
                            "seed observation missing".to_string(),
                        ))
                    },
                    |observation_id| {
                        raw_duplicate_evidence_insert(
                            &db,
                            evidence_id,
                            observation_id,
                            context_id,
                            supply_order,
                        )
                    },
                )
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    let started = Instant::now();
    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer joined"))
        .collect::<Vec<_>>();
    let duration = started.elapsed();
    let raw_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE context_id = $context_id AND id = $id",
        &params(vec![
            ("context_id", Value::Uuid(context_id.as_uuid())),
            ("id", Value::Uuid(evidence_id.as_uuid())),
        ]),
    );
    drop(db);

    let app_store = Store::open(config).expect("reopen app store");
    let app_duplicate = record_text_embedding(
        app_store,
        context_id,
        entity_id,
        evidence_id,
        EmbeddingId::new_v7(),
        1,
    );

    assert!(
        seed.is_ok()
            && results.iter().all(|result| matches!(
                result,
                Err(EngineError::UniqueViolation { table, column })
                    if table == "evidence" && column == "id"
            ))
            && raw_rows == 1
            && matches!(
                app_duplicate,
                Err(context_graph::CgError::DuplicateEvidenceId { id }) if id == evidence_id
            )
            && duration > Duration::ZERO,
        "duplicate evidence raw inserts must return engine UniqueViolation while leaving one row, and the app path must map to DuplicateEvidenceId, seed={seed:?}, results={results:?}, raw_rows={raw_rows}, app_duplicate={app_duplicate:?}, duration={duration:?}"
    );
}

#[test]
fn reindex_space_via_database_insert_vector_does_not_fire_observation_trigger() {
    let fixture = fixture("reindex-trigger");
    let evidence_id = EvidenceId::new_v7();
    let seed = record_text_embedding(
        fixture.store.clone(),
        fixture.context_id,
        fixture.entity_id,
        evidence_id,
        EmbeddingId::new_v7(),
        0,
    );
    let db_path = fixture.db_path.clone();
    let context_id = fixture.context_id;
    let entity_id = fixture.entity_id;
    drop(fixture.store);

    let db = Arc::new(Database::open(&db_path).expect("open raw contextdb"));
    let trigger_counter = register_counting_impact_trigger(&db);
    let row_id = evidence_row_id(&db, evidence_id);
    let before_vector_write = trigger_counter.load(Ordering::SeqCst);
    let vector_write = row_id
        .ok_or_else(|| "missing seeded evidence row".to_string())
        .and_then(|row_id| {
            replace_vector(Arc::clone(&db), text_index(), row_id, text_vector(2))
                .map_err(|err| format!("{err:?}"))
        });
    let after_vector_write = trigger_counter.load(Ordering::SeqCst);
    let invalidations_after_vector_write = count(
        &db,
        "SELECT COUNT(*) AS n FROM invalidations WHERE context_id = $context_id AND status = $status",
        &params(vec![
            ("context_id", Value::Uuid(context_id.as_uuid())),
            ("status", Value::Text("pending".to_string())),
        ]),
    );
    let positive_control = raw_observation_insert(&db, context_id, entity_id);
    let after_observation_insert = trigger_counter.load(Ordering::SeqCst);

    assert!(
        seed.is_ok()
            && row_id.is_some()
            && vector_write.is_ok()
            && before_vector_write == 0
            && after_vector_write == before_vector_write
            && invalidations_after_vector_write == 0
            && positive_control
                .as_ref()
                .is_ok_and(|result| result.rows_affected == 1)
            && after_observation_insert == before_vector_write + 1,
        "vector-only reindex writes must not fire the observation trigger, while the observation positive control must fire exactly once, seed={seed:?}, row_id={row_id:?}, vector_write={vector_write:?}, before_vector_write={before_vector_write}, after_vector_write={after_vector_write}, invalidations_after_vector_write={invalidations_after_vector_write}, positive_control={positive_control:?}, after_observation_insert={after_observation_insert}"
    );
}
