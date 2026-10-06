use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chrono::Utc;
use context_graph::{
    ContextId, CreateContext, CreateEntity, DistanceMetric, EmbeddingId, EmbeddingSpaceBinding,
    EmbeddingSpaceDecl, EmbeddingTarget, EntityId, EntityType, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, ModelIdentity, PrecomputedEmbedding, RecordObservation,
    RetentionStatus, Store, StoreConfig, VectorPlacement,
};
use contextdb_core::{Error as EngineError, RowId, TxId, Value, VectorIndexRef};
use contextdb_engine::Database;
use contextdb_engine::memory_accounting::MemoryAccountant;
use contextdb_engine::plugin::{CommitSource, DatabasePlugin};
use contextdb_tx::WriteSet;
use contextdb_vector::test_seam::PauseWindow;
use serde_json::json;

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";
const VISION_SPACE: &str = "vigil_face_reid_v1";
const FACE_REFERENCE_SPACE: &str = "face_reference_v1";
const ENGINE_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const ENGINE_PROBE_HNSW_COMPLETION_TIMEOUT: Duration = Duration::from_secs(60);
const HNSW_PADDING_ROWS: usize = 1000;

struct ProbeFixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    config: StoreConfig,
    store: Store,
    context_id: ContextId,
    entity_id: EntityId,
}

#[derive(Default)]
struct StagedDuplicateHandleProbe {
    target: Mutex<Option<DuplicateHandleTarget>>,
    observed: Mutex<Vec<ObservedDuplicateHandleWrite>>,
    evidence_insert_queries: AtomicUsize,
    handle_insert_queries: AtomicUsize,
    first_full_precommit_seen: AtomicBool,
    first_full_precommit_open: AtomicBool,
    concurrent_writer_reached_handle_sql: AtomicBool,
}

/// Writers that share one evidence id; each is told apart by its vector.
#[derive(Clone)]
struct DuplicateHandleTarget {
    writes: Vec<(EvidenceId, Vec<f32>)>,
}

#[derive(Clone)]
struct ObservedDuplicateHandleWrite {
    phase: DuplicateHandleObservationPhase,
    source: CommitSource,
    evidence_id: EvidenceId,
    evidence_row_id: Option<RowId>,
    vector: Vec<f32>,
    saw_observation_row: bool,
    saw_evidence_row: bool,
    saw_snapshot_row: bool,
    saw_snapshot_observation_row: bool,
    saw_handle_row: bool,
    saw_vector_insert: bool,
    staged_tx_count: usize,
    failed_unique_violation: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DuplicateHandleObservationPhase {
    PreCommit,
    CommitFailed,
}

impl std::fmt::Debug for ObservedDuplicateHandleWrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservedDuplicateHandleWrite")
            .field("phase", &self.phase)
            .field("source", &self.source)
            .field("evidence_id", &self.evidence_id)
            .field("evidence_row_id", &self.evidence_row_id)
            .field("vector_len", &self.vector.len())
            .field("saw_observation_row", &self.saw_observation_row)
            .field("saw_evidence_row", &self.saw_evidence_row)
            .field("saw_snapshot_row", &self.saw_snapshot_row)
            .field(
                "saw_snapshot_observation_row",
                &self.saw_snapshot_observation_row,
            )
            .field("saw_handle_row", &self.saw_handle_row)
            .field("saw_vector_insert", &self.saw_vector_insert)
            .field("staged_tx_count", &self.staged_tx_count)
            .field("failed_unique_violation", &self.failed_unique_violation)
            .finish()
    }
}

impl StagedDuplicateHandleProbe {
    fn arm(&self, target: DuplicateHandleTarget) {
        *self.target.lock().expect("target") = Some(target);
        self.observed.lock().expect("observed").clear();
        self.evidence_insert_queries.store(0, Ordering::SeqCst);
        self.handle_insert_queries.store(0, Ordering::SeqCst);
        self.first_full_precommit_seen
            .store(false, Ordering::SeqCst);
        self.first_full_precommit_open
            .store(false, Ordering::SeqCst);
        self.concurrent_writer_reached_handle_sql
            .store(false, Ordering::SeqCst);
    }

    fn full_staged_write_for(&self, vector: &[f32]) -> Option<ObservedDuplicateHandleWrite> {
        self.full_staged_write_for_phase(vector, DuplicateHandleObservationPhase::PreCommit)
    }

    fn failed_full_staged_write_for(&self, vector: &[f32]) -> Option<ObservedDuplicateHandleWrite> {
        self.full_staged_write_for_phase(vector, DuplicateHandleObservationPhase::CommitFailed)
            .filter(|write| write.failed_unique_violation)
    }

    fn full_staged_write_for_phase(
        &self,
        vector: &[f32],
        phase: DuplicateHandleObservationPhase,
    ) -> Option<ObservedDuplicateHandleWrite> {
        self.observed
            .lock()
            .expect("observed")
            .iter()
            .find(|write| {
                write.phase == phase
                    && write.source == CommitSource::User
                    && write.vector == vector
                    && write.saw_observation_row
                    && write.saw_evidence_row
                    && write.saw_snapshot_row
                    && write.saw_snapshot_observation_row
                    && write.saw_handle_row
                    && write.saw_vector_insert
                    && write.staged_tx_count == 1
            })
            .cloned()
    }

    fn observed_phase_for(&self, vector: &[f32], phase: DuplicateHandleObservationPhase) -> bool {
        self.observed.lock().expect("observed").iter().any(|write| {
            write.phase == phase && write.source == CommitSource::User && write.vector == vector
        })
    }

    fn observed_writes(&self) -> Vec<ObservedDuplicateHandleWrite> {
        self.observed.lock().expect("observed").clone()
    }

    fn evidence_insert_query_count(&self) -> usize {
        self.evidence_insert_queries.load(Ordering::SeqCst)
    }

    fn handle_insert_query_count(&self) -> usize {
        self.handle_insert_queries.load(Ordering::SeqCst)
    }

    fn concurrent_writer_reached_handle_sql(&self) -> bool {
        self.concurrent_writer_reached_handle_sql
            .load(Ordering::SeqCst)
    }

    fn observe_write_set(
        &self,
        ws: &WriteSet,
        source: CommitSource,
        phase: DuplicateHandleObservationPhase,
        error: Option<&EngineError>,
    ) {
        let Some(target) = self.target.lock().expect("target").clone() else {
            return;
        };

        let failed_unique_violation = matches!(
            error,
            Some(EngineError::UniqueViolation { table, column })
                if table == "evidence" && column == "id"
        );

        for (evidence_id, vector) in &target.writes {
            let evidence_row = write_set_row_for_uuid(ws, "evidence", "id", evidence_id.as_uuid());
            let observation_id = write_set_uuid_value_for_uuid(
                ws,
                "evidence",
                "id",
                evidence_id.as_uuid(),
                "observation_id",
            );
            let observation_row =
                observation_id.and_then(|id| write_set_row_for_uuid(ws, "observations", "id", id));
            let snapshot_observation_row = observation_id.and_then(|id| {
                write_set_row_for_uuid(ws, "entity_snapshot_observations", "observation_id", id)
            });
            let snapshot_id = observation_id.and_then(|id| {
                write_set_uuid_value_for_uuid(
                    ws,
                    "entity_snapshot_observations",
                    "observation_id",
                    id,
                    "snapshot_id",
                )
            });
            let snapshot_row =
                snapshot_id.and_then(|id| write_set_row_for_uuid(ws, "entity_snapshots", "id", id));
            let handle_row = write_set_row_for_uuid(
                ws,
                "evidence_embeddings",
                "target_id",
                evidence_id.as_uuid(),
            );
            let vector_tx = handle_row.and_then(|(row_id, _)| {
                write_set_vector_tx(ws, row_id, "evidence_embeddings", "vector_text", vector)
            });
            let txs = [
                observation_row.map(|(_, tx)| tx),
                evidence_row.map(|(_, tx)| tx),
                snapshot_row.map(|(_, tx)| tx),
                snapshot_observation_row.map(|(_, tx)| tx),
                handle_row.map(|(_, tx)| tx),
                vector_tx,
            ]
            .into_iter()
            .flatten()
            .collect::<BTreeSet<_>>();
            // The writers share the evidence id, so only this writer's vector attributes the write set.
            if vector_tx.is_some() {
                self.observed
                    .lock()
                    .expect("observed")
                    .push(ObservedDuplicateHandleWrite {
                        phase,
                        source,
                        evidence_id: *evidence_id,
                        evidence_row_id: evidence_row.map(|(row_id, _)| row_id),
                        vector: vector.clone(),
                        saw_observation_row: observation_row.is_some(),
                        saw_evidence_row: evidence_row.is_some(),
                        saw_snapshot_row: snapshot_row.is_some(),
                        saw_snapshot_observation_row: snapshot_observation_row.is_some(),
                        saw_handle_row: handle_row.is_some(),
                        saw_vector_insert: vector_tx.is_some(),
                        staged_tx_count: txs.len(),
                        failed_unique_violation,
                    });
            }
            if phase == DuplicateHandleObservationPhase::PreCommit
                && observation_row.is_some()
                && evidence_row.is_some()
                && snapshot_row.is_some()
                && snapshot_observation_row.is_some()
                && handle_row.is_some()
                && vector_tx.is_some()
                && txs.len() == 1
                && self
                    .first_full_precommit_seen
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                self.first_full_precommit_open.store(true, Ordering::SeqCst);
                for _ in 0..200 {
                    if self.handle_insert_queries.load(Ordering::SeqCst) >= 2 {
                        self.concurrent_writer_reached_handle_sql
                            .store(true, Ordering::SeqCst);
                        break;
                    }
                    thread::sleep(Duration::from_millis(1));
                }
                self.first_full_precommit_open
                    .store(false, Ordering::SeqCst);
            }
        }
    }
}

impl DatabasePlugin for StagedDuplicateHandleProbe {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if self.target.lock().expect("target").is_none() {
            return Ok(());
        }
        let sql = sql.trim_start();
        if sql.starts_with("INSERT INTO evidence ") {
            self.evidence_insert_queries.fetch_add(1, Ordering::SeqCst);
        }
        if sql.starts_with("INSERT INTO evidence_embeddings ") {
            let count = self.handle_insert_queries.fetch_add(1, Ordering::SeqCst) + 1;
            if count >= 2 && self.first_full_precommit_open.load(Ordering::SeqCst) {
                self.concurrent_writer_reached_handle_sql
                    .store(true, Ordering::SeqCst);
            }
        }
        Ok(())
    }

    fn pre_commit(&self, ws: &WriteSet, source: CommitSource) -> contextdb_core::Result<()> {
        self.observe_write_set(ws, source, DuplicateHandleObservationPhase::PreCommit, None);
        Ok(())
    }

    fn commit_failed(&self, ws: &WriteSet, source: CommitSource, error: &EngineError) {
        self.observe_write_set(
            ws,
            source,
            DuplicateHandleObservationPhase::CommitFailed,
            Some(error),
        );
    }
}

fn fixture(name: &str) -> ProbeFixture {
    fixture_with_plugin(name, None)
}

fn fixture_with_plugin(name: &str, plugin: Option<Arc<dyn DatabasePlugin>>) -> ProbeFixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let config = StoreConfig {
        db_path: db_path.clone(),
        extra_embedding_spaces: vec![vision_space(), face_reference_space()],
        ..Default::default()
    };
    let store = if let Some(plugin) = plugin {
        Store::open_with_database_plugin(config.clone(), plugin).expect("open with plugin")
    } else {
        Store::open(config.clone()).expect("open")
    };
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
        placement: Some(VectorPlacement::TravelsWithItem),
        media_reachable_on_every_node: false,
        producer: probe_identity(512),
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
        placement: Some(VectorPlacement::TravelsWithItem),
        media_reachable_on_every_node: false,
        producer: ModelIdentity {
            model: "face-reference".to_string(),
            weights_revision: "1".to_string(),
            preparation: "1".to_string(),
            dimension: 512,
        },
    }
}

/// The caller-filled vision space's one model: `producer("record")`'s own fields.
fn probe_identity(dimension: u32) -> ModelIdentity {
    ModelIdentity {
        model: "probe-model".to_string(),
        weights_revision: "record".to_string(),
        preparation: "1".to_string(),
        dimension,
    }
}

/// A supply of `space`'s own model: the built-in text space's, read from a store, or the probe's.
fn supply_producer(space: &str) -> EvidenceProducer {
    static TEXT_MODEL: std::sync::OnceLock<EvidenceProducer> = std::sync::OnceLock::new();
    if space != TEXT_SPACE {
        return producer("record");
    }
    let model = TEXT_MODEL.get_or_init(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Store::open(StoreConfig {
            db_path: dir.path().join("producer.db"),
            ..Default::default()
        })
        .expect("open to read the text space's model");
        EvidenceProducer::from(
            &store
                .space_producer(TEXT_SPACE)
                .expect("read the text space's model")
                .expect("the built-in text space is declared"),
        )
    });
    EvidenceProducer {
        system: "probe-record".to_string(),
        ..model.clone()
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
    VectorIndexRef::new("evidence_embeddings", "vector_text")
}

fn vision_index() -> VectorIndexRef {
    VectorIndexRef::new("evidence_embeddings_travelling", "vector_vision")
}

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn count(db: &Database, sql: &str, params: &HashMap<String, Value>) -> i64 {
    let result = db.execute(sql, params).expect("count query");
    count_from_result(result)
}

fn store_count(store: &Store, sql: &str, params: &HashMap<String, Value>) -> i64 {
    let result = store
        .execute_sql_for_test(sql, params)
        .expect("count query");
    count_from_result(result)
}

fn count_from_result(result: contextdb_engine::QueryResult) -> i64 {
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

fn raw_evidence_vector_row_insert(
    db: &Database,
    row_id: EmbeddingId,
    context_id: ContextId,
    target_id: EvidenceId,
    producer_label: &str,
    created_at: TxId,
) -> contextdb_core::Result<contextdb_engine::QueryResult> {
    db.execute(
        "INSERT INTO evidence_embeddings (id, context_id, target_id, embedding_space_id, producer_key, produced_by, producer_system, created_at) VALUES ($id, $context_id, $target_id, $space, $producer_key, $producer, $producer_system, $created_at)",
        &params(vec![
            ("id", Value::Uuid(row_id.as_uuid())),
            ("context_id", Value::Uuid(context_id.as_uuid())),
            ("target_id", Value::Uuid(target_id.as_uuid())),
            ("space", Value::Text(TEXT_SPACE.to_string())),
            ("producer_key", Value::Text(format!("raw-{producer_label}"))),
            (
                "producer",
                Value::Json(serde_json::to_value(producer(producer_label)).expect("json")),
            ),
            ("producer_system", Value::Text("cg-edge".to_string())),
            ("created_at", Value::TxId(created_at)),
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
            let (count, vector_count) = visibility_counts(&reader_store, &reader_writes);
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
    let (final_count, final_vector_count) = visibility_counts(&fixture.store, &final_writes);
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
    record_embedding_with_state_delta(store, context_id, entity_id, write, BTreeMap::new())
}

fn record_embedding_with_state_delta(
    store: Store,
    context_id: ContextId,
    entity_id: EntityId,
    write: EvidenceWrite,
    state_delta: BTreeMap<String, Option<serde_json::Value>>,
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
        state_delta,
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: write.embedding_id,
            embedding_space_id: write.embedding_space_id.to_string(),
            vector: write.vector,
            target: EmbeddingTarget::Evidence(write.evidence_id),
            produced_by: supply_producer(write.embedding_space_id),
        }],
    })
}

fn record_text_embedding_with_state_delta(
    store: Store,
    context_id: ContextId,
    entity_id: EntityId,
    evidence_id: EvidenceId,
    embedding_id: EmbeddingId,
    axis: usize,
) -> context_graph::Result<context_graph::RecordObservationResult> {
    let mut state_delta = BTreeMap::new();
    state_delta.insert(
        "atomic_rollback_probe".to_string(),
        Some(serde_json::Value::String(evidence_id.as_uuid().to_string())),
    );
    record_embedding_with_state_delta(
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
        state_delta,
    )
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

fn visibility_counts(store: &Store, writes: &[EvidenceWrite]) -> (usize, usize) {
    let evidence_ids = writes
        .iter()
        .map(|write| write.evidence_id.as_uuid())
        .collect::<Vec<_>>();
    let queries = writes
        .iter()
        .map(|write| write.vector.clone())
        .collect::<Vec<_>>();
    store
        .evidence_vector_visibility_counts_for_test(&evidence_ids, text_index(), &queries)
        .expect("visibility counts")
}

fn first_hit_id(db: &Database, index: VectorIndexRef, query: Vec<f32>) -> Option<uuid::Uuid> {
    first_hit_id_and_score(db, index, query).map(|(id, _)| id)
}

fn first_hit_id_and_score(
    db: &Database,
    index: VectorIndexRef,
    query: Vec<f32>,
) -> Option<(uuid::Uuid, f32)> {
    let table = index.table.clone();
    let hits = db
        .query_vector(index, &query, 1, None, db.snapshot())
        .expect("vector query");
    let (row_id, score) = *hits.first()?;
    let rows = db.scan(&table, db.snapshot()).expect("scan vector table");
    rows.into_iter().find_map(|row| {
        (row.row_id == row_id)
            .then(|| row.values.get("target_id").cloned())
            .flatten()
            .and_then(|value| match value {
                Value::Uuid(id) => Some(id),
                _ => None,
            })
            .map(|id| (id, score))
    })
}

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

fn write_set_uuid_value_for_uuid(
    ws: &WriteSet,
    table: &str,
    lookup_column: &str,
    lookup_value: uuid::Uuid,
    value_column: &str,
) -> Option<uuid::Uuid> {
    ws.relational_inserts
        .iter()
        .filter(|(table_name, _)| table_name == table)
        .find_map(|(_, row)| {
            (row.values.get(lookup_column) == Some(&Value::Uuid(lookup_value)))
                .then(|| row.values.get(value_column))
                .flatten()
                .and_then(|value| match value {
                    Value::Uuid(id) => Some(*id),
                    _ => None,
                })
        })
}

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

/// The row holding the item's vector in `table`.
fn vector_row_id(db: &Database, table: &str, evidence_id: EvidenceId) -> Option<RowId> {
    db.scan(table, db.snapshot())
        .expect("scan vector table")
        .into_iter()
        .find_map(|row| {
            let id = row.values.get("target_id").and_then(|value| match value {
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
    let producer_json = Value::Json(serde_json::to_value(producer("padding")).expect("json"));
    for i in 0..HNSW_PADDING_ROWS {
        let evidence_id = uuid::Uuid::now_v7();
        db.execute(
            "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, retention_status, supply_order) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $retention_status, $supply_order)",
            &params(vec![
                ("id", Value::Uuid(evidence_id)),
                ("observation_id", Value::Uuid(observation_id)),
                ("context_id", Value::Uuid(context_id.as_uuid())),
                ("kind", Value::Text("text".to_string())),
                ("source_ref", Value::Text(format!("cas://sha256:{i:064x}"))),
                ("retention_status", Value::Text("retained_external".to_string())),
                ("supply_order", Value::Int64(10_000 + i as i64)),
            ]),
        )
        .map_err(|err| format!("{err:?}"))?;
        for (table, space, column, padding, head) in [
            (
                "evidence_embeddings",
                TEXT_SPACE,
                "vector_text",
                text_vector(100),
                None,
            ),
            (
                "evidence_embeddings_travelling",
                VISION_SPACE,
                "vector_vision",
                vector(100),
                Some(true),
            ),
        ] {
            let head_column = if head.is_some() { ", is_head" } else { "" };
            let head_value = if head.is_some() { ", $is_head" } else { "" };
            let mut values = vec![
                ("id", Value::Uuid(uuid::Uuid::now_v7())),
                ("context_id", Value::Uuid(context_id.as_uuid())),
                ("target_id", Value::Uuid(evidence_id)),
                ("space", Value::Text(space.to_string())),
                ("producer_key", Value::Text("padding".to_string())),
                ("producer", producer_json.clone()),
                ("producer_system", Value::Text("probe-padding".to_string())),
                ("created_at", Value::TxId(TxId(1))),
                ("vector", Value::Vector(padding)),
            ];
            if let Some(head) = head {
                values.push(("is_head", Value::Bool(head)));
            }
            db.execute(
                &format!(
                    "INSERT INTO {table} (id, context_id, target_id, embedding_space_id, producer_key, produced_by, producer_system, created_at, {column}{head_column}) VALUES ($id, $context_id, $target_id, $space, $producer_key, $producer, $producer_system, $created_at, $vector{head_value})"
                ),
                &params(values),
            )
            .map_err(|err| format!("{err:?}"))?;
        }
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
    let vision_row_id = vector_row_id(&db, "evidence_embeddings_travelling", vision_evidence);
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
    let text_store = vector_store.clone();
    thread::spawn(move || {
        let reader = text_db
            .declared_reader_memory()
            .expect("admit the maintenance fixture reader");
        let mut executing_owner = contextdb_engine::vector_read_memory::admission(reader);
        let result = text_store
            .run_hnsw_maintenance_for_index_for_test(
                &text_index(),
                Arc::new(MemoryAccountant::no_limit()),
                &mut executing_owner,
            )
            .map_err(|error| format!("{error:?}"))
            .and_then(|_| {
                first_hit_id(&text_db, text_index(), text_vector(7))
                    .ok_or_else(|| "missing text hit".to_string())
            });
        text_done_tx.send(result).expect("send text result");
    });
    let text_reached = text_pause.wait_until_reached(ENGINE_PROBE_TIMEOUT);

    let (vision_done_tx, vision_done_rx) = mpsc::channel();
    let vision_db = Arc::clone(&db);
    let vision_store = vector_store.clone();
    let vision_row_for_thread = vision_row_id;
    thread::spawn(move || {
        let reader = vision_db
            .declared_reader_memory()
            .expect("admit the maintenance fixture reader");
        let mut executing_owner = contextdb_engine::vector_read_memory::admission(reader);
        let result = vision_row_for_thread
            .ok_or_else(|| "missing vision vector row".to_string())
            .and_then(|row_id| {
                replace_vector(Arc::clone(&vision_db), vision_index(), row_id, vector(4))
                    .map_err(|err| format!("{err:?}"))
            })
            .and_then(|()| {
                vision_store
                    .run_hnsw_maintenance_for_index_for_test(
                        &vision_index(),
                        Arc::new(MemoryAccountant::no_limit()),
                        &mut executing_owner,
                    )
                    .map_err(|error| format!("{error:?}"))?;
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
        Err(TryRecvError::Empty) => text_done_rx
            .recv_timeout(ENGINE_PROBE_HNSW_COMPLETION_TIMEOUT)
            .ok(),
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
        "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE target_id = $target_id",
        &params(vec![("target_id", Value::Uuid(first_evidence.as_uuid()))]),
    ) + count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE target_id = $target_id",
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
        "same VectorIndexRef writes must serialize without torn index state, persist both vector rows, and expose one-then-both MVCC progression through both row reads and vector search during overlap, results={results:?}, first_hit={first_hit:?}, second_hit={second_hit:?}, evidence_rows={evidence_rows}, handle_rows={handle_rows}, observed_counts={observed_counts:?}, observed_vector_counts={observed_vector_counts:?}, monotonic_counts={monotonic_counts}, monotonic_vector_counts={monotonic_vector_counts}, row_and_vector_snapshots_move_together={row_and_vector_snapshots_move_together}, duration={duration:?}"
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
            raw_evidence_vector_row_insert(
                &db,
                handle_id,
                context_id,
                target_evidence,
                &format!("cross-context-{}", handle_id.as_uuid()),
                txid,
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
        "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE id = $id",
        &params(vec![("id", Value::Uuid(valid_handle.as_uuid()))]),
    );
    let invalid_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE id = $id",
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
                }) if child_table == "evidence_embeddings"
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
    let staging_probe = Arc::new(StagedDuplicateHandleProbe::default());
    let plugin: Arc<dyn DatabasePlugin> = staging_probe.clone();
    let fixture = fixture_with_plugin("atomic-rollback", Some(plugin));
    // Both writers claim one evidence id, so exactly one commit can land; each writes its own
    // observation, snapshot, vector row and vector, told apart by its vector.
    let shared_evidence = EvidenceId::new_v7();
    let second_entity = fixture
        .store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: "probe entity duplicate-handle peer".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: fixture.context_id,
        })
        .expect("second entity");
    let barrier = Arc::new(Barrier::new(3));
    let first_entity_id = fixture.entity_id;
    let second_entity_id = second_entity.id;
    let writes = [(first_entity_id, 0_usize), (second_entity_id, 1_usize)];
    let context_params = params(vec![(
        "context_id",
        Value::Uuid(fixture.context_id.as_uuid()),
    )]);
    let entity_params = |entity_id: EntityId| {
        params(vec![
            ("context_id", Value::Uuid(fixture.context_id.as_uuid())),
            ("entity_id", Value::Uuid(entity_id.as_uuid())),
        ])
    };
    let snapshot_count_sql = "SELECT COUNT(*) AS n FROM entity_snapshots WHERE context_id = $context_id AND entity_id = $entity_id";
    let snapshot_observation_count_sql = "SELECT COUNT(*) AS n FROM entity_snapshot_observations WHERE context_id = $context_id AND entity_id = $entity_id";
    let pre_observation_rows = store_count(
        &fixture.store,
        "SELECT COUNT(*) AS n FROM observations WHERE context_id = $context_id",
        &context_params,
    );
    let pre_snapshot_rows = store_count(
        &fixture.store,
        "SELECT COUNT(*) AS n FROM entity_snapshots WHERE context_id = $context_id",
        &context_params,
    );
    let pre_snapshot_observation_rows = store_count(
        &fixture.store,
        "SELECT COUNT(*) AS n FROM entity_snapshot_observations WHERE context_id = $context_id",
        &context_params,
    );
    let pre_entity_rows = writes.map(|(entity_id, _)| {
        (
            store_count(
                &fixture.store,
                snapshot_count_sql,
                &entity_params(entity_id),
            ),
            store_count(
                &fixture.store,
                snapshot_observation_count_sql,
                &entity_params(entity_id),
            ),
        )
    });
    let fixture_context_id = fixture.context_id;
    staging_probe.arm(DuplicateHandleTarget {
        writes: writes
            .iter()
            .map(|(_, axis)| (shared_evidence, text_vector(*axis)))
            .collect(),
    });
    let handles = writes
        .into_iter()
        .map(|(entity_id, axis)| {
            let writer = fixture.store.clone();
            let barrier = Arc::clone(&barrier);
            let context_id = fixture.context_id;
            thread::spawn(move || {
                barrier.wait();
                record_text_embedding_with_state_delta(
                    writer,
                    context_id,
                    entity_id,
                    shared_evidence,
                    EmbeddingId::new_v7(),
                    axis,
                )
            })
        })
        .collect::<Vec<_>>();
    let started = Instant::now();
    barrier.wait();
    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer joined"))
        .collect::<Vec<_>>();
    let duration = started.elapsed();
    let db_path = fixture.db_path.clone();
    drop(fixture.store);
    let db = Database::open(&db_path).expect("open raw contextdb");
    let winner = results.iter().position(Result::is_ok);
    let loser = winner.map(|winner| 1 - winner);
    let winner_vector = winner.map(|winner| text_vector(writes[winner].1));
    let loser_vector = loser.map(|loser| text_vector(writes[loser].1));
    let winner_row_id = winner
        .and_then(|winner| results[winner].as_ref().ok())
        .and_then(|result| result.embedding_ids.first().copied());
    let evidence_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $id",
        &params(vec![("id", Value::Uuid(shared_evidence.as_uuid()))]),
    );
    let vector_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE target_id = $target_id",
        &params(vec![("target_id", Value::Uuid(shared_evidence.as_uuid()))]),
    );
    let winner_row_rows = winner_row_id.map(|row_id| {
        count(
            &db,
            "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE id = $id AND target_id = $target_id",
            &params(vec![
                ("id", Value::Uuid(row_id.as_uuid())),
                ("target_id", Value::Uuid(shared_evidence.as_uuid())),
            ]),
        )
    });
    let observation_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM observations WHERE context_id = $context_id",
        &context_params,
    );
    let snapshot_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_snapshots WHERE context_id = $context_id",
        &context_params,
    );
    let snapshot_observation_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_snapshot_observations WHERE context_id = $context_id",
        &context_params,
    );
    let entity_rows = writes.map(|(entity_id, _)| {
        (
            count(&db, snapshot_count_sql, &entity_params(entity_id)),
            count(
                &db,
                snapshot_observation_count_sql,
                &entity_params(entity_id),
            ),
        )
    });
    let winner_entity_progress = match (winner, loser) {
        (Some(winner), Some(loser)) => {
            entity_rows[winner].0 == pre_entity_rows[winner].0 + 1
                && entity_rows[winner].1 == pre_entity_rows[winner].1 + 1
                && entity_rows[loser] == pre_entity_rows[loser]
        }
        _ => false,
    };
    let store_evidence_insert_query_count = staging_probe.evidence_insert_query_count();
    let store_handle_insert_query_count = staging_probe.handle_insert_query_count();
    let store_concurrent_writer_reached_handle_sql =
        staging_probe.concurrent_writer_reached_handle_sql();
    let raw_duplicate_created_at = db.committed_watermark();
    let raw_duplicate_handle_insert = winner_row_id.map(|row_id| {
        raw_evidence_vector_row_insert(
            &db,
            row_id,
            fixture_context_id,
            shared_evidence,
            "raw-duplicate",
            raw_duplicate_created_at,
        )
    });
    let raw_duplicate_id_rejected = matches!(
        raw_duplicate_handle_insert,
        Some(Err(EngineError::UniqueViolation {
            ref table,
            ref column,
        })) if table == "evidence_embeddings" && column == "id"
    );
    let winner_rows_after_raw_duplicate = winner_row_id.map(|row_id| {
        count(
            &db,
            "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE id = $id",
            &params(vec![("id", Value::Uuid(row_id.as_uuid()))]),
        )
    });
    let successful_reachable = winner_vector.as_ref().is_some_and(|query| {
        first_hit_id_and_score(&db, text_index(), query.clone())
            .is_some_and(|(id, score)| id == shared_evidence.as_uuid() && score > 0.99)
    });
    let successful_staged_full_write = winner_vector
        .as_ref()
        .and_then(|vector| staging_probe.full_staged_write_for(vector))
        .is_some();
    let failed_staged_full_write = loser_vector
        .as_ref()
        .and_then(|vector| staging_probe.failed_full_staged_write_for(vector))
        .is_some();
    let loser_precommit_observed = loser_vector.as_ref().is_some_and(|vector| {
        staging_probe.observed_phase_for(vector, DuplicateHandleObservationPhase::PreCommit)
    });
    let loser_orphan_vector_leaked = loser_vector.as_ref().is_some_and(|query| {
        db.query_vector(text_index(), query, 10, None, db.snapshot())
            .expect("raw vector leak query")
            .into_iter()
            .any(|(_, score)| score > 0.99)
    });
    let observed_staged_writes = staging_probe.observed_writes();

    assert!(
        results.iter().filter(|result| result.is_ok()).count() == 1
            && results.iter().any(|result| matches!(
                result,
                Err(context_graph::CgError::DuplicateEvidenceId { id }) if *id == shared_evidence
            ))
            && evidence_rows == 1
            && vector_rows == 1
            && winner_row_rows == Some(1)
            && observation_rows == pre_observation_rows + 1
            && snapshot_rows == pre_snapshot_rows + 1
            && snapshot_observation_rows == pre_snapshot_observation_rows + 1
            && winner_entity_progress
            && raw_duplicate_id_rejected
            && winner_rows_after_raw_duplicate == Some(1)
            && successful_staged_full_write
            && failed_staged_full_write
            && successful_reachable
            && !loser_orphan_vector_leaked
            && !loser_precommit_observed
            && store_evidence_insert_query_count == 2
            && store_handle_insert_query_count == 2
            && store_concurrent_writer_reached_handle_sql
            && duration.as_millis() >= 1,
        "a duplicate evidence id under contention must commit exactly one writer: stage its observation/evidence/snapshot/vector-row/vector write in one tx, expose the losing failed WriteSet with the same full write and the structured evidence UniqueViolation before pre_commit, report DuplicateEvidenceId to the loser, keep the winner's lineage only, and roll back the losing write without committed-row or raw-vector leaks while the second Store writer reaches its vector-row SQL before the first commit returns; a raw duplicate vector-row id must also fail through the durable contextdb constraint; results={results:?}, evidence_rows={evidence_rows}, vector_rows={vector_rows}, winner_row_id={winner_row_id:?}, winner_row_rows={winner_row_rows:?}, raw_duplicate_handle_insert={raw_duplicate_handle_insert:?}, raw_duplicate_id_rejected={raw_duplicate_id_rejected}, winner_rows_after_raw_duplicate={winner_rows_after_raw_duplicate:?}, observation_rows={observation_rows}, pre_observation_rows={pre_observation_rows}, snapshot_rows={snapshot_rows}, pre_snapshot_rows={pre_snapshot_rows}, snapshot_observation_rows={snapshot_observation_rows}, pre_snapshot_observation_rows={pre_snapshot_observation_rows}, entity_rows={entity_rows:?}, pre_entity_rows={pre_entity_rows:?}, winner_entity_progress={winner_entity_progress}, successful_staged_full_write={successful_staged_full_write}, failed_staged_full_write={failed_staged_full_write}, loser_precommit_observed={loser_precommit_observed}, observed_staged_writes={observed_staged_writes:?}, successful_reachable={successful_reachable}, loser_orphan_vector_leaked={loser_orphan_vector_leaked}, store_evidence_insert_query_count={store_evidence_insert_query_count}, store_handle_insert_query_count={store_handle_insert_query_count}, store_concurrent_writer_reached_handle_sql={store_concurrent_writer_reached_handle_sql}, duration={duration:?}"
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
    let row_id = vector_row_id(&db, "evidence_embeddings", evidence_id);
    let before_vector_write = trigger_counter.load(Ordering::SeqCst);
    let vector_write = row_id
        .ok_or_else(|| "missing seeded vector row".to_string())
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
