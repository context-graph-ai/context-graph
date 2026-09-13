use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chrono::Utc;
use context_graph::Result as CgResult;
use context_graph::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use context_graph::{
    CgError, CreateContext, CreateDecision, CreateEntity, CreateIntention, DistanceMetric,
    EmbedderRegistration, EmbeddingId, EmbeddingSpace, EmbeddingSpaceBinding, EmbeddingTarget,
    EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, IntentionOrigin,
    IntentionStatus, PrecomputedEmbedding, RecordObservation, ReindexEvent, ReindexOptions,
    RetentionStatus, Store, StoreConfig,
};
use contextdb_core::{Value, VectorIndexRef};
use contextdb_engine::Database;

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";
const IMAGE_SPACE: &str = "image_clip_v1_4";
const UNBOUND_SPACE: &str = "unbound_custom_space_v1";

#[derive(Clone)]
struct RecordingConcurrencyEmbedder;

#[derive(Default)]
struct BlockingReindexControl {
    state: Mutex<BlockingReindexState>,
    changed: Condvar,
}

#[derive(Default)]
struct BlockingReindexState {
    calls: usize,
    blocked_on_second_row: bool,
    release: bool,
}

impl BlockingReindexControl {
    fn wait_until_calls_at_least(&self, expected_calls: usize) -> bool {
        let deadline = Instant::now() + Duration::from_millis(500);
        let mut state = self.state.lock().expect("blocking state");
        while state.calls < expected_calls {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let timeout = deadline.saturating_duration_since(now);
            let (next_state, wait) = self
                .changed
                .wait_timeout(state, timeout)
                .expect("call-count wait");
            state = next_state;
            if wait.timed_out() && state.calls < expected_calls {
                return false;
            }
        }
        true
    }

    fn wait_until_blocked_on_second_row(&self) -> bool {
        let deadline = Instant::now() + Duration::from_millis(500);
        let mut state = self.state.lock().expect("blocking state");
        while !state.blocked_on_second_row {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let timeout = deadline.saturating_duration_since(now);
            let (next_state, wait) = self
                .changed
                .wait_timeout(state, timeout)
                .expect("blocking wait");
            state = next_state;
            if wait.timed_out() && !state.blocked_on_second_row {
                return false;
            }
        }
        true
    }

    fn release(&self) {
        let mut state = self.state.lock().expect("blocking state");
        state.release = true;
        self.changed.notify_all();
    }
}

#[derive(Clone)]
struct BlockingSecondRowConcurrencyEmbedder {
    control: Arc<BlockingReindexControl>,
}

impl Embedder for BlockingSecondRowConcurrencyEmbedder {
    fn embed(&self, input: EmbeddingInput) -> CgResult<EmbeddingOutput> {
        if matches!(input, EmbeddingInput::Text(_)) {
            let mut state = self.control.state.lock().expect("blocking state");
            state.calls += 1;
            if state.calls == 2 {
                state.blocked_on_second_row = true;
                self.control.changed.notify_all();
                while !state.release {
                    state = self.control.changed.wait(state).expect("blocking release");
                }
            }
        }
        RecordingConcurrencyEmbedder.embed(input)
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        RecordingConcurrencyEmbedder.supported_inputs()
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        RecordingConcurrencyEmbedder.embedding_space()
    }
}

#[derive(Clone)]
struct CountingVisionConcurrencyEmbedder {
    control: Arc<BlockingReindexControl>,
}

#[derive(Clone)]
struct PhaseRecordingConcurrencyEmbedder {
    producer_version: &'static str,
    vector_salt: u64,
    calls: Arc<Mutex<Vec<PhaseReindexCall>>>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct PhaseReindexCall {
    text: String,
    vector: Vec<f32>,
}

impl Embedder for PhaseRecordingConcurrencyEmbedder {
    fn embed(&self, input: EmbeddingInput) -> CgResult<EmbeddingOutput> {
        match input {
            EmbeddingInput::Text(text) => {
                let vector = {
                    let mut calls = self.calls.lock().expect("phase call log");
                    let vector = phase_reindexed_vector(self.vector_salt, calls.len());
                    calls.push(PhaseReindexCall {
                        text: text.clone(),
                        vector: vector.clone(),
                    });
                    vector
                };
                Ok(EmbeddingOutput {
                    embedding_space_id: TEXT_SPACE.to_string(),
                    dimension: 768,
                    producer: producer_for_space(TEXT_SPACE, self.producer_version),
                    vector,
                })
            }
            EmbeddingInput::ImageFrame { .. } | EmbeddingInput::ImageBytes(..) => {
                Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::ImageFrame,
                })
            }
            EmbeddingInput::AudioSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::AudioSegment,
            }),
            EmbeddingInput::VideoSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::VideoSegment,
            }),
            EmbeddingInput::StructuredSignal(_) => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::StructuredSignal,
            }),
        }
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        text_space()
    }
}

impl Embedder for CountingVisionConcurrencyEmbedder {
    fn embed(&self, input: EmbeddingInput) -> CgResult<EmbeddingOutput> {
        if matches!(input, EmbeddingInput::ImageFrame { .. }) {
            let mut state = self.control.state.lock().expect("vision call state");
            state.calls += 1;
            self.control.changed.notify_all();
        }
        RecordingVisionConcurrencyEmbedder.embed(input)
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        RecordingVisionConcurrencyEmbedder.supported_inputs()
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        RecordingVisionConcurrencyEmbedder.embedding_space()
    }
}

impl Embedder for RecordingConcurrencyEmbedder {
    fn embed(&self, input: EmbeddingInput) -> CgResult<EmbeddingOutput> {
        match input {
            EmbeddingInput::Text(text) => Ok(EmbeddingOutput {
                embedding_space_id: TEXT_SPACE.to_string(),
                dimension: 768,
                producer: producer_for_space(TEXT_SPACE, "new"),
                vector: expected_reindexed_vector(&text),
            }),
            EmbeddingInput::ImageFrame { .. } | EmbeddingInput::ImageBytes(..) => {
                Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::ImageFrame,
                })
            }
            EmbeddingInput::AudioSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::AudioSegment,
            }),
            EmbeddingInput::VideoSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::VideoSegment,
            }),
            EmbeddingInput::StructuredSignal(_) => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::StructuredSignal,
            }),
        }
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        text_space()
    }
}

#[derive(Clone)]
struct RecordingVisionConcurrencyEmbedder;

impl Embedder for RecordingVisionConcurrencyEmbedder {
    fn embed(&self, input: EmbeddingInput) -> CgResult<EmbeddingOutput> {
        match input {
            EmbeddingInput::ImageFrame { uri } => Ok(EmbeddingOutput {
                embedding_space_id: IMAGE_SPACE.to_string(),
                dimension: 512,
                producer: producer_for_space(IMAGE_SPACE, "new"),
                vector: expected_reindexed_image_vector(&uri),
            }),
            EmbeddingInput::ImageBytes(bytes) => Ok(EmbeddingOutput {
                embedding_space_id: IMAGE_SPACE.to_string(),
                dimension: 512,
                producer: producer_for_space(IMAGE_SPACE, "new"),
                vector: expected_reindexed_image_bytes_vector(&bytes),
            }),
            EmbeddingInput::Text(_) => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::Text,
            }),
            EmbeddingInput::AudioSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::AudioSegment,
            }),
            EmbeddingInput::VideoSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::VideoSegment,
            }),
            EmbeddingInput::StructuredSignal(_) => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::StructuredSignal,
            }),
        }
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::ImageFrame]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        image_space()
    }
}

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    store: Store,
}

struct ActiveBasisDecision {
    context_id: context_graph::ContextId,
    entity_id: context_graph::EntityId,
    decision_id: context_graph::DecisionId,
}

fn fixture(name: &str) -> Fixture {
    fixture_with_embedders(
        name,
        Arc::new(RecordingConcurrencyEmbedder),
        Arc::new(RecordingVisionConcurrencyEmbedder),
    )
}

fn fixture_with_embedders(
    name: &str,
    text_embedder: Arc<dyn Embedder>,
    image_embedder: Arc<dyn Embedder>,
) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: db_path.clone(),
            extra_embedding_spaces: vec![image_space(), unbound_space()],
            ..Default::default()
        },
        vec![
            EmbedderRegistration {
                space: text_space(),
                embedder: text_embedder,
            },
            EmbedderRegistration {
                space: image_space(),
                embedder: image_embedder,
            },
        ],
    )
    .expect("store opens");
    Fixture {
        _dir: dir,
        db_path,
        store,
    }
}

fn text_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: TEXT_SPACE.to_string(),
        dimension: 768,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "embeddinggemma-300m".to_string(),
        model_version: "default".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: Some(EvidenceKind::Text),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: TEXT_SPACE.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_text".to_string(),
        }],
    }
}

fn unbound_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: UNBOUND_SPACE.to_string(),
        dimension: 384,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "unbound-local".to_string(),
        model_version: "v1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: None,
        bindings: Vec::new(),
    }
}

fn image_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: IMAGE_SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "clip-local".to_string(),
        model_version: "v1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: IMAGE_SPACE.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_vision".to_string(),
        }],
    }
}

fn producer_for_space(space: &str, version: &str) -> EvidenceProducer {
    let model_name = if space == IMAGE_SPACE {
        "clip-local"
    } else {
        "embeddinggemma-300m"
    };
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: model_name.to_string(),
        model_version: version.to_string(),
        pipeline_version: "reindex-concurrency".to_string(),
    }
}

fn producer(version: &str) -> EvidenceProducer {
    producer_for_space(TEXT_SPACE, version)
}

fn image_producer(version: &str) -> EvidenceProducer {
    producer_for_space(IMAGE_SPACE, version)
}

fn producer_event(version: &str) -> String {
    serde_json::to_string(&producer(version)).expect("producer json")
}

fn reindex_cursor_seed_versions() -> [&'static str; 5] {
    ["old-a", "old-b", "pause-new", "old-c", "old-a"]
}

fn reindex_cursor_extra_history_idx() -> usize {
    3
}

fn reindex_cursor_seeded_from_env(db_path: &Path) -> Vec<(EvidenceId, String)> {
    std::env::var("CG_REINDEX_CURSOR_EVIDENCE_IDS")
        .expect("evidence ids env")
        .split(',')
        .map(|raw| {
            let evidence_id =
                EvidenceId::from(uuid::Uuid::parse_str(raw).expect("evidence id uuid"));
            let source_text = evidence_text_projection(db_path, evidence_id)
                .expect("persisted evidence text projection");
            (evidence_id, source_text)
        })
        .collect()
}

fn reindex_cursor_evidence_ids_env(seeded: &[(EvidenceId, String)]) -> String {
    seeded
        .iter()
        .map(|(id, _)| id.as_uuid().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

const REINDEX_CURSOR_RESUME_CALLS_MARKER: &str = "CG_REINDEX_CURSOR_RESUME_CALLS=";

fn resume_call_records_from_child_stdout(stdout: &str) -> Option<Vec<PhaseReindexCall>> {
    stdout.lines().find_map(|line| {
        line.strip_prefix(REINDEX_CURSOR_RESUME_CALLS_MARKER)
            .and_then(|payload| serde_json::from_str(payload).ok())
    })
}

fn new_reindex_vector_salt(discriminator: u64) -> u64 {
    let raw = uuid::Uuid::now_v7().as_u128();
    ((raw >> 64) as u64) ^ (raw as u64) ^ discriminator
}

fn phase_reindexed_vector(vector_salt: u64, call_index: usize) -> Vec<f32> {
    let mut hash = vector_salt ^ ((call_index as u64).wrapping_mul(0x9e3779b97f4a7c15));
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
    hash ^= hash >> 33;

    let primary = 120 + (hash as usize % 480);
    let mut secondary = 620 + ((hash >> 16) as usize % 120);
    if secondary == primary {
        secondary = (secondary + 1).min(767);
    }
    let mut vector = vec![0.0; 768];
    vector[primary] = 0.8;
    vector[secondary] = 0.6;
    vector
}

fn expected_reindexed_vector(source_text: &str) -> Vec<f32> {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in source_text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let primary = 100 + (hash as usize % 500);
    let secondary = 650 + ((hash >> 16) as usize % 100);
    let mut vector = vec![0.0; 768];
    vector[primary] = 0.8;
    vector[secondary] = 0.6;
    vector
}

fn expected_reindexed_image_vector(source_ref: &str) -> Vec<f32> {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in source_ref.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let primary = 50 + (hash as usize % 350);
    let secondary = 430 + ((hash >> 16) as usize % 70);
    let mut vector = vec![0.0; 512];
    vector[primary] = 0.7;
    vector[secondary] = 0.7;
    vector
}

fn expected_reindexed_image_bytes_vector(image_bytes: &[u8]) -> Vec<f32> {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in image_bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let primary = 50 + (hash as usize % 350);
    let secondary = 430 + ((hash >> 16) as usize % 70);
    let mut vector = vec![0.0; 512];
    vector[primary] = 0.7;
    vector[secondary] = 0.7;
    vector
}

fn axis(dim: usize, idx: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dim];
    vector[idx % dim] = 1.0;
    vector
}

fn mime_for_kind(kind: EvidenceKind) -> Option<String> {
    match kind {
        EvidenceKind::ImageFrame => Some("image/png".to_string()),
        EvidenceKind::StructuredSignal => Some("application/json".to_string()),
        EvidenceKind::Text => Some("text/plain".to_string()),
        _ => None,
    }
}

fn seed_rows(
    store: &Store,
    space: &str,
    kind: EvidenceKind,
    dim: usize,
    rows: usize,
) -> Vec<(EvidenceId, String)> {
    let versions = vec!["old"; rows];
    seed_rows_with_producer_versions(store, space, kind, dim, &versions)
}

fn seed_rows_with_producer_versions(
    store: &Store,
    space: &str,
    kind: EvidenceKind,
    dim: usize,
    versions: &[&str],
) -> Vec<(EvidenceId, String)> {
    let context = store
        .create_context(CreateContext {
            name: format!("reindex-ctx-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: format!("reindex-entity-{}", uuid::Uuid::now_v7()),
            properties: Default::default(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    let mut seeded = Vec::new();
    for (idx, version) in versions.iter().enumerate() {
        let evidence_id = EvidenceId::new_v7();
        let source_text = format!("opaque reindex text {}", uuid::Uuid::now_v7());
        let source_ref = format!("cas://sha256:{:064x}", idx + 1);
        let recorded = store.record_observation(RecordObservation {
            id: context_graph::ObservationId::new_v7(),
            entity_id: entity.id,
            context_id: context.id,
            observation_type: "reindex-seed".to_string(),
            source: "test".to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: evidence_id,
                kind,
                source_ref: source_ref.clone(),
                mime_type: mime_for_kind(kind),
                retention_status: RetentionStatus::RetainedExternal,
                producer: producer_for_space(space, version),
                text_projection: (kind == EvidenceKind::Text).then_some(source_text.clone()),
                ..Default::default()
            }],
            observed_properties: Default::default(),
            state_delta: Default::default(),
            properties: Default::default(),
            embeddings: vec![PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: space.to_string(),
                vector: axis(dim, idx),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer_for_space(space, version),
            }],
        });
        assert!(recorded.is_ok(), "seed row {idx} failed: {recorded:?}");
        let reindex_source = if kind == EvidenceKind::Text {
            source_text
        } else {
            source_ref
        };
        seeded.push((evidence_id, reindex_source));
    }
    seeded
}

fn seed_active_basis_decisions(store: &Store, rows: usize) -> Vec<ActiveBasisDecision> {
    let mut seeded = Vec::new();
    for idx in 0..rows {
        let context = store
            .create_context(CreateContext {
                name: format!("fleet-reindex-decision-{idx}-{}", uuid::Uuid::now_v7()),
                labels: Vec::new(),
                properties: Default::default(),
            })
            .expect("fleet context");
        let intention = store
            .create_intention(CreateIntention {
                id: None,
                description: format!("keep fleet entity {idx} stable during reindex"),
                status: IntentionStatus::Active,
                origin: IntentionOrigin::Human,
                context_id: context.id,
                properties: Default::default(),
                tags: Vec::new(),
                blueprint_catalog_id: None,
            })
            .expect("fleet intention");
        let entity = store
            .create_entity(CreateEntity {
                entity_type: EntityType::Service,
                name: format!("fleet entity {idx}"),
                properties: BTreeMap::from([(
                    "state".to_string(),
                    serde_json::json!(format!("pre-reindex-{idx}")),
                )]),
                tags: Vec::new(),
                context_id: context.id,
            })
            .expect("fleet entity");
        let basis_fields = HashMap::from([(entity.id, vec!["state".to_string()])]);
        let decision = store
            .create_decision(CreateDecision {
                decision_type: "fleet-memory".to_string(),
                description: format!("decision pinned to fleet entity {idx}"),
                intention_ids: vec![intention.id],
                based_on_entity_ids: vec![entity.id],
                basis_fields: Some(basis_fields),
                context_id: context.id,
                ..Default::default()
            })
            .expect("fleet decision");
        seeded.push(ActiveBasisDecision {
            context_id: context.id,
            entity_id: entity.id,
            decision_id: decision.id,
        });
    }
    seeded
}

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn open_raw_database(db_path: &Path) -> Database {
    let deadline = Instant::now() + Duration::from_millis(500);
    loop {
        match Database::open(db_path) {
            Ok(db) => return db,
            Err(err) => {
                let message = format!("{err:?}");
                if !message.contains("DatabaseLocked") || Instant::now() >= deadline {
                    return Database::open(db_path).expect("raw database open after retry");
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

fn execute_query(
    db_path: &Path,
    sql: &str,
    params: &HashMap<String, Value>,
) -> contextdb_engine::QueryResult {
    let store = open_reindex_store(db_path.to_path_buf()).expect("query store opens");
    store.execute_sql_for_test(sql, params).expect("query")
}

fn count(db_path: &Path, sql: &str, params: &HashMap<String, Value>) -> i64 {
    let result = execute_query(db_path, sql, params);
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

fn cursor_count(db_path: &Path, space: &str) -> i64 {
    count(
        db_path,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space",
        &params(vec![("space", Value::Text(space.to_string()))]),
    )
}

fn cursor_at(db_path: &Path, space: &str, expected_cursor: i64) -> i64 {
    count(
        db_path,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space AND last_committed_cursor = $cursor",
        &params(vec![
            ("space", Value::Text(space.to_string())),
            ("cursor", Value::Int64(expected_cursor)),
        ]),
    )
}

fn store_count(store: &Store, query: &str, params: &HashMap<String, Value>) -> i64 {
    let result = store
        .execute_sql_for_test(query, params)
        .expect("store count query");
    match result.rows.first().and_then(|row| row.first()) {
        Some(Value::Int64(value)) => *value,
        _ => 0,
    }
}

fn store_cursor_count(store: &Store, space: &str) -> i64 {
    store_count(
        store,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space",
        &params(vec![("space", Value::Text(space.to_string()))]),
    )
}

fn store_cursor_at(store: &Store, space: &str, expected_cursor: i64) -> i64 {
    store_count(
        store,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space AND last_committed_cursor = $cursor",
        &params(vec![
            ("space", Value::Text(space.to_string())),
            ("cursor", Value::Int64(expected_cursor)),
        ]),
    )
}

fn lease_owner_run_id(db_path: &Path, space: &str) -> Option<String> {
    let result = execute_query(
        db_path,
        "SELECT owner_run_id FROM reindex_cursors WHERE embedding_space_id = $space",
        &params(vec![("space", Value::Text(space.to_string()))]),
    );
    result.rows.first().and_then(|row| match row.first() {
        Some(Value::Text(owner)) => Some(owner.clone()),
        _ => None,
    })
}

fn durable_in_progress_lease_at(db_path: &Path, space: &str, expected_cursor: i64) -> i64 {
    let Some(owner_run_id) = lease_owner_run_id(db_path, space) else {
        return 0;
    };
    if owner_run_id.trim().is_empty() || owner_run_id == "raw-contender" {
        return 0;
    }

    count(
        db_path,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space AND status = $status AND owner_run_id = $owner AND lease_epoch = $epoch AND last_committed_cursor = $cursor",
        &params(vec![
            ("space", Value::Text(space.to_string())),
            ("status", Value::Text("in_progress".to_string())),
            ("owner", Value::Text(owner_run_id)),
            ("epoch", Value::Int64(1)),
            ("cursor", Value::Int64(expected_cursor)),
        ]),
    )
}

fn raw_conditional_lease_takeover_rejected(db_path: &Path, space: &str) -> bool {
    let Some(owner_run_id) = lease_owner_run_id(db_path, space) else {
        return false;
    };
    if owner_run_id.trim().is_empty() {
        return false;
    }
    if durable_in_progress_lease_at(db_path, space, 1) != 1 {
        return false;
    }
    let available_status_result = {
        let store = open_reindex_store(db_path.to_path_buf()).expect("lease contender store opens");
        store.execute_sql_for_test(
        "UPDATE reindex_cursors SET owner_run_id = $owner, lease_epoch = $next_epoch WHERE embedding_space_id = $space AND status = $available",
        &params(vec![
            ("owner", Value::Text("raw-contender".to_string())),
            ("next_epoch", Value::Int64(2)),
            ("space", Value::Text(space.to_string())),
            ("available", Value::Text("available".to_string())),
        ]),
        )
    };
    let stale_owner_result = {
        let store = open_reindex_store(db_path.to_path_buf()).expect("lease contender store opens");
        store.execute_sql_for_test(
        "UPDATE reindex_cursors SET owner_run_id = $owner, lease_epoch = $next_epoch WHERE embedding_space_id = $space AND status = $status AND owner_run_id = $stale_owner AND lease_epoch = $epoch",
        &params(vec![
            ("owner", Value::Text("raw-contender".to_string())),
            ("next_epoch", Value::Int64(2)),
            ("space", Value::Text(space.to_string())),
            ("status", Value::Text("in_progress".to_string())),
            ("stale_owner", Value::Text(format!("{owner_run_id}-stale"))),
            ("epoch", Value::Int64(1)),
        ]),
        )
    };
    matches!(available_status_result, Ok(query) if query.rows_affected == 0)
        && matches!(stale_owner_result, Ok(query) if query.rows_affected == 0)
        && lease_owner_run_id(db_path, space).as_deref() == Some(owner_run_id.as_str())
        && durable_in_progress_lease_at(db_path, space, 1) == 1
}

fn vector_score_for(
    db_path: &Path,
    evidence_id: EvidenceId,
    column: &str,
    query: Vec<f32>,
) -> Option<f32> {
    let db = open_raw_database(db_path);
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", column),
            &query,
            32,
            None,
            db.snapshot(),
        )
        .expect("vector query");
    let rows = db.scan("evidence", db.snapshot()).expect("scan evidence");
    hits.into_iter().find_map(|(row_id, score)| {
        rows.iter().find_map(|row| {
            (row.row_id == row_id)
                .then(|| row.values.get("id").cloned())
                .flatten()
                .and_then(|value| match value {
                    Value::Uuid(id) if id == evidence_id.as_uuid() => Some(score),
                    _ => None,
                })
        })
    })
}

fn stored_text_vector(db_path: &Path, evidence_id: EvidenceId) -> Option<Vec<f32>> {
    let db = open_raw_database(db_path);
    let snapshot = db.snapshot();
    let row_id = db
        .scan("evidence", snapshot)
        .expect("scan evidence")
        .into_iter()
        .find_map(|row| match row.values.get("id") {
            Some(Value::Uuid(id)) if *id == evidence_id.as_uuid() => Some(row.row_id),
            _ => None,
        })?;
    db.live_vector_entry(row_id, snapshot)
        .filter(|entry| entry.index == VectorIndexRef::new("evidence", "vector_text"))
        .map(|entry| entry.vector)
}

fn evidence_text_projection(db_path: &Path, evidence_id: EvidenceId) -> Option<String> {
    let db = open_raw_database(db_path);
    let result = db
        .execute(
            "SELECT text_projection FROM evidence WHERE id = $id",
            &params(vec![("id", Value::Uuid(evidence_id.as_uuid()))]),
        )
        .expect("evidence text projection query");
    result.rows.first().and_then(|row| match row.first() {
        Some(Value::Text(text)) => Some(text.clone()),
        _ => None,
    })
}

fn evidence_context_uuid(store: &Store, evidence_id: EvidenceId) -> Option<uuid::Uuid> {
    let result = store
        .execute_sql_for_test(
            "SELECT context_id FROM evidence WHERE id = $id",
            &params(vec![("id", Value::Uuid(evidence_id.as_uuid()))]),
        )
        .ok()?;
    match result.rows.first().and_then(|row| row.first()) {
        Some(Value::Uuid(context_id)) => Some(*context_id),
        _ => None,
    }
}

fn insert_extra_historical_handle(store: &Store, evidence_id: EvidenceId) -> bool {
    let Some(context_id) = evidence_context_uuid(store, evidence_id) else {
        return false;
    };
    let common_params = params(vec![
        ("id", Value::Uuid(EmbeddingId::new_v7().as_uuid())),
        ("context_id", Value::Uuid(context_id)),
        ("target_id", Value::Uuid(evidence_id.as_uuid())),
        ("space", Value::Text(TEXT_SPACE.to_string())),
        (
            "producer",
            Value::Json(serde_json::to_value(producer("legacy-extra")).expect("json")),
        ),
        ("producer_system", Value::Text("edge-memory".to_string())),
        ("created_at", Value::TxId(context_graph::TxId(1))),
    ]);
    let result = store.execute_sql_for_test(
        "INSERT INTO evidence_embedding_handles (id, context_id, target_id, embedding_space_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $target_id, $space, $producer, $producer_system, $created_at)",
        &common_params,
    );
    let result = result.or_else(|_| {
        store.execute_sql_for_test(
            "INSERT INTO evidence_embedding_handles (id, context_id, target_id, embedding_space_id, produced_by, created_at) VALUES ($id, $context_id, $target_id, $space, $producer, $created_at)",
            &common_params,
        )
    });
    matches!(result, Ok(query) if query.rows_affected == 1)
}

fn text_vector_matches_exactly(db_path: &Path, evidence_id: EvidenceId, expected: &[f32]) -> bool {
    stored_text_vector(db_path, evidence_id).is_some_and(|stored| stored == expected)
}

fn producer_rows(db_path: &Path, producer: EvidenceProducer) -> i64 {
    count(
        db_path,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE produced_by = $producer",
        &params(vec![(
            "producer",
            Value::Json(serde_json::to_value(producer).expect("producer json")),
        )]),
    )
}

fn producer_rows_for_space(db_path: &Path, space: &str, producer: EvidenceProducer) -> i64 {
    count(
        db_path,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE embedding_space_id = $space AND produced_by = $producer",
        &params(vec![
            ("space", Value::Text(space.to_string())),
            (
                "producer",
                Value::Json(serde_json::to_value(producer).expect("producer json")),
            ),
        ]),
    )
}

fn handle_rows_for_target_space(db_path: &Path, evidence_id: EvidenceId, space: &str) -> i64 {
    count(
        db_path,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id AND embedding_space_id = $space",
        &params(vec![
            ("target_id", Value::Uuid(evidence_id.as_uuid())),
            ("space", Value::Text(space.to_string())),
        ]),
    )
}

fn handle_rows_for_target_producer(
    db_path: &Path,
    evidence_id: EvidenceId,
    space: &str,
    producer: EvidenceProducer,
) -> i64 {
    count(
        db_path,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id AND embedding_space_id = $space AND produced_by = $producer",
        &params(vec![
            ("target_id", Value::Uuid(evidence_id.as_uuid())),
            ("space", Value::Text(space.to_string())),
            (
                "producer",
                Value::Json(serde_json::to_value(producer).expect("producer json")),
            ),
        ]),
    )
}

fn target_handle_lineage_matches(
    db_path: &Path,
    seeded: &[(EvidenceId, String)],
    seed_versions: &[&str],
    pause_processed_rows: usize,
    resume_start: Option<usize>,
) -> bool {
    seeded.iter().zip(seed_versions.iter()).enumerate().all(
        |(idx, ((evidence_id, _), old_version))| {
            let extra_expected = i64::from(idx == reindex_cursor_extra_history_idx());
            let pause_expected =
                i64::from(*old_version == "pause-new") + i64::from(idx < pause_processed_rows);
            let resume_expected = i64::from(resume_start.is_some_and(|start| idx >= start));
            let total_expected =
                1 + extra_expected + i64::from(idx < pause_processed_rows) + resume_expected;
            let old_expected = if *old_version == "pause-new" {
                pause_expected
            } else {
                1
            };

            handle_rows_for_target_space(db_path, *evidence_id, TEXT_SPACE) == total_expected
                && handle_rows_for_target_producer(
                    db_path,
                    *evidence_id,
                    TEXT_SPACE,
                    producer(old_version),
                ) == old_expected
                && handle_rows_for_target_producer(
                    db_path,
                    *evidence_id,
                    TEXT_SPACE,
                    producer("pause-new"),
                ) == pause_expected
                && handle_rows_for_target_producer(
                    db_path,
                    *evidence_id,
                    TEXT_SPACE,
                    producer("resume-new"),
                ) == resume_expected
                && handle_rows_for_target_producer(
                    db_path,
                    *evidence_id,
                    TEXT_SPACE,
                    producer("legacy-extra"),
                ) == extra_expected
        },
    )
}

fn all_reindexed_vectors_visible(db_path: &Path, seeded: &[(EvidenceId, String)]) -> bool {
    seeded.iter().all(|(evidence_id, source_text)| {
        vector_score_for(
            db_path,
            *evidence_id,
            "vector_text",
            expected_reindexed_vector(source_text),
        )
        .is_some_and(|score| score > 0.99)
    })
}

fn phase_call_texts(calls: &[PhaseReindexCall]) -> Vec<String> {
    calls.iter().map(|call| call.text.clone()).collect()
}

fn phase_call_vectors_are_not_text_formula(calls: &[PhaseReindexCall]) -> bool {
    calls
        .iter()
        .all(|call| call.vector != expected_reindexed_vector(&call.text))
}

fn reindex_cursor_pause_vectors_visible(
    db_path: &Path,
    seeded: &[(EvidenceId, String)],
    pause_salt: u64,
) -> bool {
    seeded.iter().enumerate().all(|(idx, (evidence_id, _))| {
        let vector = phase_reindexed_vector(pause_salt, idx);
        let visible = text_vector_matches_exactly(db_path, *evidence_id, &vector);
        visible == (idx < 2)
    })
}

fn phase_output_vectors_visible_for_seeded_range(
    db_path: &Path,
    seeded: &[(EvidenceId, String)],
    start_idx: usize,
    calls: &[PhaseReindexCall],
) -> bool {
    !calls.is_empty()
        && calls.iter().enumerate().all(|(offset, call)| {
            seeded
                .get(start_idx + offset)
                .is_some_and(|(evidence_id, _)| {
                    text_vector_matches_exactly(db_path, *evidence_id, &call.vector)
                })
        })
}

fn phase_output_vectors_indexed_for_seeded_range(
    db_path: &Path,
    seeded: &[(EvidenceId, String)],
    start_idx: usize,
    calls: &[PhaseReindexCall],
) -> bool {
    !calls.is_empty()
        && calls.iter().enumerate().all(|(offset, call)| {
            seeded
                .get(start_idx + offset)
                .is_some_and(|(evidence_id, _)| {
                    vector_score_for(db_path, *evidence_id, "vector_text", call.vector.clone())
                        .is_some_and(|score| score > 0.99)
                })
        })
}

fn phase_output_vectors_indexed_for_seeded_range_in_store(
    store: &Store,
    seeded: &[(EvidenceId, String)],
    start_idx: usize,
    calls: &[PhaseReindexCall],
) -> bool {
    if calls.is_empty() {
        return false;
    }
    let Some(ids) = calls
        .iter()
        .enumerate()
        .map(|(offset, _)| {
            seeded
                .get(start_idx + offset)
                .map(|(evidence_id, _)| evidence_id.as_uuid())
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let queries = calls
        .iter()
        .map(|call| call.vector.clone())
        .collect::<Vec<_>>();
    matches!(
        store.evidence_vector_visibility_counts_for_test(
            &ids,
            VectorIndexRef::new("evidence", "vector_text"),
            &queries,
        ),
        Ok((row_count, vector_count)) if row_count == calls.len() && vector_count == calls.len()
    )
}

fn all_reindexed_image_vectors_visible(db_path: &Path, seeded: &[(EvidenceId, String)]) -> bool {
    seeded.iter().all(|(evidence_id, source_ref)| {
        vector_score_for(
            db_path,
            *evidence_id,
            "vector_vision",
            expected_reindexed_image_vector(source_ref),
        )
        .is_some_and(|score| score > 0.99)
    })
}

fn open_reindex_store_with_embedder(
    db_path: PathBuf,
    embedder: Arc<dyn Embedder>,
) -> context_graph::Result<Store> {
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: text_space(),
            embedder,
        }],
    )
}

fn open_reindex_store(db_path: PathBuf) -> context_graph::Result<Store> {
    open_reindex_store_with_embedder(db_path, Arc::new(RecordingConcurrencyEmbedder))
}

fn run_two_reindexes(
    store: Store,
    first_space: &'static str,
    second_space: &'static str,
) -> (CgResult<()>, CgResult<()>) {
    let shared = Arc::new(store);
    let barrier = Arc::new(Barrier::new(3));
    let first = {
        let store = Arc::clone(&shared);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            store.reindex_space(first_space, ReindexOptions::default())
        })
    };
    let second = {
        let store = Arc::clone(&shared);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            store.reindex_space(second_space, ReindexOptions::default())
        })
    };

    barrier.wait();
    (
        first.join().expect("first reindex thread"),
        second.join().expect("second reindex thread"),
    )
}

#[test]
fn reindex_concurrent_invocations_on_same_space_serialize_or_reject_typed() {
    let fixture = fixture("reindex-same-space");
    let seeded = seed_rows(&fixture.store, TEXT_SPACE, EvidenceKind::Text, 768, 3);
    let db_path = fixture.db_path.clone();
    drop(fixture.store);
    let control = Arc::new(BlockingReindexControl::default());
    let first_store = open_reindex_store_with_embedder(
        db_path.clone(),
        Arc::new(BlockingSecondRowConcurrencyEmbedder {
            control: Arc::clone(&control),
        }),
    )
    .expect("first reindex handle opens");
    let first = thread::spawn(move || {
        let result = first_store.reindex_space(TEXT_SPACE, ReindexOptions::default());
        let events = first_store.drain_reindex_events();
        (result, events)
    });
    let first_blocked_inside_active_reindex = control.wait_until_blocked_on_second_row();
    let paused_cursor_rows = cursor_at(&db_path, TEXT_SPACE, 1);
    let cursor_rows_during_contention = cursor_count(&db_path, TEXT_SPACE);
    let durable_lease_rows = durable_in_progress_lease_at(&db_path, TEXT_SPACE, 1);
    let raw_contender_rejected_by_durable_lease =
        raw_conditional_lease_takeover_rejected(&db_path, TEXT_SPACE);
    let second_store = open_reindex_store(db_path.clone()).expect("second reindex handle opens");
    let second_result = second_store.reindex_space(TEXT_SPACE, ReindexOptions::default());
    let second_events = second_store.drain_reindex_events();
    drop(second_store);
    control.release();
    let (first_result, first_events) = first.join().expect("first reindex thread");
    let first_completed_seeded = first_events.iter().any(|event| {
        matches!(
            event,
            ReindexEvent::Completed {
                embedding_space_id,
                rows_processed,
                ..
            } if embedding_space_id == TEXT_SPACE && *rows_processed == seeded.len()
        )
    });
    let completed_cursor_rows = cursor_at(&db_path, TEXT_SPACE, seeded.len() as i64);
    let new_producer_rows = producer_rows_for_space(&db_path, TEXT_SPACE, producer("new"));
    let old_producer_rows = producer_rows_for_space(&db_path, TEXT_SPACE, producer("old"));
    let exact_vectors = all_reindexed_vectors_visible(&db_path, &seeded);

    assert!(
        first_blocked_inside_active_reindex
            && paused_cursor_rows == 1
            && cursor_rows_during_contention == 1
            && durable_lease_rows == 1
            && raw_contender_rejected_by_durable_lease
            && matches!(
                second_result,
                Err(CgError::ReindexAlreadyInProgress { ref embedding_space_id })
                    if embedding_space_id == TEXT_SPACE
            )
            && second_events.is_empty()
            && first_result.is_ok()
            && first_completed_seeded
            && completed_cursor_rows == 1
            && new_producer_rows == seeded.len() as i64
            && old_producer_rows == seeded.len() as i64
            && exact_vectors,
        "same-space reindex must use a durable cross-handle in-progress guard while the first handle is still inside active row work: the first handle blocks on row 2 with cursor 1 committed and a DB lease row status=in_progress/nonempty-owner/lease_epoch=1, a raw conditional contender using available status or a stale owner cannot take over the lease, the second independent Store handle rejects typed without progress events, then the first handle releases and completes exact vectors/producers; first_blocked_inside_active_reindex={first_blocked_inside_active_reindex}, first_result={first_result:?}, first_events={first_events:?}, second_result={second_result:?}, second_events={second_events:?}, paused_cursor_rows={paused_cursor_rows}, cursor_rows_during_contention={cursor_rows_during_contention}, durable_lease_rows={durable_lease_rows}, raw_contender_rejected_by_durable_lease={raw_contender_rejected_by_durable_lease}, completed_cursor_rows={completed_cursor_rows}, new_producer_rows={new_producer_rows}, old_producer_rows={old_producer_rows}, exact_vectors={exact_vectors}"
    );
}

#[test]
fn reindex_concurrent_invocations_on_distinct_spaces_proceed_in_parallel() {
    let row_count = 20;
    let serial_fixture = fixture("reindex-distinct-serial");
    let serial_text_seeded = seed_rows(
        &serial_fixture.store,
        TEXT_SPACE,
        EvidenceKind::Text,
        768,
        row_count,
    );
    let serial_image_seeded = seed_rows(
        &serial_fixture.store,
        IMAGE_SPACE,
        EvidenceKind::ImageFrame,
        512,
        row_count,
    );
    let serial_text_start = Instant::now();
    let serial_text = serial_fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let serial_text_elapsed = serial_text_start.elapsed();
    let serial_image_start = Instant::now();
    let serial_image = serial_fixture
        .store
        .reindex_space(IMAGE_SPACE, ReindexOptions::default());
    let serial_image_elapsed = serial_image_start.elapsed();
    let serial_db_path = serial_fixture.db_path.clone();
    drop(serial_fixture.store);
    let serial_text_vectors = all_reindexed_vectors_visible(&serial_db_path, &serial_text_seeded);
    let serial_image_vectors =
        all_reindexed_image_vectors_visible(&serial_db_path, &serial_image_seeded);
    let serial_text_new_producer_rows =
        producer_rows_for_space(&serial_db_path, TEXT_SPACE, producer("new"));
    let serial_image_new_producer_rows =
        producer_rows_for_space(&serial_db_path, IMAGE_SPACE, image_producer("new"));

    let text_control = Arc::new(BlockingReindexControl::default());
    let image_control = Arc::new(BlockingReindexControl::default());
    let fixture = fixture_with_embedders(
        "reindex-distinct-parallel",
        Arc::new(BlockingSecondRowConcurrencyEmbedder {
            control: Arc::clone(&text_control),
        }),
        Arc::new(CountingVisionConcurrencyEmbedder {
            control: Arc::clone(&image_control),
        }),
    );
    let text_seeded = seed_rows(
        &fixture.store,
        TEXT_SPACE,
        EvidenceKind::Text,
        768,
        row_count,
    );
    let image_seeded = seed_rows(
        &fixture.store,
        IMAGE_SPACE,
        EvidenceKind::ImageFrame,
        512,
        row_count,
    );
    let db_path = fixture.db_path.clone();
    let parallel_start = Instant::now();
    let shared = Arc::new(fixture.store.clone());
    let first = {
        let store = Arc::clone(&shared);
        thread::spawn(move || store.reindex_space(TEXT_SPACE, ReindexOptions::default()))
    };
    let text_blocked_inside_active_reindex = text_control.wait_until_blocked_on_second_row();
    let second = {
        let store = Arc::clone(&shared);
        thread::spawn(move || store.reindex_space(IMAGE_SPACE, ReindexOptions::default()))
    };
    let image_progress_while_text_blocked = image_control.wait_until_calls_at_least(1);
    text_control.release();
    let first = first.join().expect("text reindex thread");
    let second = second.join().expect("image reindex thread");
    let parallel_elapsed = parallel_start.elapsed();
    drop(shared);
    drop(fixture.store);
    let text_cursors = cursor_count(&db_path, TEXT_SPACE);
    let vision_cursors = cursor_count(&db_path, IMAGE_SPACE);
    let text_vectors = all_reindexed_vectors_visible(&db_path, &text_seeded);
    let image_vectors = all_reindexed_image_vectors_visible(&db_path, &image_seeded);
    let text_new_producer_rows = producer_rows_for_space(&db_path, TEXT_SPACE, producer("new"));
    let image_new_producer_rows =
        producer_rows_for_space(&db_path, IMAGE_SPACE, image_producer("new"));

    assert!(
        serial_text.is_ok()
            && serial_image.is_ok()
            && serial_text_vectors
            && serial_image_vectors
            && serial_text_new_producer_rows == row_count as i64
            && serial_image_new_producer_rows == row_count as i64
            && text_blocked_inside_active_reindex
            && image_progress_while_text_blocked
            && first.is_ok()
            && second.is_ok()
            && text_cursors == 1
            && vision_cursors == 1
            && text_vectors
            && image_vectors
            && text_new_producer_rows == row_count as i64
            && image_new_producer_rows == row_count as i64,
        "distinct-space reindexes must each record durable cursors, re-embed exact text/image rows with post-reindex producers, and allow image row work to enter while text row work is actively blocked; this probe intentionally does not require overlapping long Database commit apply windows, row_count={row_count}, serial_text={serial_text:?}/{serial_text_elapsed:?}, serial_image={serial_image:?}/{serial_image_elapsed:?}, serial_text_vectors={serial_text_vectors}, serial_image_vectors={serial_image_vectors}, serial_text_new_producer_rows={serial_text_new_producer_rows}, serial_image_new_producer_rows={serial_image_new_producer_rows}, text_blocked_inside_active_reindex={text_blocked_inside_active_reindex}, image_progress_while_text_blocked={image_progress_while_text_blocked}, first={first:?}, second={second:?}, parallel={parallel_elapsed:?}, text_cursors={text_cursors}, vision_cursors={vision_cursors}, text_vectors={text_vectors}, image_vectors={image_vectors}, text_new_producer_rows={text_new_producer_rows}, image_new_producer_rows={image_new_producer_rows}"
    );
}

#[test]
fn reindex_on_empty_space_completes_immediately_with_zero_rows_processed() {
    let fixture = fixture("reindex-empty");
    let context = fixture
        .store
        .create_context(CreateContext {
            name: format!("reindex-empty-positive-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("positive control context");
    let entity = fixture
        .store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "empty reindex positive control".to_string(),
            properties: BTreeMap::from([("state".to_string(), serde_json::json!("known"))]),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("positive control entity");
    let intention = fixture
        .store
        .create_intention(CreateIntention {
            id: None,
            description: "prove impact-analysis trigger is active".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            properties: Default::default(),
            tags: Vec::new(),
            blueprint_catalog_id: None,
        })
        .expect("positive control intention");
    let basis_fields = HashMap::from([(entity.id, vec!["state".to_string()])]);
    let decision = fixture
        .store
        .create_decision(CreateDecision {
            decision_type: "positive-control".to_string(),
            description: "state is known".to_string(),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: Some(basis_fields),
            context_id: context.id,
            ..Default::default()
        })
        .expect("positive control decision");
    let before_events = fixture.store.drain_reindex_events();
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let events = fixture.store.drain_reindex_events();
    let cursors = cursor_count(&fixture.db_path, TEXT_SPACE);
    let zero_cursor_rows = cursor_at(&fixture.db_path, TEXT_SPACE, 0);
    let invalidation_rows = count(
        &fixture.db_path,
        "SELECT COUNT(*) AS n FROM invalidations",
        &HashMap::new(),
    );
    let completions = events
        .iter()
        .filter_map(|event| match event {
            ReindexEvent::Completed {
                embedding_space_id,
                rows_processed,
                from_producer,
                to_producer,
            } => Some((
                embedding_space_id.as_str(),
                *rows_processed,
                from_producer.as_deref(),
                to_producer.as_deref(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let positive_observation_id = context_graph::ObservationId::new_v7();
    let positive_control = fixture.store.record_observation(RecordObservation {
        id: positive_observation_id,
        entity_id: entity.id,
        context_id: context.id,
        observation_type: "positive-control".to_string(),
        source: "reindex-test".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: EvidenceId::new_v7(),
            kind: EvidenceKind::Text,
            source_ref:
                "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
            mime_type: Some("text/plain".to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer("positive-control"),
            text_projection: Some("state changed".to_string()),
            ..Default::default()
        }],
        observed_properties: Default::default(),
        state_delta: BTreeMap::from([("state".to_string(), Some(serde_json::json!("changed")))]),
        properties: Default::default(),
        embeddings: Vec::new(),
    });
    let positive_invalidations =
        fixture
            .store
            .list_invalidations(context.id, Some(decision.id), None);

    assert!(
        before_events.is_empty()
            && result.is_ok()
            && cursors == 1
            && zero_cursor_rows == 1
            && invalidation_rows == 0
            && completions == vec![(TEXT_SPACE, 0, None, None)]
            && positive_control
                .as_ref()
                .is_ok_and(|result| result.invalidations.len() == 1)
            && positive_invalidations
                .as_ref()
                .is_ok_and(|rows| rows.len() == 1
                    && rows[0].affected_decision_id == decision.id
                    && rows[0].trigger_observation_id == positive_observation_id),
        "empty-space reindex must complete with an exact zero-row cursor, one typed completion event, and no invalidation callbacks while a same-store normal observation proves the impact-analysis trigger is active, result={result:?}, before_events={before_events:?}, events={events:?}, cursors={cursors}, zero_cursor_rows={zero_cursor_rows}, invalidation_rows={invalidation_rows}, completions={completions:?}, positive_control={positive_control:?}, positive_invalidations={positive_invalidations:?}"
    );
}

#[test]
fn reindex_on_single_row_space_processes_exactly_one_row() {
    let fixture = fixture("reindex-one-row");
    seed_rows(&fixture.store, TEXT_SPACE, EvidenceKind::Text, 768, 1);
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    drop(fixture.store);
    let rows = cursor_at(&fixture.db_path, TEXT_SPACE, 1);

    assert!(
        result.is_ok() && rows == 1,
        "single-row reindex must persist last_committed_cursor=1, result={result:?}, rows={rows}"
    );
}

#[test]
fn reindex_resumable_after_process_restart_persists_cursor_to_disk() {
    if std::env::var_os("CG_REINDEX_CURSOR_CHILD").is_some() {
        let db_path = PathBuf::from(std::env::var("CG_REINDEX_CURSOR_DB").expect("db env"));
        let seeded = reindex_cursor_seeded_from_env(&db_path);
        let seed_versions = reindex_cursor_seed_versions();
        let resume_salt = new_reindex_vector_salt(0x7265_7375_6d65_6e65);
        let resume_calls = Arc::new(Mutex::new(Vec::new()));
        let store = open_reindex_store_with_embedder(
            db_path.clone(),
            Arc::new(PhaseRecordingConcurrencyEmbedder {
                producer_version: "resume-new",
                vector_salt: resume_salt,
                calls: Arc::clone(&resume_calls),
            }),
        )
        .expect("child resume store opens");
        let resume_result = store.reindex_space(TEXT_SPACE, ReindexOptions::default());
        let resume_events = store.drain_reindex_events();
        drop(store);
        let resumed_at_two = resume_events.iter().any(|event| {
            matches!(
                event,
                ReindexEvent::Resumed {
                    embedding_space_id,
                    resumed_at_cursor: 2,
                } if embedding_space_id == TEXT_SPACE
            )
        });
        let completed_five = resume_events.iter().any(|event| {
            matches!(
                event,
                ReindexEvent::Completed {
                    embedding_space_id,
                    rows_processed: 5,
                    ..
                } if embedding_space_id == TEXT_SPACE
            )
        });
        let resume_call_records = resume_calls.lock().expect("resume calls").clone();
        let resume_call_texts = phase_call_texts(&resume_call_records);
        let resume_call_shape = resume_call_texts
            == seeded[2..]
                .iter()
                .map(|(_, text)| text.clone())
                .collect::<Vec<_>>();
        let resume_vectors_not_text_formula =
            phase_call_vectors_are_not_text_formula(&resume_call_records);
        let resume_vectors_from_output_visible = resume_call_records.len() == seeded[2..].len()
            && phase_output_vectors_visible_for_seeded_range(
                &db_path,
                &seeded,
                2,
                &resume_call_records,
            );
        let resume_vectors_from_output_indexed = resume_call_records.len() == seeded[2..].len()
            && phase_output_vectors_indexed_for_seeded_range(
                &db_path,
                &seeded,
                2,
                &resume_call_records,
            );
        let cursor_rows = cursor_at(&db_path, TEXT_SPACE, 5);
        let cursor_count_final = cursor_count(&db_path, TEXT_SPACE);
        let pause_producer_rows_after = producer_rows(&db_path, producer("pause-new"));
        let resume_producer_rows = producer_rows(&db_path, producer("resume-new"));
        let final_target_lineage =
            target_handle_lineage_matches(&db_path, &seeded, &seed_versions, 2, Some(2));

        assert!(
            resume_result.is_ok()
                && resumed_at_two
                && completed_five
                && resume_call_shape
                && resume_vectors_not_text_formula
                && resume_vectors_from_output_visible
                && resume_vectors_from_output_indexed
                && cursor_rows == 1
                && cursor_count_final == 1
                && pause_producer_rows_after == 3
                && resume_producer_rows == 3
                && final_target_lineage
                && resume_vectors_from_output_visible,
            "fresh child process must resume from durable cursor 2 only, persist and index exact runtime-salted Embedder output vectors rather than text-derived vectors, and avoid parent process state or handle-derived progress, resume_result={resume_result:?}, resume_events={resume_events:?}, resume_call_texts={resume_call_texts:?}, resume_vectors_not_text_formula={resume_vectors_not_text_formula}, resume_vectors_from_output_visible={resume_vectors_from_output_visible}, resume_vectors_from_output_indexed={resume_vectors_from_output_indexed}, cursor_rows={cursor_rows}, cursor_count_final={cursor_count_final}, pause_producer_rows_after={pause_producer_rows_after}, resume_producer_rows={resume_producer_rows}, final_target_lineage={final_target_lineage}"
        );
        println!(
            "{}{}",
            REINDEX_CURSOR_RESUME_CALLS_MARKER,
            serde_json::to_string(&resume_call_records).expect("serialize resume call records")
        );
        return;
    }

    let pause_salt = new_reindex_vector_salt(0x7061_7573_652d_6e65);
    let pause_calls = Arc::new(Mutex::new(Vec::new()));
    let fixture = fixture_with_embedders(
        "reindex-cursor",
        Arc::new(PhaseRecordingConcurrencyEmbedder {
            producer_version: "pause-new",
            vector_salt: pause_salt,
            calls: Arc::clone(&pause_calls),
        }),
        Arc::new(RecordingVisionConcurrencyEmbedder),
    );
    let seed_versions = reindex_cursor_seed_versions();
    let seeded = seed_rows_with_producer_versions(
        &fixture.store,
        TEXT_SPACE,
        EvidenceKind::Text,
        768,
        &seed_versions,
    );
    let extra_history_inserted = insert_extra_historical_handle(
        &fixture.store,
        seeded[reindex_cursor_extra_history_idx()].0,
    );
    let result = fixture.store.reindex_space(
        TEXT_SPACE,
        ReindexOptions {
            interrupt_after: Some(2),
        },
    );
    let paused_events = fixture.store.drain_reindex_events();
    let pause_call_records = pause_calls.lock().expect("pause calls").clone();
    let pause_call_texts = phase_call_texts(&pause_call_records);
    let pause_call_shape = pause_call_texts == vec![seeded[0].1.clone(), seeded[1].1.clone()];
    let pause_vectors_not_text_formula =
        phase_call_vectors_are_not_text_formula(&pause_call_records);
    let pause_vectors_indexed_before_drop = pause_call_records.len() == 2
        && phase_output_vectors_indexed_for_seeded_range_in_store(
            &fixture.store,
            &seeded,
            0,
            &pause_call_records,
        );
    let paused_cursor_rows_before_drop = store_cursor_at(&fixture.store, TEXT_SPACE, 2);
    let paused_cursor_count_before_drop = store_cursor_count(&fixture.store, TEXT_SPACE);
    let premature_completed_cursor_rows_before_drop =
        store_cursor_at(&fixture.store, TEXT_SPACE, 5);
    let paused_events_shape = paused_events
        == vec![ReindexEvent::Paused {
            embedding_space_id: TEXT_SPACE.to_string(),
            last_committed_cursor: 2,
        }];
    let paused_at_two = paused_events.iter().any(|event| {
        matches!(
            event,
            ReindexEvent::Paused {
                embedding_space_id,
                last_committed_cursor: 2,
            } if embedding_space_id == TEXT_SPACE
        )
    });
    drop(fixture.store);
    let pause_vectors_from_output_visible = pause_call_records.len() == 2
        && phase_output_vectors_visible_for_seeded_range(
            &fixture.db_path,
            &seeded,
            0,
            &pause_call_records,
        );
    let pause_vectors_from_output_indexed = pause_call_records.len() == 2
        && phase_output_vectors_indexed_for_seeded_range(
            &fixture.db_path,
            &seeded,
            0,
            &pause_call_records,
        );
    let pause_vectors_only_first_two =
        reindex_cursor_pause_vectors_visible(&fixture.db_path, &seeded, pause_salt);
    let pause_producer_rows = producer_rows(&fixture.db_path, producer("pause-new"));
    let resume_producer_rows_before = producer_rows(&fixture.db_path, producer("resume-new"));
    let paused_cursor_rows = cursor_at(&fixture.db_path, TEXT_SPACE, 2);
    let paused_cursor_count = cursor_count(&fixture.db_path, TEXT_SPACE);
    let premature_completed_cursor_rows = cursor_at(&fixture.db_path, TEXT_SPACE, 5);
    let pre_resume_target_lineage =
        target_handle_lineage_matches(&fixture.db_path, &seeded, &seed_versions, 2, None);
    let child = process::Command::new(std::env::current_exe().expect("current test exe"))
        .arg("--exact")
        .arg("reindex_concurrency::reindex_resumable_after_process_restart_persists_cursor_to_disk")
        .arg("--nocapture")
        .env("CG_REINDEX_CURSOR_CHILD", "1")
        .env(
            "CG_REINDEX_CURSOR_DB",
            fixture.db_path.display().to_string(),
        )
        .env(
            "CG_REINDEX_CURSOR_EVIDENCE_IDS",
            reindex_cursor_evidence_ids_env(&seeded),
        )
        .output()
        .expect("spawn fresh reindex resume verifier process");
    let child_stdout = String::from_utf8_lossy(&child.stdout);
    let child_resume_call_records = resume_call_records_from_child_stdout(&child_stdout);
    let resume_vectors_from_output_visible_after_child =
        child_resume_call_records.as_ref().is_some_and(|records| {
            records.len() == seeded[2..].len()
                && phase_output_vectors_visible_for_seeded_range(
                    &fixture.db_path,
                    &seeded,
                    2,
                    records,
                )
        });
    let resume_vectors_from_output_indexed_after_child =
        child_resume_call_records.as_ref().is_some_and(|records| {
            records.len() == seeded[2..].len()
                && phase_output_vectors_indexed_for_seeded_range(
                    &fixture.db_path,
                    &seeded,
                    2,
                    records,
                )
        });
    let cursor_rows = cursor_at(&fixture.db_path, TEXT_SPACE, 5);
    let cursor_count_final = cursor_count(&fixture.db_path, TEXT_SPACE);
    let pause_producer_rows_after = producer_rows(&fixture.db_path, producer("pause-new"));
    let resume_producer_rows = producer_rows(&fixture.db_path, producer("resume-new"));
    let old_a_producer_rows = producer_rows(&fixture.db_path, producer("old-a"));
    let old_b_producer_rows = producer_rows(&fixture.db_path, producer("old-b"));
    let old_c_producer_rows = producer_rows(&fixture.db_path, producer("old-c"));
    let final_target_lineage =
        target_handle_lineage_matches(&fixture.db_path, &seeded, &seed_versions, 2, Some(2));
    let pause_vectors_still_visible = phase_output_vectors_visible_for_seeded_range(
        &fixture.db_path,
        &seeded,
        0,
        &pause_call_records,
    );

    assert!(
        matches!(
            result,
            Err(CgError::ReindexPaused {
                ref embedding_space_id,
                last_committed_cursor: 2,
            }) if embedding_space_id == TEXT_SPACE
        ) && paused_at_two
            && pause_call_shape
            && pause_vectors_not_text_formula
            && pause_vectors_indexed_before_drop
            && pause_vectors_from_output_visible
            && pause_vectors_from_output_indexed
            && pause_vectors_only_first_two
            && extra_history_inserted
            && pause_producer_rows == 3
            && resume_producer_rows_before == 0
            && paused_cursor_rows_before_drop == 1
            && paused_cursor_count_before_drop == 1
            && premature_completed_cursor_rows_before_drop == 0
            && paused_cursor_rows == 1
            && paused_cursor_count == 1
            && premature_completed_cursor_rows == 0
            && pre_resume_target_lineage
            && paused_events_shape
            && child.status.success()
            && cursor_rows == 1
            && cursor_count_final == 1
            && pause_producer_rows_after == 3
            && resume_producer_rows == 3
            && old_a_producer_rows == 2
            && old_b_producer_rows == 1
            && old_c_producer_rows == 1
            && final_target_lineage
            && resume_vectors_from_output_visible_after_child
            && resume_vectors_from_output_indexed_after_child
            && pause_vectors_still_visible,
        "interrupted reindex must persist cursor 2 before a fresh child-process resume, process exactly rows 0..1 before pause, emit only the typed pause event before child resume, persist and index the exact runtime-salted Embedder output vectors from persisted DB text rather than synthesized ordinal text, resume from row 2 without rewriting rows 0..1 or deriving progress from pause-new handle counts, re-embed every remaining row across mixed old producers including a post-cursor row with extra historical handle lineage, and preserve per-target old/resume producer lineage without filtering resumed candidates to the first producer, result={result:?}, paused_events={paused_events:?}, paused_events_shape={paused_events_shape}, pause_call_texts={pause_call_texts:?}, pause_vectors_not_text_formula={pause_vectors_not_text_formula}, pause_vectors_indexed_before_drop={pause_vectors_indexed_before_drop}, pause_vectors_from_output_visible={pause_vectors_from_output_visible}, pause_vectors_from_output_indexed={pause_vectors_from_output_indexed}, pause_vectors_only_first_two={pause_vectors_only_first_two}, pause_vectors_still_visible={pause_vectors_still_visible}, extra_history_inserted={extra_history_inserted}, pause_producer_rows={pause_producer_rows}, resume_producer_rows_before={resume_producer_rows_before}, paused_cursor_rows_before_drop={paused_cursor_rows_before_drop}, paused_cursor_count_before_drop={paused_cursor_count_before_drop}, premature_completed_cursor_rows_before_drop={premature_completed_cursor_rows_before_drop}, paused_cursor_rows={paused_cursor_rows}, paused_cursor_count={paused_cursor_count}, premature_completed_cursor_rows={premature_completed_cursor_rows}, pre_resume_target_lineage={pre_resume_target_lineage}, child_status={:?}, child_stdout={}, child_stderr={}, cursor_rows={cursor_rows}, cursor_count_final={cursor_count_final}, pause_producer_rows_after={pause_producer_rows_after}, resume_producer_rows={resume_producer_rows}, resume_vectors_from_output_visible_after_child={resume_vectors_from_output_visible_after_child}, resume_vectors_from_output_indexed_after_child={resume_vectors_from_output_indexed_after_child}, old_a_producer_rows={old_a_producer_rows}, old_b_producer_rows={old_b_producer_rows}, old_c_producer_rows={old_c_producer_rows}, final_target_lineage={final_target_lineage}",
        child.status.code(),
        child_stdout,
        String::from_utf8_lossy(&child.stderr)
    );
}

#[test]
fn reindex_emits_observable_pause_signal_when_underlying_io_unavailable_and_resume_signal_when_restored()
 {
    let fixture = fixture("reindex-pause-resume");
    let seeded = seed_rows(&fixture.store, TEXT_SPACE, EvidenceKind::Text, 768, 3);
    let result = fixture.store.reindex_space(
        TEXT_SPACE,
        ReindexOptions {
            interrupt_after: Some(1),
        },
    );
    let paused_events = fixture.store.drain_reindex_events();
    let resume_result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let resume_events = fixture.store.drain_reindex_events();
    drop(fixture.store);
    let paused_at_one = paused_events.iter().any(|event| {
        matches!(
            event,
            ReindexEvent::Paused {
                embedding_space_id,
                last_committed_cursor: 1,
            } if embedding_space_id == TEXT_SPACE
        )
    });
    let resumed_at_one = resume_events.iter().any(|event| {
        matches!(
            event,
            ReindexEvent::Resumed {
                embedding_space_id,
                resumed_at_cursor: 1,
            } if embedding_space_id == TEXT_SPACE
        )
    });
    let completed_three = resume_events.iter().any(|event| {
        matches!(
            event,
            ReindexEvent::Completed {
                embedding_space_id,
                rows_processed: 3,
                ..
            } if embedding_space_id == TEXT_SPACE
        )
    });
    let cursor_rows = cursor_at(&fixture.db_path, TEXT_SPACE, 3);
    let new_producer_rows = producer_rows(&fixture.db_path, producer("new"));
    let exact_vectors = all_reindexed_vectors_visible(&fixture.db_path, &seeded);

    assert!(
        matches!(
            result,
            Err(CgError::ReindexPaused {
                ref embedding_space_id,
                last_committed_cursor: 1,
            }) if embedding_space_id == TEXT_SPACE
        ) && paused_at_one
            && resume_result.is_ok()
            && resumed_at_one
            && completed_three
            && cursor_rows == 1
            && new_producer_rows == seeded.len() as i64
            && exact_vectors,
        "pause/resume must be observable through typed events, persisted cursor state, and exact post-resume vectors/producers, result={result:?}, paused_events={paused_events:?}, resume_result={resume_result:?}, resume_events={resume_events:?}, cursor_rows={cursor_rows}, new_producer_rows={new_producer_rows}, exact_vectors={exact_vectors}"
    );
}

#[test]
fn reindex_zero_invalidation_cascades_under_fleet_load_n_decisions_n_entities() {
    let fixture = fixture("reindex-fleet");
    let decisions = seed_active_basis_decisions(&fixture.store, 50);
    seed_rows(&fixture.store, TEXT_SPACE, EvidenceKind::Text, 768, 50);
    seed_rows(
        &fixture.store,
        IMAGE_SPACE,
        EvidenceKind::ImageFrame,
        512,
        50,
    );
    let db_path = fixture.db_path.clone();
    let (first, second) = run_two_reindexes(fixture.store, TEXT_SPACE, IMAGE_SPACE);
    let text_cursors = cursor_count(&db_path, TEXT_SPACE);
    let vision_cursors = cursor_count(&db_path, IMAGE_SPACE);
    let invalidation_rows = count(
        &db_path,
        "SELECT COUNT(*) AS n FROM invalidations",
        &HashMap::new(),
    );
    let positive_control = &decisions[0];
    let trigger_store = open_reindex_store(db_path).expect("reopen positive control store");
    let positive_observation_id = context_graph::ObservationId::new_v7();
    let positive_evidence_id = EvidenceId::new_v7();
    let trigger_control = trigger_store.record_observation(RecordObservation {
        id: positive_observation_id,
        entity_id: positive_control.entity_id,
        context_id: positive_control.context_id,
        observation_type: "fleet-positive-control".to_string(),
        source: "reindex-test".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: positive_evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!(
                "cas://sha256:{:064x}",
                positive_evidence_id.as_uuid().as_u128()
            ),
            mime_type: Some("text/plain".to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer_for_space(TEXT_SPACE, "positive-control"),
            text_projection: Some("fleet state changed".to_string()),
            ..Default::default()
        }],
        observed_properties: Default::default(),
        state_delta: BTreeMap::from([(
            "state".to_string(),
            Some(serde_json::json!("post-reindex-positive-control")),
        )]),
        properties: Default::default(),
        embeddings: Vec::new(),
    });
    let positive_invalidations = trigger_store.list_invalidations(
        positive_control.context_id,
        Some(positive_control.decision_id),
        None,
    );

    assert!(
        first.is_ok()
            && second.is_ok()
            && decisions.len() == 50
            && text_cursors == 1
            && vision_cursors == 1
            && invalidation_rows == 0
            && trigger_control
                .as_ref()
                .is_ok_and(|result| result.invalidations.len() == 1)
            && positive_invalidations
                .as_ref()
                .is_ok_and(|rows| rows.len() == 1
                    && rows[0].affected_decision_id == positive_control.decision_id
                    && rows[0].trigger_observation_id == positive_observation_id),
        "fleet reindex must complete without invalidation callbacks while 50 active basis Decisions exist, and a same-store normal observation must prove the impact-analysis trigger is armed, first={first:?}, second={second:?}, decisions={}, text_cursors={text_cursors}, vision_cursors={vision_cursors}, invalidation_rows={invalidation_rows}, trigger_control={trigger_control:?}, positive_invalidations={positive_invalidations:?}",
        decisions.len()
    );
}

#[test]
fn reindex_on_registered_space_with_no_bindings_returns_typed_no_op_or_error() {
    let fixture = fixture("reindex-no-binding");
    let before_events = fixture.store.drain_reindex_events();
    let result = fixture
        .store
        .reindex_space(UNBOUND_SPACE, ReindexOptions::default());
    let events = fixture.store.drain_reindex_events();
    let completed_zero_rows = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                ReindexEvent::Completed {
                    embedding_space_id,
                    rows_processed: 0,
                    from_producer: None,
                    to_producer: None,
                } if embedding_space_id == UNBOUND_SPACE
            )
        })
        .count();

    assert!(
        before_events.is_empty() && result.is_ok() && events.len() == 1 && completed_zero_rows == 1,
        "unbound registered space must complete as exactly one typed zero-row no-op event and no unrelated reindex events, result={result:?}, before_events={before_events:?}, events={events:?}, completed_zero_rows={completed_zero_rows}"
    );
}

#[test]
fn reindex_emits_typed_completion_event_distinct_from_invalidation_callbacks() {
    let fixture = fixture("reindex-completion");
    let seeded = seed_rows(&fixture.store, TEXT_SPACE, EvidenceKind::Text, 768, 2);
    let before_events = fixture.store.drain_reindex_events();
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let events = fixture.store.drain_reindex_events();
    drop(fixture.store);
    let cursor_rows = cursor_count(&fixture.db_path, TEXT_SPACE);
    let invalidation_rows = count(
        &fixture.db_path,
        "SELECT COUNT(*) AS n FROM invalidations",
        &HashMap::new(),
    );
    let completions = events
        .iter()
        .filter_map(|event| match event {
            ReindexEvent::Completed {
                embedding_space_id,
                rows_processed,
                from_producer,
                to_producer,
            } => Some((
                embedding_space_id.as_str(),
                *rows_processed,
                from_producer.as_deref(),
                to_producer.as_deref(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(
        result.is_ok()
            && cursor_rows == 1
            && invalidation_rows == 0
            && before_events.is_empty()
            && completions
                == vec![(
                    TEXT_SPACE,
                    seeded.len(),
                    Some(producer_event("old").as_str()),
                    Some(producer_event("new").as_str())
                )],
        "completion signal must name real row count and old/new producers and remain distinct from invalidation rows, result={result:?}, cursor_rows={cursor_rows}, invalidation_rows={invalidation_rows}, before_events={before_events:?}, events={events:?}, completions={completions:?}"
    );
}
