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
    BackgroundWork, CgError, CreateContext, CreateDecision, CreateEntity, CreateIntention,
    DistanceMetric, EmbedderRegistration, EmbeddingId, EmbeddingRole, EmbeddingSpace,
    EmbeddingSpaceBinding, EmbeddingTarget, EntityType, EvidenceId, EvidenceKind, EvidenceProducer,
    EvidenceRef, IntentionOrigin, IntentionStatus, ModelIdentity, OldSpace, PrecomputedEmbedding,
    RecordObservation, ReindexEvent, ReindexOptions, RetentionStatus, Store, StoreConfig,
    SwitchResult, VectorPlacement,
};
use contextdb_core::{Value, VectorIndexRef};
use contextdb_engine::Database;

/// The old text model's space: holds the text role while items are seeded.
const OLD_TEXT_SPACE: &str = "reindex_concurrency_old_text_768";
/// The new text model's space: the reindex target after the forward switch.
const TEXT_SPACE: &str = "reindex_concurrency_new_text_768";
/// The old image model's space: holds the image role while frames are seeded.
const OLD_IMAGE_SPACE: &str = "reindex_concurrency_old_image_512";
/// The new image model's space: recomputable on every node from the frame's media.
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

    fn producer(&self) -> ModelIdentity {
        self.embedding_space().producer
    }
}

#[derive(Clone)]
struct CountingVisionConcurrencyEmbedder {
    control: Arc<BlockingReindexControl>,
}

/// The new text model; each run salts its vectors so the run that wrote a row is visible.
#[derive(Clone)]
struct PhaseRecordingConcurrencyEmbedder {
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
                    producer: EvidenceProducer::from(&identity_for_space(TEXT_SPACE)),
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

    fn producer(&self) -> ModelIdentity {
        self.embedding_space().producer
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

    fn producer(&self) -> ModelIdentity {
        self.embedding_space().producer
    }
}

impl Embedder for RecordingConcurrencyEmbedder {
    fn embed(&self, input: EmbeddingInput) -> CgResult<EmbeddingOutput> {
        match input {
            EmbeddingInput::Text(text) => Ok(EmbeddingOutput {
                embedding_space_id: TEXT_SPACE.to_string(),
                dimension: 768,
                producer: EvidenceProducer::from(&identity_for_space(TEXT_SPACE)),
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

    fn producer(&self) -> ModelIdentity {
        self.embedding_space().producer
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
                producer: EvidenceProducer::from(&identity_for_space(IMAGE_SPACE)),
                vector: expected_reindexed_image_vector(&uri),
            }),
            EmbeddingInput::ImageBytes(bytes) => Ok(EmbeddingOutput {
                embedding_space_id: IMAGE_SPACE.to_string(),
                dimension: 512,
                producer: EvidenceProducer::from(&identity_for_space(IMAGE_SPACE)),
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

    fn producer(&self) -> ModelIdentity {
        self.embedding_space().producer
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

/// Which model holds each role in the declared config: the old ones while items are seeded,
/// the new ones once the roles were switched forward (every reopen after the switch).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Roles {
    Seeding,
    Switched,
}

fn store_config(db_path: PathBuf, roles: Roles) -> StoreConfig {
    let seeding = roles == Roles::Seeding;
    StoreConfig {
        db_path,
        extra_embedding_spaces: vec![
            with_default(old_text_space(), seeding.then_some(EvidenceKind::Text)),
            with_default(
                old_image_space(),
                seeding.then_some(EvidenceKind::ImageFrame),
            ),
            unbound_space(),
        ],
        ..Default::default()
    }
}

fn registrations(
    text_embedder: Arc<dyn Embedder>,
    image_embedder: Arc<dyn Embedder>,
    roles: Roles,
) -> Vec<EmbedderRegistration> {
    let switched = roles == Roles::Switched;
    vec![
        EmbedderRegistration {
            space: with_default(text_space(), switched.then_some(EvidenceKind::Text)),
            embedder: text_embedder,
        },
        EmbedderRegistration {
            space: with_default(image_space(), switched.then_some(EvidenceKind::ImageFrame)),
            embedder: image_embedder,
        },
    ]
}

fn with_default(mut space: EmbeddingSpace, default: Option<EvidenceKind>) -> EmbeddingSpace {
    space.default_for_evidence_kind = default;
    space
}

fn fixture_with_embedders(
    name: &str,
    text_embedder: Arc<dyn Embedder>,
    image_embedder: Arc<dyn Embedder>,
) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let store = Store::open_with_embedder_registrations(
        store_config(db_path.clone(), Roles::Seeding),
        registrations(text_embedder, image_embedder, Roles::Seeding),
    )
    .expect("store opens");
    Fixture {
        _dir: dir,
        db_path,
        store,
    }
}

/// Move the text and image roles forward to the new models; the old rows are rebuilt by reindex.
fn switch_forward(store: &Store) -> SwitchResult {
    let text = store
        .switch_embedding_space(EmbeddingRole::EvidenceText, text_space(), OldSpace::Reindex)
        .expect("switch the text role to the new model");
    store
        .switch_embedding_space(
            EmbeddingRole::EvidenceImageFrame,
            image_space(),
            OldSpace::Reindex,
        )
        .expect("switch the image role to the new model");
    text
}

fn identity_for_space(space: &str) -> ModelIdentity {
    let (model, revision, dimension) = match space {
        OLD_TEXT_SPACE => ("embeddinggemma-300m", "old", 768),
        TEXT_SPACE => ("embeddinggemma-300m", "new", 768),
        OLD_IMAGE_SPACE => ("clip-local", "old", 512),
        IMAGE_SPACE => ("clip-local", "new", 512),
        _ => ("unbound-local", "v1", 384),
    };
    ModelIdentity {
        model: model.to_string(),
        weights_revision: revision.to_string(),
        preparation: "reindex-concurrency".to_string(),
        dimension,
    }
}

fn node_local_space(
    id: &str,
    kind: EvidenceKind,
    vector_column: &str,
    media_reachable_on_every_node: bool,
) -> EmbeddingSpace {
    let identity = identity_for_space(id);
    EmbeddingSpace {
        embedding_space_id: id.to_string(),
        dimension: identity.dimension,
        metric: DistanceMetric::Cosine,
        model_family: if kind == EvidenceKind::Text {
            "text"
        } else {
            "vision"
        }
        .to_string(),
        model_name: identity.model.clone(),
        model_version: identity.weights_revision.clone(),
        supported_evidence_kinds: vec![kind],
        default_for_evidence_kind: None,
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: id.to_string(),
            table_name: "evidence".to_string(),
            vector_column: vector_column.to_string(),
        }],
        placement: Some(VectorPlacement::NodeLocal),
        media_reachable_on_every_node,
        producer: identity,
    }
}

fn old_text_space() -> EmbeddingSpace {
    node_local_space(OLD_TEXT_SPACE, EvidenceKind::Text, "vector_text", false)
}

fn text_space() -> EmbeddingSpace {
    node_local_space(TEXT_SPACE, EvidenceKind::Text, "vector_text", false)
}

/// Image vectors are recomputable on every node from the frame's media, so the image role is
/// node-local and reindexes like text.
fn old_image_space() -> EmbeddingSpace {
    node_local_space(
        OLD_IMAGE_SPACE,
        EvidenceKind::ImageFrame,
        "vector_vision",
        true,
    )
}

fn image_space() -> EmbeddingSpace {
    node_local_space(IMAGE_SPACE, EvidenceKind::ImageFrame, "vector_vision", true)
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
        placement: Some(VectorPlacement::TravelsWithItem),
        media_reachable_on_every_node: false,
        producer: identity_for_space(UNBOUND_SPACE),
    }
}

/// A supply of `space`'s own model, as the capture pipeline writes it.
fn producer_for_space(space: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        ..EvidenceProducer::from(&identity_for_space(space))
    }
}

fn reindex_cursor_seed_count() -> usize {
    5
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

/// Seed `rows` items whose vectors are supplied under the old model of `space`'s role.
fn seed_rows(
    store: &Store,
    space: &str,
    kind: EvidenceKind,
    dim: usize,
    rows: usize,
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
    for idx in 0..rows {
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
                producer: producer_for_space(space),
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
                produced_by: producer_for_space(space),
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

/// Node-local vector rows `space` holds, read durably (the progress a reindex has committed).
fn rows_in_space(db_path: &Path, space: &str) -> i64 {
    count(
        db_path,
        "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE embedding_space_id = $space",
        &params(vec![("space", Value::Text(space.to_string()))]),
    )
}

fn store_rows_in_space(store: &Store, space: &str) -> i64 {
    let result = store
        .execute_sql_for_test(
            "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE embedding_space_id = $space",
            &params(vec![("space", Value::Text(space.to_string()))]),
        )
        .expect("store count query");
    match result.rows.first().and_then(|row| row.first()) {
        Some(Value::Int64(value)) => *value,
        _ => 0,
    }
}

fn embedded_in(store: &Store, space: &str) -> u64 {
    store
        .embedding_space_status()
        .expect("embedding space status")
        .into_iter()
        .find(|status| status.space_id == space)
        .map(|status| status.embedded)
        .unwrap_or_else(|| panic!("status names space {space}"))
}

/// The item's vector row in `space` (the node-local table, by item and space).
fn vector_row(
    db: &Database,
    evidence_id: EvidenceId,
    space: &str,
) -> Option<contextdb_core::RowId> {
    db.scan("evidence_embeddings", db.snapshot())
        .expect("scan evidence_embeddings")
        .into_iter()
        .find_map(|row| {
            (row.values.get("target_id") == Some(&Value::Uuid(evidence_id.as_uuid()))
                && row.values.get("embedding_space_id") == Some(&Value::Text(space.to_string())))
            .then_some(row.row_id)
        })
}

fn vector_score_for(
    db_path: &Path,
    evidence_id: EvidenceId,
    column: &str,
    query: Vec<f32>,
) -> Option<f32> {
    let db = open_raw_database(db_path);
    let space = if column == "vector_vision" {
        IMAGE_SPACE
    } else {
        TEXT_SPACE
    };
    let row_id = vector_row(&db, evidence_id, space)?;
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence_embeddings", column),
            &query,
            32,
            None,
            db.snapshot(),
        )
        .expect("vector query");
    hits.into_iter()
        .find_map(|(hit_row, score)| (hit_row == row_id).then_some(score))
}

fn stored_text_vector(db_path: &Path, evidence_id: EvidenceId) -> Option<Vec<f32>> {
    let db = open_raw_database(db_path);
    let snapshot = db.snapshot();
    let row_id = vector_row(&db, evidence_id, TEXT_SPACE)?;
    db.live_vector_entry(row_id, snapshot)
        .expect("admit engine read memory")
        .filter(|entry| entry.index == VectorIndexRef::new("evidence_embeddings", "vector_text"))
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

fn text_vector_matches_exactly(db_path: &Path, evidence_id: EvidenceId, expected: &[f32]) -> bool {
    stored_text_vector(db_path, evidence_id).is_some_and(|stored| stored == expected)
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
            VectorIndexRef::new("evidence_embeddings", "vector_text"),
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

/// Reopen after the forward switch, with the new text model's embedder registered.
fn open_reindex_store_with_embedder(
    db_path: PathBuf,
    embedder: Arc<dyn Embedder>,
) -> context_graph::Result<Store> {
    Store::open_with_embedder_registrations(
        store_config(db_path, Roles::Switched),
        registrations(
            embedder,
            Arc::new(RecordingVisionConcurrencyEmbedder),
            Roles::Switched,
        ),
    )
}

fn open_reindex_store(db_path: PathBuf) -> context_graph::Result<Store> {
    open_reindex_store_with_embedder(db_path, Arc::new(RecordingConcurrencyEmbedder))
}

fn run_two_reindexes(
    store: Store,
    first_space: &'static str,
    second_space: &'static str,
) -> (
    CgResult<context_graph::ReindexReport>,
    CgResult<context_graph::ReindexReport>,
) {
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

/// Every `Completed` event as (space, embedded, skipped).
fn completions(events: &[ReindexEvent]) -> Vec<(String, u64, u64)> {
    events
        .iter()
        .filter_map(|event| match event {
            ReindexEvent::Completed {
                embedding_space_id,
                embedded,
                skipped,
            } => Some((embedding_space_id.clone(), *embedded, *skipped)),
            _ => None,
        })
        .collect()
}

#[test]
fn reindex_concurrent_invocations_on_same_space_serialize_or_reject_typed() {
    let fixture = fixture("reindex-same-space");
    let seeded = seed_rows(&fixture.store, OLD_TEXT_SPACE, EvidenceKind::Text, 768, 3);
    switch_forward(&fixture.store);
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
    let second_store = open_reindex_store(db_path.clone()).expect("second reindex handle opens");
    // Durable progress seen from an independent handle while the first is inside row 2.
    let committed_during_contention = embedded_in(&second_store, TEXT_SPACE);
    let second_result = second_store.reindex_space(TEXT_SPACE, ReindexOptions::default());
    let second_events = second_store.drain_reindex_events();
    drop(second_store);
    control.release();
    let (first_result, first_events) = first.join().expect("first reindex thread");
    let first_completed_seeded =
        completions(&first_events) == vec![(TEXT_SPACE.to_string(), seeded.len() as u64, 0)];
    let new_model_rows = rows_in_space(&db_path, TEXT_SPACE);
    let old_model_rows = rows_in_space(&db_path, OLD_TEXT_SPACE);
    let exact_vectors = all_reindexed_vectors_visible(&db_path, &seeded);

    assert!(
        first_blocked_inside_active_reindex
            && committed_during_contention == 1
            && matches!(
                second_result,
                Err(CgError::BackgroundWorkHeldByLiveWorker {
                    work: BackgroundWork::Reindex,
                    ..
                })
            )
            && second_events.is_empty()
            && first_result
                .as_ref()
                .is_ok_and(|report| report.embedded == seeded.len() as u64 && report.remaining == 0)
            && first_completed_seeded
            && new_model_rows == seeded.len() as i64
            && old_model_rows == 0 && exact_vectors,
        "same-space reindex must hold the shared background claim across handles while the first handle is inside active row work: the first handle blocks on row 2 with row 1 committed, an independent Store handle is refused with BackgroundWorkHeldByLiveWorker and emits no progress events, then the first handle releases and completes with every item rebuilt under the new model and each old row deleted with its replacement; first_blocked_inside_active_reindex={first_blocked_inside_active_reindex}, committed_during_contention={committed_during_contention}, first_result={first_result:?}, first_events={first_events:?}, second_result={second_result:?}, second_events={second_events:?}, new_model_rows={new_model_rows}, old_model_rows={old_model_rows}, exact_vectors={exact_vectors}"
    );
}

#[test]
fn reindex_concurrent_invocations_on_distinct_spaces_proceed_in_parallel() {
    let row_count = 20;
    let serial_fixture = fixture("reindex-distinct-serial");
    let serial_text_seeded = seed_rows(
        &serial_fixture.store,
        OLD_TEXT_SPACE,
        EvidenceKind::Text,
        768,
        row_count,
    );
    let serial_image_seeded = seed_rows(
        &serial_fixture.store,
        OLD_IMAGE_SPACE,
        EvidenceKind::ImageFrame,
        512,
        row_count,
    );
    switch_forward(&serial_fixture.store);
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
    let serial_text_new_model_rows = rows_in_space(&serial_db_path, TEXT_SPACE);
    let serial_image_new_model_rows = rows_in_space(&serial_db_path, IMAGE_SPACE);

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
        OLD_TEXT_SPACE,
        EvidenceKind::Text,
        768,
        row_count,
    );
    let image_seeded = seed_rows(
        &fixture.store,
        OLD_IMAGE_SPACE,
        EvidenceKind::ImageFrame,
        512,
        row_count,
    );
    switch_forward(&fixture.store);
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
    let text_vectors = all_reindexed_vectors_visible(&db_path, &text_seeded);
    let image_vectors = all_reindexed_image_vectors_visible(&db_path, &image_seeded);
    let text_new_model_rows = rows_in_space(&db_path, TEXT_SPACE);
    let image_new_model_rows = rows_in_space(&db_path, IMAGE_SPACE);
    let old_model_rows =
        rows_in_space(&db_path, OLD_TEXT_SPACE) + rows_in_space(&db_path, OLD_IMAGE_SPACE);

    assert!(
        serial_text.is_ok()
            && serial_image.is_ok()
            && serial_text_vectors
            && serial_image_vectors
            && serial_text_new_model_rows == row_count as i64
            && serial_image_new_model_rows == row_count as i64
            && text_blocked_inside_active_reindex
            && image_progress_while_text_blocked
            && first
                .as_ref()
                .is_ok_and(|report| report.embedded == row_count as u64)
            && second
                .as_ref()
                .is_ok_and(|report| report.embedded == row_count as u64)
            && text_vectors
            && image_vectors
            && text_new_model_rows == row_count as i64
            && image_new_model_rows == row_count as i64
            && old_model_rows == 0,
        "distinct-space reindexes must each re-embed exact text/image rows under their new models and allow image row work to enter while text row work is actively blocked; this probe intentionally does not require overlapping long Database commit apply windows, row_count={row_count}, serial_text={serial_text:?}/{serial_text_elapsed:?}, serial_image={serial_image:?}/{serial_image_elapsed:?}, serial_text_vectors={serial_text_vectors}, serial_image_vectors={serial_image_vectors}, serial_text_new_model_rows={serial_text_new_model_rows}, serial_image_new_model_rows={serial_image_new_model_rows}, text_blocked_inside_active_reindex={text_blocked_inside_active_reindex}, image_progress_while_text_blocked={image_progress_while_text_blocked}, first={first:?}, second={second:?}, parallel={parallel_elapsed:?}, text_vectors={text_vectors}, image_vectors={image_vectors}, text_new_model_rows={text_new_model_rows}, image_new_model_rows={image_new_model_rows}, old_model_rows={old_model_rows}"
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
    switch_forward(&fixture.store);
    let before_events = fixture.store.drain_reindex_events();
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let events = fixture.store.drain_reindex_events();
    let new_model_rows = store_rows_in_space(&fixture.store, TEXT_SPACE);
    let invalidation_rows = count(
        &fixture.db_path,
        "SELECT COUNT(*) AS n FROM invalidations",
        &HashMap::new(),
    );
    let completions = completions(&events);
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
            producer: producer_for_space(TEXT_SPACE),
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
            && result.as_ref().is_ok_and(|report| report.embedded == 0
                && report.skipped == 0
                && report.remaining == 0)
            && new_model_rows == 0
            && invalidation_rows == 0
            && completions == vec![(TEXT_SPACE.to_string(), 0, 0)]
            && positive_control
                .as_ref()
                .is_ok_and(|result| result.invalidations.len() == 1)
            && positive_invalidations
                .as_ref()
                .is_ok_and(|rows| rows.len() == 1
                    && rows[0].affected_decision_id == decision.id
                    && rows[0].trigger_observation_id == positive_observation_id),
        "empty-space reindex must complete with a zero-item report, one typed completion event, and no invalidation callbacks while a same-store normal observation proves the impact-analysis trigger is active, result={result:?}, before_events={before_events:?}, events={events:?}, new_model_rows={new_model_rows}, invalidation_rows={invalidation_rows}, completions={completions:?}, positive_control={positive_control:?}, positive_invalidations={positive_invalidations:?}"
    );
}

#[test]
fn reindex_on_single_row_space_processes_exactly_one_row() {
    let fixture = fixture("reindex-one-row");
    seed_rows(&fixture.store, OLD_TEXT_SPACE, EvidenceKind::Text, 768, 1);
    switch_forward(&fixture.store);
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let embedded = embedded_in(&fixture.store, TEXT_SPACE);
    drop(fixture.store);
    let rows = rows_in_space(&fixture.db_path, TEXT_SPACE);

    assert!(
        result
            .as_ref()
            .is_ok_and(|report| report.embedded == 1 && report.remaining == 0)
            && embedded == 1
            && rows == 1,
        "single-row reindex must embed exactly one item and leave none remaining, result={result:?}, embedded={embedded}, rows={rows}"
    );
}

#[test]
fn reindex_resumable_after_process_restart_persists_cursor_to_disk() {
    if std::env::var_os("CG_REINDEX_CURSOR_CHILD").is_some() {
        let db_path = PathBuf::from(std::env::var("CG_REINDEX_CURSOR_DB").expect("db env"));
        let seeded = reindex_cursor_seeded_from_env(&db_path);
        let resume_salt = new_reindex_vector_salt(0x7265_7375_6d65_6e65);
        let resume_calls = Arc::new(Mutex::new(Vec::new()));
        let store = open_reindex_store_with_embedder(
            db_path.clone(),
            Arc::new(PhaseRecordingConcurrencyEmbedder {
                vector_salt: resume_salt,
                calls: Arc::clone(&resume_calls),
            }),
        )
        .expect("child resume store opens");
        let resume_result = store.reindex_space(TEXT_SPACE, ReindexOptions::default());
        let resume_events = store.drain_reindex_events();
        drop(store);
        let remaining_at_start = (seeded.len() - 2) as u64;
        let resumed_with_three_left = resume_events.iter().any(|event| {
            matches!(
                event,
                ReindexEvent::Resumed {
                    embedding_space_id,
                    remaining,
                } if embedding_space_id == TEXT_SPACE && *remaining == remaining_at_start
            )
        });
        let completed_three =
            completions(&resume_events) == vec![(TEXT_SPACE.to_string(), remaining_at_start, 0)];
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
        let new_model_rows = rows_in_space(&db_path, TEXT_SPACE);
        let old_model_rows = rows_in_space(&db_path, OLD_TEXT_SPACE);

        assert!(
            resume_result
                .as_ref()
                .is_ok_and(|report| report.embedded == remaining_at_start && report.remaining == 0)
                && resumed_with_three_left
                && completed_three
                && resume_call_shape
                && resume_vectors_not_text_formula
                && resume_vectors_from_output_visible
                && resume_vectors_from_output_indexed
                && new_model_rows == seeded.len() as i64
                && old_model_rows == 0,
            "fresh child process must resume with exactly the 3 unembedded items left, persist and index exact runtime-salted Embedder output vectors rather than text-derived vectors, and avoid parent process state or handle-derived progress, resume_result={resume_result:?}, resume_events={resume_events:?}, resume_call_texts={resume_call_texts:?}, resume_vectors_not_text_formula={resume_vectors_not_text_formula}, resume_vectors_from_output_visible={resume_vectors_from_output_visible}, resume_vectors_from_output_indexed={resume_vectors_from_output_indexed}, new_model_rows={new_model_rows}, old_model_rows={old_model_rows}"
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
            vector_salt: pause_salt,
            calls: Arc::clone(&pause_calls),
        }),
        Arc::new(RecordingVisionConcurrencyEmbedder),
    );
    let seeded = seed_rows(
        &fixture.store,
        OLD_TEXT_SPACE,
        EvidenceKind::Text,
        768,
        reindex_cursor_seed_count(),
    );
    switch_forward(&fixture.store);
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
    let embedded_before_drop = embedded_in(&fixture.store, TEXT_SPACE);
    let paused_events_shape = paused_events
        == vec![ReindexEvent::Paused {
            embedding_space_id: TEXT_SPACE.to_string(),
            remaining: 3,
        }];
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
    let paused_new_model_rows = rows_in_space(&fixture.db_path, TEXT_SPACE);
    let paused_old_model_rows = rows_in_space(&fixture.db_path, OLD_TEXT_SPACE);
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
    let final_new_model_rows = rows_in_space(&fixture.db_path, TEXT_SPACE);
    let final_old_model_rows = rows_in_space(&fixture.db_path, OLD_TEXT_SPACE);
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
                remaining: 3,
            }) if embedding_space_id == TEXT_SPACE
        ) && pause_call_shape
            && pause_vectors_not_text_formula
            && pause_vectors_indexed_before_drop
            && pause_vectors_from_output_visible
            && pause_vectors_from_output_indexed
            && pause_vectors_only_first_two
            && embedded_before_drop == 2
            && paused_new_model_rows == 2
            && paused_old_model_rows == 3
            && paused_events_shape
            && child.status.success()
            && final_new_model_rows == seeded.len() as i64
            && final_old_model_rows == 0
            && resume_vectors_from_output_visible_after_child
            && resume_vectors_from_output_indexed_after_child
            && pause_vectors_still_visible,
        "interrupted reindex must commit items 0..1 before a fresh child-process resume, emit only the typed pause event naming the 3 items left, persist and index the exact runtime-salted Embedder output vectors from persisted DB text rather than synthesized ordinal text, keep each not-yet-rebuilt item's old row until its replacement is written, resume with only items 2..4 without rewriting 0..1, and end with every item under the new model and no old row left, result={result:?}, paused_events={paused_events:?}, paused_events_shape={paused_events_shape}, pause_call_texts={pause_call_texts:?}, pause_vectors_not_text_formula={pause_vectors_not_text_formula}, pause_vectors_indexed_before_drop={pause_vectors_indexed_before_drop}, pause_vectors_from_output_visible={pause_vectors_from_output_visible}, pause_vectors_from_output_indexed={pause_vectors_from_output_indexed}, pause_vectors_only_first_two={pause_vectors_only_first_two}, pause_vectors_still_visible={pause_vectors_still_visible}, embedded_before_drop={embedded_before_drop}, paused_new_model_rows={paused_new_model_rows}, paused_old_model_rows={paused_old_model_rows}, child_status={:?}, child_stdout={}, child_stderr={}, final_new_model_rows={final_new_model_rows}, final_old_model_rows={final_old_model_rows}, resume_vectors_from_output_visible_after_child={resume_vectors_from_output_visible_after_child}, resume_vectors_from_output_indexed_after_child={resume_vectors_from_output_indexed_after_child}",
        child.status.code(),
        child_stdout,
        String::from_utf8_lossy(&child.stderr)
    );
}

#[test]
fn reindex_emits_observable_pause_signal_when_underlying_io_unavailable_and_resume_signal_when_restored()
 {
    let fixture = fixture("reindex-pause-resume");
    let seeded = seed_rows(&fixture.store, OLD_TEXT_SPACE, EvidenceKind::Text, 768, 3);
    switch_forward(&fixture.store);
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
    let paused_with_two_left = paused_events.iter().any(|event| {
        matches!(
            event,
            ReindexEvent::Paused {
                embedding_space_id,
                remaining: 2,
            } if embedding_space_id == TEXT_SPACE
        )
    });
    let resumed_with_two_left = resume_events.iter().any(|event| {
        matches!(
            event,
            ReindexEvent::Resumed {
                embedding_space_id,
                remaining: 2,
            } if embedding_space_id == TEXT_SPACE
        )
    });
    let completed_two = completions(&resume_events) == vec![(TEXT_SPACE.to_string(), 2, 0)];
    let new_model_rows = rows_in_space(&fixture.db_path, TEXT_SPACE);
    let exact_vectors = all_reindexed_vectors_visible(&fixture.db_path, &seeded);

    assert!(
        matches!(
            result,
            Err(CgError::ReindexPaused {
                ref embedding_space_id,
                remaining: 2,
            }) if embedding_space_id == TEXT_SPACE
        ) && paused_with_two_left
            && resume_result
                .as_ref()
                .is_ok_and(|report| report.embedded == 2 && report.remaining == 0)
            && resumed_with_two_left
            && completed_two
            && new_model_rows == seeded.len() as i64
            && exact_vectors,
        "pause/resume must be observable through typed events naming what is left, durable vector rows, and exact post-resume vectors, result={result:?}, paused_events={paused_events:?}, resume_result={resume_result:?}, resume_events={resume_events:?}, new_model_rows={new_model_rows}, exact_vectors={exact_vectors}"
    );
}

#[test]
fn reindex_zero_invalidation_cascades_under_fleet_load_n_decisions_n_entities() {
    let fixture = fixture("reindex-fleet");
    let decisions = seed_active_basis_decisions(&fixture.store, 50);
    seed_rows(&fixture.store, OLD_TEXT_SPACE, EvidenceKind::Text, 768, 50);
    seed_rows(
        &fixture.store,
        OLD_IMAGE_SPACE,
        EvidenceKind::ImageFrame,
        512,
        50,
    );
    switch_forward(&fixture.store);
    let db_path = fixture.db_path.clone();
    let (first, second) = run_two_reindexes(fixture.store, TEXT_SPACE, IMAGE_SPACE);
    let text_rows = rows_in_space(&db_path, TEXT_SPACE);
    let image_rows = rows_in_space(&db_path, IMAGE_SPACE);
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
            producer: producer_for_space(TEXT_SPACE),
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
        first.as_ref().is_ok_and(|report| report.embedded == 50)
            && second.as_ref().is_ok_and(|report| report.embedded == 50)
            && decisions.len() == 50
            && text_rows == 50
            && image_rows == 50
            && invalidation_rows == 0
            && trigger_control
                .as_ref()
                .is_ok_and(|result| result.invalidations.len() == 1)
            && positive_invalidations
                .as_ref()
                .is_ok_and(|rows| rows.len() == 1
                    && rows[0].affected_decision_id == positive_control.decision_id
                    && rows[0].trigger_observation_id == positive_observation_id),
        "fleet reindex must complete without invalidation callbacks while 50 active basis Decisions exist, and a same-store normal observation must prove the impact-analysis trigger is armed, first={first:?}, second={second:?}, decisions={}, text_rows={text_rows}, image_rows={image_rows}, invalidation_rows={invalidation_rows}, trigger_control={trigger_control:?}, positive_invalidations={positive_invalidations:?}",
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

    assert!(
        before_events.is_empty()
            && result.as_ref().is_ok_and(|report| report.embedded == 0
                && report.skipped == 0
                && report.remaining == 0)
            && completions(&events) == vec![(UNBOUND_SPACE.to_string(), 0, 0)]
            && events.len() == 1,
        "unbound registered space must complete as exactly one typed zero-item no-op event and no unrelated reindex events, result={result:?}, before_events={before_events:?}, events={events:?}"
    );
}

#[test]
fn reindex_emits_typed_completion_event_distinct_from_invalidation_callbacks() {
    let fixture = fixture("reindex-completion");
    let seeded = seed_rows(&fixture.store, OLD_TEXT_SPACE, EvidenceKind::Text, 768, 2);
    let switched = switch_forward(&fixture.store);
    let before_events = fixture.store.drain_reindex_events();
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let events = fixture.store.drain_reindex_events();
    drop(fixture.store);
    let new_model_rows = rows_in_space(&fixture.db_path, TEXT_SPACE);
    let invalidation_rows = count(
        &fixture.db_path,
        "SELECT COUNT(*) AS n FROM invalidations",
        &HashMap::new(),
    );
    let completions = completions(&events);

    assert!(
        result.is_ok()
            && switched.from_space == OLD_TEXT_SPACE
            && switched.to_space == TEXT_SPACE
            && new_model_rows == seeded.len() as i64
            && invalidation_rows == 0
            && before_events.is_empty()
            && completions == vec![(TEXT_SPACE.to_string(), seeded.len() as u64, 0)],
        "completion signal must name the real embedded count for the switched-to space, the switch must name the old and new spaces, and both stay distinct from invalidation rows, result={result:?}, switched={switched:?}, new_model_rows={new_model_rows}, invalidation_rows={invalidation_rows}, before_events={before_events:?}, events={events:?}, completions={completions:?}"
    );
}
