//! Embedding placement per space, and reindex as shared background work.
//!
//! The contract: every embedding space declares where its vectors live. A reproducible space is
//! node-local — each node builds its own copy. Callers and media spaces declare their placement
//! too: caller-only values may be held only here, while travelling values follow their item
//! authorization and converge across nodes because a repeat replaces. A space is one model identity, so a new model or weights revision is a new
//! space; switching moves forward only. Rebuildable history is re-embedded locally, history that
//! cannot be regenerated is preserved in its old space and reported, and no item ever loses its
//! only vector. Reindex is background work on the shared crash-recovery mechanism.
//!
//! Recorded reason for a new file: the existing reindex and enrollment tests are built on the
//! contract these rulings replace (one space whose producer changes in place, a durable reindex
//! cursor, inline vector columns, random enrollment ids), so no existing fixture can host these
//! proofs; each one names the ruling behaviour it proves.

#![cfg(feature = "test-support")]

#[path = "common/fabric_common.rs"]
#[allow(dead_code)]
mod fabric_common;
#[path = "common/fake_extractor.rs"]
#[allow(dead_code)]
mod fake_extractor;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use context_graph::embedder::{EmbedderConfig, ModelSource};
use context_graph::{
    BackgroundWorkState, CgError, ContextId, CreateContext, CreateEntity, DigestSearchFilter,
    DistanceMetric, Embedder, EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingRole,
    EmbeddingSpace, EmbeddingSpaceBinding, EmbeddingTarget, EntityId, EntityReferenceMatchOptions,
    EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, ModelIdentity, ObservationId,
    OldSpace, RecordObservation, ReindexOptions, RetentionStatus, SpaceState, SpaceStatus, Store,
    StoreConfig, VectorPlacement,
};
use contextdb_core::Value;

/// More than one 32-item reindex batch.
const ITEMS: usize = 40;
const TEXT_DIMENSION: usize = 16;
const FACE_DIMENSION: usize = 8;
/// A space of a non-base dimension binds its own column; registration adds it to the space's
/// placement table (the base `vector_text` is 768-d and `vector_vision` 512-d).
const TEXT_COLUMN: &str = "vector_text_16";
const FACE_COLUMN: &str = "vector_vision_8";
const CHILD_PATH: &str = "CG_EMBEDDING_REINDEX_CHILD_PATH";
const CHILD_CLOCK: &str = "CG_EMBEDDING_REINDEX_CHILD_CLOCK";
const RECOVERY_CLOCK: u64 = 1_800_000_000_000;
/// The embedder's call on which the reindexing child dies: the first call of the second batch.
const DIE_ON_CALL: usize = 33;
const EXIT_MID_REINDEX: i32 = 72;

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

/// A deterministic text model at one weights revision: a reproducible, node-local space. It counts
/// its calls, can die on one call, and can refuse texts containing a marker.
struct TextModel {
    revision: String,
    calls: AtomicUsize,
    die_on_call: Option<usize>,
    refuse: Mutex<Option<String>>,
}

impl TextModel {
    fn new(revision: &str) -> Arc<Self> {
        Arc::new(Self {
            revision: revision.into(),
            calls: AtomicUsize::new(0),
            die_on_call: None,
            refuse: Mutex::new(None),
        })
    }

    fn dying_on(revision: &str, call: usize) -> Arc<Self> {
        Arc::new(Self {
            revision: revision.into(),
            calls: AtomicUsize::new(0),
            die_on_call: Some(call),
            refuse: Mutex::new(None),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn refuse(&self, marker: Option<&str>) {
        *self.refuse.lock().expect("refusal marker") = marker.map(str::to_string);
    }

    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }
}

fn text_space(revision: &str) -> EmbeddingSpace {
    let id = format!("placement_text_{TEXT_DIMENSION}_{revision}");
    EmbeddingSpace {
        embedding_space_id: id.clone(),
        dimension: TEXT_DIMENSION,
        metric: DistanceMetric::Cosine,
        model_family: "text".into(),
        model_name: "placement-text".into(),
        model_version: revision.into(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: Some(EvidenceKind::Text),
        bindings: vec![
            EmbeddingSpaceBinding {
                embedding_space_id: id.clone(),
                table_name: "evidence".into(),
                vector_column: TEXT_COLUMN.into(),
            },
            EmbeddingSpaceBinding {
                embedding_space_id: id,
                table_name: "digests".into(),
                vector_column: TEXT_COLUMN.into(),
            },
        ],
        placement: Some(VectorPlacement::NodeLocal),
        media_reachable_on_every_node: false,
        producer: text_identity(revision),
    }
}

fn text_vector(text: &str) -> Vec<f32> {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(text.as_bytes());
    (0..TEXT_DIMENSION)
        .map(|index| f32::from(hash[index]) / 255.0 + 0.01)
        .collect()
}

impl Embedder for TextModel {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        let EmbeddingInput::Text(text) = input else {
            return Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::ImageFrame,
            });
        };
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.die_on_call == Some(call) {
            std::process::exit(EXIT_MID_REINDEX);
        }
        if let Some(marker) = self.refuse.lock().expect("refusal marker").as_deref()
            && text.contains(marker)
        {
            return Err(CgError::Engine(format!("model refused {marker}")));
        }
        Ok(EmbeddingOutput {
            embedding_space_id: text_space(&self.revision).embedding_space_id,
            dimension: TEXT_DIMENSION,
            producer: EvidenceProducer::from(&self.producer()),
            vector: text_vector(&format!("{}:{text}", self.revision)),
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        text_space(&self.revision)
    }

    fn producer(&self) -> ModelIdentity {
        text_identity(&self.revision)
    }
}

/// A media model boundary that really reads the supplied file; a deleted file cannot be embedded.
struct MediaModel {
    revision: &'static str,
    calls: AtomicUsize,
}
impl Embedder for MediaModel {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let EmbeddingInput::ImageFrame { uri } = input else {
            return Err(CgError::UnsupportedEvidenceKind {
                kind: input.evidence_kind(),
            });
        };
        let bytes = std::fs::read(uri.strip_prefix("file://").unwrap_or(&uri))
            .map_err(|error| CgError::Engine(format!("media source unavailable: {error}")))?;
        let mut vector = axis(usize::from(bytes[0]) % FACE_DIMENSION);
        vector[(usize::from(bytes[0]) + 1) % FACE_DIMENSION] = 0.25;
        Ok(EmbeddingOutput {
            embedding_space_id: self.embedding_space().embedding_space_id,
            dimension: FACE_DIMENSION,
            producer: EvidenceProducer::from(&self.producer()),
            vector,
        })
    }
    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::ImageFrame]
    }
    fn embedding_space(&self) -> EmbeddingSpace {
        let mut space = face_space(self.revision);
        space.default_for_evidence_kind = Some(EvidenceKind::ImageFrame);
        space
    }
    fn producer(&self) -> ModelIdentity {
        face_identity(self.revision)
    }
}

/// A sealed, content-faithful event model; it runs on token bytes instead of inventing passage rows.
struct EventModel {
    calls: AtomicUsize,
}
impl EventModel {
    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }
    fn output(&self, tokens: &[u32]) -> EmbeddingOutput {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let text: String = tokens
            .iter()
            .map(|token| char::from_u32(*token).expect("byte token"))
            .collect();
        let mut vector = vec![0.0; 768];
        vector[..TEXT_DIMENSION].copy_from_slice(&text_vector(&text));
        EmbeddingOutput {
            embedding_space_id: self.embedding_space().embedding_space_id,
            dimension: 768,
            producer: EvidenceProducer::from(&self.producer()),
            vector,
        }
    }
}
impl Embedder for EventModel {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        let EmbeddingInput::Text(text) = input else {
            return Err(CgError::UnsupportedEvidenceKind {
                kind: input.evidence_kind(),
            });
        };
        Ok(self.output(&text.bytes().map(u32::from).collect::<Vec<_>>()))
    }
    fn producer(&self) -> ModelIdentity {
        ModelIdentity {
            model: "placement-event".into(),
            weights_revision: "v1".into(),
            preparation: "byte-event".into(),
            dimension: 768,
        }
    }
    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }
    fn embedding_space(&self) -> EmbeddingSpace {
        let mut space = context_graph::embedding_space::event_history_embedding_space();
        space.embedding_space_id = "placement-event-768".into();
        space.model_name = "placement-event".into();
        space.model_version = "v1".into();
        space.producer = self.producer();
        space.bindings[0].embedding_space_id = space.embedding_space_id.clone();
        space
    }
    fn encoding_profile_id(&self) -> String {
        "placement-byte-event-v1".into()
    }
    fn event_search_capability(&self) -> Option<context_graph::embedder::EventSearchCapability> {
        context_graph::embedder::EventSearchCapability::new(8192)
    }
    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &context_graph::embedder::EventSearchSeal,
    ) -> context_graph::Result<context_graph::embedder::PreparedEventSearchDocument> {
        seal.document_builder()
            .token_count(text.len())
            .byte_boundaries((0..text.len()).map(|index| (index, index + 1)).collect())
            .payload(text.bytes().map(u32::from).collect())
            .seal()
    }
    fn prepare_query_hook(
        &self,
        text: &str,
        seal: &context_graph::embedder::EventSearchSeal,
    ) -> context_graph::Result<context_graph::embedder::PreparedEventSearchQuery> {
        seal.query_builder()
            .token_count(text.len())
            .payload(text.bytes().map(u32::from).collect())
            .seal()
    }
    fn embed_prepared_document_hook(
        &self,
        input: &context_graph::embedder::EventSearchInference,
    ) -> context_graph::Result<EmbeddingOutput> {
        Ok(self.output(input.token_ids()))
    }
    fn embed_prepared_query_hook(
        &self,
        input: &context_graph::embedder::EventSearchInference,
    ) -> context_graph::Result<EmbeddingOutput> {
        Ok(self.output(input.token_ids()))
    }
}

fn text_identity(revision: &str) -> ModelIdentity {
    ModelIdentity {
        model: "placement-text".into(),
        weights_revision: revision.into(),
        preparation: "plain".into(),
        dimension: TEXT_DIMENSION as u32,
    }
}

fn plain_config(path: &Path) -> StoreConfig {
    StoreConfig {
        db_path: path.to_path_buf(),
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn open_with(path: &Path, models: &[&Arc<TextModel>]) -> Store {
    let store = Store::open_with_embedder_registrations(
        plain_config(path),
        models.iter().map(|model| model.registration()).collect(),
    )
    .expect("open with the text models");
    if models.len() > 1 {
        for model in models {
            assert_eq!(
                rows(
                    &store,
                    "SELECT default_for_evidence_kind FROM embedding_spaces WHERE embedding_space_id = $space",
                    vec![(
                        "space",
                        Value::Text(model.embedding_space().embedding_space_id)
                    )]
                ),
                vec![vec![Value::Text("text".into())]],
                "persisted model generations keep their immutable declared default after succession"
            );
        }
    }
    store
}

/// A store whose configured default text embedder is the deterministic test model at `revision`.
fn configured(path: &Path, revision: &str) -> StoreConfig {
    StoreConfig {
        db_path: path.to_path_buf(),
        default_text_embedder: Some(EmbedderConfig {
            source: ModelSource::TestLexical {
                weights_revision: revision.to_string(),
            },
            offline: true,
            cache_dir: None,
            pinned_revision: None,
            network_policy: "offline".into(),
        }),
        ..StoreConfig::default()
    }
}

fn create_context(store: &Store) -> ContextId {
    store
        .create_context(CreateContext {
            name: "placement".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id
}

fn create_entity(store: &Store, context_id: ContextId, name: &str) -> EntityId {
    store
        .create_entity(CreateEntity {
            name: name.into(),
            context_id,
            ..CreateEntity::default()
        })
        .expect("create entity")
        .id
}

fn capture_producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "placement-fixture".into(),
        model_name: "capture".into(),
        model_version: "1".into(),
        pipeline_version: "1".into(),
    }
}

/// Record one observation carrying one text evidence item; the store embeds it in the text role's
/// active space.
fn record_text(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    text: &str,
) -> EvidenceId {
    let evidence_id = EvidenceId::new_v7();
    store
        .record_observation(RecordObservation {
            id: ObservationId::new_v7(),
            entity_id,
            context_id,
            observation_type: "note".into(),
            source: "placement-fixture".into(),
            observed_at: chrono::DateTime::from_timestamp(1_790_000_000, 0).expect("fixed time"),
            evidence: vec![EvidenceRef {
                id: evidence_id,
                kind: EvidenceKind::Text,
                source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
                retention_status: RetentionStatus::RetainedExternal,
                producer: capture_producer(),
                text_projection: Some(text.to_string()),
                ..EvidenceRef::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record text evidence");
    evidence_id
}

fn note(index: usize) -> String {
    format!("note number {index}")
}

/// `ITEMS` text items recorded under the text role's active space.
fn seed_items(store: &Store) -> (ContextId, Vec<EvidenceId>) {
    let context_id = create_context(store);
    let entity_id = create_entity(store, context_id, "notebook");
    let evidence = (0..ITEMS)
        .map(|index| record_text(store, context_id, entity_id, &note(index)))
        .collect();
    (context_id, evidence)
}

fn statuses(store: &Store) -> Vec<SpaceStatus> {
    store
        .embedding_space_status()
        .expect("embedding space status")
}

fn status_of(store: &Store, space_id: &str) -> SpaceStatus {
    statuses(store)
        .into_iter()
        .find(|status| status.space_id == space_id)
        .unwrap_or_else(|| panic!("status names space {space_id}"))
}

fn active(store: &Store, role: EmbeddingRole) -> String {
    store
        .active_embedding_space(role)
        .expect("read the active space")
        .unwrap_or_else(|| panic!("{role:?} has an active space"))
}

fn rows(store: &Store, sql: &str, params: Vec<(&str, Value)>) -> Vec<Vec<Value>> {
    store
        .execute_sql_for_test(
            sql,
            &params
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect::<HashMap<_, _>>(),
        )
        .expect("inspect store rows")
        .rows
}

/// Snapshot each vector as target -> (created_at, vector, actual deterministic id).
/// Keeping the id makes every crash, skip, retention and restore snapshot compare identity too.
/// The column a space's vectors live in: the lexical mock is 768-d and uses the base column.
fn column_of(space_id: &str) -> &'static str {
    if space_id.starts_with("placement_face") {
        FACE_COLUMN
    } else if space_id.starts_with("placement_") {
        TEXT_COLUMN
    } else {
        "vector_text"
    }
}

fn vectors_in(
    store: &Store,
    table: &str,
    space_id: &str,
) -> BTreeMap<uuid::Uuid, (Value, Value, Value)> {
    let column = column_of(space_id);
    let raw = rows(
        store,
        &format!(
            "SELECT id, target_id, created_at, {column} FROM {table} WHERE embedding_space_id = $space ORDER BY target_id"
        ),
        vec![("space", Value::Text(space_id.to_string()))],
    );
    let distinct_targets = raw
        .iter()
        .map(|row| match row[1] {
            Value::Uuid(target) => target,
            _ => panic!("embedding target must be a UUID: {row:?}"),
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        raw.len(),
        distinct_targets.len(),
        "raw SQL contains exactly one row per item before collecting a snapshot"
    );
    let mut found = BTreeMap::new();
    let mut identities = std::collections::BTreeSet::new();
    for row in &raw {
        let (Value::Uuid(id), Value::Uuid(target)) = (&row[0], &row[1]) else {
            panic!("evidence embedding row {row:?}");
        };
        assert_eq!(
            id.get_version_num(),
            8,
            "embedding identity uses UUIDv8 format; snapshots and independent nodes check stability"
        );
        assert!(identities.insert(*id), "one physical row per embedding id");
        assert!(
            found
                .insert(*target, (row[2].clone(), row[3].clone(), row[0].clone()))
                .is_none(),
            "one physical embedding row per item and model, never collapsed by collection"
        );
    }
    assert_eq!(
        raw.len(),
        found.len(),
        "raw SQL cardinality equals item cardinality"
    );
    found
}

/// Assert the expected item exists before inspecting its vector: a missing production row is an
/// assertion failure at this proof's own boundary, never an unrelated map-index panic.
fn vector_row(
    snapshot: &BTreeMap<uuid::Uuid, (Value, Value, Value)>,
    target: uuid::Uuid,
) -> &(Value, Value, Value) {
    assert!(
        snapshot.contains_key(&target),
        "expected stored vector for item {target}; found {:?}",
        snapshot.keys().collect::<Vec<_>>()
    );
    snapshot.get(&target).expect("presence asserted above")
}

fn node_local(store: &Store, space_id: &str) -> BTreeMap<uuid::Uuid, (Value, Value, Value)> {
    vectors_in(store, "evidence_embeddings", space_id)
}

fn travelling(store: &Store, space_id: &str) -> BTreeMap<uuid::Uuid, (Value, Value, Value)> {
    vectors_in(store, "evidence_embeddings_travelling", space_id)
}

// ---------------------------------------------------------------------------------------------
// A crash mid-reindex resumes on reopen with an immediate rerun, repeating no committed item.
// ---------------------------------------------------------------------------------------------

#[test]
fn embedding_reindex_child() {
    let Some(path) = std::env::var_os(CHILD_PATH) else {
        return;
    };
    let path = PathBuf::from(path);
    let clock = std::env::var(CHILD_CLOCK)
        .expect("parent supplies the fixed crash clock")
        .parse::<u64>()
        .expect("fixed crash clock milliseconds");
    let _clock = contextdb_core::Wallclock::test_clock_guard(move || clock);
    let old = TextModel::new("rev-a");
    let new = TextModel::dying_on("rev-b", DIE_ON_CALL);
    let store = open_with(&path, &[&old]);
    store
        .switch_embedding_space(
            EmbeddingRole::EvidenceText,
            text_space("rev-b"),
            OldSpace::Reindex,
        )
        .expect("switch the text role to the new model");
    let _ = store.reindex_space_with_embedder(
        &text_space("rev-b").embedding_space_id,
        ReindexOptions::default(),
        new,
    );
    panic!("the reindex must reach the embedder's dying call");
}

#[test]
fn a_crash_mid_reindex_resumes_on_reopen_repeating_no_committed_item() {
    // Neither the parent nor the real child advances time to make the dead holder recoverable.
    let _clock = contextdb_core::Wallclock::test_clock_guard(|| RECOVERY_CLOCK);
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("crash-reindex.db");
    let old = TextModel::new("rev-a");
    let evidence = {
        let store = open_with(&path, &[&old]);
        let (_, evidence) = seed_items(&store);
        assert_eq!(
            active(&store, EmbeddingRole::EvidenceText),
            text_space("rev-a").embedding_space_id
        );
        evidence
    };

    let output = Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", "embedding_reindex_child", "--nocapture"])
        .env(CHILD_PATH, &path)
        .env(CHILD_CLOCK, RECOVERY_CLOCK.to_string())
        .output()
        .expect("run the reindexing child");
    assert_eq!(
        output.status.code(),
        Some(EXIT_MID_REINDEX),
        "the child died inside the second reindex batch: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let new = TextModel::new("rev-b");
    let store = open_with(&path, &[&old, &new]);
    let new_space = text_space("rev-b").embedding_space_id;
    let old_space = text_space("rev-a").embedding_space_id;
    assert_eq!(active(&store, EmbeddingRole::EvidenceText), new_space);
    let committed = node_local(&store, &new_space);
    assert_eq!(
        committed.len(),
        DIE_ON_CALL - 1,
        "exactly the first batch committed before the crash"
    );
    assert_eq!(
        status_of(&store, &new_space).work,
        BackgroundWorkState::Interrupted,
        "the killed reindex is reported interrupted, not running"
    );

    let waits = Arc::new(AtomicUsize::new(0));
    let observed_waits = Arc::clone(&waits);
    context_graph::test_support::observe_background_claim_waits_for_test(move |work, _| {
        if work == context_graph::BackgroundWork::Reindex {
            observed_waits.fetch_add(1, Ordering::SeqCst);
        }
    });
    let report = store
        .reindex_space(&new_space, ReindexOptions::default())
        .expect("the immediate rerun resumes the killed reindex");
    context_graph::test_support::observe_background_claim_waits_for_test(|_, _| {});
    assert_eq!(
        waits.load(Ordering::SeqCst),
        0,
        "dead-holder recovery never waits on a live claim"
    );
    assert_eq!(
        contextdb_core::Wallclock::now(),
        contextdb_core::Wallclock::from_raw_wire(RECOVERY_CLOCK),
        "resume completes while the clock remains at the child claim time; no lease expires"
    );
    assert_eq!(
        new.calls(),
        ITEMS - committed.len(),
        "only items whose batch never committed are embedded again"
    );
    assert_eq!(report.embedded as usize, ITEMS - committed.len());
    assert_eq!((report.skipped, report.remaining), (0, 0));
    let finished = node_local(&store, &new_space);
    assert!(
        evidence
            .iter()
            .all(|id| finished.contains_key(&id.as_uuid()))
    );
    assert_eq!(finished.len(), ITEMS);
    for (target, row) in &committed {
        assert_eq!(
            finished.get(target),
            Some(row),
            "a vector committed before the crash is kept, never rewritten"
        );
    }
    assert!(
        node_local(&store, &old_space).is_empty(),
        "each stale row was deleted in the batch that replaced it"
    );
    let status = status_of(&store, &new_space);
    assert_eq!(status.state, SpaceState::Active);
    assert_eq!((status.embedded as usize, status.unembedded), (ITEMS, 0));
    assert_eq!(status.work, BackgroundWorkState::Idle);
    assert_eq!(status_of(&store, &old_space).state, SpaceState::Stale);
}

// ---------------------------------------------------------------------------------------------
// A skipped item keeps its retained row and is counted.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_skipped_item_keeps_its_old_vector_and_is_counted_until_it_is_rebuilt() {
    // An item the new model refuses keeps its only vector, is counted, and is retried on request.
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("skipped.db");
    for builtin in [
        context_graph::embedding_space::default_text_embedding_space(),
        context_graph::embedding_space::event_history_embedding_space(),
    ] {
        assert_eq!(builtin.placement, Some(VectorPlacement::NodeLocal));
    }
    let old = TextModel::new("rev-a");
    let new = TextModel::new("rev-b");
    let store = open_with(&path, &[&old]);
    let (_, evidence) = seed_items(&store);
    let refused = evidence[7].as_uuid();
    let old_space = text_space("rev-a").embedding_space_id;
    let new_space = text_space("rev-b").embedding_space_id;
    let before_switch = node_local(&store, &old_space);
    assert!(
        before_switch.contains_key(&refused),
        "the item has a node-local vector under the old model before the switch: {:?}",
        before_switch.keys().collect::<Vec<_>>()
    );
    let old_vector = vector_row(&before_switch, refused).clone();

    store
        .switch_embedding_space(
            EmbeddingRole::EvidenceText,
            text_space("rev-b"),
            OldSpace::Reindex,
        )
        .expect("switch to the new model");
    drop(store);
    let store = open_with(&path, &[&old, &new]);
    new.refuse(Some(&note(7)));
    let report = store
        .reindex_space(&new_space, ReindexOptions::default())
        .expect("a refused item never aborts the reindex");
    assert_eq!((report.embedded as usize, report.skipped), (ITEMS - 1, 1));
    assert_eq!(report.remaining, 0);
    assert_eq!(
        node_local(&store, &old_space).get(&refused),
        Some(&old_vector),
        "the skipped item keeps its only vector"
    );
    assert_eq!(
        node_local(&store, &old_space).len(),
        1,
        "only the skipped item keeps an old row"
    );
    assert!(!node_local(&store, &new_space).contains_key(&refused));
    assert_eq!(
        rows(
            &store,
            "SELECT reason FROM embedding_skips WHERE target_id = $target AND embedding_space_id = $space",
            vec![
                ("target", Value::Uuid(refused)),
                ("space", Value::Text(new_space.clone())),
            ],
        ),
        vec![vec![Value::Text("embed_failed".into())]]
    );
    let status = status_of(&store, &new_space);
    assert_eq!((status.skipped, status.embedded as usize), (1, ITEMS - 1));

    new.refuse(None);
    let retried = store
        .reindex_space(&new_space, ReindexOptions::default())
        .expect("an explicit reindex retries the skip");
    assert_eq!((retried.embedded, retried.skipped), (1, 0));
    assert!(node_local(&store, &new_space).contains_key(&refused));
    assert!(
        node_local(&store, &old_space).is_empty(),
        "the old row goes with its replacement"
    );
    assert_eq!(status_of(&store, &new_space).skipped, 0);
}

// ---------------------------------------------------------------------------------------------
// A pin bump on open creates a new space and rebuilds; an older binary gets EmbedderSuperseded.
// After the bump every default-space caller uses the new active space.
// ---------------------------------------------------------------------------------------------

fn lexical_space(revision: &str) -> String {
    format!(
        "text_lexical_mock_768_{}",
        revision
            .bytes()
            .map(|byte| if byte.is_ascii_alphanumeric() {
                byte as char
            } else {
                '_'
            })
            .collect::<String>()
    )
}

#[test]
fn a_pin_bump_on_open_creates_a_new_space_and_an_older_binary_is_told_to_upgrade() {
    // A new weights revision is a new space that rebuilds; reopening with the old one is refused.
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("pin-bump.db");
    let evidence = {
        let store = Store::open(configured(&path, "rev-a")).expect("open under rev-a");
        let (_, evidence) = seed_items(&store);
        assert_eq!(
            active(&store, EmbeddingRole::EvidenceText),
            lexical_space("rev-a")
        );
        assert_eq!(node_local(&store, &lexical_space("rev-a")).len(), ITEMS);
        evidence
    };

    let store = Store::open(configured(&path, "rev-b")).expect("a pin bump opens");
    let new_space = lexical_space("rev-b");
    assert_eq!(active(&store, EmbeddingRole::EvidenceText), new_space);
    assert_eq!(
        status_of(&store, &lexical_space("rev-a")).state,
        SpaceState::Stale
    );
    assert_eq!(status_of(&store, &new_space).unembedded as usize, ITEMS);
    let report = store
        .reindex_space(&new_space, ReindexOptions::default())
        .expect("rebuild into the new space");
    assert_eq!(report.embedded as usize, ITEMS);
    let rebuilt = node_local(&store, &new_space);
    assert!(
        evidence
            .iter()
            .all(|id| rebuilt.contains_key(&id.as_uuid()))
    );
    drop(store);

    let refused =
        Store::open(configured(&path, "rev-a")).expect_err("an older binary never switches back");
    match &refused {
        CgError::EmbedderSuperseded {
            role,
            active_space,
            active,
            registered,
        } => {
            assert_eq!(*role, EmbeddingRole::EvidenceText);
            assert_eq!(active_space, &new_space);
            assert_eq!(active.weights_revision, "rev-b");
            assert_eq!(registered.weights_revision, "rev-a");
        }
        other => panic!("expected EmbedderSuperseded, got {other:?}"),
    }
    assert!(refused.to_string().contains("upgrade"), "{refused}");
    let still = Store::open(configured(&path, "rev-b")).expect("the current binary still opens");
    assert_eq!(
        active(&still, EmbeddingRole::EvidenceText),
        new_space,
        "the refused older binary switched nothing back"
    );
}

#[test]
fn after_a_pin_bump_every_default_space_caller_uses_the_new_active_space() {
    // New evidence, new digests and digest search all use the space the bumped pin made active.
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("default-callers.db");
    let context_id = {
        let store = Store::open(configured(&path, "rev-a")).expect("open under rev-a");
        seed_items(&store).0
    };
    let store = Store::open(configured(&path, "rev-b")).expect("pin bump");
    let new_space = lexical_space("rev-b");
    let entity_id = create_entity(&store, context_id, "fresh notebook");
    let fresh = record_text(&store, context_id, entity_id, "written after the bump");
    assert_eq!(
        rows(
            &store,
            "SELECT embedding_space_id FROM evidence_embeddings WHERE target_id = $target",
            vec![("target", Value::Uuid(fresh.as_uuid()))],
        ),
        vec![vec![Value::Text(new_space.clone())]],
        "a write after the bump embeds into the new active space only"
    );

    let digest_space = active(&store, EmbeddingRole::Digest);
    assert_eq!(
        digest_space, new_space,
        "the digest role moved with the configured model"
    );
    let report = context_graph::tier0::capture_tier0(
        &store,
        context_graph::tier0::Tier0CaptureRequest {
            context_label: "placement".into(),
            session_ref: "after-bump".into(),
            transcript: "User: we chose rust for the engine core".into(),
            source: "test".into(),
            chunk_window: 0,
        },
    )
    .expect("capture a digest after the bump");
    assert_eq!(report.digests_written, 1);
    assert_eq!(
        rows(
            &store,
            "SELECT embedding_space_id FROM digest_embeddings WHERE embedding_space_id <> $space",
            vec![("space", Value::Text(new_space.clone()))],
        ),
        Vec::<Vec<Value>>::new(),
        "the new digest is embedded in the new active space only"
    );
    let found = store
        .find_via_digests(
            "rust engine core",
            DigestSearchFilter {
                context_id: None,
                cross_context: None,
                min_similarity: None,
                limit: Some(5),
                created_after: None,
                created_before: None,
                include_inactive: false,
            },
        )
        .expect("digest search queries the new active space");
    assert!(
        !found.hits.is_empty(),
        "the new digest is found in the new space"
    );
}

// ---------------------------------------------------------------------------------------------
// Switching: an explicit Reindex switch, a retained (parallel) space, and a travelling switch.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_retained_space_stays_searchable_and_counted_until_it_is_released() {
    // Switching with the old space retained keeps every old vector until the space is released.
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("retained.db");
    let old = TextModel::new("rev-a");
    let new = TextModel::new("rev-b");
    let store = open_with(&path, &[&old]);
    let (_, evidence) = seed_items(&store);
    let old_space = text_space("rev-a").embedding_space_id;
    let new_space = text_space("rev-b").embedding_space_id;
    let before = node_local(&store, &old_space);

    let switched = store
        .switch_embedding_space(
            EmbeddingRole::EvidenceText,
            text_space("rev-b"),
            OldSpace::Retain,
        )
        .expect("switch keeping the old space");
    drop(store);
    let store = open_with(&path, &[&old, &new]);
    assert_eq!(switched.role, Some(EmbeddingRole::EvidenceText));
    assert_eq!(
        (switched.from_space.as_str(), switched.to_space.as_str()),
        (old_space.as_str(), new_space.as_str())
    );
    assert_eq!(switched.pending as usize, ITEMS);
    assert_eq!(status_of(&store, &old_space).state, SpaceState::Retained);
    assert_eq!(
        status_of(&store, &new_space).unembedded as usize,
        ITEMS,
        "items served only by the retained space count as unembedded in the role"
    );
    store
        .reindex_space(&new_space, ReindexOptions::default())
        .expect("fill the new space");
    assert_eq!(
        node_local(&store, &old_space),
        before,
        "a retained space is never stale: its rows survive the rebuild"
    );
    assert_eq!(node_local(&store, &new_space).len(), ITEMS);
    #[cfg(context_graph_internal_test_support)]
    {
        let retained = context_graph::test_support::embedding::vector_search(
            &store,
            &text_vector(&format!("rev-a:{}", note(0))),
            &old_space,
            context_graph::test_support::embedding::VectorSearchFilters {
                top_k: Some(1),
                ..Default::default()
            },
        )
        .expect("search the retained space's actual local index");
        assert_eq!(retained.hits.len(), 1);
        assert_eq!(
            retained.hits[0].target,
            EmbeddingTarget::Evidence(evidence[0])
        );
        assert_eq!(retained.hits[0].embedding_space_id, old_space);
        assert_eq!(retained.coverage.preserved_elsewhere, 0);
    }
    assert_eq!(
        new.calls(),
        ITEMS,
        "the new model actually generated distinct values"
    );
    assert_ne!(
        vector_row(&node_local(&store, &new_space), evidence[0].as_uuid()).1,
        vector_row(&before, evidence[0].as_uuid()).1,
    );

    let released = store
        .release_retained_space(EmbeddingRole::EvidenceText, &old_space)
        .expect("release the retained space");
    assert_eq!(released.from_space, old_space);
    assert_eq!(status_of(&store, &old_space).state, SpaceState::Stale);
    assert!(
        evidence
            .iter()
            .all(|id| node_local(&store, &new_space).contains_key(&id.as_uuid()))
    );
}

/// A caller-filled text space that travels with its item; succession is declared by the switch.
fn caller_space(revision: &str) -> EmbeddingSpace {
    let id = format!("placement_caller_{TEXT_DIMENSION}_{revision}");
    EmbeddingSpace {
        embedding_space_id: id.clone(),
        dimension: TEXT_DIMENSION,
        metric: DistanceMetric::Cosine,
        model_family: "caller".into(),
        model_name: "placement-caller".into(),
        model_version: revision.into(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: None,
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: id,
            table_name: "evidence".into(),
            vector_column: TEXT_COLUMN.into(),
        }],
        placement: Some(VectorPlacement::TravelsWithItem),
        media_reachable_on_every_node: false,
        producer: caller_identity(revision),
    }
}

fn caller_identity(revision: &str) -> ModelIdentity {
    ModelIdentity {
        model: "placement-caller".into(),
        weights_revision: revision.into(),
        preparation: "caller".into(),
        dimension: TEXT_DIMENSION as u32,
    }
}

#[tokio::test]
async fn a_travelling_switch_preserves_and_reports_what_this_node_cannot_regenerate() {
    // A caller-filled space switched to a new model keeps every old vector and reports them.
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("travelling-switch.db");
    let old = TextModel::new("rev-a");
    let store = open_with(&path, &[&old]);
    store
        .register_embedding_space(caller_space("v1"))
        .expect("register the caller space");
    let (_, evidence) = seed_items(&store);
    let v1 = caller_space("v1").embedding_space_id;
    for (index, id) in evidence.iter().enumerate() {
        store
            .supply_embedding(
                EmbeddingTarget::Evidence(*id),
                &v1,
                caller_identity("v1"),
                text_vector(&format!("caller {index}")),
            )
            .expect("supply a caller vector");
    }
    let before = travelling(&store, &v1);
    assert_eq!(before.len(), ITEMS);

    let v2 = caller_space("v2");
    let switched = store
        .switch_travelling_space(&v1, v2.clone())
        .expect("switch the caller space to the new model");
    assert_eq!(switched.role, None);
    assert_eq!(
        (
            switched.pending,
            switched.preserved_elsewhere as usize,
            switched.skipped
        ),
        (0, ITEMS, 0),
        "nothing here can regenerate caller vectors, so every item is preserved and reported"
    );
    assert_eq!(
        travelling(&store, &v1),
        before,
        "old vectors are never deleted or mixed"
    );
    let status = status_of(&store, &v2.embedding_space_id);
    assert_eq!(status.state, SpaceState::Travelling);
    assert_eq!(
        status.predecessors,
        vec![v1.clone()],
        "the switch records its succession"
    );
    assert_eq!(status.preserved_elsewhere as usize, ITEMS);
    #[cfg(context_graph_internal_test_support)]
    {
        let uncovered = context_graph::test_support::embedding::vector_search(
            &store,
            &text_vector("query"),
            &v2.embedding_space_id,
            Default::default(),
        )
        .expect("ordinary evidence search reports preserved-only items");
        assert!(uncovered.hits.is_empty());
        assert_eq!(uncovered.coverage.preserved_elsewhere as usize, ITEMS);
        assert_eq!(uncovered.coverage.unembedded, 0);
        let old_hits = context_graph::test_support::embedding::vector_search(
            &store,
            &text_vector("caller 0"),
            &v1,
            context_graph::test_support::embedding::VectorSearchFilters {
                top_k: Some(1),
                ..Default::default()
            },
        )
        .expect("preserved evidence remains searchable");
        assert_eq!(old_hits.hits.len(), 1);
        assert_eq!(
            old_hits.hits[0].target,
            EmbeddingTarget::Evidence(evidence[0])
        );
    }

    store
        .supply_embedding(
            EmbeddingTarget::Evidence(evidence[0]),
            &v2.embedding_space_id,
            caller_identity("v2"),
            text_vector("caller zero, new model"),
        )
        .expect("the caller re-supplies one item in the new space");
    assert_eq!(
        status_of(&store, &v2.embedding_space_id).preserved_elsewhere as usize,
        ITEMS - 1,
        "a re-supplied item is no longer served only from the old space"
    );
    #[cfg(context_graph_internal_test_support)]
    {
        let new_hits = context_graph::test_support::embedding::vector_search(
            &store,
            &text_vector("caller zero, new model"),
            &v2.embedding_space_id,
            Default::default(),
        )
        .expect("ordinary evidence search after supplying one successor");
        assert_eq!(new_hits.hits.len(), 1);
        assert_eq!(
            new_hits.hits[0].target,
            EmbeddingTarget::Evidence(evidence[0])
        );
        assert_eq!(new_hits.coverage.preserved_elsewhere as usize, ITEMS - 1);
    }
    let preserved = travelling(&store, &v1);
    store
        .reindex_space(&v2.embedding_space_id, ReindexOptions::default())
        .expect("driving a travelling space keeps irregenerable values");
    assert_eq!(travelling(&store, &v1), preserved);
    let refused = store
        .supply_embedding(
            EmbeddingTarget::Evidence(evidence[1]),
            &v2.embedding_space_id,
            caller_identity("v1"),
            text_vector("wrong model"),
        )
        .expect_err("a space holds exactly one producer");
    assert!(
        matches!(refused, CgError::EmbeddingProducerMismatch { .. }),
        "{refused:?}"
    );
    let cycle = store
        .switch_travelling_space(&v2.embedding_space_id, caller_space("v1"))
        .expect_err("succession never closes a cycle");
    assert!(
        matches!(&cycle, CgError::EmbeddingSuccessionCycle { from, to } if *from == v2.embedding_space_id && *to == v1),
        "{cycle:?}"
    );
    // cg-derived media values: the old model read real bytes, then the source really disappeared.
    let media_path = root.path().join("source-image.bin");
    std::fs::write(&media_path, [2_u8, 7, 9]).expect("retained source bytes for the first model");
    let media_old = Arc::new(MediaModel {
        revision: "media-old",
        calls: AtomicUsize::new(0),
    });
    let media_new = Arc::new(MediaModel {
        revision: "media-new",
        calls: AtomicUsize::new(0),
    });
    let media_id = EvidenceId::new_v7();
    let mut media_switch = None;
    let mut media_before_switch = None;
    let media_fixture = fabric_common::fabric_fixture_with_node_openers(
        "placement_missing_media",
        "tenant-alpha",
        "tenant-alpha",
        |label| {
            let first_generation = fabric_common::FabricNode::with_models(
                label,
                |path| StoreConfig {
                    file_uri_allowed: true,
                    ..plain_config(&path)
                },
                vec![EmbedderRegistration {
                    space: media_old.embedding_space(),
                    embedder: media_old.clone(),
                }],
            );
            let media_context = create_context(&first_generation.store);
            let media_entity = create_entity(&first_generation.store, media_context, "camera");
            let mut observation = supplied_note(
                media_context,
                media_entity,
                media_id,
                &v1,
                Vec::new(),
                &caller_identity("v1"),
            );
            observation.embeddings.clear();
            observation.evidence[0].kind = EvidenceKind::ImageFrame;
            observation.evidence[0].text_projection = None;
            observation.evidence[0].source_ref = format!("file://{}", media_path.display());
            observation.evidence[0].retention_status = RetentionStatus::RetainedExternal;
            first_generation
                .store
                .record_observation(observation)
                .expect("cg derives the first vector from actual media bytes");
            media_before_switch = Some(travelling(
                &first_generation.store,
                &media_old.embedding_space().embedding_space_id,
            ));
            std::fs::remove_file(&media_path).expect("media is no longer retained");
            media_switch = Some(
                first_generation
                    .store
                    .switch_travelling_space(
                        &media_old.embedding_space().embedding_space_id,
                        media_new.embedding_space(),
                    )
                    .expect("declare the real successor while source is unavailable"),
            );
            // Both declarations claim ImageFrame only after the production switch has made
            // them generations of the same space. Reopening resolves the successor default.
            first_generation.reopen_with_models(
                |path| StoreConfig {
                    file_uri_allowed: true,
                    ..plain_config(&path)
                },
                vec![
                    EmbedderRegistration {
                        space: media_old.embedding_space(),
                        embedder: media_old.clone(),
                    },
                    EmbedderRegistration {
                        space: media_new.embedding_space(),
                        embedder: media_new.clone(),
                    },
                ],
            )
        },
        |label| {
            // This receiver starts with one generation. The successor declaration and its
            // relationship arrive together through the real production sync path.
            fabric_common::FabricNode::with_models(
                label,
                |path| StoreConfig {
                    extra_embedding_spaces: vec![media_old.embedding_space()],
                    file_uri_allowed: true,
                    ..plain_config(&path)
                },
                Vec::new(),
            )
        },
    )
    .await;
    let media_store = &media_fixture.edge1.store;
    let media_switch = media_switch.expect("source durably declared the actual succession");
    assert_eq!(media_switch.preserved_elsewhere, 1);
    assert_eq!(
        active(&media_store, EmbeddingRole::EvidenceImageFrame),
        media_new.embedding_space().embedding_space_id
    );
    assert_eq!(media_old.calls.load(Ordering::SeqCst), 1);
    let media_before = travelling(
        &media_store,
        &media_old.embedding_space().embedding_space_id,
    );
    assert_eq!(media_before.len(), 1);
    assert_eq!(
        media_before,
        media_before_switch.expect("pre-switch media vector identity")
    );
    let mut expected = axis(2);
    expected[3] = 0.25;
    assert_eq!(
        vector_row(&media_before, media_id.as_uuid()).1,
        Value::Vector(expected.clone())
    );
    assert!(!media_path.exists());
    fabric_common::push_then_pull(&media_fixture.edge1_client, &media_fixture.edge2_client).await;
    assert_eq!(
        travelling(
            &media_fixture.edge2.store,
            &media_old.embedding_space().embedding_space_id
        ),
        media_before,
        "cg-derived media vector, identity and provenance travel after its actual source is gone"
    );
    assert_eq!(
        count_where(
            &media_fixture.edge2.store,
            "evidence",
            "id",
            media_id.as_uuid()
        ),
        1
    );

    for store in [&media_store, &media_fixture.edge2.store] {
        assert_eq!(
            rows(
                store,
                "SELECT embedding_space_id, default_for_evidence_kind FROM embedding_spaces WHERE embedding_space_id = $old OR embedding_space_id = $new ORDER BY embedding_space_id",
                vec![
                    (
                        "old",
                        Value::Text(media_old.embedding_space().embedding_space_id)
                    ),
                    (
                        "new",
                        Value::Text(media_new.embedding_space().embedding_space_id)
                    ),
                ],
            ),
            vec![
                vec![
                    Value::Text(media_new.embedding_space().embedding_space_id),
                    Value::Text("image_frame".into())
                ],
                vec![
                    Value::Text(media_old.embedding_space().embedding_space_id),
                    Value::Text("image_frame".into())
                ],
            ],
            "both generations retain their declared default after real succession sync"
        );
    }
    let source_succession = rows(
        &media_store,
        "SELECT from_space, to_space, declared_at FROM embedding_space_successions WHERE from_space = $old",
        vec![(
            "old",
            Value::Text(media_old.embedding_space().embedding_space_id),
        )],
    );
    assert_eq!(source_succession.len(), 1);
    assert_eq!(
        source_succession[0][1],
        Value::Text(media_new.embedding_space().embedding_space_id)
    );
    assert_eq!(
        rows(
            &media_fixture.edge2.store,
            "SELECT from_space, to_space, declared_at FROM embedding_space_successions WHERE from_space = $old",
            vec![(
                "old",
                Value::Text(media_old.embedding_space().embedding_space_id)
            )],
        ),
        source_succession,
        "the receiver acquired the exact source-declared succession through sync"
    );
    let media_report = media_store
        .reindex_space(
            &media_new.embedding_space().embedding_space_id,
            ReindexOptions::default(),
        )
        .expect("unavailable media is a counted skip, never aborts preservation");
    assert_eq!(
        (
            media_report.embedded,
            media_report.skipped,
            media_report.remaining
        ),
        (0, 1, 0)
    );
    assert_eq!(
        media_new.calls.load(Ordering::SeqCst),
        1,
        "successor really encountered the unavailable file"
    );
    assert_eq!(
        travelling(
            &media_store,
            &media_old.embedding_space().embedding_space_id
        ),
        media_before
    );
    let media_status = status_of(
        &media_store,
        &media_new.embedding_space().embedding_space_id,
    );
    assert_eq!(
        (
            media_status.state,
            media_status.preserved_elsewhere,
            media_status.skipped
        ),
        (SpaceState::Travelling, 1, 1)
    );
    #[cfg(context_graph_internal_test_support)]
    {
        let found = context_graph::test_support::embedding::vector_search(
            &media_store,
            &expected,
            &media_new.embedding_space().embedding_space_id,
            Default::default(),
        )
        .expect("missing-media successor search coverage");
        assert!(found.hits.is_empty());
        assert_eq!(found.coverage.preserved_elsewhere, 1);
        let old = context_graph::test_support::embedding::vector_search(
            &media_store,
            &expected,
            &media_old.embedding_space().embedding_space_id,
            Default::default(),
        )
        .expect("preserved media remains comparable");
        assert_eq!(old.hits.len(), 1);
        assert_eq!(old.hits[0].target, EmbeddingTarget::Evidence(media_id));
    }
}

/// One text note whose vector the caller supplies under `producer`.
fn supplied_note(
    context_id: ContextId,
    entity_id: EntityId,
    evidence_id: EvidenceId,
    space: &str,
    vector: Vec<f32>,
    producer: &ModelIdentity,
) -> RecordObservation {
    RecordObservation {
        id: ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "note".into(),
        source: "placement-fixture".into(),
        observed_at: chrono::DateTime::from_timestamp(1_790_000_000, 0).expect("fixed time"),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: capture_producer(),
            text_projection: Some("a supplied note".into()),
            ..EvidenceRef::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![context_graph::PrecomputedEmbedding {
            id: context_graph::EmbeddingId::new_v7(),
            embedding_space_id: space.to_string(),
            vector,
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: EvidenceProducer {
                system: "placement-fixture".into(),
                ..EvidenceProducer::from(producer)
            },
        }],
    }
}

#[test]
fn a_foreign_vector_supplied_into_a_node_local_space_is_refused_and_a_compatible_one_is_stored_as_given()
 {
    // All four write doors validate one identity component at a time and atomically refuse it.
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("supplied-vectors.db");
    let ambiguous_path = root.path().join("unrelated-defaults.db");
    let ambiguous = Store::open_with_embedder_registrations(
        plain_config(&ambiguous_path),
        vec![
            TextModel::new("unrelated-a").registration(),
            TextModel::new("unrelated-b").registration(),
        ],
    )
    .expect_err("two unrelated defaults are refused on a fresh root");
    assert!(
        matches!(ambiguous, CgError::AmbiguousEmbeddingSpace),
        "{ambiguous:?}"
    );
    assert!(
        !ambiguous_path.exists(),
        "ambiguity is refused before opening or writing the database"
    );

    let declared = text_space("rev-a");
    let space = declared.embedding_space_id.clone();
    let mut local_caller = caller_space("local-only");
    local_caller.placement = Some(VectorPlacement::NodeLocal);
    let store = Store::open(StoreConfig {
        extra_embedding_spaces: vec![declared, local_caller.clone(), face_space("v1")],
        ..plain_config(&path)
    })
    .expect("declare reproducible and caller-only local spaces without loaded models");
    let context_id = create_context(&store);
    let entity_id = create_entity(&store, context_id, "notebook");
    let unavailable = status_of(&store, &space);
    assert_eq!(unavailable.role, Some(EmbeddingRole::EvidenceText));
    assert_eq!(unavailable.placement, VectorPlacement::NodeLocal);
    assert_eq!(unavailable.state, SpaceState::Unavailable);
    let held = status_of(&store, &local_caller.embedding_space_id);
    assert_eq!(
        (held.role, held.placement, held.state),
        (None, VectorPlacement::NodeLocal, SpaceState::HeldOnlyHere)
    );
    assert_eq!(
        status_of(&store, &face_space("v1").embedding_space_id).state,
        SpaceState::Travelling
    );
    let compatible = EvidenceId::new_v7();
    let supplied_vector = text_vector("kept exactly as supplied");
    let recorded = store
        .record_observation(supplied_note(
            context_id,
            entity_id,
            compatible,
            &space,
            supplied_vector.clone(),
            &text_identity("rev-a"),
        ))
        .expect("compatible observation supply is stored exactly");
    assert_eq!(recorded.embedding_ids.len(), 1);
    assert!(matches!(recorded.supplied_vectors.as_slice(),
        [context_graph::SuppliedVectorOutcome {
            outcome: context_graph::SuppliedVector::Stored { embedding_id }, ..
        }] if *embedding_id == recorded.embedding_ids[0]));
    assert_eq!(
        vector_row(&node_local(&store, &space), compatible.as_uuid()).1,
        Value::Vector(supplied_vector)
    );
    let compatible_id = vector_row(&node_local(&store, &space), compatible.as_uuid())
        .2
        .clone();
    assert_eq!(
        compatible_id,
        Value::Uuid(recorded.embedding_ids[0].as_uuid())
    );
    let digest = context_graph::DigestId::new_v7();
    let digest_vector = text_vector("supplied digest");
    let digest_input =
        |id, producer: &ModelIdentity, vector: Vec<f32>| context_graph::CreateDigest {
            id,
            context_id,
            source: "fixture".into(),
            summary: "digest summary".into(),
            kind: EvidenceKind::Text,
            embedding: context_graph::PrecomputedEmbedding {
                id: context_graph::EmbeddingId::new_v7(),
                embedding_space_id: space.clone(),
                target: EmbeddingTarget::Digest(id),
                vector,
                produced_by: EvidenceProducer::from(producer),
            },
            ..Default::default()
        };
    let created = store
        .create_digest(digest_input(
            digest,
            &text_identity("rev-a"),
            digest_vector.clone(),
        ))
        .expect("compatible digest supply is stored exactly");
    assert!(matches!(
        created.supplied_vector,
        Some(context_graph::SuppliedVector::Stored { .. })
    ));
    assert_eq!(
        rows(
            &store,
            &format!("SELECT {TEXT_COLUMN} FROM digest_embeddings WHERE target_id = $id"),
            vec![("id", Value::Uuid(digest.as_uuid()))]
        ),
        vec![vec![Value::Vector(digest_vector)]]
    );
    let face = face_space("v1").embedding_space_id;
    let enrolled = store
        .enroll_entity_reference(
            entity_id,
            &face,
            Some("compatible".into()),
            axis(0),
            face_producer("v1"),
        )
        .expect("compatible enrollment is stored exactly");
    assert_eq!(
        enrollment_rows(&store, entity_id),
        vec![(
            Value::Text("compatible".into()),
            face.clone(),
            Value::Vector(axis(0))
        )]
    );
    assert_eq!(
        rows(
            &store,
            &format!(
                "SELECT id FROM entity_reference_embeddings_{FACE_DIMENSION} WHERE entity_id = $id"
            ),
            vec![("id", Value::Uuid(entity_id.as_uuid()))]
        ),
        vec![vec![Value::Uuid(enrolled.as_uuid())]]
    );
    let direct_vector = text_vector("compatible direct replacement");
    let direct = store
        .supply_embedding(
            EmbeddingTarget::Evidence(compatible),
            &space,
            text_identity("rev-a"),
            direct_vector.clone(),
        )
        .expect("compatible direct supply replaces exactly");
    assert!(
        matches!(direct, context_graph::SuppliedVector::Stored { embedding_id } if embedding_id == recorded.embedding_ids[0])
    );
    assert_eq!(
        vector_row(&node_local(&store, &space), compatible.as_uuid()).1,
        Value::Vector(direct_vector)
    );

    let related_tables = [
        "observations",
        "evidence",
        "entity_snapshots",
        "entity_snapshot_observations",
        "audit_log",
        "entity_current_snapshots",
        "entity_snapshot_links",
        "digests",
        "evidence_embeddings",
        "evidence_embeddings_travelling",
        "digest_embeddings",
        "embedding_pending",
        "embedding_skips",
    ];
    for component in ["model", "revision", "preparation", "dimension"] {
        let mut incompatible = text_identity("rev-a");
        match component {
            "model" => incompatible.model = "other-family-model".into(),
            "revision" => incompatible.weights_revision = "rev-foreign".into(),
            "preparation" => incompatible.preparation = "different-preprocessing".into(),
            "dimension" => incompatible.dimension += 1,
            _ => unreachable!(),
        }
        let vector = vec![0.25; incompatible.dimension as usize];
        let before: Vec<_> = related_tables
            .iter()
            .map(|table| rows(&store, &format!("SELECT * FROM {table}"), Vec::new()))
            .collect();
        let foreign = EvidenceId::new_v7();
        let mut observation = supplied_note(
            context_id,
            entity_id,
            foreign,
            &space,
            vector.clone(),
            &incompatible,
        );
        observation.state_delta.insert(
            "marker".into(),
            Some(serde_json::Value::String(component.into())),
        );
        let observation_id = observation.id;
        let error = store
            .record_observation(observation)
            .expect_err("incompatible observation supply refused");
        if component == "dimension" {
            assert!(
                matches!(error, CgError::EmbeddingDimensionMismatch { expected, found, .. }
                if expected == TEXT_DIMENSION as u32 && found == TEXT_DIMENSION as u32 + 1),
                "{error:?}"
            );
        } else {
            assert!(
                matches!(error, CgError::EmbeddingProducerMismatch { ref space, ref expected, ref supplied } if space == &text_space("rev-a").embedding_space_id && expected == &text_identity("rev-a") && supplied == &incompatible),
                "{error:?}"
            );
        }
        assert_eq!(
            count_where(&store, "observations", "id", observation_id.as_uuid()),
            0
        );
        assert_eq!(count_where(&store, "evidence", "id", foreign.as_uuid()), 0);
        let direct_error = store
            .supply_embedding(
                EmbeddingTarget::Evidence(compatible),
                &space,
                incompatible.clone(),
                vector.clone(),
            )
            .expect_err("incompatible direct supply refused");
        let digest_error = store
            .create_digest(digest_input(
                context_graph::DigestId::new_v7(),
                &incompatible,
                vector,
            ))
            .expect_err("incompatible digest supply refused");
        for error in [direct_error, digest_error] {
            if component == "dimension" {
                assert!(
                    matches!(error, CgError::EmbeddingDimensionMismatch { expected, found, .. }
                    if expected == TEXT_DIMENSION as u32 && found == TEXT_DIMENSION as u32 + 1),
                    "{error:?}"
                );
            } else {
                assert!(
                    matches!(error, CgError::EmbeddingProducerMismatch { ref space, ref expected, ref supplied } if space == &text_space("rev-a").embedding_space_id && expected == &text_identity("rev-a") && supplied == &incompatible),
                    "{error:?}"
                );
            }
        }
        let mut identity = face_identity("v1");
        match component {
            "model" => identity.model = "other-family-model".into(),
            "revision" => identity.weights_revision = "v-foreign".into(),
            "preparation" => identity.preparation = "different-preprocessing".into(),
            "dimension" => identity.dimension += 1,
            _ => unreachable!(),
        }
        let enrollment_error = store
            .enroll_entity_reference(
                entity_id,
                &face,
                Some(component.into()),
                vec![0.25; identity.dimension as usize],
                EvidenceProducer::from(&identity),
            )
            .expect_err("incompatible enrollment supply refused");
        if component == "dimension" {
            assert!(
                matches!(enrollment_error, CgError::EmbeddingDimensionMismatch { expected, found, .. }
                if expected == FACE_DIMENSION as u32 && found == FACE_DIMENSION as u32 + 1),
                "{enrollment_error:?}"
            );
        } else {
            assert!(
                matches!(enrollment_error, CgError::EmbeddingProducerMismatch { ref space, ref expected, ref supplied } if space == &face && expected == &face_identity("v1") && supplied == &identity),
                "{enrollment_error:?}"
            );
        }
        assert_eq!(
            enrollment_rows(&store, entity_id),
            vec![(
                Value::Text("compatible".into()),
                face.clone(),
                Value::Vector(axis(0))
            )]
        );
        for (table, before) in related_tables.iter().zip(before) {
            assert_eq!(
                rows(&store, &format!("SELECT * FROM {table}"), Vec::new()),
                before,
                "{component} refusal changes no related write in {table}"
            );
        }
    }
    // Producer system is provenance only: changing it does not change identity compatibility.
    let mut provenance = supplied_note(
        context_id,
        entity_id,
        EvidenceId::new_v7(),
        &space,
        text_vector("different system"),
        &text_identity("rev-a"),
    );
    provenance.embeddings[0].produced_by.system = "another-system".into();
    store
        .record_observation(provenance)
        .expect("system provenance does not cause model mismatch");
    let held_item = EvidenceId::new_v7();
    store
        .record_observation(supplied_note(
            context_id,
            entity_id,
            held_item,
            &local_caller.embedding_space_id,
            text_vector("held on this receiving node"),
            &caller_identity("local-only"),
        ))
        .expect("a caller-only space may explicitly be local");
    let held = status_of(&store, &local_caller.embedding_space_id);
    assert_eq!(
        (held.state, held.role, held.embedded, held.unembedded),
        (SpaceState::HeldOnlyHere, None, 1, 0)
    );
    assert!(travelling(&store, &local_caller.embedding_space_id).is_empty());
    let held_before = node_local(&store, &local_caller.embedding_space_id);
    drop(store);
    let model = TextModel::new("rev-a");
    let store = open_with(&path, &[&model]);
    assert_eq!(status_of(&store, &space).state, SpaceState::Active);
    store
        .reindex_space(&space, ReindexOptions::default())
        .expect("same-space repair computes local values");
    assert_eq!(
        vector_row(&node_local(&store, &space), compatible.as_uuid()).1,
        Value::Vector(text_vector("rev-a:a supplied note"))
    );
    assert_eq!(
        vector_row(&node_local(&store, &space), compatible.as_uuid()).2,
        compatible_id,
        "same-node rebuild preserves the actual item/model identity despite changing values"
    );
    assert_eq!(
        rows(
            &store,
            &format!("SELECT {TEXT_COLUMN} FROM digest_embeddings WHERE target_id = $id"),
            vec![("id", Value::Uuid(digest.as_uuid()))]
        ),
        vec![vec![Value::Vector(text_vector("rev-a:digest summary"))]]
    );
    assert_eq!(
        model.calls(),
        4,
        "three evidence projections and one digest summary were rebuilt locally"
    );
    assert_eq!(
        node_local(&store, &local_caller.embedding_space_id),
        held_before,
        "repairing the reproducible space never rewrites caller-only local values"
    );
    let held_after_repair = status_of(&store, &local_caller.embedding_space_id);
    assert_eq!(
        (held_after_repair.role, held_after_repair.state),
        (None, SpaceState::HeldOnlyHere),
        "an extra caller-only space stays unassigned after reopen and another space's repair"
    );
}

// ---------------------------------------------------------------------------------------------
// Enrollments: identity, multi-shot, replacement and preservation across models.
// ---------------------------------------------------------------------------------------------

/// A caller-filled face space at one model revision; it travels with its item.
fn face_space(revision: &str) -> EmbeddingSpace {
    let id = format!("placement_face_{revision}_{FACE_DIMENSION}");
    EmbeddingSpace {
        embedding_space_id: id.clone(),
        dimension: FACE_DIMENSION,
        metric: DistanceMetric::Cosine,
        model_family: "vision".into(),
        model_name: "placement-face".into(),
        model_version: revision.into(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: None,
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: id,
            table_name: "evidence".into(),
            vector_column: FACE_COLUMN.into(),
        }],
        placement: Some(VectorPlacement::TravelsWithItem),
        media_reachable_on_every_node: false,
        producer: face_identity(revision),
    }
}

fn face_identity(revision: &str) -> ModelIdentity {
    ModelIdentity {
        model: "placement-face".into(),
        weights_revision: revision.into(),
        preparation: "face".into(),
        dimension: FACE_DIMENSION as u32,
    }
}

/// An enrollment's producer is the face space's own model.
fn face_producer(revision: &str) -> EvidenceProducer {
    EvidenceProducer::from(&face_identity(revision))
}

fn axis(index: usize) -> Vec<f32> {
    let mut vector = vec![0.0; FACE_DIMENSION];
    vector[index % FACE_DIMENSION] = 1.0;
    vector
}

struct Enrollments {
    _root: tempfile::TempDir,
    store: Store,
    context_id: ContextId,
    v1: String,
    v2: String,
}

/// A store with face model v1 and its successor v2, switched through the travelling switch.
fn enrollments() -> Enrollments {
    let root = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: root.path().join("enrollments.db"),
        default_text_embedder: None,
        extra_embedding_spaces: vec![face_space("v1")],
        ..StoreConfig::default()
    })
    .expect("open enrollment store");
    let context_id = create_context(&store);
    let v1 = face_space("v1").embedding_space_id;
    let v2 = face_space("v2");
    store
        .switch_travelling_space(&v1, v2.clone())
        .expect("register the successor face model");
    assert_eq!(
        status_of(&store, &v1).role,
        None,
        "the old extra face space was never assigned a default role"
    );
    assert_eq!(
        status_of(&store, &v2.embedding_space_id).role,
        None,
        "activating the successor of an extra space does not invent a default role"
    );
    Enrollments {
        _root: root,
        store,
        context_id,
        v1,
        v2: v2.embedding_space_id,
    }
}

/// Every enrollment row of one entity as (label, space, vector), sorted.
fn enrollment_rows(store: &Store, entity_id: EntityId) -> Vec<(Value, String, Value)> {
    let mut found: Vec<(Value, String, Value)> = rows(
        store,
        &format!(
            "SELECT label, embedding_space_id, vector FROM entity_reference_embeddings_{FACE_DIMENSION} WHERE entity_id = $entity"
        ),
        vec![("entity", Value::Uuid(entity_id.as_uuid()))],
    )
    .into_iter()
    .map(|row| match (&row[0], &row[1], &row[2]) {
        (label, Value::Text(space), vector) => (label.clone(), space.clone(), vector.clone()),
        other => panic!("enrollment row {other:?}"),
    })
    .collect();
    found.sort_by_key(|row| format!("{row:?}"));
    found
}

fn enroll(
    fixture: &Enrollments,
    entity: EntityId,
    space: &str,
    label: Option<&str>,
    vector: Vec<f32>,
) {
    let revision = if space == fixture.v1 { "v1" } else { "v2" };
    fixture
        .store
        .enroll_entity_reference(
            entity,
            space,
            label.map(str::to_string),
            vector,
            face_producer(revision),
        )
        .expect("enroll one reference");
}

#[test]
fn multi_shot_enrollments_stay_distinct_and_a_single_shot_duplicate_replaces() {
    // Each labelled shot is its own enrollment; enrolling the same shot again replaces it.
    let fixture = enrollments();
    let entity = create_entity(&fixture.store, fixture.context_id, "visitor");
    for (label, index) in [("front", 0), ("profile", 1), ("night", 2)] {
        enroll(&fixture, entity, &fixture.v1, Some(label), axis(index));
    }
    assert_eq!(
        enrollment_rows(&fixture.store, entity).len(),
        3,
        "three shots stay distinct"
    );
    enroll(&fixture, entity, &fixture.v1, Some("front"), axis(5));
    let after = enrollment_rows(&fixture.store, entity);
    assert_eq!(
        after.len(),
        3,
        "a duplicate shot replaces, it never adds a row"
    );
    let front: Vec<_> = after
        .iter()
        .filter(|(label, _, _)| *label == Value::Text("front".into()))
        .collect();
    assert_eq!(front.len(), 1);
    assert_eq!(
        front[0].2,
        Value::Vector(axis(5)),
        "the replacement carries the new vector"
    );
}

#[test]
fn an_unlabelled_enrollment_and_an_empty_labelled_one_are_distinct() {
    // No label and the empty label are two identities, so neither replaces the other.
    let fixture = enrollments();
    let entity = create_entity(&fixture.store, fixture.context_id, "courier");
    enroll(&fixture, entity, &fixture.v1, None, axis(0));
    enroll(&fixture, entity, &fixture.v1, Some(""), axis(1));
    let found = enrollment_rows(&fixture.store, entity);
    assert_eq!(found.len(), 2, "both enrollments are kept: {found:?}");
    assert!(found.contains(&(Value::Null, fixture.v1.clone(), Value::Vector(axis(0)))));
    assert!(found.contains(&(
        Value::Text(String::new()),
        fixture.v1.clone(),
        Value::Vector(axis(1))
    )));
}

#[test]
fn re_enrolling_under_a_new_model_keeps_the_old_spaces_enrollment() {
    // Enrolling a shot in a new model's space never overwrites the old model's enrollment.
    let fixture = enrollments();
    let entity = create_entity(&fixture.store, fixture.context_id, "resident");
    enroll(&fixture, entity, &fixture.v1, Some("front"), axis(0));
    enroll(&fixture, entity, &fixture.v2, Some("front"), axis(3));
    enroll(&fixture, entity, &fixture.v2, Some("front"), axis(3));
    let found = enrollment_rows(&fixture.store, entity);
    assert_eq!(
        found.len(),
        2,
        "one row per (entity, label, space): {found:?}"
    );
    assert!(found.contains(&(
        Value::Text("front".into()),
        fixture.v1.clone(),
        Value::Vector(axis(0))
    )));
    assert!(found.contains(&(
        Value::Text("front".into()),
        fixture.v2.clone(),
        Value::Vector(axis(3))
    )));
    assert!(
        fixture
            .store
            .execute_sql_for_test(
                "SELECT id FROM entity_reference_embedding_handles",
                &HashMap::new()
            )
            .is_err(),
        "enrollment handles fold into the enrollment row; no handle table accumulates"
    );
    let old_match = fixture
        .store
        .match_entity_reference(
            &fixture.v1,
            &axis(0),
            EntityReferenceMatchOptions::default(),
        )
        .expect("match in the old space");
    assert_eq!(
        old_match.hits.len(),
        1,
        "the old model still matches its own enrollment"
    );
}

#[test]
fn an_enrollment_match_after_a_model_switch_reports_subjects_enrolled_only_in_the_old_model() {
    // A non-match in the new space never reads as unknown when the subject is enrolled only before.
    let fixture = enrollments();
    let both = create_entity(&fixture.store, fixture.context_id, "re-enrolled");
    let only_old = create_entity(&fixture.store, fixture.context_id, "not yet re-enrolled");
    enroll(&fixture, both, &fixture.v1, Some("front"), axis(0));
    enroll(&fixture, both, &fixture.v2, Some("front"), axis(0));
    enroll(&fixture, only_old, &fixture.v1, Some("front"), axis(4));

    let matched = fixture
        .store
        .match_entity_reference(
            &fixture.v2,
            &axis(4),
            EntityReferenceMatchOptions {
                context_id: Some(fixture.context_id),
                top_k: Some(5),
                min_score: Some(0.99),
            },
        )
        .expect("match in the new space");
    assert!(
        matched.hits.is_empty(),
        "the subject has no new-space enrollment to match"
    );
    assert_eq!(
        matched.coverage.preserved_elsewhere, 1,
        "the subject enrolled only under the old model is reported, never read as unknown"
    );
    let old = fixture
        .store
        .match_entity_reference(
            &fixture.v1,
            &axis(4),
            EntityReferenceMatchOptions::default(),
        )
        .expect("match in the old space");
    assert_eq!(
        old.coverage.preserved_elsewhere, 0,
        "the oldest space has no ancestor"
    );
}

// ---------------------------------------------------------------------------------------------
// Two nodes: node-local data never syncs; travelling vectors travel, converge and follow the
// item's sync direction.
// ---------------------------------------------------------------------------------------------

/// One evidence item on `store` with a caller vector in the travelling space and a node-local
/// vector in the store's active text space.
fn record_with_caller_vector(
    store: &Store,
    space: &str,
    vector: Vec<f32>,
) -> (ContextId, EvidenceId) {
    let context_id = create_context(store);
    let entity_id = create_entity(store, context_id, "shared item");
    let evidence = record_text(store, context_id, entity_id, "a shared note");
    store
        .supply_embedding(
            EmbeddingTarget::Evidence(evidence),
            space,
            caller_identity("v1"),
            vector,
        )
        .expect("supply the caller vector");
    (context_id, evidence)
}

fn count_where(store: &Store, table: &str, column: &str, id: uuid::Uuid) -> usize {
    rows(
        store,
        &format!("SELECT {column} FROM {table} WHERE {column} = $id"),
        vec![("id", Value::Uuid(id))],
    )
    .len()
}

#[tokio::test]
async fn a_second_node_receives_travelling_vectors_and_never_node_local_ones() {
    // Real interrupted source work, then independent offline rebuilds on each node.
    let clock = Arc::new(AtomicU64::new(RECOVERY_CLOCK));
    let clock_reads = Arc::clone(&clock);
    let _clock =
        contextdb_core::Wallclock::test_clock_guard(move || clock_reads.load(Ordering::SeqCst));
    let old = TextModel::new("rev-a");
    let source_new = TextModel::new("rev-b");
    let receiver_new = TextModel::new("rev-b");
    let source_event_model = Arc::new(EventModel {
        calls: AtomicUsize::new(0),
    });
    let receiver_event_model = Arc::new(EventModel {
        calls: AtomicUsize::new(0),
    });
    let fixture = fabric_common::fabric_fixture_with_node_openers(
        "placement_second_node",
        "tenant-alpha",
        "tenant-alpha",
        |label| {
            fabric_common::FabricNode::with_models(
                label,
                |path| StoreConfig {
                    event_placement: Some(context_graph::EventPlacementPolicy::TenantHub),
                    ..plain_config(&path)
                },
                vec![old.registration(), source_event_model.registration()],
            )
        },
        |label| {
            fabric_common::FabricNode::with_models(
                label,
                |path| StoreConfig {
                    event_placement: Some(context_graph::EventPlacementPolicy::TenantHub),
                    ..plain_config(&path)
                },
                vec![
                    receiver_new.registration(),
                    receiver_event_model.registration(),
                ],
            )
        },
    )
    .await;
    let space = caller_space("v1");
    for edge in [&fixture.edge1, &fixture.edge2] {
        edge.store
            .register_embedding_space(space.clone())
            .expect("declare travelling placement");
    }
    let (context_id, evidence) = seed_items(&fixture.edge1.store);
    let item = evidence[0];
    let supplied_local = text_vector("source caller-supplied local values");
    fixture
        .edge1
        .store
        .supply_embedding(
            EmbeddingTarget::Evidence(item),
            &text_space("rev-a").embedding_space_id,
            text_identity("rev-a"),
            supplied_local.clone(),
        )
        .expect("store compatible local supply as given");
    let supplied_source_rows = node_local(
        &fixture.edge1.store,
        &text_space("rev-a").embedding_space_id,
    );
    assert_eq!(
        supplied_source_rows.len(),
        ITEMS,
        "compatible supply keeps exactly one vector per seeded item"
    );
    assert!(
        supplied_source_rows.contains_key(&item.as_uuid()),
        "compatible supply stores this source item in its declared local space"
    );
    assert_eq!(
        vector_row(&supplied_source_rows, item.as_uuid()).1,
        Value::Vector(supplied_local)
    );
    fixture
        .edge1
        .store
        .supply_embedding(
            EmbeddingTarget::Evidence(item),
            &space.embedding_space_id,
            caller_identity("v1"),
            text_vector("first"),
        )
        .expect("supply travelling values");
    let id_before = rows(
        &fixture.edge1.store,
        "SELECT id FROM evidence_embeddings_travelling WHERE target_id = $id",
        vec![("id", Value::Uuid(item.as_uuid()))],
    );
    assert_eq!(id_before.len(), 1);
    let digest_id = context_graph::DigestId::new_v7();
    fixture
        .edge1
        .store
        .create_digest(context_graph::CreateDigest {
            id: digest_id,
            context_id,
            source: "fixture".into(),
            summary: "shared digest summary".into(),
            kind: EvidenceKind::Text,
            embedding: context_graph::PrecomputedEmbedding {
                id: context_graph::EmbeddingId::new_v7(),
                target: EmbeddingTarget::Digest(digest_id),
                embedding_space_id: text_space("rev-a").embedding_space_id,
                produced_by: EvidenceProducer::from(&text_identity("rev-a")),
                vector: text_vector("caller digest"),
            },
            ..Default::default()
        })
        .expect("source supplied digest");
    let source_old_evidence_before = node_local(
        &fixture.edge1.store,
        &text_space("rev-a").embedding_space_id,
    );
    assert_eq!(source_old_evidence_before.len(), ITEMS);
    let source_old_digest_before = rows(
        &fixture.edge1.store,
        &format!(
            "SELECT id, target_id, created_at, {TEXT_COLUMN} FROM digest_embeddings WHERE embedding_space_id = $space ORDER BY target_id"
        ),
        vec![("space", Value::Text(text_space("rev-a").embedding_space_id))],
    );
    assert_eq!(source_old_digest_before.len(), 1);
    assert_eq!(
        source_old_digest_before[0][1],
        Value::Uuid(digest_id.as_uuid())
    );
    // Establish the receiver's own insert/head-recheck markers before source reindex work exists.
    // This baseline comes from this receiver's production apply path, so marker reasons and ids
    // are observed rather than guessed or copied from the production deterministic-id algorithm.
    fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    let pending_snapshot = |store: &Store| {
        rows(
            store,
            "SELECT id, target_table, target_id, role, reason, created_at FROM embedding_pending WHERE context_id = $context ORDER BY id",
            vec![("context", Value::Uuid(context_id.as_uuid()))],
        )
    };
    let receiver_pending_before = pending_snapshot(&fixture.edge2.store);
    assert!(
        !receiver_pending_before.is_empty(),
        "received content creates this receiver's own local work"
    );
    for marker in &receiver_pending_before {
        assert!(
            matches!(&marker[0], Value::Uuid(_)),
            "marker has an actual id"
        );
        assert!(
            matches!(&marker[1], Value::Text(table)
            if table == "evidence" || table == "digests" || table == "evidence_embeddings_travelling"),
            "receiver's own marker concerns an authorized received item: {marker:?}"
        );
        assert!(
            matches!(&marker[2], Value::Uuid(id)
            if evidence.iter().any(|item| item.as_uuid() == *id) || digest_id.as_uuid() == *id),
            "receiver's marker identifies exact received evidence/digest: {marker:?}"
        );
    }
    fixture
        .edge1
        .store
        .switch_embedding_space(
            EmbeddingRole::EvidenceText,
            text_space("rev-b"),
            OldSpace::Reindex,
        )
        .expect("source switches evidence locally");
    fixture
        .edge1
        .store
        .switch_embedding_space(
            EmbeddingRole::Digest,
            text_space("rev-b"),
            OldSpace::Reindex,
        )
        .expect("source switches digest locally");
    let paused = fixture
        .edge1
        .store
        .reindex_space_with_embedder(
            &text_space("rev-b").embedding_space_id,
            ReindexOptions {
                interrupt_after: Some(1),
            },
            source_new.clone(),
        )
        .expect_err("source interrupts with committed work");
    assert!(
        matches!(paused, CgError::ReindexPaused { ref embedding_space_id, .. }
        if embedding_space_id == &text_space("rev-b").embedding_space_id),
        "{paused:?}"
    );
    let claims = rows(
        &fixture.edge1.store,
        "SELECT owner_token FROM background_claims WHERE work = 'reindex'",
        Vec::new(),
    );
    assert_eq!(claims.len(), 1, "a real source claim exists before sync");
    let source_pending_before_sync = pending_snapshot(&fixture.edge1.store);
    let source_status = status_of(
        &fixture.edge1.store,
        &text_space("rev-b").embedding_space_id,
    );
    assert_eq!(source_status.work, BackgroundWorkState::Interrupted);
    assert_eq!(
        source_status.unembedded, ITEMS as u64,
        "one of 41 projections committed; stale predecessor rows durably identify the remaining 40"
    );
    let source_committed = node_local(
        &fixture.edge1.store,
        &text_space("rev-b").embedding_space_id,
    );
    assert_eq!(source_committed.len(), 1);
    let source_old_remaining = node_local(
        &fixture.edge1.store,
        &text_space("rev-a").embedding_space_id,
    );
    assert_eq!(source_old_remaining.len(), ITEMS - 1);
    for (target, predecessor) in &source_old_remaining {
        assert_eq!(
            predecessor,
            vector_row(&source_old_evidence_before, *target),
            "each remaining predecessor retains its exact vector, creation transaction and id"
        );
    }
    assert_eq!(
        source_old_evidence_before
            .keys()
            .filter(|target| !source_old_remaining.contains_key(target))
            .copied()
            .collect::<Vec<_>>(),
        source_committed.keys().copied().collect::<Vec<_>>(),
        "only the committed evidence projection lost its predecessor"
    );
    assert_eq!(
        rows(
            &fixture.edge1.store,
            &format!(
                "SELECT id, target_id, created_at, {TEXT_COLUMN} FROM digest_embeddings WHERE embedding_space_id = $space ORDER BY target_id"
            ),
            vec![("space", Value::Text(text_space("rev-a").embedding_space_id))],
        ),
        source_old_digest_before,
        "the one remaining digest retains its exact predecessor identity and values"
    );
    // Both callers own this fixture history. Capture it separately through the public append
    // API: TenantHub events push to the hub and are never pulled onto another edge.
    let authorized_event_batch = context_graph::AppendEventBatch::live(
        context_id,
        context_graph::EventAdapterDeclaration {
            connector_id: "placement-event".into(),
            adapter_version: "v1".into(),
            anchor_declaration_version: "v1".into(),
            anchor_schema: serde_json::json!({"type":"object"}),
            source_position_order: vec!["/ordinal".into()],
        },
        "source-node",
        vec![context_graph::EventDraft {
            session_ref: "placement-session".into(),
            source_event_key: "event-1".into(),
            source_position: serde_json::json!({"ordinal":1}),
            occurred_at: None,
            author: "operator".into(),
            kind: "message".into(),
            render_role: context_graph::EventRenderRole::HumanMessage,
            interaction_ref: None,
            text: Some("event vectors rebuild from authorized history".into()),
            payload: serde_json::json!({}),
            anchors: serde_json::json!({"fixture":"caller-authorized"}),
            source_remainder: serde_json::json!({}),
            coverage: context_graph::EventCoverage::Complete,
        }],
    );
    let source_capture = fixture
        .edge1
        .store
        .append_event_batch(authorized_event_batch.clone())
        .expect("append caller-authorized source history");
    assert_eq!(
        (source_capture.inserted, source_capture.already_present),
        (1, 0)
    );
    assert_eq!(source_capture.events.len(), 1);
    let build_events = |store: &Store, model: &Arc<EventModel>| {
        store
            .catch_up_event_search(context_graph::EventSearchBuildRequest {
                context_id,
                embedder: model.registration(),
                profile: Default::default(),
                embedding_workers: std::num::NonZeroUsize::MIN,
                max_events_per_batch: std::num::NonZeroU32::new(8).unwrap(),
                max_passages_per_batch: std::num::NonZeroU32::new(8).unwrap(),
                max_bytes_per_batch: std::num::NonZeroUsize::new(4096).unwrap(),
            })
            .expect("build this node's own event passages")
    };
    let source_event_build = build_events(&fixture.edge1.store, &source_event_model);
    assert_eq!(
        (
            source_event_build.embedded_event_count,
            source_event_build.passage_count,
            source_event_build.state
        ),
        (1, 1, context_graph::EventSearchGenerationState::Ready)
    );
    assert_eq!(source_event_model.calls.load(Ordering::SeqCst), 1);
    for table in [
        "events",
        "event_anchors",
        "event_search_passages",
        "event_search_embedding_handles",
        "event_search_cursors",
        "event_search_session_progress",
    ] {
        assert_eq!(
            rows(
                &fixture.edge1.store,
                &format!("SELECT * FROM {table}"),
                Vec::new()
            )
            .len(),
            1,
            "source has real durable event work in {table} before sync"
        );
    }
    fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    let second = &fixture.edge2.store;
    let receiver_status = status_of(second, &text_space("rev-b").embedding_space_id);
    assert_eq!(
        (receiver_status.work, receiver_status.unembedded),
        (BackgroundWorkState::Idle, ITEMS as u64 + 1),
        "receiver has its own 41 projections and no source interrupted claim or committed result"
    );
    let receiver_pending_after = pending_snapshot(second);
    assert_eq!(
        receiver_pending_after, receiver_pending_before,
        "syncing interrupted source work adds no progress: receiver keeps only its own marker ids, kinds, reasons and creation transactions"
    );
    for source_marker in &source_pending_before_sync {
        for receiver_marker in receiver_pending_after
            .iter()
            .filter(|row| row[0] == source_marker[0])
        {
            assert!(
                receiver_pending_before.contains(receiver_marker),
                "a coinciding deterministic marker id is this receiver's own pre-existing work"
            );
            assert_ne!(
                receiver_marker[5], source_marker[5],
                "coinciding item ids do not transfer the source marker's creation transaction"
            );
        }
        assert!(
            !receiver_pending_after.contains(source_marker),
            "the exact source-local marker (id, kind, target, role, reason, source transaction) must be absent on receiver: {source_marker:?}"
        );
    }
    assert_eq!(
        rows(
            second,
            "SELECT owner_token FROM background_claims WHERE work = 'reindex'",
            Vec::new()
        ),
        Vec::<Vec<Value>>::new()
    );
    assert_eq!(count_where(second, "evidence", "id", item.as_uuid()), 1);
    assert_eq!(count_where(second, "digests", "id", digest_id.as_uuid()), 1);
    assert!(node_local(second, &text_space("rev-a").embedding_space_id).is_empty());
    assert!(node_local(second, &text_space("rev-b").embedding_space_id).is_empty());
    assert_eq!(
        rows(
            second,
            "SELECT target_id FROM digest_embeddings",
            Vec::new()
        ),
        Vec::<Vec<Value>>::new()
    );
    assert_eq!(
        vector_row(
            &travelling(second, &space.embedding_space_id),
            item.as_uuid()
        )
        .1,
        Value::Vector(text_vector("first"))
    );
    assert_eq!(
        rows(
            second,
            "SELECT id FROM evidence_embeddings_travelling WHERE target_id = $id",
            vec![("id", Value::Uuid(item.as_uuid()))]
        ),
        id_before
    );
    assert_eq!(
        rows(
            second,
            "SELECT id FROM events WHERE context_id = $context",
            vec![("context", Value::Uuid(context_id.as_uuid()))]
        ),
        Vec::<Vec<Value>>::new(),
        "TenantHub PushOnly source events never pull onto another edge"
    );
    for table in [
        "event_anchors",
        "event_search_source_events",
        "event_search_passages",
        "event_search_embedding_handles",
        "event_search_cursors",
        "event_search_session_progress",
    ] {
        assert_eq!(
            rows(second, &format!("SELECT * FROM {table}"), Vec::new()),
            Vec::<Vec<Value>>::new(),
            "event vectors, handles and source progress remain local: {table}"
        );
    }
    assert_eq!(receiver_event_model.calls.load(Ordering::SeqCst), 0);
    // A long quiet interval is represented by the canonical clock, never by sleeping. There is
    // no automatic reindex scheduler/drive exposed today: this snapshot is not credited as that
    // missing automatic-takeover proof. Explicit work below remains an explicit user drive.
    clock.store(RECOVERY_CLOCK + 365 * 24 * 60 * 60 * 1000, Ordering::SeqCst);
    assert_eq!(
        status_of(second, &text_space("rev-b").embedding_space_id).work,
        BackgroundWorkState::Idle
    );
    assert_eq!(pending_snapshot(second), receiver_pending_after);
    assert_eq!(receiver_new.calls(), 0);

    fixture
        .edge1
        .store
        .supply_embedding(
            EmbeddingTarget::Evidence(item),
            &space.embedding_space_id,
            caller_identity("v1"),
            text_vector("corrected"),
        )
        .expect("correct travelling vector");
    fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    for store in [&fixture.edge1.store, second] {
        assert_eq!(travelling(store, &space.embedding_space_id).len(), 1);
        assert_eq!(
            vector_row(
                &travelling(store, &space.embedding_space_id),
                item.as_uuid()
            )
            .1,
            Value::Vector(text_vector("corrected"))
        );
        assert_eq!(
            rows(
                store,
                "SELECT id FROM evidence_embeddings_travelling WHERE target_id = $id",
                vec![("id", Value::Uuid(item.as_uuid()))]
            ),
            id_before,
            "replacement preserves the actual deterministic id"
        );
    }

    // Allowed enrollment receipt is a generic authenticated transport journey; per-site privacy
    // is a separate unavailable surface (CG-078).
    let enrollment_space = face_space("transport-v1");
    fixture
        .edge1
        .store
        .register_embedding_space(enrollment_space.clone())
        .expect("declare travelling enrollment space");
    let enrolled_entity = create_entity(
        &fixture.edge1.store,
        context_id,
        "transport enrollment subject",
    );
    let enrollment_id = fixture
        .edge1
        .store
        .enroll_entity_reference(
            enrolled_entity,
            &enrollment_space.embedding_space_id,
            Some("front".into()),
            axis(1),
            face_producer("transport-v1"),
        )
        .expect("populate source enrollment");
    let transported_enrollment = |store: &Store| {
        rows(
            store,
            &format!(
                "SELECT id, entity_id, label, embedding_space_id, vector FROM entity_reference_embeddings_{FACE_DIMENSION} WHERE entity_id = $entity ORDER BY id"
            ),
            vec![("entity", Value::Uuid(enrolled_entity.as_uuid()))],
        )
    };
    let expected_enrollment = |vector| {
        vec![vec![
            Value::Uuid(enrollment_id.as_uuid()),
            Value::Uuid(enrolled_entity.as_uuid()),
            Value::Text("front".into()),
            Value::Text(enrollment_space.embedding_space_id.clone()),
            Value::Vector(vector),
        ]]
    };
    assert_eq!(
        transported_enrollment(&fixture.edge1.store),
        expected_enrollment(axis(1))
    );
    fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    assert_eq!(
        count_where(second, "entities", "id", enrolled_entity.as_uuid()),
        1,
        "the authorized enrollment subject itself is received"
    );
    assert_eq!(
        transported_enrollment(second),
        expected_enrollment(axis(1)),
        "authenticated receipt keeps exact subject, shot, model, values and identity"
    );
    let corrected_enrollment = fixture
        .edge1
        .store
        .enroll_entity_reference(
            enrolled_entity,
            &enrollment_space.embedding_space_id,
            Some("front".into()),
            axis(4),
            face_producer("transport-v1"),
        )
        .expect("re-supply same enrollment with corrected values");
    assert_eq!(
        corrected_enrollment, enrollment_id,
        "re-supply replaces the deterministic enrollment identity"
    );
    fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    for store in [&fixture.edge1.store, second] {
        assert_eq!(
            transported_enrollment(store),
            expected_enrollment(axis(4)),
            "one corrected enrollment converges through production sync"
        );
    }

    let receiver_pending_before_own_capture = pending_snapshot(second);
    let fixture = fixture.into_offline().await;
    assert_eq!(
        fixture.stopped_hubs, 1,
        "the populated tenant hub is observed stopped before local work"
    );
    let second = &fixture.edge2.store;
    let receiver_capture = second
        .append_event_batch(authorized_event_batch)
        .expect("receiver separately captures its own caller-authorized fixture history");
    assert_eq!(
        (receiver_capture.inserted, receiver_capture.already_present),
        (1, 0)
    );
    assert_eq!(
        receiver_capture.events, source_capture.events,
        "public capture assigns the same logical event identity and content independently"
    );
    assert_eq!(
        pending_snapshot(second),
        receiver_pending_before_own_capture,
        "own event capture does not replace receiver evidence/digest work markers"
    );
    let receiver_event_build = build_events(second, &receiver_event_model);
    assert_eq!(
        (
            receiver_event_build.embedded_event_count,
            receiver_event_build.passage_count,
            receiver_event_build.state
        ),
        (1, 1, context_graph::EventSearchGenerationState::Ready)
    );
    assert_eq!(
        receiver_event_model.calls.load(Ordering::SeqCst),
        1,
        "receiver embeds its separately captured event once"
    );
    let rendered = "User [author=\"operator\"]: event vectors rebuild from authorized history";
    let mut expected_event_vector = vec![0.0; 768];
    expected_event_vector[..TEXT_DIMENSION].copy_from_slice(&text_vector(rendered));
    for (store, build) in [
        (&fixture.edge1.store, &source_event_build),
        (second, &receiver_event_build),
    ] {
        let event = rows(
            store,
            "SELECT id, rendered_text, vector_text, dimension FROM event_search_passages WHERE context_id = $context",
            vec![("context", Value::Uuid(context_id.as_uuid()))],
        );
        assert_eq!(event.len(), 1);
        assert_eq!(
            &event[0][1..],
            &[
                Value::Text(rendered.into()),
                Value::Vector(expected_event_vector.clone()),
                Value::Int64(768),
            ]
        );
        assert_eq!(
            rows(
                store,
                "SELECT passage_id, embedding_space_id FROM event_search_embedding_handles WHERE context_id = $context",
                vec![("context", Value::Uuid(context_id.as_uuid()))]
            ),
            vec![vec![
                event[0][0].clone(),
                Value::Text("placement-event-768".into())
            ]]
        );
        assert_eq!(
            rows(
                store,
                "SELECT connector_id, source_instance_ref, session_ref, committed_sequence FROM event_search_cursors WHERE context_id = $context",
                vec![("context", Value::Uuid(context_id.as_uuid()))]
            ),
            vec![vec![
                Value::Text("placement-event".into()),
                Value::Text("source-node".into()),
                Value::Text("placement-session".into()),
                Value::Int64(1)
            ]]
        );
        assert_eq!(
            rows(
                store,
                "SELECT event_count, committed_event_count, eligible_event_count, excluded_event_count, searchable_event_count, no_searchable_text_event_count, is_published FROM event_search_session_progress WHERE context_id = $context",
                vec![("context", Value::Uuid(context_id.as_uuid()))]
            ),
            vec![vec![
                Value::Int64(1),
                Value::Int64(1),
                Value::Int64(1),
                Value::Int64(0),
                Value::Int64(1),
                Value::Int64(0),
                Value::Bool(true)
            ]]
        );
        assert_eq!(
            rows(
                store,
                "SELECT id, rendered_text FROM event_search_passages WHERE context_id = $context AND generation_id = $generation AND is_current = $current AND is_searchable = $searchable ORDER BY vector_text <=> $query LIMIT 1",
                vec![
                    ("context", Value::Uuid(context_id.as_uuid())),
                    ("generation", Value::Uuid(build.generation_id.as_uuid())),
                    ("current", Value::Bool(true)),
                    ("searchable", Value::Bool(true)),
                    ("query", Value::Vector(expected_event_vector.clone()))
                ]
            ),
            vec![vec![event[0][0].clone(), Value::Text(rendered.into())]],
            "each node searches its independently built event vector index"
        );
    }
    assert_eq!(source_event_model.calls.load(Ordering::SeqCst), 1);
    assert_eq!(receiver_event_model.calls.load(Ordering::SeqCst), 1);
    // Every client and hub has been dropped after its serving task exited; both drives are offline.
    let local_before = node_local(
        &fixture.edge1.store,
        &text_space("rev-b").embedding_space_id,
    );
    let resumed = fixture
        .edge1
        .store
        .reindex_space_with_embedder(
            &text_space("rev-b").embedding_space_id,
            ReindexOptions::default(),
            source_new.clone(),
        )
        .expect("source immediately finishes its own interrupted work");
    assert_eq!(
        (resumed.embedded, resumed.skipped, resumed.remaining),
        (ITEMS as u64, 0, 0)
    );
    let rebuilt = second
        .reindex_space(
            &text_space("rev-b").embedding_space_id,
            ReindexOptions::default(),
        )
        .expect("receiver builds its own copy");
    assert_eq!(
        (rebuilt.embedded, rebuilt.skipped, rebuilt.remaining),
        (ITEMS as u64 + 1, 0, 0)
    );
    assert_eq!(source_new.calls(), ITEMS + 1);
    assert_eq!(receiver_new.calls(), ITEMS + 1);
    for store in [&fixture.edge1.store, second] {
        let vectors = node_local(store, &text_space("rev-b").embedding_space_id);
        assert_eq!(vectors.len(), ITEMS);
        for (index, item) in evidence.iter().enumerate() {
            assert_eq!(
                vector_row(&vectors, item.as_uuid()).1,
                Value::Vector(text_vector(&format!("rev-b:{}", note(index))))
            );
        }
        assert_eq!(
            rows(
                store,
                &format!("SELECT {TEXT_COLUMN} FROM digest_embeddings WHERE target_id = $id"),
                vec![("id", Value::Uuid(digest_id.as_uuid()))]
            ),
            vec![vec![Value::Vector(text_vector(
                "rev-b:shared digest summary"
            ))]]
        );
        let final_status = status_of(store, &text_space("rev-b").embedding_space_id);
        assert_eq!(final_status.work, BackgroundWorkState::Idle);
        assert_eq!(final_status.unembedded, 0);
    }
    let source_vectors = node_local(
        &fixture.edge1.store,
        &text_space("rev-b").embedding_space_id,
    );
    let receiver_vectors = node_local(second, &text_space("rev-b").embedding_space_id);
    assert_eq!(source_vectors.len(), ITEMS);
    assert_eq!(receiver_vectors.len(), ITEMS);
    for item in &evidence {
        assert_eq!(
            vector_row(&source_vectors, item.as_uuid()).2,
            vector_row(&receiver_vectors, item.as_uuid()).2,
            "independent nodes generate the same actual identity for this item/model"
        );
    }
    let digest_identity = |store: &Store| {
        rows(
            store,
            "SELECT id FROM digest_embeddings WHERE target_id = $id AND embedding_space_id = $space",
            vec![
                ("id", Value::Uuid(digest_id.as_uuid())),
                ("space", Value::Text(text_space("rev-b").embedding_space_id)),
            ],
        )
    };
    let source_digest_id = digest_identity(&fixture.edge1.store);
    assert_eq!(source_digest_id.len(), 1);
    assert_eq!(
        digest_identity(second),
        source_digest_id,
        "independent digest rebuilds preserve the same actual item/model identity"
    );
    for (target, row) in local_before {
        assert_eq!(
            *vector_row(
                &node_local(
                    &fixture.edge1.store,
                    &text_space("rev-b").embedding_space_id
                ),
                target
            ),
            row,
            "source retains each committed row's exact values and creation transaction"
        );
    }
    #[cfg(context_graph_internal_test_support)]
    for store in [&fixture.edge1.store, second] {
        let hits = context_graph::test_support::embedding::vector_search(
            store,
            &text_vector(&format!("rev-b:{}", note(0))),
            &text_space("rev-b").embedding_space_id,
            context_graph::test_support::embedding::VectorSearchFilters {
                top_k: Some(1),
                ..Default::default()
            },
        )
        .expect("each node built its own actual vector index");
        assert_eq!(hits.hits.len(), 1);
        assert_eq!(hits.hits[0].target, EmbeddingTarget::Evidence(item));
    }
    // Real authorized sync and wipe/restore all contact the populated alpha hub.
    // Denied-recipient privacy has no typed production authorization surface today (CG-078).
    {
        let recipient = "tenant-alpha";
        let restore_model = TextModel::new("rev-a");
        let reproducible_space = restore_model.embedding_space();
        let privacy = fabric_common::fabric_fixture_with_node_openers(
            "placement_restore",
            "tenant-alpha",
            recipient,
            |label| {
                fabric_common::FabricNode::with_models(
                    label,
                    |path| StoreConfig {
                        event_placement: Some(context_graph::EventPlacementPolicy::TenantHub),
                        ..plain_config(&path)
                    },
                    vec![restore_model.registration()],
                )
            },
            |label| {
                fabric_common::FabricNode::with_models(
                    label,
                    |path| StoreConfig {
                        event_placement: Some(context_graph::EventPlacementPolicy::TenantHub),
                        extra_embedding_spaces: vec![reproducible_space.clone(), space.clone()],
                        ..plain_config(&path)
                    },
                    vec![],
                )
            },
        )
        .await;
        privacy
            .edge1
            .store
            .register_embedding_space(space.clone())
            .expect("source travelling declaration");
        let (_, target) = record_with_caller_vector(
            &privacy.edge1.store,
            &space.embedding_space_id,
            text_vector("private item"),
        );
        let authorized_source_row = travelling(&privacy.edge1.store, &space.embedding_space_id);
        assert_eq!(authorized_source_row.len(), 1);
        let source_local = node_local(&privacy.edge1.store, &reproducible_space.embedding_space_id);
        assert_eq!(
            source_local.len(),
            1,
            "the source populated this exact space"
        );
        assert_eq!(
            vector_row(&source_local, target.as_uuid()).1,
            Value::Vector(text_vector("rev-a:a shared note"))
        );
        fabric_common::push_then_pull(&privacy.edge1_client, &privacy.edge2_client).await;
        assert_eq!(
            status_of(&privacy.edge2.store, &reproducible_space.embedding_space_id).state,
            SpaceState::Unavailable
        );
        assert!(
            node_local(&privacy.edge2.store, &reproducible_space.embedding_space_id).is_empty()
        );
        assert_eq!(
            travelling(&privacy.edge2.store, &space.embedding_space_id),
            authorized_source_row
        );
        assert_eq!(
            count_where(&privacy.edge2.store, "evidence", "id", target.as_uuid()),
            1
        );
        assert_eq!(
            count_where(
                &privacy.edge2.store,
                "evidence_embeddings_travelling",
                "target_id",
                target.as_uuid()
            ),
            1
        );
        let restore_root = tempfile::tempdir().expect("restore root");
        let restore_path = restore_root.path().join("wiped-receiver.db");
        let config = || StoreConfig {
            event_placement: Some(context_graph::EventPlacementPolicy::TenantHub),
            extra_embedding_spaces: vec![reproducible_space.clone(), space.clone()],
            ..plain_config(&restore_path)
        };
        let receiver = Store::open(config()).expect("receiver without a reproducible model");
        let client = contextdb_server::SyncClient::new(
            receiver.sync_database(),
            &privacy.edge2_dial_spec,
            contextdb_core::TenantId::from(recipient),
        )
        .expect("authenticated recipient restore client");
        fabric_common::push_then_pull(&privacy.edge1_client, &client).await;
        assert_eq!(
            travelling(&receiver, &space.embedding_space_id),
            authorized_source_row,
            "authorized sync/restore preserves exact id, values and creation transaction"
        );
        assert_eq!(
            count_where(&receiver, "evidence", "id", target.as_uuid()),
            1
        );
        assert_eq!(
            count_where(
                &receiver,
                "evidence_embeddings_travelling",
                "target_id",
                target.as_uuid()
            ),
            1
        );
        assert_eq!(
            status_of(&receiver, &reproducible_space.embedding_space_id).state,
            SpaceState::Unavailable
        );
        assert!(node_local(&receiver, &reproducible_space.embedding_space_id).is_empty());
        #[cfg(context_graph_internal_test_support)]
        {
            let hits = context_graph::test_support::embedding::vector_search(
                &receiver,
                &text_vector("private item"),
                &space.embedding_space_id,
                Default::default(),
            )
            .expect("received numbers can be compared without the model");
            assert_eq!(hits.hits.len(), 1);
            assert_eq!(hits.hits[0].target, EmbeddingTarget::Evidence(target));
        }
        drop(client);
        drop(receiver);
        let receiver =
            Store::initialize_fresh(config()).expect("wipe this recipient's entire database");
        assert_eq!(
            count_where(&receiver, "evidence", "id", target.as_uuid()),
            0
        );
        assert_eq!(
            count_where(
                &receiver,
                "evidence_embeddings_travelling",
                "target_id",
                target.as_uuid()
            ),
            0
        );
        let client = contextdb_server::SyncClient::new(
            receiver.sync_database(),
            &privacy.edge2_dial_spec,
            contextdb_core::TenantId::from(recipient),
        )
        .expect("reopen retained authenticated recipient identity");
        fabric_common::push_then_pull(&privacy.edge1_client, &client).await;
        assert_eq!(
            status_of(&receiver, &reproducible_space.embedding_space_id).state,
            SpaceState::Unavailable
        );
        assert!(node_local(&receiver, &reproducible_space.embedding_space_id).is_empty());
        assert_eq!(
            travelling(&receiver, &space.embedding_space_id),
            authorized_source_row,
            "authorized sync/restore preserves exact id, values and creation transaction"
        );
        assert_eq!(
            count_where(&receiver, "evidence", "id", target.as_uuid()),
            1
        );
        assert_eq!(
            count_where(
                &receiver,
                "evidence_embeddings_travelling",
                "target_id",
                target.as_uuid()
            ),
            1
        );
        {
            assert_eq!(
                vector_row(
                    &travelling(&receiver, &space.embedding_space_id),
                    target.as_uuid()
                )
                .1,
                Value::Vector(text_vector("private item"))
            );
        }
    }
}

#[test]
fn a_travelling_table_whose_direction_differs_from_its_item_is_refused_at_open() {
    // A store whose travelling vectors would sync differently from their evidence never opens.
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("mismatch.db");
    drop(Store::open(plain_config(&path)).expect("create the store"));
    let db = contextdb_engine::Database::open(&path).expect("raw admin handle");
    db.execute("ALTER TABLE evidence SET SYNC PUSH ONLY", &HashMap::new())
        .expect("change only the item table's direction");
    db.close().expect("close the admin handle");
    let refused = Store::open(plain_config(&path)).expect_err("a mismatched direction is refused");
    match refused {
        CgError::TravellingPlacementMismatch {
            table, item_table, ..
        } => {
            assert_eq!(table, "evidence_embeddings_travelling");
            assert_eq!(item_table, "evidence");
        }
        other => panic!("expected TravellingPlacementMismatch, got {other:?}"),
    }
}

/// Whether item `target`'s row in travelling `space` is the item's head.
fn is_head(store: &Store, space: &str, target: EvidenceId) -> Value {
    let found = rows(
        store,
        "SELECT is_head FROM evidence_embeddings_travelling WHERE embedding_space_id = $space AND target_id = $target",
        vec![
            ("space", Value::Text(space.to_string())),
            ("target", Value::Uuid(target.as_uuid())),
        ],
    );
    assert_eq!(found.len(), 1, "one row for the item in {space}");
    found[0][0].clone()
}

#[tokio::test]
async fn concurrent_writes_to_an_item_in_old_and_new_spaces_converge_on_the_new_head() {
    // One node moves an item to the successor while another re-supplies it in the old space; both end agreeing.
    let fixture = fabric_common::fabric_fixture("placement_head_convergence").await;
    let old = caller_space("v1");
    let new = caller_space("v2");
    for edge in [&fixture.edge1, &fixture.edge2] {
        edge.store
            .register_embedding_space(old.clone())
            .expect("declare the old caller space on both nodes");
    }
    fixture
        .edge1
        .store
        .switch_travelling_space(&old.embedding_space_id, new.clone())
        .expect("declare the successor");
    let (_, item) = record_with_caller_vector(
        &fixture.edge1.store,
        &old.embedding_space_id,
        text_vector("old model"),
    );
    fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    // Concurrently: the first node writes the successor vector, the second re-supplies the old one.
    fixture
        .edge1
        .store
        .supply_embedding(
            EmbeddingTarget::Evidence(item),
            &new.embedding_space_id,
            caller_identity("v2"),
            text_vector("new model"),
        )
        .expect("write the item's successor vector");
    fixture
        .edge2
        .store
        .supply_embedding(
            EmbeddingTarget::Evidence(item),
            &old.embedding_space_id,
            caller_identity("v1"),
            text_vector("old model again"),
        )
        .expect("re-supply the item's old vector");

    for _ in 0..2 {
        fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
        fabric_common::push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;
        for edge in [&fixture.edge1, &fixture.edge2] {
            edge.store
                .reindex_space(&new.embedding_space_id, ReindexOptions::default())
                .expect("process the received rows' head rechecks");
        }
    }
    for store in [&fixture.edge1.store, &fixture.edge2.store] {
        assert_eq!(
            is_head(store, &old.embedding_space_id, item),
            Value::Bool(false),
            "the old-space row is not the item's head once a successor row exists"
        );
        assert_eq!(
            is_head(store, &new.embedding_space_id, item),
            Value::Bool(true)
        );
        assert_eq!(
            vector_row(&travelling(store, &old.embedding_space_id), item.as_uuid()).1,
            Value::Vector(text_vector("old model again"))
        );
        assert_eq!(
            vector_row(&travelling(store, &new.embedding_space_id), item.as_uuid()).1,
            Value::Vector(text_vector("new model"))
        );
        assert_eq!(
            status_of(store, &new.embedding_space_id).preserved_elsewhere,
            0,
            "an item with a successor vector is never counted as preserved elsewhere"
        );
    }

    // Quiescence: a converged pair writes nothing more, so corrections never echo between nodes.
    let settled = [
        all_travelling(&fixture.edge1.store),
        all_travelling(&fixture.edge2.store),
    ];
    fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    fabric_common::push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;
    for edge in [&fixture.edge1, &fixture.edge2] {
        edge.store
            .reindex_space(&new.embedding_space_id, ReindexOptions::default())
            .expect("process anything the last round received");
    }
    for (client, before, store) in [
        (&fixture.edge1_client, &settled[0], &fixture.edge1.store),
        (&fixture.edge2_client, &settled[1], &fixture.edge2.store),
    ] {
        assert!(
            !client
                .has_pending_push_changes()
                .expect("pending push check"),
            "a converged node has nothing new to push"
        );
        assert_eq!(
            &all_travelling(store),
            before,
            "a converged node rewrote no travelling row"
        );
    }
}

/// Every travelling row of the node as (space, target, is_head, vector).
fn all_travelling(store: &Store) -> Vec<Vec<Value>> {
    rows(
        store,
        &format!(
            "SELECT embedding_space_id, target_id, is_head, {TEXT_COLUMN} FROM evidence_embeddings_travelling ORDER BY embedding_space_id, target_id"
        ),
        Vec::new(),
    )
}
