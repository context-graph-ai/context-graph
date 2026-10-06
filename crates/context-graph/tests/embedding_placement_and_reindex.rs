//! Embedding placement per space, and reindex as shared background work.
//!
//! The contract: every embedding space declares where its vectors live. A reproducible space is
//! node-local — each node builds its own copy and nothing about it syncs. A space only callers
//! fill, or whose input is media, travels with its item and converges across nodes because a
//! repeat replaces. A space is one model identity, so a new model or weights revision is a new
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
use std::sync::atomic::{AtomicUsize, Ordering};
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
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: id,
            table_name: "evidence".into(),
            vector_column: TEXT_COLUMN.into(),
        }],
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
            vector: text_vector(&text),
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
    Store::open_with_embedder_registrations(
        plain_config(path),
        models.iter().map(|model| model.registration()).collect(),
    )
    .expect("open with the text models")
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

/// Every evidence vector row of one space in one physical table, as target -> (created_at, vector).
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

fn vectors_in(store: &Store, table: &str, space_id: &str) -> BTreeMap<uuid::Uuid, (Value, Value)> {
    let column = column_of(space_id);
    rows(
        store,
        &format!(
            "SELECT target_id, created_at, {column} FROM {table} WHERE embedding_space_id = $space"
        ),
        vec![("space", Value::Text(space_id.to_string()))],
    )
    .into_iter()
    .map(|row| match (&row[0], &row[1], &row[2]) {
        (Value::Uuid(target), created_at, vector) => {
            (*target, (created_at.clone(), vector.clone()))
        }
        other => panic!("evidence embedding row {other:?}"),
    })
    .collect()
}

fn node_local(store: &Store, space_id: &str) -> BTreeMap<uuid::Uuid, (Value, Value)> {
    vectors_in(store, "evidence_embeddings", space_id)
}

fn travelling(store: &Store, space_id: &str) -> BTreeMap<uuid::Uuid, (Value, Value)> {
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
    let old = TextModel::new("rev-a");
    let new = TextModel::dying_on("rev-b", DIE_ON_CALL);
    let store = open_with(&path, &[&old, &new]);
    store
        .switch_embedding_space(
            EmbeddingRole::EvidenceText,
            text_space("rev-b"),
            OldSpace::Reindex,
        )
        .expect("switch the text role to the new model");
    let _ = store.reindex_space(
        &text_space("rev-b").embedding_space_id,
        ReindexOptions::default(),
    );
    panic!("the reindex must reach the embedder's dying call");
}

#[test]
fn a_crash_mid_reindex_resumes_on_reopen_repeating_no_committed_item() {
    // A reindex killed between batches resumes at once on reopen and redoes only uncommitted items.
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

    let report = store
        .reindex_space(&new_space, ReindexOptions::default())
        .expect("the immediate rerun resumes the killed reindex");
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
    let old = TextModel::new("rev-a");
    let new = TextModel::new("rev-b");
    let store = open_with(&path, &[&old, &new]);
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
    let old_vector = before_switch[&refused].clone();

    store
        .switch_embedding_space(
            EmbeddingRole::EvidenceText,
            text_space("rev-b"),
            OldSpace::Reindex,
        )
        .expect("switch to the new model");
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
    let store = open_with(&path, &[&old, &new]);
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

#[test]
fn a_travelling_switch_preserves_and_reports_what_this_node_cannot_regenerate() {
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
    // A supplied vector is never silently discarded or recomputed: another model's vector is a
    // typed refusal that writes nothing, and the space's own model's vector is kept exactly.
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("supplied-vectors.db");
    let store = Store::open(StoreConfig {
        db_path: path,
        default_text_embedder: None,
        extra_embedding_spaces: vec![text_space("rev-a")],
        ..StoreConfig::default()
    })
    .expect("open with the space declared and no model on this node");
    let space = text_space("rev-a").embedding_space_id;
    let context_id = create_context(&store);
    let entity_id = create_entity(&store, context_id, "notebook");
    let count = |table: &str, evidence_id: EvidenceId| {
        let column = if table == "evidence" {
            "id"
        } else {
            "target_id"
        };
        rows(
            &store,
            &format!("SELECT {column} FROM {table} WHERE {column} = $id"),
            vec![("id", Value::Uuid(evidence_id.as_uuid()))],
        )
        .len()
    };

    // Another model's vector, through the observation write: refused whole, nothing written.
    let foreign = EvidenceId::new_v7();
    let refused = store.record_observation(supplied_note(
        context_id,
        entity_id,
        foreign,
        &space,
        text_vector("foreign"),
        &caller_identity("foreign"),
    ));
    assert!(
        matches!(
            &refused,
            Err(CgError::EmbeddingProducerMismatch { space: refused_space, .. }) if *refused_space == space
        ),
        "a foreign producer is refused with the typed error: {refused:?}"
    );
    assert_eq!(count("evidence", foreign), 0, "no evidence row");
    assert_eq!(count("evidence_embeddings", foreign), 0, "no vector row");
    assert!(node_local(&store, &space).is_empty(), "nothing is stored");

    // The space's own model's vector is accepted; then another model's vector for the same
    // item, through the direct supply door, is refused and changes nothing.
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
        .expect("a supply from the space's own model is accepted");
    let refused_supply = store.supply_embedding(
        EmbeddingTarget::Evidence(compatible),
        &space,
        caller_identity("foreign"),
        text_vector("foreign again"),
    );
    assert!(
        matches!(
            &refused_supply,
            Err(CgError::EmbeddingProducerMismatch { .. })
        ),
        "{refused_supply:?}"
    );

    // The space's own model's vector is stored as given, under the space's placement.
    assert_eq!(
        recorded.embedding_ids.len(),
        1,
        "one row written: {recorded:?}"
    );
    assert!(
        matches!(
            recorded.supplied_vectors.as_slice(),
            [context_graph::SuppliedVectorOutcome {
                outcome: context_graph::SuppliedVector::Stored { embedding_id },
                ..
            }] if *embedding_id == recorded.embedding_ids[0]
        ),
        "the write reports the supplied vector as stored: {:?}",
        recorded.supplied_vectors
    );
    let stored = node_local(&store, &space);
    assert_eq!(stored.len(), 1, "only the compatible vector is stored");
    assert_eq!(
        stored[&compatible.as_uuid()].1,
        Value::Vector(supplied_vector),
        "the supplied values are kept as given, never replaced by a recompute"
    );
    assert_eq!(status_of(&store, &space).unembedded, 0);
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

fn sql(store: &Store, statement: &str) {
    store
        .execute_sql_for_test(statement, &HashMap::new())
        .unwrap_or_else(|error| panic!("{statement}: {error:?}"));
}

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
    // Node-local vectors, markers and claims stay on their node; a caller vector travels and converges.
    let fixture = fabric_common::fabric_fixture("placement_second_node").await;
    let space = caller_space("v1");
    for edge in [&fixture.edge1, &fixture.edge2] {
        edge.store
            .register_embedding_space(space.clone())
            .expect("declare the caller space on both nodes");
    }
    let (_, evidence) = record_with_caller_vector(
        &fixture.edge1.store,
        &space.embedding_space_id,
        text_vector("first"),
    );
    let edge1_text = active(&fixture.edge1.store, EmbeddingRole::EvidenceText);
    assert_eq!(
        count_where(
            &fixture.edge1.store,
            "evidence_embeddings",
            "target_id",
            evidence.as_uuid()
        ),
        1,
        "the writing node embedded the item node-locally"
    );

    fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    let second = &fixture.edge2.store;
    assert_eq!(
        count_where(second, "evidence", "id", evidence.as_uuid()),
        1,
        "the item arrived"
    );
    assert_eq!(
        count_where(
            second,
            "evidence_embeddings",
            "target_id",
            evidence.as_uuid()
        ),
        0,
        "no node-local vector is ever received"
    );
    assert!(
        rows(
            second,
            "SELECT owner_token FROM background_claims",
            Vec::new()
        )
        .is_empty(),
        "no claim is ever received"
    );
    assert_eq!(
        rows(
            second,
            "SELECT reason FROM embedding_pending WHERE target_id = $id",
            vec![("id", Value::Uuid(evidence.as_uuid()))],
        ),
        vec![vec![Value::Text("pending".into())]],
        "the receiving node marks the item for its own rebuild instead of receiving a vector"
    );
    assert_eq!(
        travelling(second, &space.embedding_space_id)
            .get(&evidence.as_uuid())
            .map(|(_, vector)| vector.clone()),
        Some(Value::Vector(text_vector("first"))),
        "the caller vector travelled with its item"
    );
    let _ = edge1_text;

    fixture
        .edge1
        .store
        .supply_embedding(
            EmbeddingTarget::Evidence(evidence),
            &space.embedding_space_id,
            caller_identity("v1"),
            text_vector("corrected"),
        )
        .expect("re-supply a corrected vector");
    fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    for store in [&fixture.edge1.store, &fixture.edge2.store] {
        let vectors = travelling(store, &space.embedding_space_id);
        assert_eq!(vectors.len(), 1, "a repeat replaces, never duplicates");
        assert_eq!(
            vectors
                .get(&evidence.as_uuid())
                .map(|(_, vector)| vector.clone()),
            Some(Value::Vector(text_vector("corrected"))),
            "both nodes converge on the correction"
        );
    }
}

#[tokio::test]
async fn one_way_evidence_placement_carries_travelling_vectors_one_way() {
    // When evidence only pushes from a node, its travelling vectors push one way with it.
    let fixture = fabric_common::fabric_fixture("placement_one_way").await;
    let space = caller_space("v1");
    for edge in [&fixture.edge1, &fixture.edge2] {
        edge.store
            .register_embedding_space(space.clone())
            .expect("declare the caller space on both nodes");
    }
    sql(
        &fixture.edge1.store,
        "ALTER TABLE evidence SET SYNC PUSH ONLY",
    );
    sql(
        &fixture.edge1.store,
        "ALTER TABLE evidence_embeddings_travelling SET SYNC PUSH ONLY",
    );
    let (_, evidence) = record_with_caller_vector(
        &fixture.edge1.store,
        &space.embedding_space_id,
        text_vector("from the edge"),
    );

    fabric_common::push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    assert_eq!(
        travelling(&fixture.edge2.store, &space.embedding_space_id)
            .get(&evidence.as_uuid())
            .map(|(_, vector)| vector.clone()),
        Some(Value::Vector(text_vector("from the edge"))),
        "the pushing node's vector reaches the other node"
    );

    fixture
        .edge2
        .store
        .supply_embedding(
            EmbeddingTarget::Evidence(evidence),
            &space.embedding_space_id,
            caller_identity("v1"),
            text_vector("from elsewhere"),
        )
        .expect("another node re-supplies the vector");
    fabric_common::push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;
    assert_eq!(
        travelling(&fixture.edge1.store, &space.embedding_space_id)
            .get(&evidence.as_uuid())
            .map(|(_, vector)| vector.clone()),
        Some(Value::Vector(text_vector("from the edge"))),
        "a push-only node never receives the vector back"
    );
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
        assert!(
            travelling(store, &new.embedding_space_id).contains_key(&item.as_uuid()),
            "the successor row is present on every node"
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
