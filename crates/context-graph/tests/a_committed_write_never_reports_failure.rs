//! A committed write never reports failure.
//!
//! The contract: a write returns `Err` only when nothing committed. Once a write has committed, a
//! refused read of what it committed — the real-world trigger is a read that exhausts its
//! active-time allowance under load — never turns the call into an error. The call returns `Ok`
//! and either its value equals what a public read returns (the read moved inside the write's
//! transaction), or the value carries `Readback::Unavailable` for exactly the part it could not
//! read back.
//!
//! Each row refuses the next post-commit readback of one store with the real
//! `ReadActiveTimeExceeded` error through the one fault-injection seam, makes the call on the
//! same thread, and asserts: the call returned `Ok`, the typed readback reports `Unavailable`
//! where the family reads after committing, and the committed row is present when the store is
//! reopened. A read budget cannot produce this refusal: the readbacks need less read memory than
//! the writes' own reads, so a budget small enough to refuse them refuses the write first.

#![cfg(context_graph_internal_test_support)]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

#[path = "common/fake_extractor.rs"]
#[allow(dead_code)]
mod fake_extractor;

use context_graph::harness::{CaptureHarnessConfig, run_capture};
use context_graph::ingest::CaptureRequest;
use context_graph::test_support::refuse_next_post_commit_readback_for_test;
use context_graph::tier0::{Tier0CaptureRequest, capture_tier0, queue_counts};
use context_graph::{
    CgError, ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId,
    DecisionPatch, DecisionStatus, EntityId, EvidenceId, EvidenceKind, EvidenceRef, IntentionId,
    IntentionOrigin, IntentionPatch, IntentionStatus, ListEntityFilter, ObservationId, Readback,
    RecordObservation, RetentionStatus, Store, StoreConfig,
};
use contextdb_core::Value;

/// The refusal a loaded store gives a read that exhausts its active-time allowance.
fn refused_readback() -> CgError {
    CgError::ReadActiveTimeExceeded { limit_ms: 1 }
}

/// Every store here carries the lexical test embedder, so capture and text evidence can embed.
fn open(path: &Path) -> Store {
    let embedder =
        std::sync::Arc::new(context_graph::embedder::mock::LexicalMockEmbedder::default());
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: path.to_path_buf(),
            default_text_embedder: None,
            ..StoreConfig::default()
        },
        vec![context_graph::EmbedderRegistration {
            space: context_graph::Embedder::embedding_space(embedder.as_ref()),
            embedder,
        }],
    )
    .expect("open the store")
}

/// A seeded store, closed, that each write copies before running.
struct Template {
    _root: tempfile::TempDir,
    db_path: PathBuf,
}

impl Template {
    fn seed<T>(seed: impl FnOnce(&Store) -> T) -> (Self, T) {
        let root = tempfile::tempdir().expect("template root");
        let db_path = root.path().join("committed-write.db");
        let store = open(&db_path);
        let seeded = seed(&store);
        drop(store);
        (
            Self {
                _root: root,
                db_path,
            },
            seeded,
        )
    }

    /// A fresh copy of every store file in its own directory.
    fn copy(&self) -> StoreCopy {
        let root = tempfile::tempdir().expect("copy root");
        let source = self.db_path.parent().expect("template folder");
        for entry in std::fs::read_dir(source).expect("list the template folder") {
            let entry = entry.expect("template entry");
            if entry.file_type().expect("entry type").is_file() {
                std::fs::copy(entry.path(), root.path().join(entry.file_name()))
                    .expect("copy one store file");
            }
        }
        let db_path = root
            .path()
            .join(self.db_path.file_name().expect("store file name"));
        StoreCopy {
            _root: root,
            db_path,
        }
    }
}

struct StoreCopy {
    _root: tempfile::TempDir,
    db_path: PathBuf,
}

impl StoreCopy {
    /// The copy reopened, to read what committed.
    fn reopen(&self) -> Store {
        open(&self.db_path)
    }
}

/// Run `write` on a fresh copy whose next post-commit readback is refused; return its result and
/// the copy, closed, so the caller can reopen it and read what committed.
fn with_refused_readback<T>(
    template: &Template,
    write: impl FnOnce(&Store) -> T,
) -> (T, StoreCopy) {
    let copy = template.copy();
    let store = open(&copy.db_path);
    refuse_next_post_commit_readback_for_test(&copy.db_path, refused_readback());
    let result = write(&store);
    drop(store);
    (result, copy)
}

fn create_context(store: &Store) -> ContextId {
    store
        .create_context(CreateContext {
            name: "committed write".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id
}

fn create_intention(store: &Store, context_id: ContextId, description: &str) -> IntentionId {
    store
        .create_intention(CreateIntention {
            id: None,
            description: description.into(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
            blueprint_catalog_id: None,
        })
        .expect("create intention")
        .id
}

fn decision_input(
    context_id: ContextId,
    intention_id: IntentionId,
    description: &str,
) -> CreateDecision {
    CreateDecision {
        decision_type: "choice".into(),
        description: description.into(),
        intention_ids: vec![intention_id],
        context_id,
        ..CreateDecision::default()
    }
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
        .expect("inspect committed rows")
        .rows
}

fn seeded_decision() -> (Template, (ContextId, IntentionId, DecisionId)) {
    Template::seed(|store| {
        let context_id = create_context(store);
        let intention_id = create_intention(store, context_id, "keep the cache local");
        let decision = store
            .create_decision(decision_input(context_id, intention_id, "use sqlite"))
            .expect("seed decision");
        (context_id, intention_id, decision.id)
    })
}

fn unavailable(readback: &Readback) -> bool {
    matches!(readback, Readback::Unavailable { .. })
}

#[test]
fn a_decision_write_that_committed_never_reports_failure() {
    // Creating, updating, superseding and archiving a decision return Ok once committed.
    let (template, (context_id, intention_id, precedent)) = seeded_decision();

    let (created, copy) = with_refused_readback(&template, |store| {
        store.create_decision(CreateDecision {
            precedent_ids: vec![precedent],
            ..decision_input(context_id, intention_id, "use sqlite again")
        })
    });
    let created = created.expect("a committed decision is reported as created");
    assert!(
        unavailable(&created.precedent_readback),
        "the refused precedent readback is reported, not hidden: {created:?}"
    );
    assert!(
        created.precedent_ids.is_empty(),
        "nothing unread is reported as read"
    );
    let committed = copy
        .reopen()
        .get_decision(created.id)
        .expect("read the decision")
        .expect("the decision committed");
    assert_eq!(
        committed.precedent_ids,
        vec![precedent],
        "its precedent link committed"
    );

    let patch = DecisionPatch {
        tags: Some(vec!["revisited".into()]),
        ..DecisionPatch::default()
    };
    let (updated, copy) =
        with_refused_readback(&template, |store| store.update_decision(precedent, patch));
    let updated = updated.expect("a committed decision update is reported as done");
    let committed = copy
        .reopen()
        .get_decision(precedent)
        .expect("read the decision")
        .expect("the decision exists");
    assert_eq!(
        committed.tags,
        vec!["revisited".to_string()],
        "the update committed"
    );
    assert_eq!(
        updated, committed,
        "the reported decision is the committed one"
    );

    let (superseding, copy) = with_refused_readback(&template, |store| {
        store.supersede_decision(
            precedent,
            decision_input(context_id, intention_id, "use postgres"),
        )
    });
    let superseding = superseding.expect("a committed supersession is reported as done");
    assert!(
        unavailable(&superseding.precedent_readback),
        "{superseding:?}"
    );
    let store = copy.reopen();
    assert_eq!(
        store
            .get_decision(precedent)
            .expect("read the old decision")
            .expect("the old decision exists")
            .status,
        DecisionStatus::Superseded,
        "the supersession committed"
    );
    assert!(
        store
            .get_decision(superseding.id)
            .expect("read the new decision")
            .is_some(),
        "the superseding decision committed"
    );
    drop(store);

    let (archived, copy) = with_refused_readback(&template, |store| {
        store.archive_decision(precedent, "no longer relevant".into())
    });
    let archived = archived.expect("a committed archive is reported as done");
    let committed = copy
        .reopen()
        .get_decision(precedent)
        .expect("read the decision")
        .expect("the decision exists");
    assert_eq!(
        committed.status,
        DecisionStatus::Archived,
        "the archive committed"
    );
    assert_eq!(
        archived, committed,
        "the reported decision is the committed one"
    );
}

#[test]
fn an_intention_write_that_committed_never_reports_failure() {
    // Batch and branch intention updates return Ok once committed.
    let (template, (intention_id, predecessor)) = Template::seed(|store| {
        let context_id = create_context(store);
        let intention_id = create_intention(store, context_id, "ship the importer");
        let predecessor = store
            .history(intention_id)
            .expect("intention history")
            .last()
            .expect("one snapshot")
            .id;
        (intention_id, predecessor)
    });
    let patch = IntentionPatch {
        description: Some("ship the importer this week".into()),
        ..IntentionPatch::default()
    };

    let (batch, copy) = with_refused_readback(&template, |store| {
        store.update_intentions_batch(vec![(intention_id, patch.clone())])
    });
    let batch = batch.expect("a committed batch update is reported as done");
    let committed = copy
        .reopen()
        .get_intention(intention_id)
        .expect("read the intention")
        .expect("the intention exists");
    assert_eq!(
        committed.description, "ship the importer this week",
        "the batch committed"
    );
    assert_eq!(
        batch,
        vec![committed],
        "the reported intention is the committed one"
    );

    let (branch, copy) = with_refused_readback(&template, |store| {
        store.update_intention_on_branch(intention_id, predecessor, patch.clone())
    });
    let branch = branch.expect("a committed branch update is reported as done");
    assert!(
        unavailable(&branch.valid_from_readback),
        "an unread commit id is flagged, never returned as trusted: {branch:?}"
    );
    let successor = copy
        .reopen()
        .history(intention_id)
        .expect("intention history")
        .into_iter()
        .find(|snapshot| snapshot.id != predecessor)
        .expect("the successor snapshot committed");
    assert_eq!(branch.id, successor.id);
    assert_eq!(successor.description, "ship the importer this week");
}

#[test]
fn an_outcome_that_committed_never_reports_failure() {
    // Recording an outcome returns Ok once committed, flagging the unread commit id.
    let (template, (_, _, decision_id)) = seeded_decision();
    let (outcome, copy) = with_refused_readback(&template, |store| {
        store.record_outcome(decision_id, true, Some("it worked".into()))
    });
    let outcome = outcome.expect("a committed outcome is reported as recorded");
    assert!(
        unavailable(&outcome.recorded_at_readback),
        "an unread commit id is flagged, never returned as trusted: {outcome:?}"
    );
    assert_eq!(
        rows(
            &copy.reopen(),
            "SELECT id FROM outcomes WHERE decision_id = $decision_id",
            vec![("decision_id", Value::Uuid(decision_id.as_uuid()))],
        ),
        vec![vec![Value::Uuid(outcome.id.as_uuid())]],
        "the outcome committed"
    );
}

#[test]
fn a_capture_that_committed_never_reports_failure() {
    // Tier-0 and structured capture return Ok once committed, flagging the unread follow-up reads.
    let (template, context_id) = Template::seed(|store| {
        capture_tier0(
            store,
            Tier0CaptureRequest {
                context_label: "committed-capture".into(),
                session_ref: "seed".into(),
                transcript: "User: seed the capture context".into(),
                source: "test".into(),
                chunk_window: 0,
            },
        )
        .expect("seed capture");
        store
            .list_contexts()
            .expect("contexts")
            .into_iter()
            .find(|context| context.name == "committed-capture")
            .expect("capture Context")
            .id
    });

    let (tier0, copy) = with_refused_readback(&template, |store| {
        capture_tier0(
            store,
            Tier0CaptureRequest {
                context_label: "committed-capture".into(),
                session_ref: "probe".into(),
                transcript: "User: we chose rust for the engine core".into(),
                source: "test".into(),
                chunk_window: 0,
            },
        )
    });
    let tier0 = tier0.expect("a committed tier-0 capture is reported as captured");
    assert!(unavailable(&tier0.pending_after_readback), "{tier0:?}");
    assert_eq!(
        tier0.pending_after, 0,
        "an unread count is never reported as read"
    );
    assert_eq!(tier0.digests_written, 1);
    assert_eq!(
        queue_counts(&copy.reopen(), context_id)
            .expect("queue counts")
            .pending,
        2,
        "the captured digest committed"
    );

    let (structured, copy) = with_refused_readback(&template, |store| {
        run_capture(
            store,
            CaptureRequest {
                context_label: "committed-capture".into(),
                transcript: "Goal G_cache: keep the cache local\nEntity E_db: sqlite type=tool"
                    .into(),
                model_available: true,
            },
            &fake_extractor::LabeledTranscriptExtractor,
            &CaptureHarnessConfig::default(),
        )
    });
    let (report, _) = structured.expect("a committed structured capture is reported as captured");
    assert!(unavailable(&report.invalidation_readback), "{report:?}");
    assert!(
        report.invalidations.is_empty(),
        "nothing unread is reported as read"
    );
    assert!(
        !copy
            .reopen()
            .list_entities(ListEntityFilter {
                context_id: Some(context_id),
                ..ListEntityFilter::default()
            })
            .expect("list entities")
            .is_empty(),
        "the structured capture committed"
    );
}

#[test]
fn an_observation_that_committed_never_reports_failure() {
    // Recording an observation returns Ok once committed, flagging the unread invalidations.
    let (template, (context_id, entity_id)) = Template::seed(|store| {
        let context_id = create_context(store);
        let entity_id: EntityId = store
            .create_entity(CreateEntity {
                name: "front door".into(),
                context_id,
                ..CreateEntity::default()
            })
            .expect("create entity")
            .id;
        (context_id, entity_id)
    });
    let observation_id = ObservationId::new_v7();
    let (recorded, copy) = with_refused_readback(&template, |store| {
        store.record_observation(RecordObservation {
            id: observation_id,
            entity_id,
            context_id,
            observation_type: "door-state".into(),
            source: "sensor".into(),
            observed_at: chrono::DateTime::from_timestamp(1_790_000_000, 0)
                .expect("fixed observation time"),
            evidence: vec![EvidenceRef {
                id: EvidenceId::new_v7(),
                kind: EvidenceKind::Text,
                source_ref: "sensor://front-door".into(),
                retention_status: RetentionStatus::RetainedExternal,
                ..EvidenceRef::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
    });
    let recorded = recorded.expect("a committed observation is reported as recorded");
    assert!(unavailable(&recorded.invalidation_readback), "{recorded:?}");
    assert_eq!(recorded.observation_id, observation_id);
    assert!(
        copy.reopen().get_observation(observation_id).is_ok(),
        "the observation committed"
    );
}
