//! "This store needs a writable reopen" and "this store cannot be read" are different answers, and
//! a consumer has to be able to tell them apart.
//!
//! ## The promise
//!
//! A downstream runtime whose read-only open is refused has exactly two next moves, and they are
//! opposites. If the store's committed image merely needs writable recovery — the last writer died
//! before the storage engine could settle it — the consumer may go on to the WRITABLE
//! existing-only door, recover, and carry on serving. If the store is genuinely unreadable — a
//! permission, a mount, real damage — reaching for a writable handle is the worst available move:
//! it takes a mutating open on a store nobody has diagnosed.
//!
//! The substrate keeps the two apart by type: redb's `RepairAborted` becomes contextdb's
//! `LoadReadImageError::RequiresWriter`, surfaced as `ReadFailureKind::DirectReadRequiresWriter`.
//! The consumer-door conversion preserves that distinction so callers need not parse prose to
//! choose a recovery path.
//!
//! ## What is proved here
//!
//! A store the engine says needs a writable reopen comes back from
//! [`ConsumerReader::open`] as its OWN typed outcome — not
//! [`CgError::StoreUnreadable`], not [`CgError::StoreMissing`] — carrying the store's path and the
//! substrate's own words for the condition. A genuinely damaged store keeps
//! [`CgError::StoreUnreadable`]. The two are different typed answers,
//! not one answer with two strings in it.
//!
//! ## Why the outcome is pinned by behavior rather than by name
//!
//! The assertions compare typed discriminants: recovery-required is distinct from missing and
//! unreadable, carries the path, and carries the engine's reason. The substrate's typed kind
//! establishes the fixture condition independently.
//!
//! ## Why this is arranged end to end rather than over the conversion
//!
//! The conversion (`store::convert::map_consumer_open_err`) is `pub(crate)` and no integration
//! test can call it. The arrangement is contextdb's own damage fixture seam, which makes the
//! defect in the storage engine itself — the same seam two shipped context-graph suites already
//! use — so no marker, sidecar or test-only decoder takes part in the diagnosis. Each arm proves
//! its fixture really produces the engine condition it claims before asking context-graph anything.
//!
//! Every assertion below is about a typed value or bytes on disk. Nothing waits on elapsed time
//! and nothing sleeps.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use context_graph::owner_control::context_graph_read_session_options;
use context_graph::schema::TABLES;
use context_graph::{
    CgError, ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable, CreateContext,
    Store, StoreConfig,
};
use contextdb_core::Error as EngineError;
use contextdb_core::read_contract::ReadFailureKind;
use contextdb_engine::ReadSession;
use contextdb_engine::persistence::read_persistence_test_scaffold::{
    DurableStoreDamage, prepare_durable_store_damage_for_test,
};

/// A declaration over one context-graph table, which is all the open needs: every refusal below is
/// decided before a single row is served.
fn declaration() -> ConsumerDeclaration {
    let spec = TABLES
        .iter()
        .find(|spec| spec.name == "contexts")
        .expect("context-graph publishes a table spec for contexts");
    ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "vigil".to_string(),
            tables: vec![ConsumerTable {
                name: spec.name.to_string(),
                ddl: spec.ddl.to_string(),
            }],
        },
        scope_labels: BTreeSet::new(),
    }
}

fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    }
}

/// A store with a real committed row in it, given up by its owner, then damaged in the shape asked
/// for. The row matters: an empty file could be refused for being empty rather than for its state.
fn crash_dirty_store(db_path: &Path, damage: DurableStoreDamage) {
    let store = Store::open(store_config(db_path.to_path_buf()))
        .expect("open the store the fixture damages");
    store
        .create_context(CreateContext {
            name: format!("crash-dirty-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("commit a row so the damaged file has something to be damaged about");
    drop(store);
    prepare_durable_store_damage_for_test(db_path, damage)
        .expect("leave the store file in the state the arm is about");
}

/// What the SUBSTRATE says about this file, taken on the same read route context-graph's read-only
/// door uses. Every arm takes this first: an assertion about how context-graph reports an engine
/// condition proves nothing until the engine has been shown to produce that condition.
fn engine_refusal(db_path: &Path) -> EngineError {
    match ReadSession::open_with_options(db_path, context_graph_read_session_options()) {
        Ok(_) => panic!(
            "the fixture at {} was readable on the direct route, so it says nothing about how a \
             refusal is reported",
            db_path.display()
        ),
        Err(error) => error,
    }
}

fn assert_engine_kind(error: &EngineError, expected: ReadFailureKind, db_path: &Path) {
    let EngineError::ReadFailure(failure) = error else {
        panic!(
            "the fixture at {} must be refused as a read failure for this arm to mean anything, \
             got: {error:?}",
            db_path.display()
        );
    };
    assert_eq!(
        expected,
        failure.kind(),
        "the fixture must produce the engine condition this arm is about",
    );
}

fn consumer_refusal(db_path: &Path) -> CgError {
    match ConsumerReader::open(db_path, declaration()) {
        Ok(reader) => {
            let _ = reader.close();
            panic!(
                "the read-only door answered instead of refusing the store at {}",
                db_path.display()
            );
        }
        Err(error) => error,
    }
}

// A store whose committed image the engine says needs a writable reopen is told exactly that, by
// type. A consumer can choose recovery without risking a writable open on genuinely damaged data.
#[test]
fn a_store_needing_writable_recovery_is_its_own_typed_outcome() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let db_path = directory.path().join("needs-recovery.db");
    crash_dirty_store(&db_path, DurableStoreDamage::NonMonotonicCommitIndex);

    let engine = engine_refusal(&db_path);
    assert_engine_kind(&engine, ReadFailureKind::DirectReadRequiresWriter, &db_path);
    let engine_reason = engine.to_string();
    let bytes_before = std::fs::read(&db_path).expect("read the store file");

    let refusal = consumer_refusal(&db_path);

    assert!(
        !matches!(refusal, CgError::StoreUnreadable { .. }),
        "a store that needs writable recovery is RECOVERABLE; reporting it as unreadable either \
         costs the consumer a store it could have taken back, or drives it to open a genuinely \
         damaged one writable. Got: {refusal:?}",
    );
    assert!(
        !matches!(refusal, CgError::StoreMissing { .. }),
        "the store is right there; reporting it missing sends an operator to restore a backup \
         over a store that only needed recovering. Got: {refusal:?}",
    );

    let rendered = format!("{refusal:?}");
    assert!(
        rendered.contains(&db_path.display().to_string()),
        "the outcome must name the store it is about, so the consumer knows which one to recover; \
         got: {rendered}",
    );
    assert!(
        rendered.contains(&engine_reason),
        "the outcome must carry the substrate's own words for the condition rather than a phrase \
         context-graph invented; the engine said {engine_reason:?}, the outcome rendered as \
         {rendered}",
    );

    assert_eq!(
        bytes_before,
        std::fs::read(&db_path).expect("read the store file again"),
        "the refused read-only open must leave the file exactly as it found it: no recovery, no \
         hydration, no lock",
    );
}

// A genuinely unreadable store is reported as unreadable; the recoverable case never covers damage.
#[test]
fn a_genuinely_damaged_store_is_still_reported_unreadable() {
    let directory = tempfile::tempdir().expect("scratch directory");
    let db_path = directory.path().join("damaged.db");
    crash_dirty_store(&db_path, DurableStoreDamage::MalformedRecord);

    let engine = engine_refusal(&db_path);
    assert!(
        !matches!(
            &engine,
            EngineError::ReadFailure(failure)
                if failure.kind() == ReadFailureKind::DirectReadRequiresWriter
        ),
        "this arm is about a store the engine does NOT offer recovery for, got: {engine:?}",
    );
    let bytes_before = std::fs::read(&db_path).expect("read the store file");

    let refusal = consumer_refusal(&db_path);

    let CgError::StoreUnreadable { path, reason } = &refusal else {
        panic!(
            "a genuinely damaged store must give the unreadable answer, got: \
             {refusal:?}"
        );
    };
    assert_eq!(
        &db_path, path,
        "the refusal must name the store it could not read",
    );
    assert!(
        !reason.is_empty(),
        "the refusal must say what the substrate found, because the operator's next step depends \
         entirely on which condition it was",
    );
    assert_eq!(
        bytes_before,
        std::fs::read(&db_path).expect("read the store file again"),
        "the refused read-only open must leave the damaged file exactly as it found it",
    );
}

// A consumer branching on the typed answer distinguishes recoverable and damaged stores without
// parsing prose.
#[test]
fn recoverable_and_unreadable_are_two_typed_answers_not_one() {
    let directory = tempfile::tempdir().expect("scratch directory");

    let recoverable_path = directory.path().join("needs-recovery.db");
    crash_dirty_store(
        &recoverable_path,
        DurableStoreDamage::NonMonotonicCommitIndex,
    );
    assert_engine_kind(
        &engine_refusal(&recoverable_path),
        ReadFailureKind::DirectReadRequiresWriter,
        &recoverable_path,
    );

    let damaged_path = directory.path().join("damaged.db");
    crash_dirty_store(&damaged_path, DurableStoreDamage::MalformedRecord);
    let _ = engine_refusal(&damaged_path);

    let recoverable = consumer_refusal(&recoverable_path);
    let damaged = consumer_refusal(&damaged_path);

    assert_ne!(
        std::mem::discriminant(&recoverable),
        std::mem::discriminant(&damaged),
        "a store that can be recovered and a store that cannot must be different typed answers, \
         not the same one with different prose in it: recoverable {recoverable:?}, damaged \
         {damaged:?}",
    );
}
