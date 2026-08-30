//! A store whose file a direct read refuses still answers the plain fact that nobody owns it.
//!
//! The contract: an administrative caller asks the RUNNING owner of a store. When no process owns
//! it the caller is told exactly that — and the condition of the file behind the path never enters
//! the answer, because the file is never consulted. An operator whose process died mid-write and
//! who then asks "is anything holding this store?" must be told "nothing is", not handed a
//! diagnosis of the file; the diagnosis belongs to the writable repair they run next, and a caller
//! that gets a file-condition error here cannot tell whether it is safe to take the store.
//!
//! ## Public contract this file proves
//!
//! - `context_graph::owner_control::request_owner` on a crash-dirty store answers
//!   `OwnerControlError::OwnerNotRunning { store_path }` with `owner_absent()` true — the same
//!   answer an undamaged idle store gives.
//! - The damaged file is byte-for-byte unchanged afterwards: no hydration, no repair, no lock.
//!
//! What makes "the file was never consulted" provable rather than asserted: each fixture here is a
//! store the DIRECT read route genuinely refuses, and each arm proves that first. If consulting the
//! file produces a different typed answer and the owner ask does not produce it, the ask did not
//! consult the file. The damage is made in the storage engine itself by contextdb's own fixture
//! seam, so no marker or sidecar participates in the diagnosis.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use context_graph::owner_control::{
    OwnerControlError, context_graph_read_session_options, request_owner,
};
use context_graph::{CreateContext, Store, StoreConfig};
use contextdb_engine::ReadSession;
use contextdb_engine::persistence::read_persistence_test_scaffold::{
    DurableStoreDamage, prepare_durable_store_damage_for_test,
};

fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    }
}

/// A store with real committed rows in it, given up by its owner, and then damaged in the shape
/// asked for. The rows matter: an empty file could be refused for being empty rather than damaged.
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
        .expect("leave the store file in a shape a direct read refuses");
}

/// The control every arm runs first: reading this file directly really does fail. Without it, an
/// owner-absent answer proves nothing, because an undamaged file would answer the same.
fn direct_read_refuses(db_path: &Path, damage: DurableStoreDamage) {
    match ReadSession::open_with_options(db_path, context_graph_read_session_options()) {
        Err(error) => {
            let stated = format!("{error:?}");
            assert!(
                !stated.is_empty(),
                "the direct read refused the {damage:?} fixture and said why"
            );
        }
        Ok(_) => panic!(
            "the {damage:?} fixture was readable on the direct route, so it proves nothing about \
             an ask that never looks at the file"
        ),
    }
}

/// The whole arm, run against one shape of damage.
fn a_damaged_store_answers_owner_absent(damage: DurableStoreDamage) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("crash-dirty.db");
    crash_dirty_store(&db_path, damage);
    direct_read_refuses(&db_path, damage);

    let bytes_before = std::fs::read(&db_path).expect("read the damaged store file");

    let refusal = request_owner(&db_path, "status\n")
        .expect_err("nothing is holding the damaged store, so nothing can answer for it");

    assert!(
        matches!(&refusal, OwnerControlError::OwnerNotRunning { store_path } if store_path == &db_path),
        "a {damage:?} store nobody holds is told nobody holds it, not what its file's condition \
         is, got {refusal:?}"
    );
    assert!(
        refusal.owner_absent(),
        "the store is free for the caller to take and repair; `owner_absent` is what says so"
    );
    assert_eq!(
        std::fs::read(&db_path).expect("read the damaged store file again"),
        bytes_before,
        "the ask neither hydrated, repaired, nor locked the damaged file: it never opened it"
    );
}

#[test]
fn a_store_whose_commit_index_went_backwards_still_answers_that_no_owner_is_running() {
    a_damaged_store_answers_owner_absent(DurableStoreDamage::NonMonotonicCommitIndex);
}

#[test]
fn a_store_whose_records_are_malformed_still_answers_that_no_owner_is_running() {
    a_damaged_store_answers_owner_absent(DurableStoreDamage::MalformedRecord);
}

#[test]
fn a_damaged_store_and_an_idle_one_are_told_exactly_the_same_thing() {
    let dir = tempfile::tempdir().expect("tempdir");

    let idle_path = dir.path().join("idle.db");
    let idle_store = Store::open(store_config(idle_path.clone())).expect("open the idle store");
    drop(idle_store);

    let damaged_path = dir.path().join("damaged.db");
    crash_dirty_store(&damaged_path, DurableStoreDamage::NonMonotonicCommitIndex);
    direct_read_refuses(&damaged_path, DurableStoreDamage::NonMonotonicCommitIndex);

    let idle = request_owner(&idle_path, "status\n").expect_err("nobody holds the idle store");
    let damaged =
        request_owner(&damaged_path, "status\n").expect_err("nobody holds the damaged store");

    assert!(
        matches!(idle, OwnerControlError::OwnerNotRunning { .. })
            && matches!(damaged, OwnerControlError::OwnerNotRunning { .. }),
        "the file's condition is not part of this answer, so both stores get the same one: idle \
         {idle:?}, damaged {damaged:?}"
    );
    assert_eq!(
        std::mem::discriminant(&idle),
        std::mem::discriminant(&damaged),
        "an operator must not have to read two different refusals to learn the one fact they \
         asked for — that nothing is holding the store"
    );
}
