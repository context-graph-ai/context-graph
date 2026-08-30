//! What a refusal SAYS decides what an operator does next, so a refusal must never attribute
//! silence to an actor that is not there.
//!
//! The contract, in the two directions it can be broken:
//!
//! - Nobody is holding the store. The operator must read "nothing is holding this store" and go
//!   take it. A message that says the owner of the store refused the request tells them a live
//!   process turned them down — so they go hunting for a process that does not exist, and leave
//!   the store they were free to open sitting idle.
//! - A writer IS holding the store. The operator must NOT be told the store is free, and a caller
//!   whose fallback is to open the store itself must not take that route: `owner_absent()` false is
//!   the whole of that instruction. Opening the store behind a live writer is exactly what the
//!   owner plane exists to prevent.
//!
//! ## Public contract this file proves
//!
//! - Every no-owner answer from `context_graph::owner_control::request_owner` — an idle store, a
//!   path with no store, a store whose file a direct read refuses — renders without attributing
//!   the outcome to a live owner.
//! - A store a live writer holds never answers owner-absent, whatever that writer does with the
//!   frame, and its refusal names the store the caller asked about.
//!
//! The live-owner arm establishes by STATE that an owner really is there — contextdb's own owner
//! report says it is serving — before pinning what the caller is told. Nothing here measures
//! elapsed time and nothing sleeps.

use std::collections::BTreeMap;
use std::path::PathBuf;

use context_graph::owner_control::{
    OwnerControlError, context_graph_read_session_options, request_owner,
};
use context_graph::{CreateContext, Store, StoreConfig};
use contextdb_core::read_contract::OwnerServingState;
use contextdb_engine::ReadSession;
use contextdb_engine::persistence::read_persistence_test_scaffold::{
    DurableStoreDamage, prepare_durable_store_damage_for_test,
};

/// Wording that names a live actor as the source of the outcome. None of it may appear in an
/// answer produced when no owner is there at all.
const ATTRIBUTES_THE_OUTCOME_TO_A_LIVE_OWNER: [&str; 3] = [
    "refused",
    "the owner of the store",
    "did not answer inside the time",
];

fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    }
}

fn assert_names_no_live_owner(refusal: &OwnerControlError, fixture: &str) {
    let message = refusal.to_string();
    for wording in ATTRIBUTES_THE_OUTCOME_TO_A_LIVE_OWNER {
        assert!(
            !message.contains(wording),
            "{fixture}: no owner is alive, so the answer must not say `{wording}` and send an \
             operator hunting for a process that is not there — got `{message}` from {refusal:?}"
        );
    }
    assert!(
        refusal.owner_absent(),
        "{fixture}: nothing holds this store, so the caller is told it may open it itself, got \
         {refusal:?}"
    );
}

#[test]
fn an_idle_store_answer_never_says_a_live_owner_turned_the_caller_down() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("idle.db");
    drop(Store::open(store_config(db_path.clone())).expect("open and give up an idle store"));

    let refusal = request_owner(&db_path, "status\n").expect_err("nobody holds the idle store");
    assert_names_no_live_owner(&refusal, "a store its owner gave up");
    assert!(
        refusal.to_string().contains(&db_path.display().to_string()),
        "the answer names the store the caller asked about: {refusal:?}"
    );
}

#[test]
fn a_path_with_no_store_answer_never_says_a_live_owner_turned_the_caller_down() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("never-created.db");

    let refusal =
        request_owner(&db_path, "status\n").expect_err("no store, so no owner, so no answer");
    assert_names_no_live_owner(&refusal, "a path that never had a store");
}

#[test]
fn a_crash_dirty_store_answer_never_says_a_live_owner_turned_the_caller_down() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("crash-dirty.db");
    let store = Store::open(store_config(db_path.clone())).expect("open the store to damage");
    store
        .create_context(CreateContext {
            name: format!("wording-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("commit a row before the damage");
    drop(store);
    prepare_durable_store_damage_for_test(&db_path, DurableStoreDamage::NonMonotonicCommitIndex)
        .expect("leave the store in a shape a direct read refuses");

    let refusal =
        request_owner(&db_path, "status\n").expect_err("nobody holds the crash-dirty store");
    // This is the arm the misattribution actually reached an operator through: a file condition
    // dressed up as a live owner's refusal, on the very store they most need a plain answer about.
    assert_names_no_live_owner(&refusal, "a store whose file needs a writable repair");
}

#[test]
fn a_store_a_live_writer_holds_is_never_reported_as_free_to_open() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("held.db");

    // A plain writable open: a live owner with no context-graph control handler behind it. It is
    // still a writer holding the store, which is the fact the caller must be told.
    let _holder =
        Store::open(store_config(db_path.clone())).expect("a live writer takes the store");

    let serving = ReadSession::owner_report(&db_path, context_graph_read_session_options())
        .expect("the live writer reports on itself")
        .status
        .state;
    assert_eq!(
        serving,
        OwnerServingState::Serving,
        "the fixture must really be a live owner before anything is claimed about what it answers"
    );

    let refusal = request_owner(&db_path, "status\n")
        .expect_err("this owner serves no context-graph control frames");

    assert!(
        !refusal.owner_absent(),
        "a writer is holding this store, so the caller must never be told it may open it itself — \
         doing so is refused, and telling them otherwise turns a live-owner refusal into a failed \
         open: got {refusal:?}"
    );
    assert!(
        !matches!(
            refusal,
            OwnerControlError::OwnerNotRunning { .. } | OwnerControlError::NoStore { .. }
        ),
        "neither `no process is holding the store` nor `there is no store` is true here: {refusal:?}"
    );
    assert!(
        refusal.to_string().contains(&db_path.display().to_string()),
        "the refusal names the store the caller asked about: {refusal:?}"
    );
    assert_eq!(
        refusal.store_path(),
        db_path.as_path(),
        "the refusal carries the store path a caller renders and acts on"
    );
}
