//! A fresh Store authors its whole startup schema as one schema commit.
//!
//! A peer replays each schema commit over its whole schema. Every startup declaration -- the plain tables, the tables carrying a vector column,
//! a rank policy or DAG edges, every index, the subscription event types and the store's triggers
//! -- becomes visible together or not at all, and survives a restart.
//!
//! That the same one commit reaches a SECOND node in author order is the other half of the same
//! promise, and it lives in `a_second_node_receives_the_whole_startup_schema.rs`: it needs a real
//! sync fixture and two live stores, which is a different order of cost from opening a file, and a
//! binary's frozen wall-clock ceiling should describe one shape rather than their sum.

#[path = "common/startup_schema_log.rs"]
mod startup_schema_log;

use context_graph::{Store, StoreConfig};

use crate::startup_schema_log::{
    STORE_OPEN_SCHEMA_COMMITS, assert_whole_startup_schema_is_one_commit, schema_log,
};

#[test]
fn a_fresh_store_installs_its_whole_startup_schema_in_one_commit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("fresh.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open a fresh store");
    let first = assert_whole_startup_schema_is_one_commit(&store, "fresh open");
    assert_eq!(
        schema_log(&store).statements_per_commit.len(),
        STORE_OPEN_SCHEMA_COMMITS,
        "a fresh store's open authors exactly this many schema commits"
    );
    let authored = schema_log(&store).author_order;
    drop(store);

    let reopened = Store::open(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen the store");
    let after_restart = assert_whole_startup_schema_is_one_commit(&reopened, "after restart");
    assert_eq!(
        schema_log(&reopened).statements_per_commit.len(),
        STORE_OPEN_SCHEMA_COMMITS,
        "a restart authors no further schema commit"
    );
    assert_eq!(
        after_restart, first,
        "a restart re-reads the startup schema commit instead of authoring another"
    );
    assert_eq!(
        schema_log(&reopened).author_order,
        authored,
        "a restart preserves the order the startup schema was authored in"
    );
}
