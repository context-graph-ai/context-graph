//! Event-backed import — the schema version constant.

use context_graph::types::CG_SCHEMA_VERSION;

#[test]
fn schema_version_is_twelve() {
    assert_eq!(
        CG_SCHEMA_VERSION, 12,
        "immutable event history, node-local bindings, and placement policy require schema 12"
    );
}
