//! Fabric first-light — the schema version constant.

use context_graph::types::CG_SCHEMA_VERSION;

#[test]
fn schema_version_is_eleven() {
    assert_eq!(
        CG_SCHEMA_VERSION, 11,
        "immutable intention-history heads and links require schema 11"
    );
}
