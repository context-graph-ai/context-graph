//! Fabric first-light — the schema version constant.

use context_graph::types::CG_SCHEMA_VERSION;

#[test]
fn schema_version_is_ten() {
    assert_eq!(
        CG_SCHEMA_VERSION, 10,
        "the fabric slice bumps the cg schema version to 10 (understanding queue removed)"
    );
}
