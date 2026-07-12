//! Fabric first-light — schema 9 → 10 and the old-root migration discipline.
//!
//! At schema 10 the private understanding-queue table is gone (the ledger's computed job state
//! subsumes it); the write-once completion marker is retained. Opening an old schema-9 root with the
//! new binary loses no captured data: it is refused with a typed legacy-schema error while the raw
//! and memory it holds survive on disk, re-derivable under the pre-release re-import discipline.

use std::collections::HashMap;

use context_graph::{CgError, Store, StoreConfig};
use contextdb_core::Value as DbValue;
use contextdb_engine::Database;

use crate::fabric_common::{FabricNode, tier0_request};

const SESSION: &str = "User: Goal G1: survive the upgrade. Assistant: Decision D1: keep the auth module at src/auth.py.";

#[test]
fn fresh_schema_ten_root_carries_no_legacy_queue_state() {
    let node = FabricNode::open("fresh_schema_ten");
    let tables = node.store.sync_database().table_names();
    assert!(
        !tables.iter().any(|table| table == "cg_understanding_queue"),
        "a fresh schema-10 root holds no leftover private understanding-queue state: {tables:?}"
    );
}

#[test]
fn old_schema_nine_root_with_queue_rows_opens_without_data_loss() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("cg.db");

    // Build a schema-9 root holding captured raw/memory (and, at schema 9, queued rows), then force
    // the on-disk schema version to 9 to model an old root under the new binary.
    {
        let store = crate::fake_extractor::open_store_with_lexical_embedder(StoreConfig {
            db_path: path.clone(),
            default_text_embedder: None,
            ..Default::default()
        });
        let report =
            context_graph::tier0::capture_tier0(&store, tier0_request("daily-loop", "s1", SESSION))
                .expect("tier-0 capture");
        assert!(report.digests_written >= 1, "seed captured some raw");
        let db = store.sync_database();
        db.execute(
            "UPDATE cg_schema_version SET version = $version",
            &HashMap::from([("version".to_string(), DbValue::Int64(9))]),
        )
        .expect("force on-disk schema version to 9");
    }

    // The new binary must refuse the old root with a typed legacy-schema error — no auto-migrate, no
    // wipe, no crash.
    let result = Store::open(StoreConfig {
        db_path: path.clone(),
        ..Default::default()
    });
    assert!(
        matches!(
            result,
            Err(CgError::LegacySchemaDetected {
                found_version: 9,
                expected_version: 10,
            })
        ),
        "opening a schema-9 root under the schema-10 binary yields the typed legacy-schema error, not a crash or wipe: {result:?}"
    );

    // The underlying database still holds the captured raw — re-derivable, never lost.
    let raw = Database::open(&path).expect("raw database open");
    let digests = raw.scan("digests", raw.snapshot()).expect("scan digests");
    assert!(
        !digests.is_empty(),
        "already-captured raw survives the refused open"
    );
}
