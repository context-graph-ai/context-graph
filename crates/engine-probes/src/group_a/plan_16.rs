//! Plan 16 cg-digest-pipeline — engine-probe sweep.
//!
//! See `.claude/plans/context-graph/r0/track1-enumeration/16-cg-digest-pipeline-primitives.md`.

use contextdb_core::{Error, Value, VectorIndexRef};
use contextdb_engine::Database;
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;
use uuid::Uuid;

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn vals(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn open_digest_schema() -> Database {
    let db = Database::open_memory();
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE digests (
            id UUID PRIMARY KEY,
            source_observation_id UUID,
            ingester_kind TEXT,
            source_uri TEXT,
            source_hash TEXT UNIQUE,
            vector_text VECTOR(4),
            UNIQUE (source_observation_id, ingester_kind)
        ) IMMUTABLE",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE ingest_runs (
            id UUID PRIMARY KEY,
            mode TEXT,
            started_at INTEGER
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE edges (
            id UUID PRIMARY KEY,
            source_id UUID,
            target_id UUID,
            edge_type TEXT
        )",
        &p,
    )
    .unwrap();
    db
}

// ---------- Single-primitive ----------

/// 16 single-primitive: digests is IMMUTABLE — UPDATE rejected.
#[test]
fn digests_immutable_update_rejected() {
    let db = open_digest_schema();
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO digests (id, source_observation_id, ingester_kind, source_hash, vector_text) VALUES ($id, $sid, 'text', 'hash1', $v)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("sid", Value::Uuid(Uuid::new_v4())),
            ("v", Value::Vector(vec![1.0, 0.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();

    let result = db.execute(
        "UPDATE digests SET ingester_kind = 'modified' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    assert!(matches!(result, Err(Error::ImmutableTable(_))));
}

/// 16 single-primitive: composite UNIQUE on (source_observation_id, ingester_kind)
/// dedups concurrent digest creation for the same source.
#[test]
fn digests_composite_unique_source_observation_ingester_kind() {
    let db = open_digest_schema();
    let source_obs = Uuid::new_v4();
    db.execute(
        "INSERT INTO digests (id, source_observation_id, ingester_kind, source_hash, vector_text) VALUES ($id, $sid, 'text', 'hash1', $v)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("sid", Value::Uuid(source_obs)),
            ("v", Value::Vector(vec![1.0, 0.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();
    // Second insert with same (source_observation_id, ingester_kind) but
    // different source_hash → still must collapse on the composite UNIQUE.
    db.execute(
        "INSERT INTO digests (id, source_observation_id, ingester_kind, source_hash, vector_text) VALUES ($id, $sid, 'text', 'hash2', $v)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("sid", Value::Uuid(source_obs)),
            ("v", Value::Vector(vec![0.0, 1.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();

    let rows = db.scan("digests", db.snapshot()).unwrap();
    assert_eq!(
        rows.len(),
        1,
        "duplicate (source_observation_id, ingester_kind) must dedup to one digest"
    );

    // Different ingester_kind allows a second digest.
    db.execute(
        "INSERT INTO digests (id, source_observation_id, ingester_kind, source_hash, vector_text) VALUES ($id, $sid, 'image', 'hash3', $v)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("sid", Value::Uuid(source_obs)),
            ("v", Value::Vector(vec![0.0, 0.0, 1.0, 0.0])),
        ]),
    )
    .unwrap();
    let rows = db.scan("digests", db.snapshot()).unwrap();
    assert_eq!(rows.len(), 2);
}

/// 16 single-primitive: source_hash UNIQUE dedups by content fingerprint.
#[test]
fn digests_source_hash_unique() {
    let db = open_digest_schema();
    db.execute(
        "INSERT INTO digests (id, source_observation_id, ingester_kind, source_hash, vector_text) VALUES ($id, $sid, 'text', 'fp_abc', $v)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("sid", Value::Uuid(Uuid::new_v4())),
            ("v", Value::Vector(vec![1.0, 0.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();
    db.execute(
        "INSERT INTO digests (id, source_observation_id, ingester_kind, source_hash, vector_text) VALUES ($id, $sid, 'text', 'fp_abc', $v)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("sid", Value::Uuid(Uuid::new_v4())),
            ("v", Value::Vector(vec![0.0, 1.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();
    let rows = db.scan("digests", db.snapshot()).unwrap();
    assert_eq!(rows.len(), 1, "duplicate source_hash must dedup");
}

// ---------- Multi-primitive combinations ----------

/// 16 multi-primitive: atomic digest extraction tx — INSERT digest +
/// INSERT EXTRACTED_FROM edge + insert vector. Rollback drops all.
#[test]
fn digest_extraction_atomic_rollback_drops_all() {
    let db = open_digest_schema();
    let digest_id = Uuid::new_v4();
    let source_obs = Uuid::new_v4();

    let tx = db.begin();
    let row_id = db
        .insert_row(
            tx,
            "digests",
            vals(vec![
                ("id", Value::Uuid(digest_id)),
                ("source_observation_id", Value::Uuid(source_obs)),
                ("ingester_kind", Value::Text("text".into())),
                ("source_hash", Value::Text("fp_xyz".into())),
            ]),
        )
        .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("digests", "vector_text"),
        row_id,
        vec![1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    db.insert_edge(
        tx,
        digest_id,
        source_obs,
        "EXTRACTED_FROM".into(),
        HashMap::new(),
    )
    .unwrap();
    db.rollback(tx).unwrap();

    let snap = db.snapshot();
    assert_eq!(db.scan("digests", snap).unwrap().len(), 0);
    assert_eq!(
        db.query_vector(
            VectorIndexRef::new("digests", "vector_text"),
            &[1.0, 0.0, 0.0, 0.0],
            10,
            None,
            snap,
        )
        .unwrap()
        .len(),
        0
    );
    let bfs = db
        .query_bfs(
            digest_id,
            Some(&["EXTRACTED_FROM".to_string()]),
            contextdb_core::Direction::Outgoing,
            1,
            snap,
        )
        .unwrap();
    assert_eq!(bfs.nodes.len(), 0);
}

// ---------- Concurrent-semantics probes ----------

/// C-DG1 concurrent digest creation for same source: callers attempt
/// to create a digest for the same (source_observation_id, ingester_kind).
/// Engine composite UNIQUE must dedup. Probe runs multiple rounds.
#[test]
fn concurrent_digest_for_same_source_dedups() {
    const ROUNDS: usize = 20;
    const N: usize = 6;
    let mut max_rows = 0usize;
    for r in 0..ROUNDS {
        let db = Arc::new(open_digest_schema());
        let source_obs = Uuid::new_v4();
        let barrier = Arc::new(Barrier::new(N));
        let mut handles = vec![];
        for i in 0..N {
            let db = db.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || -> u64 {
                barrier.wait();
                db.execute(
                    "INSERT INTO digests (id, source_observation_id, ingester_kind, source_hash, vector_text) VALUES ($id, $sid, 'text', $h, $v)",
                    &params(vec![
                        ("id", Value::Uuid(Uuid::new_v4())),
                        ("sid", Value::Uuid(source_obs)),
                        ("h", Value::Text(format!("fp_r{r}_{i}"))),
                        ("v", Value::Vector(vec![1.0, 0.0, 0.0, 0.0])),
                    ]),
                )
                .map(|qr| qr.rows_affected)
                .unwrap_or(0)
            }));
        }
        let _total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let rows = db.scan("digests", db.snapshot()).unwrap();
        if rows.len() > max_rows {
            max_rows = rows.len();
        }
        if max_rows > 1 {
            break;
        }
    }
    assert_eq!(
        max_rows, 1,
        "concurrent digest INSERT against same composite UNIQUE must collapse to one row; observed {max_rows} rows in best round"
    );
}

/// C-DG2 IMMUTABLE on digests UPDATE/DELETE refused via execute_in_tx
/// (bypass-attempt probe).
#[test]
fn digests_immutable_bypass_via_execute_in_tx_refused() {
    let db = open_digest_schema();
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO digests (id, source_observation_id, ingester_kind, source_hash, vector_text) VALUES ($id, $sid, 'text', 'h', $v)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("sid", Value::Uuid(Uuid::new_v4())),
            ("v", Value::Vector(vec![1.0, 0.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();
    let tx = db.begin();
    let upd = db.execute_in_tx(
        tx,
        "UPDATE digests SET source_hash = 'tampered' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    let del = db.execute_in_tx(
        tx,
        "DELETE FROM digests WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    let _ = db.rollback(tx);
    assert!(matches!(upd, Err(Error::ImmutableTable(_))));
    assert!(matches!(del, Err(Error::ImmutableTable(_))));
}

/// C-DG3 digest extraction rolls back on partial failure: vector insert
/// fails (wrong dim) → digest row + extracted edge roll back together.
#[test]
fn digest_extraction_vector_failure_rolls_back_all() {
    let db = open_digest_schema();
    let digest_id = Uuid::new_v4();
    let source_obs = Uuid::new_v4();

    let tx = db.begin();
    let row_id = db
        .insert_row(
            tx,
            "digests",
            vals(vec![
                ("id", Value::Uuid(digest_id)),
                ("source_observation_id", Value::Uuid(source_obs)),
                ("ingester_kind", Value::Text("text".into())),
                ("source_hash", Value::Text("fp_xyz".into())),
            ]),
        )
        .unwrap();
    db.insert_edge(
        tx,
        digest_id,
        source_obs,
        "EXTRACTED_FROM".into(),
        HashMap::new(),
    )
    .unwrap();
    let bad = db.insert_vector(
        tx,
        VectorIndexRef::new("digests", "vector_text"),
        row_id,
        vec![1.0; 99],
    );
    assert!(matches!(bad, Err(Error::VectorIndexDimensionMismatch { .. })));
    let _ = db.rollback(tx);

    let snap = db.snapshot();
    assert_eq!(
        db.scan("digests", snap).unwrap().len(),
        0,
        "digest row must roll back when vector insert fails"
    );
    let bfs = db
        .query_bfs(
            digest_id,
            Some(&["EXTRACTED_FROM".to_string()]),
            contextdb_core::Direction::Outgoing,
            1,
            snap,
        )
        .unwrap();
    assert_eq!(bfs.nodes.len(), 0, "edge must roll back");
}

// ---------- BLOCKED entries ----------
// C-DG4 RawMediaNotStored is a cg-app-layer policy on EvidenceRef
// dispatch. The engine has no `evidence_kind` CHECK primitive; cg-app
// refuses raw media at API entry. Engine probes cannot exercise this
// without crossing into cg surface.
//
// C-DG5 greenfield rebuild DROP digests.embedding (384) + CREATE
// digests.vector_text (768) — single-tx vs two-step is exercised by
// reindex_atomic_delete_insert_same_tx in plan_11. Probe-coverage there
// applies; not duplicated here.
//
// C-DG6 four-way classification stable under reorder — cg-app-layer
// classifier; not engine primitive.
//
// C-DG7 ingest_dry_run no commits — cg-app-layer flag controlling
// whether the cg writer commits or rolls back. Probe-coverage:
// digest_extraction_atomic_rollback_drops_all asserts that an explicit
// rollback drops all of digests + vector + edge. Dry-run = rollback.
