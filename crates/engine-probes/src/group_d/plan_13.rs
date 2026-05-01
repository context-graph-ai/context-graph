//! Probes for plan 13 — cg-server-signature-detect.
//!
//! Plan class: primitive-owner with partial blockage on §6 HandleScopeConstraint
//! (engine refusal of Signature INSERT from edge handle is server-side and
//! pending). The unblocked surface — composite UNIQUE on signature identity,
//! `(table, column)` vector index isolation, JSON scope_labels filter,
//! atomic DELETE on space-upgrade, cluster-aggregate persistence,
//! modality-agnostic clustering within a single embedding space — is
//! probed below.
//!
//! Engine source cited:
//! - `contextdb/crates/contextdb-engine/src/database.rs::insert_row`
//!   (line ~1045) — coerce, validate, commit-time atomicity.
//! - `contextdb/crates/contextdb-engine/src/database.rs::semantic_search`
//!   (line ~2293) — vector index keyed by `(table, column)` per
//!   NamedVectorIndexes (gap §14, RELEASED).
//! - `contextdb/crates/contextdb-vector/src/store.rs` — per-index dimension
//!   + scoped state via `VectorIndexRef`.

use contextdb_core::error::Error;
use contextdb_core::types::{RowId, TxId, Value, VectorIndexRef};
use contextdb_engine::{Database, SemanticQuery};
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;

fn make_db() -> Arc<Database> {
    let db = Arc::new(Database::open_memory());
    db.execute(
        "CREATE TABLE signatures (\n\
            id UUID PRIMARY KEY,\n\
            embedding_space_id TEXT NOT NULL,\n\
            signature_kind TEXT NOT NULL,\n\
            scope_labels JSON,\n\
            tenant_id TEXT NOT NULL,\n\
            member_count INTEGER NOT NULL,\n\
            variance REAL,\n\
            last_matched_at TXID,\n\
            centroid VECTOR(4),\n\
            UNIQUE (embedding_space_id, signature_kind)\n\
        )",
        &HashMap::new(),
    )
    .expect("create signatures");
    db.execute(
        "CREATE TABLE observations (\n\
            id UUID PRIMARY KEY,\n\
            embedding_space_id TEXT NOT NULL,\n\
            evidence_kind TEXT NOT NULL,\n\
            tenant_id TEXT NOT NULL,\n\
            embedding VECTOR(4)\n\
        )",
        &HashMap::new(),
    )
    .expect("create observations");
    db
}

fn insert_signature(
    db: &Database,
    space: &str,
    kind: &str,
    scope: serde_json::Value,
    tenant: &str,
    centroid: Vec<f32>,
) -> Result<RowId, Error> {
    let id = uuid::Uuid::new_v4();
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert("embedding_space_id".into(), Value::Text(space.into()));
    row.insert("signature_kind".into(), Value::Text(kind.into()));
    row.insert("scope_labels".into(), Value::Json(scope));
    row.insert("tenant_id".into(), Value::Text(tenant.into()));
    row.insert("member_count".into(), Value::Int64(10));
    row.insert("variance".into(), Value::Float64(0.05));
    row.insert("last_matched_at".into(), Value::TxId(TxId(0)));
    row.insert("centroid".into(), Value::Vector(centroid.clone()));
    let row_id = match db.insert_row(tx, "signatures", row) {
        Ok(r) => r,
        Err(e) => {
            let _ = db.rollback(tx);
            return Err(e);
        }
    };
    if let Err(e) = db.insert_vector(
        tx,
        VectorIndexRef::new("signatures", "centroid"),
        row_id,
        centroid,
    ) {
        let _ = db.rollback(tx);
        return Err(e);
    }
    db.commit(tx)?;
    Ok(row_id)
}

fn insert_observation(db: &Database, space: &str, kind: &str, tenant: &str, vec: Vec<f32>) {
    let id = uuid::Uuid::new_v4();
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert("embedding_space_id".into(), Value::Text(space.into()));
    row.insert("evidence_kind".into(), Value::Text(kind.into()));
    row.insert("tenant_id".into(), Value::Text(tenant.into()));
    row.insert("embedding".into(), Value::Vector(vec.clone()));
    let row_id = db.insert_row(tx, "observations", row).expect("insert obs");
    db.insert_vector(
        tx,
        VectorIndexRef::new("observations", "embedding"),
        row_id,
        vec,
    )
    .expect("insert vec");
    db.commit(tx).expect("commit");
}

/// C-SD1: concurrent signature creation race on UNIQUE composite.
/// Loser MUST see UniqueViolation; never silent drop, never both succeed.
#[test]
fn concurrent_signature_creation_unique_composite_one_winner() {
    let db = make_db();
    let n_pairs = 16;
    let pairs: Vec<(String, String)> = (0..n_pairs)
        .map(|i| (format!("space-{}", i), format!("kind-{}", i)))
        .collect();

    let barrier = Arc::new(Barrier::new(4));
    let mut handles = vec![];
    for tid in 0..4 {
        let db_t = db.clone();
        let pairs_t = pairs.clone();
        let bar = barrier.clone();
        handles.push(thread::spawn(move || {
            bar.wait();
            let mut wins = 0u64;
            let mut uvs = 0u64;
            let mut others = 0u64;
            for (space, kind) in &pairs_t {
                let centroid = vec![1.0_f32, tid as f32 * 0.01, 0.0, 0.0];
                match insert_signature(&db_t, space, kind, serde_json::json!({}), "tenant-x", centroid) {
                    Ok(_) => wins += 1,
                    Err(Error::UniqueViolation { .. }) => uvs += 1,
                    Err(_) => others += 1,
                }
            }
            (wins, uvs, others)
        }));
    }
    let mut tw = 0u64;
    let mut tu = 0u64;
    let mut to = 0u64;
    for h in handles {
        let (w, u, o) = h.join().expect("worker");
        tw += w;
        tu += u;
        to += o;
    }
    assert_eq!(tw as usize, n_pairs, "expected {} winners; got {}", n_pairs, tw);
    assert_eq!(to, 0, "non-UV errors: {}", to);
    assert_eq!(tu as usize, 4 * n_pairs - n_pairs);

    let r = db
        .execute(
            "SELECT embedding_space_id FROM signatures",
            &HashMap::new(),
        )
        .expect("scan");
    assert_eq!(r.rows.len(), n_pairs);
}

/// C-SD2: signatures.centroid index disjoint from observations.embedding index.
/// Same query vector returns different rows from each — `(table, col)` isolation.
#[test]
fn signature_match_uses_centroid_index_not_observation_index() {
    let db = make_db();
    insert_signature(
        &db,
        "space-A",
        "face-cluster",
        serde_json::json!({}),
        "tenant-x",
        vec![1.0, 0.0, 0.0, 0.0],
    )
    .expect("sig");
    insert_signature(
        &db,
        "space-A",
        "voice-cluster",
        serde_json::json!({}),
        "tenant-x",
        vec![0.0, 1.0, 0.0, 0.0],
    )
    .expect("sig");
    insert_observation(&db, "space-A", "image", "tenant-x", vec![1.0, 0.0, 0.0, 0.0]);
    insert_observation(&db, "space-A", "audio", "tenant-x", vec![0.0, 1.0, 0.0, 0.0]);

    let r_sigs = db
        .semantic_search(SemanticQuery::new(
            "signatures",
            "centroid",
            vec![1.0, 0.0, 0.0, 0.0],
            5,
        ))
        .expect("sig search");
    assert_eq!(r_sigs.len(), 2);
    for sr in &r_sigs {
        assert!(sr.values.contains_key("signature_kind"));
    }

    let r_obs = db
        .semantic_search(SemanticQuery::new(
            "observations",
            "embedding",
            vec![1.0, 0.0, 0.0, 0.0],
            5,
        ))
        .expect("obs search");
    assert_eq!(r_obs.len(), 2);
    for sr in &r_obs {
        assert!(sr.values.contains_key("evidence_kind"));
    }

    let sig_uuids: std::collections::HashSet<uuid::Uuid> = r_sigs
        .iter()
        .map(|sr| *sr.values.get("id").and_then(|v| v.as_uuid()).expect("id"))
        .collect();
    let obs_uuids: std::collections::HashSet<uuid::Uuid> = r_obs
        .iter()
        .map(|sr| *sr.values.get("id").and_then(|v| v.as_uuid()).expect("id"))
        .collect();
    assert!(sig_uuids.is_disjoint(&obs_uuids), "indexes leak across (table, col)");
}

/// Unknown vector index returns typed UnknownVectorIndex (not silent fallback).
#[test]
fn unknown_vector_index_query_returns_typed_error() {
    let db = make_db();
    let result = db.query_vector(
        VectorIndexRef::new("signatures", "nonexistent_column"),
        &[1.0, 0.0, 0.0, 0.0],
        5,
        None,
        db.snapshot(),
    );
    match result {
        Err(Error::UnknownVectorIndex { index }) => {
            assert_eq!(index.table, "signatures");
            assert_eq!(index.column, "nonexistent_column");
        }
        Ok(_) => panic!("query against unknown index returned Ok"),
        Err(other) => panic!("expected UnknownVectorIndex; got {:?}", other),
    }
}

/// C-SD3: atomic DELETE+INSERT on space upgrade. Rollback restores prior state.
#[test]
fn space_upgrade_atomic_delete_then_insert() {
    let db = make_db();
    for i in 0..5 {
        let id = uuid::Uuid::new_v4();
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(id));
        row.insert("embedding_space_id".into(), Value::Text("v1".into()));
        row.insert("signature_kind".into(), Value::Text(format!("k{}", i)));
        row.insert("scope_labels".into(), Value::Json(serde_json::json!({})));
        row.insert("tenant_id".into(), Value::Text("t".into()));
        row.insert("member_count".into(), Value::Int64(5));
        row.insert("variance".into(), Value::Float64(0.1));
        row.insert("last_matched_at".into(), Value::TxId(TxId(0)));
        let v = vec![1.0_f32 - i as f32 * 0.01, 0.0, 0.0, 0.0];
        row.insert("centroid".into(), Value::Vector(v.clone()));
        let row_id = db.insert_row(tx, "signatures", row).expect("ins");
        db.insert_vector(tx, VectorIndexRef::new("signatures", "centroid"), row_id, v)
            .expect("vec");
        db.commit(tx).expect("commit");
    }
    let r_before = db
        .execute("SELECT embedding_space_id FROM signatures", &HashMap::new())
        .expect("select");
    assert_eq!(r_before.rows.len(), 5);

    // Atomic upgrade
    let tx = db.begin();
    db.execute_in_tx(
        tx,
        "DELETE FROM signatures WHERE embedding_space_id = 'v1'",
        &HashMap::new(),
    )
    .expect("delete v1");
    for i in 0..3 {
        let id = uuid::Uuid::new_v4();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(id));
        row.insert("embedding_space_id".into(), Value::Text("v2".into()));
        row.insert("signature_kind".into(), Value::Text(format!("k{}", i)));
        row.insert("scope_labels".into(), Value::Json(serde_json::json!({})));
        row.insert("tenant_id".into(), Value::Text("t".into()));
        row.insert("member_count".into(), Value::Int64(5));
        row.insert("variance".into(), Value::Float64(0.1));
        row.insert("last_matched_at".into(), Value::TxId(TxId(0)));
        let v = vec![0.0, 1.0_f32 - i as f32 * 0.01, 0.0, 0.0];
        row.insert("centroid".into(), Value::Vector(v.clone()));
        let row_id = db.insert_row(tx, "signatures", row).expect("ins v2");
        db.insert_vector(tx, VectorIndexRef::new("signatures", "centroid"), row_id, v)
            .expect("vec");
    }
    db.commit(tx).expect("commit upgrade");

    let r_after = db
        .execute("SELECT embedding_space_id FROM signatures", &HashMap::new())
        .expect("select");
    assert_eq!(r_after.rows.len(), 3);
    for row in &r_after.rows {
        assert_eq!(row[0].as_text(), Some("v2"));
    }

    // Rollback semantics
    let tx2 = db.begin();
    db.execute_in_tx(
        tx2,
        "DELETE FROM signatures WHERE embedding_space_id = 'v2'",
        &HashMap::new(),
    )
    .expect("delete staged");
    db.rollback(tx2).expect("rollback");
    let r_post = db
        .execute("SELECT embedding_space_id FROM signatures", &HashMap::new())
        .expect("select");
    assert_eq!(r_post.rows.len(), 3, "rollback failed to restore rows");
}

/// C-SD5: cross-tenant filter at engine level.
#[test]
fn cross_tenant_signature_filter_engine_enforced() {
    let db = make_db();
    for i in 0..6 {
        let tenant = if i % 2 == 0 { "tenant-a" } else { "tenant-b" };
        insert_signature(
            &db,
            &format!("space-{}", i),
            "k",
            serde_json::json!({}),
            tenant,
            vec![1.0, 0.0, 0.0, i as f32 * 0.01],
        )
        .expect("ins");
    }
    let r = db
        .execute(
            "SELECT tenant_id FROM signatures WHERE tenant_id = $t",
            &HashMap::from([("t".to_string(), Value::Text("tenant-a".to_string()))]),
        )
        .expect("scan");
    assert_eq!(r.rows.len(), 3);
    for row in &r.rows {
        assert_eq!(row[0].as_text(), Some("tenant-a"));
    }
}

/// C-SD7 / C-SD8: embedding-space filter composes with vector top-k.
#[test]
fn embedding_space_filter_composes_with_vector_search() {
    let db = make_db();
    insert_observation(&db, "space-A", "image", "t", vec![1.0, 0.0, 0.0, 0.0]);
    insert_observation(&db, "space-A", "audio", "t", vec![1.0, 0.0, 0.0, 0.0]);
    insert_observation(&db, "space-A", "text", "t", vec![1.0, 0.0, 0.0, 0.0]);
    insert_observation(&db, "space-B", "image", "t", vec![1.0, 0.0, 0.0, 0.0]);
    insert_observation(&db, "space-B", "image", "t", vec![1.0, 0.0, 0.0, 0.0]);

    let mut q = SemanticQuery::new("observations", "embedding", vec![1.0, 0.0, 0.0, 0.0], 100);
    q.where_clause = Some("embedding_space_id = 'space-A'".to_string());
    let r = db.semantic_search(q).expect("search");
    assert_eq!(r.len(), 3);
    for sr in &r {
        let s = sr
            .values
            .get("embedding_space_id")
            .and_then(|v| v.as_text())
            .expect("space");
        assert_eq!(s, "space-A");
    }
    let kinds: std::collections::HashSet<String> = r
        .iter()
        .map(|sr| {
            sr.values
                .get("evidence_kind")
                .and_then(|v| v.as_text())
                .map(String::from)
                .unwrap_or_default()
        })
        .collect();
    assert_eq!(kinds.len(), 3, "modality count: {:?}", kinds);
}

/// C-SD6: aggregate fields round-trip; JSON round-trip preserves all keys.
#[test]
fn signature_aggregate_fields_persist_and_round_trip() {
    let db = make_db();
    insert_signature(
        &db,
        "space-A",
        "face-cluster",
        serde_json::json!({"detector": "v1"}),
        "t",
        vec![0.5, 0.5, 0.5, 0.5],
    )
    .expect("ins");
    let r = db
        .execute(
            "SELECT member_count, variance, scope_labels FROM signatures WHERE embedding_space_id = $s",
            &HashMap::from([("s".to_string(), Value::Text("space-A".to_string()))]),
        )
        .expect("select");
    assert_eq!(r.rows.len(), 1);
    let row = &r.rows[0];
    assert_eq!(row[0].as_i64(), Some(10));
    match row[1] {
        Value::Float64(f) => assert!((f - 0.05).abs() < 1e-6),
        ref other => panic!("variance: {:?}", other),
    }
    match &row[2] {
        Value::Json(v) => {
            assert_eq!(v.get("detector").and_then(|x| x.as_str()), Some("v1"));
        }
        other => panic!("not JSON: {:?}", other),
    }
}

/// C-SD4: scope_labels JSON round-trip atomicity under concurrent UPDATE.
#[test]
fn signature_scope_labels_json_torn_check_under_concurrent_update() {
    let db = make_db();
    for i in 0..6 {
        insert_signature(
            &db,
            &format!("space-{}", i),
            "k",
            serde_json::json!({"version": "v1"}),
            "t",
            vec![0.0, 0.0, 0.0, i as f32 * 0.01],
        )
        .expect("ins");
    }
    let barrier = Arc::new(Barrier::new(3));
    let db_u = db.clone();
    let b_u = barrier.clone();
    let updater = thread::spawn(move || {
        b_u.wait();
        for _ in 0..50 {
            db_u.execute(
                "UPDATE signatures SET scope_labels = $j",
                &HashMap::from([(
                    "j".to_string(),
                    Value::Json(serde_json::json!({"version": "v2", "extra": "x"})),
                )]),
            )
            .ok();
            db_u.execute(
                "UPDATE signatures SET scope_labels = $j",
                &HashMap::from([(
                    "j".to_string(),
                    Value::Json(serde_json::json!({"version": "v1"})),
                )]),
            )
            .ok();
        }
    });
    let mut readers = vec![];
    for _ in 0..2 {
        let db_r = db.clone();
        let b_r = barrier.clone();
        readers.push(thread::spawn(move || {
            b_r.wait();
            for _ in 0..100 {
                let r = db_r
                    .execute("SELECT scope_labels FROM signatures", &HashMap::new())
                    .expect("select");
                for row in &r.rows {
                    if let Value::Json(v) = &row[0] {
                        let version =
                            v.get("version").and_then(|x| x.as_str()).unwrap_or_default();
                        assert!(version == "v1" || version == "v2", "torn: `{}`", version);
                        let has_extra = v.get("extra").is_some();
                        if version == "v2" {
                            assert!(has_extra, "v2 missing 'extra'");
                        } else {
                            assert!(!has_extra, "v1 has 'extra'");
                        }
                    }
                }
            }
        }));
    }
    updater.join().expect("u");
    for r in readers {
        r.join().expect("reader");
    }
}
