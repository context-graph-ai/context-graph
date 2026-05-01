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

    // Composite UNIQUE second-column-must-matter sub-case: same
    // `embedding_space_id` with different `signature_kind` values MUST both
    // succeed. A wrong engine that enforces UNIQUE only on the first column
    // would reject the second insert.
    insert_signature(&db, "shared-space", "kind-A", serde_json::json!({}), "tenant-x", vec![1.0, 0.0, 0.0, 0.0]).expect("first shared-space / kind-A");
    insert_signature(&db, "shared-space", "kind-B", serde_json::json!({}), "tenant-x", vec![0.0, 1.0, 0.0, 0.0]).expect("second shared-space / kind-B must succeed (composite UNIQUE permits)");
    let r2 = db
        .execute(
            "SELECT signature_kind FROM signatures WHERE embedding_space_id = $s",
            &HashMap::from([("s".to_string(), Value::Text("shared-space".to_string()))]),
        )
        .expect("scan shared-space");
    assert_eq!(r2.rows.len(), 2, "expected 2 rows under shared-space; got {} — composite UNIQUE may be first-column-only", r2.rows.len());

    // Symmetric: same `signature_kind` with different `embedding_space_id`
    // values MUST both succeed.
    insert_signature(&db, "space-X", "shared-kind", serde_json::json!({}), "tenant-x", vec![0.0, 0.0, 1.0, 0.0]).expect("space-X / shared-kind");
    insert_signature(&db, "space-Y", "shared-kind", serde_json::json!({}), "tenant-x", vec![0.0, 0.0, 0.0, 1.0]).expect("space-Y / shared-kind must succeed");
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

/// Mid-tx atomicity hole on space-upgrade: DELETE old rows + INSERT new rows
/// in one tx, but the INSERT phase fails (UV against a pre-staged row).
/// Rolling back must restore the deleted rows AND must not orphan any vector
/// entry from partial inserts. Catches the lens shape "vector-index write
/// succeeds but row INSERT fails — vector orphaned" plus the multi-step DELETE
/// + INSERT rollback path.
#[test]
fn space_upgrade_rollback_after_partial_inserts_no_orphan() {
    let db = make_db();
    for i in 0..4 {
        let v = vec![1.0_f32 - i as f32 * 0.01, 0.0, 0.0, 0.0];
        insert_signature(&db, "v1", &format!("k{}", i), serde_json::json!({}), "t", v).expect("v1");
    }
    insert_signature(&db, "v2", "collide", serde_json::json!({}), "t", vec![0.5, 0.5, 0.5, 0.5]).expect("v2 collide");

    let upgrade_vec = vec![0.9_f32, 0.1, 0.0, 0.0];
    let pre_search = db.semantic_search(SemanticQuery::new("signatures", "centroid", upgrade_vec.clone(), 100)).expect("pre");
    let pre_count = pre_search.len();

    let temp_row_id = {
        let tx = db.begin();
        db.execute_in_tx(tx, "DELETE FROM signatures WHERE embedding_space_id = 'v1'", &HashMap::new()).expect("delete v1");

        let temp_id = uuid::Uuid::new_v4();
        let mut ok_row = HashMap::new();
        ok_row.insert("id".into(), Value::Uuid(temp_id));
        ok_row.insert("embedding_space_id".into(), Value::Text("v2".into()));
        ok_row.insert("signature_kind".into(), Value::Text("k-good".into()));
        ok_row.insert("scope_labels".into(), Value::Json(serde_json::json!({})));
        ok_row.insert("tenant_id".into(), Value::Text("t".into()));
        ok_row.insert("member_count".into(), Value::Int64(5));
        ok_row.insert("variance".into(), Value::Float64(0.1));
        ok_row.insert("last_matched_at".into(), Value::TxId(TxId(0)));
        ok_row.insert("centroid".into(), Value::Vector(upgrade_vec.clone()));
        let temp_row_id = db.insert_row(tx, "signatures", ok_row).expect("ok ins");
        db.insert_vector(tx, VectorIndexRef::new("signatures", "centroid"), temp_row_id, upgrade_vec.clone()).expect("ok vec");

        let mut bad_row = HashMap::new();
        bad_row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        bad_row.insert("embedding_space_id".into(), Value::Text("v2".into()));
        bad_row.insert("signature_kind".into(), Value::Text("collide".into()));
        bad_row.insert("scope_labels".into(), Value::Json(serde_json::json!({})));
        bad_row.insert("tenant_id".into(), Value::Text("t".into()));
        bad_row.insert("member_count".into(), Value::Int64(5));
        bad_row.insert("variance".into(), Value::Float64(0.1));
        bad_row.insert("last_matched_at".into(), Value::TxId(TxId(0)));
        bad_row.insert("centroid".into(), Value::Vector(vec![0.7, 0.7, 0.7, 0.7]));
        let r = db.insert_row(tx, "signatures", bad_row);
        match r {
            Err(Error::UniqueViolation { .. }) => {}
            Ok(_) => panic!("expected UV"),
            Err(other) => panic!("expected UV; got {:?}", other),
        }
        db.rollback(tx).expect("rollback");
        temp_row_id
    };

    let r_v1 = db.execute("SELECT signature_kind FROM signatures WHERE embedding_space_id = $s", &HashMap::from([("s".to_string(), Value::Text("v1".to_string()))])).expect("scan v1");
    assert_eq!(r_v1.rows.len(), 4, "rollback failed to restore v1 rows: got {}", r_v1.rows.len());

    let post_search = db.semantic_search(SemanticQuery::new("signatures", "centroid", upgrade_vec, 100)).expect("post");
    for sr in &post_search {
        assert_ne!(sr.row_id, temp_row_id, "rolled-back row_id surfaced — vector orphan");
    }
    assert_eq!(post_search.len(), pre_count, "post-rollback vector hits {} != pre {} — orphan", post_search.len(), pre_count);
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

/// C-SD6: aggregate fields round-trip; multi-key JSON round-trip preserves
/// keys AND insertion order; TXID column reads back as `Value::TxId`, not
/// `Value::Integer`.
///
/// Lens: lens-question rows "JSON column round-trip" shape (b) field-order
/// canonicalisation, and "TXID column type" shape (a) "TxId round-trips
/// through INSERT but reads back as Integer".
#[test]
fn signature_aggregate_fields_persist_and_round_trip() {
    let db = make_db();
    let mut scope = serde_json::Map::new();
    scope.insert("detector".into(), serde_json::Value::String("v1".into()));
    scope.insert("zeta".into(), serde_json::Value::Number(7.into()));
    scope.insert("alpha".into(), serde_json::Value::Bool(true));
    let scope_payload = serde_json::Value::Object(scope);
    let scope_serialised = serde_json::to_string(&scope_payload).expect("ser");
    assert!(scope_serialised.starts_with("{\"detector\""), "fixture order broken: {}", scope_serialised);

    insert_signature(&db, "space-A", "face-cluster", scope_payload, "t", vec![0.5, 0.5, 0.5, 0.5]).expect("ins");
    let r = db
        .execute(
            "SELECT member_count, variance, scope_labels, last_matched_at FROM signatures WHERE embedding_space_id = $s",
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
            assert_eq!(v.get("zeta").and_then(|x| x.as_u64()), Some(7));
            assert_eq!(v.get("alpha").and_then(|x| x.as_bool()), Some(true));
            let stored_serialised = serde_json::to_string(v).expect("ser stored");
            assert_eq!(stored_serialised, scope_serialised, "JSON round-trip canonicalised key order: input `{}`, stored `{}`", scope_serialised, stored_serialised);
            if let serde_json::Value::Object(map) = v {
                let keys: Vec<&str> = map.keys().map(String::as_str).collect();
                assert_eq!(keys, vec!["detector", "zeta", "alpha"], "key iteration order broken: {:?}", keys);
            }
        }
        other => panic!("not JSON: {:?}", other),
    }
    // TXID column read-back type: MUST be Value::TxId, not Value::Integer.
    match row[3] {
        Value::TxId(_) => {}
        ref other => panic!("last_matched_at read back as wrong variant: {:?} (expected Value::TxId)", other),
    }
}

/// C-SD4: scope_labels JSON round-trip atomicity under concurrent UPDATE,
/// using multi-key payloads with non-alphabetical insertion order. Readers
/// assert (i) no torn key combinations and (ii) byte-canonical-form equality
/// with the writer's exact serialisation — catches re-ordering wrong-impls.
#[test]
fn signature_scope_labels_json_torn_check_under_concurrent_update() {
    let db = make_db();
    // Initial payload IS payload_v1 — every existing row is already in
    // the v1 state when the writer race begins. This avoids a transient
    // window where the reader sees an "initial" payload distinct from
    // either writer variant.
    let mut p_v1 = serde_json::Map::new();
    p_v1.insert("version".into(), serde_json::Value::String("v1".into()));
    p_v1.insert("region".into(), serde_json::Value::String("eu".into()));
    p_v1.insert("alpha".into(), serde_json::Value::Bool(false));
    let payload_v1 = serde_json::Value::Object(p_v1);
    for i in 0..6 {
        insert_signature(&db, &format!("space-{}", i), "k", payload_v1.clone(), "t", vec![0.0, 0.0, 0.0, i as f32 * 0.01]).expect("ins");
    }
    let canonical_v1 = serde_json::to_string(&payload_v1).expect("ser v1");
    assert!(canonical_v1.starts_with("{\"version\""), "v1 fixture broken");

    let mut p_v2 = serde_json::Map::new();
    p_v2.insert("version".into(), serde_json::Value::String("v2".into()));
    p_v2.insert("region".into(), serde_json::Value::String("eu".into()));
    p_v2.insert("alpha".into(), serde_json::Value::Bool(true));
    p_v2.insert("extra".into(), serde_json::Value::String("x".into()));
    let payload_v2 = serde_json::Value::Object(p_v2);
    let canonical_v2 = serde_json::to_string(&payload_v2).expect("ser v2");
    assert!(canonical_v2.starts_with("{\"version\""), "v2 fixture broken");

    let barrier = Arc::new(Barrier::new(3));
    let db_u = db.clone();
    let b_u = barrier.clone();
    let pv1 = payload_v1.clone();
    let pv2 = payload_v2.clone();
    let updater = thread::spawn(move || {
        b_u.wait();
        for _ in 0..50 {
            db_u.execute("UPDATE signatures SET scope_labels = $j", &HashMap::from([("j".to_string(), Value::Json(pv2.clone()))])).ok();
            db_u.execute("UPDATE signatures SET scope_labels = $j", &HashMap::from([("j".to_string(), Value::Json(pv1.clone()))])).ok();
        }
    });
    let mut readers = vec![];
    for _ in 0..2 {
        let db_r = db.clone();
        let b_r = barrier.clone();
        let cv1 = canonical_v1.clone();
        let cv2 = canonical_v2.clone();
        readers.push(thread::spawn(move || {
            b_r.wait();
            for _ in 0..100 {
                let r = db_r.execute("SELECT scope_labels FROM signatures", &HashMap::new()).expect("select");
                for row in &r.rows {
                    if let Value::Json(v) = &row[0] {
                        let version = v.get("version").and_then(|x| x.as_str()).unwrap_or_default();
                        assert!(version == "v1" || version == "v2", "torn: `{}`", version);
                        let has_extra = v.get("extra").is_some();
                        let alpha = v.get("alpha").and_then(|x| x.as_bool());
                        if version == "v2" {
                            assert!(has_extra, "v2 missing 'extra' — torn JSON across keys");
                            assert_eq!(alpha, Some(true), "v2 wrong alpha");
                            let observed = serde_json::to_string(v).expect("ser obs");
                            assert_eq!(observed, cv2, "v2 round-trip canonicalised key order: stored `{}`", observed);
                        } else {
                            assert!(!has_extra, "v1 has 'extra'");
                            assert_eq!(alpha, Some(false), "v1 wrong alpha");
                            let observed = serde_json::to_string(v).expect("ser obs");
                            assert_eq!(observed, cv1, "v1 round-trip canonicalised key order: stored `{}`", observed);
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
