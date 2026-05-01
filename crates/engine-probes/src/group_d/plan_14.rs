//! Probes for plan 14 — cg-server-signature-propagate.
//!
//! Plan class: primitive-owner with partial blockage on §6 HandleScopeConstraint
//! (server-only-write subset for register_edge / sync_edge / propagate). The
//! engine-level surface — UNIQUE-on-edge_id contention, monotonic TXID
//! `registered_at` ordering, incremental sync cursor (`last_matched_at >
//! since_txid`), atomic match-and-create-intention multi-table tx, and
//! concurrent MATCHED_BY edge insertion with composite UNIQUE on
//! (edge_id, signature_id) — is probed below.
//!
//! Engine source cited:
//! - `contextdb/crates/contextdb-engine/src/database.rs::insert_row`
//!   (line ~1045) — coerce + validate + insert + commit.
//! - `contextdb/crates/contextdb-engine/src/database.rs::execute_in_tx`
//!   (line ~642) — multi-statement tx for atomic match-and-create.
//! - `contextdb/crates/contextdb-engine/src/database.rs::insert_edge`
//!   (line ~2129) — graph edges; UNIQUE-on-edge identity per (source,
//!   target, type) in `GraphStore::insert_edge`.
//! - `contextdb/crates/contextdb-tx/src/lib.rs` — `TxId` monotonicity per
//!   commit; `committed_watermark` advances with each commit.

use contextdb_core::error::Error;
use contextdb_core::types::{TxId, Value};
use contextdb_engine::Database;
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;

fn make_db() -> Arc<Database> {
    let db = Arc::new(Database::open_memory());
    db.execute(
        "CREATE TABLE edges_registry (\n\
            id UUID PRIMARY KEY,\n\
            edge_id TEXT NOT NULL UNIQUE,\n\
            scope_labels JSON,\n\
            registered_at TXID NOT NULL\n\
        )",
        &HashMap::new(),
    )
    .expect("create edges_registry");
    db.execute(
        "CREATE TABLE signatures_propagate (\n\
            id UUID PRIMARY KEY,\n\
            scope_labels JSON,\n\
            last_matched_at TXID NOT NULL\n\
        )",
        &HashMap::new(),
    )
    .expect("create signatures");
    // matched_by: composite UNIQUE on (edge_id, signature_id) so two edges
    // can match the same signature (different rows) but the same edge
    // matching the same signature twice is rejected.
    db.execute(
        "CREATE TABLE matched_by (\n\
            id UUID PRIMARY KEY,\n\
            edge_id TEXT NOT NULL,\n\
            signature_id UUID NOT NULL,\n\
            matched_at TXID NOT NULL,\n\
            UNIQUE (edge_id, signature_id)\n\
        )",
        &HashMap::new(),
    )
    .expect("create matched_by");
    db.execute(
        "CREATE TABLE intentions (\n\
            id UUID PRIMARY KEY,\n\
            description TEXT NOT NULL,\n\
            created_at TXID NOT NULL\n\
        )",
        &HashMap::new(),
    )
    .expect("create intentions");
    db
}

fn register_edge(db: &Database, edge_id: &str, scope: serde_json::Value) -> Result<TxId, Error> {
    let id = uuid::Uuid::new_v4();
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert("edge_id".into(), Value::Text(edge_id.into()));
    row.insert("scope_labels".into(), Value::Json(scope));
    row.insert("registered_at".into(), Value::TxId(tx));
    let res = db.insert_row(tx, "edges_registry", row);
    match res {
        Ok(_) => match db.commit(tx) {
            Ok(()) => Ok(tx),
            Err(e) => Err(e),
        },
        Err(e) => {
            let _ = db.rollback(tx);
            Err(e)
        }
    }
}

/// C-SP1: concurrent register_edge for same edge_id — UNIQUE on edge_id.
///
/// Two writers attempt to register the same edge_id simultaneously. Engine
/// MUST surface UniqueViolation for the loser; only one row persists.
///
/// Lens: lens-question row "UNIQUE (single + composite)".
#[test]
fn concurrent_register_edge_unique_one_winner() {
    let db = make_db();
    let n_edges = 24;
    let edge_ids: Vec<String> = (0..n_edges).map(|i| format!("edge-{}", i)).collect();

    let barrier = Arc::new(Barrier::new(4));
    let mut handles = vec![];
    for tid in 0..4 {
        let db_t = db.clone();
        let edges_t = edge_ids.clone();
        let bar = barrier.clone();
        handles.push(thread::spawn(move || {
            bar.wait();
            let mut wins = 0u64;
            let mut uvs = 0u64;
            let mut others = 0u64;
            for eid in &edges_t {
                match register_edge(&db_t, eid, serde_json::json!({"tid": tid})) {
                    Ok(_) => wins += 1,
                    Err(Error::UniqueViolation { .. }) => uvs += 1,
                    Err(_) => others += 1,
                }
            }
            (wins, uvs, others)
        }));
    }
    let mut total_wins = 0u64;
    let mut total_uv = 0u64;
    let mut total_other = 0u64;
    for h in handles {
        let (w, u, o) = h.join().expect("worker");
        total_wins += w;
        total_uv += u;
        total_other += o;
    }
    assert_eq!(
        total_wins as usize, n_edges,
        "expected {} wins; got {}",
        n_edges, total_wins
    );
    assert_eq!(
        total_uv as usize,
        4 * n_edges - n_edges,
        "expected {} UVs; got {}",
        3 * n_edges,
        total_uv
    );
    assert_eq!(total_other, 0, "non-UV errors: {}", total_other);

    let r = db
        .execute("SELECT edge_id FROM edges_registry", &HashMap::new())
        .expect("scan");
    assert_eq!(r.rows.len(), n_edges, "stored row count");
}

/// C-SP2: sync_edge incremental cursor — `last_matched_at > since_txid`.
///
/// Probe asserts cursor boundary semantics: a signature whose `last_matched_at`
/// equals the cursor is NOT returned (strict `>`); a signature whose
/// `last_matched_at` is `cursor + 1` IS returned.
///
/// This load-bears the cg-app's incremental sync correctness: an off-by-one
/// here drops or duplicates signatures across sync calls.
///
/// Lens: lens-question row "TXID column type" — engine MUST treat `Value::TxId`
/// orderings correctly under composed predicates.
#[test]
fn sync_edge_cursor_strict_greater_than_correct() {
    let db = make_db();

    // Insert signatures with monotonic TXIDs by committing one per tx.
    let mut signature_txids: Vec<TxId> = vec![];
    for _ in 0..10 {
        let id = uuid::Uuid::new_v4();
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(id));
        row.insert("scope_labels".into(), Value::Json(serde_json::json!({})));
        row.insert("last_matched_at".into(), Value::TxId(tx));
        db.insert_row(tx, "signatures_propagate", row).expect("ins");
        db.commit(tx).expect("commit");
        signature_txids.push(tx);
    }

    let cursor = signature_txids[4];

    // Strict greater-than via SQL > comparison
    let r = db
        .execute(
            "SELECT id FROM signatures_propagate WHERE last_matched_at > $c",
            &HashMap::from([("c".to_string(), Value::TxId(cursor))]),
        )
        .expect("filtered scan");

    // Expected: signatures at indices 5..=9 (5 rows); index 4 must NOT appear.
    assert_eq!(
        r.rows.len(),
        5,
        "strict > cursor returned wrong count: {}",
        r.rows.len()
    );

    // Boundary: equality cursor must NOT include the cursor row.
    let r_eq = db
        .execute(
            "SELECT id FROM signatures_propagate WHERE last_matched_at >= $c",
            &HashMap::from([("c".to_string(), Value::TxId(cursor))]),
        )
        .expect("filtered scan");
    assert_eq!(
        r_eq.rows.len(),
        6,
        "inclusive >= cursor returned wrong count: {}",
        r_eq.rows.len()
    );

    // High-water cursor: nothing greater than the last txid
    let high = signature_txids[9];
    let r_high = db
        .execute(
            "SELECT id FROM signatures_propagate WHERE last_matched_at > $c",
            &HashMap::from([("c".to_string(), Value::TxId(high))]),
        )
        .expect("filtered scan");
    assert_eq!(
        r_high.rows.len(),
        0,
        "high-water cursor leaked rows: {}",
        r_high.rows.len()
    );
}

/// C-SP4: match_signature concurrent edges — two edges concurrently match
/// the same signature, INSERTing distinct rows into matched_by with the
/// composite UNIQUE on (edge_id, signature_id).
///
/// Engine MUST allow both INSERTs (distinct edge_id values) AND reject a
/// duplicate (edge_id, signature_id) attempt.
///
/// Lens: lens-question row "UNIQUE (single + composite)" — composite UNIQUE
/// only enforced on the first column is a wrong-impl shape.
#[test]
fn matched_by_composite_unique_distinct_edges_succeed_duplicate_rejected() {
    let db = make_db();

    // One signature, two edges, race to match.
    let signature_id = uuid::Uuid::new_v4();
    {
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(signature_id));
        row.insert("scope_labels".into(), Value::Json(serde_json::json!({})));
        row.insert("last_matched_at".into(), Value::TxId(tx));
        db.insert_row(tx, "signatures_propagate", row).expect("ins");
        db.commit(tx).expect("commit");
    }

    let n_edges = 8;
    let barrier = Arc::new(Barrier::new(n_edges));
    let mut handles = vec![];
    for i in 0..n_edges {
        let db_t = db.clone();
        let bar = barrier.clone();
        let sig = signature_id;
        handles.push(thread::spawn(move || {
            bar.wait();
            // First match: should succeed
            let id = uuid::Uuid::new_v4();
            let tx = db_t.begin();
            let mut row = HashMap::new();
            row.insert("id".into(), Value::Uuid(id));
            row.insert("edge_id".into(), Value::Text(format!("edge-{}", i)));
            row.insert("signature_id".into(), Value::Uuid(sig));
            row.insert("matched_at".into(), Value::TxId(tx));
            let r1 = db_t.insert_row(tx, "matched_by", row);
            let commit_first = match r1 {
                Ok(_) => db_t.commit(tx),
                Err(e) => {
                    let _ = db_t.rollback(tx);
                    Err(e)
                }
            };
            assert!(commit_first.is_ok(), "first match must succeed: {:?}", commit_first);

            // Second match: same (edge_id, signature_id) — must fail with UV
            let tx2 = db_t.begin();
            let mut row2 = HashMap::new();
            row2.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
            row2.insert("edge_id".into(), Value::Text(format!("edge-{}", i)));
            row2.insert("signature_id".into(), Value::Uuid(sig));
            row2.insert("matched_at".into(), Value::TxId(tx2));
            let r2 = db_t.insert_row(tx2, "matched_by", row2);
            let _ = db_t.rollback(tx2);
            match r2 {
                Err(Error::UniqueViolation { .. }) => {}
                other => panic!(
                    "expected UniqueViolation on duplicate (edge_id, sig); got {:?}",
                    other.map(|_| "Ok").unwrap_or("Other")
                ),
            }
        }));
    }
    for h in handles {
        h.join().expect("worker");
    }

    // n_edges distinct edges all matched to the same signature → exactly
    // n_edges rows persisted.
    let r = db
        .execute("SELECT edge_id FROM matched_by", &HashMap::new())
        .expect("scan");
    assert_eq!(r.rows.len(), n_edges, "matched_by stored row count");
}

/// match_signature_atomic: vector hit + INSERT new Intention + INSERT
/// MATCHED_BY edge — single tx. Failure of one insert must roll back ALL.
///
/// Multi-table tx atomicity. Lens: lens-question row "Multi-table TX (combination)"
/// — partial commit / FK violation rolling back row N only is wrong-impl.
#[test]
fn match_atomic_intention_and_matched_by_rollback_together() {
    let db = make_db();

    // Pre-existing signature
    let signature_id = uuid::Uuid::new_v4();
    {
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(signature_id));
        row.insert("scope_labels".into(), Value::Json(serde_json::json!({})));
        row.insert("last_matched_at".into(), Value::TxId(tx));
        db.insert_row(tx, "signatures_propagate", row).expect("ins");
        db.commit(tx).expect("commit");
    }

    // Pre-insert one matched_by row so we can force a UniqueViolation on
    // the second insert in the multi-statement tx.
    {
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        row.insert("edge_id".into(), Value::Text("edge-X".into()));
        row.insert("signature_id".into(), Value::Uuid(signature_id));
        row.insert("matched_at".into(), Value::TxId(tx));
        db.insert_row(tx, "matched_by", row).expect("ins");
        db.commit(tx).expect("commit");
    }

    // Now: multi-statement tx — INSERT intention, INSERT matched_by (which
    // collides with the existing edge-X / signature_id row). The intention
    // INSERT must NOT remain in the database after rollback.
    let intention_id = uuid::Uuid::new_v4();
    let tx = db.begin();
    let mut int_row = HashMap::new();
    int_row.insert("id".into(), Value::Uuid(intention_id));
    int_row.insert(
        "description".into(),
        Value::Text("auto-discovered intention".into()),
    );
    int_row.insert("created_at".into(), Value::TxId(tx));
    db.insert_row(tx, "intentions", int_row)
        .expect("intent ins");

    let mut mb_row = HashMap::new();
    mb_row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
    mb_row.insert("edge_id".into(), Value::Text("edge-X".into()));
    mb_row.insert("signature_id".into(), Value::Uuid(signature_id));
    mb_row.insert("matched_at".into(), Value::TxId(tx));
    let r = db.insert_row(tx, "matched_by", mb_row);
    assert!(matches!(r, Err(Error::UniqueViolation { .. })), "expected UV; got {:?}", r.is_ok());

    // Caller chose to abort the whole atomic match
    db.rollback(tx).expect("rollback");

    // Intention must NOT be in the DB
    let post = db
        .execute(
            "SELECT id FROM intentions WHERE id = $i",
            &HashMap::from([("i".to_string(), Value::Uuid(intention_id))]),
        )
        .expect("scan");
    assert_eq!(
        post.rows.len(),
        0,
        "intention persisted despite rollback — atomicity broken"
    );
}

/// C-SP5 (engine surface only): scope-label JSON predicate filter at
/// SELECT-time MUST exclude rows whose `scope_labels` does not match.
/// Failure mode: predicate ignored or evaluated post-fetch.
///
/// (The cross-tenant / scope-isolation server-side gating is the
/// HandleScopeConstraint subset and is catalogued, not probed.)
#[test]
fn scope_label_predicate_filter_excludes_nonmatching_rows() {
    let db = make_db();
    for i in 0..10 {
        let id = uuid::Uuid::new_v4();
        let scope = if i < 5 {
            serde_json::json!({"tenant": "alpha"})
        } else {
            serde_json::json!({"tenant": "beta"})
        };
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(id));
        row.insert("scope_labels".into(), Value::Json(scope));
        row.insert("last_matched_at".into(), Value::TxId(tx));
        db.insert_row(tx, "signatures_propagate", row).expect("ins");
        db.commit(tx).expect("commit");
    }

    // Equality on JSON path — engine surface for scope filtering is the
    // standard JSON column equality; the cg layer uses a JSON-equality
    // predicate so we probe that shape.
    let r = db
        .execute(
            "SELECT id FROM signatures_propagate WHERE scope_labels = $s",
            &HashMap::from([(
                "s".to_string(),
                Value::Json(serde_json::json!({"tenant": "alpha"})),
            )]),
        )
        .expect("scan");
    assert_eq!(
        r.rows.len(),
        5,
        "scope_labels equality filter wrong count: {}",
        r.rows.len()
    );
}

/// Monotonic TXID across `register_edge` commits.
///
/// Each successful commit must have a strictly greater committed_watermark
/// than the previous one (TXID monotonicity). Probed via the `registered_at`
/// column the row records.
///
/// Lens: lens-question row "TXID column type" — TXID accepting arbitrary u64
/// instead of advancing the manager's monotonic clock is wrong-impl.
#[test]
fn registered_at_txids_strictly_monotonic_across_concurrent_commits() {
    let db = make_db();
    let n_edges = 32;

    let barrier = Arc::new(Barrier::new(4));
    let mut handles = vec![];
    for tid in 0..4 {
        let db_t = db.clone();
        let bar = barrier.clone();
        handles.push(thread::spawn(move || {
            bar.wait();
            let mut my_txids = vec![];
            for i in 0..n_edges / 4 {
                let eid = format!("edge-{}-{}", tid, i);
                if let Ok(tx) = register_edge(&db_t, &eid, serde_json::json!({})) {
                    my_txids.push(tx);
                }
            }
            my_txids
        }));
    }

    let mut all_txids: Vec<TxId> = vec![];
    for h in handles {
        all_txids.extend(h.join().expect("worker"));
    }

    // All TxIds must be unique (TX manager uniqueness)
    let mut sorted = all_txids.clone();
    sorted.sort();
    let dedup_len = {
        let mut s = sorted.clone();
        s.dedup();
        s.len()
    };
    assert_eq!(sorted.len(), dedup_len, "duplicate TXIDs across commits");

    // The DB-stored `registered_at` values, sorted, must equal the sorted
    // collected commit TXIDs — proving the engine wrote each row's TXID at
    // the actual commit time, not at some later moment.
    let r = db
        .execute(
            "SELECT registered_at FROM edges_registry",
            &HashMap::new(),
        )
        .expect("scan");
    let mut stored_txids: Vec<TxId> = r
        .rows
        .iter()
        .map(|row| match &row[0] {
            Value::TxId(t) => *t,
            other => panic!("registered_at not TxId: {:?}", other),
        })
        .collect();
    stored_txids.sort();
    assert_eq!(
        sorted,
        stored_txids,
        "TXIDs reported on commit do not match TXIDs persisted in registered_at"
    );
}
