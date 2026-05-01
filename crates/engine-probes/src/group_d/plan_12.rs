//! Probes for plan 12 — cg-server-correlate.
//!
//! Plan class: primitive-owner with partial blockage on §6 HandleScopeConstraint.
//! The blocked subset (`correlate_observations_isolation_by_scope_labels_engine_enforced`)
//! is catalogued one-line; the rest is probed below.
//!
//! Engine surface targeted:
//! - cross-context vector query (no `WHERE context_id = ?` filter)
//! - time-window filter on `observations.timestamp`
//! - JSON `scope_labels` filter composing with vector top-k
//! - INSERT into a `correlations` table with composite UNIQUE
//! - concurrency: cross-context query under concurrent insert in another context;
//!   concurrent JSON-column UPDATE while query is running; correlate idempotency.
//!
//! Engine source cited:
//! - `contextdb/crates/contextdb-engine/src/database.rs::semantic_search`
//!   (line ~2293) — vector index keyed by `(table, column)`; snapshot-isolated.
//! - `contextdb/crates/contextdb-engine/src/database.rs::insert_row`
//!   (line ~1045) — coerce + validate + insert; commit-time atomicity.

use contextdb_core::error::Error;
use contextdb_core::types::{RowId, Value, VectorIndexRef};
use contextdb_engine::{Database, SemanticQuery};
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;

fn make_db() -> Arc<Database> {
    let db = Arc::new(Database::open_memory());
    db.execute(
        "CREATE TABLE observations (\n\
            id UUID PRIMARY KEY,\n\
            context_id TEXT NOT NULL,\n\
            timestamp INTEGER NOT NULL,\n\
            scope_labels JSON,\n\
            embedding VECTOR(4)\n\
        )",
        &HashMap::new(),
    )
    .expect("create observations");
    db.execute(
        "CREATE TABLE correlations (\n\
            id UUID PRIMARY KEY,\n\
            context_a TEXT NOT NULL,\n\
            context_b TEXT NOT NULL,\n\
            score REAL NOT NULL,\n\
            UNIQUE (context_a, context_b)\n\
        )",
        &HashMap::new(),
    )
    .expect("create correlations");
    db
}

fn insert_observation(
    db: &Database,
    context_id: &str,
    timestamp: i64,
    scope_labels: serde_json::Value,
    embedding: Vec<f32>,
) -> RowId {
    let id = uuid::Uuid::new_v4();
    let tx = db.begin();
    let mut row: HashMap<String, Value> = HashMap::new();
    row.insert("id".into(), Value::Uuid(id));
    row.insert("context_id".into(), Value::Text(context_id.into()));
    row.insert("timestamp".into(), Value::Int64(timestamp));
    row.insert("scope_labels".into(), Value::Json(scope_labels));
    row.insert("embedding".into(), Value::Vector(embedding.clone()));
    let row_id = db.insert_row(tx, "observations", row).expect("insert obs");
    db.insert_vector(
        tx,
        VectorIndexRef::new("observations", "embedding"),
        row_id,
        embedding,
    )
    .expect("insert vector");
    db.commit(tx).expect("commit");
    row_id
}

/// C-CR1: cross-context vector query under concurrent insert in another context.
/// Snapshot read MUST NOT yield torn rows or NaN scores. Vector top-k results
/// must be unique within a single search call. Per-result row data must be
/// fully populated (catches the HNSW-writes-before-row-INSERT atomicity hole).
/// After the writer commits 32 ctx-b vectors, all 32 MUST be reachable via
/// final search (catches the silent-drop-of-concurrent-tx-vectors wrong-impl).
#[test]
fn cross_context_vector_query_under_concurrent_insert_in_another_context() {
    let db = make_db();
    for i in 0..16 {
        insert_observation(
            &db,
            "ctx-a",
            1000 + i as i64,
            serde_json::json!({"labels": ["a"]}),
            vec![1.0 + i as f32 * 0.01, 0.0, 0.0, 0.0],
        );
    }
    let barrier = Arc::new(Barrier::new(3));
    let probe_query = vec![1.0_f32, 0.0, 0.0, 0.0];

    let db_w = db.clone();
    let b_w = barrier.clone();
    let writer = thread::spawn(move || {
        b_w.wait();
        for i in 0..32 {
            insert_observation(
                &db_w,
                "ctx-b",
                2000 + i as i64,
                serde_json::json!({"labels": ["b"]}),
                vec![0.99, 0.0, 0.0, i as f32 * 0.01],
            );
        }
    });

    let db_r1 = db.clone();
    let b_r1 = barrier.clone();
    let q1 = probe_query.clone();
    let reader1 = thread::spawn(move || {
        b_r1.wait();
        for _ in 0..32 {
            let q = SemanticQuery::new("observations", "embedding", q1.clone(), 5);
            let r = db_r1.semantic_search(q).expect("search");
            let mut seen: std::collections::HashSet<RowId> = std::collections::HashSet::new();
            for sr in &r {
                assert!(
                    seen.insert(sr.row_id),
                    "duplicate row_id {:?} in single search result — torn snapshot",
                    sr.row_id
                );
                assert!(
                    sr.vector_score.is_finite(),
                    "torn vector score: {}",
                    sr.vector_score
                );
                // Atomicity: HNSW vector hits MUST carry their row data. A
                // wrong engine that exposes vector hits whose row INSERT was
                // not yet committed would have `values` empty or missing
                // `context_id`. Reject that shape.
                assert!(
                    sr.values.get("context_id").is_some(),
                    "vector hit row_id {:?} missing context_id — HNSW write committed before row INSERT",
                    sr.row_id
                );
                assert!(
                    sr.values.get("timestamp").is_some(),
                    "vector hit row_id {:?} missing timestamp — atomicity hole vector vs row table",
                    sr.row_id
                );
            }
        }
    });

    let db_r2 = db.clone();
    let b_r2 = barrier.clone();
    let q2 = probe_query.clone();
    let reader2 = thread::spawn(move || {
        b_r2.wait();
        for _ in 0..32 {
            let q = SemanticQuery::new("observations", "embedding", q2.clone(), 5);
            let r = db_r2.semantic_search(q).expect("search");
            for sr in &r {
                assert!(sr.vector_score.is_finite());
                assert!(sr.values.get("context_id").is_some(),
                    "vector hit missing context_id — atomicity hole");
            }
        }
    });

    writer.join().expect("writer");
    reader1.join().expect("reader1");
    reader2.join().expect("reader2");

    // Post-test: every committed ctx-b vector MUST be reachable. A wrong
    // engine that silently dropped concurrent-tx vector inserts would
    // surface here.
    let q = SemanticQuery::new("observations", "embedding", probe_query, 100);
    let r = db.semantic_search(q).expect("final search");
    let mut ctx_b_seen = 0usize;
    for sr in &r {
        let cid = sr
            .values
            .get("context_id")
            .and_then(|v| v.as_text())
            .expect("context_id");
        if cid == "ctx-b" {
            ctx_b_seen += 1;
        }
    }
    assert_eq!(
        ctx_b_seen, 32,
        "writer committed 32 ctx-b vectors but final search reached only {} — concurrent-tx vector inserts silently dropped",
        ctx_b_seen
    );
    assert!(r.len() >= 48, "post-write search lost rows: only {} (expected >= 48)", r.len());
}

/// Mid-tx atomicity hole: vector-index write succeeds but a later row INSERT
/// in the same tx fails (PK collision). Engine MUST roll the vector entry
/// back — no orphan hit must surface in subsequent vector search.
///
/// Lens: lens-question row "Multi-table TX (combination)" shape (b)
/// "vector-index write succeeds but row INSERT fails — vector orphaned".
#[test]
fn vector_orphan_check_when_row_insert_fails_mid_tx() {
    let db = make_db();
    let collide_id = uuid::Uuid::new_v4();
    {
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(collide_id));
        row.insert("context_id".into(), Value::Text("ctx-pre".into()));
        row.insert("timestamp".into(), Value::Int64(100));
        row.insert("scope_labels".into(), Value::Json(serde_json::json!({})));
        row.insert("embedding".into(), Value::Vector(vec![0.0, 0.0, 0.0, 0.0]));
        let row_id = db.insert_row(tx, "observations", row).expect("pre-seed");
        db.insert_vector(tx, VectorIndexRef::new("observations", "embedding"), row_id, vec![0.0, 0.0, 0.0, 0.0]).expect("pre-seed vec");
        db.commit(tx).expect("commit");
    }

    let query_vec = vec![0.5_f32, 0.5, 0.5, 0.5];
    let pre = db.semantic_search(SemanticQuery::new("observations", "embedding", query_vec.clone(), 100)).expect("pre");
    let pre_count = pre.len();

    let temp_row_id = {
        let tx = db.begin();
        let mut ok_row = HashMap::new();
        ok_row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        ok_row.insert("context_id".into(), Value::Text("ctx-temp".into()));
        ok_row.insert("timestamp".into(), Value::Int64(200));
        ok_row.insert("scope_labels".into(), Value::Json(serde_json::json!({})));
        ok_row.insert("embedding".into(), Value::Vector(query_vec.clone()));
        let temp_row_id = db.insert_row(tx, "observations", ok_row).expect("temp row");
        db.insert_vector(tx, VectorIndexRef::new("observations", "embedding"), temp_row_id, query_vec.clone()).expect("temp vec");

        let mut bad_row = HashMap::new();
        bad_row.insert("id".into(), Value::Uuid(collide_id));
        bad_row.insert("context_id".into(), Value::Text("ctx-bad".into()));
        bad_row.insert("timestamp".into(), Value::Int64(300));
        bad_row.insert("scope_labels".into(), Value::Json(serde_json::json!({})));
        bad_row.insert("embedding".into(), Value::Vector(vec![0.7, 0.7, 0.7, 0.7]));
        let r = db.insert_row(tx, "observations", bad_row);
        match r {
            Err(Error::UniqueViolation { .. }) => {}
            Ok(_) => panic!("expected UV; got Ok"),
            Err(other) => panic!("expected UV; got {:?}", other),
        }
        db.rollback(tx).expect("rollback");
        temp_row_id
    };

    let post = db.semantic_search(SemanticQuery::new("observations", "embedding", query_vec, 100)).expect("post");
    for sr in &post {
        assert_ne!(sr.row_id, temp_row_id, "rolled-back row_id surfaced — orphaned HNSW entry");
    }
    assert_eq!(post.len(), pre_count, "post-rollback search hit count {} != pre {} — vector entry orphaned", post.len(), pre_count);
}

/// C-CR2: scope-label JSON filter atomicity under concurrent UPDATE.
/// Reader observes either pre-update or post-update JSON; never torn.
#[test]
fn scope_label_json_filter_atomic_under_concurrent_update() {
    let db = make_db();
    for i in 0..10 {
        insert_observation(
            &db,
            "ctx-x",
            3000 + i as i64,
            serde_json::json!({"version": "v1"}),
            vec![0.5, 0.5, 0.0, i as f32 * 0.01],
        );
    }
    let barrier = Arc::new(Barrier::new(3));

    let db_u = db.clone();
    let b_u = barrier.clone();
    let updater = thread::spawn(move || {
        b_u.wait();
        for _ in 0..50 {
            db_u.execute(
                "UPDATE observations SET scope_labels = $j WHERE context_id = $c",
                &HashMap::from([
                    (
                        "j".to_string(),
                        Value::Json(serde_json::json!({"version": "v2", "extra": "x"})),
                    ),
                    ("c".to_string(), Value::Text("ctx-x".to_string())),
                ]),
            )
            .ok();
            db_u.execute(
                "UPDATE observations SET scope_labels = $j WHERE context_id = $c",
                &HashMap::from([
                    (
                        "j".to_string(),
                        Value::Json(serde_json::json!({"version": "v1"})),
                    ),
                    ("c".to_string(), Value::Text("ctx-x".to_string())),
                ]),
            )
            .ok();
        }
    });

    let db_r1 = db.clone();
    let b_r1 = barrier.clone();
    let reader1 = thread::spawn(move || {
        b_r1.wait();
        for _ in 0..100 {
            let r = db_r1
                .execute(
                    "SELECT id, scope_labels FROM observations WHERE context_id = $c",
                    &HashMap::from([("c".to_string(), Value::Text("ctx-x".to_string()))]),
                )
                .expect("select");
            for row in &r.rows {
                let v = match &row[1] {
                    Value::Json(v) => v,
                    other => panic!("scope_labels not JSON: {:?}", other),
                };
                let version = v.get("version").and_then(|x| x.as_str()).expect("version");
                assert!(version == "v1" || version == "v2", "torn JSON `{}`", version);
                let has_extra = v.get("extra").is_some();
                if version == "v2" {
                    assert!(has_extra, "v2 missing 'extra' — torn JSON");
                } else {
                    assert!(!has_extra, "v1 has 'extra' — torn JSON");
                }
            }
        }
    });

    let db_r2 = db.clone();
    let b_r2 = barrier.clone();
    let reader2 = thread::spawn(move || {
        b_r2.wait();
        for _ in 0..100 {
            let r = db_r2
                .execute("SELECT id, scope_labels FROM observations", &HashMap::new())
                .expect("select");
            for row in &r.rows {
                if let Value::Json(v) = &row[1] {
                    let version = v.get("version").and_then(|x| x.as_str()).expect("version");
                    assert!(version == "v1" || version == "v2");
                }
            }
        }
    });

    updater.join().expect("u");
    reader1.join().expect("r1");
    reader2.join().expect("r2");
}

/// JSON field-order canonicalisation: write a multi-key JSON value with one
/// key insertion order, read it back, and assert (i) iteration order
/// preserves insertion, (ii) byte-serialised form matches the input
/// byte-for-byte. cg's byte-equality semantics depend on it.
///
/// Lens: lens-question row "JSON column round-trip" shape (b) "canonicalises
/// field order — load-bearing for byte-equality assertions in cg".
#[test]
fn scope_label_json_field_order_round_trips_byte_for_byte() {
    let db = make_db();

    let mut original = serde_json::Map::new();
    original.insert("z".into(), serde_json::Value::Number(1.into()));
    original.insert("a".into(), serde_json::Value::Number(2.into()));
    original.insert("m".into(), serde_json::Value::String("middle".into()));
    let original_payload = serde_json::Value::Object(original);
    let original_serialised = serde_json::to_string(&original_payload).expect("ser");
    assert!(
        original_serialised.starts_with("{\"z\""),
        "test fixture order broken: `{}`",
        original_serialised
    );

    insert_observation(
        &db,
        "ctx-fo",
        9999,
        original_payload.clone(),
        vec![0.1, 0.2, 0.3, 0.4],
    );

    let r = db
        .execute(
            "SELECT scope_labels FROM observations WHERE context_id = $c",
            &HashMap::from([("c".to_string(), Value::Text("ctx-fo".to_string()))]),
        )
        .expect("select");
    assert_eq!(r.rows.len(), 1);

    let stored = match &r.rows[0][0] {
        Value::Json(v) => v.clone(),
        other => panic!("scope_labels not JSON: {:?}", other),
    };
    let stored_serialised = serde_json::to_string(&stored).expect("ser stored");
    assert_eq!(
        stored_serialised, original_serialised,
        "JSON field-order canonicalisation: input `{}`, stored `{}`",
        original_serialised, stored_serialised
    );

    if let serde_json::Value::Object(map) = &stored {
        let keys: Vec<&str> = map.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["z", "a", "m"], "key iteration order broken: {:?}", keys);
    } else {
        panic!("stored not Object");
    }
}

/// C-CR4: correlate idempotency — repeated query returns same rows in same order.
#[test]
fn correlate_query_idempotent_in_absence_of_writes() {
    let db = make_db();
    for i in 0..20 {
        insert_observation(
            &db,
            if i < 10 { "ctx-a" } else { "ctx-b" },
            5000 + i as i64,
            serde_json::json!({}),
            vec![1.0, 0.0, 0.0, i as f32 * 0.01],
        );
    }
    let probe = vec![1.0_f32, 0.0, 0.0, 0.0];
    let r1 = db
        .semantic_search(SemanticQuery::new("observations", "embedding", probe.clone(), 10))
        .expect("q1");
    let r2 = db
        .semantic_search(SemanticQuery::new("observations", "embedding", probe.clone(), 10))
        .expect("q2");
    let r3 = db
        .semantic_search(SemanticQuery::new("observations", "embedding", probe, 10))
        .expect("q3");
    let ids1: Vec<RowId> = r1.iter().map(|r| r.row_id).collect();
    let ids2: Vec<RowId> = r2.iter().map(|r| r.row_id).collect();
    let ids3: Vec<RowId> = r3.iter().map(|r| r.row_id).collect();
    assert_eq!(ids1, ids2);
    assert_eq!(ids1, ids3);
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert!((a.vector_score - b.vector_score).abs() < f32::EPSILON);
    }
}

/// Composite UNIQUE on `correlations(context_a, context_b)` under contention:
/// 4 threads race on the same set of 32 pairs. Exactly one INSERT wins
/// per pair; losers see UniqueViolation. Plus a sub-case proving the
/// composite admits same first-col / different second-col (and symmetric).
#[test]
fn concurrent_correlation_insert_unique_composite_one_winner() {
    let db = make_db();
    let n_pairs = 32;
    let pairs: Vec<(String, String)> = (0..n_pairs)
        .map(|i| (format!("ctx-{}", i), format!("ctx-{}-other", i)))
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
            let mut losses = 0u64;
            for (a, b) in &pairs_t {
                let id = uuid::Uuid::new_v4();
                let mut row = HashMap::new();
                row.insert("id".into(), Value::Uuid(id));
                row.insert("context_a".into(), Value::Text(a.clone()));
                row.insert("context_b".into(), Value::Text(b.clone()));
                row.insert("score".into(), Value::Float64(0.5 + tid as f64 * 0.01));
                let tx = db_t.begin();
                match db_t.insert_row(tx, "correlations", row) {
                    Ok(_) => match db_t.commit(tx) {
                        Ok(()) => wins += 1,
                        Err(_) => losses += 1,
                    },
                    Err(_) => {
                        let _ = db_t.rollback(tx);
                        losses += 1;
                    }
                }
            }
            (wins, losses)
        }));
    }
    let mut total_wins = 0u64;
    let mut total_losses = 0u64;
    for h in handles {
        let (w, l) = h.join().expect("worker");
        total_wins += w;
        total_losses += l;
    }
    assert_eq!(total_wins as usize, n_pairs, "expected {} wins; got {}", n_pairs, total_wins);
    assert_eq!(total_losses as usize, 4 * n_pairs - n_pairs);
    let r = db
        .execute("SELECT context_a FROM correlations", &HashMap::new())
        .expect("scan");
    assert_eq!(r.rows.len(), n_pairs);

    // Composite UNIQUE second-column-must-matter: shared first-col / distinct
    // second-col MUST both succeed. A wrong engine enforcing UNIQUE only on
    // the first column rejects the second insert.
    {
        let mut row1 = HashMap::new();
        row1.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        row1.insert("context_a".into(), Value::Text("ctx-shared".into()));
        row1.insert("context_b".into(), Value::Text("ctx-other-1".into()));
        row1.insert("score".into(), Value::Float64(0.1));
        let tx = db.begin();
        db.insert_row(tx, "correlations", row1).expect("first shared-first-col");
        db.commit(tx).expect("commit");

        let mut row2 = HashMap::new();
        row2.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        row2.insert("context_a".into(), Value::Text("ctx-shared".into()));
        row2.insert("context_b".into(), Value::Text("ctx-other-2".into()));
        row2.insert("score".into(), Value::Float64(0.2));
        let tx2 = db.begin();
        db.insert_row(tx2, "correlations", row2).expect("second shared-first-col / distinct-second-col MUST succeed");
        db.commit(tx2).expect("commit");

        let r = db
            .execute(
                "SELECT context_b FROM correlations WHERE context_a = $a",
                &HashMap::from([("a".to_string(), Value::Text("ctx-shared".to_string()))]),
            )
            .expect("scan shared");
        assert_eq!(r.rows.len(), 2, "expected 2 rows under shared context_a; got {} — composite UNIQUE may be first-column-only", r.rows.len());
    }
    // Symmetric: distinct first-col / shared second-col MUST also both succeed.
    {
        let mut row1 = HashMap::new();
        row1.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        row1.insert("context_a".into(), Value::Text("ctx-distinct-1".into()));
        row1.insert("context_b".into(), Value::Text("ctx-b-shared".into()));
        row1.insert("score".into(), Value::Float64(0.3));
        let tx = db.begin();
        db.insert_row(tx, "correlations", row1).expect("first distinct-first-col / shared-second-col");
        db.commit(tx).expect("commit");

        let mut row2 = HashMap::new();
        row2.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        row2.insert("context_a".into(), Value::Text("ctx-distinct-2".into()));
        row2.insert("context_b".into(), Value::Text("ctx-b-shared".into()));
        row2.insert("score".into(), Value::Float64(0.4));
        let tx2 = db.begin();
        db.insert_row(tx2, "correlations", row2).expect("second distinct-first-col / shared-second-col MUST succeed");
        db.commit(tx2).expect("commit");
    }
}

/// Cross-context vector query + time-window filter compose under top-k.
#[test]
fn cross_context_query_with_time_window_composes_with_top_k() {
    let db = make_db();
    for ctx in &["ctx-a", "ctx-b", "ctx-c"] {
        for i in 0..5 {
            insert_observation(
                &db,
                ctx,
                7000 + i as i64,
                serde_json::json!({}),
                vec![1.0 - i as f32 * 0.05, 0.0, 0.0, 0.0],
            );
        }
        for i in 0..5 {
            insert_observation(
                &db,
                ctx,
                9000 + i as i64,
                serde_json::json!({}),
                vec![1.0 - i as f32 * 0.05, 0.0, 0.0, 0.0],
            );
        }
    }
    let mut q = SemanticQuery::new("observations", "embedding", vec![1.0, 0.0, 0.0, 0.0], 100);
    q.where_clause = Some("timestamp >= 7000 AND timestamp < 8000".to_string());
    let r = db.semantic_search(q).expect("search");
    assert_eq!(r.len(), 15, "time-band count: {}", r.len());
    for sr in &r {
        let ts = sr
            .values
            .get("timestamp")
            .and_then(|v| v.as_i64())
            .expect("ts");
        assert!((7000..8000).contains(&ts), "outside band: {}", ts);
    }
}

/// Cross-context exclude same context.
#[test]
fn cross_context_exclude_same_context() {
    let db = make_db();
    for i in 0..5 {
        insert_observation(
            &db,
            "ctx-a",
            10_000 + i as i64,
            serde_json::json!({}),
            vec![1.0, 0.0, 0.0, i as f32 * 0.001],
        );
        insert_observation(
            &db,
            "ctx-b",
            10_000 + i as i64,
            serde_json::json!({}),
            vec![1.0, 0.0, 0.0, i as f32 * 0.001 + 0.01],
        );
    }
    let mut q = SemanticQuery::new("observations", "embedding", vec![1.0, 0.0, 0.0, 0.0], 100);
    q.where_clause = Some("context_id != 'ctx-a'".to_string());
    let r = db.semantic_search(q).expect("search");
    assert!(r.len() >= 5);
    for sr in &r {
        let cid = sr.values.get("context_id").and_then(|v| v.as_text()).expect("cid");
        assert_ne!(cid, "ctx-a");
    }
}
