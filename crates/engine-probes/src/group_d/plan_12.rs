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
/// must be unique within a single search call.
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
            }
        }
    });

    writer.join().expect("writer");
    reader1.join().expect("reader1");
    reader2.join().expect("reader2");

    let q = SemanticQuery::new("observations", "embedding", probe_query, 50);
    let r = db.semantic_search(q).expect("final search");
    assert!(r.len() >= 16, "post-write search lost rows: only {}", r.len());
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
/// per pair; losers see UniqueViolation.
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
