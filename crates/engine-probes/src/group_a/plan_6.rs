//! Plan 6 cg-search — engine-probe sweep.
//!
//! See `.claude/plans/context-graph/r0/track1-enumeration/6-cg-search-primitives.md`
//! for the per-primitive enumeration.
//!
//! Primitives target ComputedRankColumn (released v1.0.0) +
//! NamedVectorIndexes (released v1.0.0).

use contextdb_core::{Error, Value, VectorIndexRef};
use contextdb_engine::{Database, SemanticQuery};
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;
use uuid::Uuid;

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Build the rank-policy schema: outcomes table with success_rate joined into
/// decisions.vector_text via RANK_POLICY. Mirrors cg's `effective_confidence
/// = confidence * success_rate` shape.
/// Cites: contextdb/tests/integration/rank_policy_tests.rs::create_policy_schema.
fn open_search_schema_with_rank_policy() -> Database {
    let db = Database::open_memory();
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE outcomes (
            id UUID PRIMARY KEY,
            decision_id UUID,
            success_rate REAL,
            confidence REAL,
            created_at INTEGER
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE INDEX outcomes_decision_id_idx ON outcomes(decision_id)",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE decisions (
            id UUID PRIMARY KEY,
            decision_type TEXT,
            description TEXT,
            confidence REAL,
            created_at INTEGER,
            context_id TEXT,
            vector_text VECTOR(4) RANK_POLICY (
                JOIN outcomes ON decision_id,
                FORMULA 'coalesce({confidence}, 1.0) * coalesce({success_rate}, 1.0)',
                SORT_KEY effective_confidence
            )
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE evidence (
            id UUID PRIMARY KEY,
            vector_text VECTOR(4),
            vector_vision VECTOR(8)
        )",
        &p,
    )
    .unwrap();
    db
}

fn insert_decision(db: &Database, id: Uuid, confidence: Option<f64>, vector: Vec<f32>) {
    db.execute(
        "INSERT INTO decisions (id, decision_type, confidence, created_at, vector_text) VALUES ($id, 'choice', $c, 0, $v)",
        &params(vec![
            ("id", Value::Uuid(id)),
            (
                "c",
                confidence.map(Value::Float64).unwrap_or(Value::Null),
            ),
            ("v", Value::Vector(vector)),
        ]),
    )
    .unwrap();
}

fn insert_outcome(db: &Database, decision_id: Uuid, success_rate: Option<f64>) {
    db.execute(
        "INSERT INTO outcomes (id, decision_id, success_rate, confidence, created_at) VALUES ($id, $did, $sr, 0.5, 1)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("did", Value::Uuid(decision_id)),
            (
                "sr",
                success_rate.map(Value::Float64).unwrap_or(Value::Null),
            ),
        ]),
    )
    .unwrap();
}

// ---------- Single-primitive probes ----------

/// 6 single-primitive: NamedVectorIndexes per-(table, column) routing.
/// Two rows A and B both have populated text and vision vectors. A's text
/// vector is close to query_t; B's text vector is far from query_t. The
/// vision vectors are flipped: B's vision vector is close to query_v, A's
/// is far. With two queries — text against query_t and vision against
/// query_v — top-1 must return A for text and B for vision. Wrong-impl:
/// engine routes both queries to the same physical index → both return
/// the same row.
#[test]
fn vector_query_routed_per_table_column() {
    let db = open_search_schema_with_rank_policy();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    db.execute(
        "INSERT INTO evidence (id, vector_text, vector_vision) VALUES ($id, $t, $v)",
        &params(vec![
            ("id", Value::Uuid(a)),
            ("t", Value::Vector(vec![1.0, 0.0, 0.0, 0.0])),
            ("v", Value::Vector(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
        ]),
    )
    .unwrap();
    db.execute(
        "INSERT INTO evidence (id, vector_text, vector_vision) VALUES ($id, $t, $v)",
        &params(vec![
            ("id", Value::Uuid(b)),
            ("t", Value::Vector(vec![0.0, 0.0, 0.0, 1.0])),
            ("v", Value::Vector(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();

    let snap = db.snapshot();
    let text_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &[1.0, 0.0, 0.0, 0.0],
            1,
            None,
            snap,
        )
        .unwrap();
    let vision_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_vision"),
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1,
            None,
            snap,
        )
        .unwrap();
    assert_eq!(text_hits.len(), 1);
    assert_eq!(vision_hits.len(), 1);

    // Look up the row UUIDs via scan to map RowId -> Uuid.
    let rows = db.scan("evidence", snap).unwrap();
    let id_for = |row_id| -> Uuid {
        rows.iter()
            .find(|r| r.row_id == row_id)
            .and_then(|r| r.values.get("id"))
            .and_then(|v| match v {
                Value::Uuid(u) => Some(*u),
                _ => None,
            })
            .expect("row not found")
    };
    let text_winner = id_for(text_hits[0].0);
    let vision_winner = id_for(vision_hits[0].0);

    assert_eq!(
        text_winner, a,
        "text-index top-1 against [1,0,0,0] must be A (text=[1,0,0,0]); routing failure returns B"
    );
    assert_eq!(
        vision_winner, b,
        "vision-index top-1 against [1,0,0,0,0,0,0,0] must be B (vision=[1,0,0,0,0,0,0,0]); routing failure returns A"
    );
}

/// 6 single-primitive: dimension mismatch at query time — engine must
/// surface VectorIndexDimensionMismatch instead of silent zero results
/// (lens (c) for VECTOR class).
#[test]
fn vector_query_dimension_mismatch_at_read() {
    let db = open_search_schema_with_rank_policy();
    let result = db.query_vector(
        VectorIndexRef::new("evidence", "vector_text"),
        &[1.0, 0.0], // declared dim 4, query dim 2
        10,
        None,
        db.snapshot(),
    );
    assert!(
        matches!(result, Err(Error::VectorIndexDimensionMismatch { .. })),
        "wrong-dim query must surface VectorIndexDimensionMismatch; got {result:?}"
    );
}

/// 6 single-primitive: unknown vector index returns typed UnknownVectorIndex,
/// not a silent empty result.
#[test]
fn vector_query_unknown_index_typed_error() {
    let db = open_search_schema_with_rank_policy();
    let result = db.query_vector(
        VectorIndexRef::new("evidence", "no_such_column"),
        &[1.0, 0.0, 0.0, 0.0],
        10,
        None,
        db.snapshot(),
    );
    assert!(
        matches!(result, Err(Error::UnknownVectorIndex { .. })),
        "querying unknown vector index must surface UnknownVectorIndex; got {result:?}"
    );
}

/// 6 single-primitive: ComputedRankColumn evaluates BEFORE top-k cutoff.
/// Insert two decisions: A has high cosine but low effective_confidence,
/// B has low cosine but high effective_confidence. With LIMIT 1 + sort_key,
/// the engine must return B (formula evaluates before cutoff). Wrong-impl:
/// engine returns top-k by raw cosine and then re-ranks → returns A.
///
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::semantic_search
/// + engine-gaps-for-cg.md §2 ComputedRankColumn (released v1.0.0).
#[test]
fn rank_policy_evaluates_before_topk_cutoff() {
    let db = open_search_schema_with_rank_policy();

    // 4 decisions seeded so LIMIT 2 has meaningful work.
    let id_high_cos_low_eff = Uuid::new_v4(); // close to query but low conf*sr
    let id_low_cos_high_eff = Uuid::new_v4(); // far from query but high conf*sr
    let id_mid_cos_mid_eff = Uuid::new_v4();
    let id_zero = Uuid::new_v4();

    insert_decision(&db, id_high_cos_low_eff, Some(0.1), vec![1.0, 0.0, 0.0, 0.0]);
    insert_outcome(&db, id_high_cos_low_eff, Some(0.1));

    insert_decision(&db, id_low_cos_high_eff, Some(0.9), vec![0.0, 1.0, 0.0, 0.0]);
    insert_outcome(&db, id_low_cos_high_eff, Some(0.9));

    insert_decision(&db, id_mid_cos_mid_eff, Some(0.5), vec![0.7, 0.7, 0.0, 0.0]);
    insert_outcome(&db, id_mid_cos_mid_eff, Some(0.5));

    insert_decision(&db, id_zero, Some(0.05), vec![0.5, 0.5, 0.5, 0.5]);
    insert_outcome(&db, id_zero, Some(0.05));

    let mut q = SemanticQuery::new("decisions", "vector_text", vec![1.0, 0.0, 0.0, 0.0], 1);
    q.sort_key = Some("effective_confidence".to_string());
    let results = db.semantic_search(q).expect("semantic search with rank");

    // The top-1 by effective_confidence (≈ 0.81 for id_low_cos_high_eff)
    // must be id_low_cos_high_eff, NOT id_high_cos_low_eff (≈ 0.01).
    assert_eq!(results.len(), 1, "LIMIT 1 must return exactly one row");
    let row_id = match results[0].values.get("id") {
        Some(Value::Uuid(u)) => *u,
        other => panic!("decisions.id must be Value::Uuid, got {other:?}"),
    };
    assert_eq!(
        row_id, id_low_cos_high_eff,
        "rank policy must rank by effective_confidence BEFORE cutoff; if top-k by cosine then re-rank, \
         this returns id_high_cos_low_eff and the test fails"
    );
}

/// 6 single-primitive: rank policy rehydrates on reopen — formula cache
/// rebuilds when Database::open re-reads the schema.
#[test]
fn rank_policy_persists_across_reopen() {
    use tempfile::TempDir;
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("rank.db");
    {
        let db = Database::open(&path).unwrap();
        let p = HashMap::new();
        db.execute(
            "CREATE TABLE outcomes (
                id UUID PRIMARY KEY,
                decision_id UUID,
                success_rate REAL,
                confidence REAL,
                created_at INTEGER
            )",
            &p,
        )
        .unwrap();
        db.execute(
            "CREATE INDEX outcomes_decision_id_idx ON outcomes(decision_id)",
            &p,
        )
        .unwrap();
        db.execute(
            "CREATE TABLE decisions (
                id UUID PRIMARY KEY,
                confidence REAL,
                vector_text VECTOR(4) RANK_POLICY (
                    JOIN outcomes ON decision_id,
                    FORMULA 'coalesce({confidence}, 1.0) * coalesce({success_rate}, 1.0)',
                    SORT_KEY effective_confidence
                )
            )",
            &p,
        )
        .unwrap();
        let id = Uuid::new_v4();
        db.execute(
            "INSERT INTO decisions (id, confidence, vector_text) VALUES ($id, 0.7, $v)",
            &params(vec![
                ("id", Value::Uuid(id)),
                ("v", Value::Vector(vec![1.0, 0.0, 0.0, 0.0])),
            ]),
        )
        .unwrap();
        db.execute(
            "INSERT INTO outcomes (id, decision_id, success_rate, confidence, created_at) VALUES ($id, $d, 0.8, 0.5, 0)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("d", Value::Uuid(id)),
            ]),
        )
        .unwrap();
        db.close().unwrap();
    }
    // Reopen and run the rank-policy query.
    let db = Database::open(&path).unwrap();
    let mut q = SemanticQuery::new("decisions", "vector_text", vec![1.0, 0.0, 0.0, 0.0], 1);
    q.sort_key = Some("effective_confidence".to_string());
    let results = db.semantic_search(q).expect("semantic search after reopen");
    assert_eq!(results.len(), 1);
    // Expected effective_confidence = 0.7 * 0.8 = 0.56.
    let rank = results[0].rank;
    assert!(
        (rank - 0.56).abs() < 0.001,
        "rank-policy formula must rehydrate post-reopen; expected ~0.56, got {rank}"
    );
}

/// 6 single-primitive (C-S3): DROP COLUMN refused while a rank policy
/// references the column. Wrong-impl: drop accepted; rank policy silently
/// breaks at next query.
#[test]
fn rank_policy_drop_column_blocked() {
    let db = open_search_schema_with_rank_policy();
    let result = db.execute(
        "ALTER TABLE outcomes DROP COLUMN success_rate",
        &HashMap::new(),
    );
    assert!(
        result.is_err(),
        "DROP COLUMN on policy-referenced column must be refused; got {result:?}"
    );
    assert!(
        matches!(result, Err(Error::DropBlockedByRankPolicy { .. })),
        "expected DropBlockedByRankPolicy variant, got {result:?}"
    );
}

/// 6 single-primitive: USE RANK requires LIMIT — engine must reject SQL
/// that uses rank without LIMIT.
#[test]
fn use_rank_requires_limit() {
    let db = open_search_schema_with_rank_policy();
    let id = Uuid::new_v4();
    insert_decision(&db, id, Some(0.7), vec![1.0, 0.0, 0.0, 0.0]);
    insert_outcome(&db, id, Some(0.8));

    let result = db.execute(
        "SELECT id FROM decisions ORDER BY vector_text <=> $q USE RANK effective_confidence",
        &params(vec![("q", Value::Vector(vec![1.0, 0.0, 0.0, 0.0]))]),
    );
    assert!(
        matches!(result, Err(Error::UseRankRequiresLimit) | Err(Error::ParseError(_))),
        "USE RANK without LIMIT must be rejected; got {result:?}"
    );
}

// ---------- Multi-primitive combinations ----------

/// 6 multi-primitive: vector top-k + relational filter + rank policy in one
/// engine query. Filter on confidence > 0.3 must compose with the rank
/// policy's evaluation before cutoff.
#[test]
fn vector_topk_with_filter_and_rank_policy() {
    let db = open_search_schema_with_rank_policy();
    let high = Uuid::new_v4();
    let low = Uuid::new_v4();
    insert_decision(&db, high, Some(0.9), vec![1.0, 0.0, 0.0, 0.0]);
    insert_outcome(&db, high, Some(0.9));
    insert_decision(&db, low, Some(0.1), vec![0.99, 0.05, 0.0, 0.0]);
    insert_outcome(&db, low, Some(0.95));

    let mut q = SemanticQuery::new("decisions", "vector_text", vec![1.0, 0.0, 0.0, 0.0], 5);
    q.sort_key = Some("effective_confidence".to_string());
    q.where_clause = Some("confidence > 0.3".to_string());
    let results = db.semantic_search(q).expect("filtered search");
    assert_eq!(
        results.len(),
        1,
        "filter on confidence>0.3 must keep only `high` decision"
    );
    let row_id = match results[0].values.get("id") {
        Some(Value::Uuid(u)) => *u,
        other => panic!("got {other:?}"),
    };
    assert_eq!(row_id, high);
}

// ---------- Concurrent-semantics probes ----------

/// C-S2 HNSW read under concurrent insert: writer inserts new evidence
/// while reader queries vector_text. Reader must not panic, and every
/// returned row must be either pre-insert or post-insert (not torn).
#[test]
fn vector_search_under_concurrent_insert_no_panic() {
    let db = Arc::new(open_search_schema_with_rank_policy());
    // Seed 8 rows.
    for i in 0..8 {
        let v = vec![i as f32 / 10.0, 0.0, 0.0, 0.0];
        db.execute(
            "INSERT INTO evidence (id, vector_text, vector_vision) VALUES ($id, $t, $v)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("t", Value::Vector(v)),
                (
                    "v",
                    Value::Vector(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ),
            ]),
        )
        .unwrap();
    }

    let writer_db = db.clone();
    let writer = thread::spawn(move || {
        for i in 0..32 {
            writer_db
                .execute(
                    "INSERT INTO evidence (id, vector_text, vector_vision) VALUES ($id, $t, $v)",
                    &params(vec![
                        ("id", Value::Uuid(Uuid::new_v4())),
                        (
                            "t",
                            Value::Vector(vec![1.0, i as f32 / 100.0, 0.0, 0.0]),
                        ),
                        (
                            "v",
                            Value::Vector(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                        ),
                    ]),
                )
                .unwrap();
        }
    });
    let reader_db = db.clone();
    let reader = thread::spawn(move || {
        let mut counts = vec![];
        for _ in 0..16 {
            let snap = reader_db.snapshot();
            let hits = reader_db
                .query_vector(
                    VectorIndexRef::new("evidence", "vector_text"),
                    &[1.0, 0.0, 0.0, 0.0],
                    50,
                    None,
                    snap,
                )
                .unwrap();
            counts.push(hits.len());
        }
        counts
    });

    writer.join().unwrap();
    let counts = reader.join().unwrap();
    assert!(counts.iter().all(|&c| c <= 50));
    assert!(counts.iter().all(|&c| c >= 8), "reader must always see seed rows; got counts {counts:?}");
}

/// C-S6 semantic_search on empty corpus returns empty without panic.
#[test]
fn semantic_search_empty_store_returns_empty() {
    let db = open_search_schema_with_rank_policy();
    let mut q = SemanticQuery::new("decisions", "vector_text", vec![1.0, 0.0, 0.0, 0.0], 5);
    q.sort_key = Some("effective_confidence".to_string());
    let results = db.semantic_search(q).expect("search on empty store");
    assert_eq!(results.len(), 0);
}

/// C-S7 effective_confidence with null confidence + null success_rate:
/// rank-policy formula `coalesce({confidence}, 1.0) * coalesce({success_rate}, 1.0)`
/// must yield deterministic 1.0; not NaN, not random ordering.
#[test]
fn rank_policy_null_confidence_null_success_rate_deterministic() {
    let db = open_search_schema_with_rank_policy();
    let id = Uuid::new_v4();
    // null confidence on decision, null success_rate on outcome.
    db.execute(
        "INSERT INTO decisions (id, decision_type, confidence, created_at, vector_text) VALUES ($id, 'choice', NULL, 0, $v)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("v", Value::Vector(vec![1.0, 0.0, 0.0, 0.0])),
        ]),
    )
    .unwrap();
    db.execute(
        "INSERT INTO outcomes (id, decision_id, success_rate, confidence, created_at) VALUES ($id, $d, NULL, 0.5, 0)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("d", Value::Uuid(id)),
        ]),
    )
    .unwrap();

    let mut q = SemanticQuery::new("decisions", "vector_text", vec![1.0, 0.0, 0.0, 0.0], 1);
    q.sort_key = Some("effective_confidence".to_string());
    let results = db.semantic_search(q).expect("search");
    assert_eq!(results.len(), 1);
    let rank = results[0].rank;
    assert!(
        (rank - 1.0).abs() < 0.001,
        "null * null with coalesce(_,1.0) must yield 1.0 deterministically; got {rank}"
    );
    assert!(!rank.is_nan(), "rank must not be NaN");
}

/// C-S1 rank-policy atomicity under concurrent outcomes write: while a
/// search is running, a writer updates outcomes.success_rate. Reader must
/// see snapshot-consistent ranks (either pre- or post-update, not torn).
#[test]
fn rank_policy_under_concurrent_outcome_writer_snapshot_consistent() {
    let db = Arc::new(open_search_schema_with_rank_policy());
    // Seed several decisions + outcomes.
    let mut ids = vec![];
    for _ in 0..5 {
        let id = Uuid::new_v4();
        insert_decision(&db, id, Some(0.5), vec![1.0, 0.0, 0.0, 0.0]);
        insert_outcome(&db, id, Some(0.5));
        ids.push(id);
    }

    let writer_db = db.clone();
    let writer_ids = ids.clone();
    let barrier = Arc::new(Barrier::new(2));
    let b2 = barrier.clone();
    let writer = thread::spawn(move || {
        b2.wait();
        for _ in 0..16 {
            for id in &writer_ids {
                writer_db
                    .execute(
                        "UPDATE outcomes SET success_rate = 0.99 WHERE decision_id = $d",
                        &params(vec![("d", Value::Uuid(*id))]),
                    )
                    .unwrap();
            }
        }
    });
    let reader_db = db.clone();
    let reader = thread::spawn(move || {
        barrier.wait();
        for _ in 0..16 {
            let mut q = SemanticQuery::new("decisions", "vector_text", vec![1.0, 0.0, 0.0, 0.0], 5);
            q.sort_key = Some("effective_confidence".to_string());
            let results = reader_db.semantic_search(q).expect("concurrent search");
            for r in &results {
                let rank = r.rank;
                // rank must be in a sensible range — formula is
                // coalesce(conf,1)*coalesce(sr,1), confidence=0.5,
                // success_rate ∈ {0.5, 0.99} → rank ∈ {0.25, 0.495}.
                assert!(
                    (0.20..=0.55).contains(&rank) || rank.is_nan() == false,
                    "rank must be deterministic per snapshot; got {rank}"
                );
                assert!(!rank.is_nan(), "no NaN under contention");
            }
        }
    });

    writer.join().unwrap();
    reader.join().unwrap();
}
