//! Plan 11 cg-embedding-spaces — engine-probe sweep.
//!
//! See `.claude/plans/context-graph/r0/track1-enumeration/11-cg-embedding-spaces-primitives.md`.
//!
//! Targets NamedVectorIndexes (engine-gaps-for-cg.md §14, released v1.0.0).

use contextdb_core::{Error, RowId, Value, VectorIndexRef};
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

fn open_embedding_schema() -> Database {
    let db = Database::open_memory();
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE evidence (
            id UUID PRIMARY KEY,
            embedding_space_id TEXT,
            vector_text VECTOR(4),
            vector_vision VECTOR(8)
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE digests (
            id UUID PRIMARY KEY,
            embedding_space_id TEXT,
            vector_text VECTOR(4)
        )",
        &p,
    )
    .unwrap();
    db
}

// ---------- Single-primitive probes ----------

/// 11 single-primitive: per-(table, column) named vector indexes coexist on
/// the same table without conflict. evidence.vector_text and
/// evidence.vector_vision keep separate dimensions and separate index state.
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::insert_vector.
#[test]
fn evidence_text_and_vision_indexes_coexist() {
    let db = open_embedding_schema();
    let id = Uuid::new_v4();
    let tx = db.begin();
    let row_id = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(id)),
                ("embedding_space_id", Value::Text("text_default".into())),
            ]),
        )
        .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"),
        row_id,
        vec![1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_vision"),
        row_id,
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    db.commit(tx).unwrap();

    // Inserting a wrong-dim vector into vector_text MUST surface dim error
    // independently of vector_vision succeeding earlier.
    let tx2 = db.begin();
    let bad = db.insert_vector(
        tx2,
        VectorIndexRef::new("evidence", "vector_text"),
        RowId(9999),
        vec![1.0, 2.0, 3.0, 4.0, 5.0], // wrong dim 5
    );
    let _ = db.rollback(tx2);
    assert!(matches!(bad, Err(Error::VectorIndexDimensionMismatch { .. })));
}

/// 11 single-primitive: digests.vector_text and evidence.vector_text are
/// distinct named indexes — same column name, different table. Inserting
/// into one does not appear in the other.
///
/// Round-2 reviewer fix: the previous version inserted DIFFERENT vectors
/// into the two tables and asserted each top-1 returned a different RowId.
/// A wrong-impl that merges all `vector_text` columns into one global
/// index keyed by column-name only (ignoring the table component of
/// VectorIndexRef) would still pass — the two RowIds differ regardless.
///
/// This version inserts the SAME vector `[1,0,0,0]` into BOTH
/// `evidence.vector_text` AND `digests.vector_text`, then queries each
/// per-table index with that exact vector. Per-table top-k must return
/// ONLY that table's row — never the other table's row, even though both
/// rows are equally close to the query. A merged-index wrong-impl would
/// return both rows in the top-2 of each query.
#[test]
fn digests_and_evidence_vector_text_are_distinct_indexes() {
    let db = open_embedding_schema();
    let evi_id = Uuid::new_v4();
    let dig_id = Uuid::new_v4();

    // Same vector value in both tables. The per-(table, column) routing
    // contract requires that querying evidence/vector_text never returns
    // the digests row, and vice versa.
    let shared_vec = vec![1.0_f32, 0.0, 0.0, 0.0];

    let tx = db.begin();
    let evi_row = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(evi_id)),
                ("embedding_space_id", Value::Text("text_default".into())),
            ]),
        )
        .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"),
        evi_row,
        shared_vec.clone(),
    )
    .unwrap();
    let dig_row = db
        .insert_row(
            tx,
            "digests",
            vals(vec![
                ("id", Value::Uuid(dig_id)),
                ("embedding_space_id", Value::Text("text_default".into())),
            ]),
        )
        .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("digests", "vector_text"),
        dig_row,
        shared_vec.clone(),
    )
    .unwrap();
    db.commit(tx).unwrap();

    let snap = db.snapshot();
    // Query each per-table index with the SHARED vector. A correct engine
    // returns exactly the one row that lives in that table's index. A
    // merged-index wrong-impl returns both rows.
    let evi_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &shared_vec,
            10,
            None,
            snap,
        )
        .unwrap();
    let dig_hits = db
        .query_vector(
            VectorIndexRef::new("digests", "vector_text"),
            &shared_vec,
            10,
            None,
            snap,
        )
        .unwrap();
    assert_eq!(
        evi_hits.len(),
        1,
        "evidence/vector_text query must return exactly 1 row (the evidence row); \
         got {} rows — merged-index wrong-impl would return both tables' rows",
        evi_hits.len()
    );
    assert_eq!(
        dig_hits.len(),
        1,
        "digests/vector_text query must return exactly 1 row (the digests row); \
         got {} rows — merged-index wrong-impl would return both tables' rows",
        dig_hits.len()
    );
    assert_eq!(
        evi_hits[0].0, evi_row,
        "evidence/vector_text top-1 must be the evidence row, not the digests row"
    );
    assert_eq!(
        dig_hits[0].0, dig_row,
        "digests/vector_text top-1 must be the digests row, not the evidence row"
    );
    assert_ne!(
        evi_hits[0].0, dig_hits[0].0,
        "per-(table, column) routing: each index must return only its own table's row"
    );
}

/// 11 single-primitive (C-E2): wrong-dim caller-supplied embedding rejected
/// at write time; the rejection does NOT silently fall through to a default.
#[test]
fn precomputed_embedding_dim_mismatch_rejected_at_write() {
    let db = open_embedding_schema();
    let tx = db.begin();
    let row_id = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("embedding_space_id", Value::Text("text_default".into())),
            ]),
        )
        .unwrap();
    let result = db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"), // declared dim 4
        row_id,
        vec![1.0, 2.0], // wrong dim 2
    );
    let _ = db.rollback(tx);
    assert!(
        matches!(result, Err(Error::VectorIndexDimensionMismatch { .. })),
        "wrong-dim precomputed embedding must surface dim mismatch; got {result:?}"
    );
}

// ---------- Multi-primitive combinations ----------

/// 11 multi-primitive: atomic reindex within one tx. Delete + reinsert
/// in same tx — engine must handle this without leaving a torn vector.
/// Wrong-impl: delete commits before insert, leaving the index empty mid-tx.
#[test]
fn reindex_atomic_delete_insert_same_tx() {
    let db = open_embedding_schema();
    let id = Uuid::new_v4();
    let tx0 = db.begin();
    let row_id = db
        .insert_row(
            tx0,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(id)),
                ("embedding_space_id", Value::Text("text_default".into())),
            ]),
        )
        .unwrap();
    db.insert_vector(
        tx0,
        VectorIndexRef::new("evidence", "vector_text"),
        row_id,
        vec![1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    db.commit(tx0).unwrap();

    let snap0 = db.snapshot();
    let pre = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &[1.0, 0.0, 0.0, 0.0],
            10,
            None,
            snap0,
        )
        .unwrap();
    assert_eq!(pre.len(), 1);

    // Reindex: delete the old vector + insert the new vector in one tx.
    let tx1 = db.begin();
    db.delete_vector(
        tx1,
        VectorIndexRef::new("evidence", "vector_text"),
        row_id,
    )
    .unwrap();
    db.insert_vector(
        tx1,
        VectorIndexRef::new("evidence", "vector_text"),
        row_id,
        vec![0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();
    db.commit(tx1).unwrap();

    let snap1 = db.snapshot();
    // Query against the new vector returns the row.
    let new_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &[0.0, 1.0, 0.0, 0.0],
            10,
            None,
            snap1,
        )
        .unwrap();
    assert_eq!(new_hits.len(), 1, "new vector must be queryable post-reindex");
    let total = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &[0.5, 0.5, 0.0, 0.0],
            10,
            None,
            snap1,
        )
        .unwrap();
    assert_eq!(
        total.len(),
        1,
        "exactly one entry per row (reindex must replace, not duplicate)"
    );
}

// ---------- Concurrent-semantics probes ----------

/// C-E1 concurrent reindex vs read: while a reindex tx is in flight, a
/// reader queries the index. Reader must see either pre- or post-reindex
/// state — never a torn vector (e.g., wrong-dim).
///
/// Round-3 reviewer fix: previous version asserted only `hits.len() <= 16`
/// and `score.is_finite()` per observation. This passes under (a) the
/// canonical lens-(a) wrong-impl that silently drops every concurrent
/// `insert_vector` after `delete_vector` (the index drains to <16 vectors,
/// every observation passes the upper bound; never asserts a lower bound)
/// and (b) the torn-vector lens shape the docstring names — a half-
/// overwritten vector produces a finite-but-wrong score that still passes
/// `is_finite`.
///
/// Strengthened per the reviewer's prescription:
///   1. Bind observed scores to the legal pre/post cosine set: per-
///      observation, every score must equal `cos(q, v_pre[i])` for some
///      `i` OR `cos(q, v_post)`, within float tolerance. A score not in
///      this set is a torn vector.
///   2. Bind the post-join state: `final_hits.len() == 16` (catches lens
///      (a) silent drop — every reindexed row vanishing leaves <16) AND
///      every score in `final_hits` equals `cos(q, v_post)` within
///      tolerance (catches a wrong-impl that retains pre-reindex vectors
///      after reindex commits).
///
/// Vector geometry (chosen so the legal pre/post cosines are distinct,
/// non-degenerate, and a torn intermediate is detectable):
///   - query `q = [1, 0, 0, 0]` (unit vector along axis 0).
///   - pre-reindex vectors `v_pre[i] = [cos(theta_i), sin(theta_i), 0, 0]`
///     for `i` in 0..16 with `theta_i = (i + 1) * PI / 64`. Each is a unit
///     vector with `cos(q, v_pre[i]) = cos(theta_i)`, distinct in
///     [cos(PI/4), cos(PI/64)] = roughly [0.7071, 0.9988].
///   - post-reindex vector `v_post = [0, 0, 1, 0]` (unit vector along axis
///     2). `cos(q, v_post) = 0` exactly. A wrong-impl that leaves the
///     pre-reindex vector in place after reindex commits would yield a
///     final score in [0.7071, 0.9988], failing the post-join score check.
///
/// Loop: 30 rounds of (re-seed → spawn writer + reader → join) to surface
/// the contention window across schedules.
#[test]
fn reindex_under_concurrent_reader_no_torn_vector() {
    use std::f32::consts::PI;

    const TOL: f32 = 1e-5;
    const ROUNDS: usize = 30;
    const N: usize = 16;

    // Pre-reindex vectors: v_pre[i] = [cos(theta_i), sin(theta_i), 0, 0].
    // theta_i = (i+1) * PI / 64 — 16 distinct angles in (0, PI/4].
    let pre_vectors: Vec<Vec<f32>> = (0..N)
        .map(|i| {
            let theta = (i as f32 + 1.0) * PI / 64.0;
            vec![theta.cos(), theta.sin(), 0.0, 0.0]
        })
        .collect();
    let post_vector: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0];
    let query: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];

    // Legal cosine set (pre + post) — each observed score must lie in this
    // set within `TOL`. q is a unit vector along axis 0, so cos(q, v) = v[0]
    // (since |q| = 1 and each v[i] has |v| = 1 by construction).
    let pre_cosines: Vec<f32> = pre_vectors.iter().map(|v| v[0]).collect();
    let post_cosine: f32 = post_vector[0]; // 0.0
    let in_legal_set = |score: f32| -> bool {
        if (score - post_cosine).abs() <= TOL {
            return true;
        }
        pre_cosines.iter().any(|c| (score - c).abs() <= TOL)
    };

    for round in 0..ROUNDS {
        let db = Arc::new(open_embedding_schema());
        let mut row_ids: Vec<RowId> = vec![];
        // Seed N rows with v_pre[i].
        {
            let tx = db.begin();
            for v_pre in &pre_vectors {
                let id = Uuid::new_v4();
                let row_id = db
                    .insert_row(
                        tx,
                        "evidence",
                        vals(vec![
                            ("id", Value::Uuid(id)),
                            ("embedding_space_id", Value::Text("text_default".into())),
                        ]),
                    )
                    .unwrap();
                db.insert_vector(
                    tx,
                    VectorIndexRef::new("evidence", "vector_text"),
                    row_id,
                    v_pre.clone(),
                )
                .unwrap();
                row_ids.push(row_id);
            }
            db.commit(tx).unwrap();
        }

        // Sanity: pre-reindex query returns exactly N hits with scores in
        // the pre-cosine set. (Catches a setup bug before the contention
        // window opens.)
        {
            let snap = db.snapshot();
            let pre_hits = db
                .query_vector(
                    VectorIndexRef::new("evidence", "vector_text"),
                    &query,
                    50,
                    None,
                    snap,
                )
                .unwrap();
            assert_eq!(
                pre_hits.len(),
                N,
                "round {round}: pre-reindex hit count must equal {N}; got {}",
                pre_hits.len()
            );
            for (_, score) in &pre_hits {
                assert!(
                    pre_cosines.iter().any(|c| (score - c).abs() <= TOL),
                    "round {round}: pre-reindex score {score} not in pre-cosine set"
                );
            }
        }

        let writer_db = db.clone();
        let writer_rows = row_ids.clone();
        let writer_post_vector = post_vector.clone();
        let writer = thread::spawn(move || {
            for rid in writer_rows {
                let tx = writer_db.begin();
                writer_db
                    .delete_vector(tx, VectorIndexRef::new("evidence", "vector_text"), rid)
                    .unwrap();
                writer_db
                    .insert_vector(
                        tx,
                        VectorIndexRef::new("evidence", "vector_text"),
                        rid,
                        writer_post_vector.clone(),
                    )
                    .unwrap();
                writer_db.commit(tx).unwrap();
            }
        });

        let reader_db = db.clone();
        let reader_query = query.clone();
        let reader_pre_cosines = pre_cosines.clone();
        let reader_post_cosine = post_cosine;
        let reader = thread::spawn(move || {
            for _ in 0..64 {
                let snap = reader_db.snapshot();
                let hits = reader_db
                    .query_vector(
                        VectorIndexRef::new("evidence", "vector_text"),
                        &reader_query,
                        50,
                        None,
                        snap,
                    )
                    .unwrap();
                // No duplicate vectors per row (upper bound).
                assert!(
                    hits.len() <= N,
                    "duplicate vectors per row would indicate non-atomic reindex; got {} hits",
                    hits.len()
                );
                // Per-observation legal-set binding: every score must be in
                // the pre/post cosine set within tolerance. A torn vector
                // (half-overwritten) would yield a finite-but-wrong score
                // that fails this membership.
                for (_, score) in &hits {
                    assert!(
                        score.is_finite(),
                        "torn vector or wrong-dim read produces NaN; got {score}"
                    );
                    let in_set = (score - reader_post_cosine).abs() <= TOL
                        || reader_pre_cosines
                            .iter()
                            .any(|c| (score - c).abs() <= TOL);
                    assert!(
                        in_set,
                        "score {score} not in legal pre/post cosine set — torn vector"
                    );
                }
            }
        });
        writer.join().unwrap();
        reader.join().unwrap();

        // Post-join: the reindex completed; the index must reflect the
        // post-reindex state for ALL N rows.
        let snap = db.snapshot();
        let final_hits = db
            .query_vector(
                VectorIndexRef::new("evidence", "vector_text"),
                &query,
                50,
                None,
                snap,
            )
            .unwrap();
        assert_eq!(
            final_hits.len(),
            N,
            "round {round}: post-reindex hit count must equal {N}; got {} \
             — wrong-impl silently dropped insert_vector after delete_vector",
            final_hits.len()
        );
        for (rid, score) in &final_hits {
            // Use legal-set check too to surface torn-vector residuals.
            assert!(
                in_legal_set(*score),
                "round {round}: rid {rid:?} post-join score {score} not in legal set"
            );
            assert!(
                (score - post_cosine).abs() <= TOL,
                "round {round}: rid {rid:?} post-join score {score} != post-reindex cosine {post_cosine} \
                 — wrong-impl retained pre-reindex vector or fabricated a torn intermediate"
            );
        }
    }
}

/// C-E2 concurrent inserts with mixed-dim precomputed embeddings: each tx
/// either succeeds or fails based on its own dim; no silent fallback.
#[test]
fn concurrent_inserts_mixed_dim_independent_outcomes() {
    let db = Arc::new(open_embedding_schema());
    let barrier = Arc::new(Barrier::new(8));
    let mut handles = vec![];
    for i in 0..8 {
        let db = db.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || -> Result<(), Error> {
            let id = Uuid::new_v4();
            let tx = db.begin();
            let row_id = db.insert_row(
                tx,
                "evidence",
                vals(vec![
                    ("id", Value::Uuid(id)),
                    ("embedding_space_id", Value::Text("text_default".into())),
                ]),
            )?;
            // Even threads: correct dim. Odd threads: wrong dim.
            let v = if i % 2 == 0 {
                vec![1.0, 0.0, 0.0, 0.0]
            } else {
                vec![1.0, 0.0]
            };
            barrier.wait();
            let r = db.insert_vector(tx, VectorIndexRef::new("evidence", "vector_text"), row_id, v);
            if r.is_ok() {
                db.commit(tx)?;
            } else {
                let _ = db.rollback(tx);
            }
            r
        }));
    }
    let mut ok = 0;
    let mut err = 0;
    for h in handles {
        match h.join().unwrap() {
            Ok(_) => ok += 1,
            Err(Error::VectorIndexDimensionMismatch { .. }) => err += 1,
            other => panic!("unexpected error: {other:?} (must be VectorIndexDimensionMismatch)"),
        }
    }
    assert_eq!(ok, 4, "4 correct-dim insertions must succeed");
    assert_eq!(err, 4, "4 wrong-dim insertions must fail with dim mismatch");
    let snap = db.snapshot();
    let rows = db.scan("evidence", snap).unwrap();
    assert_eq!(rows.len(), 4, "only the 4 successful tx commits leave rows");
}

/// C-E3 reindex idempotency: running a reindex twice on the same row
/// produces the same final state. Wrong-impl: phantom rows, vector drift.
#[test]
fn reindex_idempotent_on_same_row() {
    let db = open_embedding_schema();
    let id = Uuid::new_v4();
    let tx = db.begin();
    let row_id = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(id)),
                ("embedding_space_id", Value::Text("text_default".into())),
            ]),
        )
        .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"),
        row_id,
        vec![1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    db.commit(tx).unwrap();

    for _ in 0..3 {
        let tx = db.begin();
        db.delete_vector(tx, VectorIndexRef::new("evidence", "vector_text"), row_id)
            .unwrap();
        db.insert_vector(
            tx,
            VectorIndexRef::new("evidence", "vector_text"),
            row_id,
            vec![0.5, 0.5, 0.0, 0.0],
        )
        .unwrap();
        db.commit(tx).unwrap();
    }

    let snap = db.snapshot();
    let rows = db.scan("evidence", snap).unwrap();
    assert_eq!(rows.len(), 1, "no phantom rows from reindex");
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &[0.5, 0.5, 0.0, 0.0],
            10,
            None,
            snap,
        )
        .unwrap();
    assert_eq!(hits.len(), 1, "exactly one vector entry per row after repeated reindex");
}

// ---------- BLOCKED entries ----------
// C-E4 embedding-space declaration validation race — `Store::open` is a
// cg-side Plan 11 surface, not an engine primitive; the engine
// `Database::open` does not declare embedding spaces beyond the SQL DDL.
// Catalogued as out-of-scope for engine probes.
//
// C-E5 reindex respects scope (`embedding_space_id = A` only) — this is a
// cg-side filter applied before SELECT/UPDATE; the engine has no
// "reindex" primitive. The engine probe coverage is the per-row
// delete+insert atomicity above.
//
// C-E6 internal-vs-external metadata distinction — provenance flag is a
// cg-side JSON metadata column; the engine has no provenance-aware
// reindex skip primitive. Out of engine scope.
//
// C-E7 EmbeddingGemma 1k corpus dim 768 — Candle integration smoke test;
// not an engine primitive. Out of engine probe scope.
