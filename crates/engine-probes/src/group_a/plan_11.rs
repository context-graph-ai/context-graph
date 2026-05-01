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
#[test]
fn reindex_under_concurrent_reader_no_torn_vector() {
    let db = Arc::new(open_embedding_schema());
    let mut row_ids: Vec<RowId> = vec![];
    // Seed 16 rows.
    {
        let tx = db.begin();
        for i in 0..16 {
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
                vec![i as f32 / 16.0, 0.0, 0.0, 0.0],
            )
            .unwrap();
            row_ids.push(row_id);
        }
        db.commit(tx).unwrap();
    }

    let writer_db = db.clone();
    let writer_rows = row_ids.clone();
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
                    vec![1.0, 0.0, 0.0, 0.0],
                )
                .unwrap();
            writer_db.commit(tx).unwrap();
        }
    });
    let reader_db = db.clone();
    let reader = thread::spawn(move || {
        for _ in 0..32 {
            let snap = reader_db.snapshot();
            let hits = reader_db
                .query_vector(
                    VectorIndexRef::new("evidence", "vector_text"),
                    &[0.5, 0.0, 0.0, 0.0],
                    50,
                    None,
                    snap,
                )
                .unwrap();
            // hits.len() must always be in [0, 16] — never more (no
            // duplicate vectors per row).
            assert!(
                hits.len() <= 16,
                "duplicate vectors per row would indicate non-atomic reindex; got {} hits",
                hits.len()
            );
            // Each hit's score must be in [-1.0, 1.0] (cosine).
            for (_, score) in &hits {
                assert!(
                    score.is_finite(),
                    "torn vector or wrong-dim read produces NaN; got {score}"
                );
            }
        }
    });
    writer.join().unwrap();
    reader.join().unwrap();
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
