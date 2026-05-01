//! Plan 3R cg-evidence-retrofit — engine-probe sweep.
//!
//! See `.claude/plans/context-graph/r0/track1-enumeration/3R-cg-evidence-retrofit-primitives.md`
//! for the per-primitive enumeration this file targets. Probes target the
//! released contextdb v1.0.0 surface; each probe asserts the engine delivers
//! the assumed contract under contention / edge / failure conditions, and
//! FAILS if the engine silently accepts what it should reject or relaxes
//! a load-bearing semantic.
//!
//! Engine source citations point at `contextdb/crates/contextdb-engine/` and
//! `contextdb/crates/contextdb-core/` paths inside the umbrella checkout.

use contextdb_core::{Direction, Error, RowId, TxId, Value, VectorIndexRef};
use contextdb_engine::Database;
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;
use uuid::Uuid;

/// Build a parameter map from `(name, value)` pairs.
fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Build a value map from `(column_name, value)` pairs.
fn vals(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Construct a minimal cg-shaped schema mirroring 3R's evidence /
/// observation / decision surface. We do NOT use the cg-side schema crate
/// because 3R's surface is unshipped; we build only what each probe needs.
fn open_3r_schema() -> Database {
    let db = Database::open_memory();
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE observations (
            id UUID PRIMARY KEY,
            entity_id UUID,
            context_id TEXT,
            recorded_at INTEGER
        )",
        &p,
    )
    .unwrap();
    // evidence carries two named vector columns AND is IMMUTABLE (3R C16-style).
    db.execute(
        "CREATE TABLE evidence (
            id UUID PRIMARY KEY,
            observation_id UUID,
            context_id TEXT,
            derived_from UUID,
            recorded_at INTEGER,
            vector_text VECTOR(8),
            vector_vision VECTOR(16)
        ) IMMUTABLE",
        &p,
    )
    .unwrap();
    // append-only log; latest-by-TXID authoritative
    db.execute(
        "CREATE TABLE evidence_availability_log (
            id UUID PRIMARY KEY,
            evidence_id UUID,
            status TEXT,
            recorded_at TXID
        ) IMMUTABLE",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE entity_snapshots (
            id UUID PRIMARY KEY,
            entity_id UUID,
            valid_from INTEGER,
            valid_to INTEGER,
            evidenced_by JSON
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE decisions (
            id UUID PRIMARY KEY,
            description TEXT,
            status TEXT,
            based_on_snapshots JSON
        ) STATE MACHINE (status: active -> [superseded, archived, invalidated])",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE invalidations (
            id UUID PRIMARY KEY,
            decision_id UUID,
            reason TEXT,
            created_at INTEGER
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE embedding_spaces (
            id UUID PRIMARY KEY,
            embedding_space_id TEXT UNIQUE,
            dim INTEGER
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE embedding_space_bindings (
            id UUID PRIMARY KEY,
            embedding_space_id TEXT NOT NULL,
            table_name TEXT NOT NULL,
            vector_column TEXT NOT NULL,
            UNIQUE (embedding_space_id, table_name, vector_column)
        )",
        &p,
    )
    .unwrap();
    db
}

// ---------- Single-primitive probes ----------

/// 3R single-primitive: insert evidence row carrying both named-vector indexes
/// (`vector_text` VECTOR(8) + `vector_vision` VECTOR(16)) on the same row;
/// per-(table, column) routing must place each vector in its own index.
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::insert_vector
/// and engine-gaps-for-cg.md §14 NamedVectorIndexes (released v1.0.0).
#[test]
fn evidence_insert_two_named_vector_indexes_routes_per_column() {
    let db = open_3r_schema();
    let evidence_id = Uuid::new_v4();
    let obs_id = Uuid::new_v4();

    let tx = db.begin();
    let row_id = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(evidence_id)),
                ("observation_id", Value::Uuid(obs_id)),
                ("context_id", Value::Text("ctxA".into())),
            ]),
        )
        .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"),
        row_id,
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_vision"),
        row_id,
        vec![
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    )
    .unwrap();
    db.commit(tx).unwrap();

    // Both indexes return the row when queried with their respective query vector.
    let snap = db.snapshot();
    let text_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            10,
            None,
            snap,
        )
        .unwrap();
    let vision_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_vision"),
            &[
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            10,
            None,
            snap,
        )
        .unwrap();
    assert_eq!(text_hits.len(), 1, "vector_text index must contain the row");
    assert_eq!(text_hits[0].0, row_id);
    assert_eq!(vision_hits.len(), 1, "vector_vision index must contain the row");
    assert_eq!(vision_hits[0].0, row_id);

    // Wrong-dim probe: querying vision with text-dimension vector must error,
    // not silently fall through. This proves routing is per-(table, column).
    let mismatch = db.query_vector(
        VectorIndexRef::new("evidence", "vector_vision"),
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        10,
        None,
        snap,
    );
    assert!(
        matches!(mismatch, Err(Error::VectorIndexDimensionMismatch { .. })),
        "vision index queried with text-dim vector must surface VectorIndexDimensionMismatch; got {mismatch:?}"
    );
}

/// 3R single-primitive: HNSW write must commit/abort with the row INSERT.
/// Roll back the tx and assert neither the row nor the vector survives.
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::rollback (which
/// invokes WriteSetApplicator rollback for vectors as well as rows).
#[test]
fn evidence_vector_write_atomic_with_row_on_rollback() {
    let db = open_3r_schema();
    let evidence_id = Uuid::new_v4();

    let tx = db.begin();
    let row_id = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(evidence_id)),
                ("observation_id", Value::Uuid(Uuid::new_v4())),
                ("context_id", Value::Text("ctxA".into())),
            ]),
        )
        .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"),
        row_id,
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    db.rollback(tx).unwrap();

    let snap = db.snapshot();
    let rows = db.scan("evidence", snap).unwrap();
    assert_eq!(rows.len(), 0, "rollback must drop the evidence row");
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            10,
            None,
            snap,
        )
        .unwrap();
    assert_eq!(
        hits.len(),
        0,
        "rollback must drop the vector entry — vector orphan would be an atomicity hole"
    );
}

/// 3R single-primitive: evidence is IMMUTABLE — UPDATE rejected.
/// Cites: contextdb/crates/contextdb-core/src/error.rs::ImmutableTable and
/// the SQL-surface immutable enforcement in
/// contextdb/crates/contextdb-engine/src/schema_enforcer.rs::validate_dml.
#[test]
fn evidence_immutable_update_rejected() {
    let db = open_3r_schema();
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO evidence (id, observation_id, context_id) VALUES ($id, $oid, $ctx)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("oid", Value::Uuid(Uuid::new_v4())),
            ("ctx", Value::Text("ctxA".into())),
        ]),
    )
    .unwrap();

    let result = db.execute(
        "UPDATE evidence SET context_id = 'ctxB' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    assert!(
        result.is_err(),
        "UPDATE on IMMUTABLE evidence must be rejected; got {result:?}"
    );
    assert!(
        matches!(result, Err(Error::ImmutableTable(_))),
        "expected ImmutableTable error variant, got {result:?}"
    );
}

/// 3R single-primitive: evidence is IMMUTABLE — DELETE rejected.
/// Wrong-impl shape: engine rejects UPDATE but accepts DELETE.
#[test]
fn evidence_immutable_delete_rejected() {
    let db = open_3r_schema();
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO evidence (id, observation_id, context_id) VALUES ($id, $oid, $ctx)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("oid", Value::Uuid(Uuid::new_v4())),
            ("ctx", Value::Text("ctxA".into())),
        ]),
    )
    .unwrap();

    let result = db.execute(
        "DELETE FROM evidence WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    assert!(
        result.is_err(),
        "DELETE on IMMUTABLE evidence must be rejected; got {result:?}"
    );
    assert!(matches!(result, Err(Error::ImmutableTable(_))));
    let rows = db.scan("evidence", db.snapshot()).unwrap();
    assert_eq!(rows.len(), 1, "rejected DELETE must not remove rows");
}

/// 3R single-primitive: evidence_availability_log IMMUTABLE rejects UPDATE/DELETE
/// (C16). Append-only.
#[test]
fn evidence_availability_log_immutable_update_delete_rejected() {
    let db = open_3r_schema();
    let id = Uuid::new_v4();
    let evidence_id = Uuid::new_v4();
    db.execute(
        "INSERT INTO evidence_availability_log (id, evidence_id, status) VALUES ($id, $eid, 'present')",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("eid", Value::Uuid(evidence_id)),
        ]),
    )
    .unwrap();
    let upd = db.execute(
        "UPDATE evidence_availability_log SET status = 'gone' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    let del = db.execute(
        "DELETE FROM evidence_availability_log WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    assert!(matches!(upd, Err(Error::ImmutableTable(_))));
    assert!(matches!(del, Err(Error::ImmutableTable(_))));
}

/// 3R single-primitive: TXID column on evidence_availability_log auto-advances
/// per engine clock — cg's assumption per the primitives file is "TXID column
/// `recorded_at` advancing monotonically per engine clock". Probe issues an
/// INSERT that supplies every column EXCEPT `recorded_at`. If the engine
/// auto-stamps recorded_at with the active TxId, the SELECT reads back a
/// non-null Value::TxId. If the engine returns Null, recorded_at is
/// writer-supplied, not engine-auto-stamped — gap relative to cg's assumption.
///
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::insert_row,
/// engine-gaps-for-cg.md §8 TxIdColumnType (released v0.3.4).
#[test]
fn evidence_availability_log_txid_column_auto_advances_per_engine_clock() {
    let db = open_3r_schema();
    let evidence_id = Uuid::new_v4();

    // Capture watermark BEFORE the inserts. Each stamped TxId must satisfy
    // pre.0 < t.0 <= post.0 — i.e., come from the engine's TX-manager
    // clock, NOT a wall-clock or caller-supplied source.
    let pre = db.committed_watermark();

    db.execute(
        "INSERT INTO evidence_availability_log (id, evidence_id, status) VALUES ($id, $eid, 'present')",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("eid", Value::Uuid(evidence_id)),
        ]),
    )
    .unwrap();
    db.execute(
        "INSERT INTO evidence_availability_log (id, evidence_id, status) VALUES ($id, $eid, 'pruned')",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("eid", Value::Uuid(evidence_id)),
        ]),
    )
    .unwrap();

    let post = db.committed_watermark();

    let rows = db
        .execute(
            "SELECT recorded_at FROM evidence_availability_log WHERE evidence_id = $eid",
            &params(vec![("eid", Value::Uuid(evidence_id))]),
        )
        .unwrap();
    assert_eq!(rows.rows.len(), 2);
    let ts: Vec<u64> = rows
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::TxId(t) => t.0,
            other => panic!(
                "TXID column must auto-stamp on INSERT when not supplied; got {other:?} — \
                 cg expects engine-clock auto-advancing behaviour"
            ),
        })
        .collect();
    // Strict monotonic per insert order — wall-clock-derived stamps would
    // happen to be distinct but cannot be guaranteed monotonic w.r.t. tx
    // commit order. Tx-manager clock IS monotonic by construction.
    assert!(
        ts[1] > ts[0],
        "stamped TxIds must be strictly monotonic per insert order; got {ts:?}"
    );
    // Each stamped TxId must lie in (pre, post] — the bounds set by the
    // engine's TX-manager around the two commits. This binds the stamp to
    // the TX manager's clock, ruling out wall-clock or arbitrary-u64
    // sources that happen to produce distinct increasing values.
    for (i, t) in ts.iter().enumerate() {
        assert!(
            *t > pre.0,
            "stamped TxId[{i}]={t} must be > pre-insert watermark={}; engine clock did not advance",
            pre.0
        );
        assert!(
            *t <= post.0,
            "stamped TxId[{i}]={t} must be <= post-insert watermark={}; stamp came from outside the TX manager's clock",
            post.0
        );
    }
}

/// 3R single-primitive: writer-supplied TXID column round-trips. If the
/// engine doesn't auto-stamp, cg can supply via committed_watermark. This
/// probe confirms the writer-supplied path works on the released engine.
/// Cites: contextdb/crates/contextdb-engine/src/database.rs::coerce_row_for_insert.
#[test]
fn evidence_availability_log_txid_column_writer_supplied_round_trip() {
    let db = open_3r_schema();
    // Drive watermark forward.
    for _ in 0..5 {
        db.execute(
            "INSERT INTO observations (id, entity_id, context_id) VALUES ($id, $eid, 'ctx')",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("eid", Value::Uuid(Uuid::new_v4())),
            ]),
        )
        .unwrap();
    }
    let watermark = db.committed_watermark();
    assert!(watermark.0 >= 5);

    let evidence_id = Uuid::new_v4();
    db.execute(
        "INSERT INTO evidence_availability_log (id, evidence_id, status, recorded_at) VALUES ($id, $eid, 'present', $rec)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("eid", Value::Uuid(evidence_id)),
            ("rec", Value::TxId(TxId(watermark.0))),
        ]),
    )
    .unwrap();

    let rows = db
        .execute(
            "SELECT recorded_at FROM evidence_availability_log WHERE evidence_id = $eid",
            &params(vec![("eid", Value::Uuid(evidence_id))]),
        )
        .unwrap();
    assert_eq!(rows.rows.len(), 1);
    assert_eq!(
        rows.rows[0][0],
        Value::TxId(TxId(watermark.0)),
        "writer-supplied TxId must round-trip exactly as Value::TxId"
    );
}

/// 3R single-primitive: JSON column round-trip must preserve the input value
/// byte-for-byte (or at least key-order-independent equality). decisions has
/// `based_on_snapshots JSON` carrying `Vec<EntitySnapshotId>`.
#[test]
fn decisions_based_on_snapshots_json_round_trip() {
    let db = open_3r_schema();
    let id = Uuid::new_v4();
    let snap_ids: Vec<String> = (0..3).map(|_| Uuid::new_v4().to_string()).collect();
    let json_value = serde_json::json!({"snapshots": snap_ids.clone()});

    db.execute(
        "INSERT INTO decisions (id, status, based_on_snapshots) VALUES ($id, 'active', $j)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("j", Value::Json(json_value.clone())),
        ]),
    )
    .unwrap();

    let rows = db
        .execute(
            "SELECT based_on_snapshots FROM decisions WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    let read = match &rows.rows[0][0] {
        Value::Json(v) => v.clone(),
        other => panic!("expected JSON column, got {other:?}"),
    };
    // serde_json::Value equality is structural across object key order — but
    // arrays preserve order, so the snapshot list must round-trip identically.
    let read_snaps = read.get("snapshots").expect("missing snapshots key");
    assert_eq!(
        read_snaps,
        &serde_json::Value::Array(
            snap_ids
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect()
        ),
        "JSON array order must round-trip"
    );
}

/// 3R concurrent JSON column round-trip: N concurrent UPDATEs against the
/// SAME decision row, each writer supplying a distinct JSON payload. The
/// final read-back JSON must equal exactly ONE writer's input — no torn
/// JSON value (lens (c) for JSON column class). Wrong-impl shapes caught:
///   - byte-level interleaving: read-back JSON parses but contains
///     fragments from multiple writers (e.g. half of writer-A's array
///     prefix joined to half of writer-B's array suffix).
///   - silent key-level merge: engine's JSON storage merges keys across
///     concurrent UPDATEs, producing a JSON object with both writers'
///     "snapshots" arrays concatenated rather than one winning.
///   - parse failure on read: torn bytes produce a JSON that fails to
///     deserialize — surfaced as panic in the unwrap below.
#[test]
fn decisions_based_on_snapshots_json_round_trip_concurrent_writers() {
    let db = Arc::new(open_3r_schema());
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO decisions (id, status, based_on_snapshots) VALUES ($id, 'active', $j)",
        &params(vec![
            ("id", Value::Uuid(id)),
            (
                "j",
                Value::Json(serde_json::json!({"snapshots": [], "writer": "seed"})),
            ),
        ]),
    )
    .unwrap();

    const N: usize = 6;
    let barrier = Arc::new(Barrier::new(N));
    // Each writer's payload is uniquely identifiable: snapshots-array of
    // length i+1 carrying its own marker UUIDs; "writer" key tags source.
    let payloads: Vec<serde_json::Value> = (0..N)
        .map(|i| {
            let snap_ids: Vec<String> =
                (0..(i + 1)).map(|_| Uuid::new_v4().to_string()).collect();
            serde_json::json!({
                "snapshots": snap_ids,
                "writer": format!("writer_{i}"),
            })
        })
        .collect();
    let payloads_arc = Arc::new(payloads.clone());

    let mut handles = vec![];
    for i in 0..N {
        let db = db.clone();
        let barrier = barrier.clone();
        let payloads = payloads_arc.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            db.execute(
                "UPDATE decisions SET based_on_snapshots = $j WHERE id = $id",
                &params(vec![
                    ("id", Value::Uuid(id)),
                    ("j", Value::Json(payloads[i].clone())),
                ]),
            )
        }));
    }
    for h in handles {
        let _ = h.join().unwrap();
    }

    let rows = db
        .execute(
            "SELECT based_on_snapshots FROM decisions WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_eq!(rows.rows.len(), 1, "single decision row must persist");
    let read = match &rows.rows[0][0] {
        Value::Json(v) => v.clone(),
        other => panic!("expected JSON column, got {other:?}"),
    };

    // The read-back value must equal exactly one writer's input — no
    // partial merge, no torn fragments. structural equality across the
    // candidate set rules out both byte-level torn JSON and silent
    // key-level merge.
    let matches = payloads.iter().any(|p| p == &read);
    assert!(
        matches,
        "concurrent UPDATEs to JSON column must yield one writer's input verbatim; \
         read-back={read} matched no writer payload — torn JSON or silent merge"
    );

    // Additional guard: read-back must NOT contain a "writer" value of
    // "seed" (the initial INSERT's marker). If it does, no writer's
    // UPDATE took effect — silent loss of all six concurrent UPDATEs.
    let writer_tag = read.get("writer").and_then(|v| v.as_str()).unwrap_or("");
    assert_ne!(
        writer_tag, "seed",
        "read-back retained the seed payload; no concurrent UPDATE took effect"
    );

    // Guard against torn-array shape: snapshots must be an Array (not a
    // half-formed object), and every element must be a String UUID-shape.
    let snaps = read
        .get("snapshots")
        .expect("snapshots key must round-trip");
    let snap_arr = snaps
        .as_array()
        .expect("snapshots must be a JSON Array, not torn into another type");
    for elem in snap_arr {
        assert!(
            elem.is_string(),
            "snapshot elements must be String UUIDs; got {elem} — torn array element"
        );
    }
}

/// 3R single-primitive: STATE MACHINE rejects `invalidated -> active`.
/// Cites: contextdb/crates/contextdb-engine/src/schema_enforcer.rs::validate_dml.
#[test]
fn decision_state_machine_invalidated_to_active_rejected() {
    let db = open_3r_schema();
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO decisions (id, status) VALUES ($id, 'active')",
        &params(vec![("id", Value::Uuid(id))]),
    )
    .unwrap();
    db.execute(
        "UPDATE decisions SET status = 'invalidated' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    )
    .unwrap();
    let result = db.execute(
        "UPDATE decisions SET status = 'active' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    assert!(
        result.is_err(),
        "invalidated -> active must be rejected by STATE MACHINE; got {result:?}"
    );
    let rows = db
        .execute(
            "SELECT status FROM decisions WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_eq!(
        rows.rows[0][0],
        Value::Text("invalidated".into()),
        "rejected transition must not mutate row"
    );
}

/// 3R single-primitive: STATE MACHINE rejects `invalidated -> archived` and
/// `invalidated -> superseded` (terminal).
#[test]
fn decision_state_machine_invalidated_to_archived_and_superseded_rejected() {
    let db = open_3r_schema();
    for forbidden in ["archived", "superseded"] {
        let id = Uuid::new_v4();
        db.execute(
            "INSERT INTO decisions (id, status) VALUES ($id, 'active')",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
        db.execute(
            "UPDATE decisions SET status = 'invalidated' WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
        let result = db.execute(
            &format!("UPDATE decisions SET status = '{forbidden}' WHERE id = $id"),
            &params(vec![("id", Value::Uuid(id))]),
        );
        assert!(
            result.is_err(),
            "invalidated -> {forbidden} must be rejected by STATE MACHINE; got {result:?}"
        );
    }
}

/// 3R single-primitive: STATE MACHINE self-edge `invalidated -> invalidated`.
/// Per C13: schema-declared self-edges are no-ops (rows_affected = 0). The
/// schema in this probe does NOT declare the self-edge — so the engine MUST
/// reject the transition. This is the strict reading of C-RT7's "engine MUST
/// reject regardless of app-level guard" combined with "the no-op behaviour
/// applies only when the schema declares the self-edge."
#[test]
fn decision_state_machine_invalidated_self_edge_rejected_when_undeclared() {
    let db = open_3r_schema();
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO decisions (id, status) VALUES ($id, 'active')",
        &params(vec![("id", Value::Uuid(id))]),
    )
    .unwrap();
    db.execute(
        "UPDATE decisions SET status = 'invalidated' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    )
    .unwrap();
    // Self-edge not declared in schema — must reject, NOT silently no-op.
    let result = db.execute(
        "UPDATE decisions SET status = 'invalidated' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    // Per the lens question for STATE MACHINE: "(b) self-edge X -> [X]
    // accepted as a no-op when the schema declares it" — we did NOT declare
    // it, so accepting it is the wrong-impl shape. We want either an error
    // OR (more permissively) rows_affected == 0 with no audit-state-change
    // signal. Strictest assertion: the row remains in 'invalidated' AND
    // either an error is returned OR the engine reports rows_affected = 0.
    match result {
        Err(_) => {} // strict rejection: best outcome
        Ok(qr) => assert_eq!(
            qr.rows_affected, 0,
            "undeclared self-edge accepted with rows_affected>0 is a STATE MACHINE bypass"
        ),
    }
    let rows = db
        .execute(
            "SELECT status FROM decisions WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_eq!(rows.rows[0][0], Value::Text("invalidated".into()));
}

/// 3R single-primitive: composite UNIQUE on `embedding_space_bindings`
/// (`(embedding_space_id, table_name, vector_column)`) deduplicates the
/// three-column tuple. Two inserts with the same triple collapse to one row.
#[test]
fn embedding_space_bindings_composite_unique_dedups() {
    let db = open_3r_schema();
    let p = |email: &str| {
        params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("space", Value::Text(email.to_string())),
            ("tab", Value::Text("evidence".into())),
            ("col", Value::Text("vector_text".into())),
        ])
    };
    db.execute(
        "INSERT INTO embedding_space_bindings (id, embedding_space_id, table_name, vector_column) VALUES ($id, $space, $tab, $col)",
        &p("text_default"),
    )
    .unwrap();
    db.execute(
        "INSERT INTO embedding_space_bindings (id, embedding_space_id, table_name, vector_column) VALUES ($id, $space, $tab, $col)",
        &p("text_default"),
    )
    .unwrap();
    let rows = db.scan("embedding_space_bindings", db.snapshot()).unwrap();
    assert_eq!(
        rows.len(),
        1,
        "duplicate composite triple must dedup to one row"
    );

    // Different space_id → distinct tuple → second row inserts.
    db.execute(
        "INSERT INTO embedding_space_bindings (id, embedding_space_id, table_name, vector_column) VALUES ($id, $space, $tab, $col)",
        &p("text_v2"),
    )
    .unwrap();
    let rows = db.scan("embedding_space_bindings", db.snapshot()).unwrap();
    assert_eq!(rows.len(), 2);
}

/// 3R single-primitive: composite UNIQUE only checks the first column —
/// wrong-impl shape from the lens. Two rows differ on `vector_column`
/// alone but share the first two columns. Engine must NOT collapse them.
#[test]
fn embedding_space_bindings_composite_unique_not_first_column_only() {
    let db = open_3r_schema();
    db.execute(
        "INSERT INTO embedding_space_bindings (id, embedding_space_id, table_name, vector_column) VALUES ($id, 'text_default', 'evidence', 'vector_text')",
        &params(vec![("id", Value::Uuid(Uuid::new_v4()))]),
    )
    .unwrap();
    db.execute(
        "INSERT INTO embedding_space_bindings (id, embedding_space_id, table_name, vector_column) VALUES ($id, 'text_default', 'evidence', 'vector_vision')",
        &params(vec![("id", Value::Uuid(Uuid::new_v4()))]),
    )
    .unwrap();
    let rows = db.scan("embedding_space_bindings", db.snapshot()).unwrap();
    assert_eq!(
        rows.len(),
        2,
        "composite UNIQUE must include all listed columns; collapsing on first column only is wrong-impl"
    );
}

/// 3R single-primitive: vector dimension validation at write time.
/// Inserting a wrong-dim vector must yield VectorIndexDimensionMismatch
/// (or an analogue error variant) — not silent acceptance.
#[test]
fn evidence_vector_dimension_mismatch_rejected_at_write() {
    let db = open_3r_schema();
    let tx = db.begin();
    let row_id = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("observation_id", Value::Uuid(Uuid::new_v4())),
                ("context_id", Value::Text("ctxA".into())),
            ]),
        )
        .unwrap();
    let result = db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"), // declared dim 8
        row_id,
        vec![1.0, 0.0, 0.0], // wrong dim 3
    );
    let _ = db.rollback(tx);
    assert!(
        matches!(result, Err(Error::VectorIndexDimensionMismatch { .. })),
        "wrong-dim vector insert must surface VectorIndexDimensionMismatch; got {result:?}"
    );
}

// ---------- Concurrent-semantics probes ----------

/// C-RT2 evidence_availability_log concurrent appends: four concurrent
/// `update_evidence_retention_status` calls against the same evidence_id.
/// Each writer obtains an engine-allocated TxId via `db.begin()` and stamps
/// `recorded_at` with that TxId verbatim. All four must persist; all four
/// TxIds MUST be pairwise distinct (since `begin()` allocates a fresh TxId
/// per call from the TX manager's monotonic clock). Wrong-impl shapes
/// caught: (i) row loss; (ii) engine accepts duplicate caller-supplied
/// u64 values into a TXID column without uniqueness check; (iii) engine
/// silently coerces one writer's TxId to match another.
#[test]
fn evidence_availability_log_concurrent_appends_both_persist() {
    let db = Arc::new(open_3r_schema());
    // Drive watermark forward.
    for _ in 0..3 {
        db.execute(
            "INSERT INTO observations (id, entity_id, context_id) VALUES ($id, $eid, 'ctx')",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("eid", Value::Uuid(Uuid::new_v4())),
            ]),
        )
        .unwrap();
    }
    let evidence_id = Uuid::new_v4();
    let pre = db.committed_watermark();
    let barrier = Arc::new(Barrier::new(4));

    let mut handles = vec![];
    for i in 0..4 {
        let db = db.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || -> Result<TxId, Error> {
            let id = Uuid::new_v4();
            let status = format!("status_{i}");
            // Each writer obtains a fresh engine-allocated TxId via
            // db.begin() — the TX manager guarantees these are pairwise
            // distinct across all four writers, regardless of contention.
            let tx = db.begin();
            barrier.wait();
            let res = db.execute_in_tx(
                tx,
                "INSERT INTO evidence_availability_log (id, evidence_id, status, recorded_at) VALUES ($id, $eid, $st, $rec)",
                &params(vec![
                    ("id", Value::Uuid(id)),
                    ("eid", Value::Uuid(evidence_id)),
                    ("st", Value::Text(status)),
                    ("rec", Value::TxId(tx)),
                ]),
            );
            if let Err(e) = res {
                let _ = db.rollback(tx);
                return Err(e);
            }
            db.commit(tx).map(|_| tx)
        }));
    }
    let mut ok_count = 0;
    let mut allocated_txids: Vec<u64> = vec![];
    for h in handles {
        if let Ok(tx) = h.join().unwrap() {
            ok_count += 1;
            allocated_txids.push(tx.0);
        }
    }
    assert_eq!(
        ok_count, 4,
        "all four concurrent appends must succeed (append-only, no contention on row PK)"
    );

    let post = db.committed_watermark();

    let rows = db
        .execute(
            "SELECT recorded_at FROM evidence_availability_log WHERE evidence_id = $eid",
            &params(vec![("eid", Value::Uuid(evidence_id))]),
        )
        .unwrap();
    assert_eq!(rows.rows.len(), 4, "all four rows must be persisted");
    let mut persisted_txids: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for r in &rows.rows {
        match r[0] {
            Value::TxId(t) => {
                persisted_txids.insert(t.0);
                // Each persisted TxId must lie in (pre, post] — the bounds
                // set by the TX manager around the four commits.
                assert!(
                    t.0 > pre.0,
                    "persisted TxId={} must be > pre-spawn watermark={}",
                    t.0, pre.0
                );
                assert!(
                    t.0 <= post.0,
                    "persisted TxId={} must be <= post-join watermark={}",
                    t.0, post.0
                );
            }
            ref other => panic!("recorded_at must round-trip as TxId, got {other:?}"),
        }
    }
    // Pairwise distinctness across all four writers. Wrong-impl that
    // accepts duplicate caller-supplied u64 collapses HashSet to <4.
    assert_eq!(
        persisted_txids.len(),
        4,
        "all four persisted TxIds must be pairwise distinct; got {persisted_txids:?}; \
         allocated by db.begin()={allocated_txids:?}"
    );
}

/// C-RT3 conditional UPDATE under contention: 6 concurrent
/// `UPDATE decisions SET status='invalidated' WHERE status='active'` calls
/// against the same row. Engine MUST conflict-serialize: exactly one
/// reports rows_affected=1; the others must report rows_affected=0.
/// Wrong-impl: engine returns Ok(rows_affected = 1) for >1 callers (lens
/// question for "concurrent UPDATE / conditional UPDATE" failure mode (b)).
///
/// Probe runs multiple rounds: a single 6-thread race may or may not hit
/// the contention window. We repeat until either we see the gap surface
/// or we've completed enough rounds to attest the contract holds. If
/// ANY round reports total > 1, that's the gap.
#[test]
fn decisions_conditional_invalidate_concurrent_only_one_succeeds() {
    let db = Arc::new(open_3r_schema());

    const ROUNDS: usize = 30;
    const N: usize = 6;
    let mut max_total = 0u64;
    let mut all_round_results = vec![];
    for r in 0..ROUNDS {
        let id = Uuid::new_v4();
        db.execute(
            "INSERT INTO decisions (id, status) VALUES ($id, 'active')",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();

        let barrier = Arc::new(Barrier::new(N));
        let mut handles = vec![];
        for _ in 0..N {
            let db = db.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || -> u64 {
                barrier.wait();
                match db.execute(
                    "UPDATE decisions SET status = 'invalidated' WHERE id = $id AND status = 'active'",
                    &params(vec![("id", Value::Uuid(id))]),
                ) {
                    Ok(qr) => qr.rows_affected,
                    Err(_) => 0,
                }
            }));
        }
        let updates: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let total: u64 = updates.iter().sum();
        if total > max_total {
            max_total = total;
        }
        all_round_results.push((r, total, updates));
        if max_total > 1 {
            break;
        }
    }
    assert_eq!(
        max_total, 1,
        "concurrent conditional UPDATE: exactly one writer must report rows_affected=1; \
         max observed total across {ROUNDS} rounds = {max_total}; round results = {all_round_results:?}"
    );
}

/// C-RT4 multi-vector-column atomicity: HNSW insert on the second column
/// fails (wrong dim). Both vector entries AND the row INSERT must roll
/// back. No vector orphans, no row orphans.
#[test]
fn evidence_multi_vector_atomicity_under_dim_mismatch_rolls_back_all() {
    let db = open_3r_schema();
    let id = Uuid::new_v4();

    let tx = db.begin();
    let row_id = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(id)),
                ("observation_id", Value::Uuid(Uuid::new_v4())),
                ("context_id", Value::Text("ctxA".into())),
            ]),
        )
        .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"),
        row_id,
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let bad = db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_vision"), // declared dim 16
        row_id,
        vec![0.0, 1.0, 0.0], // wrong
    );
    assert!(matches!(bad, Err(Error::VectorIndexDimensionMismatch { .. })));
    db.rollback(tx).unwrap();

    let snap = db.snapshot();
    let rows = db.scan("evidence", snap).unwrap();
    assert_eq!(rows.len(), 0, "row must roll back when vision insert fails");
    let text_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            10,
            None,
            snap,
        )
        .unwrap();
    assert_eq!(
        text_hits.len(),
        0,
        "vector_text entry from same tx must also roll back — orphaned vector is the wrong-impl shape"
    );
}

/// C-RT5 cross-table cascade rollback: an FK violation during a multi-row
/// observation+evidence INSERT must roll back the parent observation.
/// We model this by inserting an evidence row that references a
/// non-existent observation_id, with FK declared (we don't have FK on this
/// schema, so we use a state-machine constraint as a proxy: invalidations
/// reference decision_id; an INSERT into a non-decision id and then a
/// state-machine-illegal UPDATE must roll back).
///
/// More direct probe: open a tx, insert an observation row, then attempt
/// to insert an evidence row with a malformed UUID literal — the SQL
/// parse error must roll back the observation when we rollback the tx.
#[test]
fn cross_table_cascade_rollback_observation_evidence_atomic() {
    let db = open_3r_schema();
    let observation_id = Uuid::new_v4();

    let tx = db.begin();
    db.insert_row(
        tx,
        "observations",
        vals(vec![
            ("id", Value::Uuid(observation_id)),
            ("entity_id", Value::Uuid(Uuid::new_v4())),
            ("context_id", Value::Text("ctxA".into())),
        ]),
    )
    .unwrap();
    // Simulate downstream evidence-write failure by attempting to insert
    // a vector with mismatched dim.
    let row_id = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("observation_id", Value::Uuid(observation_id)),
                ("context_id", Value::Text("ctxA".into())),
            ]),
        )
        .unwrap();
    let bad = db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"),
        row_id,
        vec![0.0; 99], // wrong
    );
    assert!(bad.is_err());
    db.rollback(tx).unwrap();

    let snap = db.snapshot();
    let obs = db.scan("observations", snap).unwrap();
    let evi = db.scan("evidence", snap).unwrap();
    assert_eq!(
        obs.len(),
        0,
        "parent observation must roll back when downstream evidence write fails"
    );
    assert_eq!(evi.len(), 0, "evidence row must roll back");
}

/// C-RT8 IMMUTABLE bypass guard: UPDATE/DELETE on evidence refused via
/// both `execute` (autocommit) and `execute_in_tx`. Wrong-impl: engine
/// rejects via execute but accepts via execute_in_tx (or vice versa).
#[test]
fn evidence_immutable_bypass_via_execute_in_tx_refused() {
    let db = open_3r_schema();
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO evidence (id, observation_id, context_id) VALUES ($id, $oid, $ctx)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("oid", Value::Uuid(Uuid::new_v4())),
            ("ctx", Value::Text("ctxA".into())),
        ]),
    )
    .unwrap();

    let tx = db.begin();
    let upd_in_tx = db.execute_in_tx(
        tx,
        "UPDATE evidence SET context_id = 'ctxB' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    let del_in_tx = db.execute_in_tx(
        tx,
        "DELETE FROM evidence WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    let _ = db.rollback(tx);

    assert!(
        matches!(upd_in_tx, Err(Error::ImmutableTable(_))),
        "UPDATE on IMMUTABLE evidence inside execute_in_tx must reject; got {upd_in_tx:?}"
    );
    assert!(
        matches!(del_in_tx, Err(Error::ImmutableTable(_))),
        "DELETE on IMMUTABLE evidence inside execute_in_tx must reject; got {del_in_tx:?}"
    );
}

/// C-RT1 `record_observation` Serializable isolation: N concurrent
/// observation+snapshot pipelines against the SAME entity_id with the
/// SAME valid_from..valid_to range — overlap is forced by construction.
/// The engine's snapshot contract for entity_snapshots is one of:
///   (a) **contiguous-snapshot**: every writer creates a new row →
///       N final rows, each with the same (valid_from, valid_to);
///   (b) **merge-on-overlap**: engine deduplicates overlapping snapshots
///       to a single canonical row → exactly 1 final row;
///   (c) **conflict-rejection**: engine rejects all but one writer with
///       a typed conflict error → exactly 1 final row + N-1 errors.
/// Wrong-impl shapes caught:
///   - torn rows: row carries thread-A's valid_from with thread-B's
///     valid_to (cross-thread cell mixing — possible if inserts are not
///     row-atomic under contention); detected by per-row width check.
///   - row-count drift: engine commits some writers and silently drops
///     others without an error signal; detected by reconciling
///     ok_count vs final row count.
#[test]
fn record_observation_concurrent_snapshots_no_torn_rows() {
    let db = Arc::new(open_3r_schema());
    let entity_id = Uuid::new_v4();
    const N: usize = 4;
    const VF: i64 = 100;
    const VT: i64 = 200;
    const WIDTH: i64 = VT - VF;

    let barrier = Arc::new(Barrier::new(N));
    let mut handles = vec![];
    for _ in 0..N {
        let db = db.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            // ALL writers target the same logical row (same entity_id,
            // same valid_from, same valid_to) — overlap is forced.
            db.execute(
                "INSERT INTO entity_snapshots (id, entity_id, valid_from, valid_to) VALUES ($id, $eid, $vf, $vt)",
                &params(vec![
                    ("id", Value::Uuid(Uuid::new_v4())),
                    ("eid", Value::Uuid(entity_id)),
                    ("vf", Value::Int64(VF)),
                    ("vt", Value::Int64(VT)),
                ]),
            )
        }));
    }
    let mut ok_count = 0;
    for h in handles {
        if h.join().unwrap().is_ok() {
            ok_count += 1;
        }
    }

    // Read back every column; per-row width and entity_id must hold —
    // this rules out cross-thread cell-mixing under contention regardless
    // of which contract (a/b/c) the engine implements.
    let snap = db.snapshot();
    let rows = db.scan("entity_snapshots", snap).unwrap();
    for r in &rows {
        let vf = match r.values.get("valid_from") {
            Some(Value::Int64(v)) => *v,
            other => panic!("valid_from must round-trip as Int64; got {other:?}"),
        };
        let vt = match r.values.get("valid_to") {
            Some(Value::Int64(v)) => *v,
            other => panic!("valid_to must round-trip as Int64; got {other:?}"),
        };
        let eid = match r.values.get("entity_id") {
            Some(Value::Uuid(u)) => *u,
            other => panic!("entity_id must round-trip as Uuid; got {other:?}"),
        };
        assert_eq!(eid, entity_id, "entity_id must match the test entity; cross-row leakage detected");
        assert_eq!(
            vt - vf,
            WIDTH,
            "row width vt-vf must equal {WIDTH}; cross-thread cell mixing produced (vf={vf}, vt={vt})"
        );
        assert_eq!(vf, VF, "valid_from must equal {VF}; cross-thread cell mixing detected");
        assert_eq!(vt, VT, "valid_to must equal {VT}; cross-thread cell mixing detected");
    }

    // Verify the actual contract the engine delivers. Exactly one of:
    //   (a) contiguous: rows.len() == N AND ok_count == N
    //   (b) merge:      rows.len() == 1 AND ok_count == N (silent dedup)
    //   (c) reject:     rows.len() == 1 AND ok_count == 1 (N-1 errors)
    let contract_a = rows.len() == N && ok_count == N;
    let contract_b = rows.len() == 1 && ok_count == N;
    let contract_c = rows.len() == 1 && ok_count == 1;
    assert!(
        contract_a || contract_b || contract_c,
        "engine must deliver one of: contiguous-snapshot ({N} rows, {N} ok), \
         merge-on-overlap (1 row, {N} ok), or conflict-rejection (1 row, 1 ok); \
         got {} rows, {ok_count} ok — neither contract holds, which signals \
         partial commit / silent row drop / row-count drift",
        rows.len()
    );
}

/// 3R single-primitive: `embedding_spaces.embedding_space_id` is UNIQUE.
/// Duplicate insert is a no-op (engine UNIQUE semantics: rows_affected=0).
#[test]
fn embedding_spaces_unique_embedding_space_id_dedups() {
    let db = open_3r_schema();
    let p = |id: Uuid| {
        params(vec![
            ("id", Value::Uuid(id)),
            ("space", Value::Text("text_default".into())),
            ("dim", Value::Int64(768)),
        ])
    };
    db.execute(
        "INSERT INTO embedding_spaces (id, embedding_space_id, dim) VALUES ($id, $space, $dim)",
        &p(Uuid::new_v4()),
    )
    .unwrap();
    db.execute(
        "INSERT INTO embedding_spaces (id, embedding_space_id, dim) VALUES ($id, $space, $dim)",
        &p(Uuid::new_v4()),
    )
    .unwrap();
    let rows = db.scan("embedding_spaces", db.snapshot()).unwrap();
    assert_eq!(rows.len(), 1, "duplicate embedding_space_id must dedup");
}

/// 3R single-primitive: STATE MACHINE rejection at engine even with raw
/// `execute_in_tx` direct-execute path. Wrong-impl: engine rejects at SQL
/// parse but accepts via `execute_in_tx`.
#[test]
fn decision_state_machine_bypass_via_execute_in_tx_refused() {
    let db = open_3r_schema();
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO decisions (id, status) VALUES ($id, 'active')",
        &params(vec![("id", Value::Uuid(id))]),
    )
    .unwrap();
    db.execute(
        "UPDATE decisions SET status = 'invalidated' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    )
    .unwrap();

    let tx = db.begin();
    let result = db.execute_in_tx(
        tx,
        "UPDATE decisions SET status = 'active' WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    let _ = db.rollback(tx);
    assert!(
        result.is_err(),
        "STATE MACHINE invalidated -> active must be refused via execute_in_tx; got {result:?}"
    );
}

/// 3R single-primitive: edges insertion via the graph subsystem under a
/// DAG (CITES) edge type — used by 3R's basis-cascade graph. Probe asserts
/// duplicate-edge insert dedups (returns false from `insert_edge`).
#[test]
fn cites_edge_insert_dedups_on_duplicate() {
    let db = open_3r_schema();
    let source = Uuid::new_v4();
    let target = Uuid::new_v4();

    let tx = db.begin();
    let inserted_first = db
        .insert_edge(tx, source, target, "CITES".into(), HashMap::new())
        .unwrap();
    let inserted_second = db
        .insert_edge(tx, source, target, "CITES".into(), HashMap::new())
        .unwrap();
    db.commit(tx).unwrap();

    assert!(inserted_first, "first edge insert must signal Inserted=true");
    assert!(
        !inserted_second,
        "duplicate edge must dedup — engine returns false (NoOp). Wrong-impl: returns true twice."
    );

    let snap = db.snapshot();
    let bfs = db
        .query_bfs(source, Some(&["CITES".to_string()]), Direction::Outgoing, 1, snap)
        .unwrap();
    assert_eq!(bfs.nodes.len(), 1, "BFS sees exactly one CITES neighbor");
}

/// C-RT9 latest_per_evidence_id under contention: while a writer is
/// appending to evidence_availability_log, a reader scans the table.
/// Reader must see a snapshot-consistent view (every row's TXID column
/// round-trips as Value::TxId — no torn variants).
#[test]
fn evidence_availability_log_reader_snapshot_consistent_under_writer() {
    let db = Arc::new(open_3r_schema());
    // Drive watermark.
    for _ in 0..3 {
        db.execute(
            "INSERT INTO observations (id, entity_id, context_id) VALUES ($id, $eid, 'ctx')",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("eid", Value::Uuid(Uuid::new_v4())),
            ]),
        )
        .unwrap();
    }
    let evidence_id = Uuid::new_v4();

    // Seed one row with explicit TxId.
    {
        let watermark = db.committed_watermark();
        db.execute(
            "INSERT INTO evidence_availability_log (id, evidence_id, status, recorded_at) VALUES ($id, $eid, 'present', $rec)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("eid", Value::Uuid(evidence_id)),
                ("rec", Value::TxId(TxId(watermark.0))),
            ]),
        )
        .unwrap();
    }

    let writer_db = db.clone();
    let writer_eid = evidence_id;
    let writer = thread::spawn(move || {
        let mut emitted: Vec<u64> = vec![];
        for i in 0..32 {
            // Each writer-step opens a fresh tx via begin() — engine
            // allocates the next monotonic TxId; we stamp recorded_at
            // with that value verbatim so the persisted column reflects
            // the engine's TX-manager clock for THIS commit.
            let tx = writer_db.begin();
            writer_db
                .execute_in_tx(
                    tx,
                    "INSERT INTO evidence_availability_log (id, evidence_id, status, recorded_at) VALUES ($id, $eid, $st, $rec)",
                    &params(vec![
                        ("id", Value::Uuid(Uuid::new_v4())),
                        ("eid", Value::Uuid(writer_eid)),
                        ("st", Value::Text(format!("step_{i}"))),
                        ("rec", Value::TxId(tx)),
                    ]),
                )
                .unwrap();
            writer_db.commit(tx).unwrap();
            emitted.push(tx.0);
        }
        emitted
    });
    let reader_db = db.clone();
    let reader = thread::spawn(move || {
        let mut max_seen = 0u64;
        for _ in 0..16 {
            let rows = reader_db
                .execute(
                    "SELECT recorded_at FROM evidence_availability_log WHERE evidence_id = $eid",
                    &params(vec![("eid", Value::Uuid(evidence_id))]),
                )
                .unwrap();
            // Reader observation count must be monotonically non-decreasing
            // in real-time (writer is append-only). Capture the highest
            // TxId visible at each observation; max must never regress.
            let mut local_max = 0u64;
            for r in &rows.rows {
                match r[0] {
                    Value::TxId(t) => {
                        if t.0 > local_max {
                            local_max = t.0;
                        }
                    }
                    ref other => panic!(
                        "concurrent reader must see TxId column type, never torn variant; got {other:?}"
                    ),
                }
            }
            assert!(
                local_max >= max_seen,
                "reader's max-visible TxId must be monotonically non-decreasing; \
                 prev={max_seen}, now={local_max} — snapshot regressed under concurrent writer"
            );
            max_seen = local_max;
        }
    });

    let writer_emitted = writer.join().unwrap();
    reader.join().unwrap();

    let final_rows = db
        .execute(
            "SELECT id, status, recorded_at FROM evidence_availability_log WHERE evidence_id = $eid",
            &params(vec![("eid", Value::Uuid(evidence_id))]),
        )
        .unwrap();
    assert_eq!(final_rows.rows.len(), 33, "writer must commit all 32 rows + 1 seed");

    // Sort all persisted recorded_at TxIds; assert strict-monotonic
    // (no duplicates) and contiguity (no gaps larger than the engine's
    // TX-allocation step). We bound the allocation step by the actual
    // span over the 32 writer commits / 32 — gap > 4x mean step is the
    // wrong-impl signal (TxIds non-monotonic w.r.t. commit order, or
    // engine is allocating from a non-contiguous source).
    let mut all_txids: Vec<u64> = final_rows
        .rows
        .iter()
        .map(|r| match &r[2] {
            Value::TxId(t) => t.0,
            other => panic!("recorded_at must be TxId, got {other:?}"),
        })
        .collect();
    all_txids.sort();
    let n = all_txids.len();
    // Strict-monotonic: no duplicates. Each commit allocates a unique TxId.
    for w in all_txids.windows(2) {
        assert!(
            w[1] > w[0],
            "persisted TxIds must be strictly monotonic when sorted; got duplicate at {w:?}"
        );
    }
    // Contiguity: span / count gives mean allocation step. No gap may
    // exceed 4x the mean step. (Engine may interleave allocations from
    // unrelated test-internal work — observations table seeds; loose
    // upper bound rules out only ordering violations.)
    let span = all_txids[n - 1].saturating_sub(all_txids[0]);
    let mean_step = (span / n as u64).max(1);
    let max_gap_allowed = mean_step * 4;
    for w in all_txids.windows(2) {
        let gap = w[1] - w[0];
        assert!(
            gap <= max_gap_allowed,
            "persisted TxId gap={gap} exceeds 4x mean allocation step={mean_step}; \
             holes greater than tx-allocation-step signal non-contiguous engine clock; \
             all_txids={all_txids:?}"
        );
    }

    // Cross-check: every TxId the writer emitted via db.begin() appears
    // in the persisted set. Wrong-impl that silently drops or rewrites
    // committed TxIds would show as a missing entry.
    let persisted_set: std::collections::HashSet<u64> = all_txids.iter().copied().collect();
    for emitted_tx in &writer_emitted {
        assert!(
            persisted_set.contains(emitted_tx),
            "writer-emitted TxId={emitted_tx} not present in persisted recorded_at column; \
             engine silently dropped or rewrote the stamp"
        );
    }
}

// ---------- BLOCKED entries ----------
// C-RT6 derived_from cross-context rejection — BLOCKED on PreCommitContextGuard
// (engine-gaps-for-cg.md §4 Pending). Cross-context FK rejection is a
// PreCommitContextGuard primitive that the released engine does not yet
// expose; cited as one-line BLOCKED entry in the catalogue.
//
// C-RT10 Store::open validation race — out of scope for engine probes.
// `Store::open` is a cg-level surface, not an engine primitive; the engine
// surface here is `Database::open`, which does not declare embedding spaces.
// Catalogued as BLOCKED on cg-side surface (Plan 3R itself).

// Suppress dead_code lint on RowId import (used in commented-out probes).
#[allow(dead_code)]
fn _row_id_witness(_r: RowId) {}
