//! Plan 25 cg-server-observe — engine-probe sweep.
//!
//! See `.claude/plans/context-graph/r0/track1-enumeration/25-cg-server-observe-primitives.md`.
//!
//! Plan 25's primary surface is HTTP webhook ingress + HMAC + LLM fallback.
//! Most of its primitives are server-layer concerns (HMAC verification,
//! HTTP routing, scheduled cron) that the engine does not expose. The
//! engine-probe coverage here targets the engine-side tail that 25
//! depends on: record_observation atomicity (3R), entity-resolution
//! UNIQUE behaviour, dimension-mismatch rejection at INSERT.

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

fn open_observe_schema() -> Database {
    let db = Database::open_memory();
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE entities (
            id UUID PRIMARY KEY,
            name TEXT,
            context_id TEXT,
            UNIQUE (name, context_id)
        )",
        &p,
    )
    .unwrap();
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
    db.execute(
        "CREATE TABLE evidence (
            id UUID PRIMARY KEY,
            observation_id UUID,
            vector_text VECTOR(8)
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE audit_log (
            id UUID PRIMARY KEY,
            event_kind TEXT,
            recorded_at INTEGER
        )",
        &p,
    )
    .unwrap();
    db
}

// ---------- Single-primitive ----------

/// 25 single-primitive (C-OB4): dimension-mismatch rejection at write
/// time — webhook with embedding of wrong dimension for declared space
/// rejected; no partial INSERT.
#[test]
fn webhook_wrong_dim_embedding_rejected_no_partial_insert() {
    let db = open_observe_schema();
    let entity_id = Uuid::new_v4();
    let observation_id = Uuid::new_v4();

    let tx = db.begin();
    db.insert_row(
        tx,
        "observations",
        vals(vec![
            ("id", Value::Uuid(observation_id)),
            ("entity_id", Value::Uuid(entity_id)),
            ("context_id", Value::Text("ctxA".into())),
        ]),
    )
    .unwrap();
    let evi_row = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("observation_id", Value::Uuid(observation_id)),
            ]),
        )
        .unwrap();
    let bad = db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"),
        evi_row,
        vec![1.0, 2.0],
    );
    assert!(matches!(bad, Err(Error::VectorIndexDimensionMismatch { .. })));
    let _ = db.rollback(tx);

    let snap = db.snapshot();
    assert_eq!(
        db.scan("observations", snap).unwrap().len(),
        0,
        "rejected webhook leaves no partial trace in observations"
    );
    assert_eq!(
        db.scan("evidence", snap).unwrap().len(),
        0,
        "rejected webhook leaves no partial trace in evidence"
    );
}

// ---------- Multi-primitive ----------

/// 25 multi-primitive: webhook end-to-end tx — observation + evidence
/// insertion atomic. Failure mid-tx rolls everything back.
#[test]
fn webhook_observation_evidence_atomic_rollback() {
    let db = open_observe_schema();
    let entity_id = Uuid::new_v4();
    let observation_id = Uuid::new_v4();

    let tx = db.begin();
    db.insert_row(
        tx,
        "entities",
        vals(vec![
            ("id", Value::Uuid(entity_id)),
            ("name", Value::Text("acme/repo".into())),
            ("context_id", Value::Text("ctxA".into())),
        ]),
    )
    .unwrap();
    db.insert_row(
        tx,
        "observations",
        vals(vec![
            ("id", Value::Uuid(observation_id)),
            ("entity_id", Value::Uuid(entity_id)),
            ("context_id", Value::Text("ctxA".into())),
        ]),
    )
    .unwrap();
    let evi_row = db
        .insert_row(
            tx,
            "evidence",
            vals(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("observation_id", Value::Uuid(observation_id)),
            ]),
        )
        .unwrap();
    db.insert_vector(
        tx,
        VectorIndexRef::new("evidence", "vector_text"),
        evi_row,
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    db.rollback(tx).unwrap();

    let snap = db.snapshot();
    assert_eq!(db.scan("entities", snap).unwrap().len(), 0);
    assert_eq!(db.scan("observations", snap).unwrap().len(), 0);
    assert_eq!(db.scan("evidence", snap).unwrap().len(), 0);
}

// ---------- Concurrent-semantics probes ----------

/// C-OB1 concurrent webhook ingress for the same entity: N threads each
/// insert (entity, observation). Engine UNIQUE on (name, context_id)
/// must dedup entity creation. Multiple rounds to surface contention.
#[test]
fn concurrent_webhook_same_entity_dedups_entity_row() {
    const ROUNDS: usize = 20;
    const N: usize = 6;
    let mut max_entity_rows = 0usize;
    for r in 0..ROUNDS {
        let db = Arc::new(open_observe_schema());
        let barrier = Arc::new(Barrier::new(N));
        let mut handles = vec![];
        let name = format!("acme/repo_{r}");
        for i in 0..N {
            let db = db.clone();
            let barrier = barrier.clone();
            let name = name.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                let _ = db.execute(
                    "INSERT INTO entities (id, name, context_id) VALUES ($id, $name, $ctx)",
                    &params(vec![
                        ("id", Value::Uuid(Uuid::new_v4())),
                        ("name", Value::Text(name)),
                        ("ctx", Value::Text("ctxA".into())),
                    ]),
                );
                let _ = db.execute(
                    "INSERT INTO observations (id, entity_id, context_id, recorded_at) VALUES ($id, $eid, $ctx, $t)",
                    &params(vec![
                        ("id", Value::Uuid(Uuid::new_v4())),
                        ("eid", Value::Uuid(Uuid::new_v4())),
                        ("ctx", Value::Text("ctxA".into())),
                        ("t", Value::Int64(i as i64)),
                    ]),
                );
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let entities = db
            .execute(
                "SELECT id FROM entities WHERE name = $n",
                &params(vec![("n", Value::Text(name))]),
            )
            .unwrap();
        if entities.rows.len() > max_entity_rows {
            max_entity_rows = entities.rows.len();
        }
        if max_entity_rows > 1 {
            break;
        }
    }
    assert_eq!(
        max_entity_rows, 1,
        "concurrent webhooks for same entity must dedup; observed {max_entity_rows} entity rows in best round"
    );
}

/// C-OB5 entity_resolution race: N threads concurrently insert the same
/// entity name. UNIQUE on (name, context_id) must collapse to ONE row.
/// Multiple rounds to surface contention reliably.
#[test]
fn entity_resolution_race_unique_collapse() {
    const ROUNDS: usize = 20;
    const N: usize = 8;
    let mut max_rows_seen = 0usize;
    for r in 0..ROUNDS {
        let db = Arc::new(open_observe_schema());
        let barrier = Arc::new(Barrier::new(N));
        let mut handles = vec![];
        let name = format!("entity_round_{r}");
        for _ in 0..N {
            let db = db.clone();
            let barrier = barrier.clone();
            let name = name.clone();
            handles.push(thread::spawn(move || -> u64 {
                barrier.wait();
                db.execute(
                    "INSERT INTO entities (id, name, context_id) VALUES ($id, $name, $ctx)",
                    &params(vec![
                        ("id", Value::Uuid(Uuid::new_v4())),
                        ("name", Value::Text(name)),
                        ("ctx", Value::Text("ctxA".into())),
                    ]),
                )
                .map(|qr| qr.rows_affected)
                .unwrap_or(0)
            }));
        }
        let _total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let entities = db
            .execute(
                "SELECT id FROM entities WHERE name = $n",
                &params(vec![("n", Value::Text(name))]),
            )
            .unwrap();
        if entities.rows.len() > max_rows_seen {
            max_rows_seen = entities.rows.len();
        }
        if max_rows_seen > 1 {
            break;
        }
    }
    assert_eq!(
        max_rows_seen, 1,
        "concurrent UNIQUE inserts must collapse to one row; observed {max_rows_seen} rows in best round"
    );
}

/// C-OB3 HMAC failure does not commit: simulated by an explicit rollback
/// of a tx that wrote partial state. Engine must drop everything; no
/// audit_log trace.
#[test]
fn rejected_webhook_no_partial_audit_trace() {
    let db = open_observe_schema();
    let tx = db.begin();
    db.insert_row(
        tx,
        "audit_log",
        vals(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("event_kind", Value::Text("observe.received".into())),
            ("recorded_at", Value::Int64(1)),
        ]),
    )
    .unwrap();
    db.insert_row(
        tx,
        "observations",
        vals(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("entity_id", Value::Uuid(Uuid::new_v4())),
            ("context_id", Value::Text("ctxA".into())),
        ]),
    )
    .unwrap();
    db.rollback(tx).unwrap();

    assert_eq!(db.scan("audit_log", db.snapshot()).unwrap().len(), 0);
    assert_eq!(db.scan("observations", db.snapshot()).unwrap().len(), 0);
}

// Round-2 reviewer fix: the former probe `engine_never_sees_inserts_when_caller_skips_them`
// was a tautology (open empty schema, scan, assert empty — no engine
// behaviour under test). Deleted. C-OB6 large-payload refusal is covered
// by the BLOCKED entry below.

// ---------- BLOCKED entries ----------
// C-OB2 scheduled cron + webhook concurrency — `observe_scheduled_via_contextdb_cron`
// requires ObservationTrigger (engine-gaps-for-cg.md §3 Pending). Cron
// scheduling is a server-layer plugin, not an engine primitive.
// BLOCKED on ObservationTrigger.
//
// C-OB6 large-payload / HTTP-layer refusal — refusal happens before any
// engine call. The engine has no payload-size primitive. Out of engine
// probe scope.
//
// HMAC verification and HTTP routing — server-layer; engine has no
// HMAC primitive. Out of engine probe scope.
//
// LLM fallback — cg-llm-client trait dispatch; not an engine primitive.
// Out of engine probe scope.
