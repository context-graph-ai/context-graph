//! Plan 22b cg-notify — engine-probe sweep.
//!
//! See `.claude/plans/context-graph/r0/track1-enumeration/22b-cg-notify-primitives.md`
//! for the per-primitive enumeration this file targets. Probes target the
//! released contextdb v1.0.0 surface.
//!
//! BLOCKED scope: §5 EventBus (Pending). The engine-event → notification
//! creation path is gated; the notification CRUD + retry-queue surface is
//! NOT gated and IS probed here. Catalogued one-line BLOCKED:
//!   - `notify_invalidation_routes_to_configured_sink`
//!   - `notify_signature_match_routes_to_configured_sink`
//!   - `notify_retry_on_sink_recovery`
//!   - `notify_sink_failure_does_not_block_engine`
//!
//! UNBLOCKED scope (probed here): notification_log table CRUD, STATE MACHINE
//! on notification status, retry-worker conditional UPDATE under contention,
//! severity filtering at SELECT, atomic notification + log creation in one tx,
//! at-least-once delivery semantics under concurrent retry workers.
//!
//! Engine source citations point at `contextdb/crates/contextdb-engine/` and
//! `contextdb/crates/contextdb-core/` paths inside the umbrella checkout.

use contextdb_core::{Error, TxId, Value};
use contextdb_engine::Database;
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;
use uuid::Uuid;

/// Build a parameter map from `(name, value)` pairs.
fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Construct a minimal cg-notify schema. Notifications carry severity;
/// notification_log captures every delivery attempt with STATE MACHINE on
/// status (pending → delivered | failed). Sinks declare a min_severity for
/// routing-side filtering.
fn open_notify_schema() -> Database {
    let db = Database::open_memory();
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE notification_sinks (
            id UUID PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            min_severity TEXT NOT NULL,
            created_at TXID NOT NULL
        )",
        &p,
    )
    .unwrap();
    db.execute(
        "CREATE TABLE notifications (
            id UUID PRIMARY KEY,
            severity TEXT NOT NULL,
            payload JSON NOT NULL,
            created_at TXID NOT NULL
        )",
        &p,
    )
    .unwrap();
    // notification_log: STATE MACHINE pending -> delivered | failed.
    // Using IMMUTABLE: false (mutable). Append on creation, UPDATE on status.
    db.execute(
        "CREATE TABLE notification_log (
            id UUID PRIMARY KEY,
            notification_id UUID NOT NULL REFERENCES notifications(id),
            sink_id UUID NOT NULL REFERENCES notification_sinks(id),
            attempt_n INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            last_error TEXT,
            created_at TXID NOT NULL,
            UNIQUE (notification_id, sink_id, attempt_n)
        ) STATE MACHINE (status: pending -> [delivered, failed])",
        &p,
    )
    .unwrap();
    db
}

fn seed_sink(db: &Database, name: &str, min_severity: &str) -> Uuid {
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO notification_sinks (id, name, min_severity, created_at) \
         VALUES ($id, $name, $sev, $ts)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("name", Value::Text(name.into())),
            ("sev", Value::Text(min_severity.into())),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    id
}

fn seed_notification(db: &Database, severity: &str) -> Uuid {
    let id = Uuid::new_v4();
    db.execute(
        "INSERT INTO notifications (id, severity, payload, created_at) \
         VALUES ($id, $sev, $pl, $ts)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("sev", Value::Text(severity.into())),
            ("pl", Value::Json(serde_json::json!({"msg": "test"}))),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    id
}

// ---------- Single-primitive probes ----------

/// 22b single-primitive: notification_log INSERT + SELECT of pending rows.
/// FK validated at commit; row visible via WHERE status='pending'.
#[test]
fn notification_log_insert_pending_row_visible() {
    let db = open_notify_schema();
    let sink = seed_sink(&db, "slack", "warning");
    let notif = seed_notification(&db, "critical");
    db.execute(
        "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
         VALUES ($id, $nid, $sid, 1, $ts)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("nid", Value::Uuid(notif)),
            ("sid", Value::Uuid(sink)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    let result = db
        .execute(
            "SELECT id, status FROM notification_log WHERE status = 'pending'",
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][1], Value::Text("pending".into()));
}

/// 22b single-primitive: notification_log STATE MACHINE pending → delivered
/// is allowed; pending → failed is allowed; delivered → pending is rejected
/// (terminal state).
/// Cites: contextdb/crates/contextdb-engine/src/schema_enforcer.rs::validate_dml.
#[test]
fn notification_log_state_machine_transitions() {
    let db = open_notify_schema();
    let sink = seed_sink(&db, "email", "info");
    let notif = seed_notification(&db, "info");
    let log_id = Uuid::new_v4();
    db.execute(
        "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
         VALUES ($id, $nid, $sid, 1, $ts)",
        &params(vec![
            ("id", Value::Uuid(log_id)),
            ("nid", Value::Uuid(notif)),
            ("sid", Value::Uuid(sink)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    // pending → delivered: allowed.
    db.execute(
        "UPDATE notification_log SET status = 'delivered' WHERE id = $id",
        &params(vec![("id", Value::Uuid(log_id))]),
    )
    .unwrap();
    // delivered → pending: terminal, must reject.
    let bad_back = db.execute(
        "UPDATE notification_log SET status = 'pending' WHERE id = $id",
        &params(vec![("id", Value::Uuid(log_id))]),
    );
    assert!(
        bad_back.is_err(),
        "delivered -> pending must be rejected by STATE MACHINE; got {bad_back:?}"
    );
    // delivered → failed: also undeclared transition, must reject.
    let bad_fwd = db.execute(
        "UPDATE notification_log SET status = 'failed' WHERE id = $id",
        &params(vec![("id", Value::Uuid(log_id))]),
    );
    assert!(
        bad_fwd.is_err(),
        "delivered -> failed must be rejected by STATE MACHINE; got {bad_fwd:?}"
    );

    // Fresh row, pending → failed: allowed.
    let log_id2 = Uuid::new_v4();
    db.execute(
        "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
         VALUES ($id, $nid, $sid, 2, $ts)",
        &params(vec![
            ("id", Value::Uuid(log_id2)),
            ("nid", Value::Uuid(notif)),
            ("sid", Value::Uuid(sink)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    db.execute(
        "UPDATE notification_log SET status = 'failed', last_error = 'sink down' WHERE id = $id",
        &params(vec![("id", Value::Uuid(log_id2))]),
    )
    .unwrap();
    let r = db
        .execute(
            "SELECT status FROM notification_log WHERE id = $id",
            &params(vec![("id", Value::Uuid(log_id2))]),
        )
        .unwrap();
    assert_eq!(r.rows[0][0], Value::Text("failed".into()));
}

/// 22b single-primitive: notification_log STATE MACHINE bypass via
/// `execute_in_tx`. Wrong-impl per STATE MACHINE lens: engine rejects via
/// `execute` but accepts via `execute_in_tx`.
#[test]
fn notification_log_state_machine_bypass_via_execute_in_tx_refused() {
    let db = open_notify_schema();
    let sink = seed_sink(&db, "pagerduty", "critical");
    let notif = seed_notification(&db, "critical");
    let log_id = Uuid::new_v4();
    db.execute(
        "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
         VALUES ($id, $nid, $sid, 1, $ts)",
        &params(vec![
            ("id", Value::Uuid(log_id)),
            ("nid", Value::Uuid(notif)),
            ("sid", Value::Uuid(sink)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    // pending → delivered through autocommit.
    db.execute(
        "UPDATE notification_log SET status = 'delivered' WHERE id = $id",
        &params(vec![("id", Value::Uuid(log_id))]),
    )
    .unwrap();

    // Now attempt delivered → failed via execute_in_tx (terminal state bypass).
    let tx = db.begin();
    let result = db.execute_in_tx(
        tx,
        "UPDATE notification_log SET status = 'failed' WHERE id = $id",
        &params(vec![("id", Value::Uuid(log_id))]),
    );
    let _ = db.rollback(tx);
    assert!(
        result.is_err(),
        "STATE MACHINE delivered -> failed must be refused via execute_in_tx; got {result:?}"
    );
}

/// 22b single-primitive: severity filter at SELECT — `WHERE severity = 'critical'`
/// must return only critical-severity notifications; engine query plan must
/// honor the predicate.
#[test]
fn notifications_severity_filter_returns_only_matching_rows() {
    let db = open_notify_schema();
    for sev in ["info", "warning", "warning", "critical", "critical", "critical"] {
        seed_notification(&db, sev);
    }
    let critical = db
        .execute(
            "SELECT id FROM notifications WHERE severity = 'critical'",
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(
        critical.rows.len(),
        3,
        "severity filter must return exactly 3 critical rows; got {}",
        critical.rows.len()
    );
    let warning = db
        .execute(
            "SELECT id FROM notifications WHERE severity = 'warning'",
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(warning.rows.len(), 2);
}

/// 22b single-primitive: composite UNIQUE on `(notification_id, sink_id,
/// attempt_n)`. Two log rows with same triple dedup.
#[test]
fn notification_log_composite_unique_attempt_dedups() {
    let db = open_notify_schema();
    let sink = seed_sink(&db, "webhook", "info");
    let notif = seed_notification(&db, "warning");
    let r1 = db
        .execute(
            "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
             VALUES ($id, $nid, $sid, 1, $ts)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("nid", Value::Uuid(notif)),
                ("sid", Value::Uuid(sink)),
                ("ts", Value::TxId(TxId(db.committed_watermark().0))),
            ]),
        )
        .unwrap();
    assert_eq!(r1.rows_affected, 1);
    // Same (notification_id, sink_id, attempt_n) — duplicate-insert no-op.
    let r2 = db
        .execute(
            "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
             VALUES ($id, $nid, $sid, 1, $ts)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("nid", Value::Uuid(notif)),
                ("sid", Value::Uuid(sink)),
                ("ts", Value::TxId(TxId(db.committed_watermark().0))),
            ]),
        )
        .unwrap();
    assert_eq!(
        r2.rows_affected, 0,
        "duplicate attempt-triple must dedup as no-op"
    );

    // attempt_n=2 → distinct triple → second row.
    let r3 = db
        .execute(
            "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
             VALUES ($id, $nid, $sid, 2, $ts)",
            &params(vec![
                ("id", Value::Uuid(Uuid::new_v4())),
                ("nid", Value::Uuid(notif)),
                ("sid", Value::Uuid(sink)),
                ("ts", Value::TxId(TxId(db.committed_watermark().0))),
            ]),
        )
        .unwrap();
    assert_eq!(r3.rows_affected, 1);
}

// ---------- Multi-primitive combination probes ----------

/// 22b multi-primitive: atomic notification + initial pending log row in one
/// tx. Either both commit or both roll back. Probe forces FK fail on
/// non-existent sink and asserts neither row persists.
#[test]
fn notification_plus_log_atomic_rollback_on_fk_failure() {
    let db = open_notify_schema();
    let bogus_sink = Uuid::new_v4();

    let notif_id = Uuid::new_v4();
    let tx = db.begin();
    db.execute_in_tx(
        tx,
        "INSERT INTO notifications (id, severity, payload, created_at) \
         VALUES ($id, 'critical', $pl, $ts)",
        &params(vec![
            ("id", Value::Uuid(notif_id)),
            ("pl", Value::Json(serde_json::json!({"msg": "alert"}))),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    db.execute_in_tx(
        tx,
        "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
         VALUES ($id, $nid, $sid, 1, $ts)",
        &params(vec![
            ("id", Value::Uuid(Uuid::new_v4())),
            ("nid", Value::Uuid(notif_id)),
            ("sid", Value::Uuid(bogus_sink)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();

    let commit_result = db.commit(tx);
    assert!(
        matches!(commit_result, Err(Error::ForeignKeyViolation { .. })),
        "commit must fail on bogus sink_id FK; got {commit_result:?}"
    );
    assert_eq!(
        db.scan("notifications", db.snapshot()).unwrap().len(),
        0,
        "notification must roll back when log row FK fails"
    );
    assert_eq!(
        db.scan("notification_log", db.snapshot()).unwrap().len(),
        0
    );
}

/// 22b multi-primitive: retry worker tx — UPDATE log row attempt_n + status
/// in one tx. Probe verifies STATE MACHINE pending → delivered transition
/// proceeds atomically with no torn rows.
#[test]
fn retry_worker_tx_state_transition_atomic() {
    let db = open_notify_schema();
    let sink = seed_sink(&db, "discord", "warning");
    let notif = seed_notification(&db, "critical");
    let log_id = Uuid::new_v4();
    db.execute(
        "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
         VALUES ($id, $nid, $sid, 1, $ts)",
        &params(vec![
            ("id", Value::Uuid(log_id)),
            ("nid", Value::Uuid(notif)),
            ("sid", Value::Uuid(sink)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();

    let tx = db.begin();
    db.execute_in_tx(
        tx,
        "UPDATE notification_log SET status = 'delivered' WHERE id = $id",
        &params(vec![("id", Value::Uuid(log_id))]),
    )
    .unwrap();
    db.commit(tx).unwrap();

    let r = db
        .execute(
            "SELECT status FROM notification_log WHERE id = $id",
            &params(vec![("id", Value::Uuid(log_id))]),
        )
        .unwrap();
    assert_eq!(r.rows[0][0], Value::Text("delivered".into()));
}

// ---------- Concurrent-semantics probes ----------

/// C-NT1 concurrent retry workers picking the same pending row: N workers
/// each issue `UPDATE notification_log SET status='delivered' WHERE id = $id
/// AND status='pending'`. Engine MUST conflict-serialize — exactly one
/// reports rows_affected=1; others must report rows_affected=0. Wrong-impl
/// per UPDATE lens: (b) engine returns Ok(rows_affected=1) for both writers
/// (double-delivery).
#[test]
fn retry_workers_conditional_update_serializes() {
    let db = Arc::new(open_notify_schema());
    let sink = seed_sink(&db, "telegram", "info");
    let notif = seed_notification(&db, "warning");

    const ROUNDS: usize = 30;
    const N: usize = 5;
    let mut max_total = 0u64;
    let mut all_rounds = vec![];

    for r in 0..ROUNDS {
        let log_id = Uuid::new_v4();
        db.execute(
            "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
             VALUES ($id, $nid, $sid, $a, $ts)",
            &params(vec![
                ("id", Value::Uuid(log_id)),
                ("nid", Value::Uuid(notif)),
                ("sid", Value::Uuid(sink)),
                ("a", Value::Int64((r as i64) + 1)),
                ("ts", Value::TxId(TxId(db.committed_watermark().0))),
            ]),
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
                    "UPDATE notification_log SET status = 'delivered' WHERE id = $id AND status = 'pending'",
                    &params(vec![("id", Value::Uuid(log_id))]),
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
        all_rounds.push((r, total, updates));
        // Final state: status must be 'delivered' (not torn).
        let row = db
            .execute(
                "SELECT status FROM notification_log WHERE id = $id",
                &params(vec![("id", Value::Uuid(log_id))]),
            )
            .unwrap();
        assert_eq!(
            row.rows[0][0],
            Value::Text("delivered".into()),
            "round {r}: conditional UPDATE must produce a single transition; final status \
             must be 'delivered'"
        );
        if max_total > 1 {
            break;
        }
    }
    assert_eq!(
        max_total, 1,
        "concurrent retry workers on conditional UPDATE: exactly one writer must report \
         rows_affected=1; max observed total across {ROUNDS} rounds = {max_total}; \
         round results = {all_rounds:?}"
    );
}

/// C-NT3 severity filter respected at routing under concurrent inserts: a
/// reader runs `SELECT WHERE severity = 'critical'` while a writer is
/// inserting a mix of severities. Reader at a pinned snapshot returns only
/// the rows visible at the snapshot.
#[test]
fn severity_filter_snapshot_isolation_under_concurrent_inserts() {
    let db = Arc::new(open_notify_schema());
    // Seed 2 critical, 2 warning.
    for _ in 0..2 {
        seed_notification(&db, "critical");
    }
    for _ in 0..2 {
        seed_notification(&db, "warning");
    }
    let pinned = db.snapshot();
    let pre = db
        .execute_at_snapshot(
            "SELECT id FROM notifications WHERE severity = 'critical'",
            &HashMap::new(),
            pinned,
        )
        .unwrap();
    assert_eq!(pre.rows.len(), 2);

    // Concurrent writer adds 5 more critical rows.
    let writer_db = db.clone();
    let writer = thread::spawn(move || {
        for _ in 0..5 {
            seed_notification(&writer_db, "critical");
        }
    });
    writer.join().unwrap();

    // Pinned read STILL sees 2 critical.
    let post_pinned = db
        .execute_at_snapshot(
            "SELECT id FROM notifications WHERE severity = 'critical'",
            &HashMap::new(),
            pinned,
        )
        .unwrap();
    assert_eq!(
        post_pinned.rows.len(),
        2,
        "pinned-snapshot severity filter must NOT see post-snapshot inserts; got {}",
        post_pinned.rows.len()
    );
    // Latest snapshot sees 2 + 5 = 7.
    let latest = db
        .execute(
            "SELECT id FROM notifications WHERE severity = 'critical'",
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(latest.rows.len(), 7);
}

/// C-NT4 at-least-once delivery semantics: simulates a crash mid-attempt by
/// rolling back a tx after staging an attempt_n+1 row, then a fresh worker
/// picks the pending row again and delivers. Probe verifies the engine
/// surface that supports this: a tx rollback drops the in-flight log row,
/// and the original pending row remains visible.
#[test]
fn retry_worker_crash_mid_attempt_rolls_back_in_flight_log_row() {
    let db = open_notify_schema();
    let sink = seed_sink(&db, "sms", "warning");
    let notif = seed_notification(&db, "critical");
    let log_id = Uuid::new_v4();
    db.execute(
        "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
         VALUES ($id, $nid, $sid, 1, $ts)",
        &params(vec![
            ("id", Value::Uuid(log_id)),
            ("nid", Value::Uuid(notif)),
            ("sid", Value::Uuid(sink)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();

    // Worker A starts a tx, stages attempt_n=2 and a status update, then
    // CRASHES (we model as rollback).
    let tx = db.begin();
    let attempt2_id = Uuid::new_v4();
    db.execute_in_tx(
        tx,
        "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
         VALUES ($id, $nid, $sid, 2, $ts)",
        &params(vec![
            ("id", Value::Uuid(attempt2_id)),
            ("nid", Value::Uuid(notif)),
            ("sid", Value::Uuid(sink)),
            ("ts", Value::TxId(TxId(db.committed_watermark().0))),
        ]),
    )
    .unwrap();
    // Worker A "crashes".
    db.rollback(tx).unwrap();

    // Original pending row is still visible.
    let pending = db
        .execute(
            "SELECT id FROM notification_log WHERE status = 'pending'",
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(
        pending.rows.len(),
        1,
        "original pending row must still be visible after worker crash"
    );
    // attempt2 row must be gone.
    let attempt2 = db
        .execute(
            "SELECT id FROM notification_log WHERE id = $id",
            &params(vec![("id", Value::Uuid(attempt2_id))]),
        )
        .unwrap();
    assert_eq!(
        attempt2.rows.len(),
        0,
        "rolled-back attempt row must NOT persist"
    );

    // Worker B resumes, delivers successfully.
    db.execute(
        "UPDATE notification_log SET status = 'delivered' WHERE id = $id AND status = 'pending'",
        &params(vec![("id", Value::Uuid(log_id))]),
    )
    .unwrap();
    let final_state = db
        .execute(
            "SELECT status FROM notification_log WHERE id = $id",
            &params(vec![("id", Value::Uuid(log_id))]),
        )
        .unwrap();
    assert_eq!(final_state.rows[0][0], Value::Text("delivered".into()));
}

/// 22b concurrent independent notifications: N concurrent writers each
/// create their own (notification + initial log row) atomic tx. All N
/// must commit; no row loss; no torn rows. Wrong-impl: contention on the
/// notifications table loses rows.
#[test]
fn concurrent_independent_notification_creation_all_persist() {
    let db = Arc::new(open_notify_schema());
    let sink = seed_sink(&db, "ops", "info");

    const N: usize = 8;
    let barrier = Arc::new(Barrier::new(N));
    let mut handles = vec![];
    for i in 0..N {
        let db = db.clone();
        let barrier = barrier.clone();
        let sink = sink;
        handles.push(thread::spawn(move || {
            let notif_id = Uuid::new_v4();
            let log_id = Uuid::new_v4();
            barrier.wait();
            let tx = db.begin();
            db.execute_in_tx(
                tx,
                "INSERT INTO notifications (id, severity, payload, created_at) \
                 VALUES ($id, $sev, $pl, $ts)",
                &params(vec![
                    ("id", Value::Uuid(notif_id)),
                    ("sev", Value::Text("warning".into())),
                    ("pl", Value::Json(serde_json::json!({"i": i}))),
                    ("ts", Value::TxId(TxId(db.committed_watermark().0))),
                ]),
            )
            .unwrap();
            db.execute_in_tx(
                tx,
                "INSERT INTO notification_log (id, notification_id, sink_id, attempt_n, created_at) \
                 VALUES ($id, $nid, $sid, 1, $ts)",
                &params(vec![
                    ("id", Value::Uuid(log_id)),
                    ("nid", Value::Uuid(notif_id)),
                    ("sid", Value::Uuid(sink)),
                    ("ts", Value::TxId(TxId(db.committed_watermark().0))),
                ]),
            )
            .unwrap();
            db.commit(tx).unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let notifications = db.scan("notifications", db.snapshot()).unwrap();
    let logs = db.scan("notification_log", db.snapshot()).unwrap();
    assert_eq!(
        notifications.len(),
        N,
        "all {N} concurrent notification creations must persist"
    );
    assert_eq!(logs.len(), N);
    // No log row is delivered yet.
    let pending = db
        .execute(
            "SELECT id FROM notification_log WHERE status = 'pending'",
            &HashMap::new(),
        )
        .unwrap();
    assert_eq!(pending.rows.len(), N);
}

/// 22b engine-blocked entries (catalogue line, not probed):
///   - `notify_invalidation_routes_to_configured_sink` — needs EventBus
///     (engine commit emits InvalidationDetected event → routing creates
///     notification). No engine surface today.
///   - `notify_signature_match_routes_to_configured_sink` — needs EventBus.
///   - `notify_retry_on_sink_recovery` — durable retry queue is probed at
///     the schema level above; the EventBus-driven re-trigger is blocked.
///   - `notify_sink_failure_does_not_block_engine` — async delivery design
///     is engine-event-bus-driven; without EventBus, the engine does not
///     even see the notification request to begin with.
/// All four catalogued one-line in
/// `.claude/plans/context-graph/r0/track1-results/22b-cg-notify-probe-results.md`.
#[allow(dead_code)]
fn _blocked_entries_marker() {}
