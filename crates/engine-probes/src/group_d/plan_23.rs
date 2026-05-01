//! Probes for plan 23 — cg-server-core.
//!
//! Plan class: primitive-owner (fully unblocked).
//!
//! Per the dispatch brief: server lifecycle (HTTP bind/listen/SIGTERM/drain),
//! NATS sync wire protocol, hosted-mode connection-pool fairness, and other
//! probes that genuinely require a `cg_server` binary plus a NATS server
//! cannot run inside the engine-only test harness against pure contextdb. Those
//! are catalogued in `r0/track1-results/23-cg-server-core-probe-results.md`
//! as `BLOCKED: requires server-binary harness; engine surface alone insufficient`.
//!
//! The engine-level primitives that surface in cg-server's sync-apply path —
//! TXID monotonicity across deliveries, last-writer-wins per TXID at engine
//! commit, concurrent multi-tx delivery into the engine, idempotent re-apply,
//! cross-context isolation under filter — DO run here against pure contextdb.
//!
//! Engine source cited:
//! - `contextdb/crates/contextdb-tx/src/lib.rs` — `TxManager::begin`,
//!   `TxManager::commit_with_lock`, `committed_watermark` AtomicTxId.
//! - `contextdb/crates/contextdb-engine/src/database.rs::begin` (line ~486),
//!   `commit_with_source` (line ~490) — atomic commit ordering, watermark
//!   advance.
//! - `contextdb/crates/contextdb-engine/src/database.rs::upsert_row`
//!   (line ~1162) — last-writer-wins via natural-key upsert.
//! - `contextdb/crates/contextdb-engine/src/database.rs::insert_row_for_sync`
//!   (line ~1082) — internal sync-apply path; we mirror its ordering by
//!   driving `insert_row` and `upsert_row` from concurrent threads.

use contextdb_core::error::Error;
use contextdb_core::types::{TxId, Value};
use contextdb_engine::Database;
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;

fn make_db() -> Arc<Database> {
    let db = Arc::new(Database::open_memory());
    db.execute(
        "CREATE TABLE replicated_rows (\n\
            id UUID PRIMARY KEY,\n\
            natural_key TEXT NOT NULL UNIQUE,\n\
            value TEXT NOT NULL,\n\
            committed_at TXID NOT NULL\n\
        )",
        &HashMap::new(),
    )
    .expect("create replicated_rows");
    db.execute(
        "CREATE TABLE ctx_rows (\n\
            id UUID PRIMARY KEY,\n\
            context_id TEXT NOT NULL,\n\
            payload TEXT NOT NULL,\n\
            committed_at TXID NOT NULL\n\
        )",
        &HashMap::new(),
    )
    .expect("create ctx_rows");
    db
}

/// C-SC1 / TXID monotonicity: under heavy concurrent commits from many
/// "delivery" threads (each a thread plays the role of one inbound NATS
/// message), every committed row carries a strictly increasing TXID and
/// no two commits share a TXID.
///
/// Lens: lens-question row "TXID column type" — TXID accepting arbitrary u64
/// instead of advancing the manager's monotonic clock is wrong-impl; TXID
/// comparisons returning wrong ordering under concurrent inserts is wrong-impl.
#[test]
fn concurrent_sync_deliveries_txid_strictly_monotonic() {
    let db = make_db();
    let n_threads = 8;
    let per_thread = 32;

    let barrier = Arc::new(Barrier::new(n_threads));
    let mut handles = vec![];
    for tid in 0..n_threads {
        let db_t = db.clone();
        let bar = barrier.clone();
        handles.push(thread::spawn(move || {
            bar.wait();
            let mut my_txids = vec![];
            for i in 0..per_thread {
                let id = uuid::Uuid::new_v4();
                let nk = format!("nk-{}-{}", tid, i);
                let tx = db_t.begin();
                let mut row = HashMap::new();
                row.insert("id".into(), Value::Uuid(id));
                row.insert("natural_key".into(), Value::Text(nk));
                row.insert("value".into(), Value::Text(format!("v-{}-{}", tid, i)));
                row.insert("committed_at".into(), Value::TxId(tx));
                if db_t.insert_row(tx, "replicated_rows", row).is_ok()
                    && db_t.commit(tx).is_ok()
                {
                    my_txids.push(tx);
                }
            }
            my_txids
        }));
    }

    let mut all_txids: Vec<TxId> = vec![];
    let mut per_thread_txids: Vec<Vec<TxId>> = vec![];
    for h in handles {
        let v = h.join().expect("worker");
        all_txids.extend(v.iter().copied());
        per_thread_txids.push(v);
    }

    // All TXIDs must be unique
    let mut sorted = all_txids.clone();
    sorted.sort();
    let mut s2 = sorted.clone();
    s2.dedup();
    assert_eq!(
        s2.len(),
        sorted.len(),
        "TXID collision across concurrent commits"
    );

    // STRICT per-thread monotonicity: each thread's successive commits
    // must have strictly increasing TxIds. Catches the wrong-impl that
    // issues unique-but-non-monotonic ids.
    for (tid, ts) in per_thread_txids.iter().enumerate() {
        for i in 1..ts.len() {
            assert!(
                ts[i].0 > ts[i - 1].0,
                "thread {} TxIds not strictly monotonic: ts[{}]={:?} <= ts[{}]={:?}",
                tid,
                i,
                ts[i],
                i - 1,
                ts[i - 1]
            );
        }
    }

    // The committed_at column read back from the DB MUST round-trip the
    // exact TXID values used at commit time (TXID column type round-trip).
    let r = db
        .execute("SELECT committed_at FROM replicated_rows", &HashMap::new())
        .expect("scan");
    let mut stored: Vec<TxId> = r
        .rows
        .iter()
        .map(|row| match &row[0] {
            Value::TxId(t) => *t,
            other => panic!("committed_at not TxId: {:?}", other),
        })
        .collect();
    stored.sort();
    assert_eq!(
        sorted, stored,
        "stored TXIDs do not match the TXIDs returned by `db.begin()` — TXID round-trip broken"
    );

    // TXID column read-back type assertion: every read-back row MUST be
    // Value::TxId, not Value::Integer. (Already implicit via the panic in
    // the map() above — make it explicit and named for the lens shape.)
    let r2 = db
        .execute("SELECT committed_at FROM replicated_rows", &HashMap::new())
        .expect("scan2");
    for row in &r2.rows {
        match row[0] {
            Value::TxId(_) => {}
            ref other => panic!(
                "committed_at read back as wrong variant: {:?} (expected Value::TxId)",
                other
            ),
        }
    }
}

/// TXID manager bypass: caller fabricates a `Value::TxId(huge)` not allocated
/// by `db.begin()` and the engine MUST reject the INSERT with `TxIdOutOfRange`.
/// Catches the wrong-impl that accepts arbitrary u64 into a TXID column,
/// allowing a peer or buggy caller to inject TXIDs that break ordering.
///
/// Lens: lens-question row "TXID column type" shape (b) "TXID column accepts
/// arbitrary u64 instead of advancing the TX manager's monotonic clock".
#[test]
fn committed_at_rejects_fabricated_txid_not_issued_by_manager() {
    let db = make_db();
    let bad_tx = TxId(u64::MAX - 1);
    let real = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
    row.insert("natural_key".into(), Value::Text("nk-fab".into()));
    row.insert("value".into(), Value::Text("v".into()));
    row.insert("committed_at".into(), Value::TxId(bad_tx));
    let r = db.insert_row(real, "replicated_rows", row);
    let _ = db.rollback(real);
    match r {
        Err(Error::TxIdOutOfRange { value, .. }) => {
            assert_eq!(value, bad_tx.0);
        }
        Err(other) => panic!("expected TxIdOutOfRange; got {:?}", other),
        Ok(_) => panic!(
            "engine accepted Value::TxId({}) which the manager never allocated — TXID-manager-bypass wrong-impl",
            bad_tx.0
        ),
    }
}

/// C-SC2 partition heal LWW per TXID: when two NATS messages target the
/// same natural key, the row state after the second message is the second
/// message's value — engine-level last-writer-wins via `upsert_row` on the
/// natural-key column.
///
/// Probed semantics: after two upserts with the same natural_key, the
/// observable row state is the second upsert's payload — NOT the first's
/// (which would be wrong-impl: out-of-order merge).
///
/// Lens: lens-question row "Multi-table TX" combined with UPSERT path. Also
/// catches engine commit ordering anomaly where writers race on the same
/// row but the latter LSN doesn't win.
#[test]
fn upsert_last_writer_wins_natural_key() {
    let db = make_db();

    // First upsert: value = "first"
    {
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        row.insert("natural_key".into(), Value::Text("nk-1".into()));
        row.insert("value".into(), Value::Text("first".into()));
        row.insert("committed_at".into(), Value::TxId(tx));
        db.upsert_row(tx, "replicated_rows", "natural_key", row)
            .expect("upsert 1");
        db.commit(tx).expect("commit 1");
    }

    // Second upsert with same natural_key: value = "second"
    {
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        row.insert("natural_key".into(), Value::Text("nk-1".into()));
        row.insert("value".into(), Value::Text("second".into()));
        row.insert("committed_at".into(), Value::TxId(tx));
        db.upsert_row(tx, "replicated_rows", "natural_key", row)
            .expect("upsert 2");
        db.commit(tx).expect("commit 2");
    }

    let r = db
        .execute(
            "SELECT value FROM replicated_rows WHERE natural_key = 'nk-1'",
            &HashMap::new(),
        )
        .expect("scan");
    assert_eq!(r.rows.len(), 1, "duplicate rows after upsert LWW");
    let v = r.rows[0][0].as_text().expect("text");
    assert_eq!(v, "second", "LWW broken: latest commit's value not visible");
}

/// Concurrent upsert race on the same natural key — engine MUST conflict-
/// serialize so the final state corresponds to exactly ONE writer's payload,
/// AND that writer MUST be the one with the highest allocated TxId at the
/// time it committed (last-writer-wins per TxId).
///
/// Each thread captures (its own TxId, its own value) on success. After the
/// race, the surviving row's `value` MUST equal the value of the writer
/// with the maximum success-TxId. A wrong engine that picks an arbitrary
/// committer (e.g. first-committer-wins) would fail this.
///
/// Lens: lens-question row "Concurrent UPDATE / conditional UPDATE" shape (b),
/// plus the LWW-by-TxId contract that "last_writer_wins" claims.
#[test]
fn concurrent_upsert_same_natural_key_collapses_to_one_row() {
    let db = make_db();

    let n = 8;
    let barrier = Arc::new(Barrier::new(n));
    let mut handles = vec![];
    for tid in 0..n {
        let db_t = db.clone();
        let bar = barrier.clone();
        handles.push(thread::spawn(move || {
            bar.wait();
            let id = uuid::Uuid::new_v4();
            let tx = db_t.begin();
            let value = format!("tid-{}", tid);
            let mut row = HashMap::new();
            row.insert("id".into(), Value::Uuid(id));
            row.insert("natural_key".into(), Value::Text("hot-key".into()));
            row.insert("value".into(), Value::Text(value.clone()));
            row.insert("committed_at".into(), Value::TxId(tx));
            let res = db_t.upsert_row(tx, "replicated_rows", "natural_key", row);
            let committed = match res {
                Ok(_) => db_t.commit(tx).is_ok(),
                Err(_) => {
                    let _ = db_t.rollback(tx);
                    false
                }
            };
            if committed {
                Some((tx, value))
            } else {
                None
            }
        }));
    }
    let mut successes: Vec<(TxId, String)> = vec![];
    for h in handles {
        if let Some(s) = h.join().expect("worker") {
            successes.push(s);
        }
    }

    // After all writers race, exactly one row visible under hot-key.
    let r = db
        .execute(
            "SELECT value, committed_at FROM replicated_rows WHERE natural_key = 'hot-key'",
            &HashMap::new(),
        )
        .expect("scan");
    assert_eq!(
        r.rows.len(),
        1,
        "concurrent upsert produced {} rows under one natural_key (UNIQUE broken)",
        r.rows.len()
    );

    let stored_value = r.rows[0][0].as_text().expect("value text").to_string();
    let stored_txid = match r.rows[0][1] {
        Value::TxId(t) => t,
        ref other => panic!("committed_at not TxId: {:?}", other),
    };
    let cw = db.snapshot();
    assert!(
        stored_txid.0 <= cw.0,
        "stored committed_at TXID {} > current snapshot {}",
        stored_txid.0,
        cw.0
    );

    // LWW-by-TxId bind: the surviving writer MUST be the one with the
    // maximum committed TxId. If the engine instead picked an arbitrary
    // committer (first-wins, last-aborts-wins, etc.) this would fail.
    if !successes.is_empty() {
        let winner = successes.iter().max_by_key(|(t, _)| t.0).unwrap();
        assert_eq!(
            stored_value, winner.1,
            "LWW-by-TxId broken: surviving value `{}` not from highest-TxId committer (TxId={:?}, value `{}`); successes were {:?}",
            stored_value, winner.0, winner.1, successes
        );
        assert_eq!(
            stored_txid, winner.0,
            "LWW-by-TxId broken: surviving committed_at TxId {:?} not max success TxId {:?}",
            stored_txid, winner.0
        );
    }
}

/// C-SC4 cross-context isolation under sync: a SELECT filtered by
/// `context_id = $X` MUST exclude every row owned by a different context.
/// Failure mode: predicate ignored or applied post-fetch.
///
/// (The full server-side context-scoped sync filter is the
/// HandleScopeConstraint subset and requires a server harness; here we
/// probe the engine's relational filter, the substrate cg-server relies on.)
#[test]
fn context_filter_excludes_other_contexts() {
    let db = make_db();

    for i in 0..6 {
        let id = uuid::Uuid::new_v4();
        let context = if i < 3 { "ctx-a" } else { "ctx-b" };
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(id));
        row.insert("context_id".into(), Value::Text(context.into()));
        row.insert("payload".into(), Value::Text(format!("p-{}", i)));
        row.insert("committed_at".into(), Value::TxId(tx));
        db.insert_row(tx, "ctx_rows", row).expect("ins");
        db.commit(tx).expect("commit");
    }

    let r = db
        .execute(
            "SELECT context_id FROM ctx_rows WHERE context_id = $c",
            &HashMap::from([("c".to_string(), Value::Text("ctx-a".to_string()))]),
        )
        .expect("filtered scan");
    assert_eq!(r.rows.len(), 3);
    for row in &r.rows {
        let cid = row[0].as_text().expect("text");
        assert_eq!(cid, "ctx-a", "ctx-b row leaked through filter");
    }
}

/// Idempotent re-apply: applying the same `(natural_key, value, committed_at)`
/// tuple twice (NATS at-least-once redelivery) must result in the same final
/// state and not corrupt the row.
///
/// Probed via two upserts with identical content. Final state is one row.
#[test]
fn idempotent_re_apply_same_payload_no_corruption() {
    let db = make_db();
    for round in 0..2 {
        let tx = db.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        row.insert("natural_key".into(), Value::Text("key-1".into()));
        row.insert("value".into(), Value::Text("payload".into()));
        row.insert("committed_at".into(), Value::TxId(tx));
        db.upsert_row(tx, "replicated_rows", "natural_key", row)
            .expect(&format!("upsert round {}", round));
        db.commit(tx).expect("commit");
    }

    let r = db
        .execute(
            "SELECT value FROM replicated_rows WHERE natural_key = 'key-1'",
            &HashMap::new(),
        )
        .expect("scan");
    assert_eq!(r.rows.len(), 1);
    assert_eq!(r.rows[0][0].as_text(), Some("payload"));
}

/// C-SC3 graceful shutdown vs in-flight write — engine-level analogue.
///
/// Simulate a writer thread that begins a tx, does some work, but never
/// commits — when the database handle is dropped (closing), uncommitted
/// state must NOT appear.
///
/// Probed by: writer begins tx, inserts a row, never commits; reader
/// (post-tx) confirms the row is absent.
///
/// Lens: lens-question row "Multi-table TX (combination)" — partial-commit
/// is wrong-impl. Also: rollback semantics on tx not explicitly committed.
#[test]
fn uncommitted_tx_invisible_to_subsequent_reader() {
    let db = make_db();
    let tx = db.begin();
    let mut row = HashMap::new();
    row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
    row.insert("natural_key".into(), Value::Text("ghost".into()));
    row.insert("value".into(), Value::Text("phantom".into()));
    row.insert("committed_at".into(), Value::TxId(tx));
    db.insert_row(tx, "replicated_rows", row).expect("insert");
    // Roll back instead of committing — analogue of crash before commit.
    db.rollback(tx).expect("rollback");

    let r = db
        .execute(
            "SELECT value FROM replicated_rows WHERE natural_key = 'ghost'",
            &HashMap::new(),
        )
        .expect("scan");
    assert_eq!(
        r.rows.len(),
        0,
        "uncommitted tx surfaced row to post-rollback reader"
    );
}

/// C-SC1 hardening: snapshot read in the middle of an uncommitted tx
/// must NOT see the in-flight tx's writes (snapshot isolation).
///
/// Probed by: thread A begins tx and inserts; thread B reads via fresh
/// snapshot; thread B sees zero rows under the new natural_key. A then
/// commits; thread C reads via fresh snapshot and sees the row.
#[test]
fn snapshot_isolation_in_flight_writes_invisible_to_other_reader() {
    let db = make_db();

    let barrier = Arc::new(Barrier::new(2));
    let db_a = db.clone();
    let bar_a = barrier.clone();
    // We use a channel-like sync via Atomic to coordinate "A has inserted
    // but not committed" → "B reads".
    let (tx_to_b, rx_from_a) = std::sync::mpsc::sync_channel::<()>(0);
    let (tx_to_a, rx_from_b) = std::sync::mpsc::sync_channel::<()>(0);

    let handle_a = thread::spawn(move || {
        bar_a.wait();
        let tx = db_a.begin();
        let mut row = HashMap::new();
        row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
        row.insert("natural_key".into(), Value::Text("iso-1".into()));
        row.insert("value".into(), Value::Text("a-write".into()));
        row.insert("committed_at".into(), Value::TxId(tx));
        db_a.insert_row(tx, "replicated_rows", row).expect("ins");
        // Tell B: "I've inserted but not committed"
        tx_to_b.send(()).expect("send-to-b");
        // Wait for B to confirm read
        rx_from_b.recv().expect("recv-from-b");
        // Now commit
        db_a.commit(tx).expect("commit");
    });

    let db_b = db.clone();
    let bar_b = barrier.clone();
    let handle_b = thread::spawn(move || {
        bar_b.wait();
        // Wait for A to insert
        rx_from_a.recv().expect("recv-from-a");
        // Read — must NOT see A's uncommitted write
        let r = db_b
            .execute(
                "SELECT value FROM replicated_rows WHERE natural_key = 'iso-1'",
                &HashMap::new(),
            )
            .expect("scan");
        assert_eq!(
            r.rows.len(),
            0,
            "B read A's uncommitted insert — snapshot isolation broken"
        );
        // Tell A to commit
        tx_to_a.send(()).expect("send-to-a");
    });

    handle_a.join().expect("a");
    handle_b.join().expect("b");

    // After both done, row IS visible
    let r = db
        .execute(
            "SELECT value FROM replicated_rows WHERE natural_key = 'iso-1'",
            &HashMap::new(),
        )
        .expect("final");
    assert_eq!(r.rows.len(), 1);
}

/// Sync-apply analogue: high-throughput concurrent commits across many
/// distinct natural keys do not produce duplicate rows.
///
/// 6 threads x 16 inserts each = 96 distinct natural keys. Final row count
/// must be exactly 96.
#[test]
fn concurrent_sync_high_throughput_no_loss_no_duplication() {
    let db = make_db();
    let nthreads = 6;
    let per = 16;
    let barrier = Arc::new(Barrier::new(nthreads));
    let mut handles = vec![];
    for tid in 0..nthreads {
        let db_t = db.clone();
        let bar = barrier.clone();
        handles.push(thread::spawn(move || {
            bar.wait();
            for i in 0..per {
                let nk = format!("hi-{}-{}", tid, i);
                let tx = db_t.begin();
                let mut row = HashMap::new();
                row.insert("id".into(), Value::Uuid(uuid::Uuid::new_v4()));
                row.insert("natural_key".into(), Value::Text(nk));
                row.insert("value".into(), Value::Text("v".into()));
                row.insert("committed_at".into(), Value::TxId(tx));
                db_t.insert_row(tx, "replicated_rows", row).expect("ins");
                db_t.commit(tx).expect("commit");
            }
        }));
    }
    for h in handles {
        h.join().expect("worker");
    }

    let r = db
        .execute("SELECT id FROM replicated_rows", &HashMap::new())
        .expect("scan");
    assert_eq!(
        r.rows.len(),
        nthreads * per,
        "concurrent commits lost or duplicated rows: got {}, expected {}",
        r.rows.len(),
        nthreads * per
    );
}
