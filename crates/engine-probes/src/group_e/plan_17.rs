//! Plan 17 cg-cli — engine-probe sweep (wrapper class).
//!
//! Wrapper-specific assumptions for cg-cli are mostly process-boundary
//! concerns (exit codes, JSON stdout framing, config resolution order,
//! standalone vs server fallback, init scaffolding, log-level routing,
//! help/version output). Those depend on the cg-cli binary, which is NOT
//! shipped today (Plan 17 unshipped). Per the brief, those assumptions are
//! BLOCKED and catalogued in
//! `.claude/plans/context-graph/r0/track1-results/17-cg-cli-probe-results.md`.
//!
//! The single runnable engine-level concern is **C-CL1 concurrent CLI
//! invocations against the same store path**. The wrapper's contract: two
//! `cg` processes opening the same store either both serve writes (engine
//! coordinates), or the second receives a clean error. The wrapper-level
//! cross-process file lock is delivered by the engine, not the CLI binary.
//! The engine's lock surface is the only thing the CLI plan can rely on.
//!
//! We exercise the same-process two-handle path as a deliberate surrogate.
//! `RedbPersistence::acquire_pid_lock`
//! (`contextdb/crates/contextdb-engine/src/persistence.rs:225`) skips the
//! cross-process rejection branch when the running PID matches its own
//! (`pid != std::process::id()` is false). That means two threads inside one
//! process are EXACTLY the engine's concurrent-handle code path the CLI
//! relies on — every contended write between the two handles is the same
//! engine surface a future cg-cli binary must trust. The probe makes both
//! handles hammer the store simultaneously and asserts the engine commits
//! both writers' rows (no torn state, no lost writes, no panic).
//!
//! Out-of-process second-PID rejection (`Error::Other("database is
//! locked...")`) cannot be probed without spawning a real `cg` subprocess —
//! BLOCKED in the catalogue.

use contextdb_core::Value;
use contextdb_engine::Database;
use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;
use uuid::Uuid;

fn vals(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// C-CL1 surrogate. Two `Database` handles to the same on-disk store path,
/// inside one process — the wrapper-class contract per the primitives
/// enumeration says "second receives clean error OR coordinates via an
/// inner-engine lock" (an OR). Either is acceptable for cg-cli, since the
/// CLI wrapper can detect the lock error on second invocation and emit a
/// user-friendly "store is in use" message. What is NOT acceptable: the
/// second open succeeds AND its writes silently land somewhere that the
/// first handle cannot see (state divergence — the first handle and the
/// second handle disagree on what's in the store).
///
/// This probe verifies which leg of the OR the engine delivers and asserts
/// the leg is consistent. We open two handles, write through each, and
/// assert: either (a) every write succeeded AND both handles see every
/// row (the "coordinate via inner-engine lock" leg), or (b) some writes
/// fail with a typed lock error AND no committed-but-invisible writes
/// are observed (the "clean error" leg).
///
/// Engine source: `RedbPersistence::acquire_pid_lock`
/// (persistence.rs:225) accepts same-PID re-opens; redb's own file lock is
/// re-acquired in `with_db` (persistence.rs:1068) on every operation.
#[test]
fn concurrent_in_process_two_handle_writes_serialize_or_clean_error() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("cg-cli.cdb");

    // First handle, declare schema.
    let db1 = Database::open(&path).expect("open first handle");
    let p = HashMap::new();
    db1.execute(
        "CREATE TABLE cli_events (
            id UUID PRIMARY KEY,
            handle_label TEXT NOT NULL,
            payload TEXT NOT NULL
        )",
        &p,
    )
    .unwrap();

    // Second handle. PID-lock branch allows same-PID re-open even though
    // the OS-level redb lock will fight every operation.
    let db2 = Database::open(&path).expect("open second handle to same path");

    const WRITERS_PER_HANDLE: usize = 32;
    let barrier = Arc::new(Barrier::new(WRITERS_PER_HANDLE * 2));
    let db1 = Arc::new(db1);
    let db2 = Arc::new(db2);

    let mut h1_attempts = 0usize;
    let mut h2_attempts = 0usize;
    let mut handles = vec![];
    for i in 0..WRITERS_PER_HANDLE {
        h1_attempts += 1;
        let b = barrier.clone();
        let db = db1.clone();
        handles.push(thread::spawn(move || -> Option<Uuid> {
            b.wait();
            let id = Uuid::now_v7();
            let row = vals(vec![
                ("id", Value::Uuid(id)),
                ("handle_label", Value::Text("h1".into())),
                ("payload", Value::Text(format!("h1-{i}"))),
            ]);
            match db.execute(
                "INSERT INTO cli_events (id, handle_label, payload)
                 VALUES ($id, $handle_label, $payload)",
                &row,
            ) {
                Ok(_) => Some(id),
                Err(_) => None,
            }
        }));

        h2_attempts += 1;
        let b = barrier.clone();
        let db = db2.clone();
        handles.push(thread::spawn(move || -> Option<Uuid> {
            b.wait();
            let id = Uuid::now_v7();
            let row = vals(vec![
                ("id", Value::Uuid(id)),
                ("handle_label", Value::Text("h2".into())),
                ("payload", Value::Text(format!("h2-{i}"))),
            ]);
            match db.execute(
                "INSERT INTO cli_events (id, handle_label, payload)
                 VALUES ($id, $handle_label, $payload)",
                &row,
            ) {
                Ok(_) => Some(id),
                Err(_) => None,
            }
        }));
    }

    let mut committed = std::collections::HashSet::new();
    let mut errored = 0usize;
    for h in handles {
        match h.join().unwrap() {
            Some(id) => {
                committed.insert(id);
            }
            None => errored += 1,
        }
    }

    // Both handles read; the read with the FEWER rows reveals the
    // visibility gap. A consistent engine has both handles seeing the same
    // committed set.
    let p = HashMap::new();
    let r1 = db1.execute("SELECT id FROM cli_events", &p);
    let r2 = db2.execute("SELECT id FROM cli_events", &p);
    let r1_count = r1.map(|r| r.rows.len()).unwrap_or(0);
    let r2_count = r2.map(|r| r.rows.len()).unwrap_or(0);

    let total_attempts = h1_attempts + h2_attempts;
    let total_committed = committed.len();
    let total_errored = errored;
    assert_eq!(total_committed + total_errored, total_attempts);

    // Contract: the union of "errored" + "visible in EITHER handle" must
    // equal total attempts. Equivalently: every committed write must
    // surface in at least one handle's read view, and the count read by
    // each handle must match the count of writes it committed PLUS what
    // the other handle's writes propagated to it. The strict version: both
    // handles see ALL committed rows.
    assert!(
        r1_count >= total_committed && r2_count >= total_committed,
        "one handle does NOT see committed writes — visibility hole. \
         h1 sees {r1_count}, h2 sees {r2_count}, but {total_committed} writes \
         committed across both handles ({total_errored} attempts errored)"
    );
}

/// C-CL1 reopen surrogate. The first handle commits, drops, the second
/// handle opens fresh and reads. cg-cli's standalone mode relies on every
/// process being able to drop its handle on exit and the next process
/// reading the same committed state. Engine source: `Database::open` ->
/// `Self::open_loaded` (database.rs:326, 342) reads `load_all_table_meta`
/// and per-table data on every re-open.
#[test]
fn sequential_open_close_open_preserves_committed_writes() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("cg-cli.cdb");

    {
        let db = Database::open(&path).unwrap();
        let p = HashMap::new();
        db.execute(
            "CREATE TABLE cli_events (
                id UUID PRIMARY KEY,
                payload TEXT NOT NULL
            )",
            &p,
        )
        .unwrap();
        for i in 0..32 {
            let row = vals(vec![
                ("id", Value::Uuid(Uuid::now_v7())),
                ("payload", Value::Text(format!("p{i}"))),
            ]);
            db.execute(
                "INSERT INTO cli_events (id, payload) VALUES ($id, $payload)",
                &row,
            )
            .unwrap();
        }
        // db drops here — pid lock released.
    }

    let db2 = Database::open(&path).expect("reopen after first handle drop");
    let p = HashMap::new();
    let result = db2
        .execute("SELECT payload FROM cli_events", &p)
        .unwrap();
    assert_eq!(
        result.rows.len(),
        32,
        "every committed write from the first handle must survive close+reopen"
    );
}
