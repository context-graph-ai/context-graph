//! Plan 22 cg-import-chat-history — engine-probe sweep (wrapper class).
//!
//! Wrapper plan covering one-shot file ingestion of chat-history exports
//! (ChatGPT, Claude.ai web export, Claude Code session, Gemini CLI
//! history). Format detection, on-disk format compatibility, and the
//! parser surface itself are wrapper code that does not exist as shipped
//! cg-side library functions today — those are catalogued BLOCKED.
//!
//! Runnable engine surface this plan depends on:
//!
//!   * **Idempotency on re-import / mid-import crash recovery** — when the
//!     import process is killed mid-import and re-run, every row already
//!     committed must be detected as a duplicate and skipped, every row
//!     not yet committed must land. The contract delivered by the engine
//!     is `UNIQUE` constraint + duplicate-insert no-op semantics. Probed.
//!
//!   * **C-IM1 concurrent imports of the same file** — dedup keeps row
//!     count stable. Same engine contract: UNIQUE + duplicate-insert
//!     no-op under concurrent writers. Probed.
//!
//!   * **C-IM2 concurrent imports of different files** — engine handles
//!     concurrent writes against the same store. Same shape as Plan 17's
//!     two-handle probe; not duplicated.
//!
//! Engine source: UNIQUE constraint + duplicate-insert no-op behaviour
//! returns `UpsertResult::NoOp` (or rows_affected=0 in the SQL surface) on
//! a duplicate. cg-app dedup via observation hash (a per-record content
//! hash mapped to a UNIQUE column) is the wrapper's mechanism; the
//! engine's role is to refuse the duplicate.

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

/// Mid-import crash + resume. The wrapper imports N records by INSERT,
/// each carrying a UNIQUE observation_hash. After half the records are
/// committed, the import process is "killed" (we drop the handle). The
/// resume opens a fresh handle and replays ALL N records — the engine
/// must accept records 0..N/2 as no-ops (already committed) and records
/// N/2..N as fresh inserts. Final row count = N.
#[test]
fn import_resume_dedups_via_unique_observation_hash() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("import.cdb");

    const N: usize = 32;
    // Pre-compute deterministic UUIDs + hashes for each record so first
    // and second pass both produce identical primary keys.
    let records: Vec<(Uuid, String)> = (0..N)
        .map(|i| (Uuid::from_u128(0x1000 + i as u128), format!("hash-{i}")))
        .collect();

    // First pass: commit half the records, then drop the handle (simulating
    // a kill mid-import).
    {
        let db = Database::open(&path).unwrap();
        let p = HashMap::new();
        db.execute(
            "CREATE TABLE chat_messages (
                id UUID PRIMARY KEY,
                observation_hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL
            )",
            &p,
        )
        .unwrap();
        for (id, hash) in &records[..N / 2] {
            let row = vals(vec![
                ("id", Value::Uuid(*id)),
                ("observation_hash", Value::Text(hash.clone())),
                ("content", Value::Text(format!("msg-{hash}"))),
            ]);
            db.execute(
                "INSERT INTO chat_messages (id, observation_hash, content)
                 VALUES ($id, $observation_hash, $content)",
                &row,
            )
            .unwrap();
        }
    }

    // Resume: replay ALL N records. Records 0..N/2 must be detected as
    // duplicates; records N/2..N must commit. Engine contract: UNIQUE
    // violation on observation_hash returns a typed error OR no-op; cg-app
    // wrapper handles either by skipping.
    let db = Database::open(&path).unwrap();
    let mut commits = 0;
    let mut skips = 0;
    for (id, hash) in &records {
        let row = vals(vec![
            ("id", Value::Uuid(*id)),
            ("observation_hash", Value::Text(hash.clone())),
            ("content", Value::Text(format!("msg-{hash}"))),
        ]);
        let result = db.execute(
            "INSERT INTO chat_messages (id, observation_hash, content)
             VALUES ($id, $observation_hash, $content)",
            &row,
        );
        match result {
            Ok(qr) => {
                if qr.rows_affected == 0 {
                    skips += 1;
                } else {
                    commits += 1;
                }
            }
            Err(_) => {
                // UNIQUE-violation error path is also valid — the wrapper
                // catches it and treats it as a skip.
                skips += 1;
            }
        }
    }

    let p = HashMap::new();
    let rows = db
        .execute("SELECT id FROM chat_messages", &p)
        .unwrap();
    assert_eq!(
        rows.rows.len(),
        N,
        "resume must produce exactly N committed rows; saw {} commits + {} skips",
        commits,
        skips
    );
    assert!(skips >= N / 2, "first half MUST be detected as already committed");
}

/// C-IM1 surrogate. Two concurrent imports of the same file content (same
/// observation_hash set) MUST settle to a stable row count. Each import
/// attempts to INSERT every record; the engine's UNIQUE constraint
/// guarantees only the first writer to land each hash succeeds.
#[test]
fn concurrent_imports_of_same_file_dedup_to_stable_row_count() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("import-same.cdb");
    let db = Arc::new(Database::open(&path).unwrap());
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE chat_messages (
            id UUID PRIMARY KEY,
            observation_hash TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL
        )",
        &p,
    )
    .unwrap();

    const N: usize = 32;
    let records: Vec<(String, String)> = (0..N)
        .map(|i| (format!("hash-{i}"), format!("msg-{i}")))
        .collect();
    let records = Arc::new(records);

    let importers = 4;
    let barrier = Arc::new(Barrier::new(importers));
    let mut handles = vec![];
    for _ in 0..importers {
        let b = barrier.clone();
        let db = db.clone();
        let recs = records.clone();
        handles.push(thread::spawn(move || {
            b.wait();
            for (hash, content) in recs.iter() {
                let row = vals(vec![
                    // Each importer generates its own primary-key UUID;
                    // only the observation_hash is stable. The UNIQUE
                    // constraint on observation_hash is what dedups.
                    ("id", Value::Uuid(Uuid::now_v7())),
                    ("observation_hash", Value::Text(hash.clone())),
                    ("content", Value::Text(content.clone())),
                ]);
                let _ = db.execute(
                    "INSERT INTO chat_messages (id, observation_hash, content)
                     VALUES ($id, $observation_hash, $content)",
                    &row,
                );
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let p = HashMap::new();
    let rows = db
        .execute("SELECT observation_hash FROM chat_messages", &p)
        .unwrap();
    assert_eq!(
        rows.rows.len(),
        N,
        "concurrent imports of same content MUST converge to N rows; UNIQUE \
         on observation_hash dedups the duplicates regardless of which \
         importer landed first"
    );
}

/// Sequential reopen contract: each import session runs as a separate `cg`
/// invocation that opens, writes, drops. After M sessions importing the
/// same content, row count is still N — the engine's durable UNIQUE
/// enforcement survives close + reopen.
#[test]
fn sequential_import_sessions_dedup_across_close_reopen() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("import-seq.cdb");

    {
        let db = Database::open(&path).unwrap();
        let p = HashMap::new();
        db.execute(
            "CREATE TABLE chat_messages (
                id UUID PRIMARY KEY,
                observation_hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL
            )",
            &p,
        )
        .unwrap();
    }

    const N: usize = 24;
    const SESSIONS: usize = 4;
    let records: Vec<(String, String)> = (0..N)
        .map(|i| (format!("hash-{i}"), format!("msg-{i}")))
        .collect();

    for _session in 0..SESSIONS {
        let db = Database::open(&path).unwrap();
        for (hash, content) in &records {
            let row = vals(vec![
                ("id", Value::Uuid(Uuid::now_v7())),
                ("observation_hash", Value::Text(hash.clone())),
                ("content", Value::Text(content.clone())),
            ]);
            let _ = db.execute(
                "INSERT INTO chat_messages (id, observation_hash, content)
                 VALUES ($id, $observation_hash, $content)",
                &row,
            );
        }
    }

    let db = Database::open(&path).unwrap();
    let p = HashMap::new();
    let rows = db
        .execute("SELECT observation_hash FROM chat_messages", &p)
        .unwrap();
    assert_eq!(
        rows.rows.len(),
        N,
        "M sequential import sessions must converge to N rows via UNIQUE \
         enforcement across close+reopen"
    );
}
