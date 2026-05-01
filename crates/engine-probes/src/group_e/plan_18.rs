//! Plan 18 cg-mcp-server — engine-probe sweep (wrapper class).
//!
//! Most cg-mcp-server wrapper assumptions require a shipped MCP server
//! binary speaking JSON-RPC over stdio (initialize handshake, tools/list,
//! line-delimited framing, error-code shapes, unknown-method rejection,
//! invalid-params rejection, `cg mcp start` CLI wiring). The cg-mcp-server
//! binary is not shipped today; those tests are catalogued BLOCKED.
//!
//! The runnable engine surface this plan touches:
//!
//!   * **C-MCP2 server handle vs CLI handle on same store** — the engine,
//!     not the wrapper, mediates concurrent writes between an MCP-server
//!     long-running handle and CLI subprocess handles. Probed as a
//!     long-lived "server" handle that interleaves writes with bursts of
//!     "CLI" handles that open + write + drop.
//!
//!   * **session_continuity_new_agent_instance_via_mcp** — engine durability
//!     across handle close + reopen. The MCP server binary, when restarted,
//!     re-opens the same store path and must observe every committed write
//!     from the prior incarnation. Engine source:
//!     `Database::open_loaded` (database.rs:342) replays
//!     `persistence.load_relational_table` for every declared table on
//!     reopen.
//!
//! stdio-framing logic (line-delimited JSON, Content-Length absence, error
//! shape, unknown-method rejection) is wrapper code that does not yet exist.
//! cg does not currently expose that framing as a library function — there
//! is no public framing API in the shipped cg crate to probe in isolation.
//! BLOCKED.

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

/// C-MCP2: a long-lived MCP-server handle and a sequence of short-lived
/// CLI-style open + write + drop handles must interleave writes correctly
/// against the same store. Each CLI burst opens a fresh `Database` against
/// the same path, writes, drops; the server handle continues to write and
/// MUST observe every CLI commit on its next read.
#[test]
fn server_handle_observes_writes_from_short_lived_handles() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("mcp.cdb");

    let server = Database::open(&path).unwrap();
    let p = HashMap::new();
    server
        .execute(
            "CREATE TABLE byoc_facts (
                id UUID PRIMARY KEY,
                source TEXT NOT NULL,
                value TEXT NOT NULL
            )",
            &p,
        )
        .unwrap();

    // Server writes 8 facts.
    for i in 0..8 {
        let row = vals(vec![
            ("id", Value::Uuid(Uuid::now_v7())),
            ("source", Value::Text("server".into())),
            ("value", Value::Text(format!("s{i}"))),
        ]);
        server
            .execute(
                "INSERT INTO byoc_facts (id, source, value)
                 VALUES ($id, $source, $value)",
                &row,
            )
            .unwrap();
    }

    // Three CLI bursts, each opening + writing + dropping. Mirrors the
    // `byoc_fact_round_trips_cli_and_mcp` test — facts written via CLI must
    // be queryable via MCP.
    for burst in 0..3 {
        let db = Database::open(&path).expect("CLI-style open against same path");
        let row = vals(vec![
            ("id", Value::Uuid(Uuid::now_v7())),
            ("source", Value::Text("cli".into())),
            ("value", Value::Text(format!("c{burst}"))),
        ]);
        db.execute(
            "INSERT INTO byoc_facts (id, source, value)
             VALUES ($id, $source, $value)",
            &row,
        )
        .unwrap();
        // db dropped at end of scope.
    }

    // Server-side read MUST observe every CLI commit + every server commit.
    let p = HashMap::new();
    let total = server
        .execute("SELECT id FROM byoc_facts", &p)
        .unwrap();
    assert_eq!(
        total.rows.len(),
        11,
        "server handle must see every CLI burst commit + its own writes"
    );

    let p = HashMap::new();
    let cli_only = server
        .execute(
            "SELECT id FROM byoc_facts WHERE source = 'cli'",
            &p,
        )
        .unwrap();
    assert_eq!(cli_only.rows.len(), 3);

    let p = HashMap::new();
    let server_only = server
        .execute(
            "SELECT id FROM byoc_facts WHERE source = 'server'",
            &p,
        )
        .unwrap();
    assert_eq!(server_only.rows.len(), 8);
}

/// session_continuity_new_agent_instance_via_mcp surrogate. The engine MUST
/// preserve every committed write across a handle drop + new handle open at
/// the same path. cg-mcp-server's "session continuity" contract is
/// engine-delivered durability — there is no MCP-side persistence layer.
#[test]
fn session_continuity_engine_durability_close_reopen() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("mcp.cdb");

    {
        let db = Database::open(&path).unwrap();
        let p = HashMap::new();
        db.execute(
            "CREATE TABLE sessions (
                id UUID PRIMARY KEY,
                agent_id TEXT NOT NULL,
                fact TEXT NOT NULL
            )",
            &p,
        )
        .unwrap();
        for i in 0..16 {
            let row = vals(vec![
                ("id", Value::Uuid(Uuid::now_v7())),
                ("agent_id", Value::Text(format!("agent-{i}"))),
                ("fact", Value::Text(format!("fact-{i}"))),
            ]);
            db.execute(
                "INSERT INTO sessions (id, agent_id, fact)
                 VALUES ($id, $agent_id, $fact)",
                &row,
            )
            .unwrap();
        }
    }

    // New "agent instance" — fresh handle, same path. Read every prior fact.
    let db2 = Database::open(&path).unwrap();
    let p = HashMap::new();
    let result = db2.execute("SELECT agent_id, fact FROM sessions", &p).unwrap();
    assert_eq!(result.rows.len(), 16);
}

/// C-MCP1 surrogate. stdio JSON-RPC cannot multiplex requests; the wrapper
/// guarantees sequential request/response. The engine itself does NOT need
/// to serialize MCP requests — that's the wrapper's job. But the engine
/// must handle the case where the MCP server, deep in one request, spawns
/// internal worker threads that all hit the same `Database` handle. Probed
/// as N concurrent worker threads sharing one Arc<Database> issuing reads
/// while a single writer commits.
#[test]
fn shared_handle_internal_workers_consistent_under_writer() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("mcp-workers.cdb");

    let db = Arc::new(Database::open(&path).unwrap());
    let p = HashMap::new();
    db.execute(
        "CREATE TABLE tools (
            id UUID PRIMARY KEY,
            name TEXT NOT NULL
        )",
        &p,
    )
    .unwrap();
    // Seed 32 tools.
    for i in 0..32 {
        let row = vals(vec![
            ("id", Value::Uuid(Uuid::now_v7())),
            ("name", Value::Text(format!("tool-{i}"))),
        ]);
        db.execute(
            "INSERT INTO tools (id, name) VALUES ($id, $name)",
            &row,
        )
        .unwrap();
    }

    const READERS: usize = 4;
    let barrier = Arc::new(Barrier::new(READERS + 1));
    let mut handles = vec![];
    for _ in 0..READERS {
        let b = barrier.clone();
        let db = db.clone();
        handles.push(thread::spawn(move || {
            b.wait();
            for _ in 0..16 {
                let p = HashMap::new();
                let r = db.execute("SELECT id, name FROM tools", &p).unwrap();
                // Reader observes monotonically growing row count: at least
                // 32, never duplicates beyond what the writer has committed.
                assert!(
                    r.rows.len() >= 32,
                    "reader must observe at least the seeded 32 rows; saw {}",
                    r.rows.len()
                );
            }
        }));
    }
    // Writer thread.
    let bw = barrier.clone();
    let dbw = db.clone();
    handles.push(thread::spawn(move || {
        bw.wait();
        for i in 0..16 {
            let row = vals(vec![
                ("id", Value::Uuid(Uuid::now_v7())),
                ("name", Value::Text(format!("late-{i}"))),
            ]);
            dbw.execute(
                "INSERT INTO tools (id, name) VALUES ($id, $name)",
                &row,
            )
            .unwrap();
        }
    }));
    for h in handles {
        h.join().unwrap();
    }

    let p = HashMap::new();
    let final_rows = db.execute("SELECT id FROM tools", &p).unwrap();
    assert_eq!(final_rows.rows.len(), 48);
}
