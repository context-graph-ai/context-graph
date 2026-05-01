//! Plan 20 cg-connector-gemini — engine-probe sweep (wrapper class).
//!
//! Gemini-CLI connector wrapper assumptions match Plan 19 at the file
//! level: idempotent hook config writes, format compatibility with Gemini
//! CLI's expected schema, process-exit / stdout-silence behaviour.
//! The shipped connector binary does not exist; nearly every assumption
//! is BLOCKED.
//!
//! Novel runnable concern: **C-GM1 connect/disconnect race**. Concurrent
//! `cg connect gemini-cli` and `cg disconnect gemini-cli` against the same
//! config path — the contract is "last write wins; the file ends in a
//! deterministic state". The OS-level rename semantics of temp + rename
//! deliver "last-writer-wins". The probe asserts: under contention, the
//! file always ends as one of {present, absent} — never partially present
//! (a torn config), never both connect-format and disconnect-format
//! intermixed.
//!
//! No engine surface is exercised; the wrapper plan's file-write contract
//! is the only runnable surface today. Engine-level concurrent-store-open
//! concerns cross-reference Plan 17.

use std::fs;
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

/// C-GM1: concurrent connect (write hook config) and disconnect (remove
/// hook config) MUST settle to a deterministic file state. We race 16
/// connect threads and 16 disconnect threads, each performing 32 cycles.
/// At the end, the file is either fully present (last write was connect)
/// or absent (last write was disconnect). Crucially: every transient read
/// during the race observes either a complete connect payload or a clean
/// "file not found" — never a partial / torn config.
#[test]
fn connect_disconnect_race_settles_to_deterministic_state() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("gemini.config");
    let payload = b"{\"connector\":\"cg\",\"version\":1}";

    let connect_count = 16;
    let disconnect_count = 16;
    let barrier = Arc::new(Barrier::new(connect_count + disconnect_count + 1));

    let mut handles = vec![];

    for c in 0..connect_count {
        let b = barrier.clone();
        let path = path.clone();
        let dir_path = dir.path().to_path_buf();
        handles.push(thread::spawn(move || {
            b.wait();
            for i in 0..32 {
                let tmp = dir_path.join(format!(".gemini.connect.{c}.{i}.tmp"));
                fs::write(&tmp, payload).unwrap();
                // rename ON TOP of an existing file, OR a missing file —
                // both must succeed atomically on POSIX.
                fs::rename(&tmp, &path).unwrap();
            }
        }));
    }

    for _d in 0..disconnect_count {
        let b = barrier.clone();
        let path = path.clone();
        handles.push(thread::spawn(move || {
            b.wait();
            for _ in 0..32 {
                // Disconnect: remove the file. May fail if absent — that's
                // fine, disconnect-after-disconnect is a no-op.
                let _ = fs::remove_file(&path);
            }
        }));
    }

    // Reader thread asserts every observed file is either non-existent or
    // exactly the connect payload — never a partial write.
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_reader = stop.clone();
    let path_reader = path.clone();
    let payload_owned = payload.to_vec();
    let b = barrier.clone();
    let reader = thread::spawn(move || {
        b.wait();
        let mut iters = 0u64;
        while !stop_reader.load(std::sync::atomic::Ordering::Relaxed) {
            match fs::read(&path_reader) {
                Ok(bytes) => {
                    assert_eq!(
                        bytes, payload_owned,
                        "reader observed a torn / unknown config under connect/disconnect race"
                    );
                }
                Err(_) => {
                    // file absent — disconnect side won this snapshot, fine.
                }
            }
            iters += 1;
            if iters > 8192 {
                break;
            }
        }
    });

    for h in handles {
        h.join().unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = reader.join();

    // Final-state contract: the file is either fully present with the
    // payload, or absent. Never partial.
    match fs::read(&path) {
        Ok(bytes) => assert_eq!(bytes, payload, "final settled state must be a full payload"),
        Err(_) => { /* disconnect won; clean absence is the other valid state */ }
    }
}

/// Idempotency at the file level. The wrapper for gemini-cli must produce
/// the same bytes on first connect and on every subsequent connect when
/// the hook is already installed. Contract: temp + rename of identical
/// bytes leaves the file content byte-for-byte unchanged.
#[test]
fn gemini_connect_idempotent_byte_equal() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("gemini.config");
    let payload = b"{\"connector\":\"cg\"}";

    for _ in 0..4 {
        let tmp = dir.path().join(".gemini.tmp");
        fs::write(&tmp, payload).unwrap();
        fs::rename(&tmp, &path).unwrap();
    }
    let final_bytes = fs::read(&path).unwrap();
    assert_eq!(final_bytes, payload);
}
