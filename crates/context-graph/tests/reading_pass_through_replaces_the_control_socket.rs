//! Asking the process that owns a store an administrative question goes through contextdb's
//! reading surface, and context-graph no longer carries a transport of its own.
//!
//! The contract: a live owner registers its `ControlHandler` on the writable open, and a client
//! names only the STORE PATH — `context_graph::owner_control::request_owner(store_path, frame)`
//! over `ReadSession::request_owner(namespace, bytes)`. There is no socket path in this crate's
//! API and no second transport in its source. A caller whose fallback is to open the store itself
//! must be able to tell "nobody holds this store" apart from "an owner is there and refused", so
//! the no-owner answer is a distinct, typed variant and no application code runs on the way to it.
//! One place declares what a context-graph read may spend, and the owner reports those same
//! ceilings back.
//!
//! ## Public contract this file proves
//!
//! - `context_graph::owner_control::{CONTROL_NAMESPACE, ControlHandler, OwnerControlError,
//!   context_graph_read_limits, owner_read_config, owner_request_handler, request_owner}`
//! - `context_graph::Store::open_with_control_handler(config: StoreConfig, handler: ControlHandler)
//!   -> context_graph::Result<Store>` — the production writable open that installs the handler
//!   through `Database::open_with_options`, so a cg store IS the owner clients ask.
//!
//! That the retired names are gone is proved by a scan of the crate's shipped source rather than by
//! a `trybuild` case. A compile-fail case proves two names no longer resolve; the contract here is
//! wider — nothing anywhere in this crate binds, dials, or takes a socket path — and only reading
//! the source proves that. It also cannot rot against compiler-message wording.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use context_graph::owner_control::{
    CONTROL_NAMESPACE, ControlHandler, OwnerControlError, context_graph_read_limits, request_owner,
};
use context_graph::{Store, StoreConfig};
use contextdb_core::read_contract::ReadLimits;
use contextdb_engine::read_session::{ReadSession, ReadSessionOptions};

/// A handler that answers control frames and counts every frame it was actually asked to run.
fn counting_handler() -> (ControlHandler, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let handler: ControlHandler = Arc::new(move |frame: String| {
        counter.fetch_add(1, Ordering::SeqCst);
        format!("handled:{}", frame.trim())
    });
    (handler, calls)
}

fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    }
}

#[test]
fn a_live_owner_answers_a_control_frame_addressed_only_by_the_store_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let (handler, calls) = counting_handler();

    let store = Store::open_with_control_handler(store_config(db_path.clone()), handler)
        .expect("open a store that serves administrative owner requests");
    assert_eq!(store.db_path(), db_path.as_path());

    let answer = request_owner(&db_path, "ping\n").expect("the live owner answers a control frame");

    assert_eq!(
        answer, "handled:ping",
        "the answer is the owner handler's own reply, carried back unaltered"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "exactly one frame reached the owner's handler"
    );
}

#[test]
fn a_frame_in_another_namespace_never_reaches_the_context_graph_handler() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let (handler, calls) = counting_handler();
    let _store = Store::open_with_control_handler(store_config(db_path.clone()), handler)
        .expect("open a store that serves administrative owner requests");

    let session = ReadSession::open_with_options(
        &db_path,
        ReadSessionOptions {
            limits: context_graph_read_limits(),
            ..ReadSessionOptions::default()
        },
    )
    .expect("open a reading session on the owned store");

    // Whatever the surface answers a foreign namespace, context-graph's application code is not it.
    let _ = session.request_owner("some-other-consumer/control", b"ping");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "a frame outside `{CONTROL_NAMESPACE}` must never run context-graph's handler"
    );
}

#[test]
fn a_store_nobody_holds_answers_owner_not_running_without_running_any_handler() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let (handler, calls) = counting_handler();

    // The store existed, served, and was then given up: the file is still there, the owner is not.
    let store = Store::open_with_control_handler(store_config(db_path.clone()), handler)
        .expect("open a store that serves administrative owner requests");
    drop(store);

    let refusal = request_owner(&db_path, "ping\n")
        .expect_err("a store nobody holds cannot answer an administrative frame");

    assert!(
        matches!(refusal, OwnerControlError::OwnerNotRunning { .. }),
        "a caller with its own fallback needs the no-owner answer typed and distinct, got {refusal:?}"
    );
    assert_eq!(
        refusal.store_path(),
        db_path.as_path(),
        "the refusal names the store that was asked"
    );
    assert!(
        refusal.owner_absent(),
        "`owner_absent` is what a fallback caller branches on"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no application code runs on the way to a no-owner refusal"
    );
}

#[test]
fn the_owner_reports_back_the_read_ceilings_this_crate_declares() {
    context_graph_read_limits()
        .validate()
        .expect("the ceilings context-graph declares are a valid read contract");

    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let (handler, _calls) = counting_handler();
    let _store = Store::open_with_control_handler(store_config(db_path.clone()), handler)
        .expect("open a store that serves administrative owner requests");

    let declared: ReadLimits = context_graph_read_limits();
    let report = ReadSession::owner_report(
        &db_path,
        ReadSessionOptions {
            limits: declared,
            ..ReadSessionOptions::default()
        },
    )
    .expect("the live owner reports how it is serving");
    let effective = report
        .serving
        .expect("a live owner reports the ceilings it is serving under")
        .effective_limits;

    for (reported, declared_value, field) in [
        (
            effective.result_rows.value,
            declared.result_rows,
            "result_rows",
        ),
        (
            effective.result_bytes.value,
            declared.result_bytes,
            "result_bytes",
        ),
        (effective.work.value, declared.work, "work"),
        (effective.active_ms.value, declared.active_ms, "active_ms"),
        (effective.memory.value, declared.memory, "memory"),
        (
            effective.cursor_page_rows.value,
            declared.cursor_page_rows,
            "cursor_page_rows",
        ),
        (
            effective.cursor_page_bytes.value,
            declared.cursor_page_bytes,
            "cursor_page_bytes",
        ),
        (
            effective.cursor_idle_ms.value,
            declared.cursor_idle_ms,
            "cursor_idle_ms",
        ),
        (
            effective.cursor_lifetime_ms.value,
            declared.cursor_lifetime_ms,
            "cursor_lifetime_ms",
        ),
    ] {
        assert_eq!(
            reported, declared_value,
            "the owner serves the `{field}` ceiling context-graph declared, not another one"
        );
    }
}

#[test]
fn the_crate_carries_no_socket_transport_and_no_path_taking_control_api() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = crate_root.join("src");

    assert!(
        !src.join("control_transport.rs").exists(),
        "the socket control transport module must be gone from {}",
        src.display()
    );

    let lib = std::fs::read_to_string(src.join("lib.rs")).expect("read the crate root module");
    for gone in [
        "control_transport",
        "request_control",
        "start_control_listener",
    ] {
        assert!(
            !lib.contains(gone),
            "`{gone}` must not appear in the crate's public surface (src/lib.rs)"
        );
    }
    assert!(
        lib.contains("owner_control"),
        "the database-backed owner plane replaces it and must be a public module"
    );

    let mut offenders: Vec<String> = Vec::new();
    for file in rust_sources(&src) {
        let text = std::fs::read_to_string(&file).expect("read a crate source file");
        for gone in [
            "UnixListener",
            "UnixStream",
            "socket_path",
            "start_control_listener",
            "request_control",
        ] {
            if text.contains(gone) {
                offenders.push(format!("{}: {gone}", file.display()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "context-graph must own no transport of its own — the store path is the whole address: {offenders:?}"
    );
}

fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).expect("read a crate source directory") {
            let path = entry.expect("read a crate source entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }
    found
}
