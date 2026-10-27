//! Asking the running owner of a store a question must never be paid for by the store's file.
//!
//! The contract a caller is promised: when a process owns the store, the answer arrives over the
//! owner route; when NONE does, the caller gets a typed owner-absent result and the file behind the
//! path is not touched at all — not hydrated, not locked, no reader breadcrumb published, nothing
//! created beside it. That is not a tidiness preference. Reading the committed image publishes a
//! reader a writer starting beside it must wait for, so a readiness probe that fell through to the
//! file would stand in the way of the very process it is waiting for.
//!
//! ## Public contract this file proves
//!
//! - `context_graph::owner_control::request_owner(store_path, frame)` answers
//!   `OwnerControlError::OwnerNotRunning { store_path }` for a store nobody holds, and
//!   `owner_absent()` is true on it — that is what a caller with its own fallback branches on.
//! - The answer does not depend on the file: a path whose store file was written and then given up
//!   and a path that never had a store at all get the SAME typed answer, because the route decides
//!   before the file is looked at. A file-absence diagnosis would disclose a file inspection and
//!   violate this owner-only query contract.
//! - Nothing on disk moves: the store file is byte-for-byte what it was, and the directory holding
//!   it gains and loses no entry across the ask.
//!
//! Every assertion here is on state — typed variants, file bytes, directory entries, handler call
//! counters. Nothing measures elapsed time and nothing sleeps.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use context_graph::owner_control::{ControlHandler, OwnerControlError, request_owner};
use context_graph::{CreateContext, Store, StoreConfig};
use std::collections::BTreeMap;

/// A handler that answers control frames and counts every frame it was actually asked to run. A
/// non-zero count on a no-owner path would mean application code ran on the way to the answer.
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

/// The exact bytes of a file, or `None` when there is no file at that path. "The file was not
/// touched" is a claim about content, so content is what is compared.
fn file_bytes(path: &Path) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

/// Every entry in a directory, sorted. A reader breadcrumb, a lock file or a hydration sidecar
/// appearing beside the store would show up here.
fn directory_entries(directory: &Path) -> Vec<String> {
    let mut entries: Vec<String> = std::fs::read_dir(directory)
        .expect("read the directory holding the store")
        .map(|entry| {
            entry
                .expect("read one entry beside the store")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    entries.sort();
    entries
}

#[test]
fn a_store_nobody_holds_answers_owner_not_running_and_leaves_its_file_alone() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("given-up.db");
    let (handler, calls) = counting_handler();

    // The store existed, served, and was then given up: the file is still there, the owner is not.
    let store = Store::open_with_control_handler(store_config(db_path.clone()), handler)
        .expect("open a store that serves administrative owner requests");
    let context_name = format!("owner-absent-{}", uuid::Uuid::now_v7());
    store
        .create_context(CreateContext {
            name: context_name.clone(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("write something the store must still hold afterwards");
    drop(store);

    let bytes_before = file_bytes(&db_path).expect("the given-up store left its file behind");
    let entries_before = directory_entries(dir.path());

    let refusal = request_owner(&db_path, "status\n")
        .expect_err("a store nobody holds cannot answer an administrative frame");

    assert!(
        matches!(&refusal, OwnerControlError::OwnerNotRunning { store_path } if store_path == &db_path),
        "a caller with its own fallback needs the no-owner answer typed, distinct, and naming the \
         store it asked about, got {refusal:?}"
    );
    assert!(
        refusal.owner_absent(),
        "`owner_absent` is what a fallback caller branches on before opening the store itself"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no application code runs on the way to a no-owner answer"
    );

    assert_eq!(
        file_bytes(&db_path).as_deref(),
        Some(bytes_before.as_slice()),
        "asking for an owner that is not there must leave the committed file byte-for-byte alone"
    );
    assert_eq!(
        directory_entries(dir.path()),
        entries_before,
        "no lock, no reader breadcrumb, no sidecar: the ask leaves nothing beside the store"
    );

    // The store is free, whole, and takes an owner again — so the refusal above was about the
    // owner's absence and about nothing that happened to the file.
    let (reopen_handler, reopen_calls) = counting_handler();
    let reopened = Store::open_with_control_handler(store_config(db_path.clone()), reopen_handler)
        .expect("the store nobody held is free for this caller to open");
    assert!(
        reopened
            .list_contexts()
            .expect("read back what the earlier owner wrote")
            .iter()
            .any(|context| context.name == context_name),
        "what the earlier owner wrote is still in the store the ask left alone"
    );
    assert_eq!(
        request_owner(&db_path, "status\n").expect("the store has an owner again, and it answers"),
        "handled:status",
        "the same path that answered owner-absent answers over the owner route once one is there"
    );
    assert_eq!(
        reopen_calls.load(Ordering::SeqCst),
        1,
        "exactly the one frame sent to the live owner reached its handler"
    );
}

#[test]
fn a_path_with_no_store_at_all_answers_the_same_owner_absent_thing_and_creates_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("never-created.db");

    let entries_before = directory_entries(dir.path());
    assert!(
        !db_path.exists(),
        "the fixture is a path no store was ever written to"
    );

    let refusal = request_owner(&db_path, "status\n")
        .expect_err("there is no owner at a path that has no store");

    // The route answers before the file is looked at, so it cannot tell a store that was given up
    // from a store that never existed. Being told "there is no store here" is knowledge only a
    // look could produce, and a look is exactly what this ask must never take.
    assert!(
        matches!(&refusal, OwnerControlError::OwnerNotRunning { store_path } if store_path == &db_path),
        "an ask that never looks at the file answers owner-absent the same way whatever is (or is \
         not) on disk, got {refusal:?}"
    );
    assert!(
        refusal.owner_absent(),
        "nobody holds this path, so the caller is free to create the store itself"
    );

    assert!(
        !db_path.exists(),
        "asking who owns a path must never bring a store into existence there"
    );
    assert_eq!(
        directory_entries(dir.path()),
        entries_before,
        "the ask created nothing at all in the directory it asked about"
    );
}
