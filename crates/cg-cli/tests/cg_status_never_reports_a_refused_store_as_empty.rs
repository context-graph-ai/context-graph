//! `cg status` must never dress a store it could not read as a store with nothing in it.
//!
//! ## The promise
//!
//! `cg status` is the command a developer and an agent both run to find out whether their memory is
//! there. "goals: 0 decisions: 0 observations: 0" is an ANSWER — it says the store was read and it
//! is empty. When the store was never read at all, printing that answer is not a missing feature,
//! it is a false statement: it tells someone whose memory is intact that their memory is gone, and
//! it tells an agent deciding whether to capture that there is nothing to preserve.
//!
//! ## What is observed today
//!
//! `memory_status_snapshot` (crates/cg-cli/src/main.rs:3955-4010) enters its body only under
//! `if let Ok(store) = open_memory_store(root)`, so EVERY open failure — a store another process is
//! writing, a store direct readers are still hydrating, an unreadable file — leaves the defaults in
//! place and is rendered as an empty store. The inner reads do the same thing again with
//! `unwrap_or_default()` (:3990-3991). There is no path by which a failure reaches the operator.
//!
//! ## The exit code
//!
//! A store another process is WRITING already has a settled answer in this CLI: `StoreLocked` is
//! rendered as `StoreBusy` and exits 4 (main.rs:5249-5267), on every other command that opens the
//! store. `cg status` is pinned at that same answer — this file asks for no new vocabulary and no
//! new code. For a store held by READERS the exit code is a parked owner proposal, so that case is
//! pinned on content and on failing rather than on a number.
//!
//! ## The third case, which cannot be pinned from out here
//!
//! A read that fails AFTER the store opened (`counts`, `queue`, `raw`, each `unwrap_or_default()`)
//! is the same defect one level in, and nothing reachable from a black-box `cg` invocation makes
//! one fail on demand. Pinning it needs a fault-injection seam in the binary; the shape is in the
//! window report rather than invented here.

#![cfg(unix)]

#[path = "support/store_contention.rs"]
mod store_contention;

use store_contention::{
    DEBUG_DUMP_FRAGMENTS, HoldingChild, ParkedReader, READS_AS_A_BROKEN_ENGINE,
    run_cg_below_runtime_root, runtime_root, seeded_data_root, store_path_in,
};

/// The parked direct reader this binary's reader-contention test borrows.
#[test]
fn a_parked_direct_reader_child() {
    store_contention::run_parked_reader_child_if_asked();
}

/// The writer that holds the store the way a running context-graph holds it.
#[test]
fn a_writer_holder_child() {
    store_contention::run_writer_holder_child_if_asked();
}

/// The counts an unread store must never be reported as having. These are the exact renderings
/// `cg status` produces from `HarvestCounts::default()` — the answer "your memory is empty".
const READS_AS_AN_EMPTY_STORE: [&str; 4] = [
    "goals: 0 decisions: 0 observations: 0",
    "\"goals\":0",
    "\"decisions\":0",
    "\"observations\":0",
];

fn assert_never_answers_as_an_empty_store(said: &str) {
    for rendering in READS_AS_AN_EMPTY_STORE {
        assert!(
            !said.contains(rendering),
            "`{rendering}` is an ANSWER — it says the store was read and is empty. This store was \
             never read, so saying it tells a developer whose memory is intact that it is gone: \
             {said}"
        );
    }
}

/// The statement `context_graph` itself writes for a store held by exactly these readers.
fn busy_statement(store_path: &std::path::Path, readers: [&ParkedReader; 2]) -> String {
    context_graph::CgError::StoreHeldByReaders {
        observed_direct_readers: 2,
        readers: readers
            .iter()
            .map(|reader| context_graph::ReaderIdentity {
                process_id: u64::from(reader.process_id),
                process_name: reader.process_name.clone(),
                process_start: reader.process_start,
            })
            .collect(),
        path: store_path.to_path_buf(),
    }
    .to_string()
}

#[test]
fn status_on_a_store_another_process_is_writing_gives_this_cli_s_settled_busy_answer() {
    let runtime = runtime_root();
    let data_root = seeded_data_root(runtime.path());
    let store_path = store_path_in(data_root.path());

    let mut holder =
        HoldingChild::hold_as_writer(&store_path, runtime.path(), "a_writer_holder_child");
    holder.wait_until_holding_as_writer();

    let status = run_cg_below_runtime_root(&["status"], data_root.path(), runtime.path());
    let said = status.everything_said();

    assert_never_answers_as_an_empty_store(&said);
    assert_eq!(
        status.code, 4,
        "a store another process is writing is refused with the same code every other command \
         gives for it; a status that exits 0 tells a supervisor the memory is healthy: {said}"
    );
    assert!(
        said.contains("store busy / already in use"),
        "`cg status` gives the same typed answer for a writer-held store as every other command \
         that opens it — no second vocabulary: {said}"
    );
    assert!(
        said.contains(&format!("pid {}", holder.process_id())),
        "the operator is given the process that actually holds the store: {said}"
    );
    assert!(
        said.contains(&store_path.display().to_string()),
        "the answer names the store it is about: {said}"
    );

    holder.release();
}

#[test]
fn status_on_a_store_readers_are_holding_says_so_instead_of_reporting_it_empty() {
    let runtime = runtime_root();
    let data_root = seeded_data_root(runtime.path());
    let store_path = store_path_in(data_root.path());

    let mut first =
        HoldingChild::park_a_reader(&store_path, runtime.path(), "a_parked_direct_reader_child");
    let mut second =
        HoldingChild::park_a_reader(&store_path, runtime.path(), "a_parked_direct_reader_child");
    let first_reader = first.wait_until_parked();
    let second_reader = second.wait_until_parked();

    let status = run_cg_below_runtime_root(&["status"], data_root.path(), runtime.path());
    let said = status.everything_said();

    assert_never_answers_as_an_empty_store(&said);
    assert_ne!(
        status.code, 0,
        "a status that never read the store must not exit as a success: the exit code is what a \
         supervisor and a shell script act on: {said}"
    );
    for fragment in DEBUG_DUMP_FRAGMENTS {
        assert!(
            !said.contains(fragment),
            "`{fragment}` is a Rust value printed AT the operator: {said}"
        );
    }
    let lowered = said.to_lowercase();
    for wording in READS_AS_A_BROKEN_ENGINE {
        assert!(
            !lowered.contains(wording),
            "nothing went wrong here — a reader is reading — so the operator must not read \
             `{wording}`: {said}"
        );
    }
    let statements = [
        busy_statement(&store_path, [&first_reader, &second_reader]),
        busy_statement(&store_path, [&second_reader, &first_reader]),
    ];
    assert!(
        statements.iter().any(|statement| said.contains(statement)),
        "the operator must read the busy statement the condition itself writes — which store, how \
         many readers, and every reader by name. Expected one of {statements:?} inside: {said}"
    );

    first.release();
    second.release();
}

#[test]
fn status_on_a_store_nobody_is_holding_still_answers_with_its_totals() {
    let runtime = runtime_root();
    let data_root = seeded_data_root(runtime.path());

    let status = run_cg_below_runtime_root(&["status"], data_root.path(), runtime.path());
    let said = status.everything_said();

    assert!(
        said.contains("goals: 0 decisions: 0 observations: 0"),
        "an EMPTY store really is empty, and honesty about refusals must not turn a readable \
         store into a refusal: {said}"
    );
    assert!(
        !said.contains("store busy"),
        "nobody is holding this store, so nothing may say it is busy: {said}"
    );
}
