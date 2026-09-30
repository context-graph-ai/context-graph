//! A `cg` command refused because somebody else is holding the store must tell the operator what
//! is happening, in the sentence the condition already knows how to write.
//!
//! ## The promise
//!
//! Nothing is wrong with the machine when another process is reading the store: the readers finish
//! and the next attempt succeeds. The operator is entitled to be told exactly that — which store,
//! how many readers, and which processes to go and look at if the wait outlasts their patience —
//! and never to be handed a Rust value printed at them, a stack trace, or a sentence claiming the
//! engine did something unexpected.
//!
//! ## What is observed today
//!
//! `map_store_open_err` (crates/cg-cli/src/main.rs:3910-3917) keeps `StoreLocked` as a typed
//! condition and funnels EVERYTHING ELSE — the busy-with-readers condition included — into
//! `MemoryCliError::Store(format!("{other:?}"))`. So the one condition that already carries a
//! written-for-humans `Display` (count, store, each verified reader's process id and process name)
//! reaches the operator as its `Debug` dump instead. The typed writer-held answer beside it is
//! unaffected, and this file pins that it stays unaffected.
//!
//! ## What this file does NOT pin
//!
//! The EXIT CODE for a reader-held store. That is a parked owner proposal, not a settled contract,
//! so asserting a number here would invent one. What is asserted is that the command FAILS — a
//! refused command that reports success would send an agent on with no memory and no warning — and
//! everything about what the operator reads. The writer-held case does have a settled mapping
//! (`StoreLocked` → 4, main.rs:5249-5267) and is pinned at it.

#![cfg(unix)]

#[path = "support/store_contention.rs"]
mod store_contention;

use store_contention::{
    DEBUG_DUMP_FRAGMENTS, HoldingChild, ParkedReader, READS_AS_A_BROKEN_ENGINE,
    run_cg_below_runtime_root, runtime_root, seeded_data_root, store_path_in,
};

/// The parked direct reader every reader-contention test in this binary borrows.
#[test]
fn a_parked_direct_reader_child() {
    store_contention::run_parked_reader_child_if_asked();
}

/// The writer that holds the store the way a running context-graph holds it.
#[test]
fn a_writer_holder_child() {
    store_contention::run_writer_holder_child_if_asked();
}

/// The statement `context_graph` itself writes for a store held by exactly these readers. Built
/// from the identities the holding processes really published, so this file pins that the CLI
/// carries the library's own sentence through rather than pinning any wording of its own.
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
fn a_command_refused_by_a_reader_held_store_names_the_readers_it_waits_on() {
    let runtime = runtime_root();
    let data_root = seeded_data_root(runtime.path());
    let store_path = store_path_in(data_root.path());

    // Two readers, so the number the operator reads cannot be a floor a single refusal carries.
    let mut first =
        HoldingChild::park_a_reader(&store_path, runtime.path(), "a_parked_direct_reader_child");
    let mut second =
        HoldingChild::park_a_reader(&store_path, runtime.path(), "a_parked_direct_reader_child");
    let first_reader = first.wait_until_parked();
    let second_reader = second.wait_until_parked();

    // The production command waits up to a minute for readers to finish. This private-feature
    // binary instead uses the core's existing zero-allowance seam, so this test proves the
    // exhausted refusal without spending the production allowance.
    std::fs::write(runtime.path().join("exhaust-store-open-retry"), b"test")
        .expect("arm the private retry allowance override");

    let refused = run_cg_below_runtime_root(&["impact"], data_root.path(), runtime.path());
    let said = refused.everything_said();

    assert_ne!(
        refused.code, 0,
        "a command that never reached the store must not report success — an agent told `ok` \
         carries on with no memory and no warning: {said}"
    );
    for fragment in DEBUG_DUMP_FRAGMENTS {
        assert!(
            !said.contains(fragment),
            "`{fragment}` is a Rust value printed AT the operator; the condition already knows \
             how to write this in a sentence: {said}"
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

    // The whole statement, not a fragment of it. The condition writes it; this asserts the CLI
    // hands that writing on rather than re-wording it or dumping the value. The two readers may be
    // named in either order, so both orderings are accepted and nothing else is.
    let statements = [
        busy_statement(&store_path, [&first_reader, &second_reader]),
        busy_statement(&store_path, [&second_reader, &first_reader]),
    ];
    assert!(
        statements.iter().any(|statement| said.contains(statement)),
        "the operator must read the busy statement the condition itself writes — which store, how \
         many readers, and every reader by name. Expected one of {statements:?} inside: {said}"
    );

    // Named again in the open, because these three are what an operator ACTS on and a future
    // re-wording must not quietly drop any of them.
    assert!(
        said.contains(&store_path.display().to_string()),
        "the operator is told WHICH store is busy, or an operator running several has nothing to \
         act on: {said}"
    );
    for reader in [&first_reader, &second_reader] {
        assert!(
            said.contains(&format!("process {}", reader.process_id)),
            "the operator is given the process to go and find, not only told to wait — \
             process {} is missing from: {said}",
            reader.process_id
        );
        assert!(
            said.contains(&reader.process_name),
            "what that process IS travels with its number, or the operator is hunting a bare \
             pid — `{}` is missing from: {said}",
            reader.process_name
        );
    }

    first.release();
    second.release();
}

#[test]
fn a_command_refused_by_a_writer_held_store_keeps_its_typed_answer() {
    let runtime = runtime_root();
    let data_root = seeded_data_root(runtime.path());
    let store_path = store_path_in(data_root.path());

    let mut holder =
        HoldingChild::hold_as_writer(&store_path, runtime.path(), "a_writer_holder_child");
    holder.wait_until_holding_as_writer();

    let refused = run_cg_below_runtime_root(&["impact"], data_root.path(), runtime.path());
    let said = refused.everything_said();

    assert_eq!(
        refused.code, 4,
        "a store another process is writing is the settled busy answer this CLI already gives, \
         and its exit code is part of it: {said}"
    );
    assert!(
        said.contains("StoreBusy"),
        "the writer-held condition keeps the name a caller branches on: {said}"
    );
    assert!(
        said.contains("store busy / already in use"),
        "the writer-held wording is unchanged by anything the reader-held answer needs: {said}"
    );
    assert!(
        said.contains(&format!("pid {}", holder.process_id())),
        "the operator is given the process that actually holds the store: {said}"
    );
    assert!(
        said.contains(&store_path.display().to_string()),
        "the writer-held answer names the store it is about: {said}"
    );

    holder.release();
}

#[test]
#[cfg(any(
    context_graph_internal_test_support,
    feature = "event-search-test-support"
))]
fn history_import_names_reader_contention_without_claiming_writer_contention() {
    let runtime = runtime_root();
    let data_root = seeded_data_root(runtime.path());
    let store_path = store_path_in(data_root.path());
    let source = data_root.path().join("held-store.jsonl");
    std::fs::write(&source, b"{}\n").expect("one source record");
    let mut reader =
        HoldingChild::park_a_reader(&store_path, runtime.path(), "a_parked_direct_reader_child");
    let observed = reader.wait_until_parked();
    std::fs::write(runtime.path().join("exhaust-store-open-retry"), b"test")
        .expect("use the private zero retry allowance");
    let source_name = source.to_str().expect("scratch path is UTF-8");
    let refused = run_cg_below_runtime_root(
        &[
            "import",
            source_name,
            "--context",
            "reader-contention",
            "--search",
            "defer",
        ],
        data_root.path(),
        runtime.path(),
    );
    let answer = refused.everything_said();
    assert_ne!(refused.code, 0, "the refused import must fail: {answer}");
    assert_ne!(refused.code, 4, "writer contention owns exit 4: {answer}");
    assert!(answer.contains("StoreHeldByReaders"), "{answer}");
    assert!(answer.contains("store_held_by_readers"), "{answer}");
    assert!(
        answer.contains(&store_path.display().to_string()),
        "{answer}"
    );
    assert!(
        answer.contains(&format!("process {}", observed.process_id)),
        "{answer}"
    );
    assert!(!answer.contains("store_busy"), "{answer}");
    reader.release();
}
