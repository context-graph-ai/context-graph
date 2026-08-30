//! `--format text` is a request about EVERY answer `cg status` gives, refusals included.
//!
//! ## The promise
//!
//! `cg status --help` advertises `[--format json|text]` without qualification, so an operator who
//! asks for text has asked for text. Today the flag is honored on the success path and dropped on
//! every refusal path: `print_memory_json` (crates/cg-cli/src/main.rs:5212) is the only rendering
//! `memory_failure` (:5279) has, and its two callers `memory_error` (:5303) and `cg_failure`
//! (:5352) never see the requested format at all. So the ONE class of answer the README's "What
//! `cg status` says when the store cannot be read" section exists for — the refusals — is the one
//! class that ignores what the caller asked for.
//!
//! What that costs is not cosmetic. A refusal is the answer an operator reads at the moment their
//! memory appears to be gone, and a person who asked for prose is instead handed a single JSON
//! line whose whole payload is a Rust `Debug` dump escaped inside a string field — the store path,
//! the condition and the remedy all buried behind `\"` sequences. A text rendering that is JSON
//! wearing a text flag is not a rendering; it is the flag being ignored.
//!
//! ## What this file pins
//!
//! On the two refusal shapes an operator actually meets on a healthy machine — a store that is
//! there and cannot be read, and a store live direct readers are still holding:
//!
//! - `--format text` renders PROSE: no JSON document on stdout, no JSON field names, and no
//!   backslash-escaped quotes from a JSON string smuggled into a text answer.
//! - The prose still carries the three things the operator and their script act on: the store
//!   path, the condition name (`cg status --help` tells them to branch on it rather than on the
//!   exit code, so text must not be the one surface that withholds it), and what happened.
//! - `--format json` is unchanged — the fix is a second rendering, never the loss of the first.
//! - The exit code is identical in both formats. Format is how an answer is written, never what
//!   the answer is. No NUMBER is asserted for the reader-held shape, whose exit code is a parked
//!   owner proposal; only that the two formats agree and that a refusal fails.
//! - A refusal rendered as text still never prints `goals: 0 decisions: 0 observations: 0`, which
//!   would say the store was read and is empty.

#![cfg(unix)]

use std::fs;
use std::path::Path;

#[path = "support/store_contention.rs"]
mod store_contention;

use store_contention::{
    CgRun, HoldingChild, run_cg_below_runtime_root, runtime_root, seeded_data_root, store_path_in,
};

/// The parked direct reader this file borrows to produce the reader-held refusal.
#[test]
fn a_parked_direct_reader_child() {
    store_contention::run_parked_reader_child_if_asked();
}

/// JSON field names, as they appear in the envelope `print_memory_json` writes. None of them is
/// something a person asked for prose is meant to read.
const READS_AS_A_JSON_DOCUMENT: [&str; 6] = [
    "\"condition\":",
    "\"bucket\":",
    "\"ok\":false",
    "\"store_path\":",
    "\"summary\":",
    "\"db_path\":",
];

/// Totals are an ANSWER: they say the store was read and is empty. A refusal has none, in either
/// format.
const READS_AS_AN_EMPTY_STORE: &str = "goals: 0 decisions: 0 observations: 0";

/// Everything a refusal rendered as text must still hand the operator, whatever words are chosen
/// around it.
struct TextRefusalExpectation<'a> {
    /// What the shape is called, for the assertion messages.
    shape: &'a str,
    /// The condition name `cg status --help` tells a script to branch on.
    condition: &'a str,
    /// Fragments the prose must carry: the store path, and whatever the operator acts on.
    must_name: Vec<String>,
}

fn assert_text_refusal_is_prose(run: &CgRun, expectation: &TextRefusalExpectation) {
    let said = run.everything_said();
    let stdout = run.stdout.trim();

    assert_ne!(
        run.code, 0,
        "the {} refusal is a refusal in any format: {said}",
        expectation.shape
    );
    assert!(
        !stdout.starts_with('{'),
        "the caller asked for text and got a JSON document: `--format text` is advertised without \
         qualification on `cg status --help`, so the {} refusal must render as prose rather than \
         the same envelope the JSON format writes: {said}",
        expectation.shape
    );
    assert!(
        !stdout.contains("\\\""),
        "a text answer must not carry backslash-escaped quotes: those are a JSON string's \
         punctuation, and their presence means the {} refusal was serialized as JSON and then \
         handed to somebody who asked for prose: {said}",
        expectation.shape
    );
    for field in READS_AS_A_JSON_DOCUMENT {
        assert!(
            !stdout.contains(field),
            "`{field}` is a JSON field name; the {} refusal was asked for as text: {said}",
            expectation.shape
        );
    }
    assert!(
        said.contains(expectation.condition),
        "`cg status --help` tells a script to branch on the condition name rather than on the exit \
         code, so the text rendering of the {} refusal must still name `{}` — text must not be the \
         one surface that withholds it: {said}",
        expectation.shape,
        expectation.condition
    );
    for fragment in &expectation.must_name {
        assert!(
            said.contains(fragment.as_str()),
            "the text rendering of the {} refusal must still name `{fragment}` — the format flag \
             chooses how the answer is written, never how much of it the operator is told: {said}",
            expectation.shape
        );
    }
    assert!(
        !said.contains(READS_AS_AN_EMPTY_STORE),
        "a refused read has no totals in any format: `{READS_AS_AN_EMPTY_STORE}` says the store \
         WAS read and is empty, which the {} refusal never established: {said}",
        expectation.shape
    );
}

/// The JSON rendering is the contract this file must not cost anybody: a fix that turns the text
/// request into prose by turning the JSON request into prose too has broken every script.
fn assert_json_refusal_is_still_a_json_document(run: &CgRun, shape: &str) {
    let said = run.everything_said();
    let stdout = run.stdout.trim();
    assert_ne!(run.code, 0, "the {shape} refusal is a refusal: {said}");
    assert!(
        stdout.starts_with('{'),
        "`--format json` is what every script parsing `cg status` already reads, so the {shape} \
         refusal must still write one JSON document on stdout: {said}"
    );
    assert!(
        stdout.contains("\"condition\":"),
        "the {shape} refusal's JSON document carries the condition name a script branches on: \
         {said}"
    );
}

/// A data root whose store is THERE and cannot be read: bytes that are not a store, at the path
/// `cg` opens. This is the shape an operator meets after a bad restore or a truncated copy.
fn data_root_with_an_unreadable_store() -> tempfile::TempDir {
    let data_root = tempfile::Builder::new()
        .prefix("cgu")
        .tempdir_in("/tmp")
        .expect("a cg data root");
    fs::write(
        store_path_in(data_root.path()),
        b"this is not a context-graph store, it is forty-odd bytes of nothing\n",
    )
    .expect("write an unreadable store where cg opens one");
    data_root
}

fn status(args: &[&str], data_root: &Path, runtime_directory: &Path) -> CgRun {
    run_cg_below_runtime_root(args, data_root, runtime_directory)
}

#[test]
fn an_unreadable_store_refusal_is_rendered_in_the_format_the_caller_asked_for() {
    let runtime = runtime_root();
    let data_root = data_root_with_an_unreadable_store();
    let store_path = store_path_in(data_root.path()).display().to_string();

    let as_text = status(
        &["status", "--format", "text"],
        data_root.path(),
        runtime.path(),
    );
    let as_json = status(
        &["status", "--format", "json"],
        data_root.path(),
        runtime.path(),
    );

    assert_json_refusal_is_still_a_json_document(&as_json, "unreadable-store");
    assert_text_refusal_is_prose(
        &as_text,
        &TextRefusalExpectation {
            shape: "unreadable-store",
            condition: "MemoryError",
            must_name: vec![store_path],
        },
    );
    assert_eq!(
        as_text.code,
        as_json.code,
        "`--format` chooses how an answer is written, never what the answer IS: the unreadable \
         store refuses with the same code either way. text said {}, json said {}",
        as_text.everything_said(),
        as_json.everything_said()
    );
}

#[test]
fn a_reader_held_refusal_is_rendered_in_the_format_the_caller_asked_for() {
    let runtime = runtime_root();
    let data_root = seeded_data_root(runtime.path());
    let store_path = store_path_in(data_root.path());

    let mut holding =
        HoldingChild::park_a_reader(&store_path, runtime.path(), "a_parked_direct_reader_child");
    let parked = holding.wait_until_parked();

    let as_text = status(
        &["status", "--format", "text"],
        data_root.path(),
        runtime.path(),
    );
    let as_json = status(
        &["status", "--format", "json"],
        data_root.path(),
        runtime.path(),
    );

    assert_json_refusal_is_still_a_json_document(&as_json, "reader-held");
    assert_text_refusal_is_prose(
        &as_text,
        &TextRefusalExpectation {
            shape: "reader-held",
            condition: "StoreHeldByReaders",
            must_name: vec![
                store_path.display().to_string(),
                // The process an operator goes and finds when the wait outlasts their patience.
                format!("process {}", parked.process_id),
            ],
        },
    );
    // No number is asserted for this shape — its exit code is a parked owner proposal. That the
    // two formats AGREE is the contract: a rendering choice must not move an exit code.
    assert_eq!(
        as_text.code,
        as_json.code,
        "the format flag must not change the code a supervisor branches on. text said {}, json \
         said {}",
        as_text.everything_said(),
        as_json.everything_said()
    );

    holding.release();
}
