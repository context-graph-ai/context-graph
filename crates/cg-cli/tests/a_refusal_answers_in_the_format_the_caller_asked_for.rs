//! `--format text` is a request about EVERY answer `cg status` gives, refusals included.
//!
//! ## The promise
//!
//! `cg status --help` advertises `[--format json|text]` without qualification. Both success and
//! refusal paths honor the requested format.
//!
//! A refusal is the answer an operator reads when their memory appears to be gone. A prose
//! request must produce readable text carrying the store path, condition, and remedy directly,
//! without burying them in an escaped JSON string or a Rust `Debug` dump.
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
//! - `--format json` renders the machine-readable refusal envelope.
//! - The exit code is identical in both formats. Format is how an answer is written, never what
//!   the answer is. No dedicated numeric code is contracted for the reader-held shape; the two
//!   formats agree and a refusal fails.
//! - A refusal rendered as text still never prints `goals: 0 decisions: 0 observations: 0`, which
//!   would say the store was read and is empty.

#![cfg(unix)]

use std::fs;
use std::path::Path;

#[path = "support/store_contention.rs"]
mod store_contention;

use store_contention::{CgRun, run_cg_below_runtime_root, runtime_root, store_path_in};

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

/// The JSON format renders a JSON document for scripts, independently of the prose rendering.
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
