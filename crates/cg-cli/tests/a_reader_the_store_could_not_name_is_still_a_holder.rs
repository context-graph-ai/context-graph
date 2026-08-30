//! A reader the substrate COUNTED but could not NAME is still holding the store, and the operator
//! must be told it is there.
//!
//! ## The promise
//!
//! `CgError::StoreHeldByReaders` carries the count and the identities apart on purpose: the count
//! is what contextdb OBSERVED, and `readers` is the smaller set it could VERIFY and name. That is
//! not an edge case invented for a test — it is the documented shape of the variant
//! (`crates/context-graph/src/error.rs:334-348`) and the reason the count is never inferred from
//! the list. So "two readers are holding this store, and here is one of them" is a real answer the
//! shipped rendering has to give.
//!
//! Today it gives half of it. `describe_readers` (`crates/context-graph/src/error.rs:446-455`)
//! walks the verified list and stops, so a store held by two readers of which one published no
//! identity renders as "2 direct readers are still reading it … Verified process 4242
//! (cg-status)". The second holder — a live process, on this machine, holding this store — leaves
//! no trace in the sentence. An operator who kills the one process they were given and finds the
//! store still busy has been sent after the wrong thing, and three surfaces told them that could
//! not happen: `cg status --help`, the crate README's refusal section, and `AGENTS.md` all promise
//! "every reader by process id and name" without qualification.
//!
//! The owner's ruling on this is that the promise is "every VERIFIED reader", and that a reader
//! observed but not named is a valid state that must be DISCLOSED rather than dropped — the same
//! move `MemoryCliError::StoreLocked` already makes for a writer that published no pid, which
//! renders as "another process (which has not published its pid)" rather than as "pid 0" or as
//! silence.
//!
//! ## What this file pins
//!
//! - The two-observed-one-named rendering presents BOTH holders: the verified one by process id
//!   and name, and the unnamed one through a generic-holder fallback.
//! - The fallback is conditional, not boilerplate: a store whose readers were all named carries no
//!   generic holder at all.
//! - A store where NONE of the observed readers could be named still presents them as holders.
//! - None of the three surfaces claims every reader is named; each says what is actually promised.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use context_graph::{CgError, ReaderIdentity};

/// The store a refused writer was trying to take.
const BUSY_STORE_PATH: &str = "/data/cg/memory.db";

/// Ways a rendering can tell an operator that a holder is there and could not be named. The
/// wording is the fixer's to choose; that the disclosure EXISTS is the contract. The first entry
/// is the vocabulary this CLI already uses for a writer that published no pid.
const DISCLOSES_A_HOLDER_IT_COULD_NOT_NAME: [&str; 7] = [
    "has not published",
    "have not published",
    "did not publish",
    "could not be identified",
    "could not identify",
    "not identified",
    "unidentified",
];

/// A claim that every holder appears by name. It is not true of a store with an observed-but-
/// unnamed reader, and it is the claim that sends an operator after the wrong process.
const CLAIMS_EVERY_READER_IS_NAMED: [&str; 2] = ["every reader", "each reader"];

/// What the promise actually is, once qualified.
const NAMES_ONLY_THE_VERIFIED_READERS: &str = "verified reader";

fn reader(process_id: u64, process_name: &str) -> ReaderIdentity {
    ReaderIdentity {
        process_id,
        process_name: process_name.to_string(),
        process_start: 9_876_543_210,
    }
}

fn held_by(observed_direct_readers: u64, readers: Vec<ReaderIdentity>) -> String {
    CgError::StoreHeldByReaders {
        observed_direct_readers,
        readers,
        path: PathBuf::from(BUSY_STORE_PATH),
    }
    .to_string()
}

fn discloses_an_unnamed_holder(rendered: &str) -> bool {
    let lowered = rendered.to_lowercase();
    DISCLOSES_A_HOLDER_IT_COULD_NOT_NAME
        .iter()
        .any(|wording| lowered.contains(wording))
}

#[test]
fn a_store_held_by_two_readers_of_which_one_is_named_presents_both_holders() {
    let rendered = held_by(2, vec![reader(4242, "cg-status")]);

    assert!(
        rendered.contains("4242") && rendered.contains("cg-status"),
        "the reader the substrate DID verify is still handed over by process id and name — the \
         disclosure of the other holder must not cost the operator the one they can act on: \
         `{rendered}`"
    );
    assert!(
        discloses_an_unnamed_holder(&rendered),
        "two readers are holding this store and only one published an identity. The other one is a \
         live process holding the store right now, so the sentence must present it as a holder \
         rather than stopping at the verified list — an operator who kills process 4242 and finds \
         the store still busy was sent after the wrong thing. Expected one of \
         {DISCLOSES_A_HOLDER_IT_COULD_NOT_NAME:?} in: `{rendered}`"
    );
}

#[test]
fn a_store_whose_readers_were_all_named_carries_no_generic_holder() {
    let rendered = held_by(
        2,
        vec![reader(4242, "cg-status"), reader(4243, "cg-worker")],
    );

    assert!(
        rendered.contains("4242") && rendered.contains("4243"),
        "both verified readers are named: `{rendered}`"
    );
    assert!(
        !discloses_an_unnamed_holder(&rendered),
        "every holder of this store published an identity, so nothing may suggest one of them is \
         anonymous — the fallback is what an unnamed holder gets, never boilerplate on every \
         refusal: `{rendered}`"
    );
}

#[test]
fn a_store_whose_readers_could_none_of_them_be_named_still_presents_them_as_holders() {
    let rendered = held_by(2, Vec::new());

    assert!(
        discloses_an_unnamed_holder(&rendered),
        "the substrate observed two readers and could name neither. They are still holding the \
         store, and the operator is entitled to be told that rather than left with a sentence that \
         trails off after the count. Expected one of \
         {DISCLOSES_A_HOLDER_IT_COULD_NOT_NAME:?} in: `{rendered}`"
    );
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate sits two levels below the repository root")
        .to_path_buf()
}

fn assert_promises_only_the_verified_readers(surface: &str, said: &str) {
    let lowered = said.to_lowercase();
    for claim in CLAIMS_EVERY_READER_IS_NAMED {
        assert!(
            !lowered.contains(claim),
            "{surface} promises `{claim}` by process id and name. That is not true of a store with \
             a reader contextdb counted and could not identify, and it is the promise that sends \
             an operator after the wrong process: {said}"
        );
    }
    assert!(
        lowered.contains(NAMES_ONLY_THE_VERIFIED_READERS),
        "{surface} must say what is actually promised — every VERIFIED reader — so a reader that \
         was observed and not named reads as the documented state it is rather than as a rendering \
         that lost somebody: {said}"
    );
}

/// The runtime surface: what `cg status --help` tells an operator to expect.
#[test]
fn cg_status_help_promises_only_the_readers_the_store_could_verify() {
    let home = tempfile::tempdir().expect("a throwaway home for a help-only invocation");
    let output = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_cg")))
        .args(["status", "--help"])
        .env("HOME", home.path())
        .env("CG_DATA_ROOT", home.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("run the compiled cg binary");
    assert_eq!(
        output.status.code(),
        Some(0),
        "asking for help is not a failure"
    );
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_promises_only_the_verified_readers("`cg status --help`", &said);
}

/// The operator surface: the crate README's "What `cg status` says when the store cannot be read".
#[test]
fn the_crate_readme_promises_only_the_readers_the_store_could_verify() {
    let path = repository_root().join("crates/context-graph/README.md");
    let said = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert_promises_only_the_verified_readers("`crates/context-graph/README.md`", &said);
}

/// The contributor surface: the routing file a contributor is handed before they touch the
/// condition.
#[test]
fn the_agents_routing_file_promises_only_the_readers_the_store_could_verify() {
    let path = repository_root().join("AGENTS.md");
    let said = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert_promises_only_the_verified_readers("`AGENTS.md`", &said);
}
