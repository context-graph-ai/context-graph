//! A store that is merely BUSY must never read to an operator as a store that is BROKEN.
//!
//! "unexpected engine error: ..." is what a consumer sees today when direct readers are still
//! hydrating a store a writer wants. That sentence tells an operator that something went wrong
//! inside the engine, on a machine where nothing has gone wrong at all: a reader is reading, and
//! the writer's own next attempt will succeed. The cost is real — a runtime that stops on it never
//! comes up, and an operator who reads it goes looking for damage to repair.
//!
//! ## Public contract this file proves
//!
//! - The condition renders as what it is: readers are holding the store. The word "unexpected"
//!   never appears, and neither does the generic engine-error wrapper it falls into today.
//! - What contextdb verified about the readers survives into what the operator reads: a process a
//!   reader was named by is in the rendered condition, so the operator can go and find it instead
//!   of only being told to wait.
//! - The store is named, so an operator running several knows which one is busy.
//!
//! The readers the condition carries are `context_graph::ReaderIdentity` — this crate's own
//! projection of what contextdb verified, not the substrate's `ReaderBreadcrumb` re-exported. That
//! is what the adapter is for, and it is why the sentence an operator reads can stay this crate's
//! to word.
//!
//! This file pins what a human reads. What the condition IS as a matchable value is pinned in
//! `a_writer_held_off_by_hydrating_readers_is_told_so_by_type.rs`; the two together are the whole
//! promise, because a typed variant nobody can read and a readable sentence nobody can match on
//! each leave half the consumer stuck.
//!
//! The condition is built here rather than provoked from a live store on purpose, and the same
//! reason applies as in the companion file: no public context-graph call reaches reader contention
//! on demand, because it exists only while a committed image is being loaded. What a value RENDERS
//! as does not depend on how it was produced, so building it proves the wording exactly as well —
//! and that a real refusal becomes this value is proved beside the converter, in the crate's own
//! unit tests.

use std::path::PathBuf;

use context_graph::{CgError, ReaderIdentity};

/// Wording that tells an operator the engine failed. None of it belongs on a store that is simply
/// being read by somebody else.
const READS_AS_A_BROKEN_ENGINE: [&str; 3] = [
    // The exact sentence a busy store produces today.
    "unexpected engine error",
    // Anything the engine calls unexpected: a reader reading is expected.
    "unexpected",
    // The generic catch-all this condition must stop falling into.
    "engine(",
];

/// The store a refused writer was trying to take.
const BUSY_STORE_PATH: &str = "/data/cg/memory.db";

/// The process contextdb named as holding the store. Carried as `u64`: the substrate reports a
/// `u32`, and this crate widens it losslessly so every process id on this error surface is one
/// width.
const HOLDING_READER_PID: u64 = 4242;

fn busy_store_condition() -> CgError {
    CgError::StoreHeldByReaders {
        observed_direct_readers: 2,
        readers: vec![ReaderIdentity {
            process_id: HOLDING_READER_PID,
            process_name: "cg-status".to_string(),
            process_start: 9_876_543_210,
        }],
        path: PathBuf::from(BUSY_STORE_PATH),
    }
}

#[test]
fn a_store_busy_with_readers_never_reads_as_an_engine_that_went_wrong() {
    let condition = busy_store_condition();
    let rendered = condition.to_string().to_lowercase();

    for wording in READS_AS_A_BROKEN_ENGINE {
        assert!(
            !rendered.contains(wording),
            "nothing went wrong here — a reader is reading — so the operator must not read \
             `{wording}`: got `{rendered}`"
        );
    }
    assert!(
        rendered.contains("reader"),
        "the operator is told what is actually true: readers are holding the store — got \
         `{rendered}`"
    );
    assert!(
        rendered.contains(&BUSY_STORE_PATH.to_lowercase()),
        "an operator running several stores is told WHICH one is busy — got `{rendered}`"
    );
}

#[test]
fn the_reader_the_substrate_verified_is_the_reader_the_operator_reads() {
    let rendered = busy_store_condition().to_string();

    assert!(
        rendered.contains(&HOLDING_READER_PID.to_string()),
        "the substrate verified process {HOLDING_READER_PID} holds this store, so that is the \
         process the operator is given to go and find — got `{rendered}`"
    );
    assert!(
        rendered.contains("cg-status"),
        "an operator hunting a process wants its name as well as its number — got `{rendered}`"
    );
}
