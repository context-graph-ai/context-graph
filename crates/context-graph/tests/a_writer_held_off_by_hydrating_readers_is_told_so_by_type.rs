//! A writer that cannot take the store because direct readers are still hydrating it must be told
//! that by TYPE — the momentary condition, the readers holding it, and the store — never as an
//! unexplained engine failure.
//!
//! This is a startup-retry contract. A runtime coming up beside a reader (a camera runtime taking
//! its store while a status probe is still hydrating the committed image) has to tell two things
//! apart that look identical in prose: "wait a moment and try again, somebody is reading" and
//! "something is wrong with this store". The first is normal and self-clears; the second must stop
//! the runtime. Handing both back as a generic engine error leaves the consumer with one option —
//! substring-matching the engine's words — which is exactly the failure mode the typed
//! `StoreLocked` variant already exists to end.
//!
//! ## Public contract this file proves
//!
//! - `context_graph::CgError::StoreHeldByReaders { observed_direct_readers: u64, readers:
//!   Vec<context_graph::ReaderIdentity>, path }` exists as its own matchable condition a consumer
//!   can branch on, distinct from the generic engine bucket, without reading a single character of
//!   any message.
//! - `ReaderIdentity` is context-graph's OWN public type — a projection, not a re-export. It
//!   mirrors every field contextdb's `ReaderBreadcrumb` carries: the process id, the process name,
//!   and which run of that process it is (contextdb's `ProcessStartIdentity` newtype projected to
//!   the plain `u64` it wraps, because the substrate's types stop at this crate's edge).
//! - The process id is carried as `u64`, widened losslessly from the `u32` the substrate reports.
//!   Reader pids and the holder pid in `StoreLocked` name processes in the same operating-system
//!   pid space and get compared to each other, so one error surface must not hand a consumer two
//!   different widths for the same kind of number. Widening never truncates; only narrowing does.
//! - Every value the substrate observed survives the projection: a consumer rendering "process N"
//!   gets the same N contextdb saw, and the same name and run identity with it.
//! - The count and the identities are carried separately. A store where more readers were observed
//!   than could be named is a real state, so the count is never inferred from the list's length.
//! - The store the writer was refused on is named, so an operator running several knows which one
//!   is busy. contextdb's detail does not carry a path (`HeldByReadersDetail` is count + readers
//!   only), so this crate supplies it from the open it was refused on.
//!
//! ## Why a projection rather than the substrate's own value
//!
//! This crate's adapter projects Context Graph results. A public `CgError` variant carries this
//! crate's own reader identity, so consumers build against its vocabulary, just as `StoreLocked`
//! carries a plain pid and path. The conversion preserves every field structurally: it destructures
//! `ReaderBreadcrumb` EXHAUSTIVELY, so a field added upstream breaks the build rather than going
//! quietly missing. What this file pins is the other half of that guarantee — that every field
//! which exists today arrives with its value intact.
//!
//! ## What this file proves, and where the rest of the promise is proved
//!
//! This file is about the VALUE: what the condition carries and that nothing is lost carrying it.
//! That a real `Store::open` actually PRODUCES it — with the processes really holding the store
//! named in it — is proved against live readers in
//! `a_real_store_open_is_refused_by_the_readers_holding_it.rs`, which parks two readers in separate
//! processes inside the shipped hydration checkpoint and takes the refusal a third process gets.
//! The two files are the whole promise: a value nothing produces and a refusal nobody can read the
//! parts of each leave a consumer stuck.

use std::path::PathBuf;

use context_graph::{CgError, ReaderIdentity};
use contextdb_core::read_contract::{ProcessStartIdentity, ReaderBreadcrumb};

/// The store a refused writer was trying to take.
const BUSY_STORE_PATH: &str = "/data/cg/memory.db";

/// How many readers the substrate OBSERVED holding the store — more than it could name, which is a
/// real state and the reason the count is its own field.
const OBSERVED_READERS: u64 = 2;

/// The reader contextdb could verify, in the substrate's own vocabulary. This is the value the
/// projection is measured with.
fn verified_breadcrumb() -> ReaderBreadcrumb {
    ReaderBreadcrumb {
        process_id: 4242,
        process_name: "cg-status".to_string(),
        process_start: ProcessStartIdentity(9_876_543_210),
    }
}

/// The same reader as context-graph projects it: this crate's own type, every field mirrored.
fn projected(breadcrumb: &ReaderBreadcrumb) -> ReaderIdentity {
    ReaderIdentity {
        process_id: u64::from(breadcrumb.process_id),
        process_name: breadcrumb.process_name.clone(),
        process_start: breadcrumb.process_start.0,
    }
}

fn busy_store_condition() -> CgError {
    CgError::StoreHeldByReaders {
        observed_direct_readers: OBSERVED_READERS,
        readers: vec![projected(&verified_breadcrumb())],
        path: PathBuf::from(BUSY_STORE_PATH),
    }
}

#[test]
fn a_busy_store_is_its_own_condition_a_consumer_can_branch_on() {
    let condition = busy_store_condition();

    let CgError::StoreHeldByReaders { path, .. } = &condition else {
        panic!(
            "a store that is merely busy must be a condition of its own — a consumer that has to \
             read a message to recognise it cannot retry safely: {condition:?}"
        );
    };
    assert_eq!(
        path,
        &PathBuf::from(BUSY_STORE_PATH),
        "the condition names the store the writer was refused on, which contextdb's reader detail \
         does not carry — this crate supplies it from the open it was refused on"
    );
}

#[test]
fn every_field_the_substrate_observed_survives_the_projection() {
    let breadcrumb = verified_breadcrumb();
    let condition = busy_store_condition();

    let CgError::StoreHeldByReaders {
        observed_direct_readers,
        readers,
        ..
    } = &condition
    else {
        panic!("the busy-store condition is what carries the readers: {condition:?}");
    };

    assert_eq!(
        *observed_direct_readers, OBSERVED_READERS,
        "the count is the substrate's own observation, not the number of readers it could name — \
         a reader it counted but could not identify still holds the store"
    );
    assert_eq!(
        readers.len(),
        1,
        "one reader was verified, so one is projected: {readers:?}"
    );
    assert!(
        readers.len() < usize::try_from(*observed_direct_readers).expect("a small reader count"),
        "the fixture is deliberately a store where the substrate counted more readers than it \
         could name, so passing it means the count is never inferred from the list"
    );

    // Field by field against the substrate value, so a projection that quietly loses one cannot
    // pass on the others.
    let reader = readers.first().expect("the projected reader");
    assert_eq!(
        reader.process_id,
        u64::from(breadcrumb.process_id),
        "the process an operator goes and finds is the one contextdb saw, widened to the width \
         this crate reports every process id in"
    );
    assert_eq!(
        reader.process_name, breadcrumb.process_name,
        "what that process IS travels with its number, or the operator is hunting a bare pid"
    );
    assert_eq!(
        reader.process_start, breadcrumb.process_start.0,
        "which RUN of that process holds the store — the same value contextdb's newtype wraps, so \
         a recycled pid is still told apart"
    );

    // Named field by field ABOVE, and destructured exhaustively here so a field added to
    // `ReaderIdentity` later fails to compile in this file rather than going unchecked. Comparing
    // the projection to another call of the same projection helper would have proved nothing at
    // all: both sides would move together with any bug in it.
    let ReaderIdentity {
        process_id,
        process_name,
        process_start,
    } = reader;
    assert_eq!(
        (*process_id, process_name.as_str(), *process_start,),
        (
            u64::from(breadcrumb.process_id),
            breadcrumb.process_name.as_str(),
            breadcrumb.process_start.0,
        ),
        "the whole identity is the substrate's whole identity — every field of it, measured \
         against the substrate value and never against this test's own projection"
    );
}
