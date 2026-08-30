//! When a writer holds the store, the caller gets exactly what contextdb knows about the holder:
//! the store path always, and a process id ONLY when the holder published one.
//!
//! An operator told "the store is held by process 0" goes looking for process 0. An operator told
//! "the store is held, and the holder has not published which process it is" stops the right way —
//! by the path — and asks the machine. So an unknown process id is a real state that must survive
//! into what the caller reads as unknown: never invented, never rendered as zero, never truncated
//! into a narrower integer than the substrate reports.
//!
//! ## Public contract this file proves
//!
//! - `context_graph::CgError::StoreLocked { holder_pid: Option<u64>, path }` — the process id is
//!   optional because "the holder did not publish one" is a state contextdb reports
//!   (`HeldByWriterDetail { process_id: Option<u64>, store_path }`), and it is 64 bits wide because
//!   that is the width contextdb reports it in. Narrowing to `u32` cannot represent either.
//! - A second writable open beside a live writer is refused with that variant, carrying the holder
//!   this machine really found and the store it really holds — matched as a typed variant, never by
//!   substring-matching the engine's Debug text.
//! - The engine-error converter behind it is NOT part of this crate's public surface. A consumer
//!   cannot import `context_graph::store::map_engine_err`, and this file pins that it cannot.
//!
//! Where the rest of the lossless claim is proved: a published id, an absent one, and an id above
//! the 32-bit range are cases only the converter itself can be handed, and no real store on this
//! machine can produce the third. They belong to the crate's own unit tests beside the converter,
//! which reach it without widening the crate's surface to a consumer — this file must not buy that
//! coverage by asking for a public converter.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use context_graph::owner_control::ControlHandler;
use context_graph::{CgError, CreateContext, Store, StoreConfig};

fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    }
}

#[test]
fn a_second_open_beside_a_live_writer_names_the_holder_it_found() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("held.db");

    let holder = Store::open(store_config(db_path.clone())).expect("a writer takes the store");
    holder
        .create_context(CreateContext {
            name: format!("holder-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("the holder is a working writer, not just an open handle");

    // A second writable open, asking for something of its own so it cannot be answered by sharing
    // the handle already open on this path.
    let handler: ControlHandler = Arc::new(|frame: String| format!("handled:{}", frame.trim()));
    let refused = Store::open_with_control_handler(store_config(db_path.clone()), handler)
        .expect_err("a live writer already holds this store");

    assert_eq!(
        refused,
        CgError::StoreLocked {
            holder_pid: Some(u64::from(std::process::id())),
            path: db_path.clone(),
        },
        "the caller is told the store is held, by whom, and which store — the holder this machine \
         actually found, carried at the width the substrate reports it in"
    );
    drop(holder);
}

#[test]
fn the_engine_error_converter_is_not_public_api() {
    // The converter is how this crate turns substrate refusals into `CgError`. What a consumer is
    // entitled to is the typed error, not the machine that produces it: a public converter would
    // make every consumer's build depend on the substrate error vocabulary this crate exists to
    // absorb. So the name must not resolve from outside — proved by a case that must fail to
    // compile, not by reading this crate's own source.
    let trybuild = trybuild::TestCases::new();
    trybuild.compile_fail("tests/compile_fail/map_engine_err_is_not_public.rs");
}
