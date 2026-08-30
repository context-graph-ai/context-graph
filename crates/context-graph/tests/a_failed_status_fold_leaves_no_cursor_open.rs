//! A bounded paged read that stops early must let go of its cursor.
//!
//! ## The promise
//!
//! The bounded reading door exists so a status total on a huge store costs a page at a time and
//! nothing more: "Only one page is ever held at a time: what a status total retains is its
//! counters, not its input" (src/status_reads.rs:9). A caller is entitled to that on the paths
//! where the read STOPS as much as on the paths where it finishes — a status that failed is exactly
//! the moment a caller retries, and a retry that meets slots and snapshots left behind by the
//! attempt before it is worse than the failure was.
//!
//! ## What is observed today
//!
//! `StatusReads::for_each_page` (src/status_reads.rs:130-148) closes its cursor on ONE path: the
//! one where every page is folded successfully. `fold(&first.rows)?`, `cursor.fetch(None)?` and
//! `fold(&page.rows)?` all return through `?` with the cursor still open, and contextdb's
//! `ReadCursor` has no `Drop` — `close()` is a method a caller calls (read_session.rs:2795), which
//! is why abandoning one is not a no-op. The cursor's slot and the snapshot it pins are held until
//! the whole session goes away. `queue_counts_observed` reaches this on its ordinary error path:
//! its fold runs two keyed reads and a work-ledger read, any of which can fail, and the digest
//! scan's own cursor is abandoned when one does.
//!
//! ## What this file needs, and why it does not exist yet
//!
//! Nothing publicly reachable makes a status fold fail on demand, and nothing publicly reachable
//! says what a reading session is still holding — `StatusReads` is `pub(crate)`. So this file is
//! written against the seam that makes the promise checkable at all, and it does not compile until
//! that seam exists. The shape asked for is the smallest one that puts no test logic in `src/`:
//!
//! ```text
//! // crates/context-graph/src/status_reads.rs — the module is already `pub`
//! pub struct StatusReads<'observer> { .. }          // was pub(crate)
//! impl<'observer> StatusReads<'observer> {
//!     pub fn open(store: &Store, observer: &'observer dyn StatusReadObserver) -> Result<Self>;
//!     pub fn for_each_page(
//!         &self,
//!         statement: &str,
//!         params: &HashMap<String, contextdb_core::Value>,
//!         fold: impl FnMut(&[Vec<contextdb_core::Value>]) -> Result<()>,
//!     ) -> Result<()>;
//!
//!     /// Cursors this session opened and has not closed.
//!     ///
//!     /// contextdb's `ReadCursor` has no `Drop`, so a cursor left behind on an error path holds
//!     /// its slot and its snapshot until the session ends. This is the session's own count —
//!     /// incremented where a cursor is opened, decremented where one is closed — so it needs no
//!     /// engine test seam and stays true in an ordinary build.
//!     pub fn open_cursors(&self) -> u64;
//! }
//! ```
//!
//! A narrower alternative, if the owner would rather not widen the reading door: keep
//! `StatusReads` crate-private and add one `#[doc(hidden)]` helper that runs a paged read whose
//! fold refuses a named page and reports the same count. This file pins the wider shape because it
//! keeps the failing fold here, in the test, instead of in `src/`.

use std::collections::HashMap;

use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::status_reads::{StatusReads, UnobservedReads};
use context_graph::tier0::{Tier0CaptureRequest, capture_tier0};
use context_graph::{
    CgError, ContextId, Embedder, EmbedderRegistration, Result, Store, StoreConfig,
};
use contextdb_core::Value as DbValue;

/// Digests captured into the fixture context. More than one bounded page (`cursor_page_rows` is
/// 100), so a fold can be failed on the first page and again after a fetch.
const CAPTURED_DIGESTS: usize = 250;

const CHUNK_WINDOW: usize = 40;

/// The paged read the queue total makes over a context's digests — the read whose fold really does
/// fail in production when a keyed ledger read below it fails.
const DIGEST_SCAN: &str = "SELECT id FROM digests WHERE context_id = $context_id";

fn fixture() -> (tempfile::TempDir, Store, ContextId) {
    let directory = tempfile::tempdir().expect("a fixture directory");
    let embedder = std::sync::Arc::new(LexicalMockEmbedder::default());
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: directory.path().join("context-graph.db"),
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("open the fixture store");

    let mut transcript = String::with_capacity(CAPTURED_DIGESTS * CHUNK_WINDOW);
    for unit in 0..CAPTURED_DIGESTS {
        let mut block = format!("User: note-{unit} ");
        while block.len() < CHUNK_WINDOW {
            block.push('x');
        }
        transcript.push_str(&block);
    }
    capture_tier0(
        &store,
        Tier0CaptureRequest {
            context_label: "folds".to_string(),
            session_ref: "folds-0".to_string(),
            transcript,
            source: "test".to_string(),
            chunk_window: CHUNK_WINDOW,
        },
    )
    .expect("tier-0 capture of the fixture");

    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|context| context.name == "folds")
        .expect("the fixture context exists")
        .id;
    (directory, store, context_id)
}

fn scan_params(context_id: ContextId) -> HashMap<String, DbValue> {
    HashMap::from([(
        "context_id".to_string(),
        DbValue::Uuid(context_id.as_uuid()),
    )])
}

/// Page through the fixture's digests, refusing the page numbered `fail_on_page` the way a keyed
/// ledger read below the fold refuses in production.
fn read_refusing_page(reads: &StatusReads<'_>, context_id: ContextId, fail_on_page: usize) {
    let mut page = 0usize;
    let outcome = reads.for_each_page(
        DIGEST_SCAN,
        &scan_params(context_id),
        |rows: &[Vec<DbValue>]| {
            let this_page = page;
            page += 1;
            assert!(
                !rows.is_empty() || this_page > 0,
                "the fixture's first page must carry rows"
            );
            if this_page == fail_on_page {
                return Err(CgError::Engine(
                    "the read below this fold refused".to_string(),
                ));
            }
            Ok(())
        },
    );
    assert!(
        outcome.is_err(),
        "the fold refused page {fail_on_page}, so the read must come back refused"
    );
}

#[test]
fn a_fold_that_refuses_its_first_page_leaves_no_cursor_behind() {
    let (_directory, store, context_id) = fixture();
    let reads = StatusReads::open(&store, &UnobservedReads).expect("open the reading door");

    read_refusing_page(&reads, context_id, 0);

    assert_eq!(
        reads.open_cursors(),
        0,
        "the read stopped, so its cursor is finished with — an abandoned cursor holds its slot and \
         the snapshot it pins until the whole session goes away, and the caller retrying after a \
         failure is exactly who pays for that"
    );
}

#[test]
fn a_fold_that_refuses_a_later_page_leaves_no_cursor_behind() {
    let (_directory, store, context_id) = fixture();
    let reads = StatusReads::open(&store, &UnobservedReads).expect("open the reading door");

    // Page 1 is reached only after a real `fetch`, so this is the abandonment that happens
    // mid-stream rather than at the open.
    read_refusing_page(&reads, context_id, 1);

    assert_eq!(
        reads.open_cursors(),
        0,
        "a read abandoned mid-stream lets go of its cursor exactly as one abandoned at the first \
         page does"
    );
}

#[test]
fn a_session_that_survived_a_failed_fold_can_still_read() {
    let (_directory, store, context_id) = fixture();
    let reads = StatusReads::open(&store, &UnobservedReads).expect("open the reading door");

    read_refusing_page(&reads, context_id, 0);

    // The point of letting go is that the session is still usable. A caller whose status failed
    // and who asks again on the same session must not be paying for the attempt before it.
    let mut rows_seen = 0usize;
    let outcome: Result<()> = reads.for_each_page(
        DIGEST_SCAN,
        &scan_params(context_id),
        |rows: &[Vec<DbValue>]| {
            rows_seen += rows.len();
            Ok(())
        },
    );
    outcome.expect("a session that let go of its cursor reads again");
    assert_eq!(
        rows_seen, CAPTURED_DIGESTS,
        "the second read pages through every digest, so nothing the failed attempt left behind is \
         standing in its way"
    );
    assert_eq!(
        reads.open_cursors(),
        0,
        "a read that finished closes its cursor, which is the path that already works today"
    );
}
