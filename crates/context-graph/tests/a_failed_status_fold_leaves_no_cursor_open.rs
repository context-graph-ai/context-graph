//! A bounded paged read releases its cursor when a fold refuses a page.
//!
//! `StatusReads::for_each_page` closes cursors on successful reads and on failures.
//! The session's `open_cursors` count returns to zero after a refusal on the first page
//! or a later page, allowing another bounded read through the same session.

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
    .expect("model-free capture of the fixture");

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
        "a successful read closes its cursor"
    );
}
