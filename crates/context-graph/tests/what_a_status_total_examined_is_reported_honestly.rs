//! What a status total COST is reported as what it really was, never as zero.
//!
//! Observed status totals account for the work each read performs, including nested reads.
//! A read cannot examine fewer rows than it returns, and a total includes the work of every
//! contributing read rather than only the largest individual observation.
//!
//! The fixture captures 250 digests so scans span several bounded pages. Queue totals include
//! keyed work-ledger and completion reads within the digest scan; raw footprint totals include
//! every captured chunk. Their observations must account for those rows.
//!
//! Exact engine work depends on the query planner. Assertions therefore use floors derived
//! from the fixture rather than pinning an exact number of examined rows.

use std::sync::Mutex;

use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::tier0::{
    ReadObservation, StatusReadObserver, Tier0CaptureRequest, capture_tier0, queue_counts_observed,
    raw_footprint_observed,
};
use context_graph::{ContextId, Embedder, EmbedderRegistration, Store, StoreConfig};

/// Digests captured into the fixture context. More than one bounded page (`cursor_page_rows` is
/// 100), so the digest scan really pages and the nested reads really run more than once.
const CAPTURED_DIGESTS: usize = 250;

/// Bytes each captured digest covers.
const CHUNK_WINDOW: usize = 40;

/// Everything a status total's reads reported, in the order they were made.
#[derive(Default)]
struct RecordedReads {
    reads: Mutex<Vec<ReadObservation>>,
}

impl RecordedReads {
    fn taken(&self) -> Vec<ReadObservation> {
        self.reads.lock().expect("recorded reads").clone()
    }
}

impl StatusReadObserver for RecordedReads {
    fn read(&self, observation: ReadObservation) {
        self.reads.lock().expect("recorded reads").push(observation);
    }
}

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
    let report = capture_tier0(
        &store,
        Tier0CaptureRequest {
            context_label: "examined".to_string(),
            session_ref: "examined-0".to_string(),
            transcript,
            source: "test".to_string(),
            chunk_window: CHUNK_WINDOW,
        },
    )
    .expect("model-free capture of the fixture");
    assert_eq!(
        report.digest_ids.len(),
        CAPTURED_DIGESTS,
        "the fixture must hold every digest it counts on"
    );

    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|context| context.name == "examined")
        .expect("the fixture context exists")
        .id;
    (directory, store, context_id)
}

/// A read cannot hand back rows it never looked at. This is the floor under every observation,
/// whatever the engine's planner does.
fn assert_no_read_returns_more_than_it_examined(reads: &[ReadObservation], what: &str) {
    assert!(
        !reads.is_empty(),
        "{what} recorded no bounded read at all — it did not go through the reading door"
    );
    for read in reads {
        assert!(
            read.rows_examined >= read.rows_returned,
            "{what} reported a read that handed back {} rows after examining {} — a read cannot \
             return rows it never looked at, and a caller reading this cannot tell a cheap status \
             from an expensive one: {}",
            read.rows_returned,
            read.rows_examined,
            read.statement
        );
    }
}

#[test]
fn the_queue_total_reports_what_its_nested_reads_examined_rather_than_zero() {
    let (_directory, store, context_id) = fixture();

    let recorded = RecordedReads::default();
    let counts =
        queue_counts_observed(&store, context_id, &recorded).expect("the queue total answers");
    assert_eq!(
        counts.pending, CAPTURED_DIGESTS,
        "the fixture's digests are all queued and untouched, so the total itself is correct — \
         what this file is about is the COST it reported making it: {counts:?}"
    );

    let reads = recorded.taken();
    assert_no_read_returns_more_than_it_examined(&reads, "the queue total");

    let examined: u64 = reads.iter().map(|read| read.rows_examined).sum();
    let deepest = reads
        .iter()
        .map(|read| read.rows_examined)
        .max()
        .expect("at least one read");

    let captured = u64::try_from(CAPTURED_DIGESTS).expect("the fixture size fits");
    assert!(
        examined >= captured,
        "the digest scan alone paged through {captured} rows, so the total cannot have examined \
         fewer than that — it reported {examined} across {} reads",
        reads.len()
    );
    assert!(
        examined > deepest,
        "the keyed work-ledger and completion reads run INSIDE the digest scan's fold, on the same \
         session; what each of them examined ADDS to what this total cost. Reporting the deepest \
         single read ({deepest}) as though it were the whole cost throws the rest away: total \
         reported {examined} across {} reads",
        reads.len()
    );
}

#[test]
fn the_raw_footprint_total_reports_the_rows_its_aggregate_walked() {
    let (_directory, store, context_id) = fixture();

    let recorded = RecordedReads::default();
    let footprint =
        raw_footprint_observed(&store, context_id, &recorded).expect("the raw footprint answers");
    assert_eq!(
        footprint.chunk_count, CAPTURED_DIGESTS,
        "the fixture keeps one raw chunk per digest, so the footprint itself is correct — what \
         this file is about is the COST it reported making it: {footprint:?}"
    );

    let reads = recorded.taken();
    assert_no_read_returns_more_than_it_examined(&reads, "the raw footprint");

    let examined: u64 = reads.iter().map(|read| read.rows_examined).sum();
    let captured = u64::try_from(CAPTURED_DIGESTS).expect("the fixture size fits");
    assert!(
        examined >= captured,
        "counting {captured} chunk rows means walking {captured} chunk rows; an aggregate that \
         hands back one row after reportedly examining {examined} tells a caller a whole-table \
         count is free"
    );
}
