//! What a status total COST is reported as what it really was, never as zero.
//!
//! ## The promise
//!
//! `context_graph::tier0`'s observing totals exist so a caller can find out what asking for a
//! status actually spent — `cg status` on a developer's year of sessions is the read that gets
//! slow, and the module's own words are that what a read examined "comes from the engine's own
//! progress reports, not from this module's arithmetic" (src/status_reads.rs:16-18). A number that
//! is reported as 0 for every read is arithmetic, and it is false arithmetic: a read that handed
//! back a hundred rows looked at at least a hundred.
//!
//! ## What is observed today
//!
//! `StatusReads::examined` (src/status_reads.rs:112-114) is
//! `self.examined.try_iter().max().unwrap_or(0)` over a channel shared by every read the session
//! makes. Two things go wrong in it and this file pins both:
//!
//! - When the engine published no progress report for a read — which is the ordinary case, because
//!   reports are withheld until a read has done enough to be worth reporting — the drain finds
//!   nothing and `unwrap_or(0)` puts a 0 in front of the caller as though it were an observation.
//!   Every read of a 250-digest store reports 0 examined today, several of them while handing back
//!   100 rows.
//! - `max` is the wrong combiner across NESTED reads. `queue_counts_observed` folds a keyed
//!   `work_jobs` read and a keyed completions read INSIDE the fold of its digest scan
//!   (src/tier0.rs:527-585), all on one session and one channel. What those nested reads examined
//!   ADDS to what the total cost; taking the largest of them reports the deepest single read and
//!   throws the rest away.
//!
//! ## Why the numbers below are floors rather than exact values
//!
//! How many items the engine walks to answer a statement is the engine's business and may change
//! with its planner. What cannot change is that it never looks at fewer rows than it hands back,
//! and never at fewer than the rows that had to be considered to produce the answer. So every
//! assertion here is a floor derived from the fixture, and passing them cannot be arranged by
//! inflating a counter: an honest report satisfies them and a zeroed one cannot.

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
    .expect("tier-0 capture of the fixture");
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
