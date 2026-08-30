//! `cg status`'s two totals stay correct on a store far larger than one read may examine.
//!
//! Public contract: the understanding-queue counts and the raw-source footprint are computed
//! through the bounded reading door — read a page at a time, or totalled by an aggregate paged to
//! completion — and never by pulling a whole table into memory. On a context holding more than
//! 62,000 chunk rows an uncapped scan cannot answer at all: it crosses the shipped `work` (50,000)
//! and `result_rows` (500) ceilings. So a status that answers CORRECTLY here, while no single read
//! it made crossed the ceilings context-graph declares, is a status that pages. Both halves are
//! asserted — a correct total reached by an unbounded scan is the defect this file exists to catch,
//! and it is invisible from the totals alone.
//!
//! The queue counts are asserted against thousands of jobs in every state, not a handful: per-state
//! counting that quietly stopped at the first page, or that answered from one budget's worth of
//! rows, is only visible once each state is deeper than a read budget. A digest carrying both a
//! ledger result and a write-once completion is one digest in one state — structured — and is
//! counted once.
//!
//! What a total costs is observed rather than inferred: `context_graph::tier0`'s
//! `queue_counts_observed` and `raw_footprint_observed` take a `StatusReadObserver` and tell it
//! about every bounded read they make, on the same code path the plain totals take, and what each
//! read examined is the engine's own progress report rather than the module's arithmetic.

#[path = "common/large_status_fixture.rs"]
mod large_status_fixture;

use std::collections::HashMap;
use std::sync::Mutex;

use context_graph::owner_control::{context_graph_read_limits, context_graph_read_session_options};
use context_graph::tier0::{
    QueueCounts, RawFootprint, ReadObservation, StatusReadObserver, queue_counts,
    queue_counts_observed, raw_footprint, raw_footprint_observed,
};
use contextdb_engine::read_session::{OwnerAdmissionReport, ReadCursor, ReadSession};

use large_status_fixture::{
    CROWD_CANCELLED, CROWD_CHUNKS_PER_CONTEXT, CROWD_CLAIMED, CROWD_CONTEXTS, CROWD_FAILED,
    CROWD_RESULTED, UNRELATED_JOBS, crowd_the_ledger, fill, force_state_spread, open_store,
};

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

/// The tables that grow with capture. A read of one of these is the read that would become a
/// whole-table scan on a real store, so what it may hand back is held to a single page.
const CAPTURE_SIZED_TABLES: [&str; 2] = ["DIGESTS", "CG_SOURCE_CHUNKS"];

fn names_a_capture_sized_table(statement: &str) -> bool {
    let statement = statement.to_uppercase();
    CAPTURE_SIZED_TABLES
        .iter()
        .any(|table| statement.contains(table))
}

/// Every read the status total made stayed inside the ceilings this crate declares, and together
/// they accounted for every row — which is only possible by paging.
fn assert_bounded(observations: &[ReadObservation], rows_in_table: usize, what: &str) {
    let limits = context_graph_read_limits();
    assert!(
        !observations.is_empty(),
        "{what} recorded no bounded read at all — it did not go through the reading door"
    );
    for observation in observations {
        assert!(
            observation.rows_examined <= limits.work,
            "{what} let one read examine {} rows against a declared `work` ceiling of {}: {}",
            observation.rows_examined,
            limits.work,
            observation.statement
        );
        assert!(
            observation.rows_returned <= limits.result_rows,
            "{what} let one read hand back {} rows against a declared `result_rows` ceiling of {}: {}",
            observation.rows_returned,
            limits.result_rows,
            observation.statement
        );
        if names_a_capture_sized_table(&observation.statement) {
            assert!(
                observation.rows_returned <= limits.cursor_page_rows,
                "{what} took {} rows out of a table that grows with capture in ONE read, past the {}-row page the ceilings allow: {}",
                observation.rows_returned,
                limits.cursor_page_rows,
                observation.statement
            );
        }
    }
    let examined: u64 = observations
        .iter()
        .map(|observation| observation.rows_examined)
        .sum();
    let returned: u64 = observations
        .iter()
        .map(|observation| observation.rows_returned)
        .sum();
    let rows = u64::try_from(rows_in_table).expect("a row count fits");
    // contextdb's read-progress stream reports only once a single read passes one reporting
    // interval (1,000 examined items or 500 assembled rows), and it never flushes a partial
    // report at completion. So a read paged at 100 rows examines everything while reporting
    // an examined count that stays under a full interval the whole way — the summed
    // `rows_examined` can undercount even though the walk covered the table. The summed
    // `rows_returned` cannot lie the same way: every row the engine handed back is counted, so
    // that total reaching the table's row count is the engine's own proof the whole table was
    // paged. Either total reaching the row count is proof of completeness.
    assert!(
        examined >= rows || returned >= rows,
        "{what} examined only {examined} and returned only {returned} of {rows} rows, so its total cannot be complete"
    );
    assert!(
        observations.len() >= 2,
        "{what} answered a {rows}-row table in one read, which the declared `work` ceiling of {} forbids",
        limits.work
    );
}

/// A total over a table that grows with capture is taken by an aggregate: at this store's size,
/// handing those rows back is the whole-table scan the ceilings forbid.
fn assert_totalled_by_aggregate(observations: &[ReadObservation], what: &str) {
    for observation in observations {
        let statement = observation.statement.to_uppercase();
        assert!(
            statement.contains("COUNT(") || statement.contains("SUM("),
            "{what} must total a capture-sized table through an aggregate, not by reading its rows back: {}",
            observation.statement
        );
    }
}

#[test]
fn status_totals_stay_correct_and_bounded_on_a_store_no_single_read_can_cross() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store(&dir.path().join("cg.db"));
    let fixture = fill(&store);

    // The raw footprint, before anything is purged: every kept chunk, and every kept byte.
    let footprint_reads = RecordedReads::default();
    let footprint = raw_footprint_observed(&store, fixture.context_id, &footprint_reads)
        .expect("raw footprint");
    assert_eq!(
        footprint,
        RawFootprint {
            chunk_count: fixture.chunk_count,
            bytes: fixture.raw_bytes,
        },
        "the raw footprint must be the arithmetic total of what capture was handed"
    );
    let footprint_reads = footprint_reads.taken();
    assert_bounded(&footprint_reads, fixture.chunk_count, "the raw footprint");
    assert_totalled_by_aggregate(&footprint_reads, "the raw footprint");

    // A queue in every state, thousands of jobs deep in each: understood, in flight, terminally
    // failed, called off, and the rest still pending.
    let forced = force_state_spread(&store, &fixture);

    let counts_reads = RecordedReads::default();
    let counts =
        queue_counts_observed(&store, fixture.context_id, &counts_reads).expect("queue counts");
    assert_eq!(
        counts,
        QueueCounts {
            pending: fixture.chunk_count - forced.stateful_jobs,
            in_flight: forced.in_flight,
            structured: forced.structured,
            failed: forced.failed,
            skipped: forced.skipped,
        },
        "the queue counts must be the arithmetic total of the states that were forced"
    );
    assert_bounded(
        &counts_reads.taken(),
        fixture.chunk_count,
        "the queue counts",
    );

    // The same question with the machine busy elsewhere: two more capture contexts, and a work
    // class context-graph never looks at, all writing to the same ledger tables.
    let crowd = crowd_the_ledger(&store);
    assert!(
        crowd.foreign_digests >= CROWD_CONTEXTS.len() * CROWD_CHUNKS_PER_CONTEXT
            && crowd.unrelated_jobs == UNRELATED_JOBS
            && crowd.foreign_ledger_rows > 0,
        "the crowd has to be there before it can be proved harmless: {crowd:?}"
    );

    let crowded_reads = RecordedReads::default();
    let crowded_counts = queue_counts_observed(&store, fixture.context_id, &crowded_reads)
        .expect("queue counts with the ledger crowded");
    assert_eq!(
        crowded_counts, counts,
        "another context's work, and another work class's work, must not move this context's counts"
    );
    assert_bounded(
        &crowded_reads.taken(),
        fixture.chunk_count,
        "the queue counts with the ledger crowded",
    );

    // And the footprint follows the purge, with the crowd present: the same bounded door, a
    // smaller correct answer.
    let after_purge_reads = RecordedReads::default();
    let after_purge = raw_footprint_observed(&store, fixture.context_id, &after_purge_reads)
        .expect("raw footprint after purge");
    assert_eq!(
        after_purge,
        RawFootprint {
            chunk_count: fixture.chunk_count - forced.chunks_purged,
            bytes: fixture.raw_bytes - forced.bytes_purged,
        },
        "the raw footprint must follow what the purge reported removing"
    );
    let after_purge_reads = after_purge_reads.taken();
    assert_bounded(
        &after_purge_reads,
        fixture.chunk_count - forced.chunks_purged,
        "the raw footprint after a purge",
    );
    assert_totalled_by_aggregate(&after_purge_reads, "the raw footprint after a purge");

    // Every reader the owner admits is taken by somebody else, and the totals still answer. A
    // process asking its own store for a total is not a client of itself: it reads the live
    // database it already holds, so a status total never waits for an admission slot and never
    // takes one from a client that needs it.
    let options = context_graph_read_session_options();
    let capacity = admission_report(store.db_path()).capacity;
    assert!(
        capacity > 0,
        "an owner serving reads admits at least one reader"
    );
    let outside: Vec<ReadSession> = (0..capacity)
        .map(|_| {
            ReadSession::open_with_options(store.db_path(), options.clone())
                .expect("open a reading session from outside the owner")
        })
        .collect();
    let held: Vec<ReadCursor> = outside
        .iter()
        .map(|session| {
            let cursor = session
                .open_cursor("SELECT id FROM digests", &HashMap::new())
                .expect("hold one of the owner's readers open");
            assert!(
                cursor.first_page().has_more,
                "a reader that finished at open would hand its slot straight back"
            );
            cursor
        })
        .collect();
    assert_eq!(
        admission_report(store.db_path()).active_readers,
        capacity,
        "every reader this owner admits is taken before the totals are asked for"
    );

    let footprint_while_saturated =
        raw_footprint(&store, fixture.context_id).expect("raw footprint while the owner is full");
    assert_eq!(
        footprint_while_saturated, after_purge,
        "a status total answers the same whether or not the owner has a reader to spare"
    );
    assert_eq!(
        admission_report(store.db_path()).active_readers,
        capacity,
        "the totals neither took one of the owner's readers nor gave one back"
    );

    let unrelated_context = *crowd
        .contexts
        .first()
        .expect("the crowd captured an unrelated context");
    let counts_while_saturated =
        queue_counts(&store, unrelated_context).expect("queue counts while the owner is full");
    assert_eq!(
        counts_while_saturated,
        QueueCounts {
            pending: CROWD_CHUNKS_PER_CONTEXT
                - (CROWD_CLAIMED + CROWD_RESULTED + CROWD_FAILED + CROWD_CANCELLED),
            in_flight: CROWD_CLAIMED,
            structured: CROWD_RESULTED,
            failed: CROWD_FAILED,
            skipped: CROWD_CANCELLED,
        },
        "a queue count answers one context's own work, and answers it whether or not the owner has \
         a reader to spare"
    );

    drop(held);
    drop(outside);
}

/// How many readers the live owner of the store at `db_path` admits, and how many are taken.
///
/// Asking bypasses admission, so a full owner still says what it is.
fn admission_report(db_path: &std::path::Path) -> OwnerAdmissionReport {
    ReadSession::owner_report(db_path, context_graph_read_session_options())
        .expect("the live owner reports how it is serving")
        .serving
        .expect("a live owner reports the readers it admits")
        .admission
}
