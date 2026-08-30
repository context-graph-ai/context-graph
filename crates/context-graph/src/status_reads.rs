//! The bounded reading door `cg status` totals go through.
//!
//! A status total on a real store is a question about a table far larger than one read may examine:
//! a developer who imported a year of sessions has hundreds of thousands of raw chunks, and the
//! shipped ceilings stop a single read at 50,000 examined items and 500 returned rows. So a total is
//! never a whole-table scan pulled back into Rust. It is an aggregate — or, where identities are
//! needed, a page of rows — opened as a CURSOR and read to completion under the ceilings
//! [`crate::owner_control::context_graph_read_limits`] declares once for the whole crate. Only one
//! page is ever held at a time: what a status total retains is its counters, not its input.
//!
//! The session is opened on the LIVE database the calling process already holds, not on the store's
//! path. A process asking its own store for a total is not a client of itself: routing that question
//! back through the owner channel would spend an owner admission slot on the owner's own work, so a
//! status total taken while every admission is occupied by outside readers still answers.
//!
//! Every read made here is reported to the caller's [`StatusReadObserver`], so what a total actually
//! cost is available to whoever asked for it rather than inferred. Each read reports its OWN cost,
//! so what a total spent is the SUM of its observations: the keyed reads a fold runs inside an
//! outer paged read ADD to that total rather than hiding behind whichever of them was deepest.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use contextdb_core::Value as DbValue;
use contextdb_engine::{ReadPhase, ReadProgress, ReadProgressObserver, ReadSession};

use crate::owner_control::context_graph_read_limits;
use crate::tier0::map_engine;
use crate::{Result, Store};

/// What one bounded read — a cursor open, or one fetch on it — did.
///
/// `rows_examined` is what the engine reported this read going through, and never fewer than the
/// rows the read handed back — a read cannot return a row it never looked at. So an observation
/// cannot flatter a scan that never paged, and a read the engine had no occasion to report on is
/// not written down as having cost nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadObservation {
    /// The statement the bounded kernel ran.
    pub statement: String,
    /// Input items this ONE read went through, as a floor: the engine's own highest report for it,
    /// or the rows it handed back, whichever is larger.
    pub rows_examined: u64,
    /// Rows that read handed back.
    pub rows_returned: u64,
}

/// Told what each bounded read a status total makes actually did, as it happens.
///
/// Public contract: a caller that wants to know what a total cost supplies one of these to the
/// observing form of the total ([`crate::tier0::queue_counts_observed`],
/// [`crate::tier0::raw_footprint_observed`]) and is told about every read that total makes, in the
/// order the reads were made. A caller that only wants the number passes [`UnobservedReads`] and
/// takes the same code path — there is no separate observed build of the totals, so what an
/// observer sees is what production does.
pub trait StatusReadObserver {
    /// One bounded read finished.
    fn read(&self, observation: ReadObservation);
}

/// The observer of a caller that only wants the total: it is told about every read and keeps
/// nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnobservedReads;

impl StatusReadObserver for UnobservedReads {
    fn read(&self, _observation: ReadObservation) {}
}

/// What the engine says a read went through, forwarded as it arrives.
///
/// Two different things arrive here and both are kept. `progress` is LIVENESS: the engine speaks
/// only once a read has done an interval's worth, so a read that finishes inside one interval says
/// nothing there and a caller adding those up would read a finished scan as costing nothing.
/// `completed` is ACCOUNTING: the totals a finished read owes, told exactly once, whatever the
/// intervals did. So the totals are what a status observation is built from, and the liveness
/// samples — which are mid-flight and can repeat a count — only ever raise a floor, never a total.
///
/// The reports travel a channel rather than a shared counter: nothing about a status read is
/// contended, and the reader end belongs to the one session that asked. Hydration reports carry no
/// work and are left alone.
struct ExaminedItems {
    reports: Sender<u64>,
}

impl ReadProgressObserver for ExaminedItems {
    fn progress(&self, progress: ReadProgress) {
        if progress.phase != ReadPhase::Executing {
            return;
        }
        let _ = self.reports.send(progress.work);
    }

    fn completed(&self, totals: ReadProgress) {
        let _ = self.reports.send(totals.work);
    }
}

/// One reading session over the live database a process already holds, with the engine reporting
/// what each of its reads examines.
///
/// Reachable outside the crate only so a test can watch what a session is still holding after a
/// read stopped early; it is `#[doc(hidden)]` and carries no published surface promise.
#[doc(hidden)]
pub struct StatusReads<'observer> {
    session: ReadSession,
    examined: Receiver<u64>,
    observer: &'observer dyn StatusReadObserver,
    /// Cursors opened on this session and not yet closed. contextdb's `ReadCursor` has no `Drop`,
    /// so this is the session's own accounting of what it would still be holding.
    open_cursors: Cell<u64>,
}

impl<'observer> StatusReads<'observer> {
    #[doc(hidden)]
    pub fn open(store: &Store, observer: &'observer dyn StatusReadObserver) -> Result<Self> {
        let (reports, examined) = channel();
        let session = store
            .internal_db()
            .read_session_with_progress(
                context_graph_read_limits(),
                Arc::new(ExaminedItems { reports }) as Arc<dyn ReadProgressObserver>,
            )
            .map_err(map_engine)?;
        Ok(Self {
            session,
            examined,
            observer,
            open_cursors: Cell::new(0),
        })
    }

    /// Cursors this session opened and has not closed.
    ///
    /// contextdb's `ReadCursor` has no `Drop`, so a cursor left behind on an error path holds its
    /// slot and the snapshot it pins until the whole session goes away. This is the session's OWN
    /// counter — incremented where a cursor is opened, decremented where one is handed to
    /// `close()` — so it needs no engine test seam and stays true in an ordinary build.
    #[doc(hidden)]
    pub fn open_cursors(&self) -> u64 {
        self.open_cursors.get()
    }

    /// What the read that just finished went through.
    ///
    /// The channel is drained once per read — a nested read drains its own reports before the read
    /// containing it drains again — so the highest running count in one drain belongs to one read.
    /// Every read's cost therefore reaches the observer as its own observation, and a caller adds
    /// them up; the largest of them never stands in for all of them.
    ///
    /// A read that examined nothing measurable and returned nothing reports nothing, and a read
    /// whose totals never reached this session is still not written down as having cost nothing: a
    /// read that handed back rows went through at least that many, and that floor is reported
    /// rather than a fabricated 0.
    fn examined(&self, rows_returned: u64) -> u64 {
        self.examined
            .try_iter()
            .max()
            .unwrap_or(0)
            .max(rows_returned)
    }

    fn record(&self, statement: &str, rows_returned: usize) {
        let rows_returned = u64::try_from(rows_returned).unwrap_or(u64::MAX);
        self.observer.read(ReadObservation {
            statement: statement.to_string(),
            rows_examined: self.examined(rows_returned),
            rows_returned,
        });
    }

    /// Read a statement to completion, handing each bounded page to `fold` as it arrives.
    ///
    /// The statement is opened as a cursor and fetched until the engine says there is no more
    /// input: every fetch spends its own work budget, so a table of any size is read through
    /// without any one read going past what it is allowed. Pages are not collected — `fold` sees
    /// one page and the page is gone, so what this costs in memory does not grow with the table.
    #[doc(hidden)]
    pub fn for_each_page(
        &self,
        statement: &str,
        params: &HashMap<String, DbValue>,
        mut fold: impl FnMut(&[Vec<DbValue>]) -> Result<()>,
    ) -> Result<()> {
        let mut cursor = self
            .session
            .open_cursor(statement, params)
            .map_err(map_engine)?;
        self.open_cursors
            .set(self.open_cursors.get().saturating_add(1));
        // A read that STOPS is exactly where a cursor gets abandoned, and the caller who retries
        // after a failed status is the one who pays for it. So every way out of the paging loop —
        // a refused fold, a refused fetch, or the last page — leaves through the same close.
        let paged = (|| -> Result<()> {
            let first = cursor.first_page();
            let mut has_more = first.has_more;
            self.record(statement, first.rows.len());
            fold(&first.rows)?;
            while has_more {
                let page = cursor.fetch(None).map_err(map_engine)?;
                self.record(statement, page.rows.len());
                fold(&page.rows)?;
                has_more = page.has_more;
            }
            Ok(())
        })();
        let closed = cursor.close().map_err(map_engine);
        self.open_cursors
            .set(self.open_cursors.get().saturating_sub(1));
        // What went wrong first is what the caller is told: a close that also failed cannot
        // displace the refusal that stopped the read.
        paged?;
        closed
    }

    /// Run one aggregate to completion and give back its single answer row.
    ///
    /// The pages before the last carry no rows and the final page carries the one-row total, so the
    /// answer is the last row any page published.
    pub(crate) fn total(
        &self,
        statement: &str,
        params: &HashMap<String, DbValue>,
    ) -> Result<Vec<DbValue>> {
        let mut answer = None;
        self.for_each_page(statement, params, |rows| {
            if let Some(row) = rows.first() {
                answer = Some(row.clone());
            }
            Ok(())
        })?;
        answer.ok_or_else(|| {
            crate::CgError::Engine(format!(
                "the aggregate `{statement}` finished without publishing a total"
            ))
        })
    }
}

/// The integer an aggregate column published, treating "no rows reached it" as zero.
pub(crate) fn aggregate_u64(row: &[DbValue], column: usize) -> u64 {
    match row.get(column) {
        Some(DbValue::Int64(value)) => (*value).max(0) as u64,
        Some(DbValue::Float64(value)) if *value > 0.0 => *value as u64,
        _ => 0,
    }
}
