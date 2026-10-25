//! The one seam a typed graph read runs its statements through.
//!
//! The typed reads — the audit trail, contexts, entities, intentions, decisions and observations —
//! are authored ONCE, as free functions over this trait, and two executors implement it: the
//! writable [`crate::Store`], which runs them on the handle it owns, and the read-only
//! [`crate::consumer_read::graph::ConsumerGraphView`], which runs them on the consumer's read
//! session. One SQL text and one decoder per read, so the two doors cannot drift into answering
//! differently; what differs between them is only WHICH executor the statement reaches.
//!
//! Refusals are already this crate's own by the time they come back: each executor converts the
//! engine's error at its own edge, so a read written here never sees the substrate's vocabulary.

use std::collections::HashMap;

use contextdb_core::Value;
use contextdb_engine::QueryResult;

use crate::Result;

#[cfg(test)]
thread_local! {
    static READ_CLOCK: std::cell::RefCell<Option<std::sync::Arc<dyn contextdb_core::read_contract::DeadlineClock>>> = const { std::cell::RefCell::new(None) };
}

/// Adapt the substrate's existing deadline seam to a synchronous Store read.
/// Thread-local and unit-test-only: other stores and production sessions keep their clock.
#[cfg(test)]
pub(super) fn with_read_clock_for_test<T>(
    clock: std::sync::Arc<dyn contextdb_core::read_contract::DeadlineClock>,
    read: impl FnOnce() -> T,
) -> T {
    struct Reset(Option<std::sync::Arc<dyn contextdb_core::read_contract::DeadlineClock>>);
    impl Drop for Reset {
        fn drop(&mut self) {
            READ_CLOCK.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let _reset = Reset(READ_CLOCK.with(|slot| slot.replace(Some(clock))));
    read()
}

/// One executor of a read statement.
pub(crate) trait GraphRows {
    /// Run one read and hand back the rows the engine answered with.
    ///
    /// The read-only executor also decides here whether the statement is inside the consumer's
    /// declaration, so a read reaching an undeclared table is refused by name before the engine is
    /// asked — which is why every typed read below runs its statements through this one call
    /// rather than reaching for a handle of its own.
    fn query_rows(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult>;
}

impl crate::Store {
    /// Read an ordinary statement through this Store's live session, preserving complete lists.
    pub(crate) fn read_query(
        &self,
        sql: &str,
        params: &HashMap<String, Value>,
    ) -> Result<QueryResult> {
        #[cfg(test)]
        let clock = READ_CLOCK.with(|slot| slot.borrow().clone());
        #[cfg(test)]
        let test_session = clock.map(|clock| {
            self.internal_db()
                .read_session_with_clock_for_test(self.read_limits(), clock)
        });
        #[cfg(test)]
        let session =
            test_session.unwrap_or_else(|| self.internal_db().read_session(self.read_limits()));
        #[cfg(not(test))]
        let session = self.internal_db().read_session(self.read_limits());
        let session = session.map_err(super::convert::map_engine_err)?;
        let mut cursor = session
            .open_cursor(sql, params)
            .map_err(super::convert::map_engine_err)?;
        let paged = (|| -> Result<QueryResult> {
            let first = cursor.first_page();
            let mut answer = QueryResult {
                columns: first.columns.clone(),
                rows: first.rows.clone(),
                ..QueryResult::empty()
            };
            let mut has_more = first.has_more;
            while has_more {
                let page = cursor.fetch(None).map_err(super::convert::map_engine_err)?;
                if answer.columns.is_empty() {
                    answer.columns = page.columns;
                }
                answer.rows.extend(page.rows);
                has_more = page.has_more;
            }
            Ok(answer)
        })();
        let closed = cursor.close().map_err(super::convert::map_engine_err);
        let answer = paged?;
        closed?;
        Ok(answer)
    }
}

impl crate::Store {
    /// Run a lookup a write makes for itself — the embedding-space registry, the entity it
    /// attaches to, what its own commit produced — as an ordinary statement read at the latest
    /// committed snapshot.
    ///
    /// The read budget bounds what a caller's read may cost, and on a file-backed store a bounded
    /// read also spends that allowance waiting for the commit boundary. A write is not a caller's
    /// read: while a reindex commits row after row, a lookup charged against that budget would
    /// refuse the write for time it spent queued behind other commits.
    pub(crate) fn write_lookup(
        &self,
        sql: &str,
        params: &HashMap<String, Value>,
    ) -> Result<QueryResult> {
        self.internal_db()
            .execute(sql, params)
            .map_err(super::convert::map_engine_err)
    }

    /// The typed graph reads, run as a write's own lookups.
    pub(crate) fn write_lookup_rows(&self) -> WriteLookupRows<'_> {
        WriteLookupRows(self)
    }
}

/// Executor for a typed graph read made on behalf of a write; see [`crate::Store::write_lookup`].
pub(crate) struct WriteLookupRows<'a>(&'a crate::Store);

impl GraphRows for WriteLookupRows<'_> {
    fn query_rows(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult> {
        self.0.require_application_ready()?;
        self.0.write_lookup(sql, params)
    }
}

/// A decoded row for the existing library table readers.
pub(crate) struct ReadRow {
    pub(crate) values: HashMap<String, Value>,
}
impl crate::Store {
    pub(crate) fn read_table(&self, table: &str) -> Result<Vec<ReadRow>> {
        let answer = self.read_query(&format!("SELECT * FROM {table}"), &HashMap::new())?;
        Ok(answer
            .rows
            .into_iter()
            .map(|row| ReadRow {
                values: answer.columns.iter().cloned().zip(row).collect(),
            })
            .collect())
    }
}
