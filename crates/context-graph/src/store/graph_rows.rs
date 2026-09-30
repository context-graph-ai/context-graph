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
        let session = self
            .internal_db()
            .read_session(self.read_limits())
            .map_err(super::convert::map_engine_err)?;
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
