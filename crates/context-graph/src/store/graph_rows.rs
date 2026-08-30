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
