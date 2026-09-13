//! The typed graph reads a consumer gets over a store it must not change.
//!
//! A downstream runtime whose store is held by another process — or whose whole job forbids it to
//! risk changing one — still has to answer "why did this happen?" and "what happened lately?".
//! Those answers are graph reads: an audit trail, an observation and the observations around it,
//! the decision that fired, the intention it served, the entity it was about, and the context all
//! of it sits in. Every one of them used to be a typed `&Store` method behind the WRITABLE open,
//! and that open mutates the store file. So a fallback had two bad choices: change a store it was
//! told not to touch, or decode the rows itself and drift from what the writable path answers.
//!
//! [`ConsumerGraphView`] is the third answer. It carries the same reads, with the same signatures
//! and the same answer types as [`crate::Store`]'s, over the read-only
//! [`ConsumerReader`]'s session — and it is the SAME code: one SQL text and one decoder per read,
//! authored once behind [`crate::store::graph_rows::GraphRows`] and run by whichever executor the
//! caller reached it through. There is no second decoder to drift and no copied statement to fall
//! behind a schema change.
//!
//! # What a consumer must declare
//!
//! The declaration is the whole of what the view may reach, exactly as it is for SQL: a read whose
//! table the consumer did not declare is refused as [`CgError::UndeclaredTableRefused`] naming
//! that table, before the engine is asked. The tables are cg's own, so a consumer declares them
//! from cg's published [`crate::schema::TABLES`] specs rather than from a copy of the DDL text.
//! Per read:
//!
//! | Read | Tables it needs |
//! | --- | --- |
//! | [`ConsumerGraphView::audit_query`] | `audit_log` |
//! | [`ConsumerGraphView::list_contexts`], [`ConsumerGraphView::get_context`] | `contexts` |
//! | [`ConsumerGraphView::get_entity`] | `entities` |
//! | [`ConsumerGraphView::get_intention`] | `intentions` |
//! | [`ConsumerGraphView::get_decision`] | `decisions`, `edges`, `decision_basis_snapshots`, `outcomes` |
//! | [`ConsumerGraphView::get_observation`], [`ConsumerGraphView::list_observations`] | `observations`, `evidence`, `evidence_availability_log` |
//! | [`ConsumerGraphView::get_event`], [`ConsumerGraphView::list_stream_events`] | `events` |
//!
//! The nested tables are there because the ANSWERS are nested: a decision carries the intentions
//! it serves, the precedents it cites, its basis snapshots and its outcome; an observation carries
//! its evidence, and each piece of evidence the retention status its availability log last
//! recorded. A consumer that declared only the headline table is told which table it still owes
//! rather than served a stripped row.
//!
//! # What the view does not change
//!
//! Row scope is the engine's, decided row by row on the session's declared labels exactly as it is
//! for SQL — a row outside them is invisible, not refused. The typed missing/unreadable/needs-
//! recovery outcomes belong to [`ConsumerReader::open`]; by the time there is a view, the store
//! has been opened. And nothing in the path holds a writable handle, so no sequence of view reads
//! changes the store file, its companions, or the directory they sit in.
//!
//! Reads run under the shipped context-graph read ceilings
//! ([`crate::owner_control::context_graph_read_limits`]), and those ceilings are never raised to
//! make a scan fit. Paging past them is this executor's job, not the caller's: a typed read whose
//! answer outgrows one bounded answer runs its OWN SQL text through a cursor, page by page under
//! the engine's work ceiling, and the concatenated rows go through the same decoder. So a site
//! with more observations than the ceiling carries is answered in full, exactly as
//! [`crate::Store`] answers it — never a refusal the writable path does not give, and never a
//! shorter list that reads as a smaller store. [`ConsumerReader::open_cursor`] stays there for a
//! consumer paging SQL of its own.

use std::collections::HashMap;

use contextdb_core::Value;
use contextdb_engine::QueryResult;

use crate::Result;
use crate::store::graph_rows::GraphRows;
use crate::store::{audit, context, decision, entity, event, intention, observation};
use crate::types::{
    AnchorKey, AnchorValue, AuditEntry, AuditFilter, Context, ContextId, ConversationEvent,
    Decision, DecisionId, Entity, EntityId, EventId, EventRange, EventStreamSelector, Intention,
    IntentionId, Observation, ObservationId,
};

use super::ConsumerReader;

/// The typed graph reads, over one [`ConsumerReader`]'s read session.
///
/// Borrowed from the reader that opened the store and carries nothing of its own: the session, the
/// declaration and the engine's row gate are the reader's, and the view is just the typed way in
/// to them. Drop it and the reader is still there to close.
#[derive(Debug)]
pub struct ConsumerGraphView<'reader> {
    reader: &'reader ConsumerReader,
}

impl<'reader> ConsumerGraphView<'reader> {
    pub(super) fn new(reader: &'reader ConsumerReader) -> Self {
        Self { reader }
    }

    /// The store this view reads.
    pub fn path(&self) -> &std::path::Path {
        self.reader.path()
    }

    /// The audit trail, filtered as the writable path filters it.
    ///
    /// Needs `audit_log` declared.
    pub fn audit_query(&self, filter: AuditFilter) -> Result<Vec<AuditEntry>> {
        audit::audit_query(self, filter)
    }

    /// Every context this reader can see.
    ///
    /// Needs `contexts` declared.
    pub fn list_contexts(&self) -> Result<Vec<Context>> {
        context::list_contexts(self)
    }

    /// One context by id, or the answered absence when there is no such row.
    ///
    /// Needs `contexts` declared.
    pub fn get_context(&self, id: ContextId) -> Result<Option<Context>> {
        context::get_context(self, id)
    }

    /// One entity by id, or the answered absence when there is no such row.
    ///
    /// Needs `entities` declared.
    pub fn get_entity(&self, id: EntityId) -> Result<Option<Entity>> {
        entity::get_entity(self, id)
    }

    /// One intention by id, or the answered absence when there is no such row.
    ///
    /// Needs `intentions` declared.
    pub fn get_intention(&self, id: IntentionId) -> Result<Option<Intention>> {
        intention::get_intention(self, id)
    }

    /// One decision by id, carrying the intentions it serves, the precedents it cites, its basis
    /// snapshots and its outcome — or the answered absence when there is no such row.
    ///
    /// Needs `decisions`, `edges`, `decision_basis_snapshots` and `outcomes` declared.
    pub fn get_decision(&self, id: DecisionId) -> Result<Option<Decision>> {
        decision::get_decision(self, id)
    }

    /// One observation by id, with the evidence it carries. An observation that is not in the
    /// store is the same typed refusal the writable path gives, not an empty answer.
    ///
    /// Needs `observations`, `evidence` and `evidence_availability_log` declared.
    pub fn get_observation(&self, id: ObservationId) -> Result<Observation> {
        observation::get_observation(self, id)
    }

    /// The observations in one context, or every observation this reader can see when no context
    /// is named, each with the evidence it carries.
    ///
    /// Needs `observations`, `evidence` and `evidence_availability_log` declared.
    pub fn list_observations(&self, context_id: Option<ContextId>) -> Result<Vec<Observation>> {
        observation::list_observations(self, context_id)
    }

    /// One immutable event inside the supplied Context. Needs `events` declared.
    pub fn get_event(
        &self,
        context_id: ContextId,
        id: EventId,
    ) -> Result<Option<ConversationEvent>> {
        event::get_event(self, context_id, id)
    }

    /// One bounded source stream in Store-assigned order. Needs `events` declared.
    pub fn list_stream_events(
        &self,
        stream: EventStreamSelector,
        range: EventRange,
    ) -> Result<Vec<ConversationEvent>> {
        event::list_stream_events(self, stream, range)
    }

    /// Events in one Context whose effective exact anchors hold `key` = `value`, answered exactly
    /// as [`crate::Store::list_events_by_anchor`] answers. Needs `events`, `event_anchors`,
    /// `event_interpretation_heads`, `event_interpretation_source_events` and
    /// `event_interpretation_anchors` declared.
    pub fn list_events_by_anchor(
        &self,
        context_id: ContextId,
        key: AnchorKey,
        value: AnchorValue,
        range: EventRange,
    ) -> Result<Vec<ConversationEvent>> {
        event::list_events_by_anchor(self, context_id, key, value, range)
    }
}

/// The read-only executor: every typed read above reaches the store through a cursor the reader's
/// own gated [`ConsumerReader::open_cursor`] opened, so the declaration check and the engine's row
/// gate apply to a typed read exactly as they do to SQL, and there is no second way out of the
/// session.
///
/// What differs from the writable executor is only HOW the rows arrive. The statement, its
/// parameters and its decoder are the read's own, authored once; this side streams the answer a
/// page at a time and hands back the concatenation, because a consumer asking for a site's
/// observations never said how many there would be and a bounded single answer would refuse the
/// large ones the writable path answers.
impl GraphRows for ConsumerGraphView<'_> {
    fn query_rows(&self, sql: &str, params: &HashMap<String, Value>) -> Result<QueryResult> {
        let mut cursor = self.reader.open_cursor(sql, params)?;
        // A read that STOPS is exactly where a cursor gets abandoned, and the consumer that only
        // wanted to look is the one left holding the snapshot. So every way out of the paging loop
        // — a refused first page, a refused fetch, or the last page — leaves through the same
        // close.
        let paged = (|| -> Result<(Vec<String>, Vec<Vec<Value>>)> {
            let first = cursor.first_page()?;
            let mut columns = first.columns.clone();
            let mut rows = first.rows.clone();
            let mut has_more = first.has_more;
            while has_more {
                let page = cursor.fetch(None)?;
                // A page that carries no rows names no columns either; the decoder needs the
                // projection, so the first page that states one is the one believed.
                if columns.is_empty() {
                    columns = page.columns;
                }
                rows.extend(page.rows);
                has_more = page.has_more;
            }
            Ok((columns, rows))
        })();
        let closed = cursor.close();
        // What went wrong first is what the caller is told: a close that also failed cannot
        // displace the refusal that stopped the read.
        let (columns, rows) = paged?;
        closed?;
        Ok(QueryResult {
            columns,
            rows,
            ..QueryResult::empty()
        })
    }
}
