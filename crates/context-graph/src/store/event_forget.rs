//! Forgetting imported history: one authoritative ContextDB `PURGE` per selection.
//!
//! A selection is one session or one Context. Its events, exact anchors, local commit records,
//! file checkpoints, source locators and ordering declarations, coverage-gap links, and every
//! interpretation and search row derived from it are erased by one standalone statement, so the
//! erasure is complete or nothing changes. The Context row and any workspace binding remain.
//!
//! Interpretation and search generations summarize a whole Context's captured watermark, so
//! forgetting one session also erases that Context's derived interpretation and search rows. The
//! search index is rebuilt from the remaining immutable events by the ordinary projection path (the
//! CLI schedules that rebuild itself). An activated interpretation is NOT restored by that path:
//! the receipt names each one it retired ([`RetiredInterpretation`]), and until the host builds
//! and activates that released connector interpretation again, the remaining events read with
//! their captured fields. Coverage-gap fingerprints are erased only when every event they
//! describe is forgotten.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use contextdb_core::Value;
use contextdb_core::read_contract::CursorPage;
use serde::{Deserialize, Serialize};

use crate::consumer_read::ConsumerCursor;
use crate::{CgError, ConnectorId, ConsumerReader, ContextId, EventPlacementState, Result, Store};

use super::convert::map_engine_err;
use super::event::rewrite_event_context_counts;

/// What to forget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "selector", rename_all = "snake_case")]
pub enum ForgetSelection {
    /// One imported session inside one Context.
    Session {
        context_id: ContextId,
        connector_id: ConnectorId,
        session_ref: String,
    },
    /// Every imported session in one Context.
    Context { context_id: ContextId },
}

impl ForgetSelection {
    pub fn context_id(&self) -> ContextId {
        match self {
            ForgetSelection::Session { context_id, .. }
            | ForgetSelection::Context { context_id } => *context_id,
        }
    }
}

impl ConsumerReader {
    /// Count the distinct sessions the forget planner selects, including sources with no events.
    /// Merge three ordered, bounded cursor streams so the count needs no set of all sessions.
    pub fn forget_selection_session_count(&self, selection: &ForgetSelection) -> Result<u64> {
        let mut params = HashMap::from([(
            "context_id".to_string(),
            Value::Uuid(selection.context_id().as_uuid()),
        )]);
        let predicate = match selection {
            ForgetSelection::Session {
                connector_id,
                session_ref,
                ..
            } => {
                params.insert(
                    "connector_id".to_string(),
                    Value::Text(connector_id.clone()),
                );
                params.insert("session_ref".to_string(), Value::Text(session_ref.clone()));
                "context_id = $context_id AND connector_id = $connector_id AND session_ref = $session_ref"
            }
            ForgetSelection::Context { .. } => "context_id = $context_id",
        };
        let mut streams = Vec::with_capacity(3);
        let read = (|| {
            for table in [
                "events",
                "event_source_checkpoints",
                "event_source_locators",
            ] {
                let cursor = self.open_cursor(
                    &format!(
                        "SELECT DISTINCT connector_id, session_ref FROM {table} WHERE {predicate} ORDER BY connector_id ASC, session_ref ASC"
                    ),
                    &params,
                )?;
                streams.push(ForgetSessionStream::new(table, cursor));
            }
            let mut heads = streams
                .iter_mut()
                .map(ForgetSessionStream::next_key)
                .collect::<Result<Vec<_>>>()?;
            let mut count = 0_u64;
            while let Some(key) = heads.iter().filter_map(Option::as_ref).min().cloned() {
                count = count
                    .checked_add(1)
                    .ok_or_else(|| CgError::Engine("forget session count overflowed".into()))?;
                for (stream, head) in streams.iter_mut().zip(&mut heads) {
                    while head.as_ref() == Some(&key) {
                        *head = stream.next_key()?;
                    }
                }
            }
            Ok(count)
        })();
        let mut close_error = None;
        for stream in &mut streams {
            if let Err(error) = stream.close()
                && close_error.is_none()
            {
                close_error = Some(error);
            }
        }
        let count = read?;
        if let Some(error) = close_error {
            return Err(error);
        }
        Ok(count)
    }
}

/// One table's sorted distinct keys, retaining at most one cursor page.
struct ForgetSessionStream {
    table: &'static str,
    cursor: ConsumerCursor,
    page: Option<CursorPage>,
    offset: usize,
}

impl ForgetSessionStream {
    fn new(table: &'static str, cursor: ConsumerCursor) -> Self {
        Self {
            table,
            cursor,
            page: None,
            offset: 0,
        }
    }

    fn next_key(&mut self) -> Result<Option<(String, String)>> {
        if self.page.is_none() {
            self.page = Some(self.cursor.first_page()?.clone());
        }
        loop {
            let page = self.page.as_ref().expect("initialized above");
            if let Some(row) = page.rows.get(self.offset) {
                self.offset += 1;
                return match (row.first(), row.get(1)) {
                    (Some(Value::Text(connector)), Some(Value::Text(session))) => {
                        Ok(Some((connector.clone(), session.clone())))
                    }
                    other => Err(CgError::Engine(format!(
                        "forget session read from {} returned {other:?}",
                        self.table
                    ))),
                };
            }
            if !page.has_more {
                return Ok(None);
            }
            self.page = Some(self.cursor.fetch(std::num::NonZeroUsize::new(128))?);
            self.offset = 0;
        }
    }

    fn close(&mut self) -> Result<()> {
        self.page = None;
        self.cursor.close()
    }
}

/// Rows one listed table lost in the erasure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgetTableCount {
    pub table: String,
    pub rows: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgetReceipt {
    pub selection: ForgetSelection,
    /// Imported sessions the selection covered.
    pub sessions: u64,
    pub events: u64,
    pub anchors: u64,
    pub passages: u64,
    pub vectors: u64,
    /// Every table the erasure listed and the rows it removed there.
    pub tables: Vec<ForgetTableCount>,
    /// Whether any row was erased.
    pub changed: bool,
    /// Activated interpretations the erasure retired with the Context's derived rows. Until each
    /// is built and activated again over the remaining history, that history reads with its
    /// captured fields.
    pub retired_interpretations: Vec<RetiredInterpretation>,
}

/// The release coordinates of one activated interpretation a forget retired: what a host builds
/// and activates again to restore it for the remaining history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetiredInterpretation {
    pub connector_id: ConnectorId,
    pub target_interpretation_version: String,
    pub mapping_contract_id: String,
    pub mapping_contract_sha256: String,
}

/// Context-wide derived tables, listed children first.
const CONTEXT_DERIVED_TABLES: &[&str] = &[
    "event_context_counts",
    "event_search_fold_progress",
    "event_search_capture_manifest",
    "event_search_heads",
    "event_search_work_policies",
    "event_search_leases",
    "event_search_cursors",
    "event_search_session_progress",
    "event_search_embedding_handles",
    "event_search_passage_sources",
    "event_search_passages",
    "event_search_source_events",
    "event_search_generations",
    "event_interpretation_heads",
    "event_interpretation_anchors",
    "event_interpretation_values",
    "event_interpretation_source_events",
    "event_interpretation_generations",
];

struct PurgePlan {
    selections: Vec<String>,
    params: HashMap<String, Value>,
}

impl PurgePlan {
    fn new(context_id: ContextId) -> Self {
        Self {
            selections: Vec::new(),
            params: HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]),
        }
    }

    fn select(&mut self, table: &str, predicate: &str) {
        self.selections.push(format!("{table} WHERE {predicate}"));
    }

    /// Bind an explicit value list; an empty list selects nothing and the table is omitted.
    fn select_in(&mut self, table: &str, prefix: &str, column: &str, values: Vec<Value>) {
        if values.is_empty() {
            return;
        }
        let mut placeholders = Vec::with_capacity(values.len());
        for (index, value) in values.into_iter().enumerate() {
            let name = format!("{prefix}_{index}");
            self.params.insert(name.clone(), value);
            placeholders.push(format!("${name}"));
        }
        self.select(
            table,
            &format!(
                "context_id = $context_id AND {column} IN ({})",
                placeholders.join(", ")
            ),
        );
    }

    fn statement(&self) -> String {
        format!("PURGE FROM {}", self.selections.join(", "))
    }
}

impl Store {
    /// Erase one imported session or one Context's imported history and everything derived from
    /// it in one authoritative erasure. A tenant-hub Store is refused: an enrolled host never
    /// holds fleet-erasing authority, and its edge-local discard is not issued by this build.
    pub fn forget_event_history(&self, selection: ForgetSelection) -> Result<ForgetReceipt> {
        let placement = self.event_placement_state()?;
        if placement != EventPlacementState::LocalOnly {
            return Err(CgError::EventHistoryForgetRequiresEdgeDiscard { placement });
        }
        self.require_application_ready()?;
        let context_id = selection.context_id();
        require_context(self, context_id)?;
        let prepared = match &selection {
            ForgetSelection::Session {
                connector_id,
                session_ref,
                ..
            } => session_plan(self, context_id, connector_id, session_ref)?,
            ForgetSelection::Context { .. } => context_plan(self, context_id)?,
        };
        let Some((plan, sessions)) = prepared else {
            return Ok(ForgetReceipt {
                selection,
                sessions: 0,
                events: 0,
                anchors: 0,
                passages: 0,
                vectors: 0,
                tables: Vec::new(),
                changed: false,
                retired_interpretations: Vec::new(),
            });
        };
        let retired_interpretations = active_interpretations(self, context_id)?;
        // The Context's captured-event count row is erased by the same authoritative erasure as the
        // history it summarizes, never retired around it. An append that commits while a forget runs
        // therefore can neither leave a pre-purge total behind (the erasure takes whatever row that
        // append wrote) nor lose its increment to a restored copy (a refused erasure writes nothing
        // at all). A crash between the erasure and the rewrite below leaves NO row, and every reader
        // (status, find) and the next append recount that Context from its events; a stale total can
        // never survive. The cost of that window is one recount per read until the next append or
        // forget writes the row again.
        let answer = match self.db.execute(&plan.statement(), &plan.params) {
            Ok(answer) => answer,
            Err(error) => return Err(map_engine_err(error)),
        };
        // The erasure committed; the receipt below is true whatever happens next. If this rewrite
        // fails the row stays absent and readers recount, so its failure does not fail the forget.
        let _ = rewrite_event_context_counts(self, context_id);
        let table_index = answer
            .columns
            .iter()
            .position(|column| column == "table")
            .ok_or_else(|| CgError::Engine("the erasure report has no table column".into()))?;
        let rows_index = answer
            .columns
            .iter()
            .position(|column| column == "rows_affected")
            .ok_or_else(|| {
                CgError::Engine("the erasure report has no rows_affected column".into())
            })?;
        let mut tables = Vec::with_capacity(answer.rows.len());
        let mut by_table = BTreeMap::new();
        for row in &answer.rows {
            let table = match row.get(table_index) {
                Some(Value::Text(table)) => table.clone(),
                other => {
                    return Err(CgError::Engine(format!(
                        "the erasure report names no table: {other:?}"
                    )));
                }
            };
            let rows = match row.get(rows_index) {
                Some(Value::Int64(rows)) => u64::try_from(*rows).map_err(|_| {
                    CgError::Engine("the erasure report has a negative count".into())
                })?,
                other => {
                    return Err(CgError::Engine(format!(
                        "the erasure report has no count for {table}: {other:?}"
                    )));
                }
            };
            by_table.insert(table.clone(), rows);
            tables.push(ForgetTableCount { table, rows });
        }
        let rows = |table: &str| by_table.get(table).copied().unwrap_or(0);
        Ok(ForgetReceipt {
            selection,
            sessions,
            events: rows("events"),
            anchors: rows("event_anchors"),
            passages: rows("event_search_passages"),
            vectors: rows("event_search_embedding_handles"),
            changed: tables.iter().any(|table| table.rows > 0),
            tables,
            retired_interpretations,
        })
    }
}

/// The release coordinates of every interpretation activated for `context_id`, by connector.
fn active_interpretations(
    store: &Store,
    context_id: ContextId,
) -> Result<Vec<RetiredInterpretation>> {
    let params = HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]);
    let heads = select_rows(
        store,
        "SELECT generation_id FROM event_interpretation_heads WHERE context_id = $context_id",
        &params,
    )?;
    let mut retired = Vec::with_capacity(heads.rows.len());
    for row in &heads.rows {
        let Some(Value::Uuid(generation_id)) = row.first() else {
            return Err(CgError::Engine(
                "forget read an interpretation head without a generation".into(),
            ));
        };
        let generation = select_rows(
            store,
            "SELECT connector_id, target_interpretation_version, mapping_contract_id, mapping_contract_sha256 FROM event_interpretation_generations WHERE context_id = $context_id AND id = $generation_id",
            &HashMap::from([
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                ("generation_id".to_string(), Value::Uuid(*generation_id)),
            ]),
        )?;
        let text = |name: &str| -> Result<String> {
            let index = column(&generation.columns, name)?;
            match generation.rows.first().and_then(|row| row.get(index)) {
                Some(Value::Text(value)) => Ok(value.clone()),
                other => Err(CgError::Engine(format!(
                    "forget read an active interpretation generation without {name}: {other:?}"
                ))),
            }
        };
        retired.push(RetiredInterpretation {
            connector_id: text("connector_id")?,
            target_interpretation_version: text("target_interpretation_version")?,
            mapping_contract_id: text("mapping_contract_id")?,
            mapping_contract_sha256: text("mapping_contract_sha256")?,
        });
    }
    retired.sort_by(|left, right| left.connector_id.cmp(&right.connector_id));
    Ok(retired)
}

/// Only the test seams below name the count table directly; the erasure itself reaches it through
/// the Context-wide table list.
#[cfg(context_graph_internal_test_support)]
use super::event_search_read::EVENT_CONTEXT_COUNTS_TABLE;

#[cfg(context_graph_internal_test_support)]
impl Store {
    /// Test seam: remove the Context's active search head and nothing else, leaving its
    /// generations and history in place with no head, the state reads must report honestly.
    #[doc(hidden)]
    pub fn retire_event_search_head_for_test(&self, context_id: ContextId) -> Result<()> {
        self.db
            .execute(
                "DELETE FROM event_search_heads WHERE context_id = $context_id",
                &HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        Ok(())
    }

    /// Test seam: retire only the Context's event count row, as a forget interrupted before it
    /// rewrote the row leaves the store.
    #[doc(hidden)]
    pub fn retire_event_context_counts_for_test(&self, context_id: ContextId) -> Result<()> {
        self.db
            .execute(
                &format!("DELETE FROM {EVENT_CONTEXT_COUNTS_TABLE} WHERE context_id = $context_id"),
                &HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]),
            )
            .map_err(map_engine_err)?;
        Ok(())
    }
}

fn require_context(store: &Store, context_id: ContextId) -> Result<()> {
    let answer = store
        .db
        .execute(
            "SELECT id FROM contexts WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Uuid(context_id.as_uuid()))]),
        )
        .map_err(map_engine_err)?;
    if answer.rows.is_empty() {
        return Err(CgError::NotFound {
            id: Some(context_id.to_string()),
            missing: Some("context"),
        });
    }
    Ok(())
}

fn select_rows(
    store: &Store,
    sql: &str,
    params: &HashMap<String, Value>,
) -> Result<contextdb_engine::QueryResult> {
    store.db.execute(sql, params).map_err(map_engine_err)
}

fn column(columns: &[String], name: &str) -> Result<usize> {
    columns
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| CgError::Engine(format!("forget selection read is missing {name}")))
}

fn session_plan(
    store: &Store,
    context_id: ContextId,
    connector_id: &str,
    session_ref: &str,
) -> Result<Option<(PurgePlan, u64)>> {
    let params = HashMap::from([
        ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
        ("connector_id".to_string(), Value::Text(connector_id.into())),
        ("session_ref".to_string(), Value::Text(session_ref.into())),
    ]);
    let events = select_rows(
        store,
        "SELECT id, source_instance_ref FROM events WHERE context_id = $context_id AND connector_id = $connector_id AND session_ref = $session_ref",
        &params,
    )?;
    let id_index = column(&events.columns, "id")?;
    let source_index = column(&events.columns, "source_instance_ref")?;
    let mut event_ids = BTreeSet::new();
    let mut sources = BTreeSet::new();
    for row in &events.rows {
        match (row.get(id_index), row.get(source_index)) {
            (Some(Value::Uuid(id)), Some(Value::Text(source))) => {
                event_ids.insert(*id);
                sources.insert(source.clone());
            }
            other => {
                return Err(CgError::Engine(format!(
                    "forget selection read an invalid event row {other:?}"
                )));
            }
        }
    }
    for table in ["event_source_checkpoints", "event_source_locators"] {
        let answer = select_rows(
            store,
            &format!(
                "SELECT source_instance_ref FROM {table} WHERE context_id = $context_id AND connector_id = $connector_id AND session_ref = $session_ref"
            ),
            &params,
        )?;
        for row in &answer.rows {
            if let Some(Value::Text(source)) = row.first() {
                sources.insert(source.clone());
            }
        }
    }
    if event_ids.is_empty() && sources.is_empty() {
        return Ok(None);
    }

    let gap_links = select_rows(
        store,
        "SELECT gap_id, event_id FROM adapter_coverage_gap_events WHERE context_id = $context_id",
        &params,
    )?;
    let gap_index = column(&gap_links.columns, "gap_id")?;
    let event_index = column(&gap_links.columns, "event_id")?;
    let mut gap_members = BTreeMap::<uuid::Uuid, bool>::new();
    for row in &gap_links.rows {
        let (Some(Value::Uuid(gap)), Some(Value::Uuid(event))) =
            (row.get(gap_index), row.get(event_index))
        else {
            return Err(CgError::Engine(
                "forget selection read an invalid coverage-gap link".into(),
            ));
        };
        let all_forgotten = gap_members.entry(*gap).or_insert(true);
        *all_forgotten &= event_ids.contains(event);
    }
    let orphaned_gaps = gap_members
        .into_iter()
        .filter_map(|(gap, all_forgotten)| all_forgotten.then_some(Value::Uuid(gap)))
        .collect::<Vec<_>>();

    let mut plan = PurgePlan::new(context_id);
    plan.params
        .insert("connector_id".into(), Value::Text(connector_id.into()));
    plan.params
        .insert("session_ref".into(), Value::Text(session_ref.into()));
    for table in CONTEXT_DERIVED_TABLES {
        plan.select(table, "context_id = $context_id");
    }
    let event_values = event_ids
        .iter()
        .map(|id| Value::Uuid(*id))
        .collect::<Vec<_>>();
    plan.select_in(
        "adapter_coverage_gap_events",
        "gap_event",
        "event_id",
        event_values.clone(),
    );
    plan.select_in("adapter_coverage_gaps", "gap", "id", orphaned_gaps);
    let session_predicate =
        "context_id = $context_id AND connector_id = $connector_id AND session_ref = $session_ref";
    // The captured-session summary is erased with its immutable events. Other sessions survive the
    // same authoritative purge, so reads never need a post-forget archive recount.
    plan.select("event_context_sessions", session_predicate);
    plan.select("event_search_pending_streams", session_predicate);
    plan.select("event_search_request_selections", session_predicate);
    plan.select("event_search_refusal_frontiers", session_predicate);
    let mut source_values = Vec::with_capacity(sources.len());
    for (index, source) in sources.into_iter().enumerate() {
        let name = format!("ordering_source_{index}");
        plan.params.insert(name.clone(), Value::Text(source));
        source_values.push(format!("${name}"));
    }
    if !source_values.is_empty() {
        plan.select(
            "event_source_ordering",
            &format!(
                "context_id = $context_id AND connector_id = $connector_id AND source_instance_ref IN ({})",
                source_values.join(", ")
            ),
        );
    }
    plan.select("event_source_checkpoints", session_predicate);
    plan.select("event_source_locators", session_predicate);
    plan.select_in(
        "event_local_commits",
        "commit_event",
        "event_id",
        event_values,
    );
    plan.select("event_anchors", session_predicate);
    plan.select("events", session_predicate);
    Ok(Some((plan, 1)))
}

fn context_plan(store: &Store, context_id: ContextId) -> Result<Option<(PurgePlan, u64)>> {
    let params = HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))]);
    let mut sessions = BTreeSet::new();
    let mut any_source_state = false;
    for table in [
        "events",
        "event_source_checkpoints",
        "event_source_locators",
    ] {
        let answer = select_rows(
            store,
            &format!(
                "SELECT DISTINCT connector_id, session_ref FROM {table} WHERE context_id = $context_id"
            ),
            &params,
        )?;
        let connector_index = column(&answer.columns, "connector_id")?;
        let session_index = column(&answer.columns, "session_ref")?;
        for row in &answer.rows {
            any_source_state = true;
            if let (Some(Value::Text(connector)), Some(Value::Text(session))) =
                (row.get(connector_index), row.get(session_index))
            {
                sessions.insert((connector.clone(), session.clone()));
            }
        }
    }
    if !any_source_state {
        return Ok(None);
    }
    let mut plan = PurgePlan::new(context_id);
    for table in CONTEXT_DERIVED_TABLES.iter().chain(
        [
            "event_search_pending_streams",
            "event_search_request_selections",
            "event_search_refusal_frontiers",
            "adapter_coverage_gap_events",
            "adapter_coverage_gaps",
            "event_context_sessions",
            "event_context_revisions",
            "event_source_ordering",
            "event_source_checkpoints",
            "event_source_locators",
            "event_local_commits",
            "event_anchors",
            "events",
        ]
        .iter(),
    ) {
        plan.select(table, "context_id = $context_id");
    }
    Ok(Some((plan, sessions.len() as u64)))
}
