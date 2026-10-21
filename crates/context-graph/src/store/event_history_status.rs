//! Event-history status: the one fold behind `cg status` and every other status caller.
//!
//! The fold is authored once over [`EventSearchRows`], so the writable [`Store`] and the read-only
//! [`ConsumerReader`] run the same statements through the same decoder. It keeps capture, delivery
//! and search coverage as separate facts: captured events are durable even while search is behind,
//! and search is `running` only while a live unexpired projection lease exists.

use std::collections::{BTreeSet, HashMap};

use contextdb_core::{Value, Wallclock};
use serde::{Deserialize, Serialize};

use crate::consumer_read::ConsumerDeclaration;
use crate::schema::{LOCAL_DERIVED_TABLES, LOCAL_RUNTIME_TABLES, TABLES};
use crate::{
    CgError, ConnectorId, ConsumerReader, ContextBindingKey, ContextId, ContextSelector,
    EventCoverageCounts, EventDeliveryState, EventDeliveryStatus, EventPlacementPolicy,
    EventPlacementState, EventSearchGenerationId, InterpretationGenerationId, Result, Store,
};

use super::event_search_read::{EventSearchRows, captured_event_counts};
use super::{ConsumerSchema, ConsumerTable};

/// Tables the read-only event-history reader declares.
const EVENT_HISTORY_READ_TABLES: &[&str] = &[
    "contexts",
    "context_bindings",
    "event_placement",
    "events",
    "event_context_counts",
    "event_context_sessions",
    "event_anchors",
    "event_source_checkpoints",
    "event_source_locators",
    "intentions",
    "decisions",
    "observations",
    "adapter_coverage_gaps",
    "event_interpretation_generations",
    "event_interpretation_heads",
    "event_interpretation_source_events",
    "event_interpretation_anchors",
    "event_search_generations",
    "event_search_source_events",
    "event_search_session_progress",
    "event_search_passages",
    "event_search_passage_sources",
    "event_search_embedding_handles",
    "event_search_heads",
    "event_search_work_policies",
    "event_search_request_selections",
    "event_search_request_headers",
    "event_search_refusal_headers",
    "event_search_refusal_frontiers",
    "event_search_cursors",
    "event_search_pending_streams",
    "event_search_leases",
    "cg_local_settings",
    "work_inputs",
];

/// The engine creates its work-ledger tables and publishes no declaration text for them; the
/// reader verifies a declared table's scope column against the store, and `work_inputs` carries
/// none, so this names the table with its key column only.
const WORK_INPUTS_DECLARATION: &str = "CREATE TABLE work_inputs (job_id TEXT NOT NULL)";

/// The declaration a read-only caller opens [`ConsumerReader`] with to run the event-history
/// reads: status, binding resolution, `find`, `show` and the typed event reads. It is built from
/// context-graph's own published table declarations, so a caller never carries DDL text.
pub fn event_history_reader_declaration() -> ConsumerDeclaration {
    let tables = EVENT_HISTORY_READ_TABLES
        .iter()
        .map(|name| {
            let ddl = TABLES
                .iter()
                .chain(LOCAL_DERIVED_TABLES.iter())
                .chain(LOCAL_RUNTIME_TABLES.iter())
                .find(|table| table.name == *name)
                .map(|table| table.ddl)
                .or((*name == "work_inputs").then_some(WORK_INPUTS_DECLARATION))
                .expect("every event-history read table is a published table");
            ConsumerTable {
                name: (*name).to_string(),
                ddl: ddl.to_string(),
            }
        })
        .collect();
    ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "context_graph_event_history_reader".to_string(),
            tables,
        },
        scope_labels: BTreeSet::new(),
    }
}

/// Graph-memory totals for one Context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EventHistoryMemoryCounts {
    pub goals: u64,
    pub decisions: u64,
    pub observations: u64,
}

/// One captured source stream: a connector's source instance inside one session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventSourceStreamStatus {
    pub connector_id: ConnectorId,
    pub source_instance_ref: String,
    pub session_ref: String,
    pub adapter_version: String,
    pub anchor_declaration_version: String,
    pub event_count: u64,
    pub latest_event_seq: Option<u64>,
}

/// Search coverage is exactly one of these states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventHistorySearchState {
    /// Captured events are not covered and no projection is running.
    Pending,
    /// A live, unexpired projection lease exists.
    Running,
    /// Captured events are not covered and the local embedding artifact is not installed.
    Blocked,
    /// The last projection attempt recorded a failure and no projection is running.
    Failed,
    /// Every captured event is covered by the active search generation.
    Ready,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventHistorySearchStatus {
    pub state: EventHistorySearchState,
    /// The Context's persisted session-selection policy, used for every retry action.
    pub work_policy: crate::EventSearchWorkPolicy,
    pub active_generation_id: Option<EventSearchGenerationId>,
    pub embedding_space_id: Option<String>,
    pub renderer_id: Option<String>,
    pub renderer_version: Option<String>,
    /// The passage profile the active generation was rendered with.
    pub profile: Option<crate::EventSearchProfile>,
    pub covered_events: u64,
    pub pending_events: u64,
    /// Captured events outside the latest accepted session-selection policy.
    #[serde(default)]
    pub outside_scope_events: u64,
    /// Outside-scope events that remain searchable from prior accepted work.
    #[serde(default)]
    pub outside_scope_indexed_events: u64,
    /// Selected events whose completed sessions produced no searchable passage membership.
    #[serde(default)]
    pub no_searchable_text_events: u64,
    /// The generation currently accumulating committed projection work, if any.
    #[serde(default)]
    pub building_generation_id: Option<EventSearchGenerationId>,
    /// The profile of the actual nonactive building generation.
    #[serde(default)]
    pub building_profile: Option<crate::EventSearchProfile>,
    /// Captured events sealed by the building generation at or below their stream cursor.
    #[serde(default)]
    pub building_covered_events: u64,
    /// Captured events not yet sealed by the building generation.
    #[serde(default)]
    pub building_pending_events: u64,
    pub live_projection_lease: bool,
    pub last_failure: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventHistoryInterpretationHead {
    pub connector_id: ConnectorId,
    pub generation_id: InterpretationGenerationId,
}

/// One tagged status snapshot for one selected Context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventHistoryContextStatus {
    pub context_id: ContextId,
    pub context_name: String,
    pub memory: EventHistoryMemoryCounts,
    pub captured_events: u64,
    pub sessions: u64,
    pub source_streams: Vec<EventSourceStreamStatus>,
    pub coverage: EventCoverageCounts,
    pub coverage_gaps: u64,
    pub active_interpretations: Vec<EventHistoryInterpretationHead>,
    pub last_interpretation_failure: Option<String>,
    /// `None` when this door cannot read the engine's delivery ledger (a tenant-hub Store read
    /// through the read-only reader); never a fabricated count.
    pub delivery: Option<EventDeliveryStatus>,
    pub search: EventHistorySearchStatus,
}

/// Store-level placement reported once, plus one tagged snapshot per selected Context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventHistoryStatusSet {
    pub placement: EventPlacementState,
    pub expected_policy: EventPlacementPolicy,
    pub selected_context_ids: Vec<ContextId>,
    pub contexts: Vec<EventHistoryContextStatus>,
}

impl EventHistoryStatusSet {
    /// Fold in whether the local event-search embedding artifact is installed. Uncovered work that
    /// no live projection is serving is `blocked` without the artifact; running and ready coverage
    /// are unaffected.
    pub fn with_search_artifact_available(mut self, available: bool) -> Self {
        if !available {
            for context in &mut self.contexts {
                if matches!(
                    context.search.state,
                    EventHistorySearchState::Pending | EventHistorySearchState::Failed
                ) {
                    context.search.state = EventHistorySearchState::Blocked;
                }
            }
        }
        self
    }
}

/// What each door supplies beyond ordinary rows: the Store-level placement and, where the door can
/// read it, the engine's delivery ledger.
pub(crate) trait EventHistoryStatusSource: EventSearchRows {
    fn placement(&self) -> Result<EventPlacementState>;
    fn tenant_delivery(&self, context_id: ContextId) -> Result<Option<EventDeliveryStatus>>;
}

impl EventHistoryStatusSource for Store {
    fn placement(&self) -> Result<EventPlacementState> {
        self.event_placement_state()
    }

    fn tenant_delivery(&self, context_id: ContextId) -> Result<Option<EventDeliveryStatus>> {
        self.event_delivery_status(context_id).map(Some)
    }
}

impl EventHistoryStatusSource for ConsumerReader {
    fn placement(&self) -> Result<EventPlacementState> {
        let answer = self.query_rows(
            "SELECT expected_policy FROM event_placement WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Int64(1))]),
        )?;
        match answer.rows.first().and_then(|row| row.first()) {
            Some(Value::Text(policy)) if policy == "local_only" => {
                Ok(EventPlacementState::LocalOnly)
            }
            // The reader open already refused an unbound tenant bootstrap, so a readable
            // tenant-hub Store is bound.
            Some(Value::Text(policy)) if policy == "tenant_hub" => {
                Ok(EventPlacementState::TenantHubBound)
            }
            other => Err(CgError::Engine(format!(
                "the event placement declaration is unreadable: {other:?}"
            ))),
        }
    }

    fn tenant_delivery(&self, _context_id: ContextId) -> Result<Option<EventDeliveryStatus>> {
        Ok(None)
    }
}

impl Store {
    /// One tagged event-history status snapshot per selected Context.
    pub fn event_history_status(&self, selector: ContextSelector) -> Result<EventHistoryStatusSet> {
        self.observe_read("event_history_status");
        event_history_status(self, selector)
    }
}

impl ConsumerReader {
    /// The same status fold as [`Store::event_history_status`], through the read-only session.
    pub fn event_history_status(&self, selector: ContextSelector) -> Result<EventHistoryStatusSet> {
        event_history_status(self, selector)
    }

    /// The Store-level placement alone, for a status caller that selected no Context.
    pub fn event_history_placement(&self) -> Result<(EventPlacementState, EventPlacementPolicy)> {
        let placement = self.placement()?;
        Ok((placement, expected_policy(placement)))
    }

    /// Resolve a caller-owned Context binding without writing the Store.
    pub fn resolve_context_binding(&self, key: &ContextBindingKey) -> Result<Option<ContextId>> {
        let answer = self.query_rows(
            "SELECT context_id FROM context_bindings WHERE namespace = $namespace AND opaque_reference = $opaque_reference",
            &HashMap::from([
                ("namespace".to_string(), Value::Text(key.namespace.clone())),
                (
                    "opaque_reference".to_string(),
                    Value::Text(key.opaque_reference.clone()),
                ),
            ]),
        )?;
        match answer.rows.first().and_then(|row| row.first()) {
            Some(Value::Uuid(id)) => Ok(Some(ContextId::from(*id))),
            None => Ok(None),
            other => Err(CgError::Engine(format!(
                "context binding lookup returned invalid row {other:?}"
            ))),
        }
    }
}

fn expected_policy(placement: EventPlacementState) -> EventPlacementPolicy {
    match placement {
        EventPlacementState::LocalOnly => EventPlacementPolicy::LocalOnly,
        EventPlacementState::TenantHubUnbound | EventPlacementState::TenantHubBound => {
            EventPlacementPolicy::TenantHub
        }
    }
}

pub(crate) fn event_history_status(
    source: &dyn EventHistoryStatusSource,
    selector: ContextSelector,
) -> Result<EventHistoryStatusSet> {
    let placement = source.placement()?;
    let selected = selector.into_context_ids();
    if placement == EventPlacementState::TenantHubUnbound {
        // An unbound tenant bootstrap holds no application data to count.
        return Ok(EventHistoryStatusSet {
            placement,
            expected_policy: expected_policy(placement),
            selected_context_ids: selected,
            contexts: Vec::new(),
        });
    }
    let names = context_names(source, &selected)?;
    let now = i64::try_from(Wallclock::now().0).unwrap_or(i64::MAX);
    let mut contexts = Vec::with_capacity(selected.len());
    for context_id in &selected {
        contexts.push(context_status(
            source,
            placement,
            *context_id,
            names[context_id].clone(),
            now,
        )?);
    }
    Ok(EventHistoryStatusSet {
        placement,
        expected_policy: expected_policy(placement),
        selected_context_ids: selected,
        contexts,
    })
}

pub(crate) fn context_names(
    source: &dyn EventSearchRows,
    selected: &[ContextId],
) -> Result<HashMap<ContextId, String>> {
    let mut names = HashMap::new();
    for context_id in selected {
        if !context_id.is_valid() {
            return Err(CgError::InvalidIdentity {
                context: "ContextId",
            });
        }
        let answer = source.query_rows(
            "SELECT id, name FROM contexts WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Uuid(context_id.as_uuid()))]),
        )?;
        let Some(row) = answer.rows.first() else {
            return Err(CgError::NotFound {
                id: Some(context_id.to_string()),
                missing: Some("context"),
            });
        };
        names.insert(*context_id, text(row, &answer.columns, "name")?);
    }
    Ok(names)
}

fn context_params(context_id: ContextId) -> HashMap<String, Value> {
    HashMap::from([("context_id".to_string(), Value::Uuid(context_id.as_uuid()))])
}

fn count(
    source: &dyn EventHistoryStatusSource,
    sql: &str,
    params: &HashMap<String, Value>,
) -> Result<u64> {
    let answer = source.query_rows(sql, params)?;
    let row = answer
        .rows
        .first()
        .ok_or_else(|| CgError::Engine("a status count returned no aggregate row".into()))?;
    match row.first() {
        Some(Value::Int64(value)) => u64::try_from(*value)
            .map_err(|_| CgError::Engine("a status count returned a negative total".into())),
        other => Err(CgError::Engine(format!(
            "a status count returned {other:?}"
        ))),
    }
}

fn context_status(
    source: &dyn EventHistoryStatusSource,
    placement: EventPlacementState,
    context_id: ContextId,
    context_name: String,
    now_ms: i64,
) -> Result<EventHistoryContextStatus> {
    let params = context_params(context_id);
    let memory = EventHistoryMemoryCounts {
        goals: count(
            source,
            "SELECT COUNT(*) AS n FROM intentions WHERE context_id = $context_id",
            &params,
        )?,
        decisions: count(
            source,
            "SELECT COUNT(*) AS n FROM decisions WHERE context_id = $context_id",
            &params,
        )?,
        observations: count(
            source,
            "SELECT COUNT(*) AS n FROM observations WHERE context_id = $context_id",
            &params,
        )?,
    };
    // The capture and coverage lines read the Context's count row; only a missing row recounts.
    let counts = captured_event_counts(source, &[context_id])?
        .remove(&context_id)
        .unwrap_or_default();
    let captured_events = counts.captured;
    let coverage = counts.coverage;

    let source_streams = source_streams(source, context_id)?;
    let sessions = source_streams
        .iter()
        .map(|stream| (&stream.connector_id, &stream.session_ref))
        .collect::<BTreeSet<_>>()
        .len() as u64;
    let coverage_gaps = count(
        source,
        "SELECT COUNT(*) AS n FROM adapter_coverage_gaps WHERE context_id = $context_id",
        &params,
    )?;

    let heads = source.query_rows(
        "SELECT connector_id, generation_id FROM event_interpretation_heads WHERE context_id = $context_id",
        &params,
    )?;
    let mut active_interpretations = heads
        .rows
        .iter()
        .map(|row| {
            Ok(EventHistoryInterpretationHead {
                connector_id: text(row, &heads.columns, "connector_id")?,
                generation_id: InterpretationGenerationId::from(uuid(
                    row,
                    &heads.columns,
                    "generation_id",
                )?),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    active_interpretations.sort_by(|left, right| left.connector_id.cmp(&right.connector_id));
    let failed_interpretations = source.query_rows(
        "SELECT sanitized_failure, started_at FROM event_interpretation_generations WHERE context_id = $context_id AND state = $state",
        &HashMap::from([
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("state".to_string(), Value::Text("failed".into())),
        ]),
    )?;
    let last_interpretation_failure = latest_failure(&failed_interpretations)?;

    let delivery = match placement {
        EventPlacementState::LocalOnly => Some(EventDeliveryStatus {
            placement,
            expected_policy: EventPlacementPolicy::LocalOnly,
            authoritative_policy: Some(EventPlacementPolicy::LocalOnly),
            authoritative_hub_id: None,
            state: EventDeliveryState::Disabled,
            eligible: 0,
            accepted: 0,
            equivalent: 0,
            pending: 0,
            refused: 0,
        }),
        _ => source.tenant_delivery(context_id)?,
    };

    let search = search_status(source, context_id, captured_events, now_ms)?;
    Ok(EventHistoryContextStatus {
        context_id,
        context_name,
        memory,
        captured_events,
        sessions,
        source_streams,
        coverage,
        coverage_gaps,
        active_interpretations,
        last_interpretation_failure,
        delivery,
        search,
    })
}

fn source_streams(
    source: &dyn EventHistoryStatusSource,
    context_id: ContextId,
) -> Result<Vec<EventSourceStreamStatus>> {
    let params = context_params(context_id);
    let groups = source.query_rows(
        "SELECT connector_id, source_instance_ref, COUNT(*) AS event_count FROM events WHERE context_id = $context_id GROUP BY connector_id, source_instance_ref",
        &params,
    )?;
    let mut streams = Vec::new();
    for row in &groups.rows {
        let connector_id = text(row, &groups.columns, "connector_id")?;
        let source_instance_ref = text(row, &groups.columns, "source_instance_ref")?;
        let mut latest_params = params.clone();
        latest_params.insert("connector".into(), Value::Text(connector_id.clone()));
        latest_params.insert("source".into(), Value::Text(source_instance_ref.clone()));
        let latest = source.query_rows(
            "SELECT session_ref, adapter_version, anchor_declaration_version, seq FROM events WHERE context_id = $context_id AND connector_id = $connector AND source_instance_ref = $source ORDER BY seq DESC LIMIT 1",
            &latest_params,
        )?;
        let newest = latest
            .rows
            .first()
            .ok_or_else(|| CgError::Engine("captured source group has no newest event".into()))?;
        streams.push(EventSourceStreamStatus {
            connector_id,
            source_instance_ref,
            session_ref: text(newest, &latest.columns, "session_ref")?,
            adapter_version: text(newest, &latest.columns, "adapter_version")?,
            anchor_declaration_version: text(
                newest,
                &latest.columns,
                "anchor_declaration_version",
            )?,
            event_count: unsigned(row, &groups.columns, "event_count")?,
            latest_event_seq: Some(unsigned(newest, &latest.columns, "seq")?),
        });
    }
    streams.sort_by(|left, right| {
        (&left.connector_id, &left.source_instance_ref)
            .cmp(&(&right.connector_id, &right.source_instance_ref))
    });
    Ok(streams)
}

fn search_status(
    source: &dyn EventHistoryStatusSource,
    context_id: ContextId,
    captured_events: u64,
    now_ms: i64,
) -> Result<EventHistorySearchStatus> {
    let params = context_params(context_id);
    let work_policy = super::event_search_projection::persisted_work_policy(source, context_id)?;
    let selected_sessions = super::event_search_projection::selected_captured_session_work(
        source,
        context_id,
        work_policy,
    )?;
    let head = source.query_rows(
        "SELECT generation_id, activated_at FROM event_search_heads WHERE context_id = $context_id",
        &params,
    )?;
    let active_generation_id = match head.rows.first() {
        Some(row) => Some(EventSearchGenerationId::from(uuid(
            row,
            &head.columns,
            "generation_id",
        )?)),
        None => None,
    };
    let mut covered_events = 0;
    let mut embedding_space_id = None;
    let mut renderer_id = None;
    let mut renderer_version = None;
    let mut profile = None;
    let mut pending_events = selected_sessions.selected_events();
    let mut outside_scope_events = captured_events.saturating_sub(pending_events);
    let mut outside_scope_indexed_events = 0;
    let mut no_searchable_text_events = 0;
    if let Some(generation_id) = active_generation_id {
        let generation = source.query_rows(
            "SELECT event_count, embedding_space_id, renderer_id, renderer_version, renderer_profile, state FROM event_search_generations WHERE id = $id AND context_id = $context_id",
            &HashMap::from([
                ("id".to_string(), Value::Uuid(generation_id.as_uuid())),
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ]),
        )?;
        let row = generation.rows.first().ok_or_else(|| {
            CgError::Engine(format!(
                "the active event-search head names missing generation {generation_id}"
            ))
        })?;
        let generation_state = text(row, &generation.columns, "state")?;
        if generation_state != "ready" && generation_state != "building" {
            return Err(CgError::Engine(format!(
                "the active event-search head names generation {generation_id}, which is neither ready nor building"
            )));
        }
        let coverage = super::event_search_projection::policy_coverage(
            source,
            context_id,
            generation_id,
            &selected_sessions,
            captured_events,
        )?;
        covered_events = coverage.covered_events;
        pending_events = coverage.pending_events;
        outside_scope_events = coverage.outside_scope_events;
        outside_scope_indexed_events = coverage.outside_scope_indexed_events;
        no_searchable_text_events = coverage.no_searchable_text_events;
        embedding_space_id = Some(text(row, &generation.columns, "embedding_space_id")?);
        renderer_id = Some(text(row, &generation.columns, "renderer_id")?);
        renderer_version = Some(text(row, &generation.columns, "renderer_version")?);
        profile = Some(
            match row.get(column(&generation.columns, "renderer_profile")?) {
                Some(Value::Json(value)) => {
                    serde_json::from_value(value.clone()).map_err(|error| {
                        CgError::Engine(format!("column renderer_profile: {error}"))
                    })?
                }
                other => {
                    return Err(CgError::Engine(format!(
                        "column renderer_profile: expected JSON, got {other:?}"
                    )));
                }
            },
        );
    }
    let live_projection_lease =
        super::event_search_projection::live_projection_lease(source, context_id, now_ms)?;
    let activated_at = head
        .rows
        .first()
        .map(|row| txid(row, &head.columns, "activated_at"))
        .transpose()?;
    // SELECT * reads generation metadata while allowing the attempt column to be absent.
    let generations = source.query_rows(
        "SELECT * FROM event_search_generations WHERE context_id = $context_id",
        &params,
    )?;
    let mut building = None;
    let mut latest_attempt: Option<(u64, EventSearchGenerationId, Option<String>)> = None;
    for row in &generations.rows {
        let id = EventSearchGenerationId::from(uuid(row, &generations.columns, "id")?);
        let state = text(row, &generations.columns, "state")?;
        if state == "retired" {
            continue;
        }
        let attempt = last_attempt_or_creation(row, &generations.columns)?;
        let failure = optional_text(row, &generations.columns, "sanitized_failure")?
            .or_else(|| (state == "failed").then(|| "projection failed".into()));
        let active = Some(id) == active_generation_id;
        // Activation supersedes other older attempts. An active generation may record its own
        // failure after activation (for example during vector maintenance); activation itself
        // clears prior failures. Its coverage remains Ready independently of that failure.
        if active || activated_at.is_none_or(|activated| attempt > activated) {
            let relevant_at = if active {
                attempt.max(activated_at.unwrap_or(0))
            } else {
                attempt
            };
            if latest_attempt
                .as_ref()
                .is_none_or(|(seen, seen_id, _)| (relevant_at, id) > (*seen, *seen_id))
            {
                latest_attempt = Some((relevant_at, id, failure));
            }
        }
        if !active
            && state == "building"
            && building
                .as_ref()
                .is_none_or(|(seen, seen_id, _)| (attempt, id) > (*seen, *seen_id))
        {
            let profile = match row.get(column(&generations.columns, "renderer_profile")?) {
                Some(Value::Json(value)) => {
                    serde_json::from_value(value.clone()).map_err(|error| {
                        CgError::Engine(format!("column renderer_profile: {error}"))
                    })?
                }
                other => {
                    return Err(CgError::Engine(format!(
                        "column renderer_profile: expected JSON, got {other:?}"
                    )));
                }
            };
            building = Some((attempt, id, profile));
        }
    }
    let last_failure = latest_attempt.and_then(|(_, _, failure)| failure);
    let (
        building_generation_id,
        building_profile,
        building_covered_events,
        building_pending_events,
    ) = match building {
        Some((_, generation_id, profile)) => {
            let coverage = super::event_search_projection::policy_coverage(
                source,
                context_id,
                generation_id,
                &selected_sessions,
                captured_events,
            )?;
            (
                Some(generation_id),
                Some(profile),
                coverage.covered_events,
                coverage.pending_events,
            )
        }
        None => (None, None, 0, 0),
    };

    let state = if pending_events == 0 {
        EventHistorySearchState::Ready
    } else if live_projection_lease {
        EventHistorySearchState::Running
    } else if last_failure.is_some() {
        EventHistorySearchState::Failed
    } else {
        EventHistorySearchState::Pending
    };
    Ok(EventHistorySearchStatus {
        state,
        work_policy,
        active_generation_id,
        embedding_space_id,
        renderer_id,
        renderer_version,
        profile,
        covered_events,
        pending_events,
        outside_scope_events,
        outside_scope_indexed_events,
        no_searchable_text_events,
        building_generation_id,
        building_profile,
        building_covered_events,
        building_pending_events,
        live_projection_lease,
        last_failure,
    })
}

fn last_attempt_or_creation(row: &[Value], columns: &[String]) -> Result<u64> {
    match row.get(column(columns, "last_attempt_at")?) {
        Some(Value::TxId(value)) => Ok(value.0),
        // A current-build generation can legitimately have no resumed attempt yet.
        Some(Value::Null) => txid(row, columns, "started_at"),
        other => Err(CgError::Engine(format!(
            "column last_attempt_at: expected txid|null, got {other:?}"
        ))),
    }
}

fn latest_failure(answer: &contextdb_engine::QueryResult) -> Result<Option<String>> {
    let mut latest: Option<(u64, Option<String>)> = None;
    for row in &answer.rows {
        let started = txid(row, &answer.columns, "started_at")?;
        if latest.as_ref().is_none_or(|(seen, _)| started > *seen) {
            latest = Some((
                started,
                optional_text(row, &answer.columns, "sanitized_failure")?,
            ));
        }
    }
    Ok(latest.map(|(_, failure)| failure.unwrap_or_else(|| "interpretation failed".into())))
}

fn column(columns: &[String], name: &str) -> Result<usize> {
    columns
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| CgError::Engine(format!("event-history status read is missing {name}")))
}

fn text(row: &[Value], columns: &[String], name: &str) -> Result<String> {
    match row.get(column(columns, name)?) {
        Some(Value::Text(value)) => Ok(value.clone()),
        other => Err(CgError::Engine(format!(
            "column {name}: expected text, got {other:?}"
        ))),
    }
}

fn optional_text(row: &[Value], columns: &[String], name: &str) -> Result<Option<String>> {
    match row.get(column(columns, name)?) {
        Some(Value::Text(value)) => Ok(Some(value.clone())),
        Some(Value::Null) | None => Ok(None),
        other => Err(CgError::Engine(format!(
            "column {name}: expected text|null, got {other:?}"
        ))),
    }
}

fn uuid(row: &[Value], columns: &[String], name: &str) -> Result<uuid::Uuid> {
    match row.get(column(columns, name)?) {
        Some(Value::Uuid(value)) => Ok(*value),
        other => Err(CgError::Engine(format!(
            "column {name}: expected UUID, got {other:?}"
        ))),
    }
}

fn unsigned(row: &[Value], columns: &[String], name: &str) -> Result<u64> {
    match row.get(column(columns, name)?) {
        Some(Value::Int64(value)) => u64::try_from(*value)
            .map_err(|_| CgError::Engine(format!("column {name}: negative integer"))),
        other => Err(CgError::Engine(format!(
            "column {name}: expected integer, got {other:?}"
        ))),
    }
}

fn txid(row: &[Value], columns: &[String], name: &str) -> Result<u64> {
    match row.get(column(columns, name)?) {
        Some(Value::TxId(value)) => Ok(value.0),
        other => Err(CgError::Engine(format!(
            "column {name}: expected txid, got {other:?}"
        ))),
    }
}

/// What a real import of one source would do, decided without writing anything.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "prefix", rename_all = "snake_case")]
pub enum EventSourcePrefixCheck {
    /// No committed progress is recorded for this locator; a real run reads the whole source.
    New { source_bytes: u64 },
    /// The committed prefix is intact and nothing follows it.
    Unchanged { committed_bytes: u64 },
    /// The committed prefix is intact and `pending_bytes` follow it.
    Appended {
        committed_bytes: u64,
        pending_bytes: u64,
    },
    /// The committed prefix no longer matches; a real run refuses this source.
    Diverged {
        committed_bytes: u64,
        reason: String,
    },
}

impl Store {
    /// Verify one source's committed prefix the way import does, writing nothing.
    pub fn check_event_source_prefix(
        &self,
        context_id: ContextId,
        declaration: &crate::EventAdapterDeclaration,
        source_locator_ref: &str,
        source: &mut dyn crate::SeekableEventSource,
    ) -> Result<EventSourcePrefixCheck> {
        check_event_source_prefix(self, context_id, declaration, source_locator_ref, source)
    }
}

impl ConsumerReader {
    /// [`Store::check_event_source_prefix`] through the read-only session.
    pub fn check_event_source_prefix(
        &self,
        context_id: ContextId,
        declaration: &crate::EventAdapterDeclaration,
        source_locator_ref: &str,
        source: &mut dyn crate::SeekableEventSource,
    ) -> Result<EventSourcePrefixCheck> {
        check_event_source_prefix(self, context_id, declaration, source_locator_ref, source)
    }
}

pub(crate) fn check_event_source_prefix(
    rows: &dyn EventSearchRows,
    context_id: ContextId,
    declaration: &crate::EventAdapterDeclaration,
    source_locator_ref: &str,
    source: &mut dyn crate::SeekableEventSource,
) -> Result<EventSourcePrefixCheck> {
    use std::io::SeekFrom;
    let io = |error: std::io::Error| CgError::Engine(error.to_string());
    let source_len = source.seek(SeekFrom::End(0)).map_err(io)?;
    let mut params = HashMap::from([
        ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
        (
            "connector_id".to_string(),
            Value::Text(declaration.connector_id.clone()),
        ),
        (
            "source_locator_ref".to_string(),
            Value::Text(source_locator_ref.to_string()),
        ),
    ]);
    let locator = rows.query_rows(
        "SELECT source_instance_ref, adapter_version, anchor_declaration_version FROM event_source_locators WHERE context_id = $context_id AND connector_id = $connector_id AND source_locator_ref = $source_locator_ref",
        &params,
    )?;
    let Some(row) = locator.rows.first() else {
        return Ok(EventSourcePrefixCheck::New {
            source_bytes: source_len,
        });
    };
    let source_instance_ref = text(row, &locator.columns, "source_instance_ref")?;
    let adapter_version = text(row, &locator.columns, "adapter_version")?;
    let anchor_version = text(row, &locator.columns, "anchor_declaration_version")?;
    params.insert(
        "source_instance_ref".to_string(),
        Value::Text(source_instance_ref),
    );
    let checkpoint = rows.query_rows(
        "SELECT committed_byte_offset, prefix_sha256 FROM event_source_checkpoints WHERE context_id = $context_id AND connector_id = $connector_id AND source_instance_ref = $source_instance_ref",
        &params,
    )?;
    let Some(row) = checkpoint.rows.first() else {
        return Ok(EventSourcePrefixCheck::Diverged {
            committed_bytes: 0,
            reason: "a recorded source locator has no committed checkpoint".into(),
        });
    };
    let committed_bytes = unsigned(row, &checkpoint.columns, "committed_byte_offset")?;
    let committed_sha256 = text(row, &checkpoint.columns, "prefix_sha256")?;
    if adapter_version != declaration.adapter_version
        || anchor_version != declaration.anchor_declaration_version
    {
        return Ok(EventSourcePrefixCheck::Diverged {
            committed_bytes,
            reason: format!(
                "the source was captured by adapter {adapter_version}/{anchor_version}, not {}/{}",
                declaration.adapter_version, declaration.anchor_declaration_version
            ),
        });
    }
    if source_len < committed_bytes {
        return Ok(EventSourcePrefixCheck::Diverged {
            committed_bytes,
            reason: format!(
                "the source is {source_len} bytes, shorter than its {committed_bytes}-byte committed prefix"
            ),
        });
    }
    let mut digest = <sha2::Sha256 as sha2::Digest>::new();
    super::event_import::hash_exact_prefix(source, committed_bytes, &mut digest)?;
    if super::event_import::hasher_hex(&digest) != committed_sha256 {
        return Ok(EventSourcePrefixCheck::Diverged {
            committed_bytes,
            reason: "the committed prefix no longer matches its recorded hash".into(),
        });
    }
    Ok(if source_len == committed_bytes {
        EventSourcePrefixCheck::Unchanged { committed_bytes }
    } else {
        EventSourcePrefixCheck::Appended {
            committed_bytes,
            pending_bytes: source_len - committed_bytes,
        }
    })
}
