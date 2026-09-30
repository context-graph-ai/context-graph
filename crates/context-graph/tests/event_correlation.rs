//! RED coverage for the scope-bound event-correlation resolver.
//!
//! The resolver must combine the borrowed in-flight batch with committed anchor
//! rows before it decides whether a projected value is unique. These cases use
//! the real Store and its immutable anchor rows; the query plugin only observes
//! the Store query so the payload boundary remains independently visible.

#![cfg(context_graph_internal_test_support)]

#[cfg(context_graph_internal_test_support)]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(context_graph_internal_test_support)]
use std::sync::{Arc, Mutex};
#[cfg(context_graph_internal_test_support)]
use std::time::Duration;

#[cfg(context_graph_internal_test_support)]
use context_graph::test_support::event_correlation::{
    EventCorrelationTestObserver, observe_public_store_resolver_for_test,
};
use context_graph::{
    AppendEventBatch, ContextId, CorrelationResolution, CreateContext, EventAdapterDeclaration,
    EventCorrelationResolver, EventCorrelationScope, EventCoverage, EventDraft, EventRenderRole,
    PriorRequestAnchorQuery, Store, StoreConfig,
};
#[cfg(context_graph_internal_test_support)]
use contextdb_core::read_contract::OwnerReadCancellation;
#[cfg(context_graph_internal_test_support)]
use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};
#[cfg(context_graph_internal_test_support)]
use contextdb_engine::read_session::{
    ReadKernelCancellationEvent, ReadKernelSource, ReadKernelSourceEvent, ReadKernelTestObserver,
};
#[cfg(context_graph_internal_test_support)]
use contextdb_parser::{BinOp, Cte, Expr, FromItem, MatchClause, SelectBody, Statement, parse};
#[cfg(context_graph_internal_test_support)]
use serde_json::Value as JsonValue;
use serde_json::json;
#[cfg(context_graph_internal_test_support)]
use sha2::{Digest, Sha256};

const CONNECTOR: &str = "provider-neutral-chat";
const SOURCE_INSTANCE: &str = "profile-17";
const SESSION: &str = "session-44";
const ANCHOR_VERSION: &str = "tool-anchors-v1";

#[cfg(context_graph_internal_test_support)]
#[derive(Default)]
struct QueryRecorder {
    enabled: AtomicBool,
    sql: Mutex<Vec<String>>,
}

#[cfg(context_graph_internal_test_support)]
impl QueryRecorder {
    fn enable_for_next_call(&self) {
        self.sql.lock().expect("query recorder lock").clear();
        self.enabled.store(true, Ordering::SeqCst);
    }

    fn queries(&self) -> Vec<String> {
        self.sql.lock().expect("query recorder lock").clone()
    }
}

#[cfg(context_graph_internal_test_support)]
impl DatabasePlugin for QueryRecorder {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if self.enabled.load(Ordering::SeqCst) {
            self.sql
                .lock()
                .expect("query recorder lock")
                .push(sql.to_string());
        }
        Ok(())
    }

    fn post_query(&self, _sql: &str, _duration: Duration, _outcome: &QueryOutcome) {}
}

#[cfg(context_graph_internal_test_support)]
#[derive(Default)]
struct KernelProbe {
    index_entries: AtomicU64,
    table_rows: AtomicU64,
    other_sources: AtomicU64,
    max_completed_items: AtomicU64,
}

#[cfg(context_graph_internal_test_support)]
impl ReadKernelTestObserver for KernelProbe {
    fn before_source_touch(
        &self,
        event: ReadKernelSourceEvent,
        _cancellation: OwnerReadCancellation,
    ) {
        match event.source {
            ReadKernelSource::IndexEntry => {
                self.index_entries.fetch_add(1, Ordering::SeqCst);
            }
            ReadKernelSource::TableRow => {
                self.table_rows.fetch_add(1, Ordering::SeqCst);
            }
            _ => {
                self.other_sources.fetch_add(1, Ordering::SeqCst);
            }
        }
        self.max_completed_items
            .fetch_max(event.completed_items, Ordering::SeqCst);
    }

    fn cancellation_observed(
        &self,
        _event: ReadKernelCancellationEvent,
        _cancellation: OwnerReadCancellation,
    ) {
    }
}

#[cfg(context_graph_internal_test_support)]
impl KernelProbe {
    fn reset(&self) {
        self.index_entries.store(0, Ordering::SeqCst);
        self.table_rows.store(0, Ordering::SeqCst);
        self.other_sources.store(0, Ordering::SeqCst);
        self.max_completed_items.store(0, Ordering::SeqCst);
    }
}

#[cfg(context_graph_internal_test_support)]
#[derive(Default)]
struct CorrelationProbe {
    pages: AtomicU64,
    continuation_pages: AtomicU64,
    pages_with_more: AtomicU64,
    max_page_rows: AtomicU64,
    candidate_observations: AtomicU64,
    max_distinct_values: AtomicU64,
    max_retained_json_bytes: AtomicU64,
    saw_positive_zero: AtomicBool,
    saw_negative_zero: AtomicBool,
    signed_zero_before_second_distinct: AtomicBool,
    saw_after_cutoff_value: AtomicBool,
}

#[cfg(context_graph_internal_test_support)]
impl EventCorrelationTestObserver for CorrelationProbe {
    fn cursor_page(&self, is_continuation: bool, rows: usize, has_more: bool) {
        self.pages.fetch_add(1, Ordering::SeqCst);
        if is_continuation {
            self.continuation_pages.fetch_add(1, Ordering::SeqCst);
        }
        if has_more {
            self.pages_with_more.fetch_add(1, Ordering::SeqCst);
        }
        self.max_page_rows.fetch_max(rows as u64, Ordering::SeqCst);
    }

    fn candidate_state(
        &self,
        candidate: &JsonValue,
        distinct_values: usize,
        retained_json_bytes: usize,
    ) {
        self.candidate_observations.fetch_add(1, Ordering::SeqCst);
        self.max_distinct_values
            .fetch_max(distinct_values as u64, Ordering::SeqCst);
        self.max_retained_json_bytes
            .fetch_max(retained_json_bytes as u64, Ordering::SeqCst);
        match serde_json::to_string(candidate)
            .expect("observed candidate serializes")
            .as_str()
        {
            "0.0" => self.saw_positive_zero.store(true, Ordering::SeqCst),
            "-0.0" => self.saw_negative_zero.store(true, Ordering::SeqCst),
            "1.0"
                if distinct_values == 2
                    && self.saw_positive_zero.load(Ordering::SeqCst)
                    && self.saw_negative_zero.load(Ordering::SeqCst) =>
            {
                self.signed_zero_before_second_distinct
                    .store(true, Ordering::SeqCst);
            }
            "2.0" => self.saw_after_cutoff_value.store(true, Ordering::SeqCst),
            _ => {}
        }
    }
}

#[cfg(context_graph_internal_test_support)]
impl CorrelationProbe {
    fn reset(&self) {
        self.pages.store(0, Ordering::SeqCst);
        self.continuation_pages.store(0, Ordering::SeqCst);
        self.pages_with_more.store(0, Ordering::SeqCst);
        self.max_page_rows.store(0, Ordering::SeqCst);
        self.candidate_observations.store(0, Ordering::SeqCst);
        self.max_distinct_values.store(0, Ordering::SeqCst);
        self.max_retained_json_bytes.store(0, Ordering::SeqCst);
        self.saw_positive_zero.store(false, Ordering::SeqCst);
        self.saw_negative_zero.store(false, Ordering::SeqCst);
        self.signed_zero_before_second_distinct
            .store(false, Ordering::SeqCst);
        self.saw_after_cutoff_value.store(false, Ordering::SeqCst);
    }
}

#[cfg(context_graph_internal_test_support)]
#[derive(Debug)]
struct DurableProbeSnapshot {
    history: u64,
    index_entries: u64,
    table_rows: u64,
    other_sources: u64,
    pages: u64,
    continuation_pages: u64,
    max_page_rows: u64,
    max_completed_items: u64,
    candidate_observations: u64,
    max_distinct_values: u64,
    max_retained_json_bytes: u64,
}

#[cfg(context_graph_internal_test_support)]
fn durable_snapshot(
    history: u64,
    kernel: &KernelProbe,
    correlation: &CorrelationProbe,
) -> DurableProbeSnapshot {
    DurableProbeSnapshot {
        history,
        index_entries: kernel.index_entries.load(Ordering::SeqCst),
        table_rows: kernel.table_rows.load(Ordering::SeqCst),
        other_sources: kernel.other_sources.load(Ordering::SeqCst),
        pages: correlation.pages.load(Ordering::SeqCst),
        continuation_pages: correlation.continuation_pages.load(Ordering::SeqCst),
        max_page_rows: correlation.max_page_rows.load(Ordering::SeqCst),
        max_completed_items: kernel.max_completed_items.load(Ordering::SeqCst),
        candidate_observations: correlation.candidate_observations.load(Ordering::SeqCst),
        max_distinct_values: correlation.max_distinct_values.load(Ordering::SeqCst),
        max_retained_json_bytes: correlation.max_retained_json_bytes.load(Ordering::SeqCst),
    }
}

fn declaration_with_anchor_keys(
    connector_id: &str,
    anchor_version: &str,
    interaction_anchor_key: &str,
    projected_anchor_key: &str,
) -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: connector_id.into(),
        adapter_version: "fixture-1".into(),
        anchor_declaration_version: anchor_version.into(),
        anchor_schema: json!({
            "type": "object",
            "properties": {
                interaction_anchor_key: { "type": "string" },
                projected_anchor_key: { "type": "string" }
            },
            "additionalProperties": false
        }),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn declaration(connector_id: &str, anchor_version: &str) -> EventAdapterDeclaration {
    declaration_with_anchor_keys(connector_id, anchor_version, "tool_use_id", "tool_name")
}

fn store() -> (tempfile::TempDir, Store) {
    let directory = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: directory.path().join("event-correlation.db"),
        ..StoreConfig::default()
    })
    .expect("open Store");
    (directory, store)
}

#[cfg(context_graph_internal_test_support)]
fn store_with_query_recorder() -> (tempfile::TempDir, Store, Arc<QueryRecorder>) {
    let directory = tempfile::tempdir().expect("tempdir");
    let recorder = Arc::new(QueryRecorder::default());
    let store = Store::open_with_database_plugin(
        StoreConfig {
            db_path: directory.path().join("event-correlation.db"),
            ..StoreConfig::default()
        },
        recorder.clone(),
    )
    .expect("open Store with query recorder");
    (directory, store, recorder)
}

fn context(store: &Store, name: &str) -> ContextId {
    store
        .create_context(CreateContext {
            name: name.into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id
}

fn tool_event(
    source_event_key: &str,
    ordinal: u64,
    role: EventRenderRole,
    interaction_ref: &str,
    tool_use_id: &str,
    tool_name: &str,
    payload_marker: &str,
) -> EventDraft {
    EventDraft {
        session_ref: SESSION.into(),
        source_event_key: source_event_key.into(),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "provider".into(),
        kind: "tool".into(),
        render_role: role,
        interaction_ref: Some(interaction_ref.into()),
        text: Some(format!("payload must not be needed: {payload_marker}")),
        payload: json!({"private_payload_marker": payload_marker}),
        anchors: json!({"tool_use_id": tool_use_id, "tool_name": tool_name}),
        source_remainder: json!({"private_remainder_marker": payload_marker}),
        coverage: EventCoverage::Complete,
    }
}

fn append(
    store: &Store,
    context_id: ContextId,
    declaration: EventAdapterDeclaration,
    source_instance_ref: &str,
    drafts: Vec<EventDraft>,
) {
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration,
            source_instance_ref,
            drafts,
        ))
        .expect("append event batch");
}

fn scope(
    context_id: ContextId,
    connector_id: &str,
    source_instance_ref: &str,
    session_ref: &str,
    anchor_declaration_version: &str,
) -> EventCorrelationScope {
    EventCorrelationScope {
        context_id,
        connector_id: connector_id.into(),
        source_instance_ref: source_instance_ref.into(),
        session_ref: session_ref.into(),
        anchor_declaration_version: anchor_declaration_version.into(),
    }
}

fn query(interaction_ref: &str) -> PriorRequestAnchorQuery {
    query_with_anchor_keys(interaction_ref, "tool_use_id", "tool_name")
}

fn query_with_anchor_keys(
    interaction_ref: &str,
    interaction_anchor_key: &str,
    projected_anchor_key: &str,
) -> PriorRequestAnchorQuery {
    PriorRequestAnchorQuery {
        interaction_ref: interaction_ref.into(),
        interaction_anchor_key: interaction_anchor_key.into(),
        projected_anchor_key: projected_anchor_key.into(),
    }
}

#[cfg(context_graph_internal_test_support)]
fn numeric_anchor_from_source(source: &str) -> JsonValue {
    let value = serde_json::from_str(source).expect("valid numeric anchor source");
    assert_eq!(
        serde_json::to_string(&value).expect("numeric anchor serializes"),
        source,
        "the fixture must retain its source numeric representation"
    );
    value
}

#[cfg(context_graph_internal_test_support)]
fn numeric_projected_request(
    source_event_key: &str,
    ordinal: u64,
    interaction_ref: &str,
    numeric_source: &str,
) -> EventDraft {
    let mut event = tool_event(
        source_event_key,
        ordinal,
        EventRenderRole::ToolRequest,
        interaction_ref,
        "unused-default-request-anchor",
        "unused-default-projected-anchor",
        source_event_key,
    );
    event.anchors = json!({"request_ref": interaction_ref});
    event
        .anchors
        .as_object_mut()
        .expect("numeric anchors remain an object")
        .insert(
            "numeric_projection".into(),
            numeric_anchor_from_source(numeric_source),
        );
    event
}

#[cfg(context_graph_internal_test_support)]
fn persisted_event_id(context_id: ContextId, source_event_key: &str) -> uuid::Uuid {
    let mut hasher = Sha256::new();
    hasher.update(b"context-graph/event/v1");
    for part in [
        context_id.as_uuid().as_bytes().as_slice(),
        CONNECTOR.as_bytes(),
        SESSION.as_bytes(),
        source_event_key.as_bytes(),
    ] {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    uuid::Uuid::from_bytes(bytes)
}

#[cfg(context_graph_internal_test_support)]
fn source_keys_in_persisted_index_order(context_id: ContextId, prefix: &str) -> Vec<String> {
    let mut keys = (0..32)
        .map(|index| format!("{prefix}-{index}"))
        .collect::<Vec<_>>();
    keys.sort_by_key(|key| persisted_event_id(context_id, key));
    keys
}

#[cfg(context_graph_internal_test_support)]
fn poison_event() -> EventDraft {
    const POISON_BYTES: usize = 20 * 1024 * 1024;
    let mut event = tool_event(
        "content-poison",
        0,
        EventRenderRole::ToolRequest,
        "unrelated-poison-interaction",
        "unrelated-poison-interaction",
        "Poison",
        "payload-present",
    );
    event.text = Some("x".repeat(POISON_BYTES));
    event.payload = json!({
        "private_payload_marker": "payload-present",
        "payload_bytes": POISON_BYTES,
    });
    event.source_remainder = json!({"private_remainder_marker": "remainder-present"});
    event
}

#[cfg(context_graph_internal_test_support)]
fn append_duplicate_history(
    store: &Store,
    context_id: ContextId,
    declaration: &EventAdapterDeclaration,
    history: u64,
) {
    for start in (0..history).step_by(256) {
        let end = (start + 256).min(history);
        let drafts = (start..end)
            .map(|ordinal| {
                tool_event(
                    &format!("duplicate-history-{ordinal}"),
                    ordinal + 1,
                    EventRenderRole::ToolRequest,
                    "duplicate-history-interaction",
                    "duplicate-history-interaction",
                    "Edit",
                    "duplicate-history-private-payload",
                )
            })
            .collect();
        append(
            store,
            context_id,
            declaration.clone(),
            SOURCE_INSTANCE,
            drafts,
        );
    }
}

#[cfg(context_graph_internal_test_support)]
fn run_durable_public_resolver_probe(history: u64) -> DurableProbeSnapshot {
    let (_directory, store) = store();
    let context_id = context(&store, "durable-correlation-probe");
    let declaration = declaration(CONNECTOR, ANCHOR_VERSION);
    append(
        &store,
        context_id,
        declaration.clone(),
        SOURCE_INSTANCE,
        vec![poison_event()],
    );
    append_duplicate_history(&store, context_id, &declaration, history);

    let kernel = Arc::new(KernelProbe::default());
    let correlation = Arc::new(CorrelationProbe::default());
    let observation = observe_public_store_resolver_for_test(kernel.clone(), correlation.clone());
    let pending: [EventDraft; 0] = [];
    let mut resolver = store.event_correlation_resolver(
        scope(
            context_id,
            CONNECTOR,
            SOURCE_INSTANCE,
            SESSION,
            ANCHOR_VERSION,
        ),
        &declaration,
        &pending,
    );
    assert_unique(
        resolver
            .resolve_prior_request_anchor(&query("duplicate-history-interaction"))
            .expect("actual public Store resolver accepts duplicate history"),
        "Edit",
    );
    drop(resolver);
    drop(observation);
    durable_snapshot(history, &kernel, &correlation)
}

#[cfg(context_graph_internal_test_support)]
#[derive(Debug)]
struct PhysicalScopeSnapshot {
    unrelated_anchors: u64,
    resolution: CorrelationResolution,
    index_entries: u64,
    table_rows: u64,
    other_sources: u64,
    pages: u64,
    continuation_pages: u64,
    max_page_rows: u64,
    candidate_observations: u64,
    max_distinct_values: u64,
    max_retained_json_bytes: u64,
}

#[cfg(context_graph_internal_test_support)]
fn physical_scope_declaration(connector_id: &str, anchor_version: &str) -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: connector_id.into(),
        adapter_version: "fixture-1".into(),
        anchor_declaration_version: anchor_version.into(),
        anchor_schema: json!({
            "type": "object",
            "properties": {
                "tool_use_id": { "type": "string" },
                "tool_name": { "type": "string" },
                "other_tool_use_id": { "type": "string" },
                "other_tool_name": { "type": "string" }
            },
            "additionalProperties": false
        }),
        source_position_order: vec!["/ordinal".into()],
    }
}

#[cfg(context_graph_internal_test_support)]
#[derive(Clone, Copy)]
enum ScopeDifference {
    Connector,
    Source,
    Session,
    Declaration,
    Role,
    Interaction,
    Key,
}

#[cfg(context_graph_internal_test_support)]
impl ScopeDifference {
    const ALL: [Self; 7] = [
        Self::Connector,
        Self::Source,
        Self::Session,
        Self::Declaration,
        Self::Role,
        Self::Interaction,
        Self::Key,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Connector => "connector",
            Self::Source => "source",
            Self::Session => "session",
            Self::Declaration => "declaration",
            Self::Role => "role",
            Self::Interaction => "interaction",
            Self::Key => "key",
        }
    }
}

#[cfg(context_graph_internal_test_support)]
fn append_unrelated_physical_scope_history(
    store: &Store,
    context_id: ContextId,
    events_per_scope_difference: u64,
) {
    const EVENTS_PER_BATCH: u64 = 125;
    for (difference_index, difference) in ScopeDifference::ALL.into_iter().enumerate() {
        let connector = match difference {
            ScopeDifference::Connector => "different-connector",
            _ => CONNECTOR,
        };
        let source = match difference {
            ScopeDifference::Source => "different-source",
            _ => SOURCE_INSTANCE,
        };
        let session = match difference {
            ScopeDifference::Session => "different-session",
            _ => SESSION,
        };
        let declaration_version = match difference {
            ScopeDifference::Declaration => "different-declaration",
            _ => ANCHOR_VERSION,
        };
        let role = match difference {
            ScopeDifference::Role => EventRenderRole::ToolResult,
            _ => EventRenderRole::ToolRequest,
        };
        let interaction = match difference {
            ScopeDifference::Interaction => "different-interaction",
            _ => "physical-scope-request",
        };
        let (interaction_key, projected_key) = match difference {
            ScopeDifference::Key => ("other_tool_use_id", "other_tool_name"),
            _ => ("tool_use_id", "tool_name"),
        };
        let declaration = physical_scope_declaration(connector, declaration_version);
        for batch_start in (0..events_per_scope_difference).step_by(EVENTS_PER_BATCH as usize) {
            let batch_end = (batch_start + EVENTS_PER_BATCH).min(events_per_scope_difference);
            let drafts = (batch_start..batch_end)
                .map(|ordinal| {
                    let mut draft = tool_event(
                        &format!("physical-noise-{}-{ordinal}", difference.name()),
                        ordinal + 1 + difference_index as u64 * events_per_scope_difference,
                        role,
                        interaction,
                        interaction,
                        "physical-noise",
                        "physical-noise",
                    );
                    draft.session_ref = session.into();
                    draft.anchors = json!({
                        interaction_key: interaction,
                        projected_key: format!("physical-noise-value-{}-{ordinal}", difference.name()),
                    });
                    draft
                })
                .collect();
            append(store, context_id, declaration.clone(), source, drafts);
        }
    }
}

#[cfg(context_graph_internal_test_support)]
fn run_physical_scope_probe(unrelated_anchors: u64) -> PhysicalScopeSnapshot {
    let (_directory, store, recorder) = store_with_query_recorder();
    let context_id = context(&store, "physical-correlation-scope");
    assert_eq!(
        unrelated_anchors % (ScopeDifference::ALL.len() as u64 * 2),
        0,
        "each valid one-dimension fixture event contributes exactly two anchors"
    );
    append_unrelated_physical_scope_history(
        &store,
        context_id,
        unrelated_anchors / (ScopeDifference::ALL.len() as u64 * 2),
    );
    let declaration = physical_scope_declaration(CONNECTOR, ANCHOR_VERSION);
    append(
        &store,
        context_id,
        declaration.clone(),
        SOURCE_INSTANCE,
        vec![tool_event(
            "physical-scope-true-request",
            10_000,
            EventRenderRole::ToolRequest,
            "physical-scope-request",
            "physical-scope-request",
            "Edit",
            "physical-scope-private-payload",
        )],
    );

    let kernel = Arc::new(KernelProbe::default());
    let correlation = Arc::new(CorrelationProbe::default());
    let observation = observe_public_store_resolver_for_test(kernel.clone(), correlation.clone());
    let pending: [EventDraft; 0] = [];
    let mut resolver = store.event_correlation_resolver(
        scope(
            context_id,
            CONNECTOR,
            SOURCE_INSTANCE,
            SESSION,
            ANCHOR_VERSION,
        ),
        &declaration,
        &pending,
    );
    kernel.reset();
    correlation.reset();
    recorder.enable_for_next_call();
    let resolution = resolver
        .resolve_prior_request_anchor(&query("physical-scope-request"))
        .expect("public Store resolver handles the one true request");
    let queries = recorder.queries();
    assert_scoped_metadata_correlation_queries(&queries, 1, 1);
    drop(resolver);
    drop(observation);
    PhysicalScopeSnapshot {
        unrelated_anchors,
        resolution,
        index_entries: kernel.index_entries.load(Ordering::SeqCst),
        table_rows: kernel.table_rows.load(Ordering::SeqCst),
        other_sources: kernel.other_sources.load(Ordering::SeqCst),
        pages: correlation.pages.load(Ordering::SeqCst),
        continuation_pages: correlation.continuation_pages.load(Ordering::SeqCst),
        max_page_rows: correlation.max_page_rows.load(Ordering::SeqCst),
        candidate_observations: correlation.candidate_observations.load(Ordering::SeqCst),
        max_distinct_values: correlation.max_distinct_values.load(Ordering::SeqCst),
        max_retained_json_bytes: correlation.max_retained_json_bytes.load(Ordering::SeqCst),
    }
}

#[cfg(context_graph_internal_test_support)]
fn run_same_event_projected_posting_probe() -> PhysicalScopeSnapshot {
    let (_directory, store) = store();
    let context_id = context(&store, "physical-correlation-posting-list");
    let declaration = EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "fixture-1".into(),
        anchor_declaration_version: ANCHOR_VERSION.into(),
        anchor_schema: json!({
            "type": "object",
            "properties": {
                "tool_use_id": { "type": "string" },
                "tool_name": { "type": "array", "items": { "type": "string" } }
            },
            "additionalProperties": false
        }),
        source_position_order: vec!["/ordinal".into()],
    };
    let source_keys = source_keys_in_persisted_index_order(context_id, "physical-scope-posting");
    let mut event = tool_event(
        &source_keys[0],
        1,
        EventRenderRole::ToolRequest,
        "physical-scope-request",
        "physical-scope-request",
        "unused",
        "physical-scope-private-payload",
    );
    event.anchors = json!({
        "tool_use_id": "physical-scope-request",
        "tool_name": ["Edit", "Delete"]
    });
    let mut later_same_scope_event = tool_event(
        &source_keys[1],
        2,
        EventRenderRole::ToolRequest,
        "physical-scope-request",
        "physical-scope-request",
        "unused",
        "same-scope-later-event-private-payload",
    );
    later_same_scope_event.anchors = json!({
        "tool_use_id": "physical-scope-request",
        "tool_name": ["MustNotReadAfterSameEventAmbiguity"]
    });
    let mut another_later_same_scope_event = tool_event(
        &source_keys[2],
        3,
        EventRenderRole::ToolRequest,
        "physical-scope-request",
        "physical-scope-request",
        "unused",
        "same-scope-later-event-private-payload-again",
    );
    another_later_same_scope_event.anchors = json!({
        "tool_use_id": "physical-scope-request",
        "tool_name": ["MustNotReadAfterSameEventAmbiguityAgain"]
    });
    append(
        &store,
        context_id,
        declaration.clone(),
        SOURCE_INSTANCE,
        vec![
            event,
            later_same_scope_event,
            another_later_same_scope_event,
        ],
    );

    let kernel = Arc::new(KernelProbe::default());
    let correlation = Arc::new(CorrelationProbe::default());
    let observation = observe_public_store_resolver_for_test(kernel.clone(), correlation.clone());
    let pending: [EventDraft; 0] = [];
    let mut resolver = store.event_correlation_resolver(
        scope(
            context_id,
            CONNECTOR,
            SOURCE_INSTANCE,
            SESSION,
            ANCHOR_VERSION,
        ),
        &declaration,
        &pending,
    );
    kernel.reset();
    correlation.reset();
    let resolution = resolver
        .resolve_prior_request_anchor(&query("physical-scope-request"))
        .expect("public Store resolver reads one event's projected posting list");
    drop(resolver);
    drop(observation);
    PhysicalScopeSnapshot {
        unrelated_anchors: 0,
        resolution,
        index_entries: kernel.index_entries.load(Ordering::SeqCst),
        table_rows: kernel.table_rows.load(Ordering::SeqCst),
        other_sources: kernel.other_sources.load(Ordering::SeqCst),
        pages: correlation.pages.load(Ordering::SeqCst),
        continuation_pages: correlation.continuation_pages.load(Ordering::SeqCst),
        max_page_rows: correlation.max_page_rows.load(Ordering::SeqCst),
        candidate_observations: correlation.candidate_observations.load(Ordering::SeqCst),
        max_distinct_values: correlation.max_distinct_values.load(Ordering::SeqCst),
        max_retained_json_bytes: correlation.max_retained_json_bytes.load(Ordering::SeqCst),
    }
}

fn assert_unique(resolution: CorrelationResolution, expected: &str) {
    assert_eq!(
        resolution,
        CorrelationResolution::Unique(json!(expected)),
        "only one distinct projected anchor value licenses an attribution"
    );
}

#[cfg(context_graph_internal_test_support)]
fn assert_scoped_metadata_correlation_queries(
    queries: &[String],
    expected_interaction_cursors: usize,
    expected_projected_cursors: usize,
) {
    let mut interaction_cursors = 0;
    let mut projected_cursors = 0;
    for sql in queries {
        match scoped_metadata_correlation_query(sql).unwrap_or_else(|reason| {
            panic!("correlation query violates its scoped metadata-only contract: {reason}\n{sql}")
        }) {
            CorrelationCursorKind::Interaction => interaction_cursors += 1,
            CorrelationCursorKind::Projected => projected_cursors += 1,
        }
    }
    assert_eq!(
        interaction_cursors, expected_interaction_cursors,
        "the resolver opens exactly the expected scoped interaction cursors"
    );
    assert_eq!(
        projected_cursors, expected_projected_cursors,
        "the resolver opens projected cursors only until its distinct-value cutoff"
    );
}

#[cfg(context_graph_internal_test_support)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CorrelationCursorKind {
    Interaction,
    Projected,
}

#[cfg(context_graph_internal_test_support)]
fn scoped_metadata_correlation_query(sql: &str) -> Result<CorrelationCursorKind, String> {
    let Statement::Select(select) = parse(sql).map_err(|error| error.to_string())? else {
        return Err("correlation must issue a SELECT query".into());
    };
    reject_content_reads_in_select_body(&select.body)?;
    for cte in &select.ctes {
        match cte {
            Cte::SqlCte { query, .. } => reject_content_reads_in_select_body(query)?,
            Cte::MatchCte { match_clause, .. } => {
                reject_content_reads_in_match_clause(match_clause)?;
            }
        }
    }
    if !select.ctes.is_empty() {
        return Err("correlation needs no auxiliary query".into());
    }
    if select.body.distinct || select.body.limit.is_some() || !select.body.order_by.is_empty() {
        return Err(
            "correlation cursor leaves uniqueness and termination to its bounded reader".into(),
        );
    }
    if !select.body.joins.is_empty() {
        return Err("correlation cursor reads anchor metadata without joining event rows".into());
    }
    if !matches!(select.body.from.as_slice(), [FromItem::Table { name, alias: None }] if name == "event_anchors")
    {
        return Err("correlation cursor reads exactly the immutable anchor table".into());
    }
    let projected_columns = [
        "event_id",
        "context_id",
        "connector_id",
        "source_instance_ref",
        "session_ref",
        "declaration_version",
        "render_role",
        "interaction_ref",
        "key",
        "original_value",
    ];
    if select.body.columns.len() != projected_columns.len()
        || !select
            .body
            .columns
            .iter()
            .map(|column| match &column.expr {
                Expr::Column(column) if column.table.is_none() => Some(column.column.as_str()),
                _ => None,
            })
            .eq(projected_columns.into_iter().map(Some))
    {
        return Err(
            "correlation cursor must project event identity and complete scope metadata before its JSON value"
                .into(),
        );
    }
    let kind = if sql.contains("event_id = $event_id") {
        CorrelationCursorKind::Projected
    } else {
        CorrelationCursorKind::Interaction
    };
    let predicates = scoped_cursor_predicates(
        select
            .body
            .where_clause
            .as_ref()
            .ok_or("correlation cursor has no scope predicates")?,
    )?;
    let mut expected = vec![
        ("correlation_scope_hash".to_owned(), "scope_hash".to_owned()),
        ("context_id".to_owned(), "context_id".to_owned()),
    ];
    if kind == CorrelationCursorKind::Projected {
        expected.push(("event_id".to_owned(), "event_id".to_owned()));
    }
    if predicates.len() != expected.len()
        || expected
            .iter()
            .any(|expected_predicate| !predicates.contains(expected_predicate))
    {
        return Err(format!(
            "correlation cursor lost or changed scoped predicate(s): {predicates:?}"
        ));
    }
    Ok(kind)
}

#[cfg(context_graph_internal_test_support)]
fn scoped_cursor_predicates(expr: &Expr) -> Result<Vec<(String, String)>, String> {
    match expr {
        Expr::BinaryOp {
            left,
            op: BinOp::And,
            right,
        } => {
            let mut predicates = scoped_cursor_predicates(left)?;
            predicates.extend(scoped_cursor_predicates(right)?);
            Ok(predicates)
        }
        Expr::BinaryOp {
            left,
            op: BinOp::Eq,
            right,
        } => match (left.as_ref(), right.as_ref()) {
            (Expr::Column(column), Expr::Parameter(parameter))
            | (Expr::Parameter(parameter), Expr::Column(column))
                if column.table.is_none() =>
            {
                Ok(vec![(column.column.clone(), parameter.clone())])
            }
            _ => Err(
                "scope predicate must compare one anchor column with its named parameter".into(),
            ),
        },
        _ => Err("scope predicate must be an AND of named equalities".into()),
    }
}

#[cfg(context_graph_internal_test_support)]
fn reject_content_reads_in_select_body(body: &SelectBody) -> Result<(), String> {
    for column in &body.columns {
        reject_content_reads_in_expr(&column.expr)?;
    }
    for from in &body.from {
        if let FromItem::GraphTable {
            match_clause,
            columns,
            ..
        } = from
        {
            reject_content_reads_in_match_clause(match_clause)?;
            for column in columns {
                reject_content_reads_in_expr(&column.expr)?;
            }
        }
    }
    for join in &body.joins {
        reject_content_reads_in_expr(&join.on)?;
    }
    if let Some(where_clause) = &body.where_clause {
        reject_content_reads_in_expr(where_clause)?;
    }
    for order in &body.order_by {
        reject_content_reads_in_expr(&order.expr)?;
    }
    if let Some(rank) = &body.use_rank {
        reject_content_column(rank)?;
    }
    Ok(())
}

#[cfg(context_graph_internal_test_support)]
fn reject_content_reads_in_match_clause(match_clause: &MatchClause) -> Result<(), String> {
    for (_, value) in &match_clause.pattern.start.properties {
        reject_content_reads_in_expr(value)?;
    }
    for edge in &match_clause.pattern.edges {
        for (_, value) in &edge.target.properties {
            reject_content_reads_in_expr(value)?;
        }
    }
    if let Some(where_clause) = &match_clause.where_clause {
        reject_content_reads_in_expr(where_clause)?;
    }
    for column in &match_clause.return_cols {
        reject_content_reads_in_expr(&column.expr)?;
    }
    Ok(())
}

#[cfg(context_graph_internal_test_support)]
fn reject_content_reads_in_expr(expr: &Expr) -> Result<(), String> {
    match expr {
        Expr::Column(column) => reject_content_column(&column.column),
        Expr::BinaryOp { left, right, .. } | Expr::CosineDistance { left, right } => {
            reject_content_reads_in_expr(left)?;
            reject_content_reads_in_expr(right)
        }
        Expr::UnaryOp { operand, .. } | Expr::IsNull { expr: operand, .. } => {
            reject_content_reads_in_expr(operand)
        }
        Expr::FunctionCall { args, .. } => {
            for argument in args {
                reject_content_reads_in_expr(argument)?;
            }
            Ok(())
        }
        Expr::Row(elements) => {
            for element in elements {
                reject_content_reads_in_expr(element)?;
            }
            Ok(())
        }
        Expr::RowVectorSource { column, key, .. } => {
            reject_content_column(column)?;
            reject_content_reads_in_expr(key)
        }
        Expr::InList { expr, list, .. } => {
            reject_content_reads_in_expr(expr)?;
            for item in list {
                reject_content_reads_in_expr(item)?;
            }
            Ok(())
        }
        Expr::Like { expr, pattern, .. } => {
            reject_content_reads_in_expr(expr)?;
            reject_content_reads_in_expr(pattern)
        }
        Expr::InSubquery { expr, subquery, .. } => {
            reject_content_reads_in_expr(expr)?;
            reject_content_reads_in_select_body(subquery)
        }
        Expr::Literal(_) | Expr::Parameter(_) => Ok(()),
    }
}

#[cfg(context_graph_internal_test_support)]
fn reject_content_column(column: &str) -> Result<(), String> {
    if column == "*"
        || [
            "text",
            "payload",
            "source_remainder",
            "source_bytes",
            "source_path",
            "path",
        ]
        .into_iter()
        .any(|content_column| column.eq_ignore_ascii_case(content_column))
    {
        Err(format!(
            "content or wildcard column `{column}` is forbidden"
        ))
    } else {
        Ok(())
    }
}

#[cfg(context_graph_internal_test_support)]
fn assert_metadata_only_correlation_oracle_self_checks() {
    let interaction = "SELECT event_id, context_id, connector_id, source_instance_ref, session_ref, declaration_version, render_role, interaction_ref, key, original_value FROM event_anchors WHERE correlation_scope_hash = $scope_hash AND context_id = $context_id";
    let projected = "SELECT event_id, context_id, connector_id, source_instance_ref, session_ref, declaration_version, render_role, interaction_ref, key, original_value FROM event_anchors WHERE correlation_scope_hash = $scope_hash AND context_id = $context_id AND event_id = $event_id";
    assert_scoped_metadata_correlation_queries(&[interaction.into(), projected.into()], 1, 1);

    for sql in [
        "SELECT * FROM event_anchors WHERE context_id = $context_id",
        "SELECT text FROM event_anchors WHERE context_id = $context_id",
        "SELECT original_value FROM event_anchors WHERE payload = $anchor_key",
        "SELECT original_value FROM event_anchors WHERE context_id = $context_id LIMIT 2",
        "SELECT original_value FROM event_anchors WHERE context_id = $context_id AND key = $anchor_key",
        "SELECT event_id, context_id, connector_id, source_instance_ref, session_ref, declaration_version, render_role, interaction_ref, key, original_value FROM event_anchors WHERE correlation_scope_hash = $scope_hash",
        "SELECT event_id, original_value FROM event_anchors WHERE correlation_scope_hash = $scope_hash AND context_id = $context_id",
    ] {
        assert!(
            scoped_metadata_correlation_query(sql).is_err(),
            "scoped metadata oracle accepted an incomplete or content-reading cursor: {sql}"
        );
    }
}

#[cfg(context_graph_internal_test_support)]
#[test]
fn committed_and_borrowed_pending_requests_resolve_before_and_after_reopen() {
    let (directory, store) = store();
    let context_id = context(&store, "correlation");
    let adapter_declaration = declaration(CONNECTOR, ANCHOR_VERSION);
    append(
        &store,
        context_id,
        adapter_declaration.clone(),
        SOURCE_INSTANCE,
        vec![tool_event(
            "committed-request",
            1,
            EventRenderRole::ToolRequest,
            "committed-call",
            "committed-call",
            "Edit",
            "committed-secret",
        )],
    );

    let pending = [tool_event(
        "pending-request",
        2,
        EventRenderRole::ToolRequest,
        "pending-call",
        "pending-call",
        "Search",
        "pending-secret",
    )];
    let bound_scope = scope(
        context_id,
        CONNECTOR,
        SOURCE_INSTANCE,
        SESSION,
        ANCHOR_VERSION,
    );
    let mut resolver =
        store.event_correlation_resolver(bound_scope, &adapter_declaration, &pending);
    assert_unique(
        resolver
            .resolve_prior_request_anchor(&query("committed-call"))
            .expect("resolve committed request"),
        "Edit",
    );
    assert_unique(
        resolver
            .resolve_prior_request_anchor(&query("pending-call"))
            .expect("resolve borrowed pending request"),
        "Search",
    );
    drop(resolver);
    drop(store);

    let reopened = Store::open(StoreConfig {
        db_path: directory.path().join("event-correlation.db"),
        ..StoreConfig::default()
    })
    .expect("reopen Store");
    let pending: [EventDraft; 0] = [];
    let mut resolver = reopened.event_correlation_resolver(
        scope(
            context_id,
            CONNECTOR,
            SOURCE_INSTANCE,
            SESSION,
            ANCHOR_VERSION,
        ),
        &adapter_declaration,
        &pending,
    );
    assert_unique(
        resolver
            .resolve_prior_request_anchor(&query("committed-call"))
            .expect("resolve committed request after reopen"),
        "Edit",
    );
    assert_eq!(
        resolver
            .resolve_prior_request_anchor(&query("pending-call"))
            .expect("resolve absent pending request after reopen"),
        CorrelationResolution::Missing
    );
}

#[cfg(context_graph_internal_test_support)]
#[test]
fn resolver_requires_the_entire_declared_scope_and_refuses_undeclared_anchor_keys() {
    let (_directory, store) = store();
    let context_id = context(&store, "bound-correlation");
    let other_context_id = context(&store, "other-context");
    let adapter_declaration = declaration(CONNECTOR, ANCHOR_VERSION);
    append(
        &store,
        context_id,
        adapter_declaration.clone(),
        SOURCE_INSTANCE,
        vec![tool_event(
            "scoped-request",
            1,
            EventRenderRole::ToolRequest,
            "scoped-call",
            "scoped-call",
            "Write",
            "scope-secret",
        )],
    );

    for (out_of_scope, scope_declaration) in [
        (
            scope(
                other_context_id,
                CONNECTOR,
                SOURCE_INSTANCE,
                SESSION,
                ANCHOR_VERSION,
            ),
            declaration(CONNECTOR, ANCHOR_VERSION),
        ),
        (
            scope(
                context_id,
                "other-provider",
                SOURCE_INSTANCE,
                SESSION,
                ANCHOR_VERSION,
            ),
            declaration("other-provider", ANCHOR_VERSION),
        ),
        (
            scope(
                context_id,
                CONNECTOR,
                "other-source-instance",
                SESSION,
                ANCHOR_VERSION,
            ),
            declaration(CONNECTOR, ANCHOR_VERSION),
        ),
        (
            scope(
                context_id,
                CONNECTOR,
                SOURCE_INSTANCE,
                "other-session",
                ANCHOR_VERSION,
            ),
            declaration(CONNECTOR, ANCHOR_VERSION),
        ),
        (
            scope(
                context_id,
                CONNECTOR,
                SOURCE_INSTANCE,
                SESSION,
                "tool-anchors-v2",
            ),
            declaration(CONNECTOR, "tool-anchors-v2"),
        ),
    ] {
        let pending: [EventDraft; 0] = [];
        let mut resolver =
            store.event_correlation_resolver(out_of_scope, &scope_declaration, &pending);
        assert_eq!(
            resolver
                .resolve_prior_request_anchor(&query("scoped-call"))
                .expect("out-of-scope lookup is a bounded absence"),
            CorrelationResolution::Missing
        );
    }

    let pending: [EventDraft; 0] = [];
    let mut resolver = store.event_correlation_resolver(
        scope(
            context_id,
            CONNECTOR,
            SOURCE_INSTANCE,
            SESSION,
            ANCHOR_VERSION,
        ),
        &adapter_declaration,
        &pending,
    );
    let malformed_query = PriorRequestAnchorQuery {
        interaction_ref: "scoped-call".into(),
        interaction_anchor_key: "undeclared-request-key".into(),
        projected_anchor_key: "tool_name".into(),
    };
    assert!(
        resolver
            .resolve_prior_request_anchor(&malformed_query)
            .is_err(),
        "the resolver must refuse a key outside the bound declaration"
    );
    let malformed_projected_query = PriorRequestAnchorQuery {
        interaction_ref: "scoped-call".into(),
        interaction_anchor_key: "tool_use_id".into(),
        projected_anchor_key: "undeclared-projected-key".into(),
    };
    assert!(
        resolver
            .resolve_prior_request_anchor(&malformed_projected_query)
            .is_err(),
        "the resolver must refuse a projected key outside the bound declaration"
    );

    let alternative_declaration = declaration_with_anchor_keys(
        CONNECTOR,
        "alternate-tool-anchors-v1",
        "request_token",
        "display_name",
    );
    let mut alternative_event = tool_event(
        "alternative-request",
        2,
        EventRenderRole::ToolRequest,
        "alternative-call",
        "unused-default-anchor",
        "unused-default-projection",
        "alternative-secret",
    );
    alternative_event.anchors = json!({
        "request_token": "alternative-call",
        "display_name": "Read"
    });
    append(
        &store,
        context_id,
        alternative_declaration.clone(),
        SOURCE_INSTANCE,
        vec![alternative_event],
    );
    let pending: [EventDraft; 0] = [];
    let mut alternative_resolver = store.event_correlation_resolver(
        scope(
            context_id,
            CONNECTOR,
            SOURCE_INSTANCE,
            SESSION,
            "alternate-tool-anchors-v1",
        ),
        &alternative_declaration,
        &pending,
    );
    assert_unique(
        alternative_resolver
            .resolve_prior_request_anchor(&query_with_anchor_keys(
                "alternative-call",
                "request_token",
                "display_name",
            ))
            .expect("resolve alternative declared anchor names"),
        "Read",
    );
}

#[cfg(context_graph_internal_test_support)]
#[test]
fn resolver_unions_distinct_committed_and_borrowed_values_before_deciding_uniqueness() {
    let (_directory, store) = store();
    let context_id = context(&store, "union-correlation");
    let adapter_declaration = declaration(CONNECTOR, ANCHOR_VERSION);
    append(
        &store,
        context_id,
        adapter_declaration.clone(),
        SOURCE_INSTANCE,
        vec![
            tool_event(
                "committed-same",
                1,
                EventRenderRole::ToolRequest,
                "same-value",
                "same-value",
                "Edit",
                "committed-same-secret",
            ),
            tool_event(
                "committed-pending-conflict",
                2,
                EventRenderRole::ToolRequest,
                "ambiguous-value",
                "ambiguous-value",
                "Edit",
                "committed-pending-conflict-secret",
            ),
            tool_event(
                "committed-ambiguous-first-duplicate",
                3,
                EventRenderRole::ToolRequest,
                "committed-duplicates-before-conflict",
                "committed-duplicates-before-conflict",
                "Edit",
                "committed-ambiguous-first-secret",
            ),
            tool_event(
                "committed-ambiguous-second-duplicate",
                4,
                EventRenderRole::ToolRequest,
                "committed-duplicates-before-conflict",
                "committed-duplicates-before-conflict",
                "Edit",
                "committed-ambiguous-second-secret",
            ),
            tool_event(
                "committed-ambiguous-conflict",
                5,
                EventRenderRole::ToolRequest,
                "committed-duplicates-before-conflict",
                "committed-duplicates-before-conflict",
                "Delete",
                "committed-ambiguous-conflict-secret",
            ),
        ],
    );
    let pending = [
        tool_event(
            "pending-same",
            6,
            EventRenderRole::ToolRequest,
            "same-value",
            "same-value",
            "Edit",
            "pending-same-secret",
        ),
        tool_event(
            "pending-ambiguous",
            7,
            EventRenderRole::ToolRequest,
            "ambiguous-value",
            "ambiguous-value",
            "Delete",
            "pending-ambiguous-secret",
        ),
        tool_event(
            "pending-wrong-session",
            8,
            EventRenderRole::ToolRequest,
            "pending-eligibility",
            "pending-eligibility",
            "WrongSessionMustNotWin",
            "pending-wrong-session-secret",
        ),
        tool_event(
            "pending-result-is-not-request",
            9,
            EventRenderRole::ToolResult,
            "pending-eligibility",
            "pending-eligibility",
            "ResultMustNotWin",
            "pending-result-secret",
        ),
        tool_event(
            "pending-matching-request",
            10,
            EventRenderRole::ToolRequest,
            "pending-eligibility",
            "pending-eligibility",
            "Inspect",
            "pending-matching-secret",
        ),
    ];
    let mut pending = pending;
    pending[2].session_ref = "other-session".into();
    let mut resolver = store.event_correlation_resolver(
        scope(
            context_id,
            CONNECTOR,
            SOURCE_INSTANCE,
            SESSION,
            ANCHOR_VERSION,
        ),
        &adapter_declaration,
        &pending,
    );
    assert_unique(
        resolver
            .resolve_prior_request_anchor(&query("same-value"))
            .expect("resolve duplicated projected value"),
        "Edit",
    );
    assert_eq!(
        resolver
            .resolve_prior_request_anchor(&query("ambiguous-value"))
            .expect("resolve conflicting projected values"),
        CorrelationResolution::Ambiguous,
        "a pending value cannot override a distinct committed value"
    );
    assert_eq!(
        resolver
            .resolve_prior_request_anchor(&query("committed-duplicates-before-conflict"))
            .expect("resolve duplicated committed values before their conflict"),
        CorrelationResolution::Ambiguous,
        "distinct values decide ambiguity before the two-row bound"
    );
    assert_unique(
        resolver
            .resolve_prior_request_anchor(&query("pending-eligibility"))
            .expect("resolve eligible pending request"),
        "Inspect",
    );
}

#[cfg(context_graph_internal_test_support)]
#[test]
fn numeric_projected_anchor_stays_ambiguous_before_and_after_the_batch_flush() {
    let (_directory, store, recorder) = store_with_query_recorder();
    let context_id = context(&store, "numeric-correlation");
    let adapter_declaration = EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "fixture-1".into(),
        anchor_declaration_version: "numeric-anchors-v1".into(),
        anchor_schema: json!({
            "type": "object",
            "properties": {
                "request_ref": { "type": "string" },
                "numeric_projection": { "type": "number" }
            },
            "additionalProperties": false
        }),
        source_position_order: vec!["/ordinal".into()],
    };
    let source_keys = source_keys_in_persisted_index_order(context_id, "numeric-cutoff");
    let pending = [
        numeric_projected_request(&source_keys[0], 1, "numeric-request", "0.0"),
        numeric_projected_request(&source_keys[1], 2, "numeric-request", "-0.0"),
        numeric_projected_request(&source_keys[2], 3, "numeric-request", "1.0"),
        numeric_projected_request(&source_keys[3], 4, "numeric-request", "2.0"),
    ];
    let bound_scope = scope(
        context_id,
        CONNECTOR,
        SOURCE_INSTANCE,
        SESSION,
        "numeric-anchors-v1",
    );
    let numeric_query =
        query_with_anchor_keys("numeric-request", "request_ref", "numeric_projection");

    let mut pending_resolver =
        store.event_correlation_resolver(bound_scope.clone(), &adapter_declaration, &pending);
    assert_eq!(
        pending_resolver
            .resolve_prior_request_anchor(&numeric_query)
            .expect("resolve the borrowed numeric requests"),
        CorrelationResolution::Ambiguous,
        "JsonValue equality collapses only the signed-zero pair before finding 1.0"
    );
    drop(pending_resolver);

    append(
        &store,
        context_id,
        adapter_declaration.clone(),
        SOURCE_INSTANCE,
        pending.to_vec(),
    );
    let committed_pending: [EventDraft; 0] = [];
    let kernel = Arc::new(KernelProbe::default());
    let correlation = Arc::new(CorrelationProbe::default());
    let observation = observe_public_store_resolver_for_test(kernel.clone(), correlation.clone());
    let mut committed_resolver =
        store.event_correlation_resolver(bound_scope, &adapter_declaration, &committed_pending);
    recorder.enable_for_next_call();
    let committed = committed_resolver
        .resolve_prior_request_anchor(&numeric_query)
        .expect("resolve the flushed numeric requests");
    let queries = recorder.queries();
    assert_scoped_metadata_correlation_queries(&queries, 1, 3);
    assert_eq!(
        committed,
        CorrelationResolution::Ambiguous,
        "the persisted 0.0, -0.0, and 1.0 anchors must use the same equality as pending drafts"
    );
    drop(committed_resolver);
    drop(observation);
    assert!(
        correlation.saw_positive_zero.load(Ordering::SeqCst)
            && correlation.saw_negative_zero.load(Ordering::SeqCst)
            && correlation
                .signed_zero_before_second_distinct
                .load(Ordering::SeqCst),
        "the actual public Store resolver must see both persisted signed zeros before 1.0 supplies the second distinct value"
    );
    assert!(
        !correlation.saw_after_cutoff_value.load(Ordering::SeqCst),
        "the actual public Store resolver must stop before the indexed 2.0 value after ambiguity"
    );
    assert_eq!(kernel.table_rows.load(Ordering::SeqCst), 0);
}

#[cfg(context_graph_internal_test_support)]
#[test]
fn public_store_resolver_keeps_cursor_pages_and_candidate_state_bounded_with_large_duplicate_history()
 {
    const SMALL_HISTORY: u64 = 4;
    const LARGE_HISTORY: u64 = 10_000;
    const POISON_TEXT_BYTES: u64 = 20 * 1024 * 1024;

    let small = run_durable_public_resolver_probe(SMALL_HISTORY);
    let large = run_durable_public_resolver_probe(LARGE_HISTORY);
    println!(
        "durable_public_resolver_probe pinned_engine=9f453bc9 history_small={} history_large={} poison_text_bytes={} small={small:?} large={large:?}",
        SMALL_HISTORY, LARGE_HISTORY, POISON_TEXT_BYTES
    );
    for snapshot in [&small, &large] {
        assert!(
            snapshot.index_entries >= snapshot.history * 2,
            "the actual cursor path must inspect its matching interaction and projected anchors"
        );
        assert_eq!(
            snapshot.table_rows, 0,
            "the 20 MiB event text is never read"
        );
        assert_eq!(
            snapshot.other_sources, 0,
            "only index entries reach the bounded kernel"
        );
        assert!(snapshot.pages >= snapshot.history * 2);
        assert!(snapshot.continuation_pages >= snapshot.history - 1);
        assert_eq!(snapshot.max_page_rows, 1);
        assert_eq!(
            snapshot.max_completed_items, 1,
            "the pinned engine reports progress within each physical one-row cursor page; history growth is proved by the page and index-entry counts above"
        );
        assert_eq!(snapshot.candidate_observations, snapshot.history);
        assert_eq!(snapshot.max_distinct_values, 1);
        assert_eq!(
            snapshot.max_retained_json_bytes, 6,
            "only JSON string `Edit` is retained"
        );
    }
    assert_eq!(
        small.max_retained_json_bytes, large.max_retained_json_bytes,
        "candidate memory stays constant as duplicate history grows"
    );
    assert!(
        large.index_entries > small.index_entries
            && large.pages > small.pages
            && large.continuation_pages > small.continuation_pages,
        "larger history produces more actual one-row cursor work without increasing retained candidates"
    );
}

#[cfg(context_graph_internal_test_support)]
#[test]
fn public_store_resolver_physically_scopes_unrelated_same_context_anchor_history() {
    const UNRELATED_ANCHORS: u64 = 10_010;

    let empty = run_physical_scope_probe(0);
    let noisy = run_physical_scope_probe(UNRELATED_ANCHORS);
    let posting = run_same_event_projected_posting_probe();
    println!(
        "physical_scope_public_resolver pinned_engine=9f453bc9 unrelated_anchors={} empty={empty:?} noisy={noisy:?} same_event_posting={posting:?}",
        UNRELATED_ANCHORS,
    );

    for snapshot in [&empty, &noisy] {
        assert_unique(snapshot.resolution.clone(), "Edit");
        assert_eq!(
            snapshot.index_entries, 2,
            "the actual resolver touches exactly one complete-scope interaction anchor and one projected anchor"
        );
        assert_eq!(
            snapshot.table_rows, 0,
            "correlation reads immutable index entries only"
        );
        assert_eq!(
            snapshot.other_sources, 0,
            "the bounded cursor reaches no other kernel source"
        );
        assert_eq!(
            snapshot.pages, 2,
            "one one-row page per cursor is sufficient"
        );
        assert_eq!(
            snapshot.continuation_pages, 0,
            "the scoped cursors need no continuation"
        );
        assert_eq!(
            snapshot.max_page_rows, 1,
            "resolver pages remain bounded to one row"
        );
        assert_eq!(snapshot.candidate_observations, 1);
        assert_eq!(snapshot.max_distinct_values, 1);
        assert_eq!(snapshot.max_retained_json_bytes, 6);
    }
    assert_eq!(
        empty.index_entries, noisy.index_entries,
        "unrelated same-Context anchors must not increase indexed correlation work"
    );
    assert_eq!(
        noisy.unrelated_anchors, UNRELATED_ANCHORS,
        "the fixture contains 10,010 valid same-Context noise anchors, each differing from the request in exactly one scoped dimension"
    );

    assert_eq!(posting.resolution, CorrelationResolution::Ambiguous);
    assert!(
        posting.index_entries <= 4,
        "the pinned bounded cursor may inspect the next index entry while stopping, but it must not walk later same-scope events; saw {} touches",
        posting.index_entries
    );
    assert_eq!(posting.table_rows, 0);
    assert_eq!(posting.other_sources, 0);
    assert_eq!(posting.pages, 3);
    assert_eq!(posting.continuation_pages, 1);
    assert_eq!(posting.max_page_rows, 1);
    assert_eq!(
        posting.candidate_observations, 2,
        "the two values posted by the first event establish ambiguity before later same-scope projected values are read"
    );
    assert_eq!(posting.max_distinct_values, 2);
    assert_eq!(
        posting.max_retained_json_bytes, 14,
        "the candidate accumulator retains only the two JSON values from the first event; this is not an allocator measurement"
    );
}

#[cfg(context_graph_internal_test_support)]
#[test]
fn resolver_uses_provider_neutral_request_and_interaction_keys_without_reading_payload_columns() {
    assert_metadata_only_correlation_oracle_self_checks();

    let (_directory, store, recorder) = store_with_query_recorder();
    let context_id = context(&store, "provider-neutral-correlation");
    let adapter_declaration = declaration(CONNECTOR, ANCHOR_VERSION);
    append(
        &store,
        context_id,
        adapter_declaration.clone(),
        SOURCE_INSTANCE,
        vec![
            tool_event(
                "result-is-not-request",
                1,
                EventRenderRole::ToolResult,
                "matched-interaction",
                "matched-interaction",
                "ResultValueMustNotWin",
                "tool-result-private-payload",
            ),
            tool_event(
                "wrong-interaction",
                2,
                EventRenderRole::ToolRequest,
                "other-interaction",
                "matched-interaction",
                "OtherInteractionMustNotWin",
                "other-interaction-private-payload",
            ),
            tool_event(
                "mismatched-interaction-anchor",
                3,
                EventRenderRole::ToolRequest,
                "matched-interaction",
                "different-anchor",
                "MismatchedAnchorMustNotWin",
                "mismatched-anchor-private-payload",
            ),
            tool_event(
                "matching-request",
                4,
                EventRenderRole::ToolRequest,
                "matched-interaction",
                "matched-interaction",
                "Inspect",
                "matching-request-private-payload",
            ),
        ],
    );

    let pending: [EventDraft; 0] = [];
    let mut resolver = store.event_correlation_resolver(
        scope(
            context_id,
            CONNECTOR,
            SOURCE_INSTANCE,
            SESSION,
            ANCHOR_VERSION,
        ),
        &adapter_declaration,
        &pending,
    );
    recorder.enable_for_next_call();
    assert_unique(
        resolver
            .resolve_prior_request_anchor(&query("matched-interaction"))
            .expect("resolve provider-neutral request"),
        "Inspect",
    );
    let queries = recorder.queries();
    assert_scoped_metadata_correlation_queries(&queries, 1, 1);
}
