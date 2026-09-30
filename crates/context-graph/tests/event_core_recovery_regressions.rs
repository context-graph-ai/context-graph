//! Event history recovery preserves captured data across interrupted work.
//!
//! These tests use the public Store surface except where an unbound tenant is
//! deliberately unreadable through the application gate. That state is checked
//! through ContextDB's read-only session, after every writable handle closes.

#![cfg(context_graph_internal_test_support)]

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

use chrono::{TimeZone, Utc};
use context_graph::{
    AppendEventBatch, CgError, ConsumerSchema, ConsumerTable, ContextBindingKey, CreateContext,
    CreateEntity, EntityType, EventAdapterDeclaration, EventCoverage, EventDraft,
    EventPlacementPolicy, EventPlacementState, EventRange, EventRenderRole, EventStreamSelector,
    Store, StoreConfig, SubscriptionError, SubscriptionFilter,
};
use contextdb_core::{Error as EngineError, Lsn, TenantId, Value};
use contextdb_engine::plugin::{DatabasePlugin, QueryOutcome};
use contextdb_engine::{MetadataBody, MetadataRequest, ReadSession};
use contextdb_server::sync_client::ApplicationTablePolicyExpectation;
use contextdb_server::{FabricIdentity, InProcessBroker, SyncClient, SyncServer};
use serde_json::json;
use tempfile::TempDir;

const CONNECTOR: &str = "core-recovery-regression";
const SESSION: &str = "core-recovery-session";
const NATURAL_KEY: &str = "same-natural-key";

fn store_config(root: &TempDir, placement: Option<EventPlacementPolicy>) -> StoreConfig {
    StoreConfig {
        db_path: root.path().join("core-recovery.db"),
        event_placement: placement,
        ..StoreConfig::default()
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "fixture-1".into(),
        anchor_declaration_version: "fixture-1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/order".into()],
    }
}

fn create_context(store: &Store) -> context_graph::ContextId {
    store
        .create_context(CreateContext {
            name: "recovery context".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id
}

fn draft(text: &str) -> EventDraft {
    EventDraft {
        session_ref: SESSION.into(),
        source_event_key: NATURAL_KEY.into(),
        source_position: json!({"order": 1}),
        occurred_at: Some(
            Utc.timestamp_opt(1_700_000_000, 0)
                .single()
                .expect("fixture timestamp"),
        ),
        author: "fixture".into(),
        kind: "fixture".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(text.into()),
        payload: json!({"text": text}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn rows(store: &Store, sql: &str, params: HashMap<String, Value>) -> Vec<Vec<Value>> {
    store
        .sync_database()
        .execute(sql, &params)
        .expect("inspect durable Store state")
        .rows
}

fn raw_rows(path: &std::path::Path, sql: &str, params: HashMap<String, Value>) -> Vec<Vec<Value>> {
    ReadSession::open(path)
        .expect("open raw read-only session")
        .execute(sql, &params)
        .expect("inspect committed image without application readiness")
        .rows
}

fn raw_table_names(path: &std::path::Path) -> BTreeSet<String> {
    let reader = ReadSession::open(path).expect("open raw read-only session");
    let mut tables = BTreeSet::new();
    let mut continuation = None;
    loop {
        let page = reader
            .metadata(MetadataRequest::Tables, continuation.as_deref())
            .expect("inspect raw table metadata");
        let MetadataBody::Tables { items, has_more } = page.body else {
            panic!("table metadata returned a different page kind");
        };
        tables.extend(items);
        if !has_more {
            assert!(page.continuation.is_none());
            return tables;
        }
        continuation = Some(
            page.continuation
                .expect("paged table metadata has a continuation"),
        );
    }
}

#[test]
fn allowed_pre_capture_tenant_transition_preserves_local_context_binding_and_graph_rows() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = store_config(&root, None).db_path;
    let store = Store::open(StoreConfig {
        db_path: path.clone(),
        ..StoreConfig::default()
    })
    .expect("open local Store");
    let context_id = create_context(&store);
    let binding = ContextBindingKey {
        namespace: "fixture".into(),
        opaque_reference: "pre-capture-workspace".into(),
    };
    store
        .bind_context(binding.clone(), context_id)
        .expect("persist node-local binding before capture");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "pre-capture graph row".into(),
            context_id,
            ..Default::default()
        })
        .expect("persist graph row before capture");
    let before_context = rows(
        &store,
        "SELECT id, name FROM contexts WHERE id = $id",
        HashMap::from([("id".into(), Value::Uuid(context_id.as_uuid()))]),
    );
    let before_binding = rows(
        &store,
        "SELECT namespace, opaque_reference, context_id FROM context_bindings WHERE namespace = $namespace AND opaque_reference = $opaque_reference",
        HashMap::from([
            ("namespace".into(), Value::Text(binding.namespace.clone())),
            (
                "opaque_reference".into(),
                Value::Text(binding.opaque_reference.clone()),
            ),
        ]),
    );
    let before_entity = rows(
        &store,
        "SELECT id, name, context_id FROM entities WHERE id = $id",
        HashMap::from([("id".into(), Value::Uuid(entity.id.as_uuid()))]),
    );
    drop(store);

    let tenant = Store::open(store_config(&root, Some(EventPlacementPolicy::TenantHub)))
        .expect("a local Store with no captured events may transition to TenantHub");
    assert_eq!(
        tenant
            .event_placement_state()
            .expect("inspect transition state"),
        EventPlacementState::TenantHubUnbound,
        "the transition changes only event placement until authenticated binding"
    );
    drop(tenant);

    let names = raw_table_names(&path);
    for table in ["contexts", "context_bindings", "entities"] {
        assert!(
            names.contains(table),
            "the allowed event-placement transition retains the pre-capture {table} schema"
        );
    }
    assert_eq!(
        raw_rows(
            &path,
            "SELECT id, name FROM contexts WHERE id = $id",
            HashMap::from([("id".into(), Value::Uuid(context_id.as_uuid()))]),
        ),
        before_context,
        "the Context row survives tenant bootstrap cleanup"
    );
    assert_eq!(
        raw_rows(
            &path,
            "SELECT namespace, opaque_reference, context_id FROM context_bindings WHERE namespace = $namespace AND opaque_reference = $opaque_reference",
            HashMap::from([
                ("namespace".into(), Value::Text(binding.namespace)),
                (
                    "opaque_reference".into(),
                    Value::Text(binding.opaque_reference)
                ),
            ]),
        ),
        before_binding,
        "the node-local binding survives tenant bootstrap cleanup"
    );
    assert_eq!(
        raw_rows(
            &path,
            "SELECT id, name, context_id FROM entities WHERE id = $id",
            HashMap::from([("id".into(), Value::Uuid(entity.id.as_uuid()))]),
        ),
        before_entity,
        "the graph row survives tenant bootstrap cleanup"
    );
}

struct StageSameNaturalKey {
    armed: AtomicBool,
    staged: AtomicUsize,
    event_id_primary_key_conflicts: AtomicUsize,
    barrier: Barrier,
}

impl StageSameNaturalKey {
    fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            staged: AtomicUsize::new(0),
            event_id_primary_key_conflicts: AtomicUsize::new(0),
            barrier: Barrier::new(2),
        }
    }

    fn arm(&self) {
        self.staged.store(0, Ordering::SeqCst);
        self.event_id_primary_key_conflicts
            .store(0, Ordering::SeqCst);
        self.armed.store(true, Ordering::SeqCst);
    }
}

impl DatabasePlugin for StageSameNaturalKey {
    fn post_query(&self, sql: &str, _elapsed: Duration, outcome: &QueryOutcome) {
        if !self.armed.load(Ordering::SeqCst)
            || !matches!(outcome, QueryOutcome::Success { .. })
            || !sql.trim_start().starts_with("INSERT INTO events (")
        {
            return;
        }
        if self.staged.fetch_add(1, Ordering::SeqCst) < 2 {
            self.barrier.wait();
        }
    }

    fn commit_failed(
        &self,
        _writes: &contextdb_tx::WriteSet,
        _source: contextdb_engine::plugin::CommitSource,
        error: &EngineError,
    ) {
        if matches!(
            error,
            EngineError::UniqueViolation { table, column } if table == "events" && column == "id"
        ) {
            self.event_id_primary_key_conflicts
                .fetch_add(1, Ordering::SeqCst);
        }
    }
}

fn race_same_natural_key(
    first_text: &'static str,
    second_text: &'static str,
) -> (
    context_graph::Result<context_graph::AppendEventReceipt>,
    context_graph::Result<context_graph::AppendEventReceipt>,
    Arc<StageSameNaturalKey>,
    TempDir,
    Store,
    context_graph::ContextId,
) {
    let root = tempfile::tempdir().expect("tempdir");
    let probe = Arc::new(StageSameNaturalKey::new());
    let store = Store::open_with_database_plugin(store_config(&root, None), probe.clone())
        .expect("open Store with deterministic staging callback");
    let context_id = create_context(&store);
    probe.arm();

    let first_store = store.clone();
    let first = thread::spawn(move || {
        first_store.append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "first-source-instance",
            vec![draft(first_text)],
        ))
    });
    let second_store = store.clone();
    let second = thread::spawn(move || {
        second_store.append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "second-source-instance",
            vec![draft(second_text)],
        ))
    });
    (
        first.join().expect("first append thread does not panic"),
        second.join().expect("second append thread does not panic"),
        probe,
        root,
        store,
        context_id,
    )
}

#[test]
fn staged_same_natural_key_appends_converge_or_diverge_after_the_real_event_id_collision() {
    let (first, second, probe, _root, store, context_id) =
        race_same_natural_key("same content", "same content");
    let first = first.expect("one same-content append completes");
    let second = second.expect("the raced same-content append becomes a no-op");
    assert_eq!(probe.staged.load(Ordering::SeqCst), 2);
    assert_eq!(
        probe.event_id_primary_key_conflicts.load(Ordering::SeqCst),
        1,
        "the deterministic race reaches the events.id primary-key collision, not a synthetic conflict"
    );
    assert_eq!(first.inserted + second.inserted, 1);
    assert_eq!(first.already_present + second.already_present, 1);
    assert_eq!(
        store
            .list_stream_events(
                EventStreamSelector {
                    context_id,
                    connector_id: CONNECTOR.into(),
                    source_instance_ref: "first-source-instance".into(),
                },
                EventRange::Count {
                    after_seq: None,
                    limit: std::num::NonZeroU32::new(8).expect("nonzero"),
                },
            )
            .expect("read one source stream")
            .len()
            + store
                .list_stream_events(
                    EventStreamSelector {
                        context_id,
                        connector_id: CONNECTOR.into(),
                        source_instance_ref: "second-source-instance".into(),
                    },
                    EventRange::Count {
                        after_seq: None,
                        limit: std::num::NonZeroU32::new(8).expect("nonzero"),
                    },
                )
                .expect("read the other source stream")
                .len(),
        1,
        "only one durable occurrence remains across the two staged source streams"
    );

    let (first, second, probe, _root, _store, _context_id) =
        race_same_natural_key("first content", "changed content");
    assert_eq!(probe.staged.load(Ordering::SeqCst), 2);
    assert_eq!(
        probe.event_id_primary_key_conflicts.load(Ordering::SeqCst),
        1
    );
    assert_eq!(first.is_ok() as usize + second.is_ok() as usize, 1);
    assert!(
        matches!(first, Err(CgError::SourceEventDiverged { .. }))
            || matches!(second, Err(CgError::SourceEventDiverged { .. })),
        "a changed payload under the same natural key is divergence after the retry rereads it"
    );
}

type EventBusMetadata = (
    Vec<(String, String, String)>,
    Vec<(String, String, bool, u64, u64, u64, u64, u64)>,
    Vec<(String, String, String)>,
);

fn event_bus_metadata(store: &Store) -> EventBusMetadata {
    let status = store.sync_database().event_bus_status();
    let mut event_types = status
        .event_types
        .into_iter()
        .map(|event| (event.name, event.trigger, event.table))
        .collect::<Vec<_>>();
    let mut sinks = status
        .sinks
        .into_iter()
        .map(|sink| {
            (
                sink.name,
                sink.sink_type,
                sink.callback_registered,
                sink.metrics.delivered,
                sink.metrics.queued,
                sink.metrics.retried,
                sink.metrics.permanent_failures,
                sink.metrics.examined,
            )
        })
        .collect::<Vec<_>>();
    let mut routes = status
        .routes
        .into_iter()
        .map(|route| (route.name, route.event_type, route.sink))
        .collect::<Vec<_>>();
    event_types.sort();
    sinks.sort();
    routes.sort();
    (event_types, sinks, routes)
}

#[test]
fn unbound_tenant_subscription_refuses_before_creating_a_sink_or_route() {
    let root = tempfile::tempdir().expect("tempdir");
    let store = Store::open(store_config(&root, Some(EventPlacementPolicy::TenantHub)))
        .expect("open unbound tenant bootstrap");
    let before = event_bus_metadata(&store);

    let result = store
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), |_| Ok(()));

    assert!(
        matches!(result, Err(SubscriptionError::Engine(message)) if message == "this import requires an authenticated tenant-hub binding; no Context binding, event schema, or event row was written. Finish tenant enrollment, then retry import"),
        "an unbound tenant has no application event schema and must refuse subscription readiness before DDL"
    );
    assert_eq!(
        event_bus_metadata(&store),
        before,
        "the rejected subscription leaves every event type, sink, route, callback flag, and metric unchanged"
    );
    assert!(
        store
            .sync_database()
            .changes_since(Lsn(0))
            .ddl
            .iter()
            .all(|ddl| !matches!(ddl, contextdb_engine::sync_types::DdlChange::CreateSink { name, .. } if name.starts_with("cg_subscription_new_decision_"))),
        "no sink DDL was recorded before the readiness refusal"
    );
}

const CONSUMER_TABLE: &str = "fixture_recovery_state";
const EVENT_POLICY: &str =
    "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE DELIVERY MANIFEST OVER event_anchors";
const ANCHOR_POLICY: &str = "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE";

fn consumer_schema() -> ConsumerSchema {
    ConsumerSchema {
        namespace: "fixture".into(),
        tables: vec![ConsumerTable {
            name: CONSUMER_TABLE.into(),
            ddl: format!(
                "CREATE TABLE {CONSUMER_TABLE} (id TEXT PRIMARY KEY, state TEXT NOT NULL)"
            ),
        }],
    }
}

fn drifted_consumer_schema() -> ConsumerSchema {
    ConsumerSchema {
        namespace: "fixture".into(),
        tables: vec![ConsumerTable {
            name: CONSUMER_TABLE.into(),
            ddl: format!(
                "CREATE TABLE {CONSUMER_TABLE} (id TEXT PRIMARY KEY, state TEXT NOT NULL, unexpected INTEGER NOT NULL)"
            ),
        }],
    }
}

struct PanicAfterConfiguredConsumerTable {
    armed: AtomicBool,
    interrupted_table: Mutex<Option<String>>,
}

impl PanicAfterConfiguredConsumerTable {
    fn arm(&self) {
        *self
            .interrupted_table
            .lock()
            .expect("interrupted table lock") = None;
        self.armed.store(true, Ordering::SeqCst);
    }
}

impl DatabasePlugin for PanicAfterConfiguredConsumerTable {
    fn post_query(&self, sql: &str, _elapsed: Duration, outcome: &QueryOutcome) {
        if !matches!(outcome, QueryOutcome::Success { .. })
            || !sql
                .trim_start()
                .starts_with(&format!("CREATE TABLE {CONSUMER_TABLE} "))
            || !self.armed.swap(false, Ordering::SeqCst)
        {
            return;
        }
        *self
            .interrupted_table
            .lock()
            .expect("interrupted table lock") = Some(CONSUMER_TABLE.into());
        panic!("interrupt after durable configured consumer table creation");
    }
}

struct AuthenticatedHub {
    broker: InProcessBroker,
    tenant: TenantId,
    stop: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<()>,
}

impl AuthenticatedHub {
    async fn stop(self) {
        self.stop.store(true, Ordering::SeqCst);
        self.task.await.expect("hub task stops");
    }
}

async fn authenticated_hub() -> AuthenticatedHub {
    let broker = InProcessBroker::new();
    let tenant = TenantId::from("consumer-recovery-tenant");
    let database = Arc::new(contextdb_engine::Database::open_memory());
    let identity = Arc::new(FabricIdentity::generate());
    let node_id = identity.node_id();
    let server = Arc::new(
        SyncServer::with_authenticated_transport_and_identity_for_test(
            database.clone(),
            broker.server_as(&node_id),
            tenant.clone(),
            node_id,
            identity,
        ),
    );
    for (table, policy) in [("events", EVENT_POLICY), ("event_anchors", ANCHOR_POLICY)] {
        database
            .execute(
                &format!("DECLARE TENANT TABLE POLICY {table} {policy}"),
                &HashMap::new(),
            )
            .expect("declare hub event policy");
    }
    let stop = Arc::new(AtomicBool::new(false));
    let task = tokio::spawn({
        let server = server.clone();
        let stop = stop.clone();
        async move { server.run_until(stop).await }
    });
    broker
        .wait_for_registered_route_for_test(&contextdb_server::subjects::binding_subject(
            tenant.as_str(),
        ))
        .await;
    AuthenticatedHub {
        broker,
        tenant,
        stop,
        task,
    }
}

fn policy_expectation() -> ApplicationTablePolicyExpectation {
    ApplicationTablePolicyExpectation::new()
        .expect_table("events", EVENT_POLICY)
        .expect("event policy grammar")
        .expect_table("event_anchors", ANCHOR_POLICY)
        .expect("anchor policy grammar")
}

async fn authenticated_binding(
    store: &Store,
    hub: &AuthenticatedHub,
) -> contextdb_engine::AuthenticatedTenantPolicyBinding {
    let identity = Arc::new(FabricIdentity::generate());
    let node_id = identity.node_id();
    let client = SyncClient::with_authenticated_transport_and_identity_for_test(
        store.sync_database(),
        hub.broker.client_as(&node_id),
        hub.tenant.clone(),
        identity,
    );
    let binding = client
        .bind_application_table_policy(policy_expectation())
        .await
        .expect("authenticate tenant policy binding");
    client.shutdown().await;
    binding
}

fn raw_query(path: &Path, sql: &str) -> contextdb_core::Result<Vec<Vec<Value>>> {
    ReadSession::open(path)?
        .execute(sql, &HashMap::new())
        .map(|result| result.rows)
}

#[tokio::test(flavor = "multi_thread")]
async fn interrupted_authenticated_install_recovers_configured_consumer_tables_without_claiming_foreign_rows()
 {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("configured-consumer-recovery.db");
    let config = StoreConfig {
        db_path: path.clone(),
        event_placement: Some(EventPlacementPolicy::TenantHub),
        ..StoreConfig::default()
    };
    let schema = consumer_schema();
    let probe = Arc::new(PanicAfterConfiguredConsumerTable {
        armed: AtomicBool::new(false),
        interrupted_table: Mutex::new(None),
    });
    let hub = authenticated_hub().await;
    let store = Arc::new(
        Store::open_with_consumer_schema_and_database_plugin(
            config.clone(),
            schema.clone(),
            probe.clone(),
        )
        .expect("open unbound tenant bootstrap with consumer declaration"),
    );
    store
        .sync_database()
        .execute(
            "CREATE TABLE foreign_recovery_marker (id TEXT PRIMARY KEY, note TEXT NOT NULL)",
            &HashMap::new(),
        )
        .expect("create an unrelated owner table before authenticated installation");
    store
        .sync_database()
        .execute(
            "INSERT INTO foreign_recovery_marker (id, note) VALUES ('outside-owner', 'retain')",
            &HashMap::new(),
        )
        .expect("persist unrelated owner row");
    let binding = authenticated_binding(&store, &hub).await;
    let binding_before = store
        .sync_database()
        .execute("SHOW SYNC BINDINGS", &HashMap::new())
        .expect("record durable authenticated binding")
        .rows;

    probe.arm();
    let interrupted_store = store.clone();
    let interrupted_binding = binding.clone();
    let interrupted = thread::spawn(move || {
        interrupted_store
            .bind_authenticated_tenant_policy(interrupted_binding)
            .expect("plugin panic interrupts authenticated installation");
    });
    assert!(
        interrupted.join().is_err(),
        "the installer thread is interrupted"
    );
    assert_eq!(
        probe
            .interrupted_table
            .lock()
            .expect("interrupted table lock")
            .as_deref(),
        Some(CONSUMER_TABLE),
        "the interruption is after the configured consumer table's successful DDL"
    );
    drop(store);

    let persisted_before_recovery = raw_table_names(&path);
    assert!(
        persisted_before_recovery.contains(CONSUMER_TABLE),
        "the configured consumer table exists in the real committed image before recovery"
    );
    assert!(
        persisted_before_recovery.contains("foreign_recovery_marker"),
        "the real committed image also contains the unrelated owner table"
    );
    assert_eq!(
        raw_query(&path, "SELECT id, note FROM foreign_recovery_marker")
            .expect("read unrelated durable row before recovery"),
        vec![vec![
            Value::Text("outside-owner".into()),
            Value::Text("retain".into())
        ]],
        "the boundary is a persisted interruption, not a synthetic partial schema"
    );
    assert_eq!(
        raw_query(&path, "SHOW SYNC BINDINGS").expect("read persisted binding before recovery"),
        binding_before,
        "authentication is already durable when installation is interrupted"
    );
    match raw_query(&path, "SELECT version FROM cg_schema_version") {
        Ok(rows) => assert!(
            rows.is_empty(),
            "partial installation has no published version row"
        ),
        Err(EngineError::TableNotFound(table)) if table == "cg_schema_version" => {}
        Err(error) => panic!("inspect interrupted readiness boundary: {error}"),
    }

    assert!(
        matches!(
            Store::open_with_consumer_schema(config.clone(), Vec::new(), drifted_consumer_schema()),
            Err(CgError::ConsumerTableDrifted { table, .. }) if table == CONSUMER_TABLE
        ),
        "a declaration that does not own the persisted consumer shape is refused before cleanup"
    );
    assert!(
        raw_table_names(&path).contains(CONSUMER_TABLE),
        "the refused unexpected declaration does not delete the configured consumer table"
    );

    let recovered = Store::open_with_consumer_schema(config.clone(), Vec::new(), schema)
        .expect("same consumer declaration recovers the unready bootstrap");
    assert_eq!(
        recovered
            .sync_database()
            .execute("SHOW SYNC BINDINGS", &HashMap::new())
            .expect("inspect retained binding after recovery")
            .rows,
        binding_before,
        "recovery retains the authenticated binding"
    );
    assert!(
        recovered
            .sync_database()
            .table_meta(CONSUMER_TABLE)
            .is_none(),
        "recovery removes only the installed consumer table from the interrupted installation"
    );
    assert_eq!(
        rows(
            &recovered,
            "SELECT id, note FROM foreign_recovery_marker",
            HashMap::new(),
        ),
        vec![vec![
            Value::Text("outside-owner".into()),
            Value::Text("retain".into())
        ]],
        "recovery retains unrelated owner rows"
    );

    recovered
        .bind_authenticated_tenant_policy(binding)
        .expect("saved binding retries the complete configured installation");
    assert!(
        recovered
            .sync_database()
            .table_meta(CONSUMER_TABLE)
            .is_some(),
        "retry reinstalls the configured consumer table"
    );
    create_context(&recovered);
    assert_eq!(
        rows(
            &recovered,
            "SELECT id, note FROM foreign_recovery_marker",
            HashMap::new(),
        ),
        vec![vec![
            Value::Text("outside-owner".into()),
            Value::Text("retain".into())
        ]],
        "the completed retry still retains unrelated owner rows"
    );
    drop(recovered);
    hub.stop().await;
}

struct RejectSecondInstallerTable {
    armed: AtomicBool,
    create_calls: AtomicUsize,
    rejected: AtomicBool,
}

impl RejectSecondInstallerTable {
    fn arm(&self) {
        self.create_calls.store(0, Ordering::SeqCst);
        self.rejected.store(false, Ordering::SeqCst);
        self.armed.store(true, Ordering::SeqCst);
    }
}

impl DatabasePlugin for RejectSecondInstallerTable {
    fn on_ddl(
        &self,
        change: &contextdb_engine::sync_types::DdlChange,
    ) -> contextdb_core::Result<()> {
        if !self.armed.load(Ordering::SeqCst)
            || !matches!(
                change,
                contextdb_engine::sync_types::DdlChange::CreateTable { .. }
            )
        {
            return Ok(());
        }
        let call = self.create_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == 2 {
            self.armed.store(false, Ordering::SeqCst);
            self.rejected.store(true, Ordering::SeqCst);
            return Err(EngineError::Other(
                "reject second authenticated installer table once".into(),
            ));
        }
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn preexisting_matching_consumer_row_survives_unbound_reopen_failed_install_and_retry() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("preexisting-consumer-recovery.db");
    let config = StoreConfig {
        db_path: path,
        event_placement: Some(EventPlacementPolicy::TenantHub),
        ..StoreConfig::default()
    };
    let schema = consumer_schema();
    let expected_row = vec![vec![
        Value::Text("preexisting-owner-row".into()),
        Value::Text("must-survive".into()),
    ]];
    let probe = Arc::new(RejectSecondInstallerTable {
        armed: AtomicBool::new(false),
        create_calls: AtomicUsize::new(0),
        rejected: AtomicBool::new(false),
    });

    let unbound = Store::open_with_consumer_schema_and_database_plugin(
        config.clone(),
        schema.clone(),
        probe.clone(),
    )
    .expect("open the unbound tenant bootstrap with the consumer declaration");
    unbound
        .sync_database()
        .execute(&schema.tables[0].ddl, &HashMap::new())
        .expect("pre-create the matching consumer table through the existing sync database API");
    unbound
        .sync_database()
        .execute(
            "INSERT INTO fixture_recovery_state (id, state) VALUES ('preexisting-owner-row', 'must-survive')",
            &HashMap::new(),
        )
        .expect("persist the pre-existing consumer row through the existing sync database API");
    drop(unbound);

    let reopened = Store::open_with_consumer_schema_and_database_plugin(
        config.clone(),
        schema.clone(),
        probe.clone(),
    )
    .expect("reopen the unbound tenant with the same consumer declaration");
    assert_eq!(
        rows(
            &reopened,
            "SELECT id, state FROM fixture_recovery_state",
            HashMap::new(),
        ),
        expected_row,
        "a matching consumer table and its prior row survive unbound bootstrap recovery"
    );

    let hub = authenticated_hub().await;
    let binding = authenticated_binding(&reopened, &hub).await;
    probe.arm();
    assert!(
        reopened
            .bind_authenticated_tenant_policy(binding.clone())
            .is_err(),
        "the real second authenticated installer CREATE TABLE is refused once"
    );
    assert!(probe.rejected.load(Ordering::SeqCst));
    assert_eq!(probe.create_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        rows(
            &reopened,
            "SELECT id, state FROM fixture_recovery_state",
            HashMap::new(),
        ),
        expected_row,
        "a failed authenticated installation does not claim the matching pre-existing consumer row"
    );
    drop(reopened);

    let recovered = Store::open_with_consumer_schema(config.clone(), Vec::new(), schema.clone())
        .expect("recover the unready tenant with the same consumer declaration");
    assert_eq!(
        rows(
            &recovered,
            "SELECT id, state FROM fixture_recovery_state",
            HashMap::new(),
        ),
        expected_row,
        "recovery after the failed installation retains the matching pre-existing row"
    );
    recovered
        .bind_authenticated_tenant_policy(binding)
        .expect("the saved binding retries the ordinary authenticated installation");
    assert_eq!(
        rows(
            &recovered,
            "SELECT id, state FROM fixture_recovery_state",
            HashMap::new(),
        ),
        expected_row,
        "the normal installer leaves the matching pre-existing consumer row intact"
    );
    drop(recovered);

    let completed = Store::open_with_consumer_schema(config, Vec::new(), schema)
        .expect("reopen the completed tenant with the same consumer declaration");
    assert_eq!(
        rows(
            &completed,
            "SELECT id, state FROM fixture_recovery_state",
            HashMap::new(),
        ),
        expected_row,
        "the completed schema reopen retains the matching pre-existing consumer row"
    );
    drop(completed);
    hub.stop().await;
}
