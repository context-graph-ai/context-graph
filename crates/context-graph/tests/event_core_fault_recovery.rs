#![cfg(context_graph_internal_test_support)]

use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, Cursor};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::Duration;

use chrono::{TimeZone, Utc};
use context_graph::{
    AdaptedSourceRecord, AppendEventBatch, CgError, ConsumerDeclaration, ConsumerReader,
    ConversationEvent, CreateContext, EndOfSource, EventAdapterDeclaration, EventCoverage,
    EventDraft, EventPlacementPolicy, EventRange, EventRenderRole, EventSourceAdapter,
    EventSourceIdentity, EventSourceRead, EventStreamSelector, ImportEventOptions,
    SourceRecordPosition, Store, StoreConfig,
};
use contextdb_core::{Error as EngineError, TenantId, Value};
use contextdb_engine::plugin::{CommitSource, DatabasePlugin, QueryOutcome};
use contextdb_engine::sync_types::DdlChange;
use contextdb_engine::{
    ApplicationTablePolicyExpectation, Database, MetadataBody, MetadataRequest, ReadSession,
};
use contextdb_server::{FabricIdentity, InProcessBroker, SyncClient, SyncServer};
use serde_json::json;
use tempfile::TempDir;

const EVENT_POLICY: &str =
    "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE DELIVERY MANIFEST OVER event_anchors";
const ANCHOR_POLICY: &str = "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE";
const TARGET_KEY: &str = "target-only-event";
const TARGET_LOCATOR: &str = "locator:target-only";
const SOURCE_INSTANCE: &str = "race-source-instance";
const SESSION: &str = "race-session";
const TARGET_BYTES: &[u8] = b"target record\n";

fn tenant_config(path: impl Into<PathBuf>) -> StoreConfig {
    StoreConfig {
        db_path: path.into(),
        event_placement: Some(EventPlacementPolicy::TenantHub),
        ..StoreConfig::default()
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: "fault-recovery-source".into(),
        adapter_version: "1".into(),
        anchor_declaration_version: "1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/order".into()],
    }
}

fn draft(key: &str, anchors: serde_json::Value) -> EventDraft {
    EventDraft {
        session_ref: SESSION.into(),
        source_event_key: key.into(),
        source_position: json!({"source_key": key, "order": 0}),
        occurred_at: Some(
            Utc.timestamp_opt(1_700_000_000, 0)
                .single()
                .expect("timestamp"),
        ),
        author: "fixture".into(),
        kind: "fixture".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(format!("event {key}")),
        payload: json!({"key": key}),
        anchors,
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn create_context(store: &Store, name: &str) -> context_graph::ContextId {
    store
        .create_context(CreateContext {
            name: name.into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id
}

fn stream(context_id: context_graph::ContextId) -> EventStreamSelector {
    EventStreamSelector {
        context_id,
        connector_id: declaration().connector_id,
        source_instance_ref: SOURCE_INSTANCE.into(),
    }
}

fn rows(store: &Store, sql: &str, params: HashMap<String, Value>) -> Vec<Vec<Value>> {
    store
        .sync_database()
        .execute(sql, &params)
        .expect("inspect durable state")
        .rows
}

fn table_fingerprint(db: &Database) -> BTreeSet<String> {
    db.table_names().into_iter().collect()
}

fn binding_fingerprint(db: &Database) -> Vec<Vec<Value>> {
    db.execute("SHOW SYNC BINDINGS", &HashMap::new())
        .expect("inspect authenticated bindings")
        .rows
}

fn version_rows(db: &Database) -> contextdb_core::Result<Vec<Vec<Value>>> {
    db.execute("SELECT version FROM cg_schema_version", &HashMap::new())
        .map(|result| result.rows)
}

fn persisted_table_names(path: &Path) -> BTreeSet<String> {
    let reader =
        ReadSession::open(path).expect("inspect interrupted committed image without recovery");
    let mut names = BTreeSet::new();
    let mut continuation = None;
    loop {
        let answer = reader
            .metadata(MetadataRequest::Tables, continuation.as_deref())
            .expect("read interrupted committed table inventory");
        let MetadataBody::Tables { items, has_more } = answer.body else {
            panic!("table inventory returned a different metadata body");
        };
        names.extend(items);
        if !has_more {
            assert!(
                answer.continuation.is_none(),
                "a complete table inventory has no continuation"
            );
            return names;
        }
        continuation = Some(
            answer
                .continuation
                .expect("a paged table inventory names its continuation"),
        );
    }
}

struct OneRecordSource {
    declaration: EventAdapterDeclaration,
    key: String,
    consumed_record: bool,
}

impl OneRecordSource {
    fn target() -> Self {
        Self {
            declaration: declaration(),
            key: TARGET_KEY.into(),
            consumed_record: false,
        }
    }
}

impl EventSourceAdapter for OneRecordSource {
    fn declaration(&self) -> &EventAdapterDeclaration {
        &self.declaration
    }

    fn identify_source(
        &mut self,
        reader: &mut dyn BufRead,
        _source_locator_ref: &str,
        _session_override: Option<&str>,
    ) -> context_graph::Result<EventSourceIdentity> {
        let mut identity_prefix = String::new();
        reader
            .read_line(&mut identity_prefix)
            .map_err(|error| CgError::Engine(error.to_string()))?;
        Ok(EventSourceIdentity {
            source_instance_ref: SOURCE_INSTANCE.into(),
            session_ref: SESSION.into(),
            provisional_empty: false,
        })
    }

    fn resume_source(
        &mut self,
        _identity: &EventSourceIdentity,
        _checkpoint: &serde_json::Value,
        _position: &SourceRecordPosition,
    ) -> context_graph::Result<()> {
        Ok(())
    }

    fn read_next_record(
        &mut self,
        reader: &mut dyn BufRead,
        _resolver: &mut dyn context_graph::EventCorrelationResolver,
    ) -> context_graph::Result<EventSourceRead> {
        let mut bytes = Vec::new();
        reader
            .read_until(b'\n', &mut bytes)
            .map_err(|error| CgError::Engine(error.to_string()))?;
        if bytes.is_empty() {
            return Ok(EventSourceRead::EndOfSource(EndOfSource::Complete));
        }
        assert!(!self.consumed_record, "fixture has exactly one record");
        self.consumed_record = true;
        Ok(EventSourceRead::Record(AdaptedSourceRecord {
            drafts: vec![draft(&self.key, json!({"target_anchor": "present"}))],
            checkpoint_after_record: json!({"target_checkpoint": 1}),
            source_position_after_record: SourceRecordPosition {
                byte_offset: bytes.len() as u64,
                coordinate: json!({"target_record_end": bytes.len()}),
            },
        }))
    }
}

struct ContentionProbe {
    planned_conflicts: usize,
    target_thread: Mutex<Option<ThreadId>>,
    arrived: mpsc::Sender<usize>,
    release: Mutex<mpsc::Receiver<()>>,
    observed_event_insert: AtomicUsize,
    observed_engine_conflict: AtomicUsize,
}

impl ContentionProbe {
    fn new(planned_conflicts: usize) -> (Arc<Self>, mpsc::Receiver<usize>, mpsc::Sender<()>) {
        let (arrived_tx, arrived_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        (
            Arc::new(Self {
                planned_conflicts,
                target_thread: Mutex::new(None),
                arrived: arrived_tx,
                release: Mutex::new(release_rx),
                observed_event_insert: AtomicUsize::new(0),
                observed_engine_conflict: AtomicUsize::new(0),
            }),
            arrived_rx,
            release_tx,
        )
    }

    fn target_current_thread(&self) {
        *self.target_thread.lock().expect("target thread lock") = Some(thread::current().id());
    }
}

impl DatabasePlugin for ContentionProbe {
    fn post_query(&self, sql: &str, _elapsed: Duration, outcome: &QueryOutcome) {
        let is_target = self
            .target_thread
            .lock()
            .expect("target thread lock")
            .is_some_and(|id| id == thread::current().id());
        if !is_target
            || !matches!(outcome, QueryOutcome::Success { .. })
            || !sql.trim_start().starts_with("INSERT INTO events (")
        {
            return;
        }
        let attempt = self.observed_event_insert.fetch_add(1, Ordering::SeqCst) + 1;
        if attempt <= self.planned_conflicts {
            self.arrived
                .send(attempt)
                .expect("controller receives target pause");
            self.release
                .lock()
                .expect("release lock")
                .recv()
                .expect("controller releases target attempt");
        }
    }

    fn commit_failed(
        &self,
        _ws: &contextdb_tx::WriteSet,
        _source: CommitSource,
        error: &EngineError,
    ) {
        if matches!(
            error,
            EngineError::UniqueViolation { table, column }
                if table == "events" && column == "context_id"
        ) {
            self.observed_engine_conflict.fetch_add(1, Ordering::SeqCst);
        }
    }
}

struct RaceObservation {
    result: context_graph::Result<context_graph::ImportEventReceipt>,
    observed_target_inserts: usize,
    observed_engine_conflicts: usize,
    stream_events: Vec<context_graph::ConversationEvent>,
    anchors: usize,
    local_commits: Vec<Vec<Value>>,
    checkpoint_txs: Vec<Value>,
    target_locator_rows: usize,
}

fn local_commit_txs_for_event(
    local_commits: &[Vec<Value>],
    event: &ConversationEvent,
) -> Vec<Value> {
    local_commits
        .iter()
        .filter_map(|row| match row.as_slice() {
            [Value::Uuid(event_id), committed_at] if *event_id == event.id.as_uuid() => {
                Some(committed_at.clone())
            }
            _ => None,
        })
        .collect()
}

fn assert_local_commit_membership(
    stream_events: &[ConversationEvent],
    local_commits: &[Vec<Value>],
) {
    assert_eq!(
        local_commits.len(),
        stream_events.len(),
        "every durable local commit belongs to one reread stream event, with no extra bookkeeping"
    );
    for event in stream_events {
        assert_eq!(
            local_commit_txs_for_event(local_commits, event).len(),
            1,
            "stream event {} has exactly one durable local commit",
            event.source_event_key
        );
    }
}

fn expected_race_keys(conflicts: usize, target_survives: bool) -> BTreeSet<String> {
    let mut keys = (1..=conflicts)
        .map(|attempt| format!("competitor-{attempt}"))
        .collect::<BTreeSet<_>>();
    if target_survives {
        keys.insert(TARGET_KEY.into());
    }
    keys
}

fn assert_exact_race_membership(
    stream_events: &[ConversationEvent],
    conflicts: usize,
    target_survives: bool,
) {
    let expected = expected_race_keys(conflicts, target_survives);
    let actual = stream_events
        .iter()
        .map(|event| event.source_event_key.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        stream_events.len(),
        expected.len(),
        "the stream has no duplicate entries"
    );
    assert_eq!(
        actual, expected,
        "the stream contains exactly the target and expected competitors"
    );
}

fn run_append_race(conflicts: usize) -> RaceObservation {
    let root = TempDir::new().expect("tempdir");
    let path = root.path().join("append-race.db");
    let (probe, arrived, release) = ContentionProbe::new(conflicts);
    let store = Arc::new(
        Store::open_with_database_plugin(
            StoreConfig {
                db_path: path,
                ..StoreConfig::default()
            },
            probe.clone(),
        )
        .expect("open store"),
    );
    let context_id = create_context(&store, "append race context");
    let target_store = store.clone();
    let target_probe = probe.clone();
    let target = thread::spawn(move || {
        target_probe.target_current_thread();
        let mut adapter = OneRecordSource::target();
        target_store.import_event_stream(
            &mut adapter,
            &mut Cursor::new(TARGET_BYTES.to_vec()),
            ImportEventOptions {
                context_id,
                source_locator_ref: TARGET_LOCATOR.into(),
                session_override: None,
                batch_limits: context_graph::EventBatchLimits {
                    max_events: NonZeroUsize::new(1).expect("nonzero"),
                    max_normalized_bytes: NonZeroUsize::new(1024).expect("nonzero"),
                },
            },
        )
    });

    for attempt in 1..=conflicts {
        assert_eq!(
            arrived
                .recv_timeout(Duration::from_secs(15))
                .expect("target reaches real event insert"),
            attempt,
            "the controller releases each target attempt only after its INSERT INTO events is staged"
        );
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                SOURCE_INSTANCE,
                vec![draft(&format!("competitor-{attempt}"), json!({}))],
            ))
            .expect("competitor commits the stream sequence while target is paused");
        release.send(()).expect("release target");
    }

    let result = target.join().expect("target thread does not panic");
    let stream_events = store
        .list_stream_events(
            stream(context_id),
            EventRange::Count {
                after_seq: None,
                limit: std::num::NonZeroU32::new(32).expect("nonzero"),
            },
        )
        .expect("read committed stream");
    let target_locator_rows = rows(
        &store,
        "SELECT source_locator_ref FROM event_source_locators WHERE context_id = $context_id AND source_locator_ref = $locator",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("locator".into(), Value::Text(TARGET_LOCATOR.into())),
        ]),
    )
    .len();
    let anchors = rows(&store, "SELECT id FROM event_anchors", HashMap::new()).len();
    let local_commits = rows(
        &store,
        "SELECT event_id, committed_at FROM event_local_commits WHERE context_id = $context_id",
        HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
    );
    let checkpoint_txs = rows(
        &store,
        "SELECT last_successful_tx FROM event_source_checkpoints WHERE context_id = $context_id AND source_instance_ref = $source_instance_ref",
        HashMap::from([
            ("context_id".into(), Value::Uuid(context_id.as_uuid())),
            ("source_instance_ref".into(), Value::Text(SOURCE_INSTANCE.into())),
        ]),
    )
    .into_iter()
    .flatten()
    .collect();

    RaceObservation {
        result,
        observed_target_inserts: probe.observed_event_insert.load(Ordering::SeqCst),
        observed_engine_conflicts: probe.observed_engine_conflict.load(Ordering::SeqCst),
        stream_events,
        anchors,
        local_commits,
        checkpoint_txs,
        target_locator_rows,
    }
}

#[test]
fn append_retries_real_stream_contention_then_refuses_the_fourth_without_a_partial_capture_unit() {
    let recovered = run_append_race(3);
    let receipt = recovered
        .result
        .expect("the fourth fresh transaction accepts the target source unit");
    assert_eq!(recovered.observed_target_inserts, 4);
    assert_eq!(recovered.observed_engine_conflicts, 3);
    assert_eq!(receipt.inserted, 1);
    assert_exact_race_membership(&recovered.stream_events, 3, true);
    assert_local_commit_membership(&recovered.stream_events, &recovered.local_commits);
    let target = recovered
        .stream_events
        .iter()
        .find(|event| event.source_event_key == TARGET_KEY)
        .expect("the accepted target is reread from the durable stream");
    assert_eq!(
        target.seq, 4,
        "the reread target uses the fresh fourth sequence"
    );
    let target_local_commit_txs = local_commit_txs_for_event(&recovered.local_commits, target);
    assert_eq!(
        target_local_commit_txs.len(),
        1,
        "the accepted target has exactly one durable local commit transaction"
    );
    assert_eq!(
        recovered.checkpoint_txs.len(),
        1,
        "the accepted target has exactly one durable source checkpoint transaction"
    );
    assert_eq!(
        target_local_commit_txs, recovered.checkpoint_txs,
        "the target local commit and checkpoint carry the same durable transaction value"
    );
    assert_eq!(recovered.anchors, 1, "the target anchor is committed once");
    assert_eq!(
        recovered.target_locator_rows, 1,
        "the target locator is committed once"
    );

    let exhausted = run_append_race(4);
    assert!(
        matches!(exhausted.result, Err(CgError::AppendContention { .. })),
        "the fourth real engine conflict is the typed append-contention outcome"
    );
    assert_eq!(exhausted.observed_target_inserts, 4);
    assert_eq!(exhausted.observed_engine_conflicts, 4);
    assert_exact_race_membership(&exhausted.stream_events, 4, false);
    assert_local_commit_membership(&exhausted.stream_events, &exhausted.local_commits);
    assert_eq!(exhausted.anchors, 0, "the target anchor did not leak");
    assert!(
        exhausted.checkpoint_txs.is_empty(),
        "the target checkpoint did not leak"
    );
    assert_eq!(
        exhausted.target_locator_rows, 0,
        "the target locator did not leak"
    );
}

struct RejectSecondCreateTable {
    armed: AtomicBool,
    create_calls: AtomicUsize,
    rejected: AtomicBool,
}

impl RejectSecondCreateTable {
    fn arm(&self) {
        self.create_calls.store(0, Ordering::SeqCst);
        self.rejected.store(false, Ordering::SeqCst);
        self.armed.store(true, Ordering::SeqCst);
    }
}

impl DatabasePlugin for RejectSecondCreateTable {
    fn on_ddl(&self, change: &DdlChange) -> contextdb_core::Result<()> {
        if !self.armed.load(Ordering::SeqCst) || !matches!(change, DdlChange::CreateTable { .. }) {
            return Ok(());
        }
        let call = self.create_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == 2 {
            self.armed.store(false, Ordering::SeqCst);
            self.rejected.store(true, Ordering::SeqCst);
            return Err(EngineError::Other(
                "reject second installer table once".into(),
            ));
        }
        Ok(())
    }
}

struct PanicAfterFirstInstallerDdl {
    armed: AtomicBool,
    interrupted_table: Mutex<Option<String>>,
}

impl PanicAfterFirstInstallerDdl {
    fn arm(&self) {
        *self
            .interrupted_table
            .lock()
            .expect("interrupted table lock") = None;
        self.armed.store(true, Ordering::SeqCst);
    }
}

impl DatabasePlugin for PanicAfterFirstInstallerDdl {
    fn post_query(&self, sql: &str, _elapsed: Duration, outcome: &QueryOutcome) {
        let Some(table) = sql
            .trim_start()
            .strip_prefix("CREATE TABLE ")
            .and_then(|ddl| ddl.split([' ', '(']).next())
        else {
            return;
        };
        if !matches!(outcome, QueryOutcome::Success { .. }) {
            return;
        }
        if self.armed.swap(false, Ordering::SeqCst) {
            *self
                .interrupted_table
                .lock()
                .expect("interrupted table lock") = Some(table.into());
            panic!("interrupt after first durable installer DDL");
        }
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
    let tenant = TenantId::from("fault-recovery-tenant");
    let database = Arc::new(Database::open_memory());
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
        .expect("authenticated policy binding");
    client.shutdown().await;
    binding
}

fn assert_bootstrap_or_unready(
    db: &Database,
    bootstrap_tables: &BTreeSet<String>,
    bindings: &[Vec<Value>],
) {
    assert_eq!(table_fingerprint(db), *bootstrap_tables);
    assert_eq!(binding_fingerprint(db), bindings);
    match version_rows(db) {
        Ok(rows) => assert!(
            rows.is_empty(),
            "a partial authenticated install never exposes a readiness version row"
        ),
        Err(EngineError::TableNotFound(table)) if table == "cg_schema_version" => {}
        Err(error) => panic!("inspect the authenticated schema version: {error}"),
    }
}

fn assert_complete_authenticated_install(store: &Store) {
    for table in context_graph::schema::TABLES {
        assert!(
            store.sync_database().table_meta(table.name).is_some(),
            "successful binding installs {}",
            table.name
        );
    }
    assert_eq!(
        version_rows(&store.sync_database())
            .expect("inspect completed authenticated schema version")
            .len(),
        1
    );
    assert_eq!(
        rows(
            store,
            "SELECT embedding_space_id FROM embedding_spaces",
            HashMap::new()
        )
        .len(),
        1,
        "successful binding seeds the default embedding space"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn ordinary_authenticated_installer_failure_cleans_to_bootstrap_and_retries_with_the_saved_binding()
 {
    let root = TempDir::new().expect("tempdir");
    let path = root.path().join("ordinary-install.db");
    let hub = authenticated_hub().await;
    let probe = Arc::new(RejectSecondCreateTable {
        armed: AtomicBool::new(false),
        create_calls: AtomicUsize::new(0),
        rejected: AtomicBool::new(false),
    });
    let store = Store::open_with_database_plugin(tenant_config(path.clone()), probe.clone())
        .expect("open unbound tenant bootstrap");
    let bootstrap_tables = table_fingerprint(&store.sync_database());
    let binding = authenticated_binding(&store, &hub).await;
    let binding_rows = binding_fingerprint(&store.sync_database());
    probe.arm();

    assert!(
        store
            .bind_authenticated_tenant_policy(binding.clone())
            .is_err(),
        "the second installer CREATE TABLE is refused by the real plugin"
    );
    assert!(probe.rejected.load(Ordering::SeqCst));
    assert_eq!(probe.create_calls.load(Ordering::SeqCst), 2);
    assert_bootstrap_or_unready(&store.sync_database(), &bootstrap_tables, &binding_rows);
    drop(store);

    let reopened =
        Store::open(tenant_config(path.clone())).expect("reopen cleans incomplete install");
    assert_bootstrap_or_unready(&reopened.sync_database(), &bootstrap_tables, &binding_rows);
    reopened
        .bind_authenticated_tenant_policy(binding)
        .expect("the preserved binding completes installation on retry");
    assert_complete_authenticated_install(&reopened);
    drop(reopened);
    hub.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn interrupted_authenticated_installer_preserves_readable_old_schema_until_retry() {
    let root = TempDir::new().expect("tempdir");
    let path = root.path().join("interrupted-install.db");
    let hub = authenticated_hub().await;
    let probe = Arc::new(PanicAfterFirstInstallerDdl {
        armed: AtomicBool::new(false),
        interrupted_table: Mutex::new(None),
    });
    let store = Arc::new(
        Store::open_with_database_plugin(tenant_config(path.clone()), probe.clone())
            .expect("open unbound tenant bootstrap"),
    );
    let bootstrap_tables = table_fingerprint(&store.sync_database());
    let binding = authenticated_binding(&store, &hub).await;
    let binding_rows = binding_fingerprint(&store.sync_database());
    probe.arm();
    let interrupted_store = store.clone();
    let interrupted_binding = binding.clone();
    let interrupted = thread::spawn(move || {
        interrupted_store
            .bind_authenticated_tenant_policy(interrupted_binding)
            .expect("plugin panic interrupts the owned installer thread");
    });
    assert!(
        interrupted.join().is_err(),
        "the owned installer thread was interrupted"
    );
    let interrupted_table = probe
        .interrupted_table
        .lock()
        .expect("interrupted table lock")
        .clone()
        .expect("the callback records the interrupted installer table");
    drop(store);

    let committed_tables = persisted_table_names(&path);
    assert_eq!(
        committed_tables, bootstrap_tables,
        "read-only metadata sees the old complete schema after the interrupted transaction"
    );
    assert!(
        !committed_tables.contains(&interrupted_table),
        "a callback before commit cannot publish its table"
    );

    let reopened = Store::open(tenant_config(path.clone()))
        .expect("the consistent old schema opens without schema recovery");
    assert_bootstrap_or_unready(&reopened.sync_database(), &bootstrap_tables, &binding_rows);
    reopened
        .bind_authenticated_tenant_policy(binding)
        .expect("saved binding installs the complete schema in one transaction on retry");
    assert_complete_authenticated_install(&reopened);
    drop(reopened);
    ConsumerReader::open(&path, ConsumerDeclaration::default())
        .expect("the consumer reader opens the completed authenticated schema");
    hub.stop().await;
}
