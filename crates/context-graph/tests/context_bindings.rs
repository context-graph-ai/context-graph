use std::collections::HashMap;
use std::sync::Arc;

use chrono::{TimeZone, Utc};
use context_graph::{
    AppendEventBatch, CgError, ContextBindingKey, ContextId, CreateContext,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventPlacementPolicy, EventPlacementState,
    EventRenderRole, Store, StoreConfig,
};
use contextdb_core::{SyncDirection, TenantId, Value, VersionedRow};
use contextdb_engine::Database;
use contextdb_server::sync_client::ApplicationTablePolicyExpectation;
use contextdb_server::{FabricIdentity, InProcessBroker, SyncClient, SyncServer};
use serde_json::json;
use tempfile::TempDir;

const EVENT_POLICY: &str =
    "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE DELIVERY MANIFEST OVER event_anchors";
const ANCHOR_POLICY: &str = "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE";

fn config(path: &std::path::Path, placement: Option<EventPlacementPolicy>) -> StoreConfig {
    StoreConfig {
        db_path: path.to_path_buf(),
        event_placement: placement,
        ..StoreConfig::default()
    }
}

fn key(reference: &str) -> ContextBindingKey {
    key_in("programmatic-test/v1", reference)
}

fn key_in(namespace: &str, reference: &str) -> ContextBindingKey {
    ContextBindingKey {
        namespace: namespace.into(),
        opaque_reference: reference.into(),
    }
}

fn create_input(name: &str) -> CreateContext {
    CreateContext {
        name: name.into(),
        labels: Vec::new(),
        properties: Default::default(),
    }
}

fn create_context(store: &Store, name: &str) -> ContextId {
    store
        .create_context(create_input(name))
        .expect("create fixture context")
        .id
}

fn binding_rows(store: &Store) -> Vec<Vec<Value>> {
    store
        .sync_database()
        .execute(
            "SELECT namespace, opaque_reference, context_id FROM context_bindings ORDER BY namespace, opaque_reference",
            &HashMap::new(),
        )
        .expect("read local binding rows")
        .rows
}

fn durable_rows(store: &Store, table: &str) -> Vec<VersionedRow> {
    let database = store.sync_database();
    let mut rows = database
        .scan(table, database.snapshot())
        .expect("scan durable rows");
    rows.sort_by_key(|row| row.row_id);
    rows
}

fn event_draft() -> EventDraft {
    EventDraft {
        session_ref: "session".into(),
        source_event_key: "event-1".into(),
        source_position: json!({"record": 1}),
        occurred_at: Some(
            Utc.timestamp_opt(1_700_000_000, 0)
                .single()
                .expect("timestamp"),
        ),
        author: "fixture".into(),
        kind: "fixture".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some("captured event".into()),
        payload: json!({"fixture": true}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

struct AuthenticatedHub {
    broker: InProcessBroker,
    tenant: TenantId,
    stop: Arc<std::sync::atomic::AtomicBool>,
    task: tokio::task::JoinHandle<()>,
}

impl AuthenticatedHub {
    async fn stop(self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        self.task.await.expect("hub task stops");
    }
}

async fn authenticated_hub() -> AuthenticatedHub {
    let broker = InProcessBroker::new();
    let tenant = TenantId::from("context-binding-tenant");
    let database = Arc::new(Database::open_memory());
    let identity = Arc::new(FabricIdentity::generate());
    let hub_node = identity.node_id();
    let server = Arc::new(
        SyncServer::with_authenticated_transport_and_identity_for_test(
            database.clone(),
            broker.server_as(&hub_node),
            tenant.clone(),
            hub_node,
            identity,
        )
        .expect("admit engine read memory"),
    );
    for (table, policy) in [("events", EVENT_POLICY), ("event_anchors", ANCHOR_POLICY)] {
        database
            .execute(
                &format!("DECLARE TENANT TABLE POLICY {table} {policy}"),
                &HashMap::new(),
            )
            .expect("hub declares event policy before enrollment");
    }
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
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

async fn bind_tenant_store(
    hub: &AuthenticatedHub,
    root: &TempDir,
    name: &str,
) -> (Store, SyncClient) {
    let store = Store::open(config(
        &root.path().join(format!("{name}.db")),
        Some(EventPlacementPolicy::TenantHub),
    ))
    .expect("open tenant Store");
    let identity = Arc::new(FabricIdentity::generate());
    let node_id = identity.node_id();
    let client = SyncClient::with_authenticated_transport_and_identity_for_test(
        store.sync_database(),
        hub.broker.client_as(&node_id),
        hub.tenant.clone(),
        identity,
    )
    .expect("admit engine read memory");
    let binding = client
        .bind_application_table_policy(policy_expectation())
        .await
        .expect("authenticated hub policy binding");
    let receipt = store
        .bind_authenticated_tenant_policy(binding)
        .expect("store installs only the authenticated binding");
    assert_eq!(receipt.state, EventPlacementState::TenantHubBound);
    (store, client)
}

#[test]
fn binding_is_absent_then_persists_across_reopen_and_same_context_is_a_noop() {
    let root = tempfile::tempdir().expect("tempdir");
    let path = root.path().join("context-graph.db");
    let store = Store::open(config(&path, None)).expect("open store");

    assert_eq!(
        store
            .resolve_context_binding(&key("opaque-machine-local-reference"))
            .expect("resolve absent binding"),
        None,
        "an absent opaque key has no implicit Context"
    );

    let context_id = create_context(&store, "bound context");
    let first = store
        .bind_context(key("opaque-machine-local-reference"), context_id)
        .expect("bind existing context");
    assert_eq!(first.context_id, context_id);
    assert!(first.changed, "the first binding writes the local row");

    let binding_versions_before_retry = durable_rows(&store, "context_bindings");
    let context_versions_before_retry = durable_rows(&store, "contexts");
    let committed_before_retry = store.sync_database().committed_watermark();

    let retry = store
        .bind_context(key("opaque-machine-local-reference"), context_id)
        .expect("repeat same binding");
    assert_eq!(retry.context_id, context_id);
    assert!(!retry.changed, "the same key and Context is idempotent");
    assert_eq!(
        durable_rows(&store, "context_bindings"),
        binding_versions_before_retry,
        "an idempotent retry preserves the complete committed binding row"
    );
    assert_eq!(
        durable_rows(&store, "contexts"),
        context_versions_before_retry,
        "an idempotent retry preserves the complete committed Context row"
    );
    assert_eq!(
        store.sync_database().committed_watermark(),
        committed_before_retry,
        "an idempotent retry creates no committed engine write"
    );
    drop(store);

    let reopened = Store::open(config(&path, None)).expect("reopen store");
    assert_eq!(
        reopened
            .resolve_context_binding(&key("opaque-machine-local-reference"))
            .expect("resolve persisted binding"),
        Some(context_id),
        "reopen retains the node-local opaque mapping"
    );
}

#[test]
fn conflicting_or_missing_context_binding_changes_neither_binding_nor_context_rows() {
    let root = tempfile::tempdir().expect("tempdir");
    let store =
        Store::open(config(&root.path().join("context-graph.db"), None)).expect("open store");
    let first_context = create_context(&store, "first context");
    let second_context = create_context(&store, "second context");
    store
        .bind_context(key("occupied-key"), first_context)
        .expect("seed binding");
    let before_binding = binding_rows(&store);
    let before_contexts = store.list_contexts().expect("list fixture contexts");

    assert!(matches!(
        store.bind_context(key("occupied-key"), second_context),
        Err(CgError::ContextBindingConflict { .. })
    ));
    assert_eq!(
        binding_rows(&store),
        before_binding,
        "a conflict cannot replace the key"
    );
    assert_eq!(
        store.list_contexts().expect("list after conflict"),
        before_contexts,
        "a binding conflict creates no Context"
    );

    assert!(
        store
            .bind_context(key("missing-context"), ContextId::new_v7())
            .is_err()
    );
    assert_eq!(
        binding_rows(&store),
        before_binding,
        "a missing Context creates no binding"
    );
    assert_eq!(
        store.list_contexts().expect("list after missing Context"),
        before_contexts,
        "binding only attaches an existing Context; it never creates one"
    );
}

#[test]
fn create_and_bind_commits_success_and_rolls_back_on_a_real_second_write_constraint() {
    let root = tempfile::tempdir().expect("tempdir");
    let store =
        Store::open(config(&root.path().join("context-graph.db"), None)).expect("open store");
    let created = store
        .create_context_with_binding(create_input("durably created"), key("created-key"))
        .expect("create and bind new Context");
    let created_id = created.context_id;
    drop(store);
    let store = Store::open(config(&root.path().join("context-graph.db"), None))
        .expect("reopen successful create-and-bind");
    assert_eq!(
        store
            .resolve_context_binding(&key("created-key"))
            .expect("resolve successful binding"),
        Some(created_id),
        "a successful create-and-bind durably records the binding"
    );
    assert_eq!(
        store
            .get_context(created_id)
            .expect("read created Context")
            .map(|context| context.name),
        Some("durably created".into()),
        "a successful create-and-bind durably records the Context"
    );

    store
        .sync_database()
        .execute(
            "ALTER TABLE context_bindings ADD COLUMN transaction_failure_probe TEXT NOT NULL",
            &HashMap::new(),
        )
        .expect("install a real engine constraint that rejects the binding write");
    let bindings_before_failure = binding_rows(&store);
    let contexts_before_failure = store
        .list_contexts()
        .expect("list before transaction failure");

    assert!(
        store
            .create_context_with_binding(create_input("would-be orphan"), key("failing-key"))
            .is_err(),
        "the binding insert must fail its declared NOT NULL constraint"
    );
    assert_eq!(
        binding_rows(&store),
        bindings_before_failure,
        "a failed binding write leaves no new binding row"
    );
    assert_eq!(
        store
            .list_contexts()
            .expect("list after transaction failure"),
        contexts_before_failure,
        "the failed second write rolls back its new Context instead of leaving an orphan"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn opaque_binding_keys_are_full_node_local_pairs_in_local_and_tenant_bound_stores() {
    let root = TempDir::new().expect("tempdir");
    let path = root.path().join("context-graph.db");
    let store = Store::open(config(&path, None)).expect("open local store");
    let first_context = create_context(&store, "first local binding context");
    let second_context = create_context(&store, "second local binding context");
    let shared_reference = "opaque:machine-local:7ad0af";
    let first_key = key_in("programmatic-test/v1", shared_reference);
    let second_namespace_key = key_in("another-caller/v1", shared_reference);
    let second_key_for_first_context = key_in("programmatic-test/v1", "opaque:second-key");

    store
        .bind_context(first_key.clone(), first_context)
        .expect("persist first opaque caller key");
    store
        .bind_context(second_namespace_key.clone(), second_context)
        .expect("same reference in a different caller namespace is independent");
    store
        .bind_context(second_key_for_first_context.clone(), first_context)
        .expect("a Context accepts multiple distinct opaque keys");
    assert_eq!(
        binding_rows(&store),
        vec![
            vec![
                Value::Text("another-caller/v1".into()),
                Value::Text(shared_reference.into()),
                Value::Uuid(second_context.as_uuid()),
            ],
            vec![
                Value::Text("programmatic-test/v1".into()),
                Value::Text(shared_reference.into()),
                Value::Uuid(first_context.as_uuid()),
            ],
            vec![
                Value::Text("programmatic-test/v1".into()),
                Value::Text("opaque:second-key".into()),
                Value::Uuid(first_context.as_uuid()),
            ],
        ],
        "the core records opaque namespace/reference pairs without interpreting either value"
    );
    drop(store);

    let reopened = Store::open(config(&path, None)).expect("reopen local store");
    for (binding_key, context_id) in [
        (&first_key, first_context),
        (&second_namespace_key, second_context),
        (&second_key_for_first_context, first_context),
    ] {
        assert_eq!(
            reopened
                .resolve_context_binding(binding_key)
                .expect("resolve persisted opaque pair"),
            Some(context_id),
            "each namespace/reference pair resolves independently after reopen"
        );
    }
    assert_eq!(
        reopened
            .sync_database()
            .table_meta("context_bindings")
            .expect("local binding table exists")
            .sync_direction,
        Some(SyncDirection::None),
        "opaque node-local binding rows have no sync transport locally"
    );
    drop(reopened);

    let hub = authenticated_hub().await;
    let (bound, client) = bind_tenant_store(&hub, &root, "tenant-bound").await;
    let bound_context = create_context(&bound, "tenant-bound binding context");
    bound
        .bind_context(
            key_in("tenant-caller/v1", "opaque:tenant-bound"),
            bound_context,
        )
        .expect("persist opaque key after authenticated tenant enrollment");
    assert_eq!(
        bound
            .sync_database()
            .table_meta("context_bindings")
            .expect("tenant-bound binding table exists")
            .sync_direction,
        Some(SyncDirection::None),
        "tenant placement still keeps opaque binding rows SYNC OFF"
    );
    client.shutdown().await;
    hub.stop().await;
}

#[test]
fn placement_change_is_allowed_before_capture_or_after_failure_but_not_after_erasure() {
    let root = tempfile::tempdir().expect("tempdir");
    let pre_capture_path = root.path().join("pre-capture.db");
    let local = Store::open(config(&pre_capture_path, None)).expect("open local store");
    drop(local);
    let changed = Store::open(config(
        &pre_capture_path,
        Some(EventPlacementPolicy::TenantHub),
    ))
    .expect("an empty standalone Store may change placement before capture");
    assert_eq!(
        changed.event_placement_state().expect("changed placement"),
        EventPlacementState::TenantHubUnbound
    );
    drop(changed);
    let persisted_unbound =
        Store::open(config(&pre_capture_path, None)).expect("reopen persisted tenant placement");
    assert_eq!(
        persisted_unbound
            .event_placement_state()
            .expect("persisted changed placement"),
        EventPlacementState::TenantHubUnbound,
        "None reuses the allowed pre-capture placement instead of restoring LocalOnly"
    );
    assert!(
        persisted_unbound
            .sync_database()
            .table_meta("events")
            .is_none(),
        "the persisted unbound placement leaves event tables unavailable"
    );
    assert!(matches!(
        persisted_unbound.create_context(create_input("unbound placement context")),
        Err(CgError::TenantPolicyBindingRequired { .. })
    ));
    assert!(matches!(
        persisted_unbound.append_event_batch(AppendEventBatch::live(
            ContextId::new_v7(),
            EventAdapterDeclaration {
                connector_id: "fixture".into(),
                adapter_version: "1".into(),
                anchor_declaration_version: "1".into(),
                anchor_schema: json!({"type": "object"}),
                source_position_order: vec!["/record".into()],
            },
            "source-instance",
            vec![event_draft()],
        )),
        Err(CgError::TenantPolicyBindingRequired { .. })
    ));
    drop(persisted_unbound);

    let failed_first_append_path = root.path().join("failed-first-append.db");
    let failed_first_append =
        Store::open(config(&failed_first_append_path, None)).expect("open local Store");
    let failed_context = create_context(&failed_first_append, "failed first append context");
    let mut divergent_duplicate = event_draft();
    divergent_duplicate.text = Some("different content for the same source event".into());
    assert!(matches!(
        failed_first_append.append_event_batch(AppendEventBatch::live(
            failed_context,
            EventAdapterDeclaration {
                connector_id: "fixture".into(),
                adapter_version: "1".into(),
                anchor_declaration_version: "1".into(),
                anchor_schema: json!({"type": "object"}),
                source_position_order: vec!["/record".into()],
            },
            "source-instance",
            vec![event_draft(), divergent_duplicate],
        )),
        Err(CgError::SourceEventDiverged { .. })
    ));
    for table in ["events", "event_anchors", "event_local_commits"] {
        assert!(
            failed_first_append
                .sync_database()
                .execute(&format!("SELECT * FROM {table}"), &HashMap::new())
                .expect("inspect rolled-back append rows")
                .rows
                .is_empty(),
            "the divergent second draft rolls back every staged {table} row"
        );
    }
    drop(failed_first_append);

    let changed_after_failed_first_append = Store::open(config(
        &failed_first_append_path,
        Some(EventPlacementPolicy::TenantHub),
    ))
    .expect("a failed first append does not fix placement");
    assert_eq!(
        changed_after_failed_first_append
            .event_placement_state()
            .expect("changed placement after failed append"),
        EventPlacementState::TenantHubUnbound
    );
    drop(changed_after_failed_first_append);

    let post_capture_path = root.path().join("post-capture.db");
    let captured = Store::open(config(&post_capture_path, None)).expect("open local capture store");
    let context_id = create_context(&captured, "captured context");
    let mut anchored_event = event_draft();
    anchored_event.anchors = json!({"fixture_anchor": "placement permanence"});
    let captured_event = captured
        .append_event_batch(AppendEventBatch::live(
            context_id,
            EventAdapterDeclaration {
                connector_id: "fixture".into(),
                adapter_version: "1".into(),
                anchor_declaration_version: "1".into(),
                anchor_schema: json!({"type": "object"}),
                source_position_order: vec!["/record".into()],
            },
            "source-instance",
            vec![anchored_event],
        ))
        .expect("capture one anchored event");
    let event_id = captured_event.events[0].event.id.as_uuid();
    let event_rows_before_erasure = captured
        .sync_database()
        .execute("SELECT * FROM events", &HashMap::new())
        .expect("read captured rows")
        .rows;
    let anchor_rows_before_erasure = captured
        .sync_database()
        .execute("SELECT * FROM event_anchors", &HashMap::new())
        .expect("read captured anchor rows")
        .rows;
    let local_commit_rows_before_erasure = captured
        .sync_database()
        .execute("SELECT * FROM event_local_commits", &HashMap::new())
        .expect("read captured local commit rows")
        .rows;
    assert_eq!(event_rows_before_erasure.len(), 1);
    assert_eq!(anchor_rows_before_erasure.len(), 1);
    assert_eq!(local_commit_rows_before_erasure.len(), 1);
    let event_table_before = captured
        .sync_database()
        .table_meta("events")
        .expect("captured event table")
        .clone();
    let anchor_table_before = captured
        .sync_database()
        .table_meta("event_anchors")
        .expect("captured anchor table")
        .clone();
    let local_commits_table_before = captured
        .sync_database()
        .table_meta("event_local_commits")
        .expect("captured local commit table")
        .clone();
    let placement_table_before = captured
        .sync_database()
        .table_meta("event_placement")
        .expect("placement table")
        .clone();
    let placement_rows_before = captured
        .sync_database()
        .execute("SELECT * FROM event_placement", &HashMap::new())
        .expect("read persisted local placement")
        .rows;

    let event_params = HashMap::from([("event_id".to_string(), Value::Uuid(event_id))]);
    captured
        .sync_database()
        .execute(
            "DISCARD FROM events WHERE id = $event_id, event_anchors WHERE event_id = $event_id, event_local_commits WHERE event_id = $event_id",
            &event_params,
        )
        .expect("the local erasure path discards every captured event row");
    for table in ["events", "event_anchors", "event_local_commits"] {
        assert!(
            captured
                .sync_database()
                .execute(&format!("SELECT * FROM {table}"), &HashMap::new())
                .expect("inspect erased event rows")
                .rows
                .is_empty(),
            "the successful local erasure removes every captured {table} row"
        );
    }
    drop(captured);

    assert!(matches!(
        Store::open(config(
            &post_capture_path,
            Some(EventPlacementPolicy::TenantHub)
        )),
        Err(CgError::EventPlacementPolicyMismatch { .. })
    ));

    let raw_after_refusal = Database::open(&post_capture_path).expect("inspect refused Store raw");
    assert_eq!(
        raw_after_refusal
            .table_meta("events")
            .expect("event table remains after refusal"),
        event_table_before,
        "the refused placement change leaves the event declaration unchanged before any Store reopen"
    );
    assert_eq!(
        raw_after_refusal
            .table_meta("event_anchors")
            .expect("anchor table remains after refusal"),
        anchor_table_before,
        "the refused placement change leaves the anchor declaration unchanged after erasure"
    );
    assert_eq!(
        raw_after_refusal
            .table_meta("event_local_commits")
            .expect("local commit table remains after refusal"),
        local_commits_table_before,
        "the refused placement change leaves node-local commit schema unchanged after erasure"
    );
    assert_eq!(
        raw_after_refusal
            .table_meta("event_placement")
            .expect("placement table remains after refusal"),
        placement_table_before,
        "the refused placement change leaves the persisted placement declaration unchanged"
    );
    assert_eq!(
        raw_after_refusal
            .execute("SELECT * FROM event_placement", &HashMap::new())
            .expect("read placement after refusal")
            .rows,
        placement_rows_before,
        "the refused placement change leaves the exact persisted policy row unchanged"
    );
    drop(raw_after_refusal);

    let reopened =
        Store::open(config(&post_capture_path, None)).expect("reopen original placement");
    assert_eq!(
        reopened
            .event_placement_state()
            .expect("original placement"),
        EventPlacementState::LocalOnly
    );
    for table in ["events", "event_anchors", "event_local_commits"] {
        assert!(
            reopened
                .sync_database()
                .execute(&format!("SELECT * FROM {table}"), &HashMap::new())
                .expect("read rows after reopen")
                .rows
                .is_empty(),
            "the legitimate erasure of {table} remains effective after reopen"
        );
    }
}
