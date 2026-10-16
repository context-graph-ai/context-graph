//! regression contracts for restoring an adapter, ordering opaque source positions,
//! and publishing the authenticated tenant schema.
//!
//! Each fixture drives the public Store and ContextDB enrollment doors. The
//! resume adapter derives its record boundaries from the bytes it actually
//! consumes; it never substitutes a scripted absolute offset for restoration.

use std::collections::HashMap;
use std::io::{BufRead, Cursor};
use std::num::NonZeroUsize;
use std::sync::Arc;

use chrono::{TimeZone, Utc};
use context_graph::schema::{INDEXES, SUBSCRIPTION_EVENT_TYPE_DDL, TABLES};
use context_graph::{
    AdaptedSourceRecord, AppendEventBatch, CG_SCHEMA_VERSION, CgError, ConsumerDeclaration,
    ConsumerReader, CreateContext, EndOfSource, EventAdapterDeclaration, EventBatchLimits,
    EventCoverage, EventPlacementPolicy, EventPlacementState, EventRenderRole, EventSourceAdapter,
    EventSourceIdentity, EventSourceRead, ImportEventOptions, SourceRecordPosition, Store,
    StoreConfig,
};
use contextdb_core::{EdgeDiscardMode, Error as DbError, TenantId, Value as DbValue};
use contextdb_engine::{Database, DeliveryOutcomeKind};
use contextdb_server::sync_client::ApplicationTablePolicyExpectation;
use contextdb_server::{FabricIdentity, InProcessBroker, SyncClient, SyncServer};
use serde_json::{Value, json};
use tempfile::TempDir;

const CONNECTOR: &str = "policy-recovery-source";
const SOURCE_INSTANCE: &str = "stateful-source-a";
const SESSION: &str = "session-a";

fn declaration(order: &[&str]) -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "1".into(),
        anchor_declaration_version: "1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: order.iter().map(|pointer| (*pointer).into()).collect(),
    }
}

fn draft(key: &str, epoch: u64, offset: u64) -> context_graph::EventDraft {
    context_graph::EventDraft {
        session_ref: SESSION.into(),
        source_event_key: key.into(),
        source_position: json!({"epoch": epoch, "offset": offset}),
        occurred_at: Some(Utc.timestamp_opt(1_700_000_000, 0).single().unwrap()),
        author: "fixture".into(),
        kind: "message".into(),
        render_role: EventRenderRole::HumanMessage,
        interaction_ref: None,
        text: Some(key.into()),
        payload: json!({"key": key}),
        anchors: json!({"file": "src/lib.rs"}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn open_store(root: &TempDir, name: &str, placement: Option<EventPlacementPolicy>) -> Store {
    Store::open(StoreConfig {
        db_path: root.path().join(format!("{name}.db")),
        event_placement: placement,
        ..StoreConfig::default()
    })
    .expect("open Store")
}

fn context(store: &Store, name: &str) -> context_graph::ContextId {
    store
        .create_context(CreateContext {
            name: name.into(),
            labels: vec![],
            properties: Default::default(),
        })
        .expect("create context after readiness")
        .id
}

fn append(
    store: &Store,
    context_id: context_graph::ContextId,
    declaration: EventAdapterDeclaration,
    drafts: Vec<context_graph::EventDraft>,
) -> context_graph::Result<context_graph::AppendEventReceipt> {
    store.append_event_batch(AppendEventBatch::live(
        context_id,
        declaration,
        SOURCE_INSTANCE,
        drafts,
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResumeCall {
    identity: EventSourceIdentity,
    checkpoint: Value,
    position: SourceRecordPosition,
}

/// An adapter whose restored cursor is derived from the actual checkpoint
/// arguments. `read_next_record` then derives every next boundary from bytes
/// consumed from the supplied reader.
struct RestoringAdapter {
    declaration: EventAdapterDeclaration,
    byte_offset: u64,
    record_ordinal: u64,
    decoded: Vec<String>,
    resumed: Vec<ResumeCall>,
}

impl RestoringAdapter {
    fn new() -> Self {
        Self {
            declaration: declaration(&["/epoch", "/offset"]),
            byte_offset: 0,
            record_ordinal: 0,
            decoded: vec![],
            resumed: vec![],
        }
    }
}

impl EventSourceAdapter for RestoringAdapter {
    fn declaration(&self) -> &EventAdapterDeclaration {
        &self.declaration
    }

    fn identify_source(
        &mut self,
        _reader: &mut dyn BufRead,
        _source_locator_ref: &str,
        _session_override: Option<&str>,
    ) -> context_graph::Result<EventSourceIdentity> {
        Ok(EventSourceIdentity {
            source_instance_ref: SOURCE_INSTANCE.into(),
            session_ref: SESSION.into(),
            provisional_empty: false,
        })
    }

    fn resume_source(
        &mut self,
        identity: &EventSourceIdentity,
        checkpoint: &Value,
        position: &SourceRecordPosition,
    ) -> context_graph::Result<()> {
        self.byte_offset = position.byte_offset;
        self.record_ordinal = position.coordinate["record"]
            .as_u64()
            .expect("the stored record coordinate is an unsigned integer");
        self.resumed.push(ResumeCall {
            identity: identity.clone(),
            checkpoint: checkpoint.clone(),
            position: position.clone(),
        });
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
        let key = std::str::from_utf8(&bytes)
            .map_err(|error| CgError::Engine(error.to_string()))?
            .trim_end_matches('\n')
            .to_owned();
        self.byte_offset += u64::try_from(bytes.len()).expect("fixture bytes fit u64");
        self.record_ordinal += 1;
        self.decoded.push(key.clone());
        Ok(EventSourceRead::Record(AdaptedSourceRecord {
            drafts: vec![draft(&key, 1, self.record_ordinal)],
            checkpoint_after_record: json!({"checkpoint": key}),
            source_position_after_record: SourceRecordPosition {
                byte_offset: self.byte_offset,
                coordinate: json!({"record": self.record_ordinal}),
            },
        }))
    }
}

fn import(
    store: &Store,
    context_id: context_graph::ContextId,
    adapter: &mut RestoringAdapter,
    bytes: &[u8],
) -> context_graph::Result<context_graph::ImportEventReceipt> {
    store.import_event_stream(
        adapter,
        &mut Cursor::new(bytes.to_vec()),
        ImportEventOptions {
            context_id,
            source_locator_ref: "fixture://stateful-source".into(),
            session_override: None,
            batch_limits: EventBatchLimits {
                max_events: NonZeroUsize::new(1).unwrap(),
                max_normalized_bytes: NonZeroUsize::new(1024 * 1024).unwrap(),
            },
        },
    )
}

#[test]
fn verified_prefix_restores_a_fresh_adapter_before_only_the_tail_is_decoded() {
    let root = TempDir::new().expect("temporary directory");
    let store = open_store(&root, "recovery", None);
    let context_id = context(&store, "recovery context");

    let mut initial = RestoringAdapter::new();
    import(&store, context_id, &mut initial, b"one\n").expect("initial committed prefix");
    assert_eq!(initial.decoded, ["one"]);
    drop(store);

    let store = open_store(&root, "recovery", None);

    let mut caught_up = RestoringAdapter::new();
    import(&store, context_id, &mut caught_up, b"one\n").expect("caught up source");
    assert!(
        caught_up.resumed.is_empty(),
        "caught-up source does not resume"
    );
    assert!(
        caught_up.decoded.is_empty(),
        "caught-up source does not decode"
    );

    let mut tail = RestoringAdapter::new();
    import(&store, context_id, &mut tail, b"one\ntwo\n").expect("verified tail import");
    assert_eq!(
        tail.resumed,
        vec![ResumeCall {
            identity: EventSourceIdentity {
                source_instance_ref: SOURCE_INSTANCE.into(),
                session_ref: SESSION.into(),
                provisional_empty: false,
            },
            checkpoint: json!({"checkpoint": "one"}),
            position: SourceRecordPosition {
                byte_offset: 4,
                coordinate: json!({"record": 1}),
            },
        }],
        "the fresh adapter receives the durable identity, opaque checkpoint, and real boundary"
    );
    assert_eq!(
        tail.decoded,
        ["two"],
        "only bytes beyond the verified prefix decode"
    );
    drop(store);

    let store = open_store(&root, "recovery", None);
    let mut resumed_again = RestoringAdapter::new();
    import(&store, context_id, &mut resumed_again, b"one\ntwo\nthree\n")
        .expect("reopened Store restores the advanced durable checkpoint");
    assert_eq!(
        resumed_again.resumed,
        vec![ResumeCall {
            identity: EventSourceIdentity {
                source_instance_ref: SOURCE_INSTANCE.into(),
                session_ref: SESSION.into(),
                provisional_empty: false,
            },
            checkpoint: json!({"checkpoint": "two"}),
            position: SourceRecordPosition {
                byte_offset: 8,
                coordinate: json!({"record": 2}),
            },
        }],
        "reopen restores the actual committed checkpoint and source boundary"
    );
    assert_eq!(resumed_again.decoded, ["three"]);

    let mut changed_prefix = RestoringAdapter::new();
    let error = import(
        &store,
        context_id,
        &mut changed_prefix,
        b"ONE\ntwo\nthree\n",
    )
    .expect_err("changed committed prefix refuses before restoration");
    assert!(matches!(error, CgError::SourceHistoryDiverged { .. }));
    assert!(changed_prefix.resumed.is_empty());
    assert!(changed_prefix.decoded.is_empty());
}

#[test]
fn live_append_uses_declared_numeric_source_order_and_persists_the_projection() {
    let root = TempDir::new().expect("temporary directory");
    let store = open_store(&root, "position-order", None);
    let context_id = context(&store, "position order context");
    let ordering = declaration(&["/epoch", "/offset"]);

    append(
        &store,
        context_id,
        ordering.clone(),
        vec![draft("nine", 9, 1)],
    )
    .expect("first ordered event");
    drop(store);

    let projection_probe = open_store(&root, "position-order", None);
    assert!(
        append(
            &projection_probe,
            context_id,
            declaration(&["/epoch"]),
            vec![draft("projection-changed", 10, 1)],
        )
        .is_err(),
        "changing the declared projection without changing adapter_version refuses on its first reopen append"
    );
    assert_eq!(
        projection_probe
            .list_stream_events(
                context_graph::EventStreamSelector {
                    context_id,
                    connector_id: CONNECTOR.into(),
                    source_instance_ref: SOURCE_INSTANCE.into(),
                },
                context_graph::EventRange::Count {
                    after_seq: None,
                    limit: std::num::NonZeroU32::new(8).unwrap(),
                },
            )
            .expect("read stream after refused projection change")
            .iter()
            .map(|event| event.source_event_key.as_str())
            .collect::<Vec<_>>(),
        ["nine"],
        "a refused first reopen projection change leaves the accepted stream unchanged"
    );
    drop(projection_probe);

    let stale_position_probe = open_store(&root, "position-order", None);
    assert!(
        append(
            &stale_position_probe,
            context_id,
            ordering.clone(),
            vec![draft("late-arrival", 8, 99)],
        )
        .is_err(),
        "a new event before the durable numeric watermark refuses on its first reopen append"
    );
    assert_eq!(
        stale_position_probe
            .list_stream_events(
                context_graph::EventStreamSelector {
                    context_id,
                    connector_id: CONNECTOR.into(),
                    source_instance_ref: SOURCE_INSTANCE.into(),
                },
                context_graph::EventRange::Count {
                    after_seq: None,
                    limit: std::num::NonZeroU32::new(8).unwrap(),
                },
            )
            .expect("read stream after refused stale position")
            .iter()
            .map(|event| event.source_event_key.as_str())
            .collect::<Vec<_>>(),
        ["nine"],
        "a refused first reopen stale position leaves the accepted stream unchanged"
    );
    drop(stale_position_probe);

    let store = open_store(&root, "position-order", None);
    let advanced = append(
        &store,
        context_id,
        ordering.clone(),
        vec![
            draft("ten", 10, 1),
            draft("ten-equal", 10, 1),
            draft("eleven-reset", 11, 0),
        ],
    )
    .expect("numeric advancement, equal tuples, and an epoch increment with reset offset append in supplied order");
    assert_eq!(advanced.inserted, 3);

    let retry = append(
        &store,
        context_id,
        ordering.clone(),
        vec![draft("nine", 9, 1)],
    )
    .expect("an older accepted natural-key retry remains a no-op after the watermark advances");
    assert_eq!(retry.inserted, 0);
    assert_eq!(retry.already_present, 1);

    assert!(
        append(
            &store,
            context_id,
            ordering.clone(),
            vec![draft("in-order", 11, 2), draft("inverted", 11, 1)],
        )
        .is_err(),
        "an inversion within one supplied batch rolls the whole batch back"
    );
    let stream = store
        .list_stream_events(
            context_graph::EventStreamSelector {
                context_id,
                connector_id: CONNECTOR.into(),
                source_instance_ref: SOURCE_INSTANCE.into(),
            },
            context_graph::EventRange::Count {
                after_seq: None,
                limit: std::num::NonZeroU32::new(8).unwrap(),
            },
        )
        .expect("read accepted stream");
    assert_eq!(
        stream
            .iter()
            .map(|event| event.source_event_key.as_str())
            .collect::<Vec<_>>(),
        ["nine", "ten", "ten-equal", "eleven-reset"],
        "the first-reopen refusals and the later inversion leave the accepted stream unchanged"
    );
    assert_eq!(
        stream.iter().map(|event| event.seq).collect::<Vec<_>>(),
        [1, 2, 3, 4],
        "accepted numeric, equal, and epoch-reset tuples receive consecutive source sequences"
    );
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
        self.task.await.expect("hub stops");
    }
}

fn event_policy(discard: &str) -> String {
    format!(
        "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE DELIVERY MANIFEST OVER event_anchors EDGE DISCARD {discard}"
    )
}

fn anchor_policy(discard: &str) -> String {
    format!("SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE EDGE DISCARD {discard}")
}

async fn authenticated_hub(name: &str, discard: &str) -> AuthenticatedHub {
    let broker = InProcessBroker::new();
    let tenant = TenantId::from(format!("core-policy-{name}"));
    let db = Arc::new(Database::open_memory());
    let identity = Arc::new(FabricIdentity::generate());
    let hub_node = identity.node_id();
    let server = Arc::new(
        SyncServer::with_authenticated_transport_and_identity_for_test(
            db.clone(),
            broker.server_as(&hub_node),
            tenant.clone(),
            hub_node,
            identity,
        ),
    );
    for (table, policy) in [
        ("events", event_policy(discard)),
        ("event_anchors", anchor_policy(discard)),
    ] {
        db.execute(
            &format!("DECLARE TENANT TABLE POLICY {table} {policy}"),
            &HashMap::<String, DbValue>::new(),
        )
        .expect("hub declares actual authenticated policy");
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

fn policy_expectation(discard: &str) -> ApplicationTablePolicyExpectation {
    ApplicationTablePolicyExpectation::new()
        .expect_table("events", event_policy(discard).as_str())
        .expect("event policy syntax")
        .expect_table("event_anchors", anchor_policy(discard).as_str())
        .expect("anchor policy syntax")
}

async fn bind(store: &Store, hub: &AuthenticatedHub, discard: &str) -> SyncClient {
    let identity = Arc::new(FabricIdentity::generate());
    let node_id = identity.node_id();
    let client = SyncClient::with_authenticated_transport_and_identity_for_test(
        store.sync_database(),
        hub.broker.client_as(&node_id),
        hub.tenant.clone(),
        identity,
    );
    let connected_hub = client
        .connected_hub_node_id()
        .expect("authenticated fixture client has a connected hub");
    if store.sync_database().retention_sync_peer().as_deref() != Some(connected_hub.as_str()) {
        client
            .change_destination(&connected_hub)
            .expect("explicitly move the fixture to its connected hub");
    }
    let binding = client
        .bind_application_table_policy(policy_expectation(discard))
        .await
        .expect("ContextDB authenticates the declared table policies");
    store
        .bind_authenticated_tenant_policy(binding)
        .expect("Store publishes the authenticated complete schema");
    client
}

#[tokio::test(flavor = "multi_thread")]
async fn unbound_store_exposes_only_bootstrap_and_binding_publishes_the_complete_schema() {
    let root = TempDir::new().expect("temporary directory");
    let db_path = root.path().join("unbound.db");
    let unbound = Store::open(StoreConfig {
        db_path: db_path.clone(),
        event_placement: Some(EventPlacementPolicy::TenantHub),
        ..StoreConfig::default()
    })
    .expect("open tenant bootstrap");
    assert_eq!(
        unbound.event_placement_state().expect("bootstrap status"),
        EventPlacementState::TenantHubUnbound
    );
    for table in ["cg_schema_version", "contexts", "events", "event_anchors"] {
        assert!(
            unbound.sync_database().table_meta(table).is_none(),
            "unbound bootstrap has no application table `{table}`"
        );
    }
    assert!(matches!(
        unbound.list_contexts(),
        Err(CgError::TenantPolicyBindingRequired { .. })
    ));
    assert!(matches!(
        unbound.create_context(CreateContext {
            name: "blocked".into(),
            labels: vec![],
            properties: Default::default(),
        }),
        Err(CgError::TenantPolicyBindingRequired { .. })
    ));
    assert!(matches!(
        append(
            &unbound,
            context_graph::ContextId::new_v7(),
            declaration(&["/epoch", "/offset"]),
            vec![draft("blocked", 1, 1)],
        ),
        Err(CgError::TenantPolicyBindingRequired { .. })
    ));
    assert!(matches!(
        ConsumerReader::open(&db_path, ConsumerDeclaration::default()),
        Err(CgError::TenantPolicyBindingRequired {
            expected: EventPlacementPolicy::TenantHub,
        })
    ));

    let rejected_hub = authenticated_hub("rejected", "NEVER").await;
    let rejected_identity = Arc::new(FabricIdentity::generate());
    let rejected_node = rejected_identity.node_id();
    let rejected_client = SyncClient::with_authenticated_transport_and_identity_for_test(
        unbound.sync_database(),
        rejected_hub.broker.client_as(&rejected_node),
        rejected_hub.tenant.clone(),
        rejected_identity,
    );
    assert!(
        rejected_client
            .bind_application_table_policy(policy_expectation("ALWAYS"))
            .await
            .is_err(),
        "a real authenticated policy mismatch does not publish application schema"
    );
    rejected_client.shutdown().await;
    rejected_hub.stop().await;
    drop(unbound);

    let reopened = Store::open(StoreConfig {
        db_path: db_path.clone(),
        event_placement: Some(EventPlacementPolicy::TenantHub),
        ..StoreConfig::default()
    })
    .expect("reopen rejected bootstrap");
    assert_eq!(
        reopened
            .event_placement_state()
            .expect("reopened bootstrap status"),
        EventPlacementState::TenantHubUnbound
    );
    assert!(reopened.sync_database().table_meta("contexts").is_none());
    assert!(matches!(
        ConsumerReader::open(&db_path, ConsumerDeclaration::default()),
        Err(CgError::TenantPolicyBindingRequired {
            expected: EventPlacementPolicy::TenantHub,
        })
    ));

    let hub = authenticated_hub("ready", "AFTER OUTCOME").await;
    let client = bind(&reopened, &hub, "AFTER OUTCOME").await;
    assert_eq!(
        reopened
            .event_placement_state()
            .expect("published placement"),
        EventPlacementState::TenantHubBound
    );
    for table in TABLES {
        assert!(
            reopened.sync_database().table_meta(table.name).is_some(),
            "ready Store publishes the complete reserved schema including `{}`",
            table.name
        );
    }
    for table in ["cg_runtime_models", "cg_changed_cursors"] {
        assert!(
            reopened.sync_database().table_meta(table).is_some(),
            "ready Store publishes the installed local runtime table `{table}`"
        );
    }
    for index in INDEXES {
        assert!(
            reopened
                .sync_database()
                .table_meta(index.table)
                .expect("ready Store publishes each indexed table")
                .indexes
                .iter()
                .any(|actual| actual.name == index.name),
            "ready Store publishes declared index `{}` on `{}`",
            index.name,
            index.table
        );
    }
    let event_types = reopened.sync_database().event_bus_status().event_types;
    for ddl in SUBSCRIPTION_EVENT_TYPE_DDL {
        let name = ddl
            .split_whitespace()
            .nth(3)
            .expect("declared event type DDL has a name");
        assert!(
            event_types.iter().any(|event_type| event_type.name == name),
            "ready Store publishes declared subscription event type `{name}`"
        );
    }
    let triggers = reopened.sync_database().list_triggers();
    let registered_callbacks = reopened.sync_database().registered_trigger_callbacks();
    for name in [
        "impact_analysis_trigger",
        "conflict_detection_on_decisions",
        "conflict_detection_on_outcomes",
        "precedent_match_on_decisions",
    ] {
        assert!(
            triggers.iter().any(|trigger| trigger.name == name),
            "ready Store publishes declared trigger `{name}`"
        );
        assert!(
            registered_callbacks.iter().any(|callback| callback == name),
            "ready Store registers callback for trigger `{name}`"
        );
    }
    assert!(
        reopened
            .sync_database()
            .execute("SELECT version FROM cg_schema_version", &HashMap::new())
            .expect("ready schema version is readable")
            .rows
            == vec![vec![DbValue::Int64(CG_SCHEMA_VERSION.into())]],
        "Ready publishes the reserved schema version, not an arbitrary version row"
    );
    assert!(
        !reopened
            .list_embedding_spaces()
            .expect("ready schema exposes its seeded embedding spaces")
            .is_empty(),
        "Ready does not publish table shells without the required seed behavior"
    );
    client.shutdown().await;
    hub.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn authenticated_discard_modes_are_enforced_from_the_bound_engine_policy() {
    for (name, clause, mode, rows_remain) in [
        ("never", "NEVER", EdgeDiscardMode::Never, true),
        (
            "after-outcome",
            "AFTER OUTCOME",
            EdgeDiscardMode::AfterOutcome,
            false,
        ),
        ("always", "ALWAYS", EdgeDiscardMode::Always, false),
    ] {
        let root = TempDir::new().expect("temporary directory");
        let hub = authenticated_hub(name, clause).await;
        let store = open_store(&root, name, Some(EventPlacementPolicy::TenantHub));
        let client = bind(&store, &hub, clause).await;
        let context_id = context(&store, &format!("{name} context"));
        let receipt = append(
            &store,
            context_id,
            declaration(&["/epoch", "/offset"]),
            vec![draft("one", 1, 1)],
        )
        .expect("write the real delivery unit");
        let event_id = receipt.events[0].event.id.as_uuid();
        let params = HashMap::from([("event_id".to_string(), DbValue::Uuid(event_id))]);
        let discard_sql =
            "DISCARD FROM events WHERE id = $event_id, event_anchors WHERE event_id = $event_id";

        if mode == EdgeDiscardMode::AfterOutcome {
            assert!(matches!(
                store.sync_database().execute(discard_sql, &params),
                Err(DbError::EdgeDiscardDenied {
                    mode: EdgeDiscardMode::AfterOutcome,
                    pending_count: 1,
                    ..
                })
            ));
            for table in ["events", "event_anchors"] {
                assert_eq!(
                    store
                        .sync_database()
                        .execute(
                            &format!(
                                "SELECT id FROM {table} WHERE {} = $event_id",
                                if table == "events" { "id" } else { "event_id" }
                            ),
                            &params,
                        )
                        .expect("inspect rows preserved while outcome is pending")
                        .rows
                        .len(),
                    1,
                    "AFTER OUTCOME preserves its {table} row while the authenticated delivery remains pending"
                );
            }
            client
                .push()
                .await
                .expect("authenticated hub records a terminal outcome");
            assert!(
                client
                    .fetch_delivery_outcomes(None)
                    .await
                    .expect("read real authenticated outcomes")
                    .iter()
                    .any(|outcome| {
                        outcome.root_table() == "events"
                            && outcome.kind() == DeliveryOutcomeKind::Accepted
                    }),
                "the unit has a real accepted outcome before local discard"
            );
        }

        let discard = store.sync_database().execute(discard_sql, &params);
        if rows_remain {
            assert!(matches!(
                discard,
                Err(DbError::EdgeDiscardDenied {
                    mode: actual_mode,
                    pending_count: 0,
                    ..
                }) if actual_mode == mode
            ));
        } else {
            discard.expect("the bound engine policy permits this local discard");
        }
        for table in ["events", "event_anchors"] {
            assert_eq!(
                store
                    .sync_database()
                    .execute(
                        &format!(
                            "SELECT id FROM {table} WHERE {} = $event_id",
                            if table == "events" { "id" } else { "event_id" }
                        ),
                        &params,
                    )
                    .expect("inspect selected rows after discard")
                    .rows
                    .len(),
                usize::from(rows_remain),
                "{name} follows the authenticated policy for its {table} row"
            );
        }
        client.shutdown().await;
        hub.stop().await;
    }
}
