//! Test-support harness for the fabric first-light acceptance tests.
//!
//! Every fabric test runs against a real authenticated, ticketed Iroh hub and two production
//! `SyncClient` edges with persisted identities. Claim exclusivity, conflict attribution, and
//! replace-on-pull come from the real engine, never a mock that hands out a winner by fiat.
//!
//! The doubles here (call-counting extractor, capture fault seam, injected-clock helper) live in
//! test support only, never in `src/`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use context_graph::fabric::CaptureFault;
use context_graph::ingest::{ConversationExtractor, ExtractionRequest, StructuredHarvest};
use context_graph::tier0::Tier0CaptureRequest;
use context_graph::{EventPlacementPolicy, EventPlacementState, Store, StoreConfig};
use contextdb_core::TenantId;
use contextdb_engine::ApplicationTablePolicyExpectation;
use contextdb_server::transport::iroh::IrohServer;
use contextdb_server::{SyncClient, SyncServer};
use tempfile::TempDir;

use crate::fake_extractor::{LabeledTranscriptExtractor, open_store_with_lexical_embedder};

/// A single node's data root + store handle.
pub struct FabricNode {
    _dir: TempDir,
    pub db_path: PathBuf,
    pub store: Store,
}

impl FabricNode {
    pub fn open(label: &str) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join(format!("{label}.db"));
        let store = open_store_with_lexical_embedder(StoreConfig {
            db_path: db_path.clone(),
            default_text_embedder: None,
            ..Default::default()
        });
        Self {
            _dir: dir,
            db_path,
            store,
        }
    }

    /// Open an ordinary node with an explicit registration set, including the empty/no-model set.
    pub fn with_models(
        label: &str,
        config: impl FnOnce(PathBuf) -> StoreConfig,
        models: Vec<context_graph::EmbedderRegistration>,
    ) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join(format!("{label}.db"));
        let store = Store::open_with_embedder_registrations(config(db_path.clone()), models)
            .expect("open node with explicit models");
        Self {
            _dir: dir,
            db_path,
            store,
        }
    }

    /// Reopen this same root after a production switch has durably declared its succession.
    pub fn reopen_with_models(
        self,
        config: impl FnOnce(PathBuf) -> StoreConfig,
        models: Vec<context_graph::EmbedderRegistration>,
    ) -> Self {
        let Self {
            _dir,
            db_path,
            store,
        } = self;
        drop(store);
        let store = Store::open_with_embedder_registrations(config(db_path.clone()), models)
            .expect("reopen node against its real persisted succession");
        Self {
            _dir,
            db_path,
            store,
        }
    }

    /// The node's data root directory (parent of the memory db) — where `cg/fabric.toml` lives.
    pub fn data_root(&self) -> PathBuf {
        self.db_path
            .parent()
            .expect("db path has a parent")
            .to_path_buf()
    }
}

/// Authenticated ticketed hubs plus two real persisted-identity edges.
pub struct FabricFixture {
    _shutdown_on_drop: ShutdownOnDrop,
    _hubs: Vec<FabricNode>,
    pub edge1: FabricNode,
    pub edge2: FabricNode,
    pub edge1_client: SyncClient,
    pub edge2_client: SyncClient,
    pub edge2_dial_spec: String,
    shutdown: Arc<AtomicBool>,
    _server_tasks: Vec<tokio::task::JoinHandle<()>>,
}

struct ShutdownOnDrop(Arc<AtomicBool>);

impl Drop for ShutdownOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Two stores after their real clients are dropped and every hub serving task has exited.
pub struct OfflineFabricFixture {
    pub edge1: FabricNode,
    pub edge2: FabricNode,
    pub stopped_hubs: usize,
}

impl FabricFixture {
    pub async fn into_offline(self) -> OfflineFabricFixture {
        let Self {
            _hubs,
            edge1,
            edge2,
            edge1_client,
            edge2_client,
            edge2_dial_spec: _,
            shutdown,
            _server_tasks,
            _shutdown_on_drop,
        } = self;
        drop(edge1_client);
        drop(edge2_client);
        shutdown.store(true, Ordering::SeqCst);
        let expected = _hubs.len();
        let mut stopped_hubs = 0;
        for task in _server_tasks {
            // run_until returns only after the serving endpoint has closed and applies drain.
            task.await
                .expect("hub serving task exits after transport shutdown");
            stopped_hubs += 1;
        }
        assert_eq!(stopped_hubs, expected, "every actual hub task was joined");
        drop(_hubs);
        drop(_shutdown_on_drop);
        OfflineFabricFixture {
            edge1,
            edge2,
            stopped_hubs,
        }
    }
}

/// Stand up the two-edge fabric harness under one tenant.
pub async fn fabric_fixture(test_name: &str) -> FabricFixture {
    fabric_fixture_with_tenants(test_name, "tenant-alpha", "tenant-alpha").await
}

/// Stand up the two-edge harness where each edge enrolls under its OWN tenant — for the
/// cross-tenant isolation tests, which must use genuinely distinct-tenant handles.
pub async fn fabric_fixture_with_tenants(
    test_name: &str,
    edge1_tenant: &str,
    edge2_tenant: &str,
) -> FabricFixture {
    fabric_fixture_with_node_openers(
        test_name,
        edge1_tenant,
        edge2_tenant,
        FabricNode::open,
        FabricNode::open,
    )
    .await
}

/// The same real transport with node-specific production model registrations.
pub async fn fabric_fixture_with_node_openers(
    test_name: &str,
    edge1_tenant: &str,
    edge2_tenant: &str,
    open_first: impl FnOnce(&str) -> FabricNode,
    open_second: impl FnOnce(&str) -> FabricNode,
) -> FabricFixture {
    let edge1 = open_first(&format!("{test_name}_edge1"));
    let edge2 = open_second(&format!("{test_name}_edge2"));
    let edge1_needs_binding = edge1
        .store
        .event_placement_state()
        .expect("first edge placement")
        == EventPlacementState::TenantHubUnbound;
    let edge2_needs_binding = edge2
        .store
        .event_placement_state()
        .expect("second edge placement")
        == EventPlacementState::TenantHubUnbound;
    let shutdown = Arc::new(AtomicBool::new(false));
    let (first_hub, first_ticket, first_task) = start_ticketed_hub(
        &format!("{test_name}_hub_{edge1_tenant}"),
        edge1_tenant,
        edge1_needs_binding || (edge1_tenant == edge2_tenant && edge2_needs_binding),
        Arc::clone(&shutdown),
    )
    .await;
    let mut hubs = vec![first_hub];
    let mut server_tasks = vec![first_task];
    let edge2_ticket = if edge2_tenant == edge1_tenant {
        first_ticket.clone()
    } else {
        let (second_hub, second_ticket, second_task) = start_ticketed_hub(
            &format!("{test_name}_hub_{edge2_tenant}"),
            edge2_tenant,
            edge2_needs_binding,
            Arc::clone(&shutdown),
        )
        .await;
        hubs.push(second_hub);
        server_tasks.push(second_task);
        second_ticket
    };

    let edge1_dial_spec = contextdb_server::peer_dial_spec(
        &first_ticket,
        &edge1.db_path.with_extension("fabric-identity.key"),
    );
    let edge1_client = SyncClient::new(
        edge1.store.sync_database(),
        &edge1_dial_spec,
        TenantId::from(edge1_tenant),
    )
    .expect("admit engine read memory");

    let edge2_dial_spec = contextdb_server::peer_dial_spec(
        &edge2_ticket,
        &edge2.db_path.with_extension("fabric-identity.key"),
    );
    let edge2_client = SyncClient::new(
        edge2.store.sync_database(),
        &edge2_dial_spec,
        TenantId::from(edge2_tenant),
    )
    .expect("admit engine read memory");

    for (edge, client, needs_binding) in [
        (&edge1, &edge1_client, edge1_needs_binding),
        (&edge2, &edge2_client, edge2_needs_binding),
    ] {
        if needs_binding {
            let expectation = ApplicationTablePolicyExpectation::new()
                .expect_table("events", EVENT_POLICY)
                .expect("event policy grammar")
                .expect_table("event_anchors", ANCHOR_POLICY)
                .expect("anchor policy grammar");
            let binding = tokio::time::timeout(
                Duration::from_secs(30),
                client.bind_application_table_policy(expectation),
            )
            .await
            .expect("authenticated tenant event binding stays bounded")
            .expect("real authenticated tenant event policy binding");
            edge.store
                .bind_authenticated_tenant_policy(binding)
                .expect("install authenticated event schema before using the edge");
        }
    }

    FabricFixture {
        _hubs: hubs,
        edge1,
        edge2,
        edge1_client,
        edge2_client,
        edge2_dial_spec,
        _shutdown_on_drop: ShutdownOnDrop(Arc::clone(&shutdown)),
        shutdown,
        _server_tasks: server_tasks,
    }
}

async fn start_ticketed_hub(
    label: &str,
    tenant: &str,
    tenant_events: bool,
    shutdown: Arc<AtomicBool>,
) -> (FabricNode, String, tokio::task::JoinHandle<()>) {
    // An event-tenant hub starts without local-only event tables. Its authority declarations
    // authenticate the edges' binding; the real sync path installs their matching schema.
    let hub = if tenant_events {
        FabricNode::with_models(
            label,
            |db_path| StoreConfig {
                db_path,
                default_text_embedder: None,
                event_placement: Some(EventPlacementPolicy::TenantHub),
                ..Default::default()
            },
            vec![],
        )
    } else {
        FabricNode::open(label)
    };
    let endpoint = tokio::time::timeout(
        Duration::from_secs(30),
        IrohServer::bind(&format!(
            "iroh:?identity={}",
            hub.db_path.with_extension("fabric-identity.key").display()
        )),
    )
    .await
    .expect("authenticated hub bind stays bounded")
    .expect("bind authenticated ticketed hub");
    let ticket = endpoint.ticket();
    let server = Arc::new(
        SyncServer::new(hub.store.sync_database(), &endpoint, TenantId::from(tenant))
            .expect("admit engine read memory"),
    );
    if tenant_events {
        for (table, policy) in [("events", EVENT_POLICY), ("event_anchors", ANCHOR_POLICY)] {
            hub.store
                .sync_database()
                .execute(
                    &format!("DECLARE TENANT TABLE POLICY {table} {policy}"),
                    &std::collections::HashMap::new(),
                )
                .expect("hub declares event policy before real edge enrollment");
        }
    }
    let task = tokio::spawn(async move { server.run_until(shutdown).await });
    (hub, ticket, task)
}

const EVENT_POLICY: &str =
    "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE DELIVERY MANIFEST OVER event_anchors";
const ANCHOR_POLICY: &str = "SYNC PUSH ONLY SYNC CONFLICT KEEP FIRST IMMUTABLE";

/// Drive one edge's push, then the other's pull, to converge the ledger.
pub async fn push_then_pull(pusher: &SyncClient, puller: &SyncClient) {
    for _ in 0..32 {
        if !pusher
            .has_pending_push_changes()
            .expect("pending push check")
        {
            break;
        }
        let before = pusher.push_watermark();
        tokio::time::timeout(Duration::from_secs(30), pusher.push())
            .await
            .expect("authenticated push stays bounded")
            .expect("push round");
        if pusher
            .has_pending_push_changes()
            .expect("pending push check")
        {
            assert!(
                pusher.push_watermark() > before,
                "push must advance while changes remain"
            );
        }
    }
    assert!(
        !pusher
            .has_pending_push_changes()
            .expect("pending push check"),
        "push did not drain within 32 authenticated rounds"
    );
    for _ in 0..32 {
        let before = puller.pull_watermark();
        tokio::time::timeout(Duration::from_secs(30), puller.pull_default())
            .await
            .expect("authenticated pull stays bounded")
            .expect("pull round");
        if puller.pull_watermark() == before {
            return;
        }
    }
    panic!("pull did not converge within 32 authenticated rounds");
}

/// A production client with a valid authenticated ticket whose endpoint has gone away.
pub async fn unavailable_authenticated_client(
    db: Arc<contextdb_engine::Database>,
    tenant: &str,
    edge_db_path: &Path,
) -> SyncClient {
    let root = tempfile::tempdir().expect("unavailable authenticated endpoint tempdir");
    let endpoint = tokio::time::timeout(
        Duration::from_secs(30),
        IrohServer::bind(&format!(
            "iroh:?identity={}",
            root.path().join("unavailable-hub.key").display()
        )),
    )
    .await
    .expect("unavailable endpoint bind stays bounded")
    .expect("mint authenticated unavailable endpoint ticket");
    let ticket = endpoint.ticket();
    drop(endpoint);
    let dial_spec = contextdb_server::peer_dial_spec(
        &ticket,
        &edge_db_path.with_extension("fabric-identity.key"),
    );
    SyncClient::new(db, &dial_spec, TenantId::from(tenant)).expect("admit engine read memory")
}

/// An extractor double that records how many times the model boundary was invoked, wrapping the
/// content-faithful lexical extractor. It never echoes the test's expected memory — it returns what
/// the lexical extractor would for the input.
#[derive(Debug, Default)]
pub struct CountingExtractor {
    inner: LabeledTranscriptExtractor,
    calls: AtomicUsize,
}

impl CountingExtractor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ConversationExtractor for CountingExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.extract(request)
    }
}

/// An extractor double that never produces a result — every call fails at the model boundary. A
/// worker running it records a `work_failures` row per attempt and no `work_results` row, so the job
/// leases-then-fails instead of completing. This is what the status tests that assert
/// leased-then-reclaimable and attempt-ceiling-reached-becomes-failed require: their stated intent is
/// a non-completing executor, which a succeeding extractor cannot exercise.
#[derive(Debug, Default)]
pub struct FailingExtractor;

impl FailingExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl ConversationExtractor for FailingExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        Err(context_graph::CgError::Engine(
            "fabric executor produced no result".to_string(),
        ))
    }
}

/// A capture fault seam that fails on demand, forcing the capture transaction to roll back after the
/// digest write.
#[derive(Debug, Default)]
pub struct ArmedCaptureFault {
    armed: Mutex<bool>,
}

impl ArmedCaptureFault {
    pub fn armed() -> Self {
        Self {
            armed: Mutex::new(true),
        }
    }
}

impl CaptureFault for ArmedCaptureFault {
    fn fail_after_digest(&self) -> context_graph::Result<()> {
        if *self.armed.lock().expect("fault lock") {
            Err(context_graph::CgError::Engine(
                "injected capture fault after digest write".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}

/// Count rows in a ledger work table on a node — the honest way to assert ledger state without
/// panicking when the table is empty (it is installed at store open).
pub fn work_row_count(store: &Store, table: &str) -> usize {
    let db = store.sync_database();
    db.scan(table, db.snapshot())
        .map(|rows| rows.len())
        .unwrap_or(0)
}

/// Acquire the per-data-root worker lock (thin re-export for the tests).
pub fn acquire_worker_lock_at(
    data_root: &std::path::Path,
) -> Option<context_graph::fabric::WorkerLock> {
    context_graph::fabric::acquire_worker_lock(data_root)
}

/// Build a tier-0 capture request for the fabric tests.
pub fn tier0_request(context: &str, session: &str, transcript: &str) -> Tier0CaptureRequest {
    Tier0CaptureRequest {
        context_label: context.to_string(),
        session_ref: session.to_string(),
        transcript: transcript.to_string(),
        source: "cg-record".to_string(),
        chunk_window: 0,
    }
}

/// The context id for a capture label, or `None` if it does not exist yet.
pub fn context_id(store: &Store, name: &str) -> Option<context_graph::ContextId> {
    store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|ctx| ctx.name == name)
        .map(|ctx| ctx.id)
}

/// Total structured memory items in a context (goals + decisions + observations).
pub fn memory_total(store: &Store, context: context_graph::ContextId) -> u32 {
    let counts = context_graph::ingest::memory_counts(store, context).expect("memory counts");
    counts.goals + counts.decisions + counts.observations
}
