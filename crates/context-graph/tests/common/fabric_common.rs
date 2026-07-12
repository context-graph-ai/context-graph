//! Test-support harness for the fabric first-light acceptance tests.
//!
//! Every fabric test runs against a REAL in-process contextdb hub (`SyncServer` in relay posture
//! over the always-compiled `InProcessBroker`) and two REAL `SyncClient` edges — no Docker, no NATS,
//! no wire transport. Claim exclusivity, conflict attribution, and replace-on-pull come from the
//! real engine, never a mock that hands out a winner by fiat.
//!
//! The doubles here (call-counting extractor, capture fault seam, injected-clock helper) live in
//! test support only, never in `src/`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use context_graph::fabric::CaptureFault;
use context_graph::ingest::{ConversationExtractor, ExtractionRequest, StructuredHarvest};
use context_graph::tier0::Tier0CaptureRequest;
use context_graph::{
    ContextGraphSyncRegistration, Store, StoreConfig, register_context_graph_sync_client,
};
use contextdb_core::TenantId;
use contextdb_engine::Database;
use contextdb_engine::sync_types::{ConflictPolicies, ConflictPolicy, SyncDirection};
use contextdb_server::{InProcessBroker, SyncClient, SyncServer};
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

    /// The node's data root directory (parent of the memory db) — where `cg/fabric.toml` lives.
    pub fn data_root(&self) -> PathBuf {
        self.db_path
            .parent()
            .expect("db path has a parent")
            .to_path_buf()
    }
}

/// A real in-process hub plus two real edges, all sharing one `InProcessBroker`.
pub struct FabricFixture {
    _broker: InProcessBroker,
    _hub: FabricNode,
    pub edge1: FabricNode,
    pub edge2: FabricNode,
    pub edge1_client: SyncClient,
    pub edge2_client: SyncClient,
    shutdown: Arc<AtomicBool>,
    _server_task: tokio::task::JoinHandle<()>,
}

impl Drop for FabricFixture {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
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
    let broker = InProcessBroker::new();
    let hub = FabricNode::open(&format!("{test_name}_hub"));
    let policies = cg_conflict_policies();
    let server = Arc::new(SyncServer::with_transport(
        hub.store.sync_database(),
        broker.server(),
        TenantId::from(edge1_tenant),
        policies,
    ));
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_task = {
        let server = Arc::clone(&server);
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            server.run_until(shutdown).await;
        })
    };

    // A genuinely two-tenant fabric gives each tenant its OWN hub view — a separate hub store, dialed
    // over its own tenant-scoped subjects on the shared broker. When the two edges enroll under
    // different tenants, stand up the second tenant's hub (a fresh, empty store) so a differently-
    // tenanted edge's push/pull resolves a responder instead of a transport miss AND sees only its own
    // tenant's data: the tenant-A edge's rows land on the tenant-A hub and never reach the tenant-B
    // hub, so a tenant-B edge pulling its own hub gets nothing — cross-tenant movement is impossible
    // at the fabric boundary. Detached and tied to the same shutdown, so it stops with the fixture.
    // (Same-tenant fixtures are unchanged.)
    if edge2_tenant != edge1_tenant {
        let second = Arc::new(SyncServer::with_transport(
            Arc::new(Database::open_memory()),
            broker.server(),
            TenantId::from(edge2_tenant),
            cg_conflict_policies(),
        ));
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            second.run_until(shutdown).await;
        });
    }

    let edge1 = FabricNode::open(&format!("{test_name}_edge1"));
    let edge2 = FabricNode::open(&format!("{test_name}_edge2"));
    let edge1_client = SyncClient::with_transport(
        edge1.store.sync_database(),
        broker.client(),
        TenantId::from(edge1_tenant),
    );
    let edge2_client = SyncClient::with_transport(
        edge2.store.sync_database(),
        broker.client(),
        TenantId::from(edge2_tenant),
    );
    register_cg_sync_tables(&edge1_client);
    register_cg_sync_tables(&edge2_client);

    FabricFixture {
        _broker: broker,
        _hub: hub,
        edge1,
        edge2,
        edge1_client,
        edge2_client,
        shutdown,
        _server_task: server_task,
    }
}

pub fn cg_conflict_policies() -> ConflictPolicies {
    context_graph::context_graph_sync_conflict_policies()
}

struct FabricSyncRegistration<'a>(&'a SyncClient);

impl ContextGraphSyncRegistration for FabricSyncRegistration<'_> {
    fn set_default_conflict_policy(&self, policy: ConflictPolicy) {
        self.0.set_default_conflict_policy(policy);
    }

    fn set_table_direction(&self, table: &str, direction: SyncDirection) {
        self.0.set_table_direction(table, direction);
    }

    fn set_conflict_policy(&self, table: &str, policy: ConflictPolicy) {
        self.0.set_conflict_policy(table, policy);
    }
}

pub fn register_cg_sync_tables(client: &SyncClient) {
    register_context_graph_sync_client(&FabricSyncRegistration(client));
}

/// Drive one edge's push, then the other's pull, to converge the ledger.
pub async fn push_then_pull(pusher: &SyncClient, puller: &SyncClient) {
    for _ in 0..32 {
        if !pusher
            .has_pending_push_changes()
            .expect("pending push check")
        {
            break;
        }
        pusher.push().await.expect("push round");
    }
    let policies = cg_conflict_policies();
    for _ in 0..32 {
        let before = puller.pull_watermark();
        puller.pull(&policies).await.expect("pull round");
        if puller.pull_watermark() == before {
            break;
        }
    }
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
