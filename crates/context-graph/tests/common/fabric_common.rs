//! Test-support harness for the fabric  acceptance tests.
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
use context_graph::{Store, StoreConfig};
use contextdb_core::TenantId;
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
    _hubs: Vec<FabricNode>,
    pub edge1: FabricNode,
    pub edge2: FabricNode,
    pub edge1_client: SyncClient,
    pub edge2_client: SyncClient,
    shutdown: Arc<AtomicBool>,
    _server_tasks: Vec<tokio::task::JoinHandle<()>>,
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
    let shutdown = Arc::new(AtomicBool::new(false));
    let (first_hub, first_ticket, first_task) = start_ticketed_hub(
        &format!("{test_name}_hub_{edge1_tenant}"),
        edge1_tenant,
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
            Arc::clone(&shutdown),
        )
        .await;
        hubs.push(second_hub);
        server_tasks.push(second_task);
        second_ticket
    };

    let edge1 = FabricNode::open(&format!("{test_name}_edge1"));
    let edge2 = FabricNode::open(&format!("{test_name}_edge2"));
    let edge1_dial_spec = contextdb_server::peer_dial_spec(
        &first_ticket,
        &edge1.db_path.with_extension("fabric-identity.key"),
    );
    let edge1_client = SyncClient::new(
        edge1.store.sync_database(),
        &edge1_dial_spec,
        TenantId::from(edge1_tenant),
    );
    let edge2_dial_spec = contextdb_server::peer_dial_spec(
        &edge2_ticket,
        &edge2.db_path.with_extension("fabric-identity.key"),
    );
    let edge2_client = SyncClient::new(
        edge2.store.sync_database(),
        &edge2_dial_spec,
        TenantId::from(edge2_tenant),
    );

    FabricFixture {
        _hubs: hubs,
        edge1,
        edge2,
        edge1_client,
        edge2_client,
        shutdown,
        _server_tasks: server_tasks,
    }
}

async fn start_ticketed_hub(
    label: &str,
    tenant: &str,
    shutdown: Arc<AtomicBool>,
) -> (FabricNode, String, tokio::task::JoinHandle<()>) {
    let hub = FabricNode::open(label);
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
    let server = Arc::new(SyncServer::new(
        hub.store.sync_database(),
        &endpoint,
        TenantId::from(tenant),
    ));
    let task = tokio::spawn(async move { server.run_until(shutdown).await });
    (hub, ticket, task)
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
    SyncClient::new(db, &dial_spec, TenantId::from(tenant))
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

/// Build a model-free capture request for the fabric tests.
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
