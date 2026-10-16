use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Duration;

use chrono::Utc;
use context_graph::ingest::{
    CaptureRequest, ConversationExtractor, ExtractionRequest, StructuredHarvest,
};
use context_graph::schema::{DIGEST_VECTOR_DIMENSION, TABLES};
use context_graph::{
    CapturePricing, ContextId, CreateContext, CreateDecision, CreateDigest, CreateEntity,
    CreateIntention, Decision, DecisionId, DecisionStatus, DigestId, EmbeddingId, EmbeddingTarget,
    EnforcementLevel, EntityId, EntitySnapshotId, EntityType, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, HostedProviderBoundary, HostedProviderReply, Intention,
    IntentionOrigin, IntentionPatch, IntentionStatus, ModelCallOutcome, ModelCallUsage,
    ObservationId, Outcome, PrecomputedEmbedding, ProgressKind, ProviderUsage, RecordObservation,
    Store, StoreConfig, SyncStatusBoundary, TxId, UsageReportState,
    context_graph_pending_push_change_count, provider_usage_from_str,
};
use contextdb_core::{Lsn, TenantId, Value as DbValue};
use contextdb_server::transport::iroh::IrohServer;
use contextdb_server::{SyncClient, SyncServer};
use tempfile::TempDir;

use crate::fake_extractor::{LabeledTranscriptExtractor, open_store_with_lexical_embedder};

const FORBIDDEN_USAGE_BASIS_VALUES: &[i64] = &[
    180, 220, 233, 377, 980, 1200, 1380, 1441, 1818, 2580, 3217, 3450,
];

pub const BACKLOG: &str = include_str!("../fixtures/backlog_two_sessions.jsonl");
pub const PROVIDER_WITH_USAGE: &str =
    include_str!("../fixtures/realistic_provider_reply_with_usage.json");
pub const PROVIDER_NO_USAGE: &str =
    include_str!("../fixtures/realistic_provider_reply_no_usage.json");

#[derive(Debug)]
pub struct FixtureHostedProvider {
    replies: Mutex<VecDeque<HostedProviderReply>>,
    calls: std::sync::atomic::AtomicUsize,
}

impl FixtureHostedProvider {
    pub fn from_fixture(raw: &str) -> Self {
        let values: Vec<serde_json::Value> =
            serde_json::from_str(raw).expect("provider fixture should be JSON array");
        Self {
            replies: Mutex::new(
                values
                    .into_iter()
                    .enumerate()
                    .map(|(index, body)| HostedProviderReply {
                        body: body.to_string(),
                        elapsed: if index == 0 {
                            Duration::from_millis(750)
                        } else {
                            Duration::from_millis(825)
                        },
                    })
                    .collect(),
            ),
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl HostedProviderBoundary for FixtureHostedProvider {
    fn call_model(&self, _prompt: &str) -> context_graph::Result<HostedProviderReply> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.replies
            .lock()
            .expect("provider replies lock")
            .pop_front()
            .ok_or_else(|| context_graph::CgError::Engine("provider fixture exhausted".to_string()))
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

pub fn provider_with_usage() -> FixtureHostedProvider {
    FixtureHostedProvider::from_fixture(PROVIDER_WITH_USAGE)
}

pub fn provider_with_unique_usage_for_chunks(
    chunks: usize,
) -> (FixtureHostedProvider, Vec<ModelCallUsage>) {
    assert!(chunks > 0, "dynamic provider requires at least one chunk");
    let usage = (0..chunks)
        .map(|index| {
            let prompt_tokens = 4100 + (index as u64 * 173);
            let completion_tokens = 211 + (index as u64 * 31);
            ProviderUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }
        })
        .collect::<Vec<_>>();
    let replies = usage
        .iter()
        .enumerate()
        .map(|(index, usage)| {
            serde_json::json!({
                "id": format!("chatcmpl-dynamic-{}", index + 1),
                "object": "chat.completion",
                "created": 1766017000 + index as u64,
                "model": "anthropic/claude-3-haiku",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": format!(
                            "{{\"summary\":\"dynamic chunk {}\",\"goals\":[],\"entities\":[],\"decisions\":[],\"observations\":[]}}",
                            index + 1
                        )
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            })
        })
        .collect::<Vec<_>>();
    let expected = usage
        .into_iter()
        .enumerate()
        .map(|(index, usage)| ModelCallUsage {
            chunk_index: index + 1,
            usage: UsageReportState::Tokens(usage),
            elapsed: dynamic_provider_elapsed(index),
            reliability_level: EnforcementLevel::Native,
            outcome: ModelCallOutcome::Captured,
            flow_tag: None,
            model_id: None,
        })
        .collect::<Vec<_>>();
    (
        FixtureHostedProvider {
            replies: Mutex::new(
                replies
                    .into_iter()
                    .enumerate()
                    .map(|(index, body)| HostedProviderReply {
                        body: body.to_string(),
                        elapsed: dynamic_provider_elapsed(index),
                    })
                    .collect(),
            ),
            calls: std::sync::atomic::AtomicUsize::new(0),
        },
        expected,
    )
}

fn dynamic_provider_elapsed(index: usize) -> Duration {
    Duration::from_millis(700 + index as u64 * 83)
}

pub fn provider_absent_and_zero_usage() -> FixtureHostedProvider {
    FixtureHostedProvider::from_fixture(PROVIDER_NO_USAGE)
}

pub fn expected_total_tokens(calls: &[ModelCallUsage]) -> u64 {
    calls
        .iter()
        .filter_map(|call| match &call.usage {
            UsageReportState::Tokens(usage) => Some(usage.total_tokens),
            UsageReportState::Zero => Some(0),
            UsageReportState::Unavailable => None,
        })
        .sum()
}

pub fn expected_chunk_count(transcript: &str, max_chars_per_chunk: usize) -> usize {
    transcript
        .as_bytes()
        .chunks(max_chars_per_chunk)
        .filter(|chunk| !chunk.is_empty())
        .count()
}

pub fn expected_projected_tokens(transcript: &str, max_chars_per_chunk: usize) -> u64 {
    expected_chunk_count(transcript, max_chars_per_chunk) as u64 * 1290
}

pub fn expected_zero_absent_and_real_usage() -> Vec<UsageReportState> {
    vec![
        UsageReportState::Tokens(ProviderUsage {
            prompt_tokens: 1200,
            completion_tokens: 180,
            total_tokens: 1380,
        }),
        UsageReportState::Unavailable,
        UsageReportState::Zero,
    ]
}

pub fn expected_progress_events_for(
    transcript: &str,
    max_chars_per_chunk: usize,
) -> Vec<context_graph::CaptureProgressEvent> {
    let total_chunks = expected_chunk_count(transcript, max_chars_per_chunk);
    let mut events = Vec::with_capacity(total_chunks * 2);
    for chunk_index in 1..=total_chunks {
        events.push(context_graph::CaptureProgressEvent {
            chunk_index,
            total_chunks,
            elapsed: Duration::ZERO,
            rough_remaining: expected_rough_remaining(
                total_chunks,
                chunk_index,
                ProgressKind::InFlight,
            ),
            kind: ProgressKind::InFlight,
        });
        events.push(context_graph::CaptureProgressEvent {
            chunk_index,
            total_chunks,
            elapsed: Duration::ZERO,
            rough_remaining: expected_rough_remaining(
                total_chunks,
                chunk_index,
                ProgressKind::Completed,
            ),
            kind: ProgressKind::Completed,
        });
    }
    events
}

fn expected_rough_remaining(
    total_chunks: usize,
    chunk_index: usize,
    kind: ProgressKind,
) -> Duration {
    let remaining_chunks = match kind {
        ProgressKind::InFlight => total_chunks.saturating_sub(chunk_index) + 1,
        ProgressKind::Completed => total_chunks.saturating_sub(chunk_index),
    };
    Duration::from_millis(remaining_chunks as u64 * 100)
}

pub fn flat_pricing() -> CapturePricing {
    CapturePricing::FlatSubscription
}

pub fn haiku_pricing() -> CapturePricing {
    CapturePricing::PayPerToken {
        usd_per_token: 0.00000025,
    }
}

pub struct TestStore {
    _dir: TempDir,
    db_path: PathBuf,
    pub store: Store,
}

pub struct TicketedMemoryNode {
    pub store: Store,
    pub client: SyncClient,
}

pub struct TicketedMemoryFixture {
    _root: TempDir,
    hub: Store,
    ticket: String,
    tenant_id: String,
    shutdown: Arc<AtomicBool>,
    server_task: tokio::task::JoinHandle<()>,
}

impl TicketedMemoryFixture {
    pub fn hub_store(&self) -> &Store {
        &self.hub
    }

    pub fn open_edge(&self, name: &str) -> TicketedMemoryNode {
        let db_path = self._root.path().join(format!("{name}.db"));
        let store = open_store_with_lexical_embedder(StoreConfig {
            db_path: db_path.clone(),
            ..Default::default()
        });
        let dial_spec = contextdb_server::peer_dial_spec(
            &self.ticket,
            &db_path.with_extension("fabric-identity.key"),
        );
        let client = SyncClient::new(
            store.sync_database(),
            &dial_spec,
            TenantId::from(&self.tenant_id),
        );
        TicketedMemoryNode { store, client }
    }

    pub async fn shutdown(self) {
        self.shutdown.store(true, Ordering::SeqCst);
        within(self.server_task)
            .await
            .expect("ticketed hub task stops within the test bound");
    }
}

pub async fn ticketed_memory_fixture(test_name: &str) -> TicketedMemoryFixture {
    let root = tempfile::tempdir().expect("ticketed fixture tempdir");
    let tenant_id = format!("tenant_{test_name}");
    let hub = open_store_with_lexical_embedder(StoreConfig {
        db_path: root.path().join("hub.db"),
        ..Default::default()
    });
    let endpoint = within(IrohServer::bind(&format!(
        "iroh:?identity={}",
        root.path().join("hub.fabric-identity.key").display()
    )))
    .await
    .expect("bind ticketed Iroh hub");
    let ticket = endpoint.ticket();
    let server = Arc::new(SyncServer::new(
        hub.sync_database(),
        &endpoint,
        TenantId::from(&tenant_id),
    ));
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_task = tokio::spawn({
        let server = Arc::clone(&server);
        let shutdown = Arc::clone(&shutdown);
        async move { server.run_until(shutdown).await }
    });
    TicketedMemoryFixture {
        _root: root,
        hub,
        ticket,
        tenant_id,
        shutdown,
        server_task,
    }
}

pub async fn within<F: std::future::Future>(future: F) -> F::Output {
    // The test guard must exceed the engine's 60-second request ceiling.
    tokio::time::timeout(Duration::from_secs(90), future)
        .await
        .expect("bounded authenticated sync operation")
}

pub async fn drain_ticketed_push(client: &SyncClient) {
    for _ in 0..32 {
        if !client
            .has_pending_push_changes()
            .expect("ticketed pending push check")
        {
            return;
        }
        let before = client.push_watermark();
        within(client.push()).await.expect("ticketed push round");
        if !client
            .has_pending_push_changes()
            .expect("ticketed pending push check")
        {
            return;
        }
        assert!(
            client.push_watermark() > before,
            "ticketed push drain must advance while work remains"
        );
    }
    panic!("ticketed push did not drain within 32 rounds");
}

pub async fn drain_ticketed_pull(client: &SyncClient) {
    for _ in 0..32 {
        let before = client.pull_watermark();
        within(client.pull_default())
            .await
            .expect("ticketed pull round");
        if client.pull_watermark() == before {
            return;
        }
    }
    panic!("ticketed pull did not drain within 32 rounds");
}

pub async fn unavailable_authenticated_client(
    db: Arc<contextdb_engine::Database>,
    tenant_id: &str,
    real_hub_ticket: &str,
    edge_db_path: &Path,
) -> SyncClient {
    let full: iroh_tickets::endpoint::EndpointTicket =
        real_hub_ticket.parse().expect("fixture hub ticket parses");
    let blackhole_addr = iroh::EndpointAddr::new(full.endpoint_addr().id)
        .with_ip_addr("240.0.0.1:38321".parse().expect("black-hole socket addr"));
    let blackhole_ticket = iroh_tickets::endpoint::EndpointTicket::new(blackhole_addr).to_string();
    let dial_spec = contextdb_server::peer_dial_spec(
        &blackhole_ticket,
        &edge_db_path.with_extension("fabric-identity.key"),
    );
    SyncClient::new(db, &dial_spec, TenantId::from(tenant_id))
}

impl TestStore {
    pub fn new(label: &str) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join(format!("{label}.db"));
        let store = open_store_with_lexical_embedder(StoreConfig {
            db_path: db_path.clone(),
            ..Default::default()
        });
        Self {
            _dir: dir,
            db_path,
            store,
        }
    }

    pub fn scoped_store(
        &self,
        contexts: impl IntoIterator<Item = context_graph::ContextId>,
    ) -> Store {
        Store::open_with_contexts(
            StoreConfig {
                db_path: self.db_path.clone(),
                ..Default::default()
            },
            contexts.into_iter().collect::<BTreeSet<_>>(),
        )
        .expect("open scoped store")
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }
}

pub fn capture_request(context_label: &str, transcript: &str) -> CaptureRequest {
    CaptureRequest {
        context_label: context_label.to_string(),
        transcript: transcript.to_string(),
        model_available: true,
    }
}

pub fn extractor() -> LabeledTranscriptExtractor {
    LabeledTranscriptExtractor
}

pub struct ProviderBackedExtractor<'a> {
    inner: LabeledTranscriptExtractor,
    provider: &'a dyn HostedProviderBoundary,
    calls: Mutex<Vec<ModelCallUsage>>,
}

impl<'a> ProviderBackedExtractor<'a> {
    pub fn new(provider: &'a dyn HostedProviderBoundary) -> Self {
        Self {
            inner: LabeledTranscriptExtractor,
            provider,
            calls: Mutex::new(Vec::new()),
        }
    }

    fn record_usage(&self, reply: HostedProviderReply) {
        let usage = match provider_usage_from_str(&reply.body) {
            Some(usage)
                if usage.prompt_tokens == 0
                    && usage.completion_tokens == 0
                    && usage.total_tokens == 0 =>
            {
                UsageReportState::Zero
            }
            Some(usage) => UsageReportState::Tokens(usage),
            None => UsageReportState::Unavailable,
        };
        let mut calls = self.calls.lock().expect("usage calls lock");
        let chunk_index = calls.len() + 1;
        calls.push(ModelCallUsage {
            chunk_index,
            usage,
            elapsed: reply.elapsed,
            reliability_level: EnforcementLevel::Native,
            outcome: ModelCallOutcome::Captured,
            flow_tag: None,
            model_id: None,
        });
    }
}

impl ConversationExtractor for ProviderBackedExtractor<'_> {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let reply = self.provider.call_model(request.transcript)?;
        self.record_usage(reply);
        self.inner.extract(request)
    }

    fn drain_model_usage(&self) -> Vec<ModelCallUsage> {
        self.calls
            .lock()
            .map(|mut calls| std::mem::take(&mut *calls))
            .unwrap_or_default()
    }
}

pub fn provider_backed_extractor(
    provider: &dyn HostedProviderBoundary,
) -> ProviderBackedExtractor<'_> {
    ProviderBackedExtractor::new(provider)
}

pub struct SeededGraph {
    pub intention: Intention,
    pub decision: Decision,
    pub context_id: context_graph::ContextId,
    pub entity_id: context_graph::EntityId,
    pub entity_name: String,
    pub entity_path: String,
    pub digest_id: DigestId,
    pub digest_summary: String,
    pub evidence_id: EvidenceId,
    pub basis_snapshot_id: EntitySnapshotId,
    pub observation_id: ObservationId,
}

pub fn seed_decision_graph(store: &Store, context_name: &str, suffix: &str) -> SeededGraph {
    let context = store
        .create_context(CreateContext {
            name: context_name.to_string(),
            ..Default::default()
        })
        .expect("create context");
    let intention = store
        .create_intention(CreateIntention {
            description: format!("resume across machines {suffix}"),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            ..Default::default()
        })
        .expect("create intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::File,
            name: format!("auth module {suffix}"),
            properties: BTreeMap::from([(
                "path".to_string(),
                serde_json::Value::from("src/auth.py"),
            )]),
            context_id: context.id,
            ..Default::default()
        })
        .expect("create entity");
    let entity_name = entity.name.clone();
    let entity_path = entity
        .properties
        .get("path")
        .and_then(serde_json::Value::as_str)
        .expect("seeded entity path")
        .to_string();
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "technical".to_string(),
            description: format!("auth module at src/auth.py {suffix}"),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            context_id: context.id,
            ..Default::default()
        })
        .expect("create decision");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let observation = store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id: entity.id,
            context_id: context.id,
            observation_type: "change".to_string(),
            source: "agent-session".to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: evidence_id,
                source_ref: format!("s3:session/{suffix}"),
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::from([(
                "path".to_string(),
                Some(serde_json::Value::from("src/auth.py")),
            )]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record observation");
    let digest_id = DigestId::new_v7();
    let digest_summary = format!("digest for {suffix}");
    store
        .create_digest(CreateDigest {
            id: digest_id,
            source: "agent-session".to_string(),
            session_ref: Some(format!("s3:session/{suffix}")),
            summary: digest_summary.clone(),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: "text_default_nomic_v1_5_768".to_string(),
                vector: digest_vector(suffix),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: EvidenceProducer::default(),
            },
            context_id: context.id,
            properties: BTreeMap::new(),
        })
        .expect("create digest");
    insert_extracted_from_edge(
        store,
        context.id,
        decision.id.as_uuid(),
        digest_id.as_uuid(),
    );
    SeededGraph {
        intention,
        context_id: context.id,
        entity_id: entity.id,
        entity_name,
        entity_path,
        digest_id,
        digest_summary,
        basis_snapshot_id: decision
            .based_on_snapshots
            .first()
            .copied()
            .expect("decision should record basis snapshot"),
        decision,
        evidence_id,
        observation_id: observation.observation_id,
    }
}

fn digest_vector(suffix: &str) -> Vec<f32> {
    let mut vector = vec![0.0; DIGEST_VECTOR_DIMENSION];
    let axis = suffix.bytes().fold(0usize, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as usize)
    }) % DIGEST_VECTOR_DIMENSION;
    vector[axis] = 1.0;
    vector
}

fn insert_extracted_from_edge(
    store: &Store,
    context_id: ContextId,
    source_id: uuid::Uuid,
    digest_id: uuid::Uuid,
) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(uuid::Uuid::now_v7())),
            ("source_id".to_string(), DbValue::Uuid(source_id)),
            ("target_id".to_string(), DbValue::Uuid(digest_id)),
            ("edge_type".to_string(), DbValue::Text("EXTRACTED_FROM".to_string())),
            ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
            ("valid_from".to_string(), DbValue::TxId(TxId(1))),
            ("valid_to".to_string(), DbValue::Null),
            ("properties".to_string(), DbValue::Json(serde_json::json!({}))),
        ]),
    )
    .expect("insert exact extracted-from edge");
}

pub async fn seed_two_edges_concurrently(
    fixture: &RealSyncFixture,
    edge1_suffix: &'static str,
    edge2_suffix: &'static str,
) -> (SeededGraph, SeededGraph) {
    let barrier = Arc::new(Barrier::new(2));
    let edge1_store = fixture.edge1.store.clone();
    let edge2_store = fixture.edge2.store.clone();
    let edge1_barrier = Arc::clone(&barrier);
    let edge2_barrier = Arc::clone(&barrier);
    let edge1 = tokio::task::spawn_blocking(move || {
        edge1_barrier.wait();
        seed_decision_graph(&edge1_store, "tenant-alpha", edge1_suffix)
    });
    let edge2 = tokio::task::spawn_blocking(move || {
        edge2_barrier.wait();
        seed_decision_graph(&edge2_store, "tenant-alpha", edge2_suffix)
    });
    let (edge1, edge2) = tokio::join!(edge1, edge2);
    (
        edge1.expect("edge1 concurrent seed"),
        edge2.expect("edge2 concurrent seed"),
    )
}

pub struct ConcurrentBasisTouch {
    pub edge1_observation_id: ObservationId,
    pub edge2_observation_id: ObservationId,
}

pub async fn touch_same_decision_graph_concurrently(
    fixture: &RealSyncFixture,
    seeded: &SeededGraph,
) -> ConcurrentBasisTouch {
    let barrier = Arc::new(Barrier::new(2));
    let edge1_store = fixture.edge1.store.clone();
    let edge2_store = fixture.edge2.store.clone();
    let edge1_seed = seeded.clone_for_concurrent_touch();
    let edge2_seed = seeded.clone_for_concurrent_touch();
    let edge1_barrier = Arc::clone(&barrier);
    let edge2_barrier = Arc::clone(&barrier);
    let edge1 = tokio::task::spawn_blocking(move || {
        edge1_barrier.wait();
        edge1_store
            .update_intention(
                edge1_seed.intention.id,
                IntentionPatch {
                    properties: Some(BTreeMap::from([(
                        "edge1_touch".to_string(),
                        serde_json::Value::from(true),
                    )])),
                    ..Default::default()
                },
            )
            .expect("edge1 touches same intention");
        record_basis_touch_observation(
            &edge1_store,
            &edge1_seed,
            "edge-one-concurrent-touch",
            "src/auth_edge1.py",
        )
    });
    let edge2 = tokio::task::spawn_blocking(move || {
        edge2_barrier.wait();
        edge2_store
            .update_intention(
                edge2_seed.intention.id,
                IntentionPatch {
                    properties: Some(BTreeMap::from([(
                        "edge2_touch".to_string(),
                        serde_json::Value::from(true),
                    )])),
                    ..Default::default()
                },
            )
            .expect("edge2 touches same intention");
        record_basis_touch_observation(
            &edge2_store,
            &edge2_seed,
            "edge-two-concurrent-touch",
            "src/auth_edge2.py",
        )
    });
    let (edge1, edge2) = tokio::join!(edge1, edge2);
    ConcurrentBasisTouch {
        edge1_observation_id: edge1.expect("edge1 concurrent basis touch"),
        edge2_observation_id: edge2.expect("edge2 concurrent basis touch"),
    }
}

pub struct ConcurrentOutcomeTouch {
    /// The outcome the hub accepts LAST for the contended decision. Both edges write
    /// concurrently, but the pushes reach the hub in a fixed order (edge1 then edge2),
    /// so the last hub commit — edge2's row — is the value the whole fleet converges on.
    pub hub_latest_winner: Outcome,
    pub surrounding_edge1: SeededGraph,
    pub surrounding_edge2: SeededGraph,
}

pub async fn record_contended_outcomes_concurrently(
    fixture: &RealSyncFixture,
    seeded: &SeededGraph,
) -> ConcurrentOutcomeTouch {
    let barrier = Arc::new(Barrier::new(2));
    let edge1_store = fixture.edge1.store.clone();
    let edge2_store = fixture.edge2.store.clone();
    let edge1_decision_id = seeded.decision.id;
    let edge2_decision_id = seeded.decision.id;
    let edge1_barrier = Arc::clone(&barrier);
    let edge2_barrier = Arc::clone(&barrier);
    let edge1 = tokio::task::spawn_blocking(move || {
        edge1_barrier.wait();
        let _outcome = edge1_store
            .record_outcome(
                edge1_decision_id,
                true,
                Some("earlier contended outcome from edge1".to_string()),
            )
            .expect("edge1 records contended outcome");
        seed_decision_graph(
            &edge1_store,
            "tenant-alpha-surrounding-edge1",
            "edge1 surrounding",
        )
    });
    let edge2 = tokio::task::spawn_blocking(move || {
        edge2_barrier.wait();
        let outcome = edge2_store
            .record_outcome(
                edge2_decision_id,
                false,
                Some("later contended outcome from edge2".to_string()),
            )
            .expect("edge2 records contended outcome");
        let surrounding = seed_decision_graph(
            &edge2_store,
            "tenant-alpha-surrounding-edge2",
            "edge2 surrounding",
        );
        (outcome, surrounding)
    });
    let (edge1, edge2) = tokio::join!(edge1, edge2);
    let surrounding_edge1 = edge1.expect("edge1 contended outcome");
    let (hub_latest_winner, surrounding_edge2) = edge2.expect("edge2 contended outcome");
    ConcurrentOutcomeTouch {
        hub_latest_winner,
        surrounding_edge1,
        surrounding_edge2,
    }
}

fn record_basis_touch_observation(
    store: &Store,
    seeded: &SeededGraph,
    source: &str,
    path: &str,
) -> ObservationId {
    let observation_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id: seeded.entity_id,
            context_id: seeded.context_id,
            observation_type: "basis_touch".to_string(),
            source: source.to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: EvidenceId::new_v7(),
                source_ref: format!("s3:session/{source}"),
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::from([(
                "path".to_string(),
                Some(serde_json::Value::from(path)),
            )]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record concurrent basis touch");
    observation_id
}

impl SeededGraph {
    fn clone_for_concurrent_touch(&self) -> Self {
        Self {
            intention: self.intention.clone(),
            decision: self.decision.clone(),
            context_id: self.context_id,
            entity_id: self.entity_id,
            entity_name: self.entity_name.clone(),
            entity_path: self.entity_path.clone(),
            digest_id: self.digest_id,
            digest_summary: self.digest_summary.clone(),
            evidence_id: self.evidence_id,
            basis_snapshot_id: self.basis_snapshot_id,
            observation_id: self.observation_id,
        }
    }
}

pub struct RealSyncFixture {
    _hub: TestStore,
    pub edge1: TestStore,
    pub edge2: TestStore,
    pub edge1_client: SyncClient,
    pub edge2_client: SyncClient,
    shutdown: Arc<AtomicBool>,
    _server_task: tokio::task::JoinHandle<()>,
    pub tenant_id: String,
    pub remote_url: String,
}

pub struct SelfEchoSyncFixture {
    _hub: TestStore,
    pub node: TestStore,
    pub client: SyncClient,
    shutdown: Arc<AtomicBool>,
    _server_task: tokio::task::JoinHandle<()>,
}

pub struct PreClientSyncFixture {
    _hub: TestStore,
    pub edge1: TestStore,
    pub edge2: TestStore,
    ticket: String,
    shutdown: Arc<AtomicBool>,
    _server_task: tokio::task::JoinHandle<()>,
    pub tenant_id: String,
}

impl PreClientSyncFixture {
    pub fn connect(&self) -> (SyncClient, SyncClient) {
        let edge1_dial_spec = contextdb_server::peer_dial_spec(
            &self.ticket,
            &self.edge1.db_path.with_extension("fabric-identity.key"),
        );
        let edge1_client = SyncClient::new(
            self.edge1.store.sync_database(),
            &edge1_dial_spec,
            TenantId::from(&self.tenant_id),
        );
        let edge2_dial_spec = contextdb_server::peer_dial_spec(
            &self.ticket,
            &self.edge2.db_path.with_extension("fabric-identity.key"),
        );
        let edge2_client = SyncClient::new(
            self.edge2.store.sync_database(),
            &edge2_dial_spec,
            TenantId::from(&self.tenant_id),
        );
        (edge1_client, edge2_client)
    }
}

impl Drop for PreClientSyncFixture {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

impl Drop for RealSyncFixture {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

impl Drop for SelfEchoSyncFixture {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

impl RealSyncFixture {
    pub fn hub_store(&self) -> &Store {
        &self._hub.store
    }

    /// Mint an additional persisted-identity edge client on this ticketed hub.
    pub fn new_edge_client(
        &self,
        db: std::sync::Arc<contextdb_engine::Database>,
        identity_path: &Path,
    ) -> SyncClient {
        let dial_spec = contextdb_server::peer_dial_spec(&self.remote_url, identity_path);
        SyncClient::new(db, &dial_spec, TenantId::from(&self.tenant_id))
    }
}

pub async fn real_sync_fixture(test_name: &str) -> RealSyncFixture {
    let tenant_id = format!("tenant_{test_name}");
    let hub = TestStore::new(&format!("{test_name}_hub"));
    let endpoint = within(IrohServer::bind(&format!(
        "iroh:?identity={}",
        hub.db_path.with_extension("fabric-identity.key").display()
    )))
    .await
    .expect("bind authenticated fixture hub");
    let remote_url = endpoint.ticket();
    let server = Arc::new(SyncServer::new(
        hub.store.sync_database(),
        &endpoint,
        TenantId::from(&tenant_id),
    ));
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_task = {
        let server = Arc::clone(&server);
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            server.run_until(shutdown).await;
        })
    };

    let edge1 = TestStore::new(&format!("{test_name}_edge1"));
    let edge2 = TestStore::new(&format!("{test_name}_edge2"));
    let edge1_dial_spec = contextdb_server::peer_dial_spec(
        &remote_url,
        &edge1.db_path.with_extension("fabric-identity.key"),
    );
    let edge1_client = SyncClient::new(
        edge1.store.sync_database(),
        &edge1_dial_spec,
        TenantId::from(&tenant_id),
    );
    let edge2_dial_spec = contextdb_server::peer_dial_spec(
        &remote_url,
        &edge2.db_path.with_extension("fabric-identity.key"),
    );
    let edge2_client = SyncClient::new(
        edge2.store.sync_database(),
        &edge2_dial_spec,
        TenantId::from(&tenant_id),
    );
    RealSyncFixture {
        _hub: hub,
        edge1,
        edge2,
        edge1_client,
        edge2_client,
        shutdown,
        _server_task: server_task,
        tenant_id,
        remote_url,
    }
}

pub async fn self_echo_sync_fixture(test_name: &str) -> SelfEchoSyncFixture {
    let tenant_id = format!("tenant_{test_name}");
    let hub = TestStore::new(&format!("{test_name}_hub"));
    let node = TestStore::new(&format!("{test_name}_node"));
    let endpoint = within(IrohServer::bind(&format!(
        "iroh:?identity={}",
        hub.db_path.with_extension("fabric-identity.key").display()
    )))
    .await
    .expect("bind authenticated self-echo hub");
    let ticket = endpoint.ticket();
    let server = Arc::new(SyncServer::new(
        hub.store.sync_database(),
        &endpoint,
        TenantId::from(&tenant_id),
    ));
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_task = {
        let server = Arc::clone(&server);
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            server.run_until(shutdown).await;
        })
    };
    let dial_spec = contextdb_server::peer_dial_spec(
        &ticket,
        &node.db_path.with_extension("fabric-identity.key"),
    );
    let client = SyncClient::new(
        node.store.sync_database(),
        &dial_spec,
        TenantId::from(&tenant_id),
    );
    SelfEchoSyncFixture {
        _hub: hub,
        node,
        client,
        shutdown,
        _server_task: server_task,
    }
}

pub async fn pre_client_sync_fixture(test_name: &str) -> PreClientSyncFixture {
    let tenant_id = format!("tenant_{test_name}");
    let hub = TestStore::new(&format!("{test_name}_hub"));
    let endpoint = within(IrohServer::bind(&format!(
        "iroh:?identity={}",
        hub.db_path.with_extension("fabric-identity.key").display()
    )))
    .await
    .expect("bind pre-client authenticated hub");
    let ticket = endpoint.ticket();
    let server = Arc::new(SyncServer::new(
        hub.store.sync_database(),
        &endpoint,
        TenantId::from(&tenant_id),
    ));
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_task = {
        let server = Arc::clone(&server);
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            server.run_until(shutdown).await;
        })
    };
    let edge1 = TestStore::new(&format!("{test_name}_edge1"));
    let edge2 = TestStore::new(&format!("{test_name}_edge2"));
    PreClientSyncFixture {
        _hub: hub,
        edge1,
        edge2,
        ticket,
        shutdown,
        _server_task: server_task,
        tenant_id,
    }
}

pub fn expected_pending_push_change_count(client: &SyncClient) -> usize {
    context_graph_pending_push_change_count(client)
        .expect("count the exact pending push set for the authenticated fixture")
}

pub async fn push_pull(edge1: &SyncClient, edge2: &SyncClient) {
    drain_push(edge1).await;
    drain_pull(edge2).await;
}

pub async fn bootstrap_sync_schema_without_rows(fixture: &RealSyncFixture) {
    within(fixture.edge1_client.push())
        .await
        .expect("edge1 pushes schema watermark");
    for _ in 0..32 {
        let before = fixture.edge2_client.pull_watermark();
        within(fixture.edge2_client.pull_default())
            .await
            .expect("edge2 admin handle pulls schema watermark");
        if fixture.edge2_client.pull_watermark() == before {
            return;
        }
    }
    assert!(
        fixture.edge2_client.pull_watermark() > Lsn(0),
        "edge2 admin schema bootstrap must advance the pull watermark"
    );
}

pub async fn bidirectional_sync(edge1: &SyncClient, edge2: &SyncClient) {
    drain_push(edge1).await;
    drain_push(edge2).await;
    drain_pull(edge1).await;
    drain_pull(edge2).await;
}

pub async fn drain_push(client: &SyncClient) {
    for _ in 0..32 {
        if !client
            .has_pending_push_changes()
            .expect("pending push check")
        {
            return;
        }
        let before = client.push_watermark();
        within(client.push()).await.expect("push sync round");
        if !client
            .has_pending_push_changes()
            .expect("pending push check")
        {
            return;
        }
        assert!(
            client.push_watermark() > before,
            "push drain must advance while pending changes remain"
        );
    }
    panic!("push did not drain pending changes");
}

pub async fn drain_pull(client: &SyncClient) {
    for _ in 0..32 {
        let before = client.pull_watermark();
        within(client.pull_default())
            .await
            .expect("pull sync round");
        if client.pull_watermark() == before {
            // Mirror the production pull path: re-derive skill on/off state from the
            // now-synced teaching-events log so a reversal recorded on a peer takes effect.
            context_graph::skill_store::reconcile_skills_from_teaching_events(client.db());
            return;
        }
    }
    panic!("pull did not drain remote changes");
}

pub fn assert_lsn_advanced(before: Lsn, after: Lsn) {
    assert!(after > before, "expected sync watermark to advance");
}

pub fn assert_decision_status_active(store: &Store, decision_id: DecisionId) {
    let decision = store
        .get_decision(decision_id)
        .expect("get decision")
        .expect("decision exists");
    assert_eq!(decision.status, context_graph::DecisionStatus::Active);
}

pub fn assert_sync_contract_has_real_table_policies() {
    let expected = expected_sync_table_policies();
    assert_eq!(
        expected.len(),
        TABLES.len(),
        "test expectation must classify every cg schema table"
    );
    for table in TABLES {
        let (expected_direction, expected_conflict) = expected
            .get(table.name)
            .unwrap_or_else(|| panic!("test expectation missing policy for {}", table.name));
        assert!(
            table.ddl.contains(expected_direction),
            "{} must declare `{expected_direction}` in its own DDL",
            table.name
        );
        match expected_conflict {
            Some(expected_conflict) => assert!(
                table.ddl.contains(expected_conflict),
                "{} must declare `{expected_conflict}` in its own DDL",
                table.name
            ),
            None => assert!(
                !table.ddl.contains("SYNC CONFLICT"),
                "{} is SYNC OFF and must not declare a conflict policy",
                table.name
            ),
        }
    }
}

fn expected_sync_table_policies() -> BTreeMap<&'static str, (&'static str, Option<&'static str>)> {
    let write_once = ("SYNC TWO WAY", Some("SYNC CONFLICT KEEP FIRST"));
    let keep_latest = ("SYNC TWO WAY", Some("SYNC CONFLICT KEEP LATEST"));
    let local_only = ("SYNC OFF", None);
    // Captured event history declares its local placement in `TABLES`: the event tables keep the
    // write-once conflict policy they carry when a tenant hub binding installs them, and the
    // placement, binding, locator, checkpoint and ordering tables stay node-local in both modes
    // (event capture atomicity, "context_bindings, event_source_checkpoints, …").
    let local_event_capture = ("SYNC OFF", Some("SYNC CONFLICT KEEP FIRST"));
    BTreeMap::from([
        ("cg_schema_version", write_once),
        ("contexts", write_once),
        ("event_placement", local_only),
        ("context_bindings", local_only),
        ("events", local_event_capture),
        ("event_anchors", local_event_capture),
        ("event_local_commits", local_only),
        ("event_source_locators", local_only),
        ("event_source_checkpoints", local_only),
        ("event_source_ordering", local_only),
        ("intentions", keep_latest),
        ("decisions", keep_latest),
        ("entities", keep_latest),
        ("observations", write_once),
        ("evidence", write_once),
        ("embedding_spaces", write_once),
        ("embedding_space_bindings", write_once),
        ("evidence_embedding_handles", write_once),
        ("entity_reference_embedding_handles", write_once),
        ("entity_reference_embeddings_128", write_once),
        ("entity_reference_embeddings_256", write_once),
        ("entity_reference_embeddings_512", write_once),
        ("entity_reference_embeddings_768", write_once),
        ("reindex_cursors", keep_latest),
        ("evidence_availability_log", write_once),
        ("outcomes", keep_latest),
        ("invalidations", keep_latest),
        ("signatures", write_once),
        ("signature_aggregates", write_once),
        ("signature_producer_lineage", write_once),
        ("digests", write_once),
        ("digest_embedding_handles", write_once),
        ("blueprints", write_once),
        ("edges", write_once),
        ("entity_snapshots", write_once),
        ("entity_snapshot_observations", write_once),
        ("decision_basis_snapshots", write_once),
        ("entity_current_snapshots", keep_latest),
        ("entity_snapshot_links", write_once),
        ("intention_snapshots", write_once),
        ("intention_current_snapshots", keep_latest),
        ("intention_snapshot_links", write_once),
        ("decision_intention_citations", write_once),
        ("approvals", write_once),
        ("audit_log", local_only),
        ("conflicts", write_once),
        ("precedent_matches", write_once),
        ("cg_learned_skills", keep_latest),
        ("cg_teaching_events", write_once),
        ("cg_skill_firing_trace", local_only),
        ("cg_review_cursor", local_only),
        ("cg_overlay_entries", write_once),
        ("cg_compensation_events", write_once),
        ("cg_pending_review_items", local_only),
        ("cg_source_chunks", local_only),
        ("cg_retrieval_touches", local_only),
        ("cg_local_settings", local_only),
        ("cg_understanding_completions", local_only),
    ])
}

pub fn assert_real_decision_content(store: &Store, id: DecisionId, expected_description: &str) {
    let actual = store
        .get_decision(id)
        .expect("read synced decision")
        .map(|decision| decision.description);
    assert_eq!(actual.as_deref(), Some(expected_description));
}

pub fn assert_decision_landed_in_context(
    store: &Store,
    context_id: ContextId,
    expected_description: &str,
) {
    let db = store.sync_database();
    let rows = db.scan("decisions", db.snapshot()).expect("scan decisions");
    assert!(
        rows.iter().any(|row| {
            row.values.get("context_id") == Some(&DbValue::Uuid(context_id.as_uuid()))
                && matches!(
                    row.values.get("description"),
                    Some(DbValue::Text(description)) if description == expected_description
                )
        }),
        "decision `{expected_description}` must persist in context {context_id}; rows={rows:?}"
    );
}

pub fn assert_real_intention_content(
    store: &Store,
    id: context_graph::IntentionId,
    expected_description: &str,
) {
    let actual = row_by_uuid(store, "intentions", id.as_uuid()).and_then(|row| {
        match row.get("description") {
            Some(DbValue::Text(description)) => Some(description.clone()),
            _ => None,
        }
    });
    assert_eq!(actual.as_deref(), Some(expected_description));
}

pub fn assert_real_observation_content(
    store: &Store,
    id: ObservationId,
    expected_source: &str,
    expected_state_field: &str,
    expected_state_value: serde_json::Value,
) {
    let observation = store.get_observation(id).expect("read synced observation");
    assert_eq!(observation.source, expected_source);
    let expected_state = Some(expected_state_value);
    assert_eq!(
        observation.state_delta.get(expected_state_field),
        Some(&expected_state),
        "synced observation must carry exact state_delta value for {expected_state_field}, got {:?}",
        observation.state_delta
    );
}

pub fn assert_real_evidence_content(
    store: &Store,
    id: EvidenceId,
    expected_observation_id: ObservationId,
    expected_kind: EvidenceKind,
    expected_source_ref: &str,
) {
    let evidence = store.get_evidence(id).expect("read synced evidence");
    assert_eq!(evidence.id, id);
    assert_eq!(evidence.observation_id, expected_observation_id);
    assert_eq!(evidence.kind, expected_kind);
    assert_eq!(evidence.source_ref, expected_source_ref);
}

pub fn assert_context_has_decision_and_digest(
    store: &Store,
    context_name: &str,
    expected_decision_description: &str,
) {
    let context = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|context| context.name == context_name)
        .unwrap_or_else(|| panic!("context {context_name} must exist"));
    assert_decision_landed_in_context(store, context.id, expected_decision_description);
    let digest_rows = store
        .sync_database()
        .scan("digests", store.sync_database().snapshot())
        .expect("scan digests")
        .into_iter()
        .filter(|row| {
            row.values.get("context_id") == Some(&DbValue::Uuid(context.id.as_uuid()))
                && matches!(
                    row.values.get("summary"),
                    Some(DbValue::Text(summary)) if summary.contains(expected_decision_description)
                )
        })
        .collect::<Vec<_>>();
    assert!(
        !digest_rows.is_empty(),
        "capture must write a Digest for `{expected_decision_description}` in context {context_name}"
    );
}

pub fn assert_real_entity_snapshot(
    store: &Store,
    id: EntitySnapshotId,
    expected_entity_id: context_graph::EntityId,
    expected_name: &str,
    expected_path: &str,
) {
    let row = row_by_uuid(store, "entity_snapshots", id.as_uuid())
        .unwrap_or_else(|| panic!("entity snapshot {id} must exist"));
    assert_eq!(row.get("id"), Some(&DbValue::Uuid(id.as_uuid())));
    assert_eq!(
        row.get("entity_id"),
        Some(&DbValue::Uuid(expected_entity_id.as_uuid()))
    );
    let data = match row.get("data") {
        Some(DbValue::Json(value)) => value,
        other => panic!("entity snapshot {id} must carry JSON data, got {other:?}"),
    };
    assert_eq!(
        data.get("name").and_then(serde_json::Value::as_str),
        Some(expected_name)
    );
    assert_eq!(
        data.pointer("/properties/path"),
        Some(&serde_json::json!(expected_path)),
        "entity snapshot must preserve exact path payload; data={data:?}"
    );
}

pub fn assert_real_digest_content(store: &Store, id: DigestId, expected_summary: &str) {
    let digest = store
        .get_digest(id)
        .expect("read synced digest")
        .expect("synced digest exists");
    assert_eq!(digest.summary, expected_summary);
}

pub fn assert_no_raw_row_by_uuid(store: &Store, table: &str, id: uuid::Uuid, message: &str) {
    let actual = row_by_uuid(store, table, id);
    assert!(actual.is_none(), "{message}; found {actual:?}");
}

pub fn assert_exact_extracted_from_edge(
    store: &Store,
    context_id: ContextId,
    source_id: uuid::Uuid,
    digest_id: DigestId,
) {
    let db = store.sync_database();
    let rows = db.scan("edges", db.snapshot()).expect("scan synced edges");
    let matching_edges = rows
        .iter()
        .filter(|row| {
            matches!(
                (
                    row.values.get("source_id"),
                    row.values.get("target_id"),
                    row.values.get("context_id"),
                    row.values.get("edge_type"),
                ),
                (
                    Some(DbValue::Uuid(actual_source)),
                    Some(DbValue::Uuid(actual_target)),
                    Some(DbValue::Uuid(actual_context)),
                    Some(DbValue::Text(edge_type)),
                ) if *actual_source == source_id
                    && *actual_target == digest_id.as_uuid()
                    && *actual_context == context_id.as_uuid()
                    && edge_type == "EXTRACTED_FROM"
            )
        })
        .count();
    assert!(
        matching_edges == 1,
        "expected exactly one synced EXTRACTED_FROM edge from {source_id} to {digest_id} in context {context_id}, got {matching_edges}; rows={rows:?}"
    );
}

pub fn assert_real_outcome_state(
    store: &Store,
    decision_id: DecisionId,
    expected: &Outcome,
    expected_count: usize,
) {
    let outcomes = store
        .admin_raw_outcomes_for_decision(decision_id)
        .expect("read synced outcomes");
    assert_eq!(outcomes.len(), expected_count, "active outcome count");
    assert!(
        outcomes.iter().any(|outcome| outcome.id == expected.id
            && outcome.success == expected.success
            && outcome.notes == expected.notes),
        "replacing outcome must sync by exact id and content; outcomes={outcomes:?}"
    );
}

pub fn assert_usage_observation_is_operational_metadata(store: &Store) {
    let db = store.sync_database();
    let rows = db
        .scan("observations", db.snapshot())
        .expect("scan observations");
    let usage_rows = rows
        .iter()
        .filter(|row| {
            matches!(
                row.values.get("observation_type"),
                Some(DbValue::Text(kind)) if kind == "provider_usage"
            )
        })
        .collect::<Vec<_>>();
    assert!(!usage_rows.is_empty(), "provider usage observation missing");
    for row in usage_rows {
        assert_provider_usage_basis_lane_empty(row.values.get("state_delta"), "state_delta");
        assert_provider_usage_basis_lane_empty(
            row.values.get("observed_properties"),
            "observed_properties",
        );
        assert!(
            format!("{:?}", row.values.get("properties")).contains("usage"),
            "usage must be persisted in operational properties: {:?}",
            row.values.get("properties")
        );
    }
}

pub fn assert_no_observation_basis_lane_contains_usage(store: &Store) {
    let db = store.sync_database();
    for row in db
        .scan("observations", db.snapshot())
        .expect("scan observations")
    {
        if matches!(
            row.values.get("observation_type"),
            Some(DbValue::Text(kind)) if kind == "provider_usage"
        ) {
            assert_provider_usage_basis_lane_empty(row.values.get("state_delta"), "state_delta");
            assert_provider_usage_basis_lane_empty(
                row.values.get("observed_properties"),
                "observed_properties",
            );
        } else {
            assert_no_usage_payload(
                row.values.get("state_delta"),
                "state_delta",
                "no observation state_delta may carry provider usage",
            );
            assert_no_usage_payload(
                row.values.get("observed_properties"),
                "observed_properties",
                "no observation observed_properties may carry provider usage",
            );
        }
    }
}

pub fn seed_decisions_based_on_usage_entities(store: &Store) -> Vec<DecisionId> {
    let targets = usage_observation_targets(store);
    assert!(
        !targets.is_empty(),
        "provider usage observation missing; cannot seed the model-entity basis guard"
    );
    targets
        .into_iter()
        .enumerate()
        .map(|(index, (context_id, entity_id))| {
            let intention = store
                .create_intention(CreateIntention {
                    description: format!("keep provider usage operational {}", index + 1),
                    status: IntentionStatus::Active,
                    origin: IntentionOrigin::Human,
                    context_id,
                    ..Default::default()
                })
                .expect("create provider usage guard intention");
            store
                .create_decision(CreateDecision {
                    decision_type: "operational-boundary".to_string(),
                    description: format!("provider usage target remains non-basis {}", index + 1),
                    intention_ids: vec![intention.id],
                    based_on_entity_ids: vec![entity_id],
                    context_id,
                    ..Default::default()
                })
                .expect("create provider usage guard decision")
                .id
        })
        .collect()
}

fn usage_observation_targets(store: &Store) -> Vec<(ContextId, EntityId)> {
    let db = store.sync_database();
    let mut targets = BTreeSet::new();
    for row in db
        .scan("observations", db.snapshot())
        .expect("scan observations")
    {
        if !matches!(
            row.values.get("observation_type"),
            Some(DbValue::Text(kind)) if kind == "provider_usage"
        ) {
            continue;
        }
        let context_id = match row.values.get("context_id") {
            Some(DbValue::Uuid(id)) => ContextId::from(*id),
            other => panic!("provider usage observation missing context_id: {other:?}"),
        };
        let entity_id = match row.values.get("entity_id") {
            Some(DbValue::Uuid(id)) => EntityId::from(*id),
            other => panic!("provider usage observation missing entity_id: {other:?}"),
        };
        targets.insert((context_id, entity_id));
    }
    targets.into_iter().collect()
}

pub fn assert_usage_does_not_affect_basis_warnings(store: &Store, guarded: &[DecisionId]) {
    for decision_id in guarded {
        let decision = store
            .get_decision(*decision_id)
            .expect("read provider usage guard decision")
            .expect("provider usage guard decision exists");
        assert_eq!(
            decision.status,
            DecisionStatus::Active,
            "provider usage metadata must not invalidate a decision even when it cites the usage observation target"
        );
    }
    for context in store.list_contexts().expect("list contexts") {
        let standing = context_graph::ingest::standing_impact(store, context.id)
            .expect("standing impact must read");
        assert_warning_surface_mentions_no_usage("standing impact", &standing, guarded);
        let changed = context_graph::ingest::changed_since_context_cursor(store, context.id)
            .expect("changed impact must read");
        assert_warning_surface_mentions_no_usage("changed impact", &changed, guarded);
    }
}

fn assert_warning_surface_mentions_no_usage(
    surface: &str,
    notices: &[context_graph::ingest::MemoryInvalidationNotice],
    guarded: &[DecisionId],
) {
    let rendered = format!("{notices:?}");
    assert!(
        !rendered.contains("usage")
            && !rendered.contains("token")
            && !rendered.contains("cost")
            && !rendered.contains("elapsed"),
        "usage/cost metadata must not appear in {surface} notices: {rendered}"
    );
    assert!(
        notices.iter().all(|notice| {
            notice
                .affected_decision_id
                .is_none_or(|decision_id| !guarded.contains(&decision_id))
        }),
        "provider usage metadata must not create {surface} warnings for decisions based on the usage observation target: {notices:?}"
    );
}

fn assert_no_usage_payload(value: Option<&DbValue>, lane: &str, message: &str) {
    let rendered = format!("{value:?}");
    let lowered = rendered.to_ascii_lowercase();
    assert!(
        !lowered.contains("usage")
            && !lowered.contains("prompt_tokens")
            && !lowered.contains("completion_tokens")
            && !lowered.contains("total_tokens")
            && !lowered.contains("provider_tokens")
            && !lowered.contains("token_count")
            && !contains_forbidden_usage_basis_value(value),
        "{message} ({lane}): {rendered}"
    );
}

fn assert_provider_usage_basis_lane_empty(value: Option<&DbValue>, lane: &str) {
    match (lane, value) {
        ("observed_properties", Some(DbValue::Json(serde_json::Value::Object(properties))))
            if properties.is_empty() => {}
        ("state_delta", Some(DbValue::Json(serde_json::Value::Object(delta)))) => {
            let format = delta
                .get("format")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let ops = delta
                .get("ops")
                .and_then(serde_json::Value::as_object)
                .unwrap_or_else(|| {
                    panic!("provider usage state_delta missing ops object: {value:?}")
                });
            assert_eq!(
                format, "cg_state_delta_v1",
                "provider usage state_delta must use the state-delta envelope: {value:?}"
            );
            assert!(
                ops.is_empty(),
                "provider usage must not persist any basis-facing state_delta entries, even under neutral keys: {value:?}"
            );
        }
        _ => {
            panic!(
                "provider usage must leave {lane} structurally empty rather than hiding cost metadata under neutral keys: {value:?}"
            );
        }
    }
}

fn contains_forbidden_usage_basis_value(value: Option<&DbValue>) -> bool {
    match value {
        Some(DbValue::Int64(value)) => FORBIDDEN_USAGE_BASIS_VALUES.contains(value),
        Some(DbValue::Float64(value)) => {
            value.fract() == 0.0 && FORBIDDEN_USAGE_BASIS_VALUES.contains(&(*value as i64))
        }
        Some(DbValue::Text(text)) => FORBIDDEN_USAGE_BASIS_VALUES
            .iter()
            .any(|forbidden| text.contains(&forbidden.to_string())),
        Some(DbValue::Json(value)) => json_contains_forbidden_usage_basis_value(value),
        _ => false,
    }
}

fn json_contains_forbidden_usage_basis_value(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Number(number) => {
            number
                .as_i64()
                .is_some_and(|value| FORBIDDEN_USAGE_BASIS_VALUES.contains(&value))
                || number
                    .as_u64()
                    .and_then(|value| i64::try_from(value).ok())
                    .is_some_and(|value| FORBIDDEN_USAGE_BASIS_VALUES.contains(&value))
        }
        serde_json::Value::String(text) => FORBIDDEN_USAGE_BASIS_VALUES
            .iter()
            .any(|forbidden| text.contains(&forbidden.to_string())),
        serde_json::Value::Array(values) => {
            values.iter().any(json_contains_forbidden_usage_basis_value)
        }
        serde_json::Value::Object(values) => values
            .values()
            .any(json_contains_forbidden_usage_basis_value),
        _ => false,
    }
}

pub struct TestSyncStatusBoundary<'a>(pub &'a SyncClient);

impl SyncStatusBoundary for TestSyncStatusBoundary<'_> {
    fn push_watermark(&self) -> Lsn {
        self.0.push_watermark()
    }

    fn pull_watermark(&self) -> Lsn {
        self.0.pull_watermark()
    }

    fn has_pending_push_changes(&self) -> context_graph::Result<bool> {
        self.0
            .has_pending_push_changes()
            .map_err(|err| context_graph::CgError::Engine(err.to_string()))
    }

    fn pending_push_change_count(&self) -> context_graph::Result<usize> {
        Ok(expected_pending_push_change_count(self.0))
    }

    fn is_connected(&self) -> bool {
        true
    }
}

fn row_by_uuid(store: &Store, table: &str, id: uuid::Uuid) -> Option<HashMap<String, DbValue>> {
    let db = store.sync_database();
    db.point_lookup(table, "id", &DbValue::Uuid(id), db.snapshot())
        .expect("point lookup")
        .map(|row| row.values)
}
