//! A projection run keeps committing while direct readers continuously open, read status, and drop
//! their Store views.
//!
//! Split from `event_search_projection_correctness.rs` into its own test binary: this reader
//! pressure test alone peaks near 3 GB of resident memory, and sharing a binary with the other
//! correctness tests pushed that binary past the per-binary memory cap.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::panic::{self, AssertUnwindSafe, PanicHookInfo};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use context_graph::ConsumerReader;
use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, CgError, ContextId, ContextSelector, CreateContext, DistanceMetric, Embedder,
    EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchBuildReceipt,
    EventSearchProfile, EventSearchProjectionRunner, EventSearchRunRequest, EvidenceKind,
    EvidenceProducer, Result, Store, StoreConfig, event_history_reader_declaration,
};
use contextdb_core::Value;
use serde_json::json;

const CONNECTOR: &str = "event-search-correctness";
const SESSION: &str = "event-search-correctness-session";
const SPACE: &str = "event-search-correctness-768";

type Observer = Box<dyn Fn() + Send + Sync>;

type PanicHook = dyn for<'a> Fn(&PanicHookInfo<'a>) + Send + Sync + 'static;

/// The panic hook is process-global. This binary has one test, but the lock also keeps a future
/// sibling from replacing its hook while this proof is observing projection-owned background work.
struct OwnedBackgroundPanicProbe {
    _serial: MutexGuard<'static, ()>,
    previous: Arc<PanicHook>,
    panics: Arc<Mutex<Vec<String>>>,
}

impl OwnedBackgroundPanicProbe {
    fn install() -> Self {
        static PANIC_HOOK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

        let serial = PANIC_HOOK_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("panic-hook serialization lock");
        let panics = Arc::new(Mutex::new(Vec::new()));
        let observed_panics = Arc::clone(&panics);
        let previous: Arc<PanicHook> = Arc::from(std::panic::take_hook());
        let chained = Arc::clone(&previous);
        std::panic::set_hook(Box::new(move |info| {
            observed_panics
                .lock()
                .expect("background panic log")
                .push(info.to_string());
            chained(info);
        }));
        Self {
            _serial: serial,
            previous,
            panics,
        }
    }

    fn assert_no_background_panics(&self) {
        let panics = self.panics.lock().expect("background panic log").clone();
        assert!(
            panics.is_empty(),
            "projection-owned background work panicked under reader pressure: {panics:#?}"
        );
    }
    fn restore(self) {
        let previous = Arc::clone(&self.previous);
        std::panic::set_hook(Box::new(move |info| previous(info)));
    }
}

fn with_owned_background_panic_probe<T>(body: impl FnOnce(&OwnedBackgroundPanicProbe) -> T) -> T {
    let probe = OwnedBackgroundPanicProbe::install();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let result = body(&probe);
        probe.assert_no_background_panics();
        result
    }));
    probe.restore();
    match result {
        Ok(result) => result,
        Err(payload) => panic::resume_unwind(payload),
    }
}

#[derive(Debug)]
enum ReaderOpenRefusal {
    StoreLocked,
    StoreHeldByReaders,
}

fn classify_reader_open_refusal(error: CgError) -> std::result::Result<ReaderOpenRefusal, CgError> {
    match error {
        CgError::StoreLocked { .. } => Ok(ReaderOpenRefusal::StoreLocked),
        CgError::StoreHeldByReaders { .. } => Ok(ReaderOpenRefusal::StoreHeldByReaders),
        other => Err(other),
    }
}

/// Role-aware fixture tokenization (five model tokens plus two per scalar), with two controls:
/// a one-shot failure for any document containing a marker, and an observer run at every
/// document preparation.
struct ControlledEmbedder {
    fail_once_on: Mutex<Option<char>>,
    observer: Mutex<Option<Observer>>,
    /// Run at every prepared-document embed call.
    embed_observer: Mutex<Option<Observer>>,
    embedded_documents: Mutex<Vec<String>>,
}

impl ControlledEmbedder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            fail_once_on: Mutex::new(None),
            observer: Mutex::new(None),
            embed_observer: Mutex::new(None),
            embedded_documents: Mutex::new(Vec::new()),
        })
    }

    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }
}

impl Embedder for ControlledEmbedder {
    fn embed(&self, _: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search uses role-aware preparation")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "test".into(),
            model_name: "correctness-fixture".into(),
            model_version: "1".into(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: SPACE.into(),
                table_name: "event_search_passages".into(),
                vector_column: "vector_text".into(),
            }],
        }
    }

    fn encoding_profile_id(&self) -> String {
        "event-search-correctness-v1".into()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(256)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        if let Some(observer) = self.observer.lock().expect("observer lock").as_ref() {
            observer();
        }
        let mut payload = vec![0; 5];
        let mut boundaries = vec![(0, 0); 5];
        for (start, scalar) in text.char_indices() {
            let end = start + scalar.len_utf8();
            payload.extend([scalar as u32, scalar as u32]);
            boundaries.extend([(start, end), (start, end)]);
        }
        seal.document_builder()
            .token_count(payload.len())
            .byte_boundaries(boundaries)
            .payload(payload)
            .seal()
    }

    fn prepare_query_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        let payload = text
            .chars()
            .flat_map(|scalar| [scalar as u32, scalar as u32])
            .collect::<Vec<_>>();
        seal.query_builder()
            .token_count(payload.len())
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        if let Some(observer) = self.embed_observer.lock().expect("observer lock").as_ref() {
            observer();
        }
        let text: String = inference
            .token_ids()
            .iter()
            .filter(|token| **token != 0)
            .map(|token| char::from_u32(*token).expect("fixture token is Unicode"))
            .collect();
        let mut fail = self.fail_once_on.lock().expect("failure control lock");
        if fail.is_some_and(|marker| text.contains(marker)) {
            *fail = None;
            return Err(CgError::Engine(
                "fixture worker killed during embedding".into(),
            ));
        }
        drop(fail);
        self.embedded_documents
            .lock()
            .expect("embedded document log")
            .push(text);
        Ok(EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: CONNECTOR.into(),
                model_name: "correctness-fixture".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: vec![inference.token_count() as f32; 768],
        })
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: CONNECTOR.into(),
                model_name: "correctness-fixture".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: vec![inference.token_count() as f32; 768],
        })
    }
}

fn config(path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path: path,
        extra_embedding_spaces: vec![ControlledEmbedder::new().embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "correctness-v1".into(),
        anchor_declaration_version: "correctness-v1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn draft(key: &str, ordinal: u64, role: EventRenderRole, text: &str) -> EventDraft {
    EventDraft {
        session_ref: SESSION.into(),
        source_event_key: key.into(),
        source_position: json!({"ordinal": ordinal}),
        occurred_at: None,
        author: "Renée".into(),
        kind: "message".into(),
        render_role: role,
        interaction_ref: Some("correctness-pair".into()),
        text: Some(text.into()),
        payload: json!({}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn profile() -> EventSearchProfile {
    EventSearchProfile {
        target_passage_tokens: NonZeroU32::new(177).expect("nonzero target"),
        hard_passage_tokens: NonZeroU32::new(256).expect("nonzero hard limit"),
        ordinary_overlap_events: NonZeroU32::new(1).expect("nonzero overlap"),
        oversized_event_overlap_tokens: NonZeroU32::new(4).expect("nonzero oversized overlap"),
        ..EventSearchProfile::default()
    }
}

fn run(
    config: &StoreConfig,
    context_id: ContextId,
    embedder: &Arc<ControlledEmbedder>,
    max_events_per_batch: u32,
    lease_ms: u64,
) -> Result<EventSearchBuildReceipt> {
    EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config.clone(),
        contexts: ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: profile(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(max_events_per_batch).expect("nonzero batch"),
        max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
        max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
        lease_duration_ms: NonZeroU64::new(lease_ms).expect("nonzero lease"),
    })
}

fn open(config: &StoreConfig) -> Store {
    Store::open(config.clone()).expect("open Store")
}

fn create_context(store: &Store) -> ContextId {
    store
        .create_context(CreateContext {
            name: "event-search correctness".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id
}

fn lease_rows(store: &Store, context_id: ContextId) -> usize {
    store
        .execute_sql_for_test(
            "SELECT owner_token FROM event_search_leases WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("inspect projection leases")
        .rows
        .len()
}

#[test]
fn a_run_reopening_the_store_under_continuous_readers_commits_every_batch() {
    with_owned_background_panic_probe(|_| {
        a_run_reopening_the_store_under_continuous_readers_body()
    });
}

fn a_run_reopening_the_store_under_continuous_readers_body() {
    // Long enough that nothing here is about lease expiry: this test is about readers.
    let unexpiring_lease_ms = 600_000;
    for scenario in 0..6 {
        let root = tempfile::tempdir().expect("tempdir");
        let config = config(root.path().join(format!("readers-{scenario}.db")));
        let store = open(&config);
        let context_id = create_context(&store);
        let drafts = (1..=8)
            .map(|sequence| {
                let role = if sequence % 2 == 1 {
                    EventRenderRole::HumanMessage
                } else {
                    EventRenderRole::AgentMessage
                };
                draft(
                    &format!("event-{sequence}"),
                    sequence,
                    role,
                    &format!("passage text {sequence}"),
                )
            })
            .collect::<Vec<_>>();
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                "reader-pressure",
                drafts,
            ))
            .expect("append the stream");
        drop(store);

        let stop = AtomicBool::new(false);
        let build_active = AtomicBool::new(false);
        let reader_attempts = AtomicUsize::new(0);
        let active_read_successes = AtomicUsize::new(0);
        let contention_refusals = Mutex::new(Vec::new());
        let unexpected_refusals = Mutex::new(Vec::new());
        let receipt = std::thread::scope(|scope| {
            for _ in 0..3 {
                let config = &config;
                let stop = &stop;
                let build_active = &build_active;
                let reader_attempts = &reader_attempts;
                let active_read_successes = &active_read_successes;
                let contention_refusals = &contention_refusals;
                let unexpected_refusals = &unexpected_refusals;
                scope.spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        reader_attempts.fetch_add(1, Ordering::Relaxed);
                        match ConsumerReader::open(
                            &config.db_path,
                            event_history_reader_declaration(),
                        ) {
                            Ok(reader) => match reader
                                .event_history_status(ContextSelector::one(context_id))
                            {
                                Ok(_) if build_active.load(Ordering::Acquire) => {
                                    active_read_successes.fetch_add(1, Ordering::Relaxed);
                                }
                                Ok(_) => {}
                                Err(error) => match classify_reader_open_refusal(error) {
                                    Ok(contention) => contention_refusals
                                        .lock()
                                        .expect("typed contention log")
                                        .push(contention),
                                    Err(error) => unexpected_refusals
                                        .lock()
                                        .expect("unexpected reader refusal log")
                                        .push(error),
                                },
                            },
                            Err(error) => match classify_reader_open_refusal(error) {
                                Ok(contention) => contention_refusals
                                    .lock()
                                    .expect("typed contention log")
                                    .push(contention),
                                Err(error) => unexpected_refusals
                                    .lock()
                                    .expect("unexpected reader refusal log")
                                    .push(error),
                            },
                        }
                    }
                });
            }
            let embedder = ControlledEmbedder::new();
            build_active.store(true, Ordering::Release);
            let receipt = run(&config, context_id, &embedder, 1, unexpiring_lease_ms);
            build_active.store(false, Ordering::Release);
            stop.store(true, Ordering::Relaxed);
            receipt
        });
        let receipt =
            receipt.expect("the run commits every batch it claims while readers hold the Store");
        assert!(
            reader_attempts.load(Ordering::Relaxed) >= 3,
            "scenario {scenario} kept all three reader loops working"
        );
        assert!(
            active_read_successes.load(Ordering::Relaxed) > 0,
            "scenario {scenario} has at least one successful status read while the build is active"
        );
        let unexpected_refusals = unexpected_refusals
            .into_inner()
            .expect("unexpected reader refusal log");
        let classified_contention_refusals = contention_refusals
            .into_inner()
            .expect("typed contention log");
        assert!(
            unexpected_refusals.is_empty(),
            "scenario {scenario} saw a reader refusal outside the established typed busy/held \
             cases: {unexpected_refusals:#?}; classified typed contention: \
             {classified_contention_refusals:#?}"
        );
        assert_eq!(
            receipt.source_watermark.event_count, 8,
            "scenario {scenario} covers the whole stream"
        );
        assert_eq!(
            receipt.embedded_event_count, 8,
            "scenario {scenario} embedded every event"
        );

        let after = open(&config);
        assert_eq!(
            lease_rows(&after, context_id),
            0,
            "scenario {scenario} left no claim behind"
        );
    }
}
