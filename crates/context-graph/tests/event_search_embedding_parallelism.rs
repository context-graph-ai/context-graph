//! Contract for the default concurrent event-search embedding runner.
//! The controlled embedder reports entered calls over per-instance channels and holds only the
//! first two calls, so overlap is an observed event rather than an elapsed-time success oracle.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use cg_node_fs::{read_event_search_worker_policy, write_event_search_worker_policy};
use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
};
use context_graph::{
    AppendEventBatch, ContextId, CreateContext, DistanceMetric, Embedder, EmbedderRegistration,
    EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchBuildRequest,
    EventSearchProfile, EventSearchProjectionRunner, EventSearchRunRequest, EventSearchWorkSpec,
    EvidenceKind, EvidenceProducer, Result, Store, StoreConfig,
};
use serde_json::json;

const SPACE: &str = "event-search-parallelism-red-768";

struct EnteredCall {
    ordinal: usize,
    text: String,
    #[cfg(feature = "test-support")]
    helper_threads: usize,
    release: Option<Sender<()>>,
}

impl EnteredCall {
    fn release(self) {
        if let Some(release) = self.release {
            release.send(()).expect("release held embedding call");
        }
    }
}

struct ControlledEmbedder {
    entered: Sender<EnteredCall>,
    call_index: AtomicUsize,
    held_calls: usize,
}

impl ControlledEmbedder {
    fn new() -> (Arc<Self>, Receiver<EnteredCall>) {
        Self::with_held_calls(2)
    }

    fn with_held_calls(held_calls: usize) -> (Arc<Self>, Receiver<EnteredCall>) {
        let (entered, observed) = channel();
        (
            Arc::new(Self {
                entered,
                call_index: AtomicUsize::new(0),
                held_calls,
            }),
            observed,
        )
    }

    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn output(&self, inference: &EventSearchInference) -> EmbeddingOutput {
        let mut vector = vec![0.0; 768];
        vector[0] = inference.token_count() as f32;
        EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: vector.len(),
            producer: EvidenceProducer {
                system: "parallelism-red".into(),
                model_name: "controlled".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector,
        }
    }

    fn input_text(inference: &EventSearchInference) -> String {
        inference
            .token_ids()
            .iter()
            .filter_map(|token| char::from_u32(*token))
            .collect()
    }
}

impl Embedder for ControlledEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search must use prepared document inference")
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
            model_name: "controlled".into(),
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
        SPACE.into()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(512)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let mut payload = Vec::new();
        let mut boundaries = Vec::new();
        for (start, scalar) in text.char_indices() {
            payload.push(scalar as u32);
            boundaries.push((start, start + scalar.len_utf8()));
        }
        seal.document_builder()
            .token_count(payload.len())
            .byte_boundaries(boundaries)
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        let ordinal = self.call_index.fetch_add(1, Ordering::SeqCst);
        let text = Self::input_text(inference);
        #[cfg(feature = "test-support")]
        let helper_threads = rayon::current_num_threads();
        if ordinal < self.held_calls {
            let (release, held) = channel();
            self.entered
                .send(EnteredCall {
                    ordinal,
                    text,
                    #[cfg(feature = "test-support")]
                    helper_threads,
                    release: Some(release),
                })
                .map_err(|_| {
                    context_graph::CgError::Engine("parallelism observer went away".into())
                })?;
            held.recv().map_err(|_| {
                context_graph::CgError::Engine("parallelism test release disconnected".into())
            })?;
        } else {
            self.entered
                .send(EnteredCall {
                    ordinal,
                    text,
                    #[cfg(feature = "test-support")]
                    helper_threads,
                    release: None,
                })
                .map_err(|_| {
                    context_graph::CgError::Engine("parallelism observer went away".into())
                })?;
        }
        Ok(self.output(inference))
    }
}

#[cfg(feature = "test-support")]
#[test]
fn concurrent_embedding_calls_run_with_one_rayon_helper_each_for_standalone_and_held_store() {
    let standalone = start_projection(NonZeroUsize::new(2).expect("positive worker cap"), None);
    let first = standalone
        .entered
        .recv()
        .expect("standalone first embedding call");
    let second = standalone
        .entered
        .recv()
        .expect("standalone second embedding call");
    assert_eq!(
        first.helper_threads, 1,
        "standalone first call helper count"
    );
    assert_eq!(
        second.helper_threads, 1,
        "standalone second call helper count"
    );
    let (outcome, held) = finish_projection(standalone, vec![first, second]);
    assert_eq!(held, 2, "standalone fixture observed both held calls");
    assert!(
        outcome.result.is_ok(),
        "standalone projection failed: {:?}",
        outcome.result
    );

    let held_store =
        start_held_store_projection(NonZeroUsize::new(2).expect("positive worker cap"));
    let first = held_store
        .entered
        .recv()
        .expect("held-Store first embedding call");
    let second = held_store
        .entered
        .recv()
        .expect("held-Store second embedding call");
    assert_eq!(
        first.helper_threads, 1,
        "held-Store first call helper count"
    );
    assert_eq!(
        second.helper_threads, 1,
        "held-Store second call helper count"
    );
    first.release();
    second.release();
    assert!(
        held_store
            .runner
            .join()
            .expect("held-Store runner thread joins")
            .is_ok(),
        "held-Store projection failed"
    );
}

fn config(path: std::path::PathBuf, space: EmbeddingSpace) -> StoreConfig {
    StoreConfig {
        db_path: path,
        extra_embedding_spaces: vec![space],
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn captured_context(store: &Store, close_every_two_events: bool) -> ContextId {
    let context_id = store
        .create_context(CreateContext {
            name: "parallel embedding red".into(),
            ..CreateContext::default()
        })
        .expect("create isolated Context")
        .id;
    let drafts = (0..4)
        .map(|ordinal| EventDraft {
            session_ref: if close_every_two_events {
                format!("parallel-session-{}", ordinal / 2)
            } else {
                "parallel-session".into()
            },
            source_event_key: format!("human-{ordinal}"),
            source_position: json!({"ordinal": ordinal}),
            occurred_at: None,
            author: "fixture".into(),
            kind: "message".into(),
            render_role: EventRenderRole::HumanMessage,
            interaction_ref: None,
            text: Some(format!("independent passage number {ordinal}")),
            payload: json!({}),
            anchors: json!({}),
            source_remainder: json!({}),
            coverage: EventCoverage::Complete,
        })
        .collect();
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            EventAdapterDeclaration {
                connector_id: "parallelism-red".into(),
                adapter_version: "1".into(),
                anchor_declaration_version: "1".into(),
                anchor_schema: json!({"type": "object", "additionalProperties": true}),
                source_position_order: vec!["/ordinal".into()],
            },
            "parallel-source",
            drafts,
        ))
        .expect("capture independent passage inputs");
    context_id
}

struct RunningProjection {
    _root: tempfile::TempDir,
    entered: Receiver<EnteredCall>,
    waves: Receiver<usize>,
    runner: std::thread::JoinHandle<context_graph::EventSearchRunOutcome>,
}

struct HeldStoreProjection {
    _root: tempfile::TempDir,
    entered: Receiver<EnteredCall>,
    waves: Receiver<usize>,
    probe: Store,
    context_id: ContextId,
    spec: EventSearchWorkSpec,
    runner: std::thread::JoinHandle<Result<context_graph::EventSearchBuildReceipt>>,
}

fn start_projection(
    embedding_workers: NonZeroUsize,
    resolutions: Option<Arc<AtomicUsize>>,
) -> RunningProjection {
    start_projection_with_event_batch(embedding_workers, resolutions, NonZeroU32::new(8).unwrap())
}

fn start_projection_with_event_batch(
    embedding_workers: NonZeroUsize,
    resolutions: Option<Arc<AtomicUsize>>,
    max_events_per_batch: NonZeroU32,
) -> RunningProjection {
    let root = tempfile::tempdir().expect("isolated store root");
    let (embedder, entered) = ControlledEmbedder::new();
    start_projection_with_embedder(
        root,
        embedder,
        entered,
        embedding_workers,
        resolutions.map(|resolutions| {
            Box::new(move || {
                resolutions.fetch_add(1, Ordering::SeqCst);
                NonZeroUsize::new(2).unwrap()
            }) as Box<dyn FnMut() -> NonZeroUsize + Send>
        }),
        max_events_per_batch,
        false,
    )
}

fn start_projection_with_embedder(
    root: tempfile::TempDir,
    embedder: Arc<ControlledEmbedder>,
    entered: Receiver<EnteredCall>,
    embedding_workers: NonZeroUsize,
    mut resolver: Option<Box<dyn FnMut() -> NonZeroUsize + Send>>,
    max_events_per_batch: NonZeroU32,
    close_every_two_events: bool,
) -> RunningProjection {
    let config = config(
        root.path().join("parallelism.db"),
        embedder.embedding_space(),
    );
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open isolated store");
    let context_id = captured_context(&store, close_every_two_events);
    drop(store);

    let request = EventSearchRunRequest {
        store: config,
        contexts: context_graph::ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: EventSearchProfile {
            target_passage_tokens: NonZeroU32::new(1).unwrap(),
            hard_passage_tokens: NonZeroU32::new(256).unwrap(),
            ordinary_overlap_events: NonZeroU32::new(1).unwrap(),
            oversized_event_overlap_tokens: NonZeroU32::new(8).unwrap(),
            ..EventSearchProfile::default()
        },
        embedding_workers,
        max_events_per_batch,
        max_passages_per_batch: NonZeroU32::new(16).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(32_768).unwrap(),
        lease_duration_ms: NonZeroU64::new(30_000).unwrap(),
    };
    let (wave_tx, waves) = channel();
    let runner = std::thread::spawn(move || {
        context_graph::test_support::observe_embedding_waves_for_test(move |width| {
            let _ = wave_tx.send(width);
        });
        match resolver.as_mut() {
            Some(resolver) => {
                EventSearchProjectionRunner::catch_up_observed_with_embedding_workers(
                    request, resolver,
                )
            }
            None => EventSearchProjectionRunner::catch_up_observed(request),
        }
    });
    RunningProjection {
        _root: root,
        entered,
        waves,
        runner,
    }
}

fn start_held_store_projection(embedding_workers: NonZeroUsize) -> HeldStoreProjection {
    let root = tempfile::tempdir().expect("isolated held-Store root");
    let (embedder, entered) = ControlledEmbedder::new();
    let config = config(
        root.path().join("held-store-parallelism.db"),
        embedder.embedding_space(),
    );
    let store = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("open held-Store fixture");
    let context_id = captured_context(&store, false);
    let profile = EventSearchProfile {
        target_passage_tokens: NonZeroU32::new(1).unwrap(),
        hard_passage_tokens: NonZeroU32::new(256).unwrap(),
        ordinary_overlap_events: NonZeroU32::new(1).unwrap(),
        oversized_event_overlap_tokens: NonZeroU32::new(8).unwrap(),
        ..EventSearchProfile::default()
    };
    let registration = embedder.registration();
    let spec = EventSearchWorkSpec {
        profile: profile.clone(),
        embedding_space_id: registration.space.embedding_space_id.clone(),
        encoding_profile_id: registration.embedder.encoding_profile_id(),
    };
    let request = EventSearchBuildRequest {
        context_id,
        embedder: registration,
        profile,
        embedding_workers,
        max_events_per_batch: NonZeroU32::new(8).unwrap(),
        max_passages_per_batch: NonZeroU32::new(16).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(32_768).unwrap(),
        lease_duration_ms: NonZeroU64::new(30_000).unwrap(),
    };
    let probe = store.clone();
    let (wave_tx, waves) = channel();
    let runner = std::thread::spawn(move || {
        context_graph::test_support::observe_embedding_waves_for_test(move |width| {
            let _ = wave_tx.send(width);
        });
        store.catch_up_event_search(request)
    });
    HeldStoreProjection {
        _root: root,
        entered,
        waves,
        probe,
        context_id,
        spec,
        runner,
    }
}

fn held_store_has_live_lease(run: &HeldStoreProjection) -> bool {
    let work = run
        .probe
        .event_search_work(
            context_graph::ContextSelector::one(run.context_id),
            &run.spec,
        )
        .expect("inspect the held Store's projection lease");
    assert_eq!(work.len(), 1, "the held-Store fixture has one Context");
    work[0].live_projection_lease
}

fn finish_projection(
    run: RunningProjection,
    observed: Vec<EnteredCall>,
) -> (context_graph::EventSearchRunOutcome, usize) {
    finish_projection_with_held_count(run, observed, 2)
}

fn finish_projection_with_held_count(
    run: RunningProjection,
    observed: Vec<EnteredCall>,
    expected_held: usize,
) -> (context_graph::EventSearchRunOutcome, usize) {
    let mut held_observed = observed
        .iter()
        .filter(|call| call.release.is_some())
        .count();
    for call in observed {
        call.release();
    }
    while held_observed < expected_held {
        let Ok(call) = run.entered.recv() else {
            break;
        };
        if call.release.is_some() {
            held_observed += 1;
        }
        call.release();
    }
    (
        run.runner.join().expect("projection runner thread joins"),
        held_observed,
    )
}

#[test]
fn default_runner_enters_two_embedding_calls_before_either_is_released() {
    let available = std::thread::available_parallelism().expect("host parallelism is available");
    if available.get() <= 1 {
        return;
    }
    let run = start_projection(available, None);
    let first = run.entered.recv().ok();
    let first_observed = first.is_some();
    let second = run.entered.recv();
    let overlap_observed = second.is_ok();
    let mut observed = first.into_iter().collect::<Vec<_>>();
    if let Ok(second) = second {
        observed.push(second);
    }
    let (outcome, held_observed) = finish_projection(run, observed);
    assert!(
        first_observed,
        "fixture never entered its first embedding call"
    );
    assert_eq!(held_observed, 2, "fixture did not observe both held calls");
    assert!(
        outcome.result.is_ok(),
        "fixture projection must otherwise succeed: {:?}",
        outcome.result
    );
    assert!(
        outcome.embedding_invocations >= 2,
        "fixture must create at least two actual passage inference calls"
    );
    assert!(
        overlap_observed,
        "default event-search runner serialized the first two embedding calls"
    );
}

#[test]
fn embedding_worker_cap_one_keeps_the_second_call_out_until_the_first_returns() {
    let run = start_projection(NonZeroUsize::MIN, None);
    let first = run.entered.recv().ok();
    let first_observed = first.is_some();
    assert_eq!(run.waves.recv().expect("wave admission"), 1);
    let second_before_release = run.entered.try_recv();
    let overlap_observed = second_before_release.is_ok();
    let mut observed = first.into_iter().collect::<Vec<_>>();
    if let Ok(second) = second_before_release {
        observed.push(second);
    }
    let (outcome, held_observed) = finish_projection(run, observed);

    assert!(
        first_observed,
        "fixture never entered its first embedding call"
    );
    assert_eq!(held_observed, 2, "fixture did not observe both held calls");
    assert!(
        outcome.result.is_ok(),
        "cap-one fixture projection must otherwise succeed: {:?}",
        outcome.result
    );
    assert!(
        outcome.embedding_invocations >= 2,
        "cap-one fixture must make at least two actual passage calls"
    );
    assert!(!overlap_observed, "embedding worker cap 1 allowed overlap");
}

#[test]
fn embedding_worker_cap_two_enters_two_calls_but_never_a_third_before_release() {
    let run = start_projection(NonZeroUsize::new(2).unwrap(), None);
    let first = run.entered.recv().ok();
    let first_observed = first.is_some();
    let second = run.entered.recv();
    assert_eq!(run.waves.recv().expect("wave admission"), 2);
    let third_before_release = second
        .as_ref()
        .ok()
        .and_then(|_| run.entered.try_recv().ok());
    let second_observed = second.is_ok();
    let third_ordinal = third_before_release.as_ref().map(|call| call.ordinal);
    let mut observed = first.into_iter().collect::<Vec<_>>();
    if let Ok(second) = second {
        observed.push(second);
    }
    if let Some(third) = third_before_release {
        observed.push(third);
    }
    let (outcome, held_observed) = finish_projection(run, observed);

    assert!(
        first_observed,
        "fixture never entered its first embedding call"
    );
    assert_eq!(held_observed, 2, "fixture did not observe both held calls");
    assert!(
        outcome.result.is_ok(),
        "cap-two fixture projection must otherwise succeed: {:?}",
        outcome.result
    );
    assert!(
        second_observed,
        "embedding worker cap 2 was ignored as serial"
    );
    assert_eq!(
        third_ordinal, None,
        "embedding worker cap 2 admitted a third call before release"
    );
}

#[test]
fn held_store_cap_two_overlaps_two_calls_while_its_claim_lease_stays_live() {
    let run = start_held_store_projection(NonZeroUsize::new(2).unwrap());
    let first = run.entered.recv().ok();
    let second = run.entered.recv().ok();
    assert_eq!(
        run.waves
            .recv()
            .expect("complete held-Store wave admission"),
        2
    );
    let third = second.as_ref().and_then(|_| run.entered.try_recv().ok());

    assert!(
        first.is_some(),
        "held Store never entered its first embedding call"
    );
    assert!(
        second.is_some(),
        "held Store cap 2 serialized embedding calls"
    );
    assert!(
        third.is_none(),
        "held Store cap 2 admitted a third embedding call before either gated call returned"
    );
    assert!(
        held_store_has_live_lease(&run),
        "the held Store lost its projection claim while two gated embedding calls were active"
    );

    first.expect("checked first call").release();
    second.expect("checked second call").release();
    assert!(
        run.runner
            .join()
            .expect("held-Store runner thread joins")
            .is_ok(),
        "held-Store cap-two projection failed"
    );
}

#[test]
fn held_store_cap_one_serializes_calls_while_its_claim_lease_stays_live() {
    let run = start_held_store_projection(NonZeroUsize::MIN);
    let first = run.entered.recv().ok();
    assert_eq!(run.waves.recv().expect("wave admission"), 1);
    let second_before_release = run.entered.try_recv();

    assert!(
        first.is_some(),
        "held Store never entered its first embedding call"
    );
    assert!(
        second_before_release.is_err(),
        "held Store cap 1 admitted a second embedding call before the first returned"
    );
    assert!(
        held_store_has_live_lease(&run),
        "the held Store lost its projection claim while its gated embedding call was active"
    );

    first.expect("checked first call").release();
    let second = run
        .entered
        .recv()
        .expect("held Store enters the second call after the first returns");
    second.release();
    assert!(
        run.runner
            .join()
            .expect("held-Store runner thread joins")
            .is_ok(),
        "held-Store cap-one projection failed"
    );
}

#[test]
fn claim_boundary_resolver_is_called_and_its_worker_cap_is_honored() {
    let resolutions = Arc::new(AtomicUsize::new(0));
    let run = start_projection(NonZeroUsize::MIN, Some(Arc::clone(&resolutions)));
    let first = run.entered.recv().ok();
    let second = run.entered.recv();
    let overlap_observed = second.is_ok();
    let mut observed = first.into_iter().collect::<Vec<_>>();
    if let Ok(second) = second {
        observed.push(second);
    }
    let (outcome, held_observed) = finish_projection(run, observed);

    assert!(
        outcome.result.is_ok(),
        "callback fixture projection must otherwise succeed: {:?}",
        outcome.result
    );
    assert_eq!(held_observed, 2, "fixture did not observe both held calls");
    assert_eq!(
        resolutions.load(Ordering::SeqCst),
        1,
        "one projection claim must resolve its worker cap once"
    );
    assert!(
        overlap_observed,
        "the claim ignored the callback's cap and stayed serial"
    );
}

#[test]
fn standalone_runner_resolves_the_worker_cap_again_for_later_claims() {
    let resolutions = Arc::new(AtomicUsize::new(0));
    let run = start_projection_with_event_batch(
        NonZeroUsize::MIN,
        Some(Arc::clone(&resolutions)),
        NonZeroU32::MIN,
    );
    let first = run
        .entered
        .recv()
        .expect("fixture enters its first embedding call");
    let (outcome, held_observed) = finish_projection(run, vec![first]);

    assert!(
        outcome.result.is_ok(),
        "multi-claim fixture projection must otherwise succeed: {:?}",
        outcome.result
    );
    assert_eq!(held_observed, 2, "fixture did not join both held calls");
    assert!(
        resolutions.load(Ordering::SeqCst) >= 2,
        "the standalone runner did not resolve the worker cap for a later claim"
    );
}

#[derive(Debug)]
struct BatchObservation {
    first_entered: bool,
    second_entered: bool,
    overlapped: bool,
    passage_texts: Vec<String>,
}

fn observe_two_passage_batch(
    entered: &Receiver<EnteredCall>,
    waves: &Receiver<usize>,
    expect_overlap: bool,
    observed: &mut Vec<EnteredCall>,
) -> BatchObservation {
    let first = entered.recv().ok();
    let first_entered = first.is_some();
    let width = waves.recv().expect("complete wave admission");
    assert_eq!(width, if expect_overlap { 2 } else { 1 });
    let second = if expect_overlap {
        entered.recv().ok()
    } else {
        entered.try_recv().ok()
    };
    let overlapped = second.is_some();

    let mut passage_texts = Vec::new();
    if let Some(first) = first {
        passage_texts.push(first.text.clone());
        observed.push(first);
    }
    if let Some(second) = second {
        passage_texts.push(second.text.clone());
        observed.push(second);
    }
    for call in observed.drain(..) {
        call.release();
    }
    let second_after_release = if !overlapped {
        entered.recv().ok()
    } else {
        None
    };
    if second_after_release.is_some() {
        assert_eq!(waves.recv().expect("serial second wave"), 1);
    }
    let second_entered = overlapped || second_after_release.is_some();
    if let Some(second) = second_after_release {
        passage_texts.push(second.text.clone());
        second.release();
    }
    BatchObservation {
        first_entered,
        second_entered,
        overlapped,
        passage_texts,
    }
}

fn changing_cap_run(
    caps: [usize; 2],
) -> (Vec<BatchObservation>, context_graph::EventSearchRunOutcome) {
    let root = tempfile::tempdir().expect("isolated changing-cap root");
    let (embedder, entered) = ControlledEmbedder::with_held_calls(4);
    let expected_caps = caps;
    let mut cap_values = caps.into_iter();
    let (claim_boundary, claim_boundaries) = channel();
    let run = start_projection_with_embedder(
        root,
        embedder,
        entered,
        NonZeroUsize::MIN,
        Some(Box::new(move || {
            let cap =
                NonZeroUsize::new(cap_values.next().unwrap_or(1)).expect("fixture cap is positive");
            claim_boundary
                .send(cap.get())
                .expect("runner reports claimed batch before embedding");
            cap
        })),
        NonZeroU32::new(2).expect("two events per claim"),
        true,
    );
    let mut observations = Vec::new();
    let mut held = Vec::new();
    assert_eq!(
        claim_boundaries
            .recv()
            .expect("first actual claim reaches worker-cap callback"),
        expected_caps[0],
    );
    observations.push(observe_two_passage_batch(
        &run.entered,
        &run.waves,
        expected_caps[0] > 1,
        &mut held,
    ));
    assert_eq!(
        claim_boundaries
            .recv()
            .expect("second actual claim reaches worker-cap callback"),
        expected_caps[1],
    );
    observations.push(observe_two_passage_batch(
        &run.entered,
        &run.waves,
        expected_caps[1] > 1,
        &mut held,
    ));
    let (outcome, _) = finish_projection_with_held_count(run, held, 0);
    (observations, outcome)
}

#[test]
fn standalone_runner_honors_one_then_two_at_successive_two_passage_claims() {
    let (observed, outcome) = changing_cap_run([1, 2]);
    assert!(
        outcome.result.is_ok(),
        "fixture run failed: {:?}",
        outcome.result
    );
    assert_eq!(observed.len(), 2);
    assert!(
        observed.iter().all(|batch| {
            batch.first_entered
                && batch.second_entered
                && batch.passage_texts.len() == 2
                && batch.passage_texts[0] != batch.passage_texts[1]
        }),
        "each claim must contain at least two actual passage calls: {observed:?}"
    );
    assert!(
        !observed[0].overlapped,
        "cap 1 allowed overlap in the first claim"
    );
    assert!(
        observed[1].overlapped,
        "cap 2 was not adopted for the second claim"
    );
}

#[test]
fn standalone_runner_honors_two_then_one_at_successive_two_passage_claims() {
    let root = tempfile::tempdir().expect("isolated changing-cap root");
    let (embedder, entered) = ControlledEmbedder::with_held_calls(4);
    let mut caps = [2_usize, 1].into_iter();
    let (claim_boundary, claim_boundaries) = channel();
    let run = start_projection_with_embedder(
        root,
        embedder,
        entered,
        NonZeroUsize::MIN,
        Some(Box::new(move || {
            let cap = NonZeroUsize::new(caps.next().unwrap_or(1)).expect("fixture cap is positive");
            claim_boundary
                .send(cap.get())
                .expect("runner reports claimed batch before embedding");
            cap
        })),
        NonZeroU32::new(2).expect("two events per claim"),
        true,
    );
    let mut held = Vec::new();
    assert_eq!(
        claim_boundaries
            .recv()
            .expect("first actual claim reaches worker-cap callback"),
        2,
    );
    let first = observe_two_passage_batch(&run.entered, &run.waves, true, &mut held);
    assert_eq!(
        claim_boundaries
            .recv()
            .expect("second actual claim reaches worker-cap callback"),
        1,
    );
    let second = observe_two_passage_batch(&run.entered, &run.waves, false, &mut held);
    let (outcome, _) = finish_projection_with_held_count(run, held, 0);

    assert!(
        outcome.result.is_ok(),
        "fixture run failed: {:?}",
        outcome.result
    );
    assert!(
        first.first_entered
            && first.second_entered
            && second.first_entered
            && second.second_entered
            && first.passage_texts.len() == 2
            && second.passage_texts.len() == 2
            && first.passage_texts[0] != first.passage_texts[1]
            && second.passage_texts[0] != second.passage_texts[1],
        "each claim must contain at least two actual passage calls: {first:?}, {second:?}"
    );
    assert!(first.overlapped, "cap 2 was not used for the first claim");
    assert!(
        !second.overlapped,
        "cap 1 allowed overlap in the second claim"
    );
}

#[test]
fn claim_callback_rereads_the_worker_policy_file_before_the_next_two_passage_batch() {
    let root = tempfile::tempdir().expect("isolated policy-refresh root");
    let initial_write = write_event_search_worker_policy(root.path(), r#"{"embedding_workers":1}"#);
    let (embedder, entered) = ControlledEmbedder::with_held_calls(4);
    let policy_root = root.path().to_path_buf();
    let run = start_projection_with_embedder(
        root,
        embedder,
        entered,
        NonZeroUsize::MIN,
        Some(Box::new(move || {
            read_event_search_worker_policy(&policy_root)
                .ok()
                .flatten()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
                .and_then(|value| value.get("embedding_workers")?.as_u64())
                .and_then(|value| usize::try_from(value).ok())
                .and_then(NonZeroUsize::new)
                .unwrap_or(NonZeroUsize::MIN)
        })),
        NonZeroU32::new(2).expect("two events per claim"),
        true,
    );

    let first = run.entered.recv().ok();
    assert_eq!(run.waves.recv().expect("first wave admission"), 1);
    let first_overlap = run.entered.try_recv().ok();
    let first_overlapped = first_overlap.is_some();
    if let Some(call) = first {
        call.release();
    }
    let first_second = if first_overlap.is_none() {
        run.entered.recv().ok()
    } else {
        first_overlap
    };
    let replacement =
        write_event_search_worker_policy(run._root.path(), r#"{"embedding_workers":2}"#);
    let first_had_two_passages = first_second.is_some();
    if let Some(call) = first_second {
        call.release();
    }
    let second_first = run.entered.recv().ok();
    let second_second = run.entered.recv().ok();
    let second_first_entered = second_first.is_some();
    let second_overlapped = second_second.is_some();
    if let Some(call) = second_first {
        call.release();
    }
    if let Some(call) = second_second {
        call.release();
    }
    let (outcome, _) = finish_projection_with_held_count(run, Vec::new(), 4);

    assert!(
        initial_write.is_ok(),
        "initial policy write failed: {initial_write:?}"
    );
    assert!(
        replacement.is_ok(),
        "replacement policy write failed: {replacement:?}"
    );
    assert!(
        outcome.result.is_ok(),
        "fixture run failed: {:?}",
        outcome.result
    );
    assert!(
        first_had_two_passages,
        "the first claim had fewer than two passage calls"
    );
    assert!(
        second_first_entered,
        "the second claim never entered its first passage call"
    );
    assert!(!first_overlapped, "initial cap 1 allowed overlap");
    assert!(
        second_overlapped,
        "the next claim did not adopt the replacement cap 2"
    );
}
