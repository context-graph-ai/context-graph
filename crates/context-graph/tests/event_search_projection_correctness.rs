//! Projection correctness under interruption and concurrent readers (intent "Projection
//! recovery" and the incremental-tail snapshot contract).
//!
//! The standalone runner releases its Store around rendering and embedding, so the embedder hooks
//! below act at real claim boundaries: one simulates a worker killed after a committed passage
//! batch, the other reads the committed projection between every claim of an incremental tail
//! replacement. The finalization test pauses inside the full rebuild's head-switch transaction
//! through the internal test seam.

#![cfg(context_graph_internal_test_support)]

use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use context_graph::ConsumerReader;
use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::owner_control::{ReaderRelease, wait_for_reader_release_within};
use context_graph::test_support::pause_next_event_search_finalization_for_test;
use context_graph::{
    AppendEventBatch, CgError, ContextId, ContextSelector, CreateContext, DistanceMetric, Embedder,
    EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchBuildReceipt,
    EventSearchProfile, EventSearchProjectionRunner, EventSearchRebuildRequest,
    EventSearchRunRequest, EvidenceKind, EvidenceProducer, Result, Store, StoreConfig,
    event_history_reader_declaration,
};
use contextdb_core::{Value, Wallclock};
use serde_json::json;

const CONNECTOR: &str = "event-search-correctness";
const SESSION: &str = "event-search-correctness-session";
const SPACE: &str = "event-search-correctness-768";
/// Short enough that the resume test can outwait a dead worker's lease.
const LEASE_MS: u64 = 1_000;
/// Long enough that a reader opening a Store between claims never expires the live claim.
const OBSERVED_LEASE_MS: u64 = 60_000;

type Observer = Box<dyn Fn() + Send + Sync>;

/// Role-aware fixture tokenization (five model tokens plus two per scalar), with two controls:
/// a one-shot failure for any document containing a marker, and an observer run at every
/// document preparation.
struct ControlledEmbedder {
    fail_once_on: Mutex<Option<char>>,
    observer: Mutex<Option<Observer>>,
    embedded_documents: Mutex<Vec<String>>,
}

impl ControlledEmbedder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            fail_once_on: Mutex::new(None),
            observer: Mutex::new(None),
            embedded_documents: Mutex::new(Vec::new()),
        })
    }

    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn embedded(&self) -> Vec<String> {
        self.embedded_documents
            .lock()
            .expect("embedded document log")
            .clone()
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

/// Every current passage in the Context as (passage id, rendered text), in passage order.
fn current_passages(store: &Store, context_id: ContextId) -> Vec<(uuid::Uuid, String)> {
    let result = store
        .execute_sql_for_test(
            "SELECT id, rendered_text, passage_ordinal FROM event_search_passages WHERE context_id = $context_id AND is_current = $current ORDER BY passage_ordinal ASC",
            &HashMap::from([
                ("context_id".into(), Value::Uuid(context_id.as_uuid())),
                ("current".into(), Value::Bool(true)),
            ]),
        )
        .expect("inspect current passages");
    result
        .rows
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Uuid(id), Value::Text(text)) => (*id, text.clone()),
            other => panic!("unexpected passage row {other:?}"),
        })
        .collect()
}

fn all_passage_count(store: &Store, context_id: ContextId) -> usize {
    store
        .execute_sql_for_test(
            "SELECT id FROM event_search_passages WHERE context_id = $context_id",
            &HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]),
        )
        .expect("inspect every passage row")
        .rows
        .len()
}

#[test]
fn a_worker_killed_after_a_committed_batch_resumes_without_duplicates_or_re_embedding() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("resume.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    for (source, first, second) in [("a-stream", "α", "β"), ("b-stream", "γ", "δ")] {
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                source,
                vec![
                    draft(
                        &format!("{source}-1"),
                        1,
                        EventRenderRole::HumanMessage,
                        first,
                    ),
                    draft(
                        &format!("{source}-2"),
                        2,
                        EventRenderRole::AgentMessage,
                        second,
                    ),
                ],
            ))
            .expect("append one stream");
    }
    drop(store);

    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control lock") = Some('γ');
    let killed = run(&config, context_id, &embedder, 8, LEASE_MS)
        .expect_err("the worker dies while embedding the second stream's batch");
    assert!(
        matches!(&killed, CgError::Engine(message) if message.contains("fixture worker killed")),
        "the simulated kill surfaces unchanged: {killed:?}"
    );
    // A claim whose lease expires before its commit is re-rendered, so the first stream's batch may
    // be embedded more than once before it commits; the second stream's batch never completes.
    let first_run = embedder.embedded();
    assert!(
        !first_run.is_empty()
            && first_run
                .iter()
                .all(|text| text.contains('α') && text.contains('β') && !text.contains('γ')),
        "only the first stream's batch was embedded before the kill: {first_run:?}"
    );

    let inspect = open(&config);
    let committed = current_passages(&inspect, context_id);
    assert_eq!(
        committed.len(),
        1,
        "the first batch committed atomically before the kill: {committed:?}"
    );
    let committed_id = committed[0].0;
    drop(inspect);

    // The killed worker's lease is left behind exactly as a dead process would leave it. The rerun
    // reads the seamed clock past that lease's expiry instead of waiting it out.
    let past_expiry = Wallclock::now().0 + LEASE_MS * 2;
    let resumed = {
        let _clock = Wallclock::test_clock_guard(move || past_expiry);
        run(&config, context_id, &embedder, 8, LEASE_MS)
    }
    .expect("the rerun resumes after the lease expires");
    let resumed_inputs = embedder.embedded()[first_run.len()..].to_vec();
    assert!(
        !resumed_inputs.is_empty()
            && resumed_inputs
                .iter()
                .all(|text| !text.contains('α') && !text.contains('β')),
        "the completed passage is never re-embedded: {resumed_inputs:?}"
    );
    assert!(
        resumed_inputs
            .iter()
            .any(|text| text.contains('γ') && text.contains('δ')),
        "the interrupted range is embedded on resume: {resumed_inputs:?}"
    );
    assert_eq!(resumed.source_watermark.event_count, 4);
    assert_eq!(resumed.embedded_event_count, 4);

    let after = open(&config);
    let current = current_passages(&after, context_id);
    assert_eq!(
        current.len(),
        2,
        "one current passage per stream, no duplicate: {current:?}"
    );
    assert_eq!(
        current[0].0, committed_id,
        "the committed batch's passage is reused, not rebuilt"
    );
    assert_eq!(
        all_passage_count(&after, context_id),
        2,
        "the resume wrote no duplicate passage rows, current or retired"
    );
    for marker in ['α', 'β', 'γ', 'δ'] {
        assert_eq!(
            current
                .iter()
                .filter(|(_, text)| text.contains(marker))
                .count(),
            1,
            "event {marker} is covered by exactly one current passage"
        );
    }
    drop(after);

    let before_unchanged = embedder.embedded().len();
    run(&config, context_id, &embedder, 8, LEASE_MS).expect("an unchanged rerun completes");
    assert_eq!(
        embedder.embedded().len(),
        before_unchanged,
        "a second unchanged run performs zero embedder calls"
    );
}

#[test]
fn readers_between_incremental_tail_claims_see_the_complete_old_or_new_tail_never_a_mix() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("tail-snapshot.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "tail-stream",
            vec![
                draft("closed-human", 1, EventRenderRole::HumanMessage, "α"),
                draft("closed-request", 2, EventRenderRole::ToolRequest, "β"),
                draft("tail-result", 3, EventRenderRole::ToolResult, "γ"),
                draft("tail-human", 4, EventRenderRole::HumanMessage, "δ"),
            ],
        ))
        .expect("append initial history");
    drop(store);

    let embedder = ControlledEmbedder::new();
    run(&config, context_id, &embedder, 8, LEASE_MS).expect("initial projection");
    let before = open(&config);
    let initial = current_passages(&before, context_id);
    assert_eq!(
        initial.len(),
        2,
        "the fixture grouping has one closed passage and one open tail: {initial:?}"
    );
    let old_tail = initial
        .iter()
        .find(|(_, text)| text.contains('γ') && text.contains('δ'))
        .expect("the open tail holds γ and δ")
        .0;
    let initial_ids = initial.iter().map(|(id, _)| *id).collect::<BTreeSet<_>>();
    before
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "tail-stream",
            vec![
                draft("new-request", 5, EventRenderRole::ToolRequest, "ε"),
                draft("new-result", 6, EventRenderRole::ToolResult, "ζ"),
                draft("new-human", 7, EventRenderRole::HumanMessage, "η"),
                draft("new-agent", 8, EventRenderRole::AgentMessage, "θ"),
            ],
        ))
        .expect("append after the ready head");
    drop(before);

    let observations = Arc::new(Mutex::new(Vec::<Vec<(uuid::Uuid, String)>>::new()));
    let observer_config = config.clone();
    let observed = observations.clone();
    *embedder.observer.lock().expect("observer lock") = Some(Box::new(move || {
        let reader = Store::open(observer_config.clone()).expect("reader opens between claims");
        observed
            .lock()
            .expect("observation log")
            .push(current_passages(&reader, context_id));
    }));
    // One new event per claim, so the tail replacement spans several committed claims.
    run(&config, context_id, &embedder, 1, OBSERVED_LEASE_MS).expect("incremental catch-up");
    *embedder.observer.lock().expect("observer lock") = None;
    let after = open(&config);
    let mut observations = observations.lock().expect("observation log").clone();
    observations.push(current_passages(&after, context_id));
    assert!(
        observations.len() >= 3,
        "readers observed the projection between several claims: {}",
        observations.len()
    );

    for (index, current) in observations.iter().enumerate() {
        for marker in ['α', 'β', 'γ', 'δ'] {
            assert!(
                current.iter().any(|(_, text)| text.contains(marker)),
                "observation {index}: previously searchable event {marker} vanished: {current:?}"
            );
        }
        let old_tail_current = current.iter().any(|(id, _)| *id == old_tail);
        eprintln!(
            "observation {index}: old tail current = {old_tail_current}, current passages = {:?}",
            current.iter().map(|(_, text)| text).collect::<Vec<_>>()
        );
        let replacement_current = current.iter().any(|(id, text)| {
            !initial_ids.contains(id) && (text.contains('γ') || text.contains('δ'))
        });
        assert_ne!(
            old_tail_current, replacement_current,
            "observation {index}: exactly one of the old tail or its replacement is current: \
             {current:?}"
        );
    }
    let last = observations.last().expect("final observation");
    assert!(
        !last.iter().any(|(id, _)| *id == old_tail),
        "the final projection retired the old tail"
    );
    for marker in ['ε', 'ζ', 'η', 'θ'] {
        assert!(
            last.iter().any(|(_, text)| text.contains(marker)),
            "the final projection exposes appended event {marker}: {last:?}"
        );
    }
}

#[test]
fn an_append_during_rebuild_finalization_is_caught_up_before_the_head_switches() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("finalization.db"));
    let embedder = ControlledEmbedder::new();
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open Store");
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "finalization-stream",
            vec![draft("before", 1, EventRenderRole::HumanMessage, "α")],
        ))
        .expect("append the event present when the rebuild begins");
    run(&config, context_id, &embedder, 8, LEASE_MS).expect("an existing ready head");

    // The only way a second event exists is this append, made inside the rebuild's final
    // head-switch transaction after its revision guard and live count check.
    pause_next_event_search_finalization_for_test(move |paused: &Store| {
        paused
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                "finalization-stream",
                vec![draft("during", 2, EventRenderRole::AgentMessage, "β")],
            ))
            .expect("append while the rebuild is finalizing");
    });
    let receipt = store
        .rebuild_event_search(EventSearchRebuildRequest {
            context_id,
            interpretation_generation_id: None,
            embedder: embedder.registration(),
            profile: profile(),
            max_events_per_batch: NonZeroU32::new(8).expect("nonzero batch"),
            max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
            max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
            lease_duration_ms: NonZeroU64::new(OBSERVED_LEASE_MS).expect("nonzero lease"),
        })
        .expect("the rebuild completes");

    let params = HashMap::from([("context_id".into(), Value::Uuid(context_id.as_uuid()))]);
    let head = store
        .execute_sql_for_test(
            "SELECT generation_id FROM event_search_heads WHERE context_id = $context_id",
            &params,
        )
        .expect("inspect search head")
        .rows[0][0]
        .clone();
    let head_event_count = store
        .execute_sql_for_test(
            "SELECT event_count FROM event_search_generations WHERE id = $id",
            &HashMap::from([("id".into(), head)]),
        )
        .expect("inspect head generation")
        .rows[0][0]
        .clone();

    assert_eq!(
        receipt.source_watermark.event_count, 2,
        "the rebuild's receipt covers the event appended during its finalization"
    );
    assert_eq!(
        head_event_count,
        Value::Int64(2),
        "the head never switched to a generation that misses the appended event"
    );
}

/// Every projection lease row the Context holds, whatever its expiry.
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
fn a_run_that_gives_up_after_claiming_releases_its_claim_so_the_next_run_claims_at_once() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("release.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "a-stream",
            vec![
                draft("a-1", 1, EventRenderRole::HumanMessage, "α"),
                draft("a-2", 2, EventRenderRole::AgentMessage, "β"),
            ],
        ))
        .expect("append one stream");
    drop(store);

    // A ten-minute lease: if the run that gives up left its claim behind, nothing could claim this
    // Context again for ten minutes, and the rerun below would be refused rather than slow.
    let long_lease_ms = 600_000;
    let embedder = ControlledEmbedder::new();
    *embedder.fail_once_on.lock().expect("failure control lock") = Some('α');
    let abandoned = run(&config, context_id, &embedder, 8, long_lease_ms)
        .expect_err("the run gives up while embedding the batch it claimed");
    assert!(
        matches!(&abandoned, CgError::Engine(message) if message.contains("fixture worker killed")),
        "the simulated give-up surfaces unchanged: {abandoned:?}"
    );

    let inspect = open(&config);
    assert_eq!(
        lease_rows(&inspect, context_id),
        0,
        "a run that gives up after claiming releases the claim it was holding"
    );
    drop(inspect);

    // No clock guard anywhere: the next run claims the released work at once.
    let resumed = run(&config, context_id, &embedder, 8, long_lease_ms)
        .expect("the next run claims the released work immediately");
    assert_eq!(resumed.source_watermark.event_count, 2);
    assert_eq!(resumed.embedded_event_count, 2);
    assert_eq!(
        current_passages(&open(&config), context_id).len(),
        1,
        "the retry covers the Context with one current passage"
    );
}

/// A run whose reopens meet direct readers still commits every batch.
///
/// The runner claims a batch, CLOSES the Store to render and embed, and reopens to commit. A reader
/// loading the committed image at one of those reopens refuses the writer — and a run that took
/// that refusal for an answer would return holding its claim, leaving the Context uncovered with
/// nothing running and the next attempt locked out by the orphan. Readers finish; the reopen must
/// simply be asked again.
///
/// One event per batch, so the run closes and reopens the Store for every event there is, with
/// three readers opening the store as fast as they can throughout. Whether any single reopen is
/// actually refused is up to the machine, which is why the scenario runs several times: the
/// assertion that must hold every time is that the run commits all of its work.
#[test]
fn a_run_reopening_the_store_under_continuous_readers_commits_every_batch() {
    use std::sync::atomic::{AtomicBool, Ordering};

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
        std::thread::scope(|scope| {
            for _ in 0..3 {
                let config = &config;
                let stop = &stop;
                scope.spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        if let Ok(reader) = ConsumerReader::open(
                            &config.db_path,
                            event_history_reader_declaration(),
                        ) {
                            drop(reader);
                        }
                    }
                });
            }
            let embedder = ControlledEmbedder::new();
            let receipt = run(&config, context_id, &embedder, 1, unexpiring_lease_ms)
                .expect("the run commits every batch it claims while readers hold the Store");
            stop.store(true, Ordering::Relaxed);
            assert_eq!(
                receipt.source_watermark.event_count, 8,
                "scenario {scenario} covers the whole stream"
            );
            assert_eq!(
                receipt.embedded_event_count, 8,
                "scenario {scenario} embedded every event"
            );
        });

        let after = open(&config);
        assert_eq!(
            lease_rows(&after, context_id),
            0,
            "scenario {scenario} left no claim behind"
        );
    }
}

/// Waiting for direct readers never outlasts the allowance the caller gave it.
///
/// `wait_for_reader_release` has no deadline of its own: it wakes when the last reader lets go or
/// when its cancellation token is cancelled, and otherwise waits forever. A writer that hands it a
/// token nobody ever cancels therefore has no allowance at all, and one reader that never finishes
/// loading would stop the projection for good. The bounded form owns the token and cancels it the
/// moment the allowance is spent, so the wait always answers.
#[test]
fn waiting_for_readers_never_outlasts_the_allowance() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    let root = tempfile::tempdir().expect("tempdir");
    let config = config(root.path().join("bounded-wait.db"));
    let store = open(&config);
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            "bounded-wait",
            vec![
                draft("w-1", 1, EventRenderRole::HumanMessage, "α"),
                draft("w-2", 2, EventRenderRole::AgentMessage, "β"),
            ],
        ))
        .expect("append one stream");
    drop(store);

    // A deadline already spent is answered at once, without waiting on anything at all.
    assert!(
        matches!(
            wait_for_reader_release_within(&config.db_path, Instant::now()),
            ReaderRelease::Stopped
        ),
        "a spent allowance is answered immediately"
    );

    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..3 {
            let config = &config;
            let stop = &stop;
            scope.spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if let Ok(reader) =
                        ConsumerReader::open(&config.db_path, event_history_reader_declaration())
                    {
                        drop(reader);
                    }
                }
            });
        }
        for allowance in [Duration::from_millis(50), Duration::from_millis(400)] {
            // Either verdict is legitimate — the readers may let go inside the allowance, or the
            // allowance may run out first. What must never happen is not answering, and the test
            // runner's terminate-after owns that.
            let verdict =
                wait_for_reader_release_within(&config.db_path, Instant::now() + allowance);
            assert!(
                matches!(verdict, ReaderRelease::Released | ReaderRelease::Stopped),
                "a {allowance:?} allowance under churning readers answered {verdict:?}"
            );
        }
        stop.store(true, Ordering::Relaxed);
    });
}
