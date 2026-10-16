//! Behavior proofs for event-history projection.  Expected rendering lives in an independent fixture;
//! this file only captures real immutable events and drives the public runner.

#![cfg(context_graph_internal_test_support)]

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, ContextId, ConversationEvent, CreateContext, DistanceMetric, Embedder,
    EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchEmbedder,
    EventSearchFilter, EventSearchProfile, EventSearchProjectionRunner, EventSearchRunRequest,
    EvidenceKind, EvidenceProducer, Result, Store, StoreConfig,
};
use serde::Deserialize;
use serde_json::json;

const FIXTURE: &str = include_str!("fixtures/event_search/renderer_v1_utf8.json");
const CONNECTOR: &str = "event-search-red";
const SOURCE: &str = "event-search-fixture";
const SESSION: &str = "renderer-v1";

#[derive(Deserialize)]
struct RendererFixture {
    expected_passage: String,
    expected_source_spans: Vec<ExpectedSpan>,
}

#[derive(Deserialize)]
struct ExpectedSpan {
    event: String,
    event_start: u64,
    event_end: u64,
    passage_start: u64,
    passage_end: u64,
}

/// A deterministic 768-dimensional test embedder.  It implements the existing role-aware
/// capability and records calls; it is deliberately not a substitute for the Nomic runtime.
struct ProjectionEmbedder {
    query_prepare_calls: AtomicUsize,
    document_preparations: Mutex<Vec<DocumentObservation>>,
    document_inferences: Mutex<Vec<DocumentObservation>>,
    query_embed_token_counts: Mutex<Vec<usize>>,
}

impl ProjectionEmbedder {
    fn new() -> Self {
        Self {
            query_prepare_calls: AtomicUsize::new(0),
            document_preparations: Mutex::new(Vec::new()),
            document_inferences: Mutex::new(Vec::new()),
            query_embed_token_counts: Mutex::new(Vec::new()),
        }
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
            embedding_space_id: self.encoding_profile_id(),
            dimension: 768,
            producer: EvidenceProducer {
                system: "event-search-red".into(),
                model_name: "deterministic-768".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector,
        }
    }

    /// This test tokenizer models one Unicode scalar per content token, plus the actual role
    /// prefix and two model-special tokens.  Prefix/special tokens deliberately have zero-width
    /// source ranges; source ranges remain scalar and UTF-8 aligned.
    fn document_tokens(text: &str) -> (Vec<u32>, Vec<(usize, usize)>) {
        const DOCUMENT_PREFIX: &str = "search_document: ";
        const MODEL_SPECIALS: usize = 2;
        let mut payload = vec![0; DOCUMENT_PREFIX.chars().count() + MODEL_SPECIALS];
        let mut boundaries = vec![(0, 0); payload.len()];
        for (start, scalar) in text.char_indices() {
            let end = start + scalar.len_utf8();
            payload.push(scalar as u32);
            boundaries.push((start, end));
        }
        (payload, boundaries)
    }

    fn query_tokens(text: &str) -> Vec<u32> {
        const QUERY_PREFIX: &str = "search_query: ";
        const MODEL_SPECIALS: usize = 2;
        let mut payload = vec![0; QUERY_PREFIX.chars().count() + MODEL_SPECIALS];
        payload.extend(text.chars().map(|scalar| scalar as u32));
        payload
    }

    fn document_inference_observation(inference: &EventSearchInference) -> DocumentObservation {
        DocumentObservation {
            text: inference
                .token_ids()
                .iter()
                .filter(|token| **token != 0)
                .map(|token| {
                    char::from_u32(*token)
                        .expect("fixture document token ids are Unicode scalar values")
                })
                .collect(),
            token_count: inference.token_count(),
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DocumentObservation {
    text: String,
    token_count: usize,
}

fn assert_persisted_passages_inferred_once(
    preparations: &[DocumentObservation],
    inferences: &[DocumentObservation],
    passages: impl IntoIterator<Item = String>,
    hard_limit: usize,
) {
    let mut expected = passages
        .into_iter()
        .map(|text| DocumentObservation {
            token_count: fixture_document_token_count(&text),
            text,
        })
        .collect::<Vec<_>>();
    let mut actual = inferences.to_vec();
    expected.sort();
    actual.sort();
    assert_eq!(
        actual, expected,
        "each persisted passage has exactly one inference with its final text and token count"
    );
    for inference in inferences {
        assert!(
            preparations.contains(inference),
            "every inference comes from matching prepared document data"
        );
        assert!(
            inference.token_count <= hard_limit,
            "every inferred passage stays within the declared hard token budget"
        );
    }
}

impl Embedder for ProjectionEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search must use explicit document/query preparation")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: "event-search-red-deterministic-768".into(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "test".into(),
            model_name: "deterministic-768".into(),
            model_version: "1".into(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: "event-search-red-deterministic-768".into(),
                table_name: "event_search_passages".into(),
                vector_column: "vector_text".into(),
            }],
        }
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(1_536)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let (payload, boundaries) = Self::document_tokens(text);
        self.document_preparations
            .lock()
            .expect("document preparation log")
            .push(DocumentObservation {
                text: text.into(),
                token_count: payload.len(),
            });
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
        self.query_prepare_calls.fetch_add(1, Ordering::SeqCst);
        let payload = Self::query_tokens(text);
        seal.query_builder()
            .token_count(payload.len())
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        self.document_inferences
            .lock()
            .expect("document hook log")
            .push(Self::document_inference_observation(inference));
        Ok(self.output(inference))
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        self.query_embed_token_counts
            .lock()
            .expect("query hook log")
            .push(inference.token_count());
        Ok(self.output(inference))
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "fixture-v1".into(),
        anchor_declaration_version: "fixture-v1".into(),
        anchor_schema: json!({"type":"object", "additionalProperties":true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn store_config(path: std::path::PathBuf, space: EmbeddingSpace) -> StoreConfig {
    StoreConfig {
        db_path: path,
        extra_embedding_spaces: vec![space],
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn create_context(store: &Store) -> ContextId {
    store
        .create_context(CreateContext {
            name: "event-search-projection-red".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id
}

fn append_fixture_events(store: &Store, context_id: ContextId) -> Vec<ConversationEvent> {
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![
                EventDraft {
                    session_ref: SESSION.into(),
                    source_event_key: "human-1".into(),
                    source_position: json!({"ordinal": 1}),
                    occurred_at: None,
                    author: "Renée".into(),
                    kind: "message".into(),
                    render_role: EventRenderRole::HumanMessage,
                    interaction_ref: None,
                    text: Some("Keep café ☕ open.".into()),
                    payload: json!({"ignored":"because text exists"}),
                    anchors: json!({}),
                    source_remainder: json!({"not_a_renderer_input":true}),
                    coverage: EventCoverage::Complete,
                },
                EventDraft {
                    session_ref: SESSION.into(),
                    source_event_key: "tool-1".into(),
                    source_position: json!({"ordinal": 2}),
                    occurred_at: None,
                    author: "agent:main".into(),
                    kind: "tool".into(),
                    render_role: EventRenderRole::ToolRequest,
                    interaction_ref: Some("call-1".into()),
                    text: None,
                    payload: json!({"z":1,"a":"β"}),
                    anchors: json!({}),
                    source_remainder: json!({"not_a_renderer_input":true}),
                    coverage: EventCoverage::Complete,
                },
            ],
        ))
        .expect("append captured events")
        .events
        .into_iter()
        .map(|result| result.event)
        .collect()
}

fn custom_profile() -> EventSearchProfile {
    EventSearchProfile {
        target_passage_tokens: NonZeroU32::new(1_000).unwrap(),
        hard_passage_tokens: NonZeroU32::new(1_500).unwrap(),
        ordinary_overlap_events: NonZeroU32::new(2).unwrap(),
        oversized_event_overlap_tokens: NonZeroU32::new(64).unwrap(),
        ..EventSearchProfile::default()
    }
}

const OVERSIZED_HUMAN_PREFIX: &str = "User [author=\"Renée\"]: ";
const DOCUMENT_ROLE_INSTRUCTION: &str = "search_document: ";
const DOCUMENT_SPECIAL_TOKENS: usize = 2;

/// This is deliberately independent of the production renderer and embedder: it fixes the
/// renderer-v1 human prefix and the fixture tokenizer's scalar accounting in the Behavior proofs.
fn expected_oversized_passage(body: &str) -> String {
    format!("{OVERSIZED_HUMAN_PREFIX}{body}")
}

fn fixture_document_token_count(passage: &str) -> usize {
    DOCUMENT_ROLE_INSTRUCTION.chars().count() + DOCUMENT_SPECIAL_TOKENS + passage.chars().count()
}

fn expected_oversized_ranges(
    body: &str,
    hard_limit: usize,
    overlap_tokens: usize,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    loop {
        let end = body[start..]
            .char_indices()
            .map(|(offset, scalar)| start + offset + scalar.len_utf8())
            .take_while(|end| {
                fixture_document_token_count(&expected_oversized_passage(&body[start..*end]))
                    <= hard_limit
            })
            .last()
            .expect("the declared hard limit leaves room for one body scalar");
        ranges.push((start, end));
        if end == body.len() {
            return ranges;
        }

        let scalar_starts: Vec<_> = body[start..end]
            .char_indices()
            .map(|(offset, _)| start + offset)
            .collect();
        assert!(
            scalar_starts.len() > overlap_tokens,
            "an unfinished oversized slice must contain the configured overlap"
        );
        let next_start = scalar_starts[scalar_starts.len() - overlap_tokens];
        assert!(next_start > start, "oversized slicing must advance");
        start = next_start;
    }
}

#[test]
fn fixture_unicode_budget_uses_scalar_ranges_and_role_overhead() {
    let fixture: RendererFixture = serde_json::from_str(FIXTURE).expect("fixture parses");
    let embedder = ProjectionEmbedder::new();
    let prepared = embedder
        .prepare_document(&fixture.expected_passage)
        .expect("fixture rendering prepares under the test profile");
    let expected_overhead = "search_document: ".chars().count() + 2;
    assert_eq!(
        prepared.token_count(),
        expected_overhead + fixture.expected_passage.chars().count(),
        "the fixture budget includes the document role prefix and model specials, not UTF-8 bytes"
    );
    assert!(fixture.expected_passage.is_char_boundary(44));
    assert_eq!(
        &prepared.byte_boundaries()[..expected_overhead],
        vec![(0, 0); expected_overhead].as_slice(),
        "role prefix and model specials have zero-width source spans"
    );
    assert!(
        prepared.byte_boundaries()[expected_overhead..]
            .iter()
            .all(
                |(start, end)| fixture.expected_passage.is_char_boundary(*start)
                    && fixture.expected_passage.is_char_boundary(*end)
                    && end > start
            )
    );
}

#[test]
fn projection_renders_frozen_utf8_grammar_with_ordered_source_spans_and_document_preparation() {
    let fixture: RendererFixture = serde_json::from_str(FIXTURE).expect("fixture parses");
    assert_eq!(fixture.expected_passage.len(), 94);
    assert_eq!(fixture.expected_source_spans.len(), 2);
    assert_eq!(fixture.expected_source_spans[0].event, "human-1");
    assert_eq!(fixture.expected_source_spans[0].event_end, 20);
    assert_eq!(fixture.expected_source_spans[0].passage_start, 24);
    assert_eq!(fixture.expected_source_spans[0].passage_end, 44);
    assert_eq!(fixture.expected_source_spans[1].event, "tool-1");
    assert_eq!(fixture.expected_source_spans[1].event_start, 0);
    assert_eq!(fixture.expected_source_spans[1].event_end, 16);
    assert_eq!(fixture.expected_source_spans[1].passage_start, 78);
    assert_eq!(fixture.expected_source_spans[1].passage_end, 94);

    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(ProjectionEmbedder::new());
    let config = store_config(
        root.path().join("event-search.db"),
        embedder.embedding_space(),
    );
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open store");
    let context_id = create_context(&store);
    let captured = append_fixture_events(&store, context_id);
    assert_eq!(captured.len(), 2);
    drop(store);

    let receipt = EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config.clone(),
        contexts: context_graph::ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: custom_profile(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(2).unwrap(),
        max_passages_per_batch: NonZeroU32::new(2).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(4_096).unwrap(),
        lease_duration_ms: NonZeroU64::new(1_000).unwrap(),
    })
    .expect("projection must render and commit captured events");

    assert_eq!(receipt.profile, custom_profile());
    assert_eq!(receipt.renderer_id, "context_graph_event_history");
    assert_eq!(receipt.renderer_version, "1");
    let reopened = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("open a fresh Store handle after the public runner returns");
    let result = reopened
        .find_event_history(
            context_graph::ContextSelector::one(context_id),
            "open café",
            EventSearchFilter::default(),
        )
        .expect("projection result is searchable");
    assert_eq!(embedder.query_prepare_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        embedder
            .query_embed_token_counts
            .lock()
            .expect("query hook log")
            .len(),
        1,
        "every prepared query must reach the matching checked embedding hook"
    );
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].text, fixture.expected_passage);
    let preparations = embedder
        .document_preparations
        .lock()
        .expect("document preparation log")
        .clone();
    let inferences = embedder
        .document_inferences
        .lock()
        .expect("document inference log")
        .clone();
    assert_persisted_passages_inferred_once(
        &preparations,
        &inferences,
        result.hits.iter().map(|hit| hit.text.clone()),
        custom_profile().hard_passage_tokens.get() as usize,
    );
    assert_eq!(result.hits[0].sources.len(), 2);
    let first = &result.hits[0].sources[0];
    assert_eq!(first.event_id, captured[0].id);
    assert_eq!(first.context_id, context_id);
    assert_eq!(first.connector_id, CONNECTOR);
    assert_eq!(first.session_ref, SESSION);
    assert_eq!(first.source_instance_ref, SOURCE);
    assert_eq!((first.event_start_byte, first.event_end_byte), (None, None));
    assert_eq!((first.passage_start_byte, first.passage_end_byte), (24, 44));
    let second = &result.hits[0].sources[1];
    assert_eq!(second.event_id, captured[1].id);
    assert_eq!(second.context_id, context_id);
    assert_eq!(second.connector_id, CONNECTOR);
    assert_eq!(second.session_ref, SESSION);
    assert_eq!(second.source_instance_ref, SOURCE);
    assert_eq!(
        (second.event_start_byte, second.event_end_byte),
        (None, None)
    );
    assert_eq!(
        (second.passage_start_byte, second.passage_end_byte),
        (78, 94)
    );
    let persisted = reopened
        .get_event_search_passage(
            context_graph::ContextSelector::one(context_id),
            result.hits[0].passage_id,
        )
        .expect("public passage read")
        .expect("search hit must name a persisted passage with an embedding handle");
    assert_eq!(persisted.id, result.hits[0].passage_id);
    assert_eq!(
        persisted.embedding_space_id,
        "event-search-red-deterministic-768"
    );
    assert_ne!(persisted.embedding_id.as_uuid(), uuid::Uuid::nil());
}

#[test]
fn oversized_event_slices_advance_cover_every_body_byte_and_honor_the_declared_profile_limit() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(ProjectionEmbedder::new());
    let config = store_config(
        root.path().join("oversized-event-search.db"),
        embedder.embedding_space(),
    );
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open store");
    let context_id = create_context(&store);
    let oversized = "é".repeat(1_600);
    let captured = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![EventDraft {
                session_ref: SESSION.into(),
                source_event_key: "oversized-1".into(),
                source_position: json!({"ordinal": 1}),
                occurred_at: None,
                author: "Renée".into(),
                kind: "message".into(),
                render_role: EventRenderRole::HumanMessage,
                interaction_ref: None,
                text: Some(oversized.clone()),
                payload: json!({}),
                anchors: json!({}),
                source_remainder: json!({}),
                coverage: EventCoverage::Complete,
            }],
        ))
        .expect("append oversized captured event")
        .events;
    assert_eq!(captured.len(), 1);
    drop(store);

    EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config.clone(),
        contexts: context_graph::ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: custom_profile(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(1).unwrap(),
        max_passages_per_batch: NonZeroU32::new(8).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(oversized.len() + 512).unwrap(),
        lease_duration_ms: NonZeroU64::new(1_000).unwrap(),
    })
    .expect("oversized event must produce advancing bounded slices");

    let reopened = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("open a fresh Store handle after the public runner returns");
    let found = reopened
        .find_event_history(
            context_graph::ContextSelector::one(context_id),
            "é",
            EventSearchFilter::default(),
        )
        .expect("search oversized passages");
    assert!(
        found.hits.len() > 1,
        "oversized body must be sliced, never truncated"
    );
    let profile = custom_profile();
    let hard_limit = profile.hard_passage_tokens.get() as usize;
    let overlap_tokens = profile.oversized_event_overlap_tokens.get() as usize;
    let expected_ranges = expected_oversized_ranges(&oversized, hard_limit, overlap_tokens);
    assert_eq!(
        found.hits.len(),
        expected_ranges.len(),
        "every independently derived maximal slice is persisted"
    );

    let preparations = embedder
        .document_preparations
        .lock()
        .expect("document preparation log")
        .clone();
    let inferences = embedder
        .document_inferences
        .lock()
        .expect("document inference log")
        .clone();
    assert_persisted_passages_inferred_once(
        &preparations,
        &inferences,
        found.hits.iter().map(|hit| hit.text.clone()),
        hard_limit,
    );

    let mut previous = None;
    for (hit, &(expected_start, expected_end)) in found.hits.iter().zip(&expected_ranges) {
        assert_eq!(hit.sources.len(), 1);
        let source = &hit.sources[0];
        let start = source.event_start_byte.expect("slice start");
        let end = source.event_end_byte.expect("slice end");
        let passage_start = source.passage_start_byte as usize;
        let passage_end = source.passage_end_byte as usize;
        assert_eq!(source.event_id, captured[0].event.id);
        assert_eq!(source.context_id, context_id);
        assert_eq!(source.connector_id, CONNECTOR);
        assert_eq!(source.session_ref, SESSION);
        assert_eq!(source.source_instance_ref, SOURCE);
        assert_eq!((start, end), (expected_start as u64, expected_end as u64));
        assert!(start < end && end <= oversized.len() as u64);
        assert!(oversized.is_char_boundary(start as usize));
        assert!(oversized.is_char_boundary(end as usize));
        assert!(hit.text.is_char_boundary(passage_start));
        assert!(hit.text.is_char_boundary(passage_end));
        assert_eq!(
            hit.text,
            expected_oversized_passage(&oversized[expected_start..expected_end])
        );
        assert_eq!(
            &hit.text[passage_start..passage_end],
            &oversized[start as usize..end as usize],
            "the persisted source range names exactly the copied body bytes"
        );
        assert_eq!(passage_start, OVERSIZED_HUMAN_PREFIX.len());
        assert_eq!(passage_end, passage_start + (end - start) as usize);
        if expected_end < oversized.len() {
            let next_scalar_end = expected_end
                + oversized[expected_end..]
                    .chars()
                    .next()
                    .expect("unfinished slice has another scalar")
                    .len_utf8();
            assert!(
                fixture_document_token_count(&expected_oversized_passage(
                    &oversized[expected_start..next_scalar_end],
                )) > hard_limit,
                "each non-final cut is the maximal UTF-8-aligned cut under the declared hard limit"
            );
        }
        if let Some((previous_start, previous_end)) = previous {
            assert!(start > previous_start, "source starts strictly advance");
            assert!(end > previous_end, "source ends strictly advance");
            assert_eq!(
                oversized[start as usize..previous_end as usize]
                    .chars()
                    .count(),
                overlap_tokens,
                "consecutive oversized slices share exactly the configured scalar-token overlap"
            );
        } else {
            assert_eq!(start, 0, "the first slice starts at the first body byte");
        }
        previous = Some((start, end));
    }
    assert_eq!(
        previous.expect("at least one oversized slice").1,
        oversized.len() as u64,
        "slices cover the final body byte"
    );
}

/// The runner a detached worker calls leaves the passage vectors on ContextDB's maintained route
/// before it returns: a fresh read-only reader, as `cg find` opens, is served by `INDEXED`, which
/// refuses whenever a selected partition has no complete maintained graph. The statement selects
/// the one (Context, active generation) partition, keeps `is_current` as a residual filter, and
/// the engine discloses exactly that partition key and scope.
#[test]
fn the_public_runner_leaves_the_active_partition_on_the_maintained_indexed_route() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(ProjectionEmbedder::new());
    let config = store_config(
        root.path().join("event-search.db"),
        embedder.embedding_space(),
    );
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open store");
    let context_id = create_context(&store);
    append_fixture_events(&store, context_id);
    drop(store);

    let receipt = EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config.clone(),
        contexts: context_graph::ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: custom_profile(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(2).unwrap(),
        max_passages_per_batch: NonZeroU32::new(2).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(4_096).unwrap(),
        lease_duration_ms: NonZeroU64::new(1_000).unwrap(),
    })
    .expect("projection commits and reports ready");

    let reader = context_graph::ConsumerReader::open(
        &config.db_path,
        context_graph::event_history_reader_declaration(),
    )
    .expect("open the read-only reader in place of a fresh cg find process");
    let mut query = vec![0.0_f32; 768];
    query[0] = 1.0;
    let params = std::collections::HashMap::from([
        (
            "context_id".to_string(),
            contextdb_core::Value::Uuid(context_id.as_uuid()),
        ),
        (
            "generation_id".to_string(),
            contextdb_core::Value::Uuid(receipt.generation_id.as_uuid()),
        ),
        ("is_current".to_string(), contextdb_core::Value::Bool(true)),
        ("query".to_string(), contextdb_core::Value::Vector(query)),
    ]);
    let answer = reader
        .execute(
            "SELECT id FROM event_search_passages WHERE context_id = $context_id AND generation_id = $generation_id AND is_current = $is_current ORDER BY vector_text <=> $query USE VECTOR INDEXED LIMIT 10",
            &params,
        )
        .expect("the maintained indexed route serves the active partition");
    assert_eq!(answer.rows.len(), 1, "the one current passage is found");
    let vector = answer
        .trace
        .vector_search
        .as_ref()
        .expect("the vector statement discloses its route");
    assert_eq!(
        vector.partition_key_columns,
        vec!["context_id".to_string(), "generation_id".to_string()]
    );
    assert_eq!(vector.scope, contextdb_engine::VectorSearchScopeShape::One);
    assert_eq!(
        vector.refusal, None,
        "no selected partition lacks a maintained route"
    );
}
