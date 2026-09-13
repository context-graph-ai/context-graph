//! Independent public-contract regressions for the event-history renderer.
//!
//! Expected bytes and ranges live here, not in the projection implementation.

#![cfg(context_graph_internal_test_support)]

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::{Arc, Mutex};

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, CgError, ContextId, CreateContext, DistanceMetric, Embedder,
    EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchEmbedder,
    EventSearchFilter, EventSearchProfile, EventSearchProjectionRunner, EventSearchRunRequest,
    EvidenceKind, EvidenceProducer, Result, Store, StoreConfig,
};
use serde_json::json;

const SPACE: &str = "event-search-red-role-aware-768";
const CONNECTOR: &str = "renderer-contracts";
const SOURCE: &str = "independent-red";
const SESSION: &str = "renderer-contracts-session";
const DOCUMENT_OVERHEAD: usize = 5;

/// A deliberately non-1:1 role-aware tokenizer: every Unicode scalar has two tokens sharing its
/// UTF-8 byte span, and document preparation has five zero-width model tokens.
struct RoleAwareEmbedder {
    embedded_document_counts: Mutex<Vec<usize>>,
    token_limit: usize,
    declared_space_id: Option<String>,
    declared_binding_owner: Option<String>,
}

impl RoleAwareEmbedder {
    fn new() -> Self {
        Self::with_token_limit(64)
    }

    fn with_token_limit(token_limit: usize) -> Self {
        Self {
            embedded_document_counts: Mutex::new(Vec::new()),
            token_limit,
            declared_space_id: None,
            declared_binding_owner: None,
        }
    }

    fn with_declared_binding_owner(token_limit: usize, binding_owner: &str) -> Self {
        Self {
            embedded_document_counts: Mutex::new(Vec::new()),
            token_limit,
            declared_space_id: None,
            declared_binding_owner: Some(binding_owner.into()),
        }
    }

    fn with_declared_space_and_binding_owner(
        token_limit: usize,
        space_id: &str,
        binding_owner: &str,
    ) -> Self {
        Self {
            embedded_document_counts: Mutex::new(Vec::new()),
            token_limit,
            declared_space_id: Some(space_id.into()),
            declared_binding_owner: Some(binding_owner.into()),
        }
    }

    fn space() -> EmbeddingSpace {
        Self::space_with_binding_owner(SPACE)
    }

    fn space_with_binding_owner(binding_owner: &str) -> EmbeddingSpace {
        Self::space_with_ids(SPACE, binding_owner)
    }

    fn space_with_ids(space_id: &str, binding_owner: &str) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: space_id.into(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "test".into(),
            model_name: "role-aware-non-scalar".into(),
            model_version: "1".into(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: binding_owner.into(),
                table_name: "event_search_passages".into(),
                vector_column: "vector_text".into(),
            }],
        }
    }

    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: Self::space(),
            embedder: self.clone(),
        }
    }

    fn declared_registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn document_payload(text: &str) -> (Vec<u32>, Vec<(usize, usize)>) {
        let mut payload = vec![0; DOCUMENT_OVERHEAD];
        let mut boundaries = vec![(0, 0); DOCUMENT_OVERHEAD];
        for (start, scalar) in text.char_indices() {
            let end = start + scalar.len_utf8();
            // Two model tokens, both mapped to the same valid source-byte span.
            payload.extend([scalar as u32, scalar as u32]);
            boundaries.extend([(start, end), (start, end)]);
        }
        (payload, boundaries)
    }

    fn output(&self, count: usize) -> EmbeddingOutput {
        EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: "renderer-contract-red".into(),
                model_name: "role-aware-non-scalar".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: vec![count as f32; 768],
        }
    }
}

impl Embedder for RoleAwareEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event-search uses role-aware preparation")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        Self::space_with_ids(
            self.declared_space_id.as_deref().unwrap_or(SPACE),
            self.declared_binding_owner.as_deref().unwrap_or(SPACE),
        )
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(self.token_limit)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let (payload, boundaries) = Self::document_payload(text);
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
        let (payload, _) = Self::document_payload(text);
        seal.query_builder()
            .token_count(payload.len())
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        self.embedded_document_counts
            .lock()
            .expect("document count log")
            .push(inference.token_count());
        Ok(self.output(inference.token_count()))
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(self.output(inference.token_count()))
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "independent-contract-v1".into(),
        anchor_declaration_version: "independent-contract-v1".into(),
        anchor_schema: json!({"type": "object", "additionalProperties": true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn bound_config(path: std::path::PathBuf) -> StoreConfig {
    StoreConfig {
        db_path: path,
        default_text_embedder: None,
        extra_embedding_spaces: vec![RoleAwareEmbedder::space()],
        ..StoreConfig::default()
    }
}

fn profile() -> EventSearchProfile {
    EventSearchProfile {
        target_passage_tokens: NonZeroU32::new(56).unwrap(),
        hard_passage_tokens: NonZeroU32::new(64).unwrap(),
        ordinary_overlap_events: NonZeroU32::new(1).unwrap(),
        oversized_event_overlap_tokens: NonZeroU32::new(4).unwrap(),
        ..EventSearchProfile::default()
    }
}

fn ordinary_profile() -> EventSearchProfile {
    EventSearchProfile {
        // Independently counted with this model: user/tool-request/tool-result is 177 prepared
        // tokens, all four units are 227, and the request/result pair is 127. This makes the
        // user boundary a legal soft cut while retaining a one-event ordinary overlap.
        target_passage_tokens: NonZeroU32::new(177).unwrap(),
        hard_passage_tokens: NonZeroU32::new(256).unwrap(),
        ordinary_overlap_events: NonZeroU32::new(1).unwrap(),
        oversized_event_overlap_tokens: NonZeroU32::new(4).unwrap(),
        ..EventSearchProfile::default()
    }
}

fn ordinary_overlap_three_profile() -> EventSearchProfile {
    EventSearchProfile {
        // The independent scalar-count oracle gives 177 prepared tokens for the first three
        // units. With three retained ordinary events, each later cut is 227 prepared tokens.
        target_passage_tokens: NonZeroU32::new(177).unwrap(),
        hard_passage_tokens: NonZeroU32::new(256).unwrap(),
        ordinary_overlap_events: NonZeroU32::new(3).unwrap(),
        oversized_event_overlap_tokens: NonZeroU32::new(4).unwrap(),
        ..EventSearchProfile::default()
    }
}

fn create_context(store: &Store) -> ContextId {
    store
        .create_context(CreateContext {
            name: "event-search renderer contracts".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create context")
        .id
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
        interaction_ref: Some("tool-pair-1".into()),
        text: Some(text.into()),
        payload: json!({}),
        anchors: json!({}),
        source_remainder: json!({}),
        coverage: EventCoverage::Complete,
    }
}

fn run_request(
    config: StoreConfig,
    context_id: ContextId,
    embedder: &Arc<RoleAwareEmbedder>,
    profile: EventSearchProfile,
    max_events_per_batch: u32,
) {
    try_run_request(config, context_id, embedder, profile, max_events_per_batch)
        .expect("the public projection runner must commit the configured contract");
}

fn try_run_request(
    config: StoreConfig,
    context_id: ContextId,
    embedder: &Arc<RoleAwareEmbedder>,
    profile: EventSearchProfile,
    max_events_per_batch: u32,
) -> Result<context_graph::EventSearchBuildReceipt> {
    EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config,
        contexts: context_graph::ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile,
        max_events_per_batch: NonZeroU32::new(max_events_per_batch).unwrap(),
        max_passages_per_batch: NonZeroU32::new(32).unwrap(),
        max_bytes_per_batch: NonZeroUsize::new(32_768).unwrap(),
        lease_duration_ms: NonZeroU64::new(1_000).unwrap(),
    })
}

fn all_passages(store: &Store, context_id: ContextId) -> Vec<context_graph::EventSearchPassage> {
    let result = store
        .find_event_history(
            context_graph::ContextSelector::one(context_id),
            "éα",
            EventSearchFilter {
                limit: NonZeroU32::new(100).unwrap(),
                ..EventSearchFilter::default()
            },
        )
        .expect("public search readback");
    let mut passages = result
        .hits
        .into_iter()
        .map(|hit| {
            store
                .get_event_search_passage(
                    context_graph::ContextSelector::one(context_id),
                    hit.passage_id,
                )
                .expect("public passage readback")
                .expect("ranked passage persists")
        })
        .collect::<Vec<_>>();
    passages.sort_by_key(|passage| passage.ordinal);
    passages
}

fn expected_oversized_ranges(
    embedder: &RoleAwareEmbedder,
    prefix: &str,
    body: &str,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    loop {
        let end = body[start..]
            .char_indices()
            .map(|(offset, scalar)| start + offset + scalar.len_utf8())
            .filter(|end| {
                embedder
                    .prepare_document(&format!("{prefix}{}", &body[start..*end]))
                    .expect("independent exact preparation")
                    .token_count()
                    <= profile().hard_passage_tokens.get() as usize
            })
            .last()
            .expect("configured hard limit admits one body scalar");
        ranges.push((start, end));
        if end == body.len() {
            return ranges;
        }
        let mut token_starts = Vec::new();
        for (offset, _) in body[start..end].char_indices() {
            token_starts.extend([start + offset, start + offset]);
        }
        let next = token_starts
            [token_starts.len() - profile().oversized_event_overlap_tokens.get() as usize];
        assert!(
            next > start,
            "configured overlap must advance the next UTF-8 slice"
        );
        start = next;
    }
}

#[test]
fn matching_custom_event_search_binding_is_accepted_and_persists_a_public_projection() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = bound_config(root.path().join("bound.db"));
    let embedder = Arc::new(RoleAwareEmbedder::new());
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("the explicit matching event-search vector binding is declared and usable");
    let context_id = create_context(&store);
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![draft("one", 1, EventRenderRole::HumanMessage, "é")],
        ))
        .expect("public capture");
    drop(store);
    run_request(config.clone(), context_id, &embedder, profile(), 1);
    let reopened = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("reopen with the same explicit declaration and registration");
    let passages = all_passages(&reopened, context_id);
    assert_eq!(passages.len(), 1);
    assert_eq!(passages[0].embedding_space_id, SPACE);
}

#[test]
fn missing_or_wrong_event_search_binding_is_refused() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(RoleAwareEmbedder::new());
    let mut missing = bound_config(root.path().join("missing.db"));
    missing.extra_embedding_spaces[0].bindings.clear();
    let missing_result =
        Store::open_with_embedder_registrations(missing, vec![embedder.registration()]);

    let mut wrong = bound_config(root.path().join("wrong.db"));
    wrong.extra_embedding_spaces[0].bindings[0] = EmbeddingSpaceBinding {
        embedding_space_id: SPACE.into(),
        table_name: "evidence".into(),
        vector_column: "vector_text".into(),
    };
    let wrong_result =
        Store::open_with_embedder_registrations(wrong, vec![embedder.registration()]);
    assert_binding_mismatch(
        missing_result
            .expect_err("the configured space must declare the registered passage binding"),
        "",
        "event_search_passages:vector_text",
    );
    assert_binding_mismatch(
        wrong_result.expect_err("the configured space must declare the registered passage binding"),
        "evidence:vector_text",
        "event_search_passages:vector_text",
    );
}

fn assert_binding_mismatch(error: CgError, expected: &str, actual: &str) {
    match error {
        CgError::EmbeddingSpaceConfigMismatch {
            field,
            expected: observed_expected,
            actual: observed_actual,
        } => {
            assert_eq!(field, "bindings");
            assert_eq!(observed_expected, expected);
            assert_eq!(observed_actual, actual);
        }
        other => panic!(
            "the declared-versus-registered event-search binding must return EmbeddingSpaceConfigMismatch; got {other:?}"
        ),
    }
}

#[test]
fn utf8_oversized_slices_use_prepared_tokens_for_maximal_hard_bound_and_exact_overlap() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = bound_config(root.path().join("utf8.db"));
    let embedder = Arc::new(RoleAwareEmbedder::new());
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open explicit bound store");
    let context_id = create_context(&store);
    let body = "éα".repeat(24);
    let captured = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![draft("oversized", 1, EventRenderRole::HumanMessage, &body)],
        ))
        .expect("public capture")
        .events;
    let event_id = captured[0].event.id;
    drop(store);
    run_request(config.clone(), context_id, &embedder, profile(), 1);
    let reopened = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("reopen explicit bound store");
    let passages = all_passages(&reopened, context_id);
    let prefix = "User [author=\"Renée\"]: ";
    let expected = expected_oversized_ranges(embedder.as_ref(), prefix, &body);
    assert_eq!(passages.len(), expected.len());
    for (passage, &(start, end)) in passages.iter().zip(&expected) {
        assert_eq!(passage.text, format!("{prefix}{}", &body[start..end]));
        assert_eq!(passage.sources.len(), 1);
        let source = &passage.sources[0];
        assert_eq!(source.event_id, event_id);
        assert_eq!(
            (source.event_start_byte, source.event_end_byte),
            (Some(start as u64), Some(end as u64))
        );
        assert_eq!(
            &passage.text[source.passage_start_byte as usize..source.passage_end_byte as usize],
            &body[start..end]
        );
        let actual = embedder
            .prepare_document(&passage.text)
            .expect("actual prepared count");
        assert!(actual.token_count() <= profile().hard_passage_tokens.get() as usize);
    }
    for (left, right) in expected.windows(2).map(|pair| (pair[0], pair[1])) {
        let overlapped_tokens = body[left.0..left.1]
            .char_indices()
            .flat_map(|(offset, _)| [left.0 + offset, left.0 + offset])
            .filter(|start| *start >= right.0)
            .count();
        assert_eq!(
            overlapped_tokens,
            profile().oversized_event_overlap_tokens.get() as usize
        );
    }
    assert!(
        embedder
            .embedded_document_counts
            .lock()
            .expect("document count log")
            .iter()
            .all(|count| *count <= profile().hard_passage_tokens.get() as usize),
        "the checked inference path receives only prepared model-bounded documents"
    );
}

#[test]
fn ordinary_user_tool_boundaries_and_configured_overlap_do_not_depend_on_claim_batch_size() {
    fn build(max_events: u32) -> Vec<(String, Vec<String>)> {
        let root = tempfile::tempdir().expect("tempdir");
        let config = bound_config(root.path().join("ordinary.db"));
        let embedder = Arc::new(RoleAwareEmbedder::with_token_limit(256));
        let store =
            Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
                .expect("open explicit bound store");
        let context_id = create_context(&store);
        let captured = store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                SOURCE,
                vec![
                    draft("user-before", 1, EventRenderRole::HumanMessage, "α"),
                    draft("tool-request", 2, EventRenderRole::ToolRequest, "β"),
                    draft("tool-result", 3, EventRenderRole::ToolResult, "γ"),
                    draft("user-after", 4, EventRenderRole::HumanMessage, "δ"),
                ],
            ))
            .expect("public capture")
            .events;
        let keys = captured
            .iter()
            .map(|captured| (captured.event.id, captured.event.source_event_key.clone()))
            .collect::<HashMap<_, _>>();
        drop(store);
        run_request(
            config.clone(),
            context_id,
            &embedder,
            ordinary_profile(),
            max_events,
        );
        let reopened =
            Store::open_with_embedder_registrations(config, vec![embedder.registration()])
                .expect("reopen explicit bound store");
        all_passages(&reopened, context_id)
            .into_iter()
            .map(|passage| {
                (
                    passage.text,
                    passage
                        .sources
                        .iter()
                        .map(|source| keys[&source.event_id].clone())
                        .collect(),
                )
            })
            .collect()
    }

    let expected = vec![
        (
            "User [author=\"Renée\"]: α\nTool call [author=\"Renée\"]: β\nTool result [author=\"Renée\"]: γ".to_string(),
            vec![
                "user-before".to_string(),
                "tool-request".to_string(),
                "tool-result".to_string(),
            ],
        ),
        (
            "Tool result [author=\"Renée\"]: γ\nUser [author=\"Renée\"]: δ".to_string(),
            vec!["tool-result".to_string(), "user-after".to_string()],
        ),
    ];
    let one_at_a_time = build(1);
    let one_claim = build(4);
    assert_eq!(
        one_at_a_time, expected,
        "one-event claims must render the frozen passage contract"
    );
    assert_eq!(
        one_claim, expected,
        "four-event claims must render the frozen passage contract"
    );
    assert_eq!(
        one_claim[0].1.last(),
        one_claim[1].1.first(),
        "the only adjacent ordinary overlap is the final pre-cut event"
    );
    assert_eq!(one_claim[1].1[0], "tool-result");
}

#[test]
fn three_event_overlap_has_exact_bytes_and_sources_without_claim_replay_duplicates() {
    fn build(max_events: u32) -> Vec<(String, Vec<String>)> {
        let root = tempfile::tempdir().expect("tempdir");
        let config = bound_config(root.path().join("three-overlap.db"));
        let embedder = Arc::new(RoleAwareEmbedder::with_token_limit(256));
        let store =
            Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
                .expect("open explicit bound store");
        let context_id = create_context(&store);
        let captured = store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                declaration(),
                SOURCE,
                vec![
                    draft("user-one", 1, EventRenderRole::HumanMessage, "α"),
                    draft("request", 2, EventRenderRole::ToolRequest, "β"),
                    draft("result", 3, EventRenderRole::ToolResult, "γ"),
                    draft("user-two", 4, EventRenderRole::HumanMessage, "δ"),
                    draft("user-three", 5, EventRenderRole::HumanMessage, "ε"),
                ],
            ))
            .expect("public capture")
            .events;
        let keys = captured
            .iter()
            .map(|captured| (captured.event.id, captured.event.source_event_key.clone()))
            .collect::<HashMap<_, _>>();
        drop(store);
        run_request(
            config.clone(),
            context_id,
            &embedder,
            ordinary_overlap_three_profile(),
            max_events,
        );
        let reopened =
            Store::open_with_embedder_registrations(config, vec![embedder.registration()])
                .expect("reopen explicit bound store");
        all_passages(&reopened, context_id)
            .into_iter()
            .map(|passage| {
                (
                    passage.text,
                    passage
                        .sources
                        .iter()
                        .map(|source| keys[&source.event_id].clone())
                        .collect(),
                )
            })
            .collect()
    }

    // These are independently rendered frozen-grammar units. The tokenizer contributes two
    // tokens per Unicode scalar and five document-only tokens: the first three units total 177,
    // and each three-event overlap plus its new user unit totals 227, below the 256 hard limit.
    let expected = vec![
        (
            "User [author=\"Renée\"]: α\nTool call [author=\"Renée\"]: β\nTool result [author=\"Renée\"]: γ".to_string(),
            vec!["user-one".to_string(), "request".to_string(), "result".to_string()],
        ),
        (
            "User [author=\"Renée\"]: α\nTool call [author=\"Renée\"]: β\nTool result [author=\"Renée\"]: γ\nUser [author=\"Renée\"]: δ".to_string(),
            vec![
                "user-one".to_string(),
                "request".to_string(),
                "result".to_string(),
                "user-two".to_string(),
            ],
        ),
        (
            "Tool call [author=\"Renée\"]: β\nTool result [author=\"Renée\"]: γ\nUser [author=\"Renée\"]: δ\nUser [author=\"Renée\"]: ε".to_string(),
            vec![
                "request".to_string(),
                "result".to_string(),
                "user-two".to_string(),
                "user-three".to_string(),
            ],
        ),
    ];
    let four_event_claims = build(4);
    let five_event_claims = build(5);
    assert_eq!(
        four_event_claims, expected,
        "four-event claims must not replay a prior cut"
    );
    assert_eq!(
        five_event_claims, expected,
        "five-event claims must render the same source groups"
    );
}

#[test]
fn preferred_user_boundary_keeps_a_fitting_assistant_with_the_request_activity() {
    let root = tempfile::tempdir().expect("tempdir");
    let config = bound_config(root.path().join("assistant-boundary.db"));
    let embedder = Arc::new(RoleAwareEmbedder::with_token_limit(256));
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open explicit bound store");
    let context_id = create_context(&store);
    let captured = store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![
                draft("user", 1, EventRenderRole::HumanMessage, "α"),
                draft("request", 2, EventRenderRole::ToolRequest, "β"),
                draft("result", 3, EventRenderRole::ToolResult, "γ"),
                draft("assistant", 4, EventRenderRole::AgentMessage, "δ"),
                draft("next-user", 5, EventRenderRole::HumanMessage, "ε"),
            ],
        ))
        .expect("public capture")
        .events;
    let keys = captured
        .iter()
        .map(|captured| (captured.event.id, captured.event.source_event_key.clone()))
        .collect::<HashMap<_, _>>();
    drop(store);
    run_request(config.clone(), context_id, &embedder, ordinary_profile(), 5);
    let reopened = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("reopen explicit bound store");
    let actual = all_passages(&reopened, context_id)
        .into_iter()
        .map(|passage| {
            (
                passage.text,
                passage
                    .sources
                    .iter()
                    .map(|source| keys[&source.event_id].clone())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    // The first four frozen units are 237 prepared tokens (including the five model-only
    // tokens), so they fit the 256 hard limit. The 180 target is only a reason to cut at the
    // following preferred User boundary; it cannot split at 177 before the Assistant.
    let expected = vec![
        (
            "User [author=\"Renée\"]: α\nTool call [author=\"Renée\"]: β\nTool result [author=\"Renée\"]: γ\nAssistant [author=\"Renée\"]: δ".to_string(),
            vec![
                "user".to_string(),
                "request".to_string(),
                "result".to_string(),
                "assistant".to_string(),
            ],
        ),
        (
            "Assistant [author=\"Renée\"]: δ\nUser [author=\"Renée\"]: ε".to_string(),
            vec!["assistant".to_string(), "next-user".to_string()],
        ),
    ];
    assert_eq!(actual, expected);
}

#[test]
fn projection_refuses_configured_or_embedder_binding_with_a_wrong_owner_id() {
    const WRONG_OWNER: &str = "wrong-event-search-owner";

    let configured_root = tempfile::tempdir().expect("tempdir");
    let mut configured = bound_config(configured_root.path().join("configured-owner.db"));
    configured.extra_embedding_spaces[0].bindings[0].embedding_space_id = WRONG_OWNER.into();
    let configured_embedder = Arc::new(RoleAwareEmbedder::with_token_limit(256));
    let configured_result = match Store::open_with_embedder_registrations(
        configured.clone(),
        vec![configured_embedder.registration()],
    ) {
        Err(error) => Err(error),
        Ok(store) => {
            let context_id = create_context(&store);
            drop(store);
            try_run_request(
                configured,
                context_id,
                &configured_embedder,
                ordinary_profile(),
                1,
            )
        }
    };

    let declared_root = tempfile::tempdir().expect("tempdir");
    let declared = bound_config(declared_root.path().join("declared-owner.db"));
    let declared_embedder = Arc::new(RoleAwareEmbedder::with_declared_binding_owner(
        256,
        WRONG_OWNER,
    ));
    let declared_result = match Store::open_with_embedder_registrations(
        declared.clone(),
        vec![declared_embedder.registration()],
    ) {
        Err(error) => Err(error),
        Ok(store) => {
            let context_id = create_context(&store);
            drop(store);
            try_run_request(
                declared,
                context_id,
                &declared_embedder,
                ordinary_profile(),
                1,
            )
        }
    };

    match (configured_result, declared_result) {
        (Err(configured_error), Err(declared_error)) => {
            assert_binding_mismatch(
                configured_error,
                "wrong-event-search-owner:event_search_passages:vector_text",
                "event-search-red-role-aware-768:event_search_passages:vector_text",
            );
            assert_binding_mismatch(
                declared_error,
                "event-search-red-role-aware-768:event_search_passages:vector_text",
                "wrong-event-search-owner:event_search_passages:vector_text",
            );
        }
        (configured, declared) => panic!(
            "configured and embedder-declared bindings must both refuse a wrong owner ID; configured={configured:?}; declared={declared:?}"
        ),
    }
}

#[test]
fn matching_wrong_binding_owner_is_refused_before_persistence_and_corrected_declaration_reopens() {
    const CUSTOM_SPACE: &str = "event-search.binding-owner-self-consistency/v1";
    const WRONG_OWNER: &str = "different-owner.binding-owner-self-consistency/v1";

    let root = tempfile::tempdir().expect("tempdir");
    let db_path = root.path().join("binding-owner-self-consistency.db");
    let mut corrected = bound_config(db_path);
    corrected.extra_embedding_spaces[0].embedding_space_id = CUSTOM_SPACE.into();
    corrected.extra_embedding_spaces[0].bindings[0].embedding_space_id = CUSTOM_SPACE.into();
    let expected_space = corrected.extra_embedding_spaces[0].clone();

    let mut malformed = corrected.clone();
    malformed.extra_embedding_spaces[0].bindings[0].embedding_space_id = WRONG_OWNER.into();
    let malformed_embedder = Arc::new(RoleAwareEmbedder::with_declared_space_and_binding_owner(
        256,
        CUSTOM_SPACE,
        WRONG_OWNER,
    ));
    assert_binding_mismatch(
        Store::open_with_embedder_registrations(
            malformed,
            vec![malformed_embedder.declared_registration()],
        )
        .expect_err("a matching malformed binding owner must be refused before persistence"),
        CUSTOM_SPACE,
        WRONG_OWNER,
    );

    let corrected_embedder = Arc::new(RoleAwareEmbedder::with_declared_space_and_binding_owner(
        256,
        CUSTOM_SPACE,
        CUSTOM_SPACE,
    ));
    let store = Store::open_with_embedder_registrations(
        corrected.clone(),
        vec![corrected_embedder.declared_registration()],
    )
    .expect("the corrected declaration must open after the malformed declaration was refused");
    assert!(
        store
            .list_embedding_spaces()
            .expect("list persisted embedding spaces")
            .contains(&expected_space),
        "the corrected declaration must persist its own space and binding rows"
    );
    drop(store);
    Store::open_with_embedder_registrations(
        corrected,
        vec![corrected_embedder.declared_registration()],
    )
    .expect("the corrected declaration must reopen without poisoned rows");
}
