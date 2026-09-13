//! RED proof that an ordinary append extends the ready generation by replacing only its open tail.

#![cfg(context_graph_internal_test_support)]

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, ContextId, CreateContext, DistanceMetric, Embedder, EmbedderRegistration,
    EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchFilter,
    EventSearchPassage, EventSearchProfile, EventSearchProjectionRunner, EventSearchRunRequest,
    EvidenceKind, EvidenceProducer, Result, Store, StoreConfig,
};
use contextdb_core::Value;
use serde_json::json;

const CONNECTOR: &str = "event-search-incremental-tail-red";
const SOURCE: &str = "event-search-incremental-tail-source";
const SESSION: &str = "event-search-incremental-tail-session";
const SPACE: &str = "event-search-incremental-tail-768";

/// This is the adopted renderer's non-scalar prepared-token shape: five model tokens plus two
/// prepared tokens per Unicode scalar. The profile below is the valid ordinary grouping profile.
struct TailEmbedder {
    embedded_documents: Mutex<Vec<String>>,
}

impl TailEmbedder {
    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn output(&self, inference: &EventSearchInference) -> EmbeddingOutput {
        EmbeddingOutput {
            embedding_space_id: SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: CONNECTOR.into(),
                model_name: "role-aware-fixture".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: vec![inference.token_count() as f32; 768],
        }
    }

    fn embedded_len(&self) -> usize {
        self.embedded_documents
            .lock()
            .expect("embedded document log")
            .len()
    }

    fn embedded_since(&self, start: usize) -> Vec<String> {
        self.embedded_documents
            .lock()
            .expect("embedded document log")[start..]
            .to_vec()
    }
}

impl Embedder for TailEmbedder {
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
            model_name: "role-aware-fixture".into(),
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
        "incremental-tail-role-aware-v1".into()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(256)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
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
        self.embedded_documents
            .lock()
            .expect("embedded document log")
            .push(
                inference
                    .token_ids()
                    .iter()
                    .filter(|token| **token != 0)
                    .map(|token| char::from_u32(*token).expect("fixture token is Unicode"))
                    .collect(),
            );
        Ok(self.output(inference))
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(self.output(inference))
    }
}

fn config(path: PathBuf, space: EmbeddingSpace) -> StoreConfig {
    StoreConfig {
        db_path: path,
        extra_embedding_spaces: vec![space],
        default_text_embedder: None,
        ..StoreConfig::default()
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "incremental-tail-v1".into(),
        anchor_declaration_version: "incremental-tail-v1".into(),
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
        interaction_ref: Some("tail-fixture-pair".into()),
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
    config: StoreConfig,
    context_id: ContextId,
    embedder: &Arc<TailEmbedder>,
) -> context_graph::EventSearchBuildReceipt {
    EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
        store: config,
        contexts: context_graph::ContextSelector::one(context_id),
        embedder: embedder.registration(),
        profile: profile(),
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero event batch"),
        max_passages_per_batch: NonZeroU32::new(16).expect("nonzero passage batch"),
        max_bytes_per_batch: NonZeroUsize::new(32_768).expect("nonzero byte batch"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease"),
    })
    .expect("public projection runner completes")
}

fn current_passages(store: &Store, context_id: ContextId) -> Vec<EventSearchPassage> {
    let result = store
        .find_event_history(
            context_graph::ContextSelector::one(context_id),
            "incremental tail",
            EventSearchFilter {
                limit: NonZeroU32::new(16).expect("nonzero result limit"),
                ..EventSearchFilter::default()
            },
        )
        .expect("public search read");
    let mut passages = result
        .hits
        .into_iter()
        .map(|hit| {
            store
                .get_event_search_passage(
                    context_graph::ContextSelector::one(context_id),
                    hit.passage_id,
                )
                .expect("public show read")
                .expect("search hit has a passage")
        })
        .collect::<Vec<_>>();
    passages.sort_by_key(|passage| passage.ordinal);
    passages
}

fn vector(store: &Store, passage: &EventSearchPassage) -> Value {
    store
        .execute_sql_for_test(
            "SELECT vector_text FROM event_search_passages WHERE context_id = $context_id AND id = $passage_id",
            &std::collections::HashMap::from([
                ("context_id".into(), Value::Uuid(passage.context_id.as_uuid())),
                ("passage_id".into(), Value::Uuid(passage.id.as_uuid())),
            ]),
        )
        .expect("inspect persisted vector")
        .rows[0][0]
        .clone()
}

#[test]
fn ready_generation_reuses_closed_passage_and_replaces_only_the_open_tail() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(TailEmbedder {
        embedded_documents: Mutex::new(Vec::new()),
    });
    let config = config(
        root.path().join("incremental-tail.db"),
        embedder.embedding_space(),
    );
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open Store");
    let context_id = store
        .create_context(CreateContext {
            name: "incremental event-search tail".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create Context")
        .id;
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![
                draft("closed-human", 1, EventRenderRole::HumanMessage, "α"),
                draft("closed-request", 2, EventRenderRole::ToolRequest, "β"),
                draft("tail-result", 3, EventRenderRole::ToolResult, "γ"),
                draft("tail-human", 4, EventRenderRole::HumanMessage, "δ"),
            ],
        ))
        .expect("append initial history");
    drop(store);

    let first = run(config.clone(), context_id, &embedder);
    let before =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("reopen initial projection");
    let initial = current_passages(&before, context_id);
    assert_eq!(
        initial.len(),
        2,
        "the adopted grouping has one closed passage and one open tail"
    );
    let closed = initial[0].clone();
    let tail = initial[1].clone();
    let closed_vector = vector(&before, &closed);
    assert!(closed.text.contains('α') && closed.text.contains('β'));
    assert!(tail.text.contains('γ') && tail.text.contains('δ'));
    let initial_inferences = embedder.embedded_len();
    before
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![
                draft("new-request", 5, EventRenderRole::ToolRequest, "ε"),
                draft("new-result", 6, EventRenderRole::ToolResult, "ζ"),
                draft("new-human", 7, EventRenderRole::HumanMessage, "η"),
            ],
        ))
        .expect("append after the ready head");
    drop(before);

    let second = run(config.clone(), context_id, &embedder);
    assert_eq!(
        second.generation_id, first.generation_id,
        "ordinary append extends the ready head"
    );
    assert!(
        !second.active_generation_changed,
        "incremental catch-up does not switch heads"
    );

    let after = Store::open_with_embedder_registrations(config, vec![embedder.registration()])
        .expect("reopen incremental projection");
    let preserved = after
        .get_event_search_passage(context_graph::ContextSelector::one(context_id), closed.id)
        .expect("show closed passage")
        .expect("closed passage remains available");
    assert_eq!(
        preserved, closed,
        "closed ID, text, handle, and source links are reused exactly"
    );
    assert_eq!(
        vector(&after, &preserved),
        closed_vector,
        "closed vector is not rewritten"
    );
    let retired = after
        .get_event_search_passage(context_graph::ContextSelector::one(context_id), tail.id)
        .expect("show retired tail")
        .expect("retired tail remains available to show");
    assert!(
        !retired.is_current,
        "the former open tail is retired only after replacement"
    );
    assert_eq!(retired.text, tail.text);
    assert_eq!(
        retired.sources, tail.sources,
        "retired tail source links remain immutable"
    );

    let current = current_passages(&after, context_id);
    assert!(current.iter().any(|passage| passage.id == closed.id));
    assert!(
        current.iter().all(|passage| passage.id != tail.id),
        "public search excludes retired tails"
    );
    assert!(
        current.iter().any(|passage| passage.text.contains('η')),
        "public search exposes the replacement tail"
    );
    let incremental_inputs = embedder.embedded_since(initial_inferences);
    assert!(
        !incremental_inputs.is_empty(),
        "the replacement tail is embedded"
    );
    assert!(
        incremental_inputs
            .iter()
            .all(|text| !text.contains('α') && !text.contains('β')),
        "closed history is never re-embedded"
    );
    assert!(
        incremental_inputs
            .iter()
            .any(|text| text.contains('γ') && text.contains('δ')),
        "the previous open tail is reopened for rendering"
    );
    assert!(
        incremental_inputs.iter().any(|text| text.contains('η')),
        "newly appended events join the replacement tail"
    );
}
