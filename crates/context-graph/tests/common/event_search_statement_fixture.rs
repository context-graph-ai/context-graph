//! A store whose event history a statement-cost proof can be run against.
//!
//! Shared by the proofs that measure what a `find` or a `status` COSTS in statements: they need a
//! real store, real captured events and a real search projection, and they must not need minutes of
//! a real model to get one. The embedder here is deterministic and declares a profile of its own --
//! it never claims the canonical event-history identity.

#![allow(dead_code)]

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, ContextId, ContextSelector, CreateContext, DistanceMetric, Embedder,
    EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchProfile,
    EventSearchProjectionRunner, EventSearchRunRequest, EvidenceKind, EvidenceProducer, Result,
    Store, StoreConfig,
};
use serde_json::json;

pub const CONNECTOR: &str = "find-statement-cost";
pub const SOURCE: &str = "find-statement-cost-fixture";
const SPACE_ID: &str = "find-statement-cost-deterministic-768";

/// A deterministic 768-dimensional embedder under a profile of its own. It is not the canonical
/// event-history identity and never claims to be: this file measures statement counts, and a real
/// model would only add minutes.
pub struct CountingEmbedder;

impl CountingEmbedder {
    pub fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn output(&self, inference: &EventSearchInference) -> EmbeddingOutput {
        let mut vector = vec![0.0; 768];
        vector[0] = inference.token_count() as f32;
        EmbeddingOutput {
            embedding_space_id: SPACE_ID.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: "find-statement-cost".into(),
                model_name: "deterministic-768".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector,
        }
    }

    /// One token per Unicode scalar, after a fixed prefix, with scalar-aligned byte boundaries.
    fn document_tokens(text: &str) -> (Vec<u32>, Vec<(usize, usize)>) {
        let mut payload = vec![0; 4];
        let mut boundaries = vec![(0, 0); payload.len()];
        for (start, scalar) in text.char_indices() {
            boundaries.push((start, start + scalar.len_utf8()));
            payload.push(scalar as u32);
        }
        (payload, boundaries)
    }
}

impl Embedder for CountingEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event search uses explicit document/query preparation")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: SPACE_ID.into(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "test".into(),
            model_name: "deterministic-768".into(),
            model_version: "1".into(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: SPACE_ID.into(),
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
        let mut payload = vec![0; 4];
        payload.extend(text.chars().map(|scalar| scalar as u32));
        seal.query_builder()
            .token_count(payload.len())
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(self.output(inference))
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(self.output(inference))
    }
}

fn adapter() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "fixture-v1".into(),
        anchor_declaration_version: "fixture-v1".into(),
        anchor_schema: json!({"type":"object", "additionalProperties":true}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn profile() -> EventSearchProfile {
    EventSearchProfile::default()
}

/// A store holding `contexts` Contexts, each with two captured events. When `project` is true the
/// search projection is run over all of them, so every Context has a ready head.
pub struct Fixture {
    _root: tempfile::TempDir,
    pub config: StoreConfig,
    pub context_ids: Vec<ContextId>,
    pub registration: EmbedderRegistration,
}

pub fn fixture(contexts: usize, project: bool) -> Fixture {
    let embedder = Arc::new(CountingEmbedder);
    let root = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: root.path().join("find-statement-cost.db"),
        extra_embedding_spaces: vec![embedder.embedding_space()],
        default_text_embedder: None,
        ..StoreConfig::default()
    };
    let store =
        Store::open_with_embedder_registrations(config.clone(), vec![embedder.registration()])
            .expect("open the fixture store");
    let mut context_ids = Vec::with_capacity(contexts);
    for index in 0..contexts {
        let context_id = store
            .create_context(CreateContext {
                name: format!("find-statement-cost-{index}"),
                labels: Vec::new(),
                properties: Default::default(),
            })
            .expect("create context")
            .id;
        store
            .append_event_batch(AppendEventBatch::live(
                context_id,
                adapter(),
                SOURCE,
                vec![
                    EventDraft {
                        session_ref: format!("session-{index}"),
                        source_event_key: "human-1".into(),
                        source_position: json!({"ordinal": 1}),
                        occurred_at: None,
                        author: "human".into(),
                        kind: "message".into(),
                        render_role: EventRenderRole::HumanMessage,
                        interaction_ref: None,
                        text: Some(format!(
                            "why did context {index} choose the broker-free path?"
                        )),
                        payload: json!({}),
                        anchors: json!({}),
                        source_remainder: json!({}),
                        coverage: EventCoverage::Complete,
                    },
                    EventDraft {
                        session_ref: format!("session-{index}"),
                        source_event_key: "agent-1".into(),
                        source_position: json!({"ordinal": 2}),
                        occurred_at: None,
                        author: "agent:main".into(),
                        kind: "message".into(),
                        render_role: EventRenderRole::AgentMessage,
                        interaction_ref: None,
                        text: Some(format!("context {index} chose direct peer transport")),
                        payload: json!({}),
                        anchors: json!({}),
                        source_remainder: json!({}),
                        coverage: EventCoverage::Complete,
                    },
                ],
            ))
            .expect("append captured events");
        context_ids.push(context_id);
    }
    drop(store);

    // One build receipt describes exactly one Context, so the fixture projects them one at a time.
    if project {
        for context_id in &context_ids {
            EventSearchProjectionRunner::catch_up(EventSearchRunRequest {
                store: config.clone(),
                contexts: ContextSelector::one(*context_id),
                embedder: embedder.registration(),
                profile: profile(),
                embedding_workers: NonZeroUsize::MIN,
                max_events_per_batch: NonZeroU32::new(64).unwrap(),
                max_passages_per_batch: NonZeroU32::new(64).unwrap(),
                max_bytes_per_batch: NonZeroUsize::new(1 << 20).unwrap(),
                lease_duration_ms: NonZeroU64::new(30_000).unwrap(),
            })
            .expect("the projection commits and reports ready");
        }
    }

    Fixture {
        _root: root,
        config,
        context_ids,
        registration: embedder.registration(),
    }
}
