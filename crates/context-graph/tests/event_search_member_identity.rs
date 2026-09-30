//! A retained generation and another Context both reuse ordinal zero; their commit guards must
//! therefore identify the persisted membership row rather than its generation-local ordinal.

#![cfg(context_graph_internal_test_support)]

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    AppendEventBatch, CreateContext, DistanceMetric, Embedder, EmbedderRegistration,
    EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EventAdapterDeclaration, EventCoverage, EventDraft, EventRenderRole, EventSearchBuildRequest,
    EventSearchProfile, EventSearchRebuildRequest, EvidenceKind, EvidenceProducer, Result, Store,
    StoreConfig,
};
use serde_json::json;

const CONNECTOR: &str = "event-search-member-identity";
const SOURCE: &str = "event-search-member-identity-source";
const SESSION: &str = "event-search-member-identity-session";
const SEARCH_SPACE: &str = "event-search-member-identity-768";

struct IdentityEmbedder;

impl IdentityEmbedder {
    fn registration(self: &Arc<Self>) -> EmbedderRegistration {
        EmbedderRegistration {
            space: self.embedding_space(),
            embedder: self.clone(),
        }
    }

    fn output(&self) -> EmbeddingOutput {
        EmbeddingOutput {
            embedding_space_id: SEARCH_SPACE.into(),
            dimension: 768,
            producer: EvidenceProducer {
                system: "event-search-member-identity".into(),
                model_name: "deterministic".into(),
                model_version: "1".into(),
                pipeline_version: "1".into(),
            },
            vector: vec![0.0; 768],
        }
    }
}

impl Embedder for IdentityEmbedder {
    fn embed(&self, _: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("event-search uses role-aware preparation")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: SEARCH_SPACE.into(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "test".into(),
            model_name: "member-identity".into(),
            model_version: "1".into(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: SEARCH_SPACE.into(),
                table_name: "event_search_passages".into(),
                vector_column: "vector_text".into(),
            }],
        }
    }

    fn encoding_profile_id(&self) -> String {
        "event-search-member-identity-v1".into()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(4_096)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        seal.document_builder()
            .token_count(text.len())
            .byte_boundaries((0..text.len()).map(|index| (index, index + 1)).collect())
            .payload(text.bytes().map(u32::from).collect())
            .seal()
    }

    fn prepare_query_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        seal.query_builder()
            .token_count(text.len())
            .payload(text.bytes().map(u32::from).collect())
            .seal()
    }

    fn embed_prepared_document_hook(&self, _: &EventSearchInference) -> Result<EmbeddingOutput> {
        Ok(self.output())
    }

    fn embed_prepared_query_hook(&self, _: &EventSearchInference) -> Result<EmbeddingOutput> {
        Ok(self.output())
    }
}

fn declaration() -> EventAdapterDeclaration {
    EventAdapterDeclaration {
        connector_id: CONNECTOR.into(),
        adapter_version: "member-identity-v1".into(),
        anchor_declaration_version: "member-identity-v1".into(),
        anchor_schema: json!({"type": "object"}),
        source_position_order: vec!["/ordinal".into()],
    }
}

fn append_event(store: &Store, context_id: context_graph::ContextId, ordinal: u64, text: &str) {
    store
        .append_event_batch(AppendEventBatch::live(
            context_id,
            declaration(),
            SOURCE,
            vec![EventDraft {
                session_ref: SESSION.into(),
                source_event_key: format!("member-identity-{ordinal}"),
                source_position: json!({"ordinal": ordinal}),
                occurred_at: None,
                author: "fixture".into(),
                kind: "message".into(),
                render_role: EventRenderRole::HumanMessage,
                interaction_ref: None,
                text: Some(text.into()),
                payload: json!({}),
                anchors: json!({}),
                source_remainder: json!({}),
                coverage: EventCoverage::Complete,
            }],
        ))
        .expect("append one source event");
}

fn build_request(
    context_id: context_graph::ContextId,
    embedder: EmbedderRegistration,
) -> EventSearchBuildRequest {
    EventSearchBuildRequest {
        context_id,
        embedder,
        profile: EventSearchProfile::default(),
        embedding_workers: NonZeroUsize::MIN,
        max_events_per_batch: NonZeroU32::new(8).expect("nonzero event batch"),
        max_passages_per_batch: NonZeroU32::new(8).expect("nonzero passage batch"),
        max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero byte batch"),
        lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease"),
    }
}

#[test]
fn retained_generation_and_other_context_commit_their_ordinal_zero_members() {
    let root = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(IdentityEmbedder);
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: root.path().join("member-identity.db"),
            extra_embedding_spaces: vec![embedder.embedding_space()],
            default_text_embedder: None,
            ..StoreConfig::default()
        },
        vec![embedder.registration()],
    )
    .expect("open store");
    let first_context = store
        .create_context(CreateContext {
            name: "first member identity context".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create first Context")
        .id;
    let other_context = store
        .create_context(CreateContext {
            name: "other member identity context".into(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("create other Context")
        .id;
    append_event(&store, first_context, 1, "first generation source");
    append_event(&store, other_context, 1, "other Context source");

    let first = store
        .catch_up_event_search(build_request(first_context, embedder.registration()))
        .expect("first nonempty generation commits");
    let other = store
        .catch_up_event_search(build_request(other_context, embedder.registration()))
        .expect("other Context ordinal-zero membership commits after the first generation");
    let rebuilt = store
        .rebuild_event_search(EventSearchRebuildRequest {
            context_id: first_context,
            interpretation_generation_id: None,
            embedder: embedder.registration(),
            profile: EventSearchProfile::default(),
            embedding_workers: NonZeroUsize::MIN,
            max_events_per_batch: NonZeroU32::new(8).expect("nonzero event batch"),
            max_passages_per_batch: NonZeroU32::new(8).expect("nonzero passage batch"),
            max_bytes_per_batch: NonZeroUsize::new(4_096).expect("nonzero byte batch"),
            lease_duration_ms: NonZeroU64::new(1_000).expect("nonzero lease"),
        })
        .expect("second generation retains and commits its own ordinal-zero membership");

    assert_ne!(first.generation_id, rebuilt.generation_id);
    assert_eq!(first.embedded_event_count, 1);
    assert_eq!(other.embedded_event_count, 1);
    assert_eq!(rebuilt.embedded_event_count, 1);
}
