//! `PreparedEventSearchDocument`/`PreparedEventSearchQuery` carry a sealing embedder's private
//! prepared model input (tokenizer ids). Nothing about that payload is meant to be visible outside
//! the sealing embedder itself. Debug output exposes only metadata, keeping token ids out of
//! logs, panic messages, and test failures. Every distinctive payload id below must be absent
//! from the Debug string.

use context_graph::embedder::{
    Embedder, EventSearchCapability, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    DistanceMetric, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EventSearchEmbedder,
    EvidenceKind, Result,
};

/// Deliberately unusual token ids that would never coincidentally appear in any other field of the
/// struct (a token count, a byte boundary, a profile id) so their presence in the Debug string
/// unambiguously means the payload leaked.
const DISTINCTIVE_TOKEN_IDS: [u32; 3] = [7_654_321, 7_654_322, 7_654_323];

struct FixedPayloadEmbedder;

impl Embedder for FixedPayloadEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unimplemented!("this double only exercises Debug redaction")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: "fixed_payload_test_double".to_string(),
            dimension: 1,
            metric: DistanceMetric::Cosine,
            model_family: "test-double".to_string(),
            model_name: "test-double".to_string(),
            model_version: "1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: Vec::new(),
        }
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(64)
    }

    fn prepare_document_hook(
        &self,
        _text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        seal.document_builder()
            .token_count(DISTINCTIVE_TOKEN_IDS.len())
            .byte_boundaries(vec![(0, 1), (1, 2), (2, 3)])
            .payload(DISTINCTIVE_TOKEN_IDS.to_vec())
            .seal()
    }

    fn prepare_query_hook(
        &self,
        _text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        seal.query_builder()
            .token_count(DISTINCTIVE_TOKEN_IDS.len())
            .payload(DISTINCTIVE_TOKEN_IDS.to_vec())
            .seal()
    }
}

#[test]
fn prepared_document_debug_output_contains_no_token_payload() {
    let embedder: &dyn Embedder = &FixedPayloadEmbedder;
    let prepared = embedder
        .prepare_document("irrelevant")
        .expect("preparation must succeed");
    let debug_output = format!("{prepared:?}");
    for id in DISTINCTIVE_TOKEN_IDS {
        assert!(
            !debug_output.contains(&id.to_string()),
            "PreparedEventSearchDocument's Debug output must never print raw token ids from its \
             sealed payload, but {id} appeared in {debug_output:?}"
        );
    }
}

#[test]
fn prepared_query_debug_output_contains_no_token_payload() {
    let embedder: &dyn Embedder = &FixedPayloadEmbedder;
    let prepared = embedder
        .prepare_query("irrelevant")
        .expect("preparation must succeed");
    let debug_output = format!("{prepared:?}");
    for id in DISTINCTIVE_TOKEN_IDS {
        assert!(
            !debug_output.contains(&id.to_string()),
            "PreparedEventSearchQuery's Debug output must never print raw token ids from its \
             sealed payload, but {id} appeared in {debug_output:?}"
        );
    }
}
