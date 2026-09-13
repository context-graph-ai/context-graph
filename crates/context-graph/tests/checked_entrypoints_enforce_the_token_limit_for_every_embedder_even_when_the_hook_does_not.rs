//! Event-search support and its finite model ceiling are one declaration. An embedder whose hooks
//! do no size checking declares a deliberately small ceiling here; the checked public entrypoints
//! must refuse larger prepared values before either inference hook runs.

use std::sync::atomic::{AtomicUsize, Ordering};

use context_graph::embedder::{
    Embedder, EventSearchCapability, EventSearchInference, EventSearchSeal,
    PreparedEventSearchDocument, PreparedEventSearchQuery,
};
use context_graph::{
    DistanceMetric, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EventSearchEmbedder,
    EvidenceKind, EvidenceProducer, Result,
};

/// Implements every event-search hook without checking size itself. Its inference hooks accept
/// whatever they receive, so a refusal can only come from the checked entrypoint.
struct SmallCeilingEmbedder {
    document_inference_calls: AtomicUsize,
    query_inference_calls: AtomicUsize,
}

impl SmallCeilingEmbedder {
    fn new() -> Self {
        Self {
            document_inference_calls: AtomicUsize::new(0),
            query_inference_calls: AtomicUsize::new(0),
        }
    }

    fn document_inference_calls(&self) -> usize {
        self.document_inference_calls.load(Ordering::SeqCst)
    }

    fn query_inference_calls(&self) -> usize {
        self.query_inference_calls.load(Ordering::SeqCst)
    }
}

impl Embedder for SmallCeilingEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("the checked event-search entrypoints must not call plain embedding")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: "small-ceiling-capability-test-space".to_string(),
            dimension: 1,
            metric: DistanceMetric::Cosine,
            model_family: "test".to_string(),
            model_name: "small-ceiling-capability".to_string(),
            model_version: "1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: Vec::new(),
        }
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(4)
    }

    fn prepare_document_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let payload: Vec<u32> = text.bytes().map(u32::from).collect();
        let boundaries = (0..text.len()).map(|byte| (byte, byte + 1)).collect();
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
        let payload: Vec<u32> = text.bytes().map(u32::from).collect();
        seal.query_builder()
            .token_count(payload.len())
            .payload(payload)
            .seal()
    }

    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        self.document_inference_calls.fetch_add(1, Ordering::SeqCst);
        Ok(EmbeddingOutput {
            embedding_space_id: self.encoding_profile_id(),
            dimension: 1,
            producer: producer(),
            vector: vec![inference.token_count() as f32],
        })
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        self.query_inference_calls.fetch_add(1, Ordering::SeqCst);
        Ok(EmbeddingOutput {
            embedding_space_id: self.encoding_profile_id(),
            dimension: 1,
            producer: producer(),
            vector: vec![inference.token_count() as f32],
        })
    }
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "test".to_string(),
        model_name: "small-ceiling-capability".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

#[test]
fn small_declared_event_search_ceiling_is_enforced_before_inference() {
    let embedder = SmallCeilingEmbedder::new();
    let checked: &dyn Embedder = &embedder;

    let document = checked
        .prepare_document("document input")
        .expect("the double deliberately implements document preparation");
    let query = checked
        .prepare_query("query input")
        .expect("the double deliberately implements query preparation");
    assert_eq!(
        checked
            .event_search_capability()
            .expect("the fixture declares event-search support")
            .token_limit(),
        4,
        "the fixture needs the deliberately small ceiling for this proof to mean anything"
    );

    let document_result = checked.embed_prepared_document(&document);
    let query_result = checked.embed_prepared_query(&query);

    assert!(
        matches!(
            document_result,
            Err(context_graph::CgError::PreparedInputExceedsModelLimit {
                token_count: 14,
                limit: 4,
            })
        ) && matches!(
            query_result,
            Err(context_graph::CgError::PreparedInputExceedsModelLimit {
                token_count: 11,
                limit: 4,
            })
        ) && embedder.document_inference_calls() == 0
            && embedder.query_inference_calls() == 0,
        "a prepared value over the declared event-search ceiling must be refused before either \
         inference hook runs; document_result={document_result:?}, query_result={query_result:?}, \
         document_inference_calls={}, query_inference_calls={}",
        embedder.document_inference_calls(),
        embedder.query_inference_calls(),
    );
}
