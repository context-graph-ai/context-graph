//! An `Embedder` implementation this crate does not own (test support only,
//! never Nomic) can produce a real `PreparedEventSearchDocument`/`PreparedEventSearchQuery`
//! purely through the implementer-facing hook (`prepare_document_hook`/`prepare_query_hook` plus
//! the library-created `EventSearchSeal`'s `document_builder`/`query_builder`). This file links
//! against `context_graph` exactly the way an external consumer would, so if the hook were not
//! public and reachable from `context_graph::embedder`, this file would not compile.
//!
//! A value sealed under one embedder's encoding profile is refused, with the
//! typed `CgError::EventSearchPreparationProfileMismatch`, by a different embedder's
//! `embed_prepared_document`/`embed_prepared_query` — even though the double's own
//! `embed_prepared_document_hook`/`embed_prepared_query_hook` never calls a verification helper,
//! because the library-owned `embed_prepared_document`/`embed_prepared_query` wrapper verifies
//! before the hook ever runs.
//!
//! These are guards on the sealing/verification mechanism itself, not Behavior tests against product
//! behavior: no embedder shipped by this crate (Nomic included) implements real preparation here,
//! only this file's own deterministic test doubles.

use context_graph::embedder::{
    EventSearchCapability, EventSearchInference, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    CgError, DistanceMetric, Embedder, EmbeddingInput, EmbeddingOutput, EmbeddingSpace,
    EventSearchEmbedder, EvidenceKind, EvidenceProducer, Result,
};

/// A deterministic, non-Nomic `Embedder` double: test support only. Its "preparation" and
/// "embedding" are trivial byte arithmetic, never a real model — it exists only to prove the
/// implementer-facing hook is usable by an `Embedder` this crate does not own.
struct DeterministicEventSearchEmbedder {
    profile_suffix: &'static str,
}

impl DeterministicEventSearchEmbedder {
    fn new(profile_suffix: &'static str) -> Self {
        Self { profile_suffix }
    }

    fn deterministic_vector(&self, ids: &[u32]) -> Vec<f32> {
        ids.iter().map(|id| *id as f32).collect()
    }

    fn producer() -> EvidenceProducer {
        EvidenceProducer {
            system: "test".to_string(),
            model_name: "deterministic-test-double".to_string(),
            model_version: "1".to_string(),
            pipeline_version: "1".to_string(),
        }
    }
}

impl Embedder for DeterministicEventSearchEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unimplemented!("this double only exercises event-search preparation")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: format!("deterministic_event_search_{}", self.profile_suffix),
            dimension: 1,
            metric: DistanceMetric::Cosine,
            model_family: "deterministic-test-double".to_string(),
            model_name: "deterministic-test-double".to_string(),
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
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        let ids: Vec<u32> = text.bytes().map(u32::from).collect();
        let boundaries = (0..text.len()).map(|index| (index, index + 1)).collect();
        seal.document_builder()
            .token_count(ids.len())
            .byte_boundaries(boundaries)
            .payload(ids)
            .seal()
    }

    // Deliberately does NOT call `verify_event_search_document_profile`: the point of this proof
    // is that the library-owned `embed_prepared_document` wrapper verifies before this hook ever
    // runs, so a hook that skips verification entirely is still safe.
    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(EmbeddingOutput {
            embedding_space_id: self.embedding_space().embedding_space_id,
            dimension: 1,
            producer: Self::producer(),
            vector: self.deterministic_vector(inference.token_ids()),
        })
    }

    fn prepare_query_hook(
        &self,
        text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        let ids: Vec<u32> = text.bytes().map(u32::from).collect();
        seal.query_builder()
            .token_count(ids.len())
            .payload(ids)
            .seal()
    }

    // Deliberately does NOT call `verify_event_search_query_profile`, mirroring
    // `embed_prepared_document_hook` above.
    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(EmbeddingOutput {
            embedding_space_id: self.embedding_space().embedding_space_id,
            dimension: 1,
            producer: Self::producer(),
            vector: self.deterministic_vector(inference.token_ids()),
        })
    }
}

#[test]
fn external_embedder_implements_preparation_through_the_hook_and_produces_a_prepared_value() {
    let embedder = DeterministicEventSearchEmbedder::new("a");
    let embedder: &dyn Embedder = &embedder;
    assert_eq!(embedder.event_search_slice_token_loss_bound("User: "), None);
    let prepared = embedder
        .prepare_document("hello event")
        .expect("preparation through the implementer-facing hook must succeed");
    assert_eq!(prepared.token_count(), "hello event".len());
    assert_eq!(
        prepared.encoding_profile_id(),
        "deterministic_event_search_a"
    );

    let output = embedder
        .embed_prepared_document(&prepared)
        .expect("embedding a value this embedder sealed itself must succeed");
    assert_eq!(output.embedding_space_id, "deterministic_event_search_a");
}

#[test]
fn prepared_document_sealed_by_one_profile_is_refused_by_a_different_embedders_embed_prepared_document()
 {
    let sealing_embedder = DeterministicEventSearchEmbedder::new("a");
    let other_embedder = DeterministicEventSearchEmbedder::new("b");
    let sealing_embedder: &dyn Embedder = &sealing_embedder;
    let other_embedder: &dyn Embedder = &other_embedder;

    let prepared_by_a = sealing_embedder
        .prepare_document("hello event")
        .expect("preparation must succeed");

    let result = other_embedder.embed_prepared_document(&prepared_by_a);
    assert!(
        matches!(
            &result,
            Err(CgError::EventSearchPreparationProfileMismatch {
                expected_profile,
                actual_profile,
            }) if expected_profile == "deterministic_event_search_b"
                && actual_profile == "deterministic_event_search_a"
        ),
        "a prepared document sealed under one embedder's encoding profile must be refused by a \
         different embedder's embed_prepared_document with the typed mismatch error, got \
         {result:?}"
    );
}

#[test]
fn prepared_query_sealed_by_one_profile_is_refused_by_a_different_embedders_embed_prepared_query() {
    let sealing_embedder = DeterministicEventSearchEmbedder::new("a");
    let other_embedder = DeterministicEventSearchEmbedder::new("b");
    let sealing_embedder: &dyn Embedder = &sealing_embedder;
    let other_embedder: &dyn Embedder = &other_embedder;

    let prepared_by_a = sealing_embedder
        .prepare_query("hello query")
        .expect("preparation must succeed");

    let result = other_embedder.embed_prepared_query(&prepared_by_a);
    assert!(
        matches!(
            &result,
            Err(CgError::EventSearchPreparationProfileMismatch {
                expected_profile,
                actual_profile,
            }) if expected_profile == "deterministic_event_search_b"
                && actual_profile == "deterministic_event_search_a"
        ),
        "a prepared query sealed under one embedder's encoding profile must be refused by a \
         different embedder's embed_prepared_query with the typed mismatch error, got {result:?}"
    );
}
