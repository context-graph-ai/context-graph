//! Same-profile-only preparation must be enforced, not advisory: a value sealed under one
//! embedder's encoding profile must be refused by a DIFFERENT embedder's checked
//! `embed_prepared_document`/`embed_prepared_query`, even when that different embedder is
//! deliberately designed to accept anything. `Bypasser` below overrides every hook it can to
//! answer unconditionally for any prepared value's payload, regardless of the profile it was
//! actually sealed under — proving the refusal comes from the checked entrypoint itself, not from
//! an implementer choosing to behave.
//!
//! This file is called through `&dyn Embedder` throughout, deliberately: that is how every real
//! caller in this crate holds an embedder (`EmbedderRegistration.embedder: Arc<dyn Embedder>`), so
//! this is the exact boundary the enforcement must hold at.

use context_graph::embedder::{
    Embedder, EventSearchCapability, EventSearchInference, EventSearchSeal,
    PreparedEventSearchDocument, PreparedEventSearchQuery,
};
use context_graph::{
    CgError, DistanceMetric, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EventSearchEmbedder,
    EvidenceKind, EvidenceProducer, Result,
};

fn embedding_space_named(id: &str) -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: id.to_string(),
        dimension: 1,
        metric: DistanceMetric::Cosine,
        model_family: "test".to_string(),
        model_name: "test".to_string(),
        model_version: "1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: None,
        bindings: Vec::new(),
    }
}

fn bypass_output(profile: &str, payload: &[u32]) -> EmbeddingOutput {
    EmbeddingOutput {
        embedding_space_id: profile.to_string(),
        dimension: 1,
        producer: EvidenceProducer {
            system: "test".to_string(),
            model_name: "bypasser".to_string(),
            model_version: "1".to_string(),
            pipeline_version: "1".to_string(),
        },
        vector: payload.iter().map(|id| *id as f32).collect(),
    }
}

/// A real, distinct `Embedder` used only to legitimately seal values under ITS OWN declared
/// profile (`"sealer_profile"`) — never `Bypasser`'s.
struct SealingEmbedder;

impl Embedder for SealingEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unimplemented!("this double only seals prepared values, it never plain-embeds")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        embedding_space_named("sealer_profile")
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
}

/// Declares a DIFFERENT profile (`"bypasser_profile"`) than `SealingEmbedder`, and overrides every
/// hook it can to answer unconditionally for any prepared value's payload, regardless of the
/// profile it was actually sealed under — the maximally adversarial implementer.
struct Bypasser;

impl Embedder for Bypasser {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unimplemented!("this double never legitimately plain-embeds")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        embedding_space_named("bypasser_profile")
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(64)
    }

    // `embed_prepared_document_hook`/`embed_prepared_query_hook` are the only extension points
    // left: overridden here to answer unconditionally for any prepared value's payload, exactly
    // the maximally adversarial implementation. `embed_prepared_document`/`embed_prepared_query`
    // themselves are not `Embedder` trait methods at all — there is nothing to override that
    // could skip the checked entrypoint's own profile verification, so these hooks are never
    // reached for a mismatched profile.
    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(bypass_output("bypasser_profile", inference.token_ids()))
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        Ok(bypass_output("bypasser_profile", inference.token_ids()))
    }
}

#[test]
fn prepared_document_sealed_under_a_different_profile_is_refused_even_by_an_embedder_designed_to_bypass_the_check()
 {
    let sealer: &dyn Embedder = &SealingEmbedder;
    let bypasser: &dyn Embedder = &Bypasser;

    let prepared_document = sealer
        .prepare_document("hello event")
        .expect("the sealing embedder's own preparation must succeed");
    let result = bypasser.embed_prepared_document(&prepared_document);
    assert!(
        matches!(
            &result,
            Err(CgError::EventSearchPreparationProfileMismatch {
                expected_profile,
                actual_profile,
            }) if expected_profile == "bypasser_profile" && actual_profile == "sealer_profile"
        ),
        "a value sealed under a different embedder's profile must be refused with the typed \
         profile-mismatch error even by an embedder deliberately designed to accept anything, \
         got {result:?}"
    );
}

#[test]
fn prepared_query_sealed_under_a_different_profile_is_refused_even_by_an_embedder_designed_to_bypass_the_check()
 {
    let sealer: &dyn Embedder = &SealingEmbedder;
    let bypasser: &dyn Embedder = &Bypasser;

    let prepared_query = sealer
        .prepare_query("hello query")
        .expect("the sealing embedder's own preparation must succeed");
    let result = bypasser.embed_prepared_query(&prepared_query);
    assert!(
        matches!(
            &result,
            Err(CgError::EventSearchPreparationProfileMismatch {
                expected_profile,
                actual_profile,
            }) if expected_profile == "bypasser_profile" && actual_profile == "sealer_profile"
        ),
        "a value sealed under a different embedder's profile must be refused with the typed \
         profile-mismatch error even by an embedder deliberately designed to accept anything, \
         got {result:?}"
    );
}
