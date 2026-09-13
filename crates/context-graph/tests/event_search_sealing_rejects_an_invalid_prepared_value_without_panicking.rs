//! A third-party `Embedder` implementation's preparation hook is the one place outside this
//! crate that can build a sealed `PreparedEventSearchDocument`/`PreparedEventSearchQuery`, through
//! `EventSearchSeal::document_builder`/`query_builder` (there is no other way to construct one:
//! `EventSearchSeal` itself has no public constructor). A hook that forgets a required field, or
//! supplies a `token_count` that does not match `payload`'s actual length, is an implementer
//! mistake — and the promise this proves is that a mistake like that comes back through the
//! checked `prepare_document`/`prepare_query` entrypoints on `&dyn Embedder` as a typed
//! `CgError::EventSearchPreparedValueInvalid`, never a panic that takes the whole embedding
//! pipeline down with it. Every case below is wrapped in `catch_unwind` so a regression that
//! reintroduces a panic fails loudly and distinctly from a regression that merely stops returning
//! the typed error.

use std::panic::{self, AssertUnwindSafe};

use context_graph::embedder::{
    Embedder, EventSearchCapability, EventSearchSeal, PreparedEventSearchDocument,
    PreparedEventSearchQuery,
};
use context_graph::{
    CgError, DistanceMetric, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EventSearchEmbedder,
    EvidenceKind, Result,
};

/// Which structurally-invalid sealed value this double's hook builds when asked to prepare.
#[derive(Clone, Copy)]
enum SealCase {
    OmittedTokenCount,
    OmittedPayload,
    OmittedBoundaries,
    InconsistentCount,
}

struct SealCaseEmbedder(SealCase);

impl Embedder for SealCaseEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unimplemented!("this double only exercises sealing validation")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: "seal_case_test_double".to_string(),
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
        match self.0 {
            SealCase::OmittedTokenCount => seal.document_builder().payload(vec![1, 2, 3]).seal(),
            SealCase::OmittedPayload => seal.document_builder().token_count(3).seal(),
            SealCase::OmittedBoundaries => seal
                .document_builder()
                .token_count(3)
                .payload(vec![1, 2, 3])
                .seal(),
            SealCase::InconsistentCount => seal
                .document_builder()
                .token_count(9999)
                .byte_boundaries(vec![(0, 1), (1, 2), (2, 3)])
                .payload(vec![1, 2, 3])
                .seal(),
        }
    }

    fn prepare_query_hook(
        &self,
        _text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        match self.0 {
            SealCase::OmittedTokenCount => seal.query_builder().payload(vec![1, 2, 3]).seal(),
            SealCase::OmittedPayload => seal.query_builder().token_count(3).seal(),
            // The query role has no boundaries field at all; reuse "omitted boundaries" as
            // another inconsistent-count probe so every query case still exercises a distinct
            // invalid shape.
            SealCase::OmittedBoundaries => seal
                .query_builder()
                .token_count(3)
                .payload(vec![1, 2, 3])
                .seal(),
            SealCase::InconsistentCount => seal
                .query_builder()
                .token_count(9999)
                .payload(vec![1, 2, 3])
                .seal(),
        }
    }
}

fn assert_no_panic_and_typed_error(label: &str, case: SealCase, query_role: bool) {
    let embedder = SealCaseEmbedder(case);
    let embedder: &dyn Embedder = &embedder;
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
        if query_role {
            embedder.prepare_query("probe text").map(|_| ())
        } else {
            embedder.prepare_document("probe text").map(|_| ())
        }
    }));
    let result = match outcome {
        Ok(result) => result,
        Err(_) => panic!("{label} must never panic the host on an invalid sealed value"),
    };
    assert!(
        matches!(
            &result,
            Err(CgError::EventSearchPreparedValueInvalid { .. })
        ),
        "{label} must return the typed validation error for an invalid sealed value, got {result:?}"
    );
}

#[test]
fn sealing_a_document_with_omitted_token_count_is_a_typed_error_not_a_panic() {
    assert_no_panic_and_typed_error(
        "sealing a document with an omitted token_count",
        SealCase::OmittedTokenCount,
        false,
    );
}

#[test]
fn sealing_a_document_with_omitted_payload_is_a_typed_error_not_a_panic() {
    assert_no_panic_and_typed_error(
        "sealing a document with an omitted payload",
        SealCase::OmittedPayload,
        false,
    );
}

#[test]
fn sealing_a_document_with_omitted_boundaries_is_a_typed_error_not_a_panic() {
    assert_no_panic_and_typed_error(
        "sealing a document with omitted byte_boundaries",
        SealCase::OmittedBoundaries,
        false,
    );
}

#[test]
fn sealing_a_document_with_a_token_count_inconsistent_with_payload_length_is_a_typed_error_not_a_panic()
 {
    assert_no_panic_and_typed_error(
        "sealing a document whose token_count does not match payload's length",
        SealCase::InconsistentCount,
        false,
    );
}

#[test]
fn sealing_a_query_with_omitted_token_count_is_a_typed_error_not_a_panic() {
    assert_no_panic_and_typed_error(
        "sealing a query with an omitted token_count",
        SealCase::OmittedTokenCount,
        true,
    );
}

#[test]
fn sealing_a_query_with_omitted_payload_is_a_typed_error_not_a_panic() {
    assert_no_panic_and_typed_error(
        "sealing a query with an omitted payload",
        SealCase::OmittedPayload,
        true,
    );
}

#[test]
fn sealing_a_query_with_a_token_count_inconsistent_with_payload_length_is_a_typed_error_not_a_panic()
 {
    assert_no_panic_and_typed_error(
        "sealing a query whose token_count does not match payload's length",
        SealCase::InconsistentCount,
        true,
    );
}
