//! The canonical event-history identity belongs only to this library's exact Nomic runtime types.
//! This external integration test uses a `repr(C)` wrapper whose first field is a genuine Nomic
//! embedder, declares the reserved profile and space, and implements every role-aware hook.
//! Preparation, store registration, and inference must each refuse the outside wrapper before
//! those hooks can run.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
use context_graph::embedder::{
    Embedder, EmbedderConfig, EventSearchCapability, EventSearchInference, EventSearchSeal,
    PreparedEventSearchDocument, PreparedEventSearchQuery,
};
use context_graph::embedding_space::{BuiltinSpaceRegistry, event_history_embedding_space};
use context_graph::{
    CgError, EmbedderRegistration, EmbeddingInput, EmbeddingOutput, EventSearchEmbedder,
    EvidenceKind, EvidenceProducer, Result, Store, StoreConfig,
};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

fn build_model() -> Arc<NomicTextEmbedder> {
    Arc::new(NomicTextEmbedder::new(
        nomic_real_model::local_offline_config(nomic_real_model::canonical_model_dir()),
    ))
}

#[test]
fn document_prepared_by_the_raw_embedder_is_accepted_by_the_wrapper_and_produces_the_same_vector() {
    let model = build_model();
    let raw: &dyn Embedder = model.as_ref();
    let wrapper =
        EventHistoryTextEmbedder::over(Arc::clone(&model)).expect("registration must succeed");
    let wrapped: &dyn Embedder = &wrapper;

    let text = "the incident review closed with three follow-up actions";
    let prepared_by_raw = raw
        .prepare_document(text)
        .expect("the raw embedder's role-aware preparation must succeed");
    let actual = wrapped
        .embed_prepared_document(&prepared_by_raw)
        .expect("the wrapper must accept the raw embedder's preparation");
    let expected = wrapped
        .embed_prepared_document(&wrapped.prepare_document(text).expect("wrapper preparation"))
        .expect("the wrapper embedding its own preparation must succeed");

    assert_eq!(
        actual.vector, expected.vector,
        "raw preparation embedded through the wrapper must produce the wrapper's exact vector"
    );
}

#[test]
fn document_prepared_by_the_wrapper_is_accepted_by_the_raw_embedders_embed_prepared_document() {
    let model = build_model();
    let raw: &dyn Embedder = model.as_ref();
    let wrapper =
        EventHistoryTextEmbedder::over(Arc::clone(&model)).expect("registration must succeed");
    let wrapped: &dyn Embedder = &wrapper;

    let text = "find the incident review from last quarter";
    let prepared_by_wrapper = wrapped
        .prepare_document(text)
        .expect("the wrapper's role-aware preparation must succeed");
    let actual = raw
        .embed_prepared_document(&prepared_by_wrapper)
        .expect("the raw embedder must accept the wrapper's preparation");
    let expected = raw
        .embed_prepared_document(&raw.prepare_document(text).expect("raw preparation"))
        .expect("the raw embedder embedding its own preparation must succeed");

    assert_eq!(
        actual.vector, expected.vector,
        "wrapper preparation embedded through raw Nomic must produce raw Nomic's exact vector"
    );
}

/// Models a third party whose first field is a genuine canonical Nomic embedder. The atomic
/// counters prove the checked public surfaces use runtime type identity and reject before invoking
/// either of this unrelated implementation's hooks.
#[repr(C)]
struct OutsideNomicWrapper {
    canonical: NomicTextEmbedder,
    preparation_calls: AtomicUsize,
    inference_calls: AtomicUsize,
}

impl OutsideNomicWrapper {
    fn new(canonical: NomicTextEmbedder) -> Self {
        Self {
            canonical,
            preparation_calls: AtomicUsize::new(0),
            inference_calls: AtomicUsize::new(0),
        }
    }

    fn preparation_calls(&self) -> usize {
        self.preparation_calls.load(Ordering::SeqCst)
    }

    fn inference_calls(&self) -> usize {
        self.inference_calls.load(Ordering::SeqCst)
    }
}

impl Embedder for OutsideNomicWrapper {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("the role-aware proof must never call plain embedding")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> context_graph::EmbeddingSpace {
        event_history_embedding_space()
    }

    fn encoding_profile_id(&self) -> String {
        BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID.to_string()
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        let _genuine_inner_model = &self.canonical;
        EventSearchCapability::new(8192)
    }

    fn prepare_document_hook(
        &self,
        _text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchDocument> {
        self.preparation_calls.fetch_add(1, Ordering::SeqCst);
        seal.document_builder()
            .token_count(1)
            .byte_boundaries(vec![(0, 1)])
            .payload(vec![7])
            .seal()
    }

    fn prepare_query_hook(
        &self,
        _text: &str,
        seal: &EventSearchSeal,
    ) -> Result<PreparedEventSearchQuery> {
        self.preparation_calls.fetch_add(1, Ordering::SeqCst);
        seal.query_builder().token_count(1).payload(vec![7]).seal()
    }

    fn embed_prepared_document_hook(
        &self,
        _inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        self.inference_calls.fetch_add(1, Ordering::SeqCst);
        Ok(EmbeddingOutput {
            embedding_space_id: self.encoding_profile_id(),
            dimension: 768,
            producer: producer(),
            vector: vec![0.0; 768],
        })
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        self.embed_prepared_document_hook(inference)
    }
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "test".to_string(),
        model_name: "outside-nomic-wrapper".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

#[test]
fn external_embedder_cannot_borrow_canonical_event_history_authority() {
    let canonical = NomicTextEmbedder::new(nomic_real_model::local_offline_config(
        nomic_real_model::canonical_model_dir(),
    ));
    let impersonator = Arc::new(OutsideNomicWrapper::new(NomicTextEmbedder::new(
        nomic_real_model::local_offline_config(nomic_real_model::canonical_model_dir()),
    )));
    let checked_impersonator: &dyn Embedder = impersonator.as_ref();

    let preparation = checked_impersonator.prepare_document("relay attempt");

    let directory = tempfile::tempdir().expect("registration test directory");
    let registration = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: directory.path().join("relayed-authority.db"),
            extra_embedding_spaces: vec![event_history_embedding_space()],
            default_text_embedder: Some(EmbedderConfig::disabled()),
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: event_history_embedding_space(),
            embedder: Arc::clone(&impersonator) as Arc<dyn Embedder>,
        }],
    );

    let checked_canonical: &dyn Embedder = &canonical;
    let canonical_prepared = checked_canonical
        .prepare_document("a genuine canonical event-history document")
        .expect("the installed canonical model must prepare its own document offline");
    let inference = checked_impersonator.embed_prepared_document(&canonical_prepared);

    let expected_profile = BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID;
    let preparation_refused = matches!(
        &preparation,
        Err(CgError::EventHistoryProfileIdentityUnverified { declared_profile })
            if declared_profile == expected_profile
    );
    let registration_refused = matches!(
        &registration,
        Err(CgError::EventHistoryProfileIdentityUnverified { declared_profile })
            if declared_profile == expected_profile
    );
    let inference_refused = matches!(
        &inference,
        Err(CgError::EventHistoryProfileIdentityUnverified { declared_profile })
            if declared_profile == expected_profile
    );

    assert!(
        preparation_refused
            && registration_refused
            && inference_refused
            && impersonator.preparation_calls() == 0
            && impersonator.inference_calls() == 0,
        "a third-party embedder must not borrow canonical event-history authority by placing a \
         real Nomic embedder first in its layout: preparation, registration, and inference must each refuse \
         before its hooks run; preparation={preparation:?}, registration={registration:?}, \
         inference={inference:?}, preparation_calls={}, inference_calls={}",
        impersonator.preparation_calls(),
        impersonator.inference_calls(),
    );
}
