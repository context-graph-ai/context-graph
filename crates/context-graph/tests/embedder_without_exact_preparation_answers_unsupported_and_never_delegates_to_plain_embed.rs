//! An embedder that does not implement exact role-aware event-search preparation answers
//! `EventSearchPreparationUnsupported` from all four event-search entrypoints and never silently
//! falls back to `Embedder::embed` on plain text: `embed`'s call count stays 0. A host that plugs
//! in a half-written embedder gets a typed refusal, never an event-history vector produced from an
//! unlabelled plain embedding.
//!
//! There are two distinct ways an embedder can be "using the defaults", and each is answered by a
//! different piece of the library, so both are proved here:
//!
//! - An embedder that declares NO event-search capability is answered by the capability gate, which
//!   runs first in every one of the four checked entrypoints, before any profile verification and
//!   before any hook. The first two tests below cover that embedder for `prepare_*` and for
//!   `embed_prepared_*`.
//! - An embedder that DOES declare a capability but leaves the four `Embedder` hooks at their trait
//!   defaults passes the capability gate, so for it the hook defaults are the only thing standing
//!   between the checked entrypoint and a silent plain-embed fallback. The third test covers that
//!   embedder, and is the only place in the estate where those four defaults actually execute.
//!
//! The `embed_prepared_*` arms need real sealed values, which only an embedder that implements the
//! preparation hooks can mint. A separate, test-only sealer produces them under the profile of the
//! embedder under test, so profile verification passes and the refusal comes from the embedder
//! under test rather than from a profile mismatch.

use std::sync::atomic::{AtomicUsize, Ordering};

use context_graph::embedder::mock::MockEmbedder;
use context_graph::embedder::{
    EventSearchCapability, EventSearchSeal, PreparedEventSearchDocument, PreparedEventSearchQuery,
};
use context_graph::{
    CgError, Embedder, EmbeddingInput, EmbeddingOutput, EventSearchEmbedder, EvidenceKind, Result,
};

/// Counts `embed` calls on an inner embedder while inheriting every other method's default,
/// including the four event-search preparation methods under test. Test support only: it
/// forwards to `MockEmbedder`, adding nothing but the counter.
struct CountingEmbedder {
    inner: MockEmbedder,
    embed_calls: AtomicUsize,
}

impl CountingEmbedder {
    fn new() -> Self {
        Self {
            inner: MockEmbedder::new("text_default_nomic_v1_5_768", 768),
            embed_calls: AtomicUsize::new(0),
        }
    }

    fn embed_call_count(&self) -> usize {
        self.embed_calls.load(Ordering::SeqCst)
    }
}

impl Embedder for CountingEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        self.embed_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.embed(input)
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        self.inner.supported_inputs()
    }

    fn embedding_space(&self) -> context_graph::EmbeddingSpace {
        self.inner.embedding_space()
    }

    // Deliberately no override of prepare_document / embed_prepared_document / prepare_query /
    // embed_prepared_query: the trait's own defaults are what this test proves.
}

#[test]
fn embedder_without_exact_preparation_answers_unsupported_and_never_delegates_to_plain_embed() {
    let embedder = CountingEmbedder::new();
    let checked: &dyn Embedder = &embedder;

    let document_result = checked.prepare_document("some event text");
    assert!(
        matches!(
            document_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "an embedder using the defaults must answer EventSearchPreparationUnsupported from \
         prepare_document, got {document_result:?}"
    );

    let query_result = checked.prepare_query("some query text");
    assert!(
        matches!(
            query_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "an embedder using the defaults must answer EventSearchPreparationUnsupported from \
         prepare_query, got {query_result:?}"
    );

    assert_eq!(
        embedder.embed_call_count(),
        0,
        "preparation failing must never silently delegate to plain-text embed"
    );
}

/// A test-only `Embedder` double whose sole job is producing real, sealed prepared values under
/// SOMEONE ELSE's encoding profile — never its own. It implements the preparation hooks, but its
/// `encoding_profile_id` is overridden to equal the profile of the embedder under test
/// (`CountingEmbedder`'s declared space id), so the values it seals are accepted, by profile, when
/// handed to that other embedder's checked `embed_prepared_document`/`embed_prepared_query`. This
/// double never implements `embed_prepared_*_hook` itself — the point of this file is what the
/// UNSUPPORTED embedder does with these values, never this sealer's own inference.
struct CompatibleProfileSealer {
    target_profile: String,
}

impl CompatibleProfileSealer {
    fn new(target_profile: String) -> Self {
        Self { target_profile }
    }
}

impl Embedder for CompatibleProfileSealer {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unimplemented!("this double only seals prepared values under another embedder's profile")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> context_graph::EmbeddingSpace {
        unimplemented!("this double is never registered as its own embedder")
    }

    fn encoding_profile_id(&self) -> String {
        self.target_profile.clone()
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

#[test]
fn embedder_without_exact_preparation_answers_unsupported_from_embed_prepared_too_and_never_delegates_to_plain_embed()
 {
    let embedder = CountingEmbedder::new();
    let sealer = CompatibleProfileSealer::new(embedder.embedding_space().embedding_space_id);
    assert_eq!(
        sealer.encoding_profile_id(),
        embedder.embedding_space().embedding_space_id,
        "the sealer must share the embedder-under-test's own profile so profile verification \
         passes and the answer comes from that embedder rather than from a profile mismatch"
    );
    let checked_sealer: &dyn Embedder = &sealer;
    let checked_embedder: &dyn Embedder = &embedder;

    let prepared_document = checked_sealer
        .prepare_document("some event text")
        .expect("the compatible-profile sealer's own preparation must succeed");
    let prepared_query = checked_sealer
        .prepare_query("some query text")
        .expect("the compatible-profile sealer's own preparation must succeed");

    let document_result = checked_embedder.embed_prepared_document(&prepared_document);
    assert!(
        matches!(
            document_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "an embedder using the defaults must answer EventSearchPreparationUnsupported from \
         embed_prepared_document even for a value prepared under its own matching profile by a \
         different embedder, got {document_result:?}"
    );

    let query_result = checked_embedder.embed_prepared_query(&prepared_query);
    assert!(
        matches!(
            query_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "an embedder using the defaults must answer EventSearchPreparationUnsupported from \
         embed_prepared_query even for a value prepared under its own matching profile by a \
         different embedder, got {query_result:?}"
    );

    assert_eq!(
        embedder.embed_call_count(),
        0,
        "embed_prepared_document/embed_prepared_query failing on an unsupported embedder must \
         never silently delegate to plain-text embed"
    );
}

/// A test-only `Embedder` that DECLARES the event-search capability and then leaves every one of
/// the four preparation hooks at its trait default — the half-written plug-in a programmatic host
/// can realistically ship. Its declared space is the plain text-default one, never the reserved
/// event-history identity, so nothing about it claims canonical authority. `embed` is counted, as
/// on the no-capability double above, because the contract under test is that it is never reached.
struct CapabilityWithoutPreparationEmbedder {
    inner: MockEmbedder,
    embed_calls: AtomicUsize,
}

impl CapabilityWithoutPreparationEmbedder {
    fn new() -> Self {
        Self {
            inner: MockEmbedder::new("text_default_nomic_v1_5_768", 768),
            embed_calls: AtomicUsize::new(0),
        }
    }

    fn embed_call_count(&self) -> usize {
        self.embed_calls.load(Ordering::SeqCst)
    }
}

impl Embedder for CapabilityWithoutPreparationEmbedder {
    fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
        self.embed_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.embed(input)
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        self.inner.supported_inputs()
    }

    fn embedding_space(&self) -> context_graph::EmbeddingSpace {
        self.inner.embedding_space()
    }

    /// The one and only override beyond the plain-embedding basics: a declared, finite ceiling.
    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(64)
    }

    // Deliberately no override of prepare_document_hook / prepare_query_hook /
    // embed_prepared_document_hook / embed_prepared_query_hook: those four trait defaults are what
    // this file's third test proves, and the capability above is what lets execution reach them.
}

#[test]
fn an_embedder_that_declares_the_capability_without_implementing_preparation_answers_unsupported_and_never_delegates_to_plain_embed()
 {
    let embedder = CapabilityWithoutPreparationEmbedder::new();
    assert!(
        embedder.event_search_capability().is_some(),
        "this double must declare a capability, so the capability gate passes and execution \
         actually reaches the four hook defaults under test"
    );
    let checked: &dyn Embedder = &embedder;

    let document_result = checked.prepare_document("some event text");
    assert!(
        matches!(
            document_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "an embedder that declares the event-search capability but leaves prepare_document_hook \
         at its default must answer EventSearchPreparationUnsupported, got {document_result:?}"
    );

    let query_result = checked.prepare_query("some query text");
    assert!(
        matches!(
            query_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "an embedder that declares the event-search capability but leaves prepare_query_hook at \
         its default must answer EventSearchPreparationUnsupported, got {query_result:?}"
    );

    let sealer = CompatibleProfileSealer::new(embedder.embedding_space().embedding_space_id);
    let checked_sealer: &dyn Embedder = &sealer;
    let prepared_document = checked_sealer
        .prepare_document("some event text")
        .expect("the compatible-profile sealer's own preparation must succeed");
    let prepared_query = checked_sealer
        .prepare_query("some query text")
        .expect("the compatible-profile sealer's own preparation must succeed");

    let embed_document_result = checked.embed_prepared_document(&prepared_document);
    assert!(
        matches!(
            embed_document_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "an embedder that declares the event-search capability but leaves \
         embed_prepared_document_hook at its default must answer \
         EventSearchPreparationUnsupported for a value sealed under its own matching profile, got \
         {embed_document_result:?}"
    );

    let embed_query_result = checked.embed_prepared_query(&prepared_query);
    assert!(
        matches!(
            embed_query_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "an embedder that declares the event-search capability but leaves \
         embed_prepared_query_hook at its default must answer EventSearchPreparationUnsupported \
         for a value sealed under its own matching profile, got {embed_query_result:?}"
    );

    assert_eq!(
        embedder.embed_call_count(),
        0,
        "a declared capability with unimplemented preparation hooks must never silently delegate \
         to plain-text embed: an event-history vector produced from an unlabelled plain embedding \
         is exactly what the refusal exists to prevent"
    );
}
