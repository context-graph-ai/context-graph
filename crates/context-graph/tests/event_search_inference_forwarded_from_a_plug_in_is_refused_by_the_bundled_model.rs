//! An `EventSearchInference` an external plug-in receives from its OWN checked entrypoint carries
//! nothing today that names which embedder's profile authorized it. So a plug-in embedder that
//! prepares under its own encoding profile, then forwards the inference it is handed straight into
//! the bundled Nomic embedder's implementer-facing hook (`embed_prepared_document_hook`/
//! `embed_prepared_query_hook`) — bypassing the checked, library-owned `embed_prepared_document`/
//! `embed_prepared_query` entrypoints on the Nomic side entirely — gets back a real vector stamped
//! with the reserved event-history identity, from tokens the canonical Nomic preparer never
//! produced and never authorized.
//!
//! Every case below must be refused before Nomic's forward pass produces anything: a plug-in
//! forwarding its own document inference into Nomic's document hook, its own query inference into
//! Nomic's query hook, and — the role-mismatch case — its own document inference into Nomic's
//! *query* hook. An inference that was never authorized for the bundled hooks' own profile must
//! never be accepted and stamped with the reserved `event_history_nomic_v1_5_document_query_v1_768`
//! identity.

use std::sync::Arc;

use context_graph::embedder::nomic::NomicTextEmbedder;
use context_graph::embedder::{
    Embedder, EventSearchCapability, EventSearchInference, EventSearchSeal,
    PreparedEventSearchDocument, PreparedEventSearchQuery,
};
use context_graph::embedding_space::BuiltinSpaceRegistry;
use context_graph::{
    CgError, DistanceMetric, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EventSearchEmbedder,
    EvidenceKind, Result,
};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const PLUGIN_PROFILE_ID: &str = "external_plugin_over_canonical_nomic_v1";

fn build_canonical_nomic() -> Arc<NomicTextEmbedder> {
    let fixture = nomic_real_model::canonical_model_dir();
    assert!(
        fixture.join("model.safetensors").is_file(),
        "the canonical Nomic fixture must already be installed at {} for this proof to run \
         offline; a missing cache is BLOCKED, not a reason to download",
        fixture.display()
    );
    Arc::new(NomicTextEmbedder::new(
        nomic_real_model::local_offline_config(fixture),
    ))
}

/// A test-support double modelling an external plug-in embedder: its own encoding profile, its
/// own `event_search_capability`, and its own byte-arithmetic preparation — never Nomic's
/// tokenizer. It holds the real canonical `Arc<NomicTextEmbedder>` and, inside its own
/// implementer-facing hooks, forwards the `&EventSearchInference` it was handed straight to one of
/// Nomic's raw hooks, calling the hook function directly rather than going through Nomic's checked
/// `embed_prepared_document`/`embed_prepared_query` entrypoints.
struct ForwardingPluginEmbedder {
    nomic: Arc<NomicTextEmbedder>,
    /// When set, `embed_prepared_document_hook` forwards into Nomic's *query* hook and
    /// `embed_prepared_query_hook` forwards into Nomic's *document* hook — the role-mismatch case.
    swap_role: bool,
}

impl ForwardingPluginEmbedder {
    fn new(nomic: Arc<NomicTextEmbedder>, swap_role: bool) -> Self {
        Self { nomic, swap_role }
    }
}

impl Embedder for ForwardingPluginEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        unreachable!("this double only exercises event-search preparation forwarding")
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: PLUGIN_PROFILE_ID.to_string(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "external-plugin".to_string(),
            model_name: "external-plugin".to_string(),
            model_version: "1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: Vec::new(),
        }
    }

    fn event_search_capability(&self) -> Option<EventSearchCapability> {
        EventSearchCapability::new(8192)
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

    // Deliberately forwards the received inference to the bundled Nomic model's own raw hook,
    // calling it directly rather than through Nomic's checked `embed_prepared_document`/
    // `embed_prepared_query` entrypoints — that direct call is the boundary this proof targets.
    fn embed_prepared_document_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        if self.swap_role {
            self.nomic.embed_prepared_query_hook(inference)
        } else {
            self.nomic.embed_prepared_document_hook(inference)
        }
    }

    fn embed_prepared_query_hook(
        &self,
        inference: &EventSearchInference,
    ) -> Result<EmbeddingOutput> {
        if self.swap_role {
            self.nomic.embed_prepared_document_hook(inference)
        } else {
            self.nomic.embed_prepared_query_hook(inference)
        }
    }
}

fn assert_refused_for_nomics_reserved_profile(result: &Result<EmbeddingOutput>, context: &str) {
    let reserved = BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID;
    assert!(
        matches!(
            result,
            Err(CgError::EventSearchPreparationProfileMismatch {
                expected_profile,
                actual_profile,
            }) if expected_profile == reserved && actual_profile == PLUGIN_PROFILE_ID
        ),
        "{context}: an inference authorized only for the plug-in's own profile ({PLUGIN_PROFILE_ID}) \
         must be refused by the bundled Nomic hook with EventSearchPreparationProfileMismatch \
         naming the reserved profile as expected and the plug-in's profile as actual, and must \
         never produce an EmbeddingOutput stamped with the reserved identity ({reserved}); got \
         {result:?}"
    );
    if let Ok(output) = result {
        assert_ne!(
            output.embedding_space_id, reserved,
            "{context}: a forwarded, unauthorized inference must never be stamped with the \
             reserved event-history identity"
        );
    }
}

#[test]
fn forwarded_document_inference_from_a_plug_in_is_refused_by_the_bundled_nomic_document_hook() {
    let nomic = build_canonical_nomic();
    let plugin = ForwardingPluginEmbedder::new(Arc::clone(&nomic), false);
    let plugin: &dyn Embedder = &plugin;

    let prepared = plugin
        .prepare_document("the incident review closed with three follow-up actions")
        .expect("the plug-in's own role-aware preparation must succeed under its own profile");

    let result = plugin.embed_prepared_document(&prepared);
    assert_refused_for_nomics_reserved_profile(&result, "document forwarded into document hook");
}

#[test]
fn forwarded_query_inference_from_a_plug_in_is_refused_by_the_bundled_nomic_query_hook() {
    let nomic = build_canonical_nomic();
    let plugin = ForwardingPluginEmbedder::new(Arc::clone(&nomic), false);
    let plugin: &dyn Embedder = &plugin;

    let prepared = plugin
        .prepare_query("find the incident review from last quarter")
        .expect("the plug-in's own role-aware preparation must succeed under its own profile");

    let result = plugin.embed_prepared_query(&prepared);
    assert_refused_for_nomics_reserved_profile(&result, "query forwarded into query hook");
}

#[test]
fn forwarded_document_inference_from_a_plug_in_is_refused_even_when_relayed_into_the_bundled_nomic_query_hook()
 {
    let nomic = build_canonical_nomic();
    let plugin = ForwardingPluginEmbedder::new(Arc::clone(&nomic), true);
    let plugin: &dyn Embedder = &plugin;

    let prepared = plugin
        .prepare_document("a role-mismatched document forwarded into the query hook")
        .expect("the plug-in's own role-aware preparation must succeed under its own profile");

    // The plug-in's `embed_prepared_document_hook` relays this document inference into Nomic's
    // *query* hook: a role mismatch layered on top of the profile mismatch. Both must be refused
    // before Nomic's forward pass runs.
    let result = plugin.embed_prepared_document(&prepared);
    assert_refused_for_nomics_reserved_profile(
        &result,
        "document inference relayed into the query hook (role mismatch)",
    );
}
