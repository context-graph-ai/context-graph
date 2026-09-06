//! The reserved event-history encoding identity
//! (`event_history_nomic_v1_5_document_query_v1_768`) must come from the canonical, verified
//! Nomic weights — never merely from an embedder whose concrete type is the library's own
//! `NomicTextEmbedder` and whose loaded dimension happens to match. `NomicTextEmbedder::over`
//! (the `EventHistoryTextEmbedder` wrapper's constructor) already verifies the installed
//! `REVISION` marker before granting that identity. Raw `NomicTextEmbedder::prepare_document` /
//! `prepare_query` do not: `dyn Embedder::verify_event_history_authority`
//! (`src/embedder.rs:741-773`) checks only the concrete type and the dimension, and
//! `NomicTextEmbedder::load_model` (`src/embedder/nomic.rs`) skips the revision check entirely
//! when `EmbedderConfig::pinned_revision` is `None` — exactly the shape of a `LocalPath` config
//! with no pin.
//!
//! This test builds a raw `NomicTextEmbedder` from a `LocalPath` config, with no revision pin,
//! over a copy of the canonical installed fixture whose `REVISION` marker has been rewritten to a
//! noncanonical value at the same dimension (768). The required shape is: refuse with
//! `EventSearchPreparationUnsupported`, never mint the reserved identity from dimension and
//! concrete type alone. Today, raw preparation still succeeds and claims the reserved profile, so
//! this test fails until preparation shares the same canonical admission
//! `EventHistoryTextEmbedder::over` already enforces.

use context_graph::embedder::nomic::NomicTextEmbedder;
use context_graph::embedding_space::BuiltinSpaceRegistry;
use context_graph::{CgError, Embedder, EmbedderConfig, EventSearchEmbedder, ModelSource};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
const NONCANONICAL_REVISION: &str = "noncanonical-test-revision-same-dimension";

fn canonical_fixture_dir() -> std::path::PathBuf {
    nomic_real_model::canonical_model_dir()
}

/// A local copy of the canonical fixture with its `REVISION` marker rewritten to a noncanonical
/// value; every other file (model weights, tokenizer, config) is untouched, so the dimension
/// (768) and the underlying weights' shape are exactly canonical — only the revision identity
/// differs, which is the one thing that makes weights "verified canonical" rather than merely
/// "same-shaped."
fn noncanonical_same_dimension_fixture(destination: &std::path::Path) {
    let source = canonical_fixture_dir();
    assert!(
        source.join("model.safetensors").is_file(),
        "the canonical Nomic fixture must already be installed at {} for this proof to run \
         offline; a missing cache is BLOCKED, not a reason to download",
        source.display()
    );
    std::fs::create_dir_all(destination).expect("create destination model directory");
    for file in ["model.safetensors", "tokenizer.json", "config.json"] {
        std::fs::copy(source.join(file), destination.join(file))
            .unwrap_or_else(|err| panic!("copy canonical fixture file {file}: {err}"));
    }
    std::fs::write(destination.join("REVISION"), NONCANONICAL_REVISION)
        .expect("write noncanonical REVISION marker");
}

#[test]
fn raw_nomic_preparation_over_noncanonical_same_dimension_weights_refuses_the_reserved_identity() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join(NOMIC_MODEL_ID);
    noncanonical_same_dimension_fixture(&model_dir);

    // A LocalPath config with no revision pin: `load_model` skips `check_revision_marker`
    // entirely in this shape, so nothing today stops these noncanonical weights from loading.
    let config = EmbedderConfig {
        source: ModelSource::LocalPath(model_dir),
        offline: true,
        cache_dir: None,
        pinned_revision: None,
        network_policy: "deny-download".to_string(),
    };
    let embedder = NomicTextEmbedder::new(config);
    let embedder: &dyn Embedder = &embedder;

    let document_result = embedder.prepare_document("the incident review closed today");
    assert!(
        matches!(
            document_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "raw preparation over noncanonical same-dimension weights must refuse with \
         EventSearchPreparationUnsupported rather than mint the reserved event-history identity \
         from dimension and concrete type alone; got {document_result:?}"
    );

    let query_result = embedder.prepare_query("find the incident review");
    assert!(
        matches!(
            query_result,
            Err(CgError::EventSearchPreparationUnsupported { .. })
        ),
        "raw preparation over noncanonical same-dimension weights must refuse with \
         EventSearchPreparationUnsupported for the query role too; got {query_result:?}"
    );

    // Never a stamped, unlabeled claim to the reserved identity: even the plain embedding-space
    // declaration this embedder answers with must not be the reserved event-history id.
    assert_ne!(
        embedder.embedding_space().embedding_space_id,
        BuiltinSpaceRegistry::EVENT_HISTORY_DOCUMENT_QUERY_ID,
        "an embedder over unverified, noncanonical weights must never declare the reserved \
         event-history space as its own identity"
    );
}
