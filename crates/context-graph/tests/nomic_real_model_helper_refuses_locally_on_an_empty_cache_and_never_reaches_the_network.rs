//! Proof that the real-model test helper's local, offline, download-denying construction
//! actually refuses locally when the cache is empty, rather than silently reaching the network.
//!
//! `ModelSource::LocalPath` pointed at an isolated, empty temp directory can never take the
//! Hugging-Face download branch: `NomicTextEmbedder::ensure_cache_ready` checks
//! `!matches!(source, ModelSource::HuggingFace { .. })` and returns the typed
//! `EmbedderModelNotAvailable` refusal before any code path that could construct an HTTP client
//! (`download_hf_file`/`ureq::get`) is even reached — so no network client is constructed, and the
//! refusal is observed here as the typed error, not inferred from elapsed time or a log line.

use context_graph::{CgError, Embedder, EventSearchEmbedder};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

#[test]
fn embedding_over_an_isolated_empty_local_cache_refuses_locally_with_a_typed_error_and_never_reaches_the_network()
 {
    let empty_cache = tempfile::tempdir().expect("isolated empty temp dir");
    let empty_model_dir = empty_cache.path().join(nomic_real_model::NOMIC_MODEL_ID);
    let embedder = context_graph::embedder::nomic::NomicTextEmbedder::new(
        nomic_real_model::local_offline_config(empty_model_dir.clone()),
    );
    let embedder: &dyn Embedder = &embedder;

    // Plain embedding: the model must be loaded, and loading over an empty local directory is
    // refused locally.
    let plain_result = embedder.embed(context_graph::EmbeddingInput::Text(
        "probe text".to_string(),
    ));
    assert!(
        matches!(
            &plain_result,
            Err(CgError::EmbedderModelNotAvailable { source, path })
                if source.starts_with("LocalPath(") && path == &empty_model_dir.display().to_string()
        ),
        "loading a local-source embedder over an empty, isolated cache directory must refuse \
         locally with the typed EmbedderModelNotAvailable error naming the LocalPath source and \
         the exact missing directory, never attempt a Hugging Face download; got {plain_result:?}"
    );

    // Event-search preparation tokenizes through the same `load_model` path (it needs the
    // tokenizer to report exact token counts and byte boundaries), so the same local refusal must
    // hold there too — never a silent fallback that reaches the network.
    let prepare_result = embedder.prepare_document("probe text");
    assert!(
        matches!(
            &prepare_result,
            Err(CgError::EmbedderModelNotAvailable { source, path })
                if source.starts_with("LocalPath(") && path == &empty_model_dir.display().to_string()
        ),
        "preparing an event-search document over an empty, isolated cache directory must refuse \
         locally with the same typed EmbedderModelNotAvailable error, never reach the network; \
         got {prepare_result:?}"
    );
}
