//! A caller choosing to store a passage for later retrieval and a caller searching for one must
//! never be able to reach the wrong role by accident. `EventHistoryTextEmbedder` offers the
//! explicit `prepare_document`/`embed_prepared_document` and `prepare_query`/`embed_prepared_query`
//! entrypoints through `EventSearchEmbedder` for exactly that reason. Plain `Embedder::embed(Text)`
//! on this wrapper names neither role, so it refuses rather than silently picking one.

use std::sync::Arc;

use context_graph::CgError;
use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
use context_graph::embedder::{Embedder, EmbeddingInput};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

#[test]
fn event_history_wrapper_refuses_plain_text_embedding_without_an_explicit_role() {
    let model = Arc::new(NomicTextEmbedder::new(
        nomic_real_model::local_offline_config(nomic_real_model::canonical_model_dir()),
    ));
    let wrapper =
        EventHistoryTextEmbedder::over(model).expect("registration over a fit model must succeed");
    let checked: &dyn Embedder = &wrapper;

    let result = checked.embed(EmbeddingInput::Text(
        "the incident review closed with three follow-up actions".to_string(),
    ));

    assert!(
        matches!(
            &result,
            Err(CgError::EventHistoryPlainEmbedRequiresExplicitRole)
        ),
        "plain Embedder::embed(Text) on the event-history wrapper must refuse with the typed \
         explicit-role-required error rather than silently embedding as a document, got {result:?}"
    );
}
