//! C5 (assertion 74, preparer arm) — source or query text that itself begins with the literal
//! words `search_document:` / `search_query:` is never inspected or rejected for that reason: the
//! preparer still adds exactly one system-added prefix outside the untouched content, so the
//! literal words end up appearing twice in the model input — the system-added instruction first,
//! the caller's own content second. A later retrieval proof covers the "searchable" half end to
//! end; here the arm is the preparer never special-casing this content.
//!
//! The proof compares VECTORS, not a token-count proxy: `embed_prepared_*`'s output must
//! equal the same cached model's plain `embed` output over the exact reference string (one
//! system-added prefix, then the caller's literal content untouched).

use context_graph::EmbeddingInput;
use context_graph::EventSearchEmbedder;

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const EVENT_HISTORY_SPACE_ID: &str = "event_history_nomic_v1_5_document_query_v1_768";

#[test]
fn literal_source_and_query_text_starting_with_nomic_prefix_words_is_preserved_and_searchable() {
    let embedder = nomic_real_model::nomic_live_embedder();

    let literal_document = "search_document: this is a user's literal content, not a system prefix";
    let prepared_document = embedder.prepare_document(literal_document).unwrap_or_else(|err| {
        panic!(
            "document preparation must preserve literal `search_document:` content untouched, got {err:?}"
        )
    });
    let prepared_document_output = embedder
        .embed_prepared_document(&prepared_document)
        .expect("embedding a prepared literal document must succeed");
    let reference_document_output = embedder
        .embed(EmbeddingInput::Text(format!(
            "search_document: {literal_document}"
        )))
        .expect("plain embed of the exact reference string must succeed");
    assert_eq!(
        prepared_document_output.vector, reference_document_output.vector,
        "literal `search_document:` content must be preserved verbatim after exactly one \
         system-added prefix, never inspected or rejected for starting with those words"
    );
    assert_eq!(
        prepared_document_output.embedding_space_id, EVENT_HISTORY_SPACE_ID,
        "a prepared literal document must embed under the dedicated event-history space id"
    );

    let literal_query = "search_query: what did the user literally type here";
    let prepared_query = embedder.prepare_query(literal_query).unwrap_or_else(|err| {
        panic!(
            "query preparation must preserve literal `search_query:` content untouched, got {err:?}"
        )
    });
    let prepared_query_output = embedder
        .embed_prepared_query(&prepared_query)
        .expect("embedding a prepared literal query must succeed");
    let reference_query_output = embedder
        .embed(EmbeddingInput::Text(format!(
            "search_query: {literal_query}"
        )))
        .expect("plain embed of the exact reference string must succeed");
    assert_eq!(
        prepared_query_output.vector, reference_query_output.vector,
        "literal `search_query:` content must be preserved verbatim after exactly one \
         system-added prefix, never inspected or rejected for starting with those words"
    );
    assert_eq!(
        prepared_query_output.embedding_space_id, EVENT_HISTORY_SPACE_ID,
        "a prepared literal query must embed under the dedicated event-history space id"
    );
}
