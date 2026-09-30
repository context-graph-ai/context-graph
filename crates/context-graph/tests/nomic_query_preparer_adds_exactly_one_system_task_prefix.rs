//! Query preparation over the real cached Nomic embedder adds exactly one system-added
//! `search_query: ` task prefix outside the untouched content, mirroring the document role.
//!
//! The proof compares VECTORS, not a token-count proxy or a space label. The document
//! preparation proof explains the comparison.

use context_graph::EmbeddingInput;
use context_graph::EventSearchEmbedder;

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const QUERY_PREFIX: &str = "search_query: ";
const EVENT_HISTORY_SPACE_ID: &str = "event_history_nomic_v1_5_document_query_v1_768";

#[test]
fn nomic_query_preparer_adds_exactly_one_system_task_prefix() {
    let embedder = nomic_real_model::nomic_live_embedder();
    let tokenizer = nomic_real_model::nomic_tokenizer();

    for content in [
        "what did we decide about the rollback window",
        "second unrelated query about the deploy freeze",
    ] {
        let prepared = embedder.prepare_query(content).unwrap_or_else(|err| {
            panic!(
                "query preparation must succeed over the bundled Nomic embedder for {content:?}, got {err:?}"
            )
        });
        let expected_ids = tokenizer
            .encode(format!("{QUERY_PREFIX}{content}"), true)
            .expect("tokenizer encode of exactly one prefix plus the untouched content")
            .get_ids()
            .to_vec();
        assert_eq!(
            prepared.token_count(),
            expected_ids.len(),
            "the prepared query's token count for {content:?} must be the exact final count the \
             model sees — exactly one `{QUERY_PREFIX}` prefix plus the untouched content, special \
             tokens included — measured by the same cached tokenizer loaded independently of the \
             embedder"
        );

        let prepared_output = embedder
            .embed_prepared_query(&prepared)
            .expect("embedding a prepared event-search query must succeed");

        let reference_output = embedder
            .embed(EmbeddingInput::Text(format!("{QUERY_PREFIX}{content}")))
            .expect("plain embed of the exact reference string must succeed");
        assert_eq!(
            prepared_output.vector, reference_output.vector,
            "embed_prepared_query for {content:?} must produce the exact same vector as \
             plain-embedding exactly one `{QUERY_PREFIX}` prefix plus the untouched content \
             through the same cached model"
        );

        let plain_output = embedder
            .embed(EmbeddingInput::Text(content.to_string()))
            .expect("plain embed of the unprefixed content must succeed");
        assert_ne!(
            prepared_output.vector, plain_output.vector,
            "a role-prepared query vector for {content:?} must differ from embedding the \
             unprefixed content"
        );

        let double_prefixed_output = embedder
            .embed(EmbeddingInput::Text(format!(
                "{QUERY_PREFIX}{QUERY_PREFIX}{content}"
            )))
            .expect("plain embed of a double-added instruction must succeed");
        assert_ne!(
            prepared_output.vector, double_prefixed_output.vector,
            "a role-prepared query vector for {content:?} must differ from a double-added \
             `{QUERY_PREFIX}` instruction"
        );

        assert_eq!(
            prepared_output.embedding_space_id, EVENT_HISTORY_SPACE_ID,
            "a prepared event-search query must embed under the dedicated event-history space id"
        );
    }
}
