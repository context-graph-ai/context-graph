//! Document preparation over the real cached Nomic embedder adds exactly one system-added
//! `search_document: ` task prefix outside the untouched content.
//!
//! The proof compares VECTORS, not a token-count proxy or a space label: for each fixed
//! content string, `embed_prepared_document`'s output vector must equal the SAME cached model's
//! plain `Embedder::embed` output over the exact reference string (`search_document: ` + content),
//! and must differ from embedding the unprefixed content and from a double-added instruction. An
//! implementation that discarded the prepared text, or added the prefix zero or two times, would
//! fail here even if it happened to report a plausible token count.

use context_graph::EmbeddingInput;
use context_graph::EventSearchEmbedder;

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const DOCUMENT_PREFIX: &str = "search_document: ";
const EVENT_HISTORY_SPACE_ID: &str = "event_history_nomic_v1_5_document_query_v1_768";

#[test]
fn nomic_document_preparer_adds_exactly_one_system_task_prefix() {
    let embedder = nomic_real_model::nomic_live_embedder();
    let tokenizer = nomic_real_model::nomic_tokenizer();
    assert_eq!(
        embedder.event_search_slice_token_loss_bound("User: "),
        Some(200)
    );
    assert_eq!(embedder.event_search_slice_token_loss_bound("User:"), None);

    for content in [
        "the quarterly review meeting covered three open action items",
        "a second unrelated sentence about deployment rollback",
    ] {
        let prepared = embedder.prepare_document(content).unwrap_or_else(|err| {
            panic!(
                "document preparation must succeed over the bundled Nomic embedder for {content:?}, got {err:?}"
            )
        });
        let expected_ids = tokenizer
            .encode(format!("{DOCUMENT_PREFIX}{content}"), true)
            .expect("tokenizer encode of exactly one prefix plus the untouched content")
            .get_ids()
            .to_vec();
        assert_eq!(
            prepared.token_count(),
            expected_ids.len(),
            "the prepared document's token count for {content:?} must be the exact final count \
             the model sees — exactly one `{DOCUMENT_PREFIX}` prefix plus the untouched content, \
             special tokens included — measured by the same cached tokenizer loaded independently \
             of the embedder"
        );

        let prepared_output = embedder
            .embed_prepared_document(&prepared)
            .expect("embedding a prepared event-search document must succeed");

        let reference_output = embedder
            .embed(EmbeddingInput::Text(format!("{DOCUMENT_PREFIX}{content}")))
            .expect("plain embed of the exact reference string must succeed");
        assert_eq!(
            prepared_output.vector, reference_output.vector,
            "embed_prepared_document for {content:?} must produce the exact same vector as \
             plain-embedding exactly one `{DOCUMENT_PREFIX}` prefix plus the untouched content \
             through the same cached model"
        );

        let plain_output = embedder
            .embed(EmbeddingInput::Text(content.to_string()))
            .expect("plain embed of the unprefixed content must succeed");
        assert_ne!(
            prepared_output.vector, plain_output.vector,
            "a role-prepared document vector for {content:?} must differ from embedding the \
             unprefixed content"
        );

        let double_prefixed_output = embedder
            .embed(EmbeddingInput::Text(format!(
                "{DOCUMENT_PREFIX}{DOCUMENT_PREFIX}{content}"
            )))
            .expect("plain embed of a double-added instruction must succeed");
        assert_ne!(
            prepared_output.vector, double_prefixed_output.vector,
            "a role-prepared document vector for {content:?} must differ from a double-added \
             `{DOCUMENT_PREFIX}` instruction"
        );

        assert_eq!(
            prepared_output.embedding_space_id, EVENT_HISTORY_SPACE_ID,
            "a prepared event-search document must embed under the dedicated event-history \
             space id"
        );
    }
}
