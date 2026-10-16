//! Preparing an oversized query never silently shortens it. A caller hands in plain text and gets
//! back the exact final token count the model would see — one system-added `search_query: ` task
//! instruction plus the untouched content, special tokens included — even when that count is far
//! past what the model window can hold. Only embedding a still-oversized prepared value is
//! refused, by exact count against the embedder's own declared event-search ceiling.
//!
//! Clamping the encoding before sealing can produce a vector with the canonical event-history
//! identity and an internally consistent token count while silently truncating the query. Oversized
//! fixtures and an independent tokenizer expose that loss of content.
//!
//! The oracle is the cached tokenizer loaded directly, independently of the embedder, and the
//! ceiling is read from the embedder's own declared capability, never a hardcoded constant. The
//! query role deliberately carries no byte-boundary map — it maps no source spans — so this is the
//! count-and-refusal half only.

use context_graph::CgError;
use context_graph::EventSearchEmbedder;

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const QUERY_PREFIX: &str = "search_query: ";

#[test]
fn an_oversized_event_search_query_is_prepared_in_full_and_refused_only_at_embed_time() {
    let embedder = nomic_real_model::nomic_live_embedder();
    let tokenizer = nomic_real_model::nomic_tokenizer();
    let token_limit = embedder
        .event_search_capability()
        .expect("the bundled Nomic embedder must declare its event-search capability")
        .token_limit();

    let oversized = "what did the incident review decide about the rollback window ".repeat(3000);
    let independent_ids = tokenizer
        .encode(format!("{QUERY_PREFIX}{oversized}"), true)
        .expect("tokenizer encode of the oversized prefixed query")
        .get_ids()
        .to_vec();
    assert!(
        independent_ids.len() > token_limit,
        "the fixture must actually exceed the embedder's declared event-search token limit \
         ({token_limit}) for this test to prove anything; got {} tokens",
        independent_ids.len()
    );

    let prepared = embedder.prepare_query(&oversized).unwrap_or_else(|err| {
        panic!(
            "preparing an oversized query must still succeed with the complete count — a caller \
             needs the real number to decide what to do about it — got {err:?}"
        )
    });
    assert_eq!(
        prepared.token_count(),
        independent_ids.len(),
        "the prepared oversized query's token count must be the exact final count, exactly one \
         `{QUERY_PREFIX}` prefix plus the untouched content with special tokens included, never \
         clamped to the model window"
    );
    assert!(
        prepared.token_count() > token_limit,
        "the prepared oversized query must still report a count past the declared ceiling of \
         {token_limit}; a count at or below it means preparation shortened the query"
    );

    let result = embedder.embed_prepared_query(&prepared);
    assert!(
        matches!(
            &result,
            Err(CgError::PreparedInputExceedsModelLimit { token_count, limit })
                if *limit == token_limit && *token_count == prepared.token_count()
        ),
        "embedding a still-oversized prepared query must be refused by exact token count against \
         the embedder's declared maximum, never silently truncated; got {result:?}"
    );
}
