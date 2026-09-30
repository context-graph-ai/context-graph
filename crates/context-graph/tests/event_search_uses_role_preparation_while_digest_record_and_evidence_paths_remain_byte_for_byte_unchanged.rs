//! Role-aware event search keeps the existing text embedding path intact:
//!
//! 1. Existing text path: the existing `Embedder::embed` text path, for the same fixed texts recorded
//!    in the checked-in golden fixture `tests/fixtures/text_default_space_goldens.json`, still
//!    produces byte-for-byte identical model input (tokenizer ids) and vectors through the mock
//!    and the real cached Nomic embedder. Digest/Record/Evidence embedding never moves.
//! 2. Event search path: the real Nomic embedder answers role-aware event-search preparation
//!    end-to-end (`prepare_document` -> `embed_prepared_document`) and the resulting vector
//!    equals the same cached model's plain `embed` output over the exact reference string
//!    (`search_document: ` + content) — proved by vector equality, not just a label — and is
//!    reported under the dedicated event-history space id, never the unlabelled text-default one.
//!    Import projection also exercises this behavior through the full ingest path.
//!
//! The checked-in golden fixture is treated as an immutable oracle: this test verifies
//! its recorded sha256 before trusting its contents, and never writes to it or to any other path
//! — only the separate, `#[ignore]`d `text_default_space_goldens_capture` tool may regenerate it,
//! and only into a caller-supplied destination outside the repository.

use std::fs;
use std::path::PathBuf;

use context_graph::embedder::mock::MockEmbedder;
use context_graph::{Embedder, EmbeddingInput, EventSearchEmbedder};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

/// Recorded sha256 of `tests/fixtures/text_default_space_goldens.json` at the moment this
/// checksum was pinned. A change to the fixture's bytes — whether an honest rebase of the
/// baseline or a silent replacement by the regression it is supposed to catch — must be a
/// deliberate, reviewed edit to this constant, never a side effect of running `cargo test`.
const GOLDEN_FIXTURE_SHA256: &str =
    "aba4629616a919e6a9bb6bacf6a4f7eb614935c78dead8f48292a97eee457bf7";

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct GoldenEntry {
    text: String,
    mock_token_len_proxy: usize,
    mock_vector_sha256: String,
    real_nomic_tokenizer_ids: Vec<u32>,
    real_nomic_vector_sha256: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct GoldenFixture {
    model_artifacts_sha256: String,
    entries: Vec<GoldenEntry>,
}

fn sha256_of_vector(vector: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in vector {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[test]
fn event_search_uses_role_preparation_while_digest_record_and_evidence_paths_remain_byte_for_byte_unchanged()
 {
    // -- Unchanged half: re-derive the same golden entries from the live `embed` path and diff
    // against the checked-in fixture.
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/text_default_space_goldens.json");
    let fixture_json = fs::read_to_string(&fixture_path).expect("read checked-in golden fixture");
    let mut hasher = Sha256::new();
    hasher.update(fixture_json.as_bytes());
    let fixture_sha256 = format!("{:x}", hasher.finalize());
    assert_eq!(
        fixture_sha256, GOLDEN_FIXTURE_SHA256,
        "the checked-in golden fixture's bytes must match the recorded checksum exactly; a \
         mismatch means the immutable oracle moved outside a deliberate, reviewed edit to \
         GOLDEN_FIXTURE_SHA256"
    );
    let expected: GoldenFixture =
        serde_json::from_str(&fixture_json).expect("parse golden fixture");

    assert_eq!(
        nomic_real_model::model_artifacts_sha256(),
        expected.model_artifacts_sha256,
        "the installed Nomic model artifacts must match the artifacts that produced the golden vectors"
    );

    let mock = MockEmbedder::new("text_default_nomic_v1_5_768", 768);
    let real = nomic_real_model::nomic_live_embedder();
    let tokenizer = nomic_real_model::nomic_tokenizer();

    for entry in &expected.entries {
        let mock_out = mock
            .embed(EmbeddingInput::Text(entry.text.clone()))
            .expect("mock embed");
        let real_out = real
            .embed(EmbeddingInput::Text(entry.text.clone()))
            .expect("real nomic embed (cached, offline)");
        let ids = tokenizer
            .encode(entry.text.as_str(), true)
            .expect("tokenizer encode")
            .get_ids()
            .to_vec();

        assert_eq!(
            ids, entry.real_nomic_tokenizer_ids,
            "existing embed path must tokenize {:?} identically to the artifact-bound golden",
            entry.text
        );
        assert_eq!(
            sha256_of_vector(&mock_out.vector),
            entry.mock_vector_sha256,
            "mock embed path must match the artifact-bound golden for {:?}",
            entry.text
        );
        assert_eq!(
            sha256_of_vector(&real_out.vector),
            entry.real_nomic_vector_sha256,
            "real Nomic embed path must be byte-for-byte unchanged for {:?}",
            entry.text
        );
    }

    // -- Capability half: role-aware preparation over the same real embedder, proved by VECTOR
    // equality against the frozen legacy inference path over the exact reference string, not by
    // a label alone.
    let content = "the quarterly review meeting covered three open action items";
    let prepared = real.prepare_document(content).unwrap_or_else(|err| {
        panic!("event-search document preparation must be supported over the bundled Nomic embedder, got {err:?}")
    });
    let output = real
        .embed_prepared_document(&prepared)
        .expect("embedding a prepared event-search document must succeed");

    let reference_output = real
        .embed(EmbeddingInput::Text(format!("search_document: {content}")))
        .expect("plain embed of the exact reference string must succeed");
    assert_eq!(
        output.vector, reference_output.vector,
        "embed_prepared_document must produce the exact same vector as plain-embedding exactly \
         one `search_document: ` prefix plus the untouched content through the same cached model"
    );

    assert_eq!(
        output.embedding_space_id, "event_history_nomic_v1_5_document_query_v1_768",
        "a prepared event-search document must embed under the dedicated event-history space id, \
         never the unlabelled text-default space"
    );
    assert_ne!(
        output.embedding_space_id, "text_default_nomic_v1_5_768",
        "event-search embedding must never be reported under the Digest/Evidence text-default space"
    );
}
