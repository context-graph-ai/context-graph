//! Manual baseline-capture TOOL, not a regression test: this is deliberately excluded
//! from the ordinary `cargo test` run (`#[ignore]`) and never writes to the checked-in fixture,
//! the repository, or any build working directory. Rewriting the checked-in oracle from an ordinary test
//! run would let the exact regression it is supposed to catch replace the baseline instead of
//! failing against it — the immutable-oracle promise
//! `event_search_uses_role_preparation_while_digest_record_and_evidence_paths_remain_byte_for_byte_unchanged`
//! depends on.
//!
//! To (re)capture a baseline, run explicitly with the destination named by `CG_GOLDENS_OUT`:
//!
//! ```text
//! CG_GOLDENS_OUT=/tmp/goldens.json cargo test --test text_default_space_goldens_capture \
//!     -- --ignored
//! ```
//!
//! The output is written ONLY to that caller-supplied path. Promoting a freshly captured file to
//! `tests/fixtures/text_default_space_goldens.json` — and updating the recorded
//! `GOLDEN_FIXTURE_SHA256` checksum in the regression test — is a separate, deliberate, reviewed
//! step this tool never performs itself.

use std::fs;
use std::path::PathBuf;

use context_graph::embedder::mock::MockEmbedder;
use context_graph::{Embedder, EmbeddingInput};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

const FIXED_TEXTS: &[&str] = &[
    "the quarterly review meeting covered three open action items",
    "search_document: a literal prefix word sequence used as ordinary content",
    "caf\u{e9} na\u{ef}ve r\u{e9}sum\u{e9} with combining e\u{301} and stray \u{1f600} emoji",
    "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{30c6}\u{30ad}\u{30b9}\u{30c8}\u{3092}\u{542b}\u{3080}\u{4f8b}\u{6587}",
    "",
];

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct GoldenEntry {
    text: String,
    mock_token_len_proxy: usize,
    mock_vector_sha256: String,
    real_nomic_tokenizer_ids: Vec<u32>,
    real_nomic_vector_sha256: String,
}

fn sha256_of_vector(vector: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in vector {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[test]
#[ignore = "manual baseline capture; writes only to CG_GOLDENS_OUT"]
fn text_default_space_base_golden_capture() {
    let destination = std::env::var_os("CG_GOLDENS_OUT").unwrap_or_else(|| {
        panic!(
            "this tool writes only to a caller-supplied destination: set CG_GOLDENS_OUT to an \
             explicit path outside the repository before running it"
        )
    });
    let destination = PathBuf::from(destination);

    let mock = MockEmbedder::new("text_default_nomic_v1_5_768", 768);
    let real = nomic_real_model::nomic_live_embedder();
    let tokenizer = nomic_real_model::nomic_tokenizer();

    let mut entries = Vec::new();
    for text in FIXED_TEXTS {
        let mock_out = mock
            .embed(EmbeddingInput::Text((*text).to_string()))
            .expect("mock embed");
        let real_out = real
            .embed(EmbeddingInput::Text((*text).to_string()))
            .expect("real nomic embed (cached, offline)");
        let ids = tokenizer
            .encode(*text, true)
            .expect("tokenizer encode")
            .get_ids()
            .to_vec();
        entries.push(GoldenEntry {
            text: (*text).to_string(),
            mock_token_len_proxy: text.len(),
            mock_vector_sha256: sha256_of_vector(&mock_out.vector),
            real_nomic_tokenizer_ids: ids,
            real_nomic_vector_sha256: sha256_of_vector(&real_out.vector),
        });
    }

    let json = serde_json::to_string_pretty(&entries).expect("serialize goldens");

    // Writes ONLY to the caller-supplied destination — never to the checked-in fixture path,
    // never anywhere under the repository or a build working directory.
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!(
                "failed to create parent directory for CG_GOLDENS_OUT destination {}: {err}",
                destination.display()
            )
        });
    }
    fs::write(&destination, &json).unwrap_or_else(|err| {
        panic!(
            "failed to write captured goldens to CG_GOLDENS_OUT destination {}: {err}",
            destination.display()
        )
    });

    assert_eq!(entries.len(), FIXED_TEXTS.len());
}
