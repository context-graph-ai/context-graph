//! `EventHistoryTextEmbedder::embed` must report which evidence kind it actually refused, the
//! same way the underlying `NomicTextEmbedder::embed` already does (`src/embedder/nomic.rs`,
//! `Embedder::embed`) — never a hardcoded `Text` label regardless of what was actually passed.
//! An operator or caller reading `CgError::UnsupportedEvidenceKind { kind }` must see the real
//! kind they sent, not the one kind the wrapper accepts.

use std::sync::Arc;

use context_graph::embedder::nomic::{EventHistoryTextEmbedder, NomicTextEmbedder};
use context_graph::embedder::{Embedder, EmbeddingInput};
use context_graph::{CgError, EvidenceKind};

#[path = "common/nomic_real_model.rs"]
mod nomic_real_model;

fn wrapper() -> EventHistoryTextEmbedder {
    let model = Arc::new(NomicTextEmbedder::new(
        nomic_real_model::local_offline_config(nomic_real_model::canonical_model_dir()),
    ));
    EventHistoryTextEmbedder::over(model).expect("registration over a fit model must succeed")
}

fn assert_reports_kind(input: EmbeddingInput, expected: EvidenceKind, label: &str) {
    let wrapper = wrapper();
    let result = wrapper.embed(input);
    assert!(
        matches!(
            &result,
            Err(CgError::UnsupportedEvidenceKind { kind }) if *kind == expected
        ),
        "the event-history wrapper must report {label} as the unsupported kind for its own \
         {label} input, got {result:?}"
    );
}

#[test]
fn event_history_wrapper_names_the_actual_unsupported_input_kind() {
    assert_reports_kind(
        EmbeddingInput::ImageFrame {
            uri: "file:///tmp/frame.png".to_string(),
        },
        EvidenceKind::ImageFrame,
        "ImageFrame",
    );
    assert_reports_kind(
        EmbeddingInput::ImageBytes(vec![0u8, 1, 2, 3]),
        EvidenceKind::ImageFrame,
        "ImageBytes (maps to EvidenceKind::ImageFrame)",
    );
    assert_reports_kind(
        EmbeddingInput::AudioSegment {
            uri: "file:///tmp/clip.wav".to_string(),
        },
        EvidenceKind::AudioSegment,
        "AudioSegment",
    );
    assert_reports_kind(
        EmbeddingInput::VideoSegment {
            uri: "file:///tmp/clip.mp4".to_string(),
        },
        EvidenceKind::VideoSegment,
        "VideoSegment",
    );
    assert_reports_kind(
        EmbeddingInput::StructuredSignal(serde_json::json!({"k": "v"})),
        EvidenceKind::StructuredSignal,
        "StructuredSignal",
    );
}
