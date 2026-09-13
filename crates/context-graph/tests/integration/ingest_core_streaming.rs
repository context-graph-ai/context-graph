use std::sync::{Arc, Mutex};

use context_graph::ingest::{
    CaptureMode, CaptureRequest, ConversationExtractor, EntitySeed, ExtractedEntityType,
    ExtractionRequest, GoalSeed, HarvestCounts, ObservationSeed, RollingCaptureRequest,
    RollingIngestRequest, RuntimeModelInfo, StructuredHarvest,
    capture_rolling_conversation_for_model_with_extractor,
    capture_rolling_conversation_with_extractor, embed_digest_summary_for_connected_embedders,
    record_conversation, register_runtime_model_info,
};
use context_graph::{
    CgError, Embedder, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EvidenceKind, Store,
    StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;

use super::fake_extractor::open_store_with_lexical_embedder;

#[path = "../common/nomic_real_model.rs"]
mod nomic_real_model;

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn persisted_digest_count(db_path: &std::path::Path) -> usize {
    let db = Database::open(db_path).expect("raw db");
    db.scan("digests", db.snapshot())
        .expect("scan digests")
        .len()
}

fn text_values(db_path: &std::path::Path, table: &str, column: &str) -> Vec<String> {
    let db = Database::open(db_path).expect("raw db");
    let mut values = db
        .scan(table, db.snapshot())
        .expect("scan table")
        .into_iter()
        .filter_map(|row| match row.values.get(column) {
            Some(Value::Text(value)) => Some(value.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    values.sort();
    values
}

fn persisted_stream_snapshot(db_path: &std::path::Path) -> (Vec<uuid::Uuid>, usize, usize, usize) {
    let db = Database::open(db_path).expect("raw db");
    let snapshot = db.snapshot();
    let mut digest_ids = db
        .scan("digests", snapshot)
        .expect("scan digests")
        .into_iter()
        .filter_map(|row| match row.values.get("id") {
            Some(Value::Uuid(id)) => Some(*id),
            _ => None,
        })
        .collect::<Vec<_>>();
    digest_ids.sort();
    let intentions = db
        .scan("intentions", snapshot)
        .expect("scan intentions")
        .len();
    let decisions = db
        .scan("decisions", snapshot)
        .expect("scan decisions")
        .len();
    let observations = db
        .scan("observations", snapshot)
        .expect("scan observations")
        .len();
    (digest_ids, intentions, decisions, observations)
}

fn generated_multi_chunk_session() -> String {
    [
        "chunk 001 EARLY_ONLY_SENTINEL Goal I_early: preserve password-less login for acme. Decision D_old: auth module at src/auth.py.",
        &"neutral filler without auth facts. ".repeat(180),
        "chunk 002 MID_ONLY_SENTINEL Observation O_mid: token lifetime policy reviewed. Decision D_mid: JWT access token lifetime is 15 minutes.",
        &"more neutral filler without auth facts. ".repeat(180),
        "chunk 003 LATE_ONLY_SENTINEL Observation O_late: auth module renamed src/auth.py -> src/authentication.py. Decision D_new: auth module at src/authentication.py.",
    ]
    .join("\n")
}

fn rolling_request(
    transcript: String,
    context_label: &str,
    model_key: Option<&str>,
    prompt_overhead_tokens: usize,
    output_budget_tokens: usize,
    running_state_cap_tokens: usize,
    explicit_chunk_tokens: Option<usize>,
) -> RollingIngestRequest {
    RollingIngestRequest {
        context_label: context_label.to_string(),
        model_key: model_key.map(ToOwned::to_owned),
        transcript,
        prompt_overhead_tokens,
        output_budget_tokens,
        running_state_cap_tokens,
        explicit_chunk_tokens,
        context_fraction: None,
    }
}

fn expected_window(
    model_context: usize,
    prompt_overhead_tokens: usize,
    output_budget_tokens: usize,
    running_state_cap_tokens: usize,
    explicit_chunk_tokens: Option<usize>,
) -> usize {
    explicit_chunk_tokens.unwrap_or_else(|| {
        (model_context - prompt_overhead_tokens - output_budget_tokens - running_state_cap_tokens)
            .min(model_context * 3 / 5)
    })
}

fn register_runtime_model(
    store: &Store,
    context_label: &str,
    model_key: &str,
    advertised_context_tokens: usize,
    default_for_context: bool,
) {
    register_runtime_model_info(
        store,
        RuntimeModelInfo {
            context_label: context_label.to_string(),
            model_key: model_key.to_string(),
            advertised_context_tokens,
            default_for_context,
        },
    )
    .expect("register runtime model metadata");
}

#[derive(Clone, Default)]
struct SpyRollingExtractor {
    prompts: Arc<Mutex<Vec<String>>>,
}

impl SpyRollingExtractor {
    fn prompts(&self) -> Vec<String> {
        self.prompts.lock().expect("spy prompts").clone()
    }
}

impl ConversationExtractor for SpyRollingExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        self.prompts
            .lock()
            .expect("spy prompts")
            .push(request.transcript.to_string());
        let source = source_chunk_from_prompt(request.transcript);
        let mut harvest = StructuredHarvest::default();
        if source.contains("SENTINEL_OPEN") || source.contains("SENTINEL_CLOSE") {
            harvest.goals.push(GoalSeed {
                key: "G_sentinel".to_string(),
                text: "preserve sentinel memory across a long session".to_string(),
                context_id: None,
            });
            harvest.entities.push(EntitySeed {
                key: "E_sentinel".to_string(),
                name: "sentinel session memory".to_string(),
                path: "memory://sentinel-session".to_string(),
                entity_type: ExtractedEntityType::Document,
                entity_type_raw: None,
                context_id: None,
            });
        }
        if source.contains("SENTINEL_OPEN") {
            harvest.observations.push(ObservationSeed {
                key: "O_open".to_string(),
                text: "SENTINEL_OPEN appeared near the start of the long session".to_string(),
                entity_key: Some("E_sentinel".to_string()),
                path_change: None,
                ..Default::default()
            });
        }
        if source.contains("SENTINEL_CLOSE") {
            harvest.observations.push(ObservationSeed {
                key: "O_close".to_string(),
                text: "SENTINEL_CLOSE appeared near the end of the long session".to_string(),
                entity_key: Some("E_sentinel".to_string()),
                path_change: None,
                ..Default::default()
            });
        }
        Ok(harvest)
    }
}

fn source_chunk_from_prompt(prompt: &str) -> String {
    prompt
        .rsplit_once("Next session chunk:\n")
        .map(|(_, chunk)| chunk)
        .unwrap_or(prompt)
        .to_string()
}

struct CapturingSequenceEmbedder {
    space_id: &'static str,
    max_sequence: usize,
    seen_lengths: Arc<Mutex<Vec<usize>>>,
}

struct RecordingNomicDigestEmbedder {
    ordinary_limit: usize,
    inputs: Arc<Mutex<Vec<String>>>,
}

impl Embedder for RecordingNomicDigestEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        let EmbeddingInput::Text(text) = input else {
            unreachable!("Digest embedding seam supplies text")
        };
        self.inputs
            .lock()
            .expect("recorded digest inputs")
            .push(text);
        Ok(EmbeddingOutput {
            embedding_space_id: "recording-nomic-digest".to_string(),
            dimension: 1,
            producer: Default::default(),
            vector: vec![1.0],
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: "recording-nomic-digest".to_string(),
            dimension: 1,
            supported_evidence_kinds: vec![EvidenceKind::Text],
            ..Default::default()
        }
    }

    fn max_sequence_length(&self) -> usize {
        self.ordinary_limit
    }
}

impl Embedder for CapturingSequenceEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        if let EmbeddingInput::Text(text) = input {
            self.seen_lengths
                .lock()
                .expect("seen lengths")
                .push(text.len());
        }
        Ok(EmbeddingOutput {
            embedding_space_id: self.space_id.to_string(),
            dimension: 1,
            producer: Default::default(),
            vector: vec![1.0],
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: self.space_id.to_string(),
            dimension: 1,
            supported_evidence_kinds: vec![EvidenceKind::Text],
            ..Default::default()
        }
    }

    fn max_sequence_length(&self) -> usize {
        self.max_sequence
    }
}

#[test]
fn long_session_rolling_condensation_covers_both_ends_no_overflow() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let transcript = format!(
        "SENTINEL_OPEN{}SENTINEL_CLOSE",
        "plainprosefiller".repeat(420)
    );
    let extractor = SpyRollingExtractor::default();
    let report = capture_rolling_conversation_with_extractor(
        &store,
        RollingCaptureRequest {
            context_label: "rolling-import".to_string(),
            transcript: transcript.clone(),
            chunk_window: 1_024,
            running_state_cap_tokens: 4_096,
            model_context_tokens: Some(200_000),
        },
        &extractor,
    )
    .expect("rolling seam returns");
    let prompts = extractor.prompts();
    let source_chunks = prompts
        .iter()
        .map(|prompt| source_chunk_from_prompt(prompt))
        .collect::<Vec<_>>();
    drop(store);
    let observation_types = text_values(&db_path, "observations", "observation_type");

    assert_eq!(
        (
            report.import_chunks,
            prompts.len(),
            prompts.len() > 1,
            source_chunks.len(),
            source_chunks
                .iter()
                .all(|chunk| !chunk.is_empty() && chunk.len() <= 1_024),
            source_chunks.join(""),
            persisted_digest_count(&db_path),
            observation_types
                .iter()
                .any(|text| text.contains("SENTINEL_OPEN")),
            observation_types
                .iter()
                .any(|text| text.contains("SENTINEL_CLOSE")),
        ),
        (
            Some(prompts.len()),
            prompts.len(),
            true,
            source_chunks.len(),
            true,
            transcript,
            1,
            true,
            true,
        )
    );
}

#[test]
fn streaming_small_context_fails_loud_not_degenerate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open store");
    register_runtime_model(&store, "small-context", "tiny-runtime", 8, true);
    drop(store);
    let before = persisted_stream_snapshot(&db_path);
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen store");
    let extractor = SpyRollingExtractor::default();
    let result = capture_rolling_conversation_for_model_with_extractor(
        &store,
        rolling_request(
            generated_multi_chunk_session(),
            "small-context",
            None,
            4,
            4,
            4,
            None,
        ),
        &extractor,
    );
    drop(store);
    let after = persisted_stream_snapshot(&db_path);

    assert_eq!(
        (
            matches!(
                result,
                Err(CgError::TranscriptTooLargeForModel {
                    model_context: 8,
                    required: _
                })
            ),
            before,
            after,
        ),
        (true, (Vec::new(), 0, 0, 0), (Vec::new(), 0, 0, 0))
    );
}

#[test]
fn chunk_window_derives_from_connected_model_context_default_haiku_overridable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open store");
    let big_context = 160_000 + 40_000;
    let small_context = 5_000 + 3_000;
    let prompt_overhead = 1_000;
    let output = 2_000;
    let carried = 4_000;
    register_runtime_model(
        &store,
        "metadata-window",
        "large-runtime",
        big_context,
        true,
    );
    register_runtime_model(
        &store,
        "metadata-window",
        "small-runtime",
        small_context,
        false,
    );
    drop(store);
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    });
    let extractor = SpyRollingExtractor::default();
    let big = capture_rolling_conversation_for_model_with_extractor(
        &store,
        rolling_request(
            "metadata-driven big model transcript".to_string(),
            "metadata-window",
            None,
            prompt_overhead,
            output,
            carried,
            None,
        ),
        &extractor,
    )
    .expect("big model rolling ingest seam returns");
    let small = capture_rolling_conversation_for_model_with_extractor(
        &store,
        rolling_request(
            "metadata-driven small model transcript".to_string(),
            "metadata-window",
            Some("small-runtime"),
            prompt_overhead,
            output,
            carried,
            None,
        ),
        &extractor,
    )
    .expect("small model rolling ingest seam returns");
    let explicit = capture_rolling_conversation_for_model_with_extractor(
        &store,
        rolling_request(
            "metadata-driven override transcript".to_string(),
            "metadata-window",
            Some("large-runtime"),
            prompt_overhead,
            output,
            carried,
            Some(777),
        ),
        &extractor,
    )
    .expect("override rolling ingest seam returns");

    assert_eq!(
        (
            big.model_context_tokens,
            big.resolved_chunk_window,
            small.model_context_tokens,
            small.resolved_chunk_window,
            small.resolved_chunk_window < big.resolved_chunk_window,
            explicit.resolved_chunk_window,
        ),
        (
            Some(big_context),
            Some(expected_window(
                big_context,
                prompt_overhead,
                output,
                carried,
                None,
            )),
            Some(small_context),
            Some(expected_window(
                small_context,
                prompt_overhead,
                output,
                carried,
                None,
            )),
            true,
            Some(777),
        )
    );
}

/// Digest embedding uses Nomic's ordinary embedding declaration, not its event-search-only
/// ceiling. The production model is constructed only to read that ordinary declaration; the
/// recording embedder then drives the Digest seam without running real inference, so a ceiling
/// accidentally reintroduced into `max_sequence_length` cannot hide behind a mock or a returned
/// summary value.
#[test]
fn nomic_event_search_limit_cannot_truncate_digest_embedding_text() {
    let full = "incident review summary paragraph ".repeat(500);
    assert!(
        full.len() > 8_192,
        "the fixture must actually exceed 8,192 characters for this regression to prove anything; \
         got {} characters",
        full.len()
    );
    let inputs = Arc::new(Mutex::new(Vec::new()));
    let model = context_graph::embedder::nomic::NomicTextEmbedder::new(
        nomic_real_model::local_offline_config(nomic_real_model::canonical_model_dir()),
    );
    let ordinary_limit = model.max_sequence_length();
    let embedder = RecordingNomicDigestEmbedder {
        ordinary_limit,
        inputs: Arc::clone(&inputs),
    };
    let (embedded, outputs) = embed_digest_summary_for_connected_embedders(&full, &[&embedder])
        .expect("embedding seam returns");

    assert_eq!(
        (
            ordinary_limit,
            embedded,
            inputs.lock().expect("recorded digest inputs").clone(),
            outputs.len(),
        ),
        (usize::MAX, full.clone(), vec![full], 1),
        "Nomic's event-search limit must never become its ordinary Digest embedding limit: the \
         cached real model must receive the full Digest input byte-for-byte"
    );
}

#[test]
fn digest_summary_bounded_to_connected_embedder_max_sequence() {
    let full = "a".repeat(10_000);
    let wide_lengths = Arc::new(Mutex::new(Vec::new()));
    let narrow_lengths = Arc::new(Mutex::new(Vec::new()));
    let wide = CapturingSequenceEmbedder {
        space_id: "wide-sequence",
        max_sequence: 8_192,
        seen_lengths: Arc::clone(&wide_lengths),
    };
    let narrow = CapturingSequenceEmbedder {
        space_id: "narrow-sequence",
        max_sequence: 2_048,
        seen_lengths: Arc::clone(&narrow_lengths),
    };
    let (embedded, outputs) =
        embed_digest_summary_for_connected_embedders(&full, &[&wide, &narrow])
            .expect("embedding seam returns");

    assert_eq!(
        embedded.len(),
        wide.max_sequence_length().min(narrow.max_sequence_length())
    );
    assert_eq!(
        (
            outputs
                .iter()
                .map(|output| output.embedding_space_id.as_str())
                .collect::<Vec<_>>(),
            wide_lengths.lock().expect("wide lengths").clone(),
            narrow_lengths.lock().expect("narrow lengths").clone(),
        ),
        (
            vec!["wide-sequence", "narrow-sequence"],
            vec![2_048],
            vec![2_048],
        )
    );
}

#[test]
fn chunk_boundary_exact_window_is_single_pass() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open store");
    let model_context = 100;
    let prompt_overhead = 10;
    let output = 10;
    let carried = 10;
    let window = expected_window(model_context, prompt_overhead, output, carried, None);
    register_runtime_model(
        &store,
        "boundary-window",
        "boundary-runtime",
        model_context,
        true,
    );
    drop(store);
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    });
    let extractor = SpyRollingExtractor::default();
    let exact = capture_rolling_conversation_for_model_with_extractor(
        &store,
        rolling_request(
            "x".repeat(window),
            "boundary-window",
            None,
            prompt_overhead,
            output,
            carried,
            None,
        ),
        &extractor,
    )
    .expect("exact");
    let overflow = capture_rolling_conversation_for_model_with_extractor(
        &store,
        rolling_request(
            "x".repeat(window + 1),
            "boundary-window",
            None,
            prompt_overhead,
            output,
            carried,
            None,
        ),
        &extractor,
    )
    .expect("overflow");

    assert_eq!(
        (
            exact.resolved_chunk_window,
            exact.import_chunks,
            exact.wrote_digest,
            overflow.resolved_chunk_window,
            overflow.import_chunks,
            overflow.wrote_digest,
        ),
        (Some(window), Some(1), true, Some(window), Some(2), true)
    );
}

#[test]
fn import_empty_jsonl_yields_honest_zero_harvest() {
    let (_dir, store) = fresh_store();
    let report = record_conversation(
        &store,
        CaptureRequest {
            context_label: "empty-import".to_string(),
            transcript: String::new(),
            model_available: true,
        },
    )
    .expect("capture seam returns");

    assert_eq!(report.last_harvest, HarvestCounts::default());
    assert_eq!(report.message, "nothing to capture: empty import");
    assert!(!report.wrote_digest);
}

#[test]
fn record_tool_only_without_extractor_stores_verbatim() {
    let (_dir, store) = fresh_store();
    let report = record_conversation(
        &store,
        CaptureRequest {
            context_label: "tool-only-import".to_string(),
            transcript: "tool_use grep\ntool_result src/auth.py".to_string(),
            model_available: false,
        },
    )
    .expect("capture seam returns");

    assert_eq!(report.last_harvest, HarvestCounts::default());
    assert_eq!(report.mode, CaptureMode::RawVerbatim);
    assert_eq!(
        report.message,
        "model-free capture: stored verbatim text; no derived graph or impact analysis"
    );
    assert!(report.wrote_digest);
}
