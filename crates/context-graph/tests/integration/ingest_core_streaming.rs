use std::sync::{Arc, Mutex};

use context_graph::ingest::{
    CaptureRequest, HarvestCounts, RollingIngestRequest, RuntimeModelInfo,
    embed_digest_summary_for_connected_embedders, record_conversation, register_runtime_model_info,
    rolling_condense_for_connected_model, rolling_condense_into_store,
};
use context_graph::{
    CgError, Embedder, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EvidenceKind, Store,
    StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open store");
    (dir, store)
}

fn persisted_digest_count(db_path: &std::path::Path) -> usize {
    let db = Database::open(db_path).expect("raw db");
    db.scan("digests", db.snapshot())
        .expect("scan digests")
        .len()
}

fn persisted_extracted_from_edge_targets(db_path: &std::path::Path) -> Vec<uuid::Uuid> {
    let db = Database::open(db_path).expect("raw db");
    let mut targets = db
        .scan("edges", db.snapshot())
        .expect("scan edges")
        .into_iter()
        .filter_map(|row| {
            let relationship = row.values.get("edge_type")?;
            let target = row.values.get("target_id")?;
            match (relationship, target) {
                (Value::Text(edge_type), Value::Uuid(target_id))
                    if edge_type == "EXTRACTED_FROM" =>
                {
                    Some(*target_id)
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    targets.sort();
    targets
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

fn uuid_text_rows(
    db_path: &std::path::Path,
    table: &str,
    text_column: &str,
) -> Vec<(uuid::Uuid, String)> {
    let db = Database::open(db_path).expect("raw db");
    let mut rows = db
        .scan(table, db.snapshot())
        .expect("scan table")
        .into_iter()
        .filter_map(|row| {
            let id = match row.values.get("id")? {
                Value::Uuid(id) => *id,
                _ => return None,
            };
            let text = match row.values.get(text_column)? {
                Value::Text(text) => text.clone(),
                _ => return None,
            };
            Some((id, text))
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    rows
}

fn persisted_extracted_from_edges(
    db_path: &std::path::Path,
) -> Vec<(uuid::Uuid, uuid::Uuid, String)> {
    let db = Database::open(db_path).expect("raw db");
    let mut edges = db
        .scan("edges", db.snapshot())
        .expect("scan edges")
        .into_iter()
        .filter_map(|row| {
            let relationship = match row.values.get("edge_type")? {
                Value::Text(edge_type) if edge_type == "EXTRACTED_FROM" => edge_type.clone(),
                _ => return None,
            };
            let source = match row.values.get("source_id")? {
                Value::Uuid(id) => *id,
                _ => return None,
            };
            let target = match row.values.get("target_id")? {
                Value::Uuid(id) => *id,
                _ => return None,
            };
            Some((source, target, relationship))
        })
        .collect::<Vec<_>>();
    edges.sort();
    edges
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

fn generated_final_summary_only_session() -> String {
    [
        "chunk 001 EARLY_PIN_SENTINEL Goal I_early: keep the original auth path decision discoverable.",
        &"intermediate text with no early sentinel. ".repeat(240),
        "chunk 009 final summary mentions only LATE_SUMMARY_SENTINEL and says auth module now lives at src/authentication.py.",
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

struct CapturingSequenceEmbedder {
    space_id: &'static str,
    max_sequence: usize,
    seen_lengths: Arc<Mutex<Vec<usize>>>,
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
    let transcript = generated_multi_chunk_session();
    let report =
        rolling_condense_into_store(&store, &transcript, 1_024).expect("rolling seam returns");
    drop(store);
    let source_digest = report.provenance_edges.first().map(|edge| edge.digest_id);
    let expected_items = vec!["I_early", "D_old", "O_mid", "D_mid", "O_late", "D_new"];
    let persisted_edge_targets = persisted_extracted_from_edge_targets(&db_path);
    let expected_persisted_targets = source_digest
        .map(|digest_id| vec![digest_id.as_uuid(); expected_items.len()])
        .unwrap_or_default();
    let reopened = Store::open(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen after rolling ingest");
    let text_vectors = reopened
        .count_vectors_in_space("text_default_nomic_v1_5_768")
        .expect("count text vectors");
    drop(reopened);
    let intention_rows = uuid_text_rows(&db_path, "intentions", "description");
    let decision_rows = uuid_text_rows(&db_path, "decisions", "description");
    let observation_types = text_values(&db_path, "observations", "observation_type");
    let persisted_edges = persisted_extracted_from_edges(&db_path);
    let expected_persisted_edges = source_digest
        .map(|digest_id| {
            expected_items
                .iter()
                .filter_map(|item_id| {
                    report.stable_item_ids.get(item_id).map(|source_id| {
                        (
                            *source_id,
                            digest_id.as_uuid(),
                            "EXTRACTED_FROM".to_string(),
                        )
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    assert_eq!(report.pass_count, 4);
    assert!(report.max_prompt_tokens_seen <= 1_024);
    assert!(report.wrote_digest);
    assert!(source_digest.is_some());
    assert_eq!(report.discovered_item_ids, expected_items);
    assert_eq!(
        report
            .provenance_edges
            .iter()
            .map(|edge| {
                (
                    edge.item_id.as_str(),
                    edge.relationship.as_str(),
                    Some(edge.digest_id),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            ("I_early", "EXTRACTED_FROM", source_digest),
            ("D_old", "EXTRACTED_FROM", source_digest),
            ("O_mid", "EXTRACTED_FROM", source_digest),
            ("D_mid", "EXTRACTED_FROM", source_digest),
            ("O_late", "EXTRACTED_FROM", source_digest),
            ("D_new", "EXTRACTED_FROM", source_digest),
        ]
    );
    assert_eq!(
        (
            persisted_digest_count(&db_path),
            text_vectors,
            intention_rows
                .iter()
                .map(|(_, text)| text.as_str())
                .collect::<Vec<_>>(),
            decision_rows
                .iter()
                .map(|(_, text)| text.as_str())
                .collect::<Vec<_>>(),
            observation_types,
            report
                .stable_item_ids
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            persisted_edge_targets.len(),
            persisted_edge_targets,
            persisted_edges,
        ),
        (
            1,
            1,
            vec!["Goal I_early: preserve password-less login for acme"],
            vec![
                "Decision D_mid: JWT access token lifetime is 15 minutes",
                "Decision D_new: auth module at src/authentication.py",
                "Decision D_old: auth module at src/auth.py",
            ],
            vec![
                "Observation O_late: auth module renamed src/auth.py -> src/authentication.py"
                    .to_string(),
                "Observation O_mid: token lifetime policy reviewed".to_string(),
            ],
            expected_items.clone(),
            report.discovered_item_ids.len(),
            expected_persisted_targets,
            expected_persisted_edges,
        )
    );
}

#[test]
fn rolling_condensation_pins_items_on_discovery_not_rederived() {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let transcript = generated_final_summary_only_session();
    let final_summary_only = transcript
        .rsplit("chunk 009")
        .next()
        .unwrap_or_default()
        .to_string();
    let report =
        rolling_condense_into_store(&store, &transcript, 512).expect("store-backed rolling ingest");
    drop(store);
    let source_digest = report.provenance_edges.first().map(|edge| edge.digest_id);
    let intention_rows = uuid_text_rows(&db_path, "intentions", "description");
    let persisted_edges = persisted_extracted_from_edges(&db_path);
    let early_item_id = report.stable_item_ids.get("I_early").copied();

    assert_eq!(
        (
            final_summary_only.contains("EARLY_PIN_SENTINEL"),
            report.discovered_item_ids.iter().any(|id| id == "I_early"),
            report
                .provenance_edges
                .iter()
                .find(|edge| edge.item_id == "I_early")
                .map(|edge| (edge.relationship.as_str(), edge.digest_id)),
            source_digest.is_some(),
            early_item_id.is_some(),
            intention_rows,
            persisted_edges,
        ),
        (
            false,
            true,
            source_digest.map(|digest_id| ("EXTRACTED_FROM", digest_id)),
            true,
            true,
            report
                .stable_item_ids
                .get("I_early")
                .map(|id| {
                    vec![(
                        *id,
                        "Goal I_early: keep the original auth path decision discoverable"
                            .to_string(),
                    )]
                })
                .unwrap_or_default(),
            early_item_id
                .zip(source_digest)
                .map(|(source_id, digest_id)| {
                    vec![(source_id, digest_id.as_uuid(), "EXTRACTED_FROM".to_string())]
                })
                .unwrap_or_default(),
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
    let result = rolling_condense_for_connected_model(
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
    let store = Store::open(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen store with runtime metadata");
    let big = rolling_condense_for_connected_model(
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
    )
    .expect("big model rolling ingest seam returns");
    let small = rolling_condense_for_connected_model(
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
    )
    .expect("small model rolling ingest seam returns");
    let explicit = rolling_condense_for_connected_model(
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
    let store = Store::open(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("reopen store with runtime metadata");
    let exact = rolling_condense_for_connected_model(
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
    )
    .expect("exact");
    let overflow = rolling_condense_for_connected_model(
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
    )
    .expect("overflow");

    assert_eq!(
        (
            exact.resolved_chunk_window,
            exact.pass_count,
            exact.wrote_digest,
            overflow.resolved_chunk_window,
            overflow.pass_count,
            overflow.wrote_digest,
        ),
        (Some(window), 1, true, Some(window), 2, true)
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
fn import_tool_only_session_is_honest_empty() {
    let (_dir, store) = fresh_store();
    let report = record_conversation(
        &store,
        CaptureRequest {
            context_label: "tool-only-import".to_string(),
            transcript: "tool_use grep\ntool_result src/auth.py".to_string(),
            model_available: true,
        },
    )
    .expect("capture seam returns");

    assert_eq!(report.last_harvest, HarvestCounts::default());
    assert_eq!(
        report.message,
        "nothing to capture: session contained tool traffic only"
    );
    assert!(!report.wrote_digest);
}
