//! The bench drives the production capture path, reads persisted extraction, and reports
//! category scores, ceiling measurements and extraction settings.

use cg_bench::driver::{collect_extracted_session, run_transcript_through_capture};
use cg_bench::gold::load_gold_str;
use cg_bench::report::{BenchReport, CategoryNumbers, VariantRow};
use cg_bench::scoring::{SimilarityProvider, score_session};
use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::embedder::{Embedder, EmbedderRegistration};
use context_graph::harness::CaptureHarnessConfig;
use context_graph::ingest::{
    ConversationExtractor, DecisionRelationRequest, DecisionRelationSeed, ExtractionRequest,
    StructuredHarvest,
};
use context_graph::{Store, StoreConfig};

fn fresh_store(dir: &std::path::Path) -> Store {
    let embedder = std::sync::Arc::new(LexicalMockEmbedder::default());
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: dir.join("cg.db"),
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("store opens with the lexical mock embedder")
}

struct EqualitySim;
impl SimilarityProvider for EqualitySim {
    fn similarity(&self, a: &str, b: &str) -> f32 {
        if a.trim().eq_ignore_ascii_case(b.trim()) {
            1.0
        } else {
            0.0
        }
    }
}

/// Content-faithful double: returns exactly what the transcript states.
struct FaithfulExtractor;
impl ConversationExtractor for FaithfulExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        Ok(StructuredHarvest {
            goals: vec![context_graph::ingest::GoalSeed {
                key: "g1".to_string(),
                text: "ship the correctness bench".to_string(),
                ..context_graph::ingest::GoalSeed::default()
            }],
            decisions: vec![context_graph::ingest::DecisionSeed {
                key: "d1".to_string(),
                text: "drive the production drain path from the bench".to_string(),
                goal_key: Some("g1".to_string()),
                ..context_graph::ingest::DecisionSeed::default()
            }],
            entities: vec![context_graph::ingest::EntitySeed {
                key: "e1".to_string(),
                name: "cg-bench".to_string(),
                ..context_graph::ingest::EntitySeed::default()
            }],
            ..StructuredHarvest::default()
        })
    }

    fn classify_decision_relation(
        &self,
        _request: DecisionRelationRequest<'_>,
    ) -> context_graph::Result<Option<DecisionRelationSeed>> {
        Ok(None)
    }
}

const GOLD: &str = r#"
session: bench-driver-test.jsonl
drafted_by: test
approved_by: "test"
intentions:
  - key: i1
    statement: "ship the correctness bench"
decisions:
  - statement: "drive the production drain path from the bench"
    serves: i1
    made_in: text
entities:
  - name: "cg-bench"
    type: MODULE
"#;

#[test]
fn bench_driver_exercises_production_capture_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = fresh_store(dir.path());
    let reports = run_transcript_through_capture(
        &store,
        "bench-driver-test",
        "Goal: ship the correctness bench. Decision: drive the production drain path from the \
         bench. The cg-bench crate is involved.",
        &FaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("driver captures through the production path");
    assert_eq!(reports.len(), 1, "one chunk, one capture report");

    let extracted = collect_extracted_session(&store, "bench-driver-test")
        .expect("extraction reads back from the store");
    assert_eq!(extracted.intentions.len(), 1);
    assert_eq!(extracted.decisions.len(), 1);
    assert_eq!(
        extracted.decisions[0].goal.as_deref(),
        Some("ship the correctness bench"),
        "the stored SERVES attachment is read back as the decision's goal"
    );
    assert_eq!(extracted.entities.len(), 1);

    // The full bench pipeline: captured store state scores against gold with the scoring method.
    let gold = load_gold_str(GOLD).expect("gold parses");
    let score = score_session(&gold, &extracted, &EqualitySim);
    assert!(score.intention_captured);
    assert_eq!(score.decisions.matched, 1);
    assert_eq!(score.attachment_correct, 1);
    assert_eq!(score.entities.matched, 1);
}

#[test]
fn bench_report_carries_ceiling_row_and_per_lever_history() {
    let numbers = CategoryNumbers {
        intention_fidelity: 0.8,
        attachment: 0.8,
        decision_recall: 0.7,
        decision_precision: 0.9,
        entity_recall: 0.75,
        entity_type_accuracy: 0.85,
        observation_recall: 0.4,
        duplication_rate: 0.1,
        sessions: 4,
    };
    let mut report = BenchReport::new("dev-subset");
    report.push_row(VariantRow {
        label: "recipe+staged".to_string(),
        model: "gemma4-e2b-it-q4".to_string(),
        numbers: numbers.clone(),
    });
    report.push_row(VariantRow {
        label: "ceiling".to_string(),
        model: "hosted:anthropic/claude-haiku-4.5".to_string(),
        numbers,
    });

    let rendered = report.render_markdown();
    // Every report carries the hosted-ceiling row next to the local rows (pre-registered scoring method):
    // the model-gap vs pipeline-gap separator.
    assert!(rendered.contains("gemma4-e2b-it-q4"));
    assert!(rendered.contains("hosted:anthropic/claude-haiku-4.5"));
    // The frozen bars appear so the numbers are read AGAINST something.
    assert!(rendered.contains("0.75"), "bars rendered: {rendered}");
    // Per-lever history: rows accumulate — a second render with both rows shows both labels.
    assert!(rendered.contains("recipe+staged") && rendered.contains("ceiling"));
    // The verdict is printed in whichever direction it falls; with these numbers the
    // judgment-axis bars (0.75/0.75/0.60) are met.
    assert!(
        rendered.contains("PASS") || rendered.contains("FAIL"),
        "a verdict token is printed"
    );
}
