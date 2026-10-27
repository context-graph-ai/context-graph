//! The reliability wrapper re-asks once when a large chunk produces no entities.
//! Every wrapper action appears in the capture report.

use std::cell::RefCell;

use context_graph::harness::{CaptureHarnessConfig, run_capture};
use context_graph::ingest::{
    CaptureRequest, ConversationExtractor, DecisionRelationRequest, DecisionRelationSeed,
    EntitySeed, ExtractionPass, ExtractionRequest, StructuredHarvest,
};
use context_graph::{Result as IngestResult, StoreConfig};

use super::fake_extractor::open_store_with_lexical_embedder;

fn fresh_store() -> (tempfile::TempDir, context_graph::Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

/// A big decision-and-entity-rich transcript (≥ 8KB) on which an all-empty entity harvest is
/// SUSPICIOUS — the lazy-empty shape, not a plausible reading.
fn big_transcript() -> String {
    let mut text = String::from(
        "Goal: make the correctness bench trustworthy. Decision: the scoring thresholds are \
         frozen before measurement. The files scoring.rs and gold.rs in the cg-bench crate \
         carry the implementation; drain_context_queue is the production path.\n",
    );
    while text.len() < 9_000 {
        text.push_str(
            "Further discussion of cg-bench, scoring.rs, gold.rs, and the drain continues with \
             concrete detail about entities and decisions in this working session.\n",
        );
    }
    text
}

/// Lazy-then-diligent double: the first entities-bearing ask returns empty entities; the
/// re-ask returns the entities that
/// were genuinely in the transcript. Counts every pass invocation.
struct LazyOnceExtractor {
    full_calls: RefCell<usize>,
    entity_reasks: RefCell<usize>,
}

impl LazyOnceExtractor {
    fn new() -> Self {
        Self {
            full_calls: RefCell::new(0),
            entity_reasks: RefCell::new(0),
        }
    }
}

impl ConversationExtractor for LazyOnceExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> IngestResult<StructuredHarvest> {
        self.extract_pass(request, ExtractionPass::Full)
    }

    fn extract_pass(
        &self,
        _request: ExtractionRequest<'_>,
        pass: ExtractionPass,
    ) -> IngestResult<StructuredHarvest> {
        match pass {
            ExtractionPass::Full => {
                *self.full_calls.borrow_mut() += 1;
                // Lazy-empty: goals/decisions present, entities EMPTY on an entity-rich chunk.
                Ok(StructuredHarvest {
                    goals: vec![context_graph::ingest::GoalSeed {
                        key: "g1".to_string(),
                        text: "make the correctness bench trustworthy".to_string(),
                        ..context_graph::ingest::GoalSeed::default()
                    }],
                    decisions: vec![context_graph::ingest::DecisionSeed {
                        key: "d1".to_string(),
                        text: "the scoring thresholds are frozen before measurement".to_string(),
                        goal_key: Some("g1".to_string()),
                        ..context_graph::ingest::DecisionSeed::default()
                    }],
                    entities: vec![],
                    ..StructuredHarvest::default()
                })
            }
            ExtractionPass::EntitiesOnly => {
                *self.entity_reasks.borrow_mut() += 1;
                // The diligent read: the entities genuinely present in the transcript.
                Ok(StructuredHarvest {
                    entities: vec![
                        EntitySeed {
                            key: "e1".to_string(),
                            name: "scoring.rs".to_string(),
                            ..EntitySeed::default()
                        },
                        EntitySeed {
                            key: "e2".to_string(),
                            name: "cg-bench".to_string(),
                            ..EntitySeed::default()
                        },
                    ],
                    ..StructuredHarvest::default()
                })
            }
            _ => Ok(StructuredHarvest::default()),
        }
    }

    fn classify_decision_relation(
        &self,
        _request: DecisionRelationRequest<'_>,
    ) -> IngestResult<Option<DecisionRelationSeed>> {
        Ok(None)
    }
}

/// Persistently-lazy double: the re-ask ALSO returns empty. The wrapper must ask exactly once —
/// never loop chasing a model that will not produce.
struct AlwaysLazyExtractor {
    entity_reasks: RefCell<usize>,
}

impl ConversationExtractor for AlwaysLazyExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> IngestResult<StructuredHarvest> {
        self.extract_pass(request, ExtractionPass::Full)
    }

    fn extract_pass(
        &self,
        _request: ExtractionRequest<'_>,
        pass: ExtractionPass,
    ) -> IngestResult<StructuredHarvest> {
        if pass == ExtractionPass::EntitiesOnly {
            *self.entity_reasks.borrow_mut() += 1;
        }
        Ok(StructuredHarvest {
            goals: vec![context_graph::ingest::GoalSeed {
                key: "g1".to_string(),
                text: "make the correctness bench trustworthy".to_string(),
                ..context_graph::ingest::GoalSeed::default()
            }],
            ..StructuredHarvest::default()
        })
    }
}

#[test]
fn suspicious_empty_result_triggers_exactly_one_reask() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let (_dir, store) = fresh_store();
    let extractor = LazyOnceExtractor::new();
    let (_report, harness_report) = run_capture(
        &store,
        CaptureRequest {
            context_label: "reask-test".to_string(),
            transcript: big_transcript(),
            model_available: true,
        },
        &extractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("capture succeeds");

    assert_eq!(
        *extractor.entity_reasks.borrow(),
        1,
        "exactly one entities re-ask"
    );
    // The re-asked entities joined the harvest.
    assert!(
        harness_report
            .stage3_resolution_verdicts
            .iter()
            .any(|v| v.kind == context_graph::harness::ResolutionKind::Entity),
        "re-asked entities were captured"
    );
    // The wrapper action is VISIBLE — silent recovery reads as 'the model did fine'.
    assert!(
        harness_report
            .reliability_actions
            .iter()
            .any(|action| action.contains("reask")),
        "the re-ask is recorded on the harness report: {:?}",
        harness_report.reliability_actions
    );
}

#[test]
fn reask_happens_once_even_when_still_empty() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let (_dir, store) = fresh_store();
    let extractor = AlwaysLazyExtractor {
        entity_reasks: RefCell::new(0),
    };
    let (_report, harness_report) = run_capture(
        &store,
        CaptureRequest {
            context_label: "reask-once-test".to_string(),
            transcript: big_transcript(),
            model_available: true,
        },
        &extractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("capture succeeds");
    assert_eq!(
        *extractor.entity_reasks.borrow(),
        1,
        "one re-ask, never a loop, even when the model stays lazy"
    );
    assert!(
        harness_report
            .reliability_actions
            .iter()
            .any(|action| action.contains("reask")),
        "the attempted re-ask is still recorded"
    );
}

#[test]
fn small_or_populated_chunks_do_not_reask() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let (_dir, store) = fresh_store();
    // Small chunk (< 8KB) with empty entities: plausibly nothing there — no re-ask.
    let extractor = LazyOnceExtractor::new();
    let (_report, harness_report) = run_capture(
        &store,
        CaptureRequest {
            context_label: "no-reask-test".to_string(),
            transcript: "Goal: small note. Decision: nothing entity-shaped here.".to_string(),
            model_available: true,
        },
        &extractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("capture succeeds");
    assert_eq!(
        *extractor.entity_reasks.borrow(),
        0,
        "small chunks never re-ask"
    );
    assert!(
        harness_report.reliability_actions.is_empty(),
        "no wrapper action is invented when none ran"
    );
}
