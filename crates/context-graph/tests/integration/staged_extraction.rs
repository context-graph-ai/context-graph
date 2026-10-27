//! Focused passes preserve full-pass findings and session goals reconcile additively.

use std::cell::RefCell;
use std::collections::BTreeMap;

use context_graph::harness::{CaptureHarnessConfig, ResolutionKind, run_capture};
use context_graph::ingest::{
    CaptureRequest, ConversationExtractor, DecisionRelationClassification, DecisionRelationRequest,
    DecisionRelationSeed, DecisionSeed, EntitySeed, ExtractionPass, ExtractionRequest, GoalSeed,
    StructuredHarvest, reconcile_session_goals,
};
use context_graph::{DiscoveryFilter, Result as IngestResult, StoreConfig};

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

fn goal(key: &str, text: &str) -> GoalSeed {
    GoalSeed {
        key: key.to_string(),
        text: text.to_string(),
        ..GoalSeed::default()
    }
}

fn decision(key: &str, text: &str, goal_key: Option<&str>) -> DecisionSeed {
    DecisionSeed {
        key: key.to_string(),
        text: text.to_string(),
        goal_key: goal_key.map(str::to_string),
        ..DecisionSeed::default()
    }
}

fn entity(key: &str, name: &str) -> EntitySeed {
    EntitySeed {
        key: key.to_string(),
        name: name.to_string(),
        ..EntitySeed::default()
    }
}

/// Deterministic extractor with per-pass harvests and a call log. The focused entity pass
/// includes entities absent from the full pass; every returned item matches the input transcript.
struct StagedSpyExtractor {
    calls: RefCell<Vec<ExtractionPass>>,
    /// Per-pass harvest builders keyed by pass; None = return empty harvest for that pass.
    harvests: BTreeMap<&'static str, StructuredHarvest>,
}

impl StagedSpyExtractor {
    fn new(harvests: BTreeMap<&'static str, StructuredHarvest>) -> Self {
        Self {
            calls: RefCell::new(Vec::new()),
            harvests,
        }
    }

    fn pass_key(pass: ExtractionPass) -> &'static str {
        match pass {
            ExtractionPass::Full => "full",
            ExtractionPass::EntitiesOnly => "entities",
            ExtractionPass::DecisionsOnly => "decisions",
            ExtractionPass::SessionIntention => "session",
        }
    }
}

impl ConversationExtractor for StagedSpyExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> IngestResult<StructuredHarvest> {
        self.extract_pass(request, ExtractionPass::Full)
    }

    fn extract_pass(
        &self,
        _request: ExtractionRequest<'_>,
        pass: ExtractionPass,
    ) -> IngestResult<StructuredHarvest> {
        self.calls.borrow_mut().push(pass);
        Ok(self
            .harvests
            .get(Self::pass_key(pass))
            .cloned()
            .unwrap_or_default())
    }

    fn classify_decision_relation(
        &self,
        _request: DecisionRelationRequest<'_>,
    ) -> IngestResult<Option<DecisionRelationSeed>> {
        Ok(None)
    }
}

const STAGED_TRANSCRIPT: &str = "Goal: ship the correctness bench so the local model is \
measurable. Decision: drive the production drain path from the bench, serves that goal. \
Decision: pin the scoring thresholds before any measurement. Entities involved: cg-bench, \
scoring.rs, drain_context_queue.";

fn full_harvest() -> StructuredHarvest {
    StructuredHarvest {
        goals: vec![goal(
            "g1",
            "ship the correctness bench so the local model is measurable",
        )],
        decisions: vec![decision(
            "d1",
            "drive the production drain path from the bench",
            Some("g1"),
        )],
        entities: vec![entity("e1", "cg-bench")],
        ..StructuredHarvest::default()
    }
}

fn entities_pass_harvest() -> StructuredHarvest {
    StructuredHarvest {
        entities: vec![
            entity("e2", "scoring.rs"),
            entity("e3", "drain_context_queue"),
        ],
        ..StructuredHarvest::default()
    }
}

fn decisions_pass_harvest() -> StructuredHarvest {
    StructuredHarvest {
        decisions: vec![
            // The same decision re-found (must dedup) plus one the full pass missed.
            decision(
                "d1b",
                "drive the production drain path from the bench",
                Some("g1"),
            ),
            decision(
                "d2",
                "pin the scoring thresholds before any measurement",
                Some("g1"),
            ),
        ],
        ..StructuredHarvest::default()
    }
}

#[test]
fn staged_passes_execute_on_capture_when_config_enabled() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let (_dir, store) = fresh_store();
    let harvests = BTreeMap::from([
        ("full", full_harvest()),
        ("entities", entities_pass_harvest()),
        ("decisions", decisions_pass_harvest()),
    ]);
    let spy = StagedSpyExtractor::new(harvests);
    let config = CaptureHarnessConfig {
        staged_extraction: true,
        ..CaptureHarnessConfig::default()
    };
    let (report, harness_report) = run_capture(
        &store,
        CaptureRequest {
            context_label: "staged-test".to_string(),
            transcript: STAGED_TRANSCRIPT.to_string(),
            model_available: true,
        },
        &spy,
        &config,
    )
    .expect("staged capture succeeds");

    let calls = spy.calls.borrow();
    assert!(calls.contains(&ExtractionPass::Full), "full pass runs");
    assert!(
        calls.contains(&ExtractionPass::EntitiesOnly),
        "entities pass runs when staged_extraction is on; calls: {calls:?}"
    );
    assert!(
        calls.contains(&ExtractionPass::DecisionsOnly),
        "decisions pass runs"
    );
    // Union: full's entity + the 2 the focused pass recovered; both decisions, deduped.
    let entity_resolutions = harness_report
        .stage3_resolution_verdicts
        .iter()
        .filter(|v| v.kind == ResolutionKind::Entity)
        .count();
    assert_eq!(
        entity_resolutions, 3,
        "focused-pass entities join the harvest"
    );
    assert_eq!(
        report.total_harvest.decisions, 2,
        "duplicate decision collapses; new one joins"
    );
}

#[test]
fn default_config_runs_single_full_pass_only() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let (_dir, store) = fresh_store();
    let spy = StagedSpyExtractor::new(BTreeMap::from([("full", full_harvest())]));
    let config = CaptureHarnessConfig::default();
    run_capture(
        &store,
        CaptureRequest {
            context_label: "unstaged-test".to_string(),
            transcript: STAGED_TRANSCRIPT.to_string(),
            model_available: true,
        },
        &spy,
        &config,
    )
    .expect("default capture succeeds");
    let calls = spy.calls.borrow();
    assert!(
        !calls.contains(&ExtractionPass::EntitiesOnly)
            && !calls.contains(&ExtractionPass::DecisionsOnly),
        "staging is opt-in config; default runs the full pass only, calls: {calls:?}"
    );
}

#[test]
fn staged_zero_with_full_nonzero_keeps_full_result() {
    let _alongside = crate::runs_alone::alongside_siblings();
    // An empty focused pass never erases items found by the full pass.
    let (_dir, store) = fresh_store();
    let harvests = BTreeMap::from([
        ("full", full_harvest()),
        // entities + decisions passes both return EMPTY.
    ]);
    let spy = StagedSpyExtractor::new(harvests);
    let config = CaptureHarnessConfig {
        staged_extraction: true,
        ..CaptureHarnessConfig::default()
    };
    let (report, harness_report) = run_capture(
        &store,
        CaptureRequest {
            context_label: "consistency-test".to_string(),
            transcript: STAGED_TRANSCRIPT.to_string(),
            model_available: true,
        },
        &spy,
        &config,
    )
    .expect("capture succeeds");
    // The staged passes must have actually RUN and returned empty — otherwise this test would
    // pass vacuously on a build where staging does not exist at all.
    let calls = spy.calls.borrow();
    assert!(
        calls.contains(&ExtractionPass::EntitiesOnly)
            && calls.contains(&ExtractionPass::DecisionsOnly),
        "the focused passes ran and returned empty; calls: {calls:?}"
    );
    assert_eq!(
        report.total_harvest.decisions, 1,
        "full pass's decision survives an empty DEC pass"
    );
    let entity_resolutions = harness_report
        .stage3_resolution_verdicts
        .iter()
        .filter(|v| v.kind == ResolutionKind::Entity)
        .count();
    assert_eq!(
        entity_resolutions, 1,
        "full pass's entity survives an empty ENT pass"
    );
}

/// Identity double for reconciliation: goals with the same normalized head are the same goal.
struct SameGoalJudge;

impl ConversationExtractor for SameGoalJudge {
    fn extract(&self, _request: ExtractionRequest<'_>) -> IngestResult<StructuredHarvest> {
        Ok(StructuredHarvest::default())
    }

    fn classify_decision_relation(
        &self,
        request: DecisionRelationRequest<'_>,
    ) -> IngestResult<Option<DecisionRelationSeed>> {
        // Content-faithful judgment double: same-meaning variants of the bench goal are
        // duplicates; anything else is unrelated. Mirrors what a real model would answer for
        // these two specific texts.
        let candidate = request.candidate_decision.to_ascii_lowercase();
        let same = request
            .prior_decisions
            .iter()
            .find(|prior| {
                let prior_text = prior.text.to_ascii_lowercase();
                candidate.contains("correctness bench") && prior_text.contains("correctness bench")
            })
            .map(|prior| DecisionRelationSeed {
                prior_decision_id: prior.id.to_string(),
                classification: DecisionRelationClassification::ExactDuplicate,
            });
        Ok(same)
    }
}

#[test]
fn session_goal_reconciliation_merges_duplicate_goals_across_digests() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let (_dir, store) = fresh_store();
    let config = CaptureHarnessConfig::default();

    // Two chunks of one session, understood independently (the drain path's shape): each
    // produced a VARIANT of the same session goal with one decision attached.
    for (chunk, goal_text, decision_text) in [
        (
            "chunk one: Goal: ship the correctness bench. Decision: drive the production drain.",
            "ship the correctness bench",
            "drive the production drain",
        ),
        (
            "chunk two: Goal: deliver the correctness bench. Decision: pin scoring thresholds.",
            "deliver the correctness bench",
            "pin scoring thresholds",
        ),
    ] {
        let harvest = StructuredHarvest {
            goals: vec![goal("g1", goal_text)],
            decisions: vec![decision("d1", decision_text, Some("g1"))],
            ..StructuredHarvest::default()
        };
        let spy = StagedSpyExtractor::new(BTreeMap::from([("full", harvest)]));
        run_capture(
            &store,
            CaptureRequest {
                context_label: "reconcile-test".to_string(),
                transcript: chunk.to_string(),
                model_available: true,
            },
            &spy,
            &config,
        )
        .expect("chunk capture succeeds");
    }

    let summary = reconcile_session_goals(&store, "reconcile-test", &SameGoalJudge, &config)
        .expect("reconciliation runs");
    assert_eq!(
        summary.merged_goals, 1,
        "the two goal variants merge into one canonical"
    );

    // The canonical intention now serves BOTH decisions; the merge was additive (the variant
    // is superseded/absorbed, not deleted).
    let canonical = summary
        .canonical_intentions
        .first()
        .expect("a canonical intention is reported");
    let serving = store
        .by_intention(*canonical, DiscoveryFilter::default())
        .expect("by_intention reads");
    assert_eq!(
        serving.len(),
        2,
        "both chunks' decisions attach to the canonical session intention"
    );
}

#[test]
fn unattached_decision_is_linked_or_flagged_by_reconciliation() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let (_dir, store) = fresh_store();
    let config = CaptureHarnessConfig::default();

    // A decision with NO goal attachment and no plausible goal in the harvest: after capture +
    // reconciliation it must be FLAGGED (visible, reviewable) — never silently unattached.
    let harvest = StructuredHarvest {
        goals: vec![],
        decisions: vec![decision("d1", "adopt the two-phase import parse", None)],
        ..StructuredHarvest::default()
    };
    let spy = StagedSpyExtractor::new(BTreeMap::from([("full", harvest)]));
    run_capture(
        &store,
        CaptureRequest {
            context_label: "unattached-test".to_string(),
            transcript: "Decision: adopt the two-phase import parse.".to_string(),
            model_available: true,
        },
        &spy,
        &config,
    )
    .expect("capture succeeds");

    let summary = reconcile_session_goals(&store, "unattached-test", &SameGoalJudge, &config)
        .expect("reconciliation runs");
    assert_eq!(
        summary.flagged_unattached, 1,
        "an unattachable decision is surfaced for review, never silently unattached"
    );
}
