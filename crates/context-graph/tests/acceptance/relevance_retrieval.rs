//! Relevance retrieval acceptance tests.
//!
//! Four acceptance tests verifying that Stage-3 retrieval makes the right number of
//! judge calls with the right narrowed candidate set. Every assertion fails on a placeholder
//! value, never on a compile error or panic.
//!
//! Primary observables: the `CountingExtractor` spy records `judge_call_count()` and
//! `received_candidate_sets()` — these are process-observable doubles, not report fields.
//! Placeholder values (from `CaptureHarnessReport` defaults are secondary confirmations):
//!   - `judge_call_count: 0` (stays 0 even when candidates exist)
//!   - `judge_candidate_set: []` (always empty)
//!   - `stage2_model_calls: 1` (placeholder; correct is 0)

use context_graph::harness::{CaptureHarnessConfig, VerdictValue, run_capture};
use context_graph::ingest::CaptureRequest;
use context_graph::{ListEntityFilter, Store, StoreConfig};

use super::fake_extractor::open_store_with_lexical_embedder;
use super::refactor_extractor::{CountingExtractor, SpanFaithfulExtractor};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn default_config() -> CaptureHarnessConfig {
    CaptureHarnessConfig::default()
}

fn request(label: &str, transcript: &str) -> CaptureRequest {
    CaptureRequest {
        context_label: label.to_string(),
        transcript: transcript.to_string(),
        model_available: true,
    }
}

// ---------------------------------------------------------------------------
// Test 1 (retrieval): In a fresh store (no prior neighbors), Stage-3 must issue zero
// judge calls — cost floor. The placeholder run_capture calls the judge for every
// harvested decision even when the store has zero prior candidates.
//
// Primary load-bearing observable: `counter.judge_call_count() == 0` fails because the
// placeholder calls the judge once per decision in the harvest, even with zero store candidates.
// The transcript includes a decision so the placeholder pipeline has a decision to iterate over.
// ---------------------------------------------------------------------------
#[test]
fn zero_candidates_makes_no_judge_call() {
    let (_dir, store) = fresh_store();

    // Transcript includes a decision so the placeholder pipeline's per-decision judge loop fires.
    // Correct impl: zero prior candidates → zero judge calls.
    // Placeholder: calls judge once per harvested decision even with zero candidates.
    let counter = CountingExtractor::new(SpanFaithfulExtractor);
    let (_report, harness_report) = run_capture(
        &store,
        request(
            "fresh-e-context",
            "Goal G1: first capture in empty store\nDecision D1: choose option_a serves G1",
        ),
        &counter,
        &default_config(),
    )
    .expect("fresh-store capture");

    // Primary load-bearing spy assertion: fresh store has no prior candidates —
    // the judge must not be called. Placeholder calls it once per decision
    // even with zero candidates → spy observes 1 call → assertion fails.
    assert_eq!(
        counter.judge_call_count(),
        0,
        "fresh store has no prior candidates — judge must not be called; \
         wrong run_capture called the judge {} time(s) even with zero candidates",
        counter.judge_call_count()
    );

    // Spy: received_candidate_sets must be empty (zero judge calls → zero recorded sets).
    let received = counter.received_candidate_sets();
    assert!(
        received.is_empty(),
        "fresh store: received_candidate_sets must be empty (no judge calls); \
         spy recorded {} call(s) with sets: {:?}",
        received.len(),
        received
    );

    // Report field: no candidates → empty judge_candidate_set.
    assert!(
        harness_report.judge_candidate_set.is_empty(),
        "fresh store has no candidates — judge_candidate_set must be empty; got {:?}",
        harness_report.judge_candidate_set
    );

    // Secondary: Stage-2 structural fill uses zero model calls.
    assert_eq!(
        harness_report.stage2_model_calls, 0,
        "Stage-2 structural fill must use zero model calls even in a fresh store"
    );

    // Spy: extraction ran exactly once (one Stage-1 call for a single-chunk capture).
    assert_eq!(
        counter.extraction_call_count(),
        1,
        "extraction must be called exactly once for a single capture; \
         got {} extraction calls",
        counter.extraction_call_count()
    );
}

// ---------------------------------------------------------------------------
// Test 2 (retrieval): When candidates exist, the judge is called once per resolved item
// with a NARROWED candidate set — not the whole store.
//
// Primary observable: `counter.received_candidate_sets()` — the spy records what the
// judge actually received. The set contents must be the near few, not all 20.
// Placeholder: judge is never called via classify_decision_relation → received_candidate_sets
// is empty → assertion on received.len() fails.
// ---------------------------------------------------------------------------
#[test]
fn candidates_present_makes_exactly_one_small_judge_call_per_item() {
    let (_dir, store) = fresh_store();

    // Seed 20 decisions so the retrieval block has a population to narrow against.
    // Each transcript uses \n so SpanFaithfulExtractor sees Goal and Decision on separate lines.
    for i in 0..20 {
        let _ = run_capture(
            &store,
            request(
                "e-candidates-context",
                &format!(
                    "Goal G{i}: seed intention #{i} for neighbor population\nDecision D{i}: use option_{i} serves G{i}"
                ),
            ),
            &SpanFaithfulExtractor,
            &default_config(),
        )
        .expect("seed neighbor");
    }

    // Capture a new decision that has prior neighbors to judge against.
    let counter = CountingExtractor::new(SpanFaithfulExtractor);
    let (_report, harness_report) = run_capture(
        &store,
        request(
            "e-candidates-context",
            "Goal G_new: new intention\nDecision D_new: use option_0 — relates to prior serves G_new",
        ),
        &counter,
        &default_config(),
    )
    .expect("narrowed capture");

    // Primary spy observable: judge must have been called at least once.
    // Placeholder: classify_decision_relation is never forwarded → judge_call_count == 0.
    assert!(
        counter.judge_call_count() >= 1,
        "at least one judge call must have been made when prior decisions exist; \
         spy observed {} judge calls — wrong stub never calls classify_decision_relation",
        counter.judge_call_count()
    );

    // Spy: received_candidate_sets must be non-empty.
    let received = counter.received_candidate_sets();
    assert!(
        !received.is_empty(),
        "received_candidate_sets must be non-empty when the judge was called; \
         spy recorded 0 sets — wrong stub bypasses judge"
    );

    // Spy: each call must have received a NARROWED set (fewer than all 20 seeded decisions).
    for (i, candidate_set) in received.iter().enumerate() {
        assert!(
            candidate_set.len() < 20,
            "judge call #{} received {} candidates — must be narrowed (< 20, the full corpus); \
             wrong stub would pass all or none",
            i,
            candidate_set.len()
        );
        assert!(
            !candidate_set.is_empty(),
            "judge call #{} received an empty candidate set — retrieval must narrow to \
             the relevant few, not collapse to zero",
            i
        );
    }

    // Report confirms: at least one judge call and non-empty candidate set.
    assert!(
        harness_report.judge_call_count >= 1,
        "harness report: at least one judge call must have been made; \
         wrong stub always returns judge_call_count=0"
    );
    assert!(
        !harness_report.judge_candidate_set.is_empty(),
        "when candidates exist, judge_candidate_set must be non-empty (narrowed neighbors); \
         wrong stub always returns empty judge_candidate_set"
    );
    assert!(
        harness_report.judge_candidate_set.len() <= 10,
        "judge_candidate_set must be a narrowed few (≤10), not the entire corpus (20); \
         got {} candidates",
        harness_report.judge_candidate_set.len()
    );
}

// ---------------------------------------------------------------------------
// Test 3 (retrieval): The retrieval+judge `unsure` outcome keeps items separate
// (false-merge-worse-than-miss). The contract: when the judge returns unsure
// (None from classify_decision_relation), the harness must NOT silently merge two distinct items.
//
// Primary load-bearing spy assertion: the judge must have been consulted (called at least once)
// to decide separateness — `counter.judge_call_count() >= 1`.
// Placeholder run_capture: the entity-identity judge seam does not exist (only
// `classify_decision_relation` exists, which is per-decision, not per-entity).
// With entity-only transcripts (no decisions), placeholder behavior 2 never calls the judge →
// `counter.judge_call_count() == 0` → assertion `>= 1` FAILS.
//
// NOTE for impl: the correct impl needs a dedicated entity-identity judge seam invoked
// during Stage-3 entity resolution; `classify_decision_relation` is the only available
// judge hook and is not called for entity-only harvests in the placeholder pipeline.
// ---------------------------------------------------------------------------
#[test]
fn relevance_retrieval_unsure_keeps_separate() {
    let (_dir, store) = fresh_store();

    // Capture a first entity to populate the neighbor space.
    // Entity-only transcript: no decisions, so the placeholder's per-decision judge loop
    // never fires, leaving judge_call_count at 0.
    let _ = run_capture(
        &store,
        request(
            "unsure-retrieval-context",
            "Entity E_first: auth-service type=service",
        ),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("first entity capture");

    // Capture a second entity with the same name — Stage-3 should judge it as unsure.
    // CountingExtractor wrapping SpanFaithfulExtractor: classify_decision_relation returns
    // None (unsure) via SpanFaithfulExtractor's default impl. This is the "unsure" judge double.
    let counter = CountingExtractor::new(SpanFaithfulExtractor);
    let (_report, harness_report) = run_capture(
        &store,
        request(
            "unsure-retrieval-context",
            "Entity E_second: auth-service type=service",
        ),
        &counter,
        &default_config(),
    )
    .expect("second entity capture");

    // Primary load-bearing spy assertion: the judge MUST be consulted to decide entity
    // separateness. Placeholder run_capture's per-decision judge loop never fires for entity-only
    // harvests → counter.judge_call_count() == 0 → this assertion FAILS.
    // Correct impl: calls the entity-identity judge, which returns unsure (None) → separate.
    assert!(
        counter.judge_call_count() >= 1,
        "judge must be consulted at least once to decide entity separateness; \
         wrong run_capture does not call the judge for entity-only harvests — \
         spy observed {} judge calls",
        counter.judge_call_count()
    );

    // No silent Same-merge verdict: an unsure judge result must NOT produce VerdictValue::Same.
    let silently_merged = harness_report
        .stage3_resolution_verdicts
        .iter()
        .any(|v| v.verdict == VerdictValue::Same);
    assert!(
        !silently_merged,
        "unsure judge verdict must not produce a silent VerdictValue::Same merge; \
         if Same appears, items were merged without clear evidence"
    );

    // Store assertion: entities must be in the store (pipeline ran).
    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|c| c.name == "unsure-retrieval-context")
        .map(|c| c.id);
    let entities = store
        .list_entities(ListEntityFilter {
            context_id,
            name_contains: Some("auth-service".to_string()),
            ..Default::default()
        })
        .expect("list entities");
    assert!(
        !entities.is_empty(),
        "at least one entity must be in the store after two captures"
    );

    // Secondary: structural fill uses zero model calls (now correct in run_capture).
    assert_eq!(
        harness_report.stage2_model_calls, 0,
        "Stage-2 structural fill must be model-free"
    );
}

// ---------------------------------------------------------------------------
// Test 4 (retrieval): Skill selection retrieves only the skills whose description
// matches the chunk — not every learned skill in the store.
//
// Primary spy observable: when neighbors exist, the judge must be called with a
// non-empty, narrowed candidate set via `received_candidate_sets()`.
// Placeholder: judge is never invoked → received_candidate_sets is empty → fails.
// ---------------------------------------------------------------------------
#[test]
fn skill_selection_retrieves_the_relevant_few_not_the_whole_corpus() {
    let (_dir, store) = fresh_store();

    // Seed 20 decisions so there are many potential neighbors.
    // Each transcript uses \n so SpanFaithfulExtractor sees Goal and Decision on separate lines.
    for i in 0..20 {
        let _ = run_capture(
            &store,
            request(
                "skill-selection-context",
                &format!(
                    "Goal G{i}: seed skill context intention #{i}\nDecision D{i}: use strategy_{i} serves G{i}"
                ),
            ),
            &SpanFaithfulExtractor,
            &default_config(),
        )
        .expect("seed skill context");
    }

    // Capture a decision that overlaps with only a few of the seeds.
    let counter = CountingExtractor::new(SpanFaithfulExtractor);
    let (_report, harness_report) = run_capture(
        &store,
        request(
            "skill-selection-context",
            "Goal G_match: matching goal\nDecision D_match: use strategy_0 serves G_match",
        ),
        &counter,
        &default_config(),
    )
    .expect("skill-selection capture");

    // Primary spy observable: the judge must have been called with a narrowed set.
    // Placeholder: classify_decision_relation is never delegated → judge_call_count == 0.
    let received = counter.received_candidate_sets();
    assert!(
        !received.is_empty(),
        "when prior neighbors exist, judge must be called and received_candidate_sets \
         must be non-empty; wrong stub never calls classify_decision_relation → empty"
    );

    // Spy: each received set must be narrowed (far fewer than all 20 seeded decisions).
    for (i, candidate_set) in received.iter().enumerate() {
        assert!(
            candidate_set.len() <= 5,
            "judge call #{} received {} candidates — skill selection must narrow to \
             the relevant few (≤5 of 20); wrong stub would pass all or none",
            i,
            candidate_set.len()
        );
        assert!(
            !candidate_set.is_empty(),
            "judge call #{} received an empty set — retrieval must select the relevant few, \
             not collapse to zero when neighbors exist",
            i
        );
    }

    // Report fields confirm the spy's observations.
    assert!(
        !harness_report.judge_candidate_set.is_empty(),
        "when prior neighbors exist, judge_candidate_set must be non-empty (the relevant few); \
         wrong stub always returns empty judge_candidate_set"
    );
    assert!(
        harness_report.judge_candidate_set.len() <= 5,
        "skill selection must retrieve only the relevant few (≤5 candidates); \
         wrong stub returns judge_candidate_set={:?} (empty, but the correct impl \
         returns a non-empty narrowed set which must still be small)",
        harness_report.judge_candidate_set
    );
}
