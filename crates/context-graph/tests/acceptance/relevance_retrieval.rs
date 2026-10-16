//! Relevance retrieval acceptance tests.
//!
//! Four acceptance tests verifying that Stage-3 retrieval makes the right number of
//! judge calls with the right narrowed candidate set.
//!
//! Primary observables: the `CountingExtractor` spy records `judge_call_count()` and
//! `received_candidate_sets()` — these are process-observable doubles, not report fields.
//! Report fields provide secondary confirmation of candidate selection and model calls.

use std::collections::HashMap;

use context_graph::harness::{CaptureHarnessConfig, ResolutionKind, VerdictValue, run_capture};
use context_graph::ingest::{CaptureRequest, DecisionRelationClassification};
use context_graph::{ListEntityFilter, Store, StoreConfig, TxId};
use contextdb_core::Value as DbValue;

use super::action_test_extractors::{
    CountingExtractor, DecisionRelationExtractor, SpanFaithfulExtractor,
};
use super::fake_extractor::open_store_with_lexical_embedder;

/// Insert a learned skill row directly into `cg_learned_skills` (test seeding). The rule text is a
/// non-retype placeholder — fine here because these rows are only used as description-match
/// distractors that must never be selected in the first place.
fn seed_skill(store: &Store, id: &str, description: &str) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_learned_skills (id, description, rule, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("description".to_string(), DbValue::Text(description.to_string())),
            ("rule".to_string(), DbValue::Text(format!("rule for {id}"))),
            ("provenance".to_string(), DbValue::Text("human-correction".to_string())),
            ("active".to_string(), DbValue::Bool(true)),
            ("held_for_review".to_string(), DbValue::Bool(false)),
            ("fire_right".to_string(), DbValue::Int64(0)),
            ("fire_wrong".to_string(), DbValue::Int64(0)),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("seed learned skill");
}

/// Insert a learned skill row that is meant to be SELECTED (unlike `seed_skill`'s distractors): a
/// real, derivable retype rule naming `entity_name`, so the fold-seam withhold contract (an
/// effect-less row never applies) does not withhold it. `entity_name` never appears as a harvested
/// entity in this file's fixtures, so the retype never actually fires — it is a scope-matching,
/// selectable, no-op skill, exactly what these relevance tests need to prove selection narrows to
/// the relevant few by description, not by which skills happen to change a value.
fn seed_retype_skill(store: &Store, id: &str, description: &str, entity_name: &str) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_learned_skills (id, description, rule, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("description".to_string(), DbValue::Text(description.to_string())),
            (
                "rule".to_string(),
                DbValue::Text(format!(
                    "when an entity named {entity_name} is typed Service → Cache"
                )),
            ),
            ("provenance".to_string(), DbValue::Text("human-correction".to_string())),
            ("active".to_string(), DbValue::Bool(true)),
            ("held_for_review".to_string(), DbValue::Bool(false)),
            ("fire_right".to_string(), DbValue::Int64(0)),
            ("fire_wrong".to_string(), DbValue::Int64(0)),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("seed retype-capable learned skill");
}

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

#[test]
fn zero_candidates_makes_no_judge_call() {
    let (_dir, store) = fresh_store();

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

    // A fresh store has no prior candidates, so the judge must not be called.
    assert_eq!(
        counter.judge_call_count(),
        0,
        "fresh store has no prior candidates — judge must not be called; \
         observed {} judge calls with zero candidates",
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
#[test]
fn candidates_present_makes_exactly_one_small_judge_call_per_item() {
    let (_dir, store) = fresh_store();

    // Seed a population of 20 prior decisions. A FEW share the lexical core of the new decision
    // ("graphql gateway") — the KNOWN-NEAR identity set — and the rest are FAR/irrelevant rows on
    // unrelated topics. Retrieval must narrow to exactly the near set and exclude every far row.
    let near_texts = [
        "adopt the graphql gateway for the public api",
        "tune the graphql gateway cache for the public api",
    ];
    for (i, text) in near_texts.iter().enumerate() {
        let _ = run_capture(
            &store,
            request(
                "e-candidates-context",
                &format!("Goal GN{i}: gateway intention\nDecision DN{i}: {text} serves GN{i}"),
            ),
            &SpanFaithfulExtractor,
            &default_config(),
        )
        .expect("seed near neighbor");
    }
    // 18 FAR rows on topics with no lexical overlap with "graphql gateway".
    let far_topics = [
        "rotate the tls certificate",
        "schedule nightly database backups",
        "raise the pod memory limit",
        "migrate logging to opentelemetry",
        "pin the rust toolchain version",
        "add a smoke test to the pipeline",
        "document the on-call rotation",
        "archive the legacy billing module",
        "throttle the email digest job",
        "shard the search index by tenant",
        "deprecate the old metrics dashboard",
        "encrypt the audit log at rest",
        "cache the avatar thumbnails",
        "retry failed webhook deliveries",
        "compress the cold storage tier",
        "rename the staging namespace",
        "batch the analytics ingestion",
        "vendor the protobuf definitions",
    ];
    for (i, text) in far_topics.iter().enumerate() {
        let _ = run_capture(
            &store,
            request(
                "e-candidates-context",
                &format!("Goal GF{i}: unrelated intention\nDecision DF{i}: {text} serves GF{i}"),
            ),
            &SpanFaithfulExtractor,
            &default_config(),
        )
        .expect("seed far neighbor");
    }

    // Capture a new decision lexically close to the near set, far from everything else.
    let counter = CountingExtractor::new(SpanFaithfulExtractor);
    let (_report, harness_report) = run_capture(
        &store,
        request(
            "e-candidates-context",
            "Goal G_new: gateway latency\nDecision D_new: extend the graphql gateway for the public api serves G_new",
        ),
        &counter,
        &default_config(),
    )
    .expect("narrowed capture");

    // The judge must have run on the dedicated decision-relation seam.
    assert!(
        counter.judge_call_count() >= 1,
        "at least one decision-relation judge call must have been made when prior decisions exist; \
         spy observed {} judge calls",
        counter.judge_call_count()
    );

    let received = counter.received_candidate_sets();
    assert!(
        !received.is_empty(),
        "received_candidate_sets must be non-empty when the judge was called"
    );

    // LOAD-BEARING: assert the received set BY IDENTITY. The near texts must be PRESENT and EVERY
    // far text must be ABSENT — a positive-exclusion check, not just a size bound. A stub that
    // passes the whole store (far rows included) fails; one that returns the right size but wrong
    // members fails.
    let last_set = received.last().expect("at least one candidate set");
    for near in near_texts.iter() {
        assert!(
            last_set.iter().any(|c| c.contains("graphql gateway"))
                && last_set.iter().any(|c| c == near),
            "the known-near candidate {near:?} must be present in the judge's candidate set by \
             identity; got: {last_set:?}"
        );
    }
    for far in far_topics.iter() {
        assert!(
            !last_set.iter().any(|c| c == far),
            "the far/irrelevant candidate {far:?} must be ABSENT from the judge's candidate set \
             (positive exclusion); got: {last_set:?}"
        );
    }

    // Report mirrors the spy.
    assert!(
        harness_report.judge_call_count >= 1,
        "harness report: at least one judge call must have been made"
    );
    assert!(
        !harness_report.judge_candidate_set.is_empty(),
        "when candidates exist, judge_candidate_set must be non-empty (narrowed neighbors)"
    );
    for far in far_topics.iter() {
        assert!(
            !harness_report.judge_candidate_set.iter().any(|c| c == far),
            "harness report judge_candidate_set must exclude the far candidate {far:?}; got: {:?}",
            harness_report.judge_candidate_set
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3 (retrieval): The retrieval+judge `unsure` outcome keeps items separate
// (false-merge-worse-than-miss). The contract: when the judge returns unsure
// (None from classify_decision_relation), the harness must NOT silently merge two distinct items.
#[test]
fn relevance_retrieval_unsure_keeps_separate() {
    let (_dir, store) = fresh_store();

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

    // Primary load-bearing spy assertion: the DEDICATED entity-identity judge seam must be the one
    // consulted to decide entity separateness — NOT classify_decision_relation, which only
    // classifies decisions. Entity identity is its own question, so it must run on its own seam.
    assert!(
        counter.entity_identity_call_count() >= 1,
        "the dedicated entity-identity judge must be consulted at least once for the resolved \
         entity; spy observed {} entity-identity calls",
        counter.entity_identity_call_count()
    );
    // The decision-relation classifier must NOT have been used to decide entity identity (routing
    // entity identity through classify_decision_relation is the wrong seam — its counter stays 0).
    assert_eq!(
        counter.judge_call_count(),
        0,
        "entity identity must NOT be routed through classify_decision_relation (a decision judge); \
         that counter must stay 0, but observed {}",
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

    // False-merge guard: an any-relation (related-but-not-same) model answer must NOT collapse to
    // Same — only an explicit exact-duplicate (Same) verdict merges. Drive a SECOND, separately
    // named service entity whose identity judge returns a generic relation (Refines) rather than
    // exact-duplicate, and assert it is kept separate (not merged into a prior).
    let any_relation = DecisionRelationExtractor {
        relation: DecisionRelationClassification::RefinesExisting,
        ..Default::default()
    };
    let (_third_report, hr3) = run_capture(
        &store,
        request(
            "unsure-retrieval-context",
            "Entity E_third: auth-service type=service",
        ),
        &any_relation,
        &default_config(),
    )
    .expect("third entity capture with any-relation judge");
    let any_relation_merged = hr3
        .stage3_resolution_verdicts
        .iter()
        .any(|v| v.kind == ResolutionKind::Entity && v.verdict == VerdictValue::Same);
    assert!(
        !any_relation_merged,
        "an any-relation (Refines) entity-identity answer must NOT collapse to Same; \
         only an explicit exact-duplicate verdict may merge entities"
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

    // Structural fill makes no model calls.
    assert_eq!(
        harness_report.stage2_model_calls, 0,
        "Stage-2 structural fill must be model-free"
    );
}

// ---------------------------------------------------------------------------
// Test 4 (retrieval): Skill selection surfaces ONLY the learned skills whose DESCRIPTION matches
// the captured chunk — exactly the matching few by identity, not every active skill and not a
// different set.
//
// Seed ~20 learned skills with varying descriptions, only 2 of which match the chunk. A real
// capture must record firings for exactly those 2 skills by id. Application reporting additionally
// requires a persisted effect.
// ---------------------------------------------------------------------------
#[test]
fn skill_selection_retrieves_the_relevant_few_not_the_whole_corpus() {
    let (_dir, store) = fresh_store();

    // Seed 20 learned skills with VARYING descriptions. Only the two "redis cache" skills match the
    // captured chunk below; the other 18 are on unrelated topics.
    let matching = ["sk-redis-ttl", "sk-redis-evict"];
    seed_retype_skill(
        &store,
        matching[0],
        "tune the redis cache TTL for hot keys",
        "redis",
    );
    seed_retype_skill(
        &store,
        matching[1],
        "set the redis cache eviction policy",
        "redis",
    );
    let unrelated_topics = [
        "rotate the tls certificate",
        "schedule nightly database backups",
        "raise the pod memory limit",
        "migrate logging to opentelemetry",
        "pin the rust toolchain version",
        "add a smoke test to the pipeline",
        "document the on-call rotation",
        "archive the legacy billing module",
        "throttle the email digest job",
        "shard the search index by tenant",
        "deprecate the old metrics dashboard",
        "encrypt the audit log at rest",
        "compress the cold storage tier",
        "rename the staging namespace",
        "batch the analytics ingestion",
        "vendor the protobuf definitions",
        "wire the feature flag service",
        "back up the secrets vault",
    ];
    for (i, topic) in unrelated_topics.iter().enumerate() {
        seed_skill(&store, &format!("sk-other-{i}"), topic);
    }

    // Capture a chunk matching ONLY the two redis-cache skills.
    let (_report, _harness_report) = run_capture(
        &store,
        request(
            "skill-selection-context",
            "Goal G1: speed up reads\nDecision D1: tune the redis cache for hot keys serves G1",
        ),
        &SpanFaithfulExtractor,
        &default_config(),
    )
    .expect("skill-selection capture");

    // EXACTLY the two matching skills (by id) must be SELECTED — not 20 (selecting all active
    // skills fails), and not a different 2 (selecting by anything but description match fails).
    // Selection is proven by the persisted firing trace: applied_skill_ids reports only skills
    // whose effect actually landed (receipt honesty), which is not this test's subject.
    let db = store.sync_database();
    let mut fired: Vec<String> = db
        .scan("cg_skill_firing_trace", db.snapshot())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| match row.values.get("skill_id") {
            Some(DbValue::Text(value)) => Some(value.clone()),
            _ => None,
        })
        .collect();
    fired.sort();
    fired.dedup();
    let mut expected: Vec<String> = matching.iter().map(|s| s.to_string()).collect();
    expected.sort();
    assert_eq!(
        fired, expected,
        "skill selection must fire EXACTLY the two description-matching skills by id ({expected:?}); \
         got {fired:?}"
    );
}
