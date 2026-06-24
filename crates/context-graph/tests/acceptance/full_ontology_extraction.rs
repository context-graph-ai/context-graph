use context_graph::ingest::{CaptureRequest, record_conversation_with_extractor};
use context_graph::{ContextId, DecisionId, EntityType, ListEntityFilter, Store, StoreConfig};

use super::fake_extractor::open_store_with_lexical_embedder;
use super::refactor_extractor::{
    AgentIdExtractor, AllThirteenTypesExtractor, DecisionTypeExtractor,
    DecisionWithMetadataExtractor, NullConfidenceExtractor, ObservationLanesExtractor,
    OutcomeExtractor, RepositoryModuleExtractor,
};

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn capture_with<E: context_graph::ingest::ConversationExtractor>(
    store: &Store,
    context_label: &str,
    transcript: &str,
    extractor: &E,
) -> context_graph::ingest::CaptureReport {
    record_conversation_with_extractor(
        store,
        CaptureRequest {
            context_label: context_label.to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        extractor,
    )
    .expect("capture succeeded")
}

fn context_id_by_name(store: &Store, name: &str) -> Option<ContextId> {
    store
        .list_contexts()
        .ok()?
        .into_iter()
        .find(|c| c.name == name)
        .map(|c| c.id)
}

fn decision_id_from_stable(
    report: &context_graph::ingest::CaptureReport,
    key: &str,
) -> Option<DecisionId> {
    report
        .stable_ids
        .get(key)
        .and_then(|s| uuid::Uuid::parse_str(s).ok())
        .map(DecisionId::from)
}

// The placeholder From<ExtractedEntityType> maps Repository and Module to File.
// Assert stored entities carry Repository and Module, not File.
#[test]
fn repository_and_module_transcript_resolve_to_repository_and_module_not_file() {
    let (_dir, store) = fresh_store();
    capture_with(
        &store,
        "repo-module-context",
        "capturing repository and module",
        &RepositoryModuleExtractor,
    );

    let context_id = context_id_by_name(&store, "repo-module-context");
    let entities = store
        .list_entities(ListEntityFilter {
            context_id,
            ..Default::default()
        })
        .expect("list entities");

    let repo_entity = entities.iter().find(|e| e.name.contains("context-graph"));
    let module_entity = entities.iter().find(|e| e.name.contains("ingest"));

    // Placeholder: From<ExtractedEntityType> maps Repository→File, Module→File.
    assert_eq!(
        repo_entity.map(|e| e.entity_type),
        Some(EntityType::Repository),
        "Repository entity must be typed Repository, not File"
    );
    assert_eq!(
        module_entity.map(|e| e.entity_type),
        Some(EntityType::Module),
        "Module entity must be typed Module, not File"
    );
    assert!(
        entities
            .iter()
            .filter(|e| e.name.contains("context-graph") || e.name.contains("ingest"))
            .all(|e| e.entity_type != EntityType::File),
        "Neither Repository nor Module entity should collapse to File"
    );
}

// Assert each of the 13 entity types maps to its own variant per named seed.
// The placeholder implementation misroutes several to wrong variants.
#[test]
fn all_thirteen_entity_types_are_reachable_through_capture() {
    let (_dir, store) = fresh_store();
    capture_with(
        &store,
        "all-types-context",
        "all 13 entity types",
        &AllThirteenTypesExtractor,
    );

    let context_id = context_id_by_name(&store, "all-types-context");
    let entities = store
        .list_entities(ListEntityFilter {
            context_id,
            ..Default::default()
        })
        .expect("list entities");

    // Per-seed identity assertions: each name maps to exactly its expected type.
    let expected: Vec<(&str, EntityType)> = vec![
        ("auth service", EntityType::Service),
        ("Alice person", EntityType::Person),
        ("edge device", EntityType::Device),
        ("ingest module", EntityType::Module),
        ("context-graph repo", EntityType::Repository),
        ("app config", EntityType::Configuration),
        ("prod infra", EntityType::Infrastructure),
        ("us-east-1 location", EntityType::Location),
        ("src/main.rs file", EntityType::File),
        ("ci pipeline", EntityType::Pipeline),
        ("postgres database", EntityType::Database),
        ("platform team", EntityType::Team),
        ("spec document", EntityType::Document),
    ];

    assert_eq!(entities.len(), 13, "All 13 entity types must be stored");

    for (name_fragment, expected_type) in &expected {
        let found = entities.iter().find(|e| e.name.contains(name_fragment));
        assert!(
            found.is_some(),
            "Entity with name containing '{}' not found in store",
            name_fragment
        );
        assert_eq!(
            found.unwrap().entity_type,
            *expected_type,
            "Entity '{}' should have type {:?} but stored as {:?}",
            name_fragment,
            expected_type,
            found.unwrap().entity_type
        );
    }
}

// The placeholder harness stores decision_type="code_reference" unconditionally.
// Assert the stored decision carries the model-emitted "technology_choice".
// Look up by stable_id to avoid depending on the placeholder captured_decisions list.
#[test]
fn decision_type_comes_from_model_not_hardcoded_code_reference() {
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "decision-type-context",
        "database technology choice",
        &DecisionTypeExtractor,
    );

    // Get the decision from the store directly via stable_id.
    let decision_id = decision_id_from_stable(&report, "D1")
        .expect("D1 must be in stable_ids from DecisionTypeExtractor");
    let decision = store
        .get_decision(decision_id)
        .expect("get decision")
        .expect("decision exists");

    // Placeholder: stores "code_reference". Assert the model-emitted value is stored.
    assert_eq!(
        decision.decision_type, "technology_choice",
        "decision_type must be 'technology_choice' from model, not the hardcoded 'code_reference'"
    );
    assert_ne!(
        decision.decision_type, "code_reference",
        "decision_type must not be the hardcoded 'code_reference'"
    );
}

// The placeholder harness stores agent_id="cg-ingest" unconditionally.
// Assert the stored decision carries the model-emitted "senior-eng-alice".
#[test]
fn agent_id_comes_from_model_not_hardcoded_cg_ingest() {
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "agent-id-context",
        "agent attributed decision",
        &AgentIdExtractor,
    );

    let decision_id = decision_id_from_stable(&report, "D1")
        .expect("D1 must be in stable_ids from AgentIdExtractor");
    let decision = store
        .get_decision(decision_id)
        .expect("get decision")
        .expect("decision exists");

    // Placeholder: stores "cg-ingest". Assert the model-emitted value is stored.
    assert_eq!(
        decision.agent_id.as_deref(),
        Some("senior-eng-alice"),
        "agent_id must be 'senior-eng-alice' from model, not the hardcoded 'cg-ingest'"
    );
    assert_ne!(
        decision.agent_id.as_deref(),
        Some("cg-ingest"),
        "agent_id must not be the hardcoded 'cg-ingest'"
    );
}

// The placeholder harness fabricates confidence=0.9 at capture time.
// Assert confidence is None in the stored decision immediately after capture.
#[test]
fn decision_confidence_is_null_at_capture_never_fabricated_point_nine() {
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "confidence-context",
        "null confidence capture",
        &NullConfidenceExtractor,
    );

    let decision_id = decision_id_from_stable(&report, "D1")
        .expect("D1 must be in stable_ids from NullConfidenceExtractor");
    let decision = store
        .get_decision(decision_id)
        .expect("get decision")
        .expect("decision exists");

    // Placeholder: fabricates Some(0.9). Assert it must be None.
    assert!(
        decision.confidence.is_none(),
        "confidence must be None at capture (evidence-derived), got Some({:?})",
        decision.confidence
    );
    assert_ne!(
        decision.confidence,
        Some(0.9_f32),
        "confidence must not be the fabricated 0.9"
    );
}

// After Outcomes accrue, the library DERIVES a decision's confidence from its outcome
// record. X (3 successes) must outrank Y (1 success + 2 failures). The derivation itself is
// the system under test: the test records outcomes and invokes the library's derivation
// operation, then reads back from the store. The test never computes or writes the derived
// values. The placeholder derivation is a no-op, so both reads come back None and the
// ordering assertion fails.
#[test]
fn decision_confidence_becomes_evidence_derived_after_outcomes_accrue() {
    let (_dir, store) = fresh_store();

    // Capture X and Y in separate contexts so dedup cannot collapse them.
    let report_x = capture_with(
        &store,
        "evidence-context-x",
        "first decision for evidence",
        &NullConfidenceExtractor,
    );
    let report_y = capture_with(
        &store,
        "evidence-context-y",
        "second decision for evidence",
        &NullConfidenceExtractor,
    );

    let decision_id_x =
        decision_id_from_stable(&report_x, "D1").expect("D1 must be in stable_ids for X");
    let decision_id_y =
        decision_id_from_stable(&report_y, "D1").expect("D1 must be in stable_ids for Y");

    assert_ne!(
        decision_id_x, decision_id_y,
        "X and Y must be distinct decisions"
    );

    // X: 3 successes.
    store
        .record_outcome(decision_id_x, true, Some("success 1".to_string()))
        .expect("x outcome 1");
    store
        .record_outcome(decision_id_x, true, Some("success 2".to_string()))
        .expect("x outcome 2");
    store
        .record_outcome(decision_id_x, true, Some("success 3".to_string()))
        .expect("x outcome 3");

    // Y: 1 success + 2 failures.
    store
        .record_outcome(decision_id_y, true, Some("success".to_string()))
        .expect("y success");
    store
        .record_outcome(decision_id_y, false, Some("failure 1".to_string()))
        .expect("y failure 1");
    store
        .record_outcome(decision_id_y, false, Some("failure 2".to_string()))
        .expect("y failure 2");

    // Invoke the library derivation — the system under test. The test does NOT compute or
    // write any confidence value; the library must read the outcome record and persist a
    // derived confidence. The placeholder derivation is a no-op, so confidence stays None.
    context_graph::harness::derive_decision_confidence(&store, decision_id_x)
        .expect("derive X confidence");
    context_graph::harness::derive_decision_confidence(&store, decision_id_y)
        .expect("derive Y confidence");

    let persisted_x = store
        .get_decision(decision_id_x)
        .expect("get x")
        .expect("x exists");
    let persisted_y = store
        .get_decision(decision_id_y)
        .expect("get y")
        .expect("y exists");

    assert!(
        persisted_x.confidence.is_some(),
        "X confidence must be Some after evidence-derivation, got None (derivation did not run)"
    );
    assert!(
        persisted_y.confidence.is_some(),
        "Y confidence must be Some after evidence-derivation, got None (derivation did not run)"
    );
    assert!(
        persisted_x.confidence.unwrap() > persisted_y.confidence.unwrap(),
        "X (3 successes) must have higher derived confidence than Y (1 success, 2 failures); got X={:?} Y={:?}",
        persisted_x.confidence,
        persisted_y.confidence
    );
}

// The placeholder harness drops the outcomes key; captured_outcomes is always empty.
// Assert: one Outcome linked to the decision; recording a second REPLACES the first
// (Invariant 9 — at most one active outcome per decision).
#[test]
fn outcome_is_extracted_and_attached_single_active() {
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "outcome-context",
        "deploy on Kubernetes",
        &OutcomeExtractor,
    );

    let decision_id = decision_id_from_stable(&report, "D1")
        .expect("D1 must be in stable_ids from OutcomeExtractor");

    // Read back from store: the extractor emitted an outcome, so the store must hold exactly
    // one outcome for this decision. The placeholder drops the outcome at capture, so the
    // store has none and this deep read fails.
    let first_outcomes = store
        .admin_raw_outcomes_for_decision(decision_id)
        .expect("list outcomes for decision");
    assert_eq!(
        first_outcomes.len(),
        1,
        "exactly one outcome must be stored after capture when the extractor emits one"
    );
    assert_eq!(
        first_outcomes[0].notes.as_deref(),
        Some("first"),
        "first outcome notes must be 'first'"
    );
    assert!(first_outcomes[0].success, "first outcome success=true");
    let first_id = first_outcomes[0].id;

    // Record a second outcome — Invariant 9: must REPLACE the first.
    store
        .record_outcome(decision_id, false, Some("second".to_string()))
        .expect("record second outcome");

    let after_second = store
        .admin_raw_outcomes_for_decision(decision_id)
        .expect("list outcomes after second");
    assert_eq!(
        after_second.len(),
        1,
        "still exactly one outcome after second (Invariant 9)"
    );
    assert_eq!(
        after_second[0].notes.as_deref(),
        Some("second"),
        "survivor must be the second outcome"
    );
    assert!(!after_second[0].success, "second outcome success=false");
    assert_ne!(
        after_second[0].id, first_id,
        "second outcome has a distinct id (first was replaced)"
    );
}

// The placeholder harness drops observation lanes (observation_type, source, observed_properties, state_delta).
// Assert all four lanes are stored with the emitted values.
#[test]
fn observation_lanes_observed_properties_state_delta_source_type_all_captured() {
    let (_dir, store) = fresh_store();
    capture_with(
        &store,
        "observation-lanes-context",
        "benchmark observation",
        &ObservationLanesExtractor,
    );

    // Read back from the STORE by context — the product contract is that the captured
    // observation is queryable from the store.
    // The placeholder drops the observation at capture, so the store holds none and this
    // deep read fails.
    let context_id = context_id_by_name(&store, "observation-lanes-context");
    let observations = store
        .list_observations(context_id)
        .expect("list observations");
    assert_eq!(
        observations.len(),
        1,
        "exactly one observation must be stored after capture when the extractor emits one"
    );

    let obs = &observations[0];
    assert_eq!(
        obs.observation_type, "benchmark",
        "observation_type must be 'benchmark' from model"
    );
    assert_eq!(
        obs.source, "agent-session",
        "source must be 'agent-session' from model"
    );

    let latency = obs.observed_properties.get("latency_ms");
    assert!(
        latency.is_some(),
        "observed_properties must contain latency_ms"
    );
    assert_eq!(
        latency.unwrap().to_string(),
        "42",
        "latency_ms must be 42 from model"
    );
}

// The placeholder harness drops reasoning, tags, and properties on Decision.
// Assert all three are populated from the model extraction.
#[test]
fn decision_reasoning_tags_properties_are_captured() {
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "decision-metadata-context",
        "messaging decision with metadata",
        &DecisionWithMetadataExtractor,
    );

    // Read the decision from the STORE via stable_id. The placeholder drops reasoning, tags, and
    // properties at capture, so the stored decision has them empty/default and these deep reads fail.
    let decision_id = decision_id_from_stable(&report, "D1")
        .expect("D1 must be in stable_ids from DecisionWithMetadataExtractor");
    let decision = store
        .get_decision(decision_id)
        .expect("get decision")
        .expect("decision exists");

    // Placeholder: drops reasoning, tags, properties — they are empty/default.
    assert_eq!(
        decision.reasoning.len(),
        2,
        "reasoning must have exactly 2 entries from model"
    );
    assert!(
        decision.tags.contains(&"messaging".to_string()),
        "tags must contain 'messaging'"
    );
    assert!(
        decision.tags.contains(&"async".to_string()),
        "tags must contain 'async'"
    );
    assert!(
        decision.properties.contains_key("protocol"),
        "properties must contain 'protocol' key"
    );
}
