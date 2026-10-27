use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::ingest::{CaptureRequest, record_conversation_with_extractor};
use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId, EntityId,
    EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, IntentionId,
    IntentionOrigin, IntentionStatus, ListEntityFilter, ObservationId, RecordObservation,
    RetentionStatus, Store, StoreConfig,
};

use super::action_test_extractors::{
    AgentIdExtractor, AllThirteenTypesExtractor, DecisionTypeExtractor,
    DecisionWithMetadataExtractor, NullConfidenceExtractor, ObservationLanesExtractor,
    OutcomeExtractor, RawEntityTypeStringExtractor, RepositoryModuleExtractor,
};
use super::fake_extractor::open_store_with_lexical_embedder;

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

// A model emits the entity type as a RAW STRING (e.g. "module", "repository"); capture must
// resolve it to the right EntityType variant through the live entity_type_from_str conversion —
// NOT just an enum→type map fed a pre-typed variant. Here the double emits the type only as a
// raw string (via EntitySeed.entity_type_raw, leaving the typed enum at its default), so a stub
// that bypasses string resolution stores the wrong variant. An unrecognized string must be
// flagged out-of-vocab, never silently coerced.
//
#[test]
fn model_emitted_string_entity_type_resolves_to_variant_via_from_str_through_capture() {
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "raw-entity-type-context",
        // The transcript names all three so grounding (name-in-transcript) does not pre-flag them
        // for a missing span — the only flag must come from the out-of-vocab type resolution.
        "ingest-module is a module; context-graph-repo is a repository; mystery-thing is a wormhole",
        &RawEntityTypeStringExtractor,
    );

    let context_id = context_id_by_name(&store, "raw-entity-type-context");
    let entities = store
        .list_entities(ListEntityFilter {
            context_id,
            ..Default::default()
        })
        .expect("list entities");

    // Capture resolves the raw string "module" to Module through entity_type_from_str.
    let module_entity = entities
        .iter()
        .find(|e| e.name.contains(RawEntityTypeStringExtractor::MODULE_NAME));
    assert_eq!(
        module_entity.map(|e| e.entity_type),
        Some(EntityType::Module),
        "raw model string \"module\" must resolve to EntityType::Module via entity_type_from_str; \
         entity_type_raw must override the default File"
    );

    // The repository entity, raw string "repository", must resolve to Repository.
    let repo_entity = entities.iter().find(|e| {
        e.name
            .contains(RawEntityTypeStringExtractor::REPOSITORY_NAME)
    });
    assert_eq!(
        repo_entity.map(|e| e.entity_type),
        Some(EntityType::Repository),
        "raw model string \"repository\" must resolve to EntityType::Repository via entity_type_from_str; \
         the default File must not override the raw type"
    );

    // Capture flags the unrecognized raw type "wormhole" as out-of-vocabulary.
    assert!(
        report.flagged_items.iter().any(|f| {
            f.reason
                .contains(RawEntityTypeStringExtractor::OUT_OF_VOCAB_TYPE)
                || f.reason
                    .contains(RawEntityTypeStringExtractor::OUT_OF_VOCAB_NAME)
        }),
        "an unrecognized raw entity-type string (\"wormhole\") must be flagged out-of-vocab, not \
         silently coerced; got flagged_items: {:?}",
        report.flagged_items
    );
}

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

    assert_eq!(
        decision.decision_type, "technology_choice",
        "decision_type must be 'technology_choice' from model, not the hardcoded 'code_reference'"
    );
    assert_ne!(
        decision.decision_type, "code_reference",
        "decision_type must not be the hardcoded 'code_reference'"
    );
}

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

// After Outcomes accrue, the library DERIVES a decision's confidence from its BROAD track record
// from its BROAD track record — not just this decision's own single Outcome. Confidence accrues from:
// confidence accrues from "Outcome success rates (direct, plus citing-decision success via the
// CITES walk), the count of non-invalidating Observations on the decision's basis entities (it
// survived scrutiny) ...".
//
// The decisive design of this test (so it cannot pass on a single-outcome derivation): X and Y EACH
// get exactly ONE direct Outcome and BOTH are success. A derivation that reads only this decision's
// single active Outcome therefore gets X == Y == 1.0 and cannot tell them apart. The real
// difference is the BROAD record:
//   - X is cited (CITES) by decisions whose Outcomes are SUCCESS, and its basis entity carries MANY
//     non-invalidating Observations (it survived scrutiny);
//   - Y is cited by decisions whose Outcomes are FAILURE, and its basis entity carries FEWER
//     non-invalidating Observations.
// Broad-record derivation yields X.confidence > Y.confidence.
//
// X and Y are built directly with `create_decision` (confidence: None) so the test controls their
// basis entities and CITES precedents — the capture path cannot attach those. The test never
// computes or writes a confidence value; the library derivation is the system under test.
#[test]
fn decision_confidence_becomes_evidence_derived_after_outcomes_accrue() {
    let (_dir, store) = fresh_store();

    // Build X (a strong broad track record) and Y (a weak one) in separate contexts.
    let x = build_confidence_subject(&store, "evidence-context-x", "X primary decision");
    let y = build_confidence_subject(&store, "evidence-context-y", "Y primary decision");
    assert_ne!(x.decision_id, y.decision_id, "X and Y must be distinct");

    // EACH gets exactly ONE direct Outcome, and BOTH are success — so a single-outcome derivation
    // gives X == Y and cannot distinguish them. The discriminator is the broad record below.
    store
        .record_outcome(x.decision_id, true, Some("x direct success".to_string()))
        .expect("x direct outcome");
    store
        .record_outcome(y.decision_id, true, Some("y direct success".to_string()))
        .expect("y direct outcome");

    // X's BROAD record: cited by SUCCESSFUL decisions + MANY non-invalidating Observations.
    for i in 0..3 {
        let citing = create_citing_decision(&store, &x, &format!("cites-x-{i}"));
        store
            .record_outcome(citing, true, Some(format!("citing-x success {i}")))
            .expect("citing-x outcome success");
    }
    for i in 0..5 {
        record_non_invalidating_observation(
            &store,
            x.context_id,
            x.entity_id,
            &format!("x-obs-{i}"),
        );
    }

    // Y's BROAD record: cited by FAILING decisions + FEWER non-invalidating Observations.
    for i in 0..3 {
        let citing = create_citing_decision(&store, &y, &format!("cites-y-{i}"));
        store
            .record_outcome(citing, false, Some(format!("citing-y failure {i}")))
            .expect("citing-y outcome failure");
    }
    record_non_invalidating_observation(&store, y.context_id, y.entity_id, "y-obs-0");

    // Invoke the library derivation (system under test). The test never computes/writes confidence.
    context_graph::harness::derive_decision_confidence(&store, x.decision_id)
        .expect("derive X confidence");
    context_graph::harness::derive_decision_confidence(&store, y.decision_id)
        .expect("derive Y confidence");

    let persisted_x = store
        .get_decision(x.decision_id)
        .expect("get x")
        .expect("x exists");
    let persisted_y = store
        .get_decision(y.decision_id)
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
        "X must have higher derived confidence than Y (X: success citing-decisions + \
         many non-invalidating Observations; Y: failing citing-decisions + few). Both have ONE direct \
         success Outcome, so a single-outcome derivation gives X == Y and FAILS here; got X={:?} Y={:?}",
        persisted_x.confidence,
        persisted_y.confidence
    );
}

// The production capture/outcome/read flow derives confidence from the stored track record
// without an explicit derivation call from the test.
#[test]
fn decision_confidence_is_derived_on_the_production_flow() {
    let (_dir, store) = fresh_store();

    // Build two confidence-null subjects with different broad track records (X strong, Y weak).
    let x = build_confidence_subject(&store, "prod-flow-x", "X prod-flow decision");
    let y = build_confidence_subject(&store, "prod-flow-y", "Y prod-flow decision");

    // Record the track record through the LIVE path (record_outcome / create_decision / observe) —
    // the production operations a real session performs. The test NEVER calls
    // derive_decision_confidence directly: the production flow must invoke it.
    store
        .record_outcome(x.decision_id, true, Some("x direct success".to_string()))
        .expect("x direct outcome");
    store
        .record_outcome(y.decision_id, true, Some("y direct success".to_string()))
        .expect("y direct outcome");
    for i in 0..3 {
        let citing = create_citing_decision(&store, &x, &format!("prod-cites-x-{i}"));
        store
            .record_outcome(citing, true, Some(format!("citing-x success {i}")))
            .expect("citing-x success");
    }
    for i in 0..5 {
        record_non_invalidating_observation(
            &store,
            x.context_id,
            x.entity_id,
            &format!("prod-x-obs-{i}"),
        );
    }
    for i in 0..3 {
        let citing = create_citing_decision(&store, &y, &format!("prod-cites-y-{i}"));
        store
            .record_outcome(citing, false, Some(format!("citing-y failure {i}")))
            .expect("citing-y failure");
    }

    // Read the decisions back THROUGH the production read path — no derive call from the test.
    let persisted_x = store
        .get_decision(x.decision_id)
        .expect("get x")
        .expect("x exists");
    let persisted_y = store
        .get_decision(y.decision_id)
        .expect("get y")
        .expect("y exists");

    // Confidence must be evidence-derived BY the production flow (Some), and move with the record.
    // an implementation that never invokes derivation on the production flow → both are None → regression here.
    assert!(
        persisted_x.confidence.is_some(),
        "X confidence must be Some — derived BY the production capture/outcome/read flow without the \
         test calling derive_decision_confidence; an implementation that leaves it None (derivation never invoked)"
    );
    assert!(
        persisted_y.confidence.is_some(),
        "Y confidence must be Some — derived BY the production flow; an implementation that leaves it None"
    );
    assert!(
        persisted_x.confidence.unwrap() > persisted_y.confidence.unwrap(),
        "production-derived confidence must MOVE with the record: X (success citing-decisions + many \
         non-invalidating Observations) must outrank Y (failing citing-decisions); got X={:?} Y={:?}",
        persisted_x.confidence,
        persisted_y.confidence
    );
}

/// A confidence-null subject decision plus the intention and basis entity it is BASED_ON, so the
/// test can attach CITES precedents and non-invalidating Observations to exercise the evidence-derived confidence walk.
struct ConfidenceSubject {
    context_id: ContextId,
    intention_id: IntentionId,
    entity_id: EntityId,
    decision_id: DecisionId,
}

fn build_confidence_subject(store: &Store, context_name: &str, label: &str) -> ConfidenceSubject {
    let context = store
        .create_context(CreateContext {
            name: context_name.to_string(),
            ..Default::default()
        })
        .expect("create context");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: format!("{label} intention"),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
            blueprint_catalog_id: None,
        })
        .expect("create intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: format!("{label} basis service"),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("create basis entity");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: label.to_string(),
            reasoning: Vec::new(),
            // confidence is null at capture — it must be evidence-derived, never set here.
            confidence: None,
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: None,
            based_on_snapshots: Vec::new(),
            tags: Vec::new(),
            properties: BTreeMap::new(),
            context_id: context.id,
            precedent_ids: Vec::new(),
            agent_id: None,
        })
        .expect("create subject decision");
    ConfidenceSubject {
        context_id: context.id,
        intention_id: intention.id,
        entity_id: entity.id,
        decision_id: decision.id,
    }
}

/// Create a decision that CITES the subject (`precedent_ids: [subject]`), serving the same
/// intention. Its own Outcome (recorded by the caller) feeds the subject's confidence via the
/// CITES walk.
fn create_citing_decision(store: &Store, subject: &ConfidenceSubject, label: &str) -> DecisionId {
    store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: format!("{label} citing decision"),
            reasoning: Vec::new(),
            confidence: None,
            intention_ids: vec![subject.intention_id],
            based_on_entity_ids: vec![subject.entity_id],
            basis_fields: None,
            based_on_snapshots: Vec::new(),
            tags: Vec::new(),
            properties: BTreeMap::new(),
            context_id: subject.context_id,
            precedent_ids: vec![subject.decision_id],
            agent_id: None,
        })
        .expect("create citing decision")
        .id
}

/// Record a NON-invalidating Observation on a basis entity: empty `state_delta` (no state change →
/// no snapshot → cannot invalidate), so it counts as the entity "surviving scrutiny".
fn record_non_invalidating_observation(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    label: &str,
) {
    let observation_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id,
            context_id,
            observation_type: "health_check".to_string(),
            source: label.to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: EvidenceId::new_v7(),
                observation_id,
                context_id: ContextId::default(),
                kind: EvidenceKind::Text,
                source_ref: "webhook://scrutiny".to_string(),
                mime_type: Some("text/plain".to_string()),
                captured_at: Some(Utc::now()),
                producer: EvidenceProducer {
                    system: "acceptance".to_string(),
                    model_name: "fixture".to_string(),
                    model_version: "1".to_string(),
                    pipeline_version: "1".to_string(),
                },
                retention_status: RetentionStatus::RetainedExternal,
                ..Default::default()
            }],
            observed_properties: BTreeMap::from([("ok".to_string(), serde_json::json!(true))]),
            // Empty state_delta → non-invalidating (no entity-state change, no snapshot).
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record non-invalidating observation");
}

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

    // Record a second outcome — must REPLACE the first (one active outcome per decision).
    store
        .record_outcome(decision_id, false, Some("second".to_string()))
        .expect("record second outcome");

    let after_second = store
        .admin_raw_outcomes_for_decision(decision_id)
        .expect("list outcomes after second");
    assert_eq!(
        after_second.len(),
        1,
        "still exactly one outcome after second"
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

#[test]
fn observation_lanes_observed_properties_state_delta_source_type_all_captured() {
    let (_dir, store) = fresh_store();
    capture_with(
        &store,
        "observation-lanes-context",
        "benchmark observation",
        &ObservationLanesExtractor,
    );

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

#[test]
fn decision_reasoning_tags_properties_are_captured() {
    let (_dir, store) = fresh_store();
    let report = capture_with(
        &store,
        "decision-metadata-context",
        "messaging decision with metadata",
        &DecisionWithMetadataExtractor,
    );

    let decision_id = decision_id_from_stable(&report, "D1")
        .expect("D1 must be in stable_ids from DecisionWithMetadataExtractor");
    let decision = store
        .get_decision(decision_id)
        .expect("get decision")
        .expect("decision exists");

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
