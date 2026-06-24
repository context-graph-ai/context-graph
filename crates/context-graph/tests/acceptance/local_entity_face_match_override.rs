use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::{
    CgError, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId,
    DistanceMetric, EmbeddingId, EmbeddingSpaceBinding, EmbeddingSpaceDecl, EmbeddingTarget,
    EntityId, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, IntentionOrigin,
    IntentionStatus, Invalidation, InvalidationSeverity, InvalidationStatus, OverrideFailureReason,
    PrecomputedEmbedding, RecordObservation, RetentionStatus, Store, StoreConfig,
    test_support::embedding::entity_reference_cosine_match,
};

const FACE_REID_SPACE: &str = "vigil_face_reid_v1";
const FACE_MISMATCH_SPACE: &str = "vigil_face_reid_v2";

struct Scenario {
    store: Store,
    context_id: context_graph::ContextId,
    entity_id: EntityId,
    detected_entity_id: EntityId,
    decision_id: DecisionId,
}

fn store(name: &str) -> Store {
    Store::open(StoreConfig {
        db_path: std::env::temp_dir().join(format!("{name}-{}.db", EntityId::new_v7())),
        extra_embedding_spaces: vec![
            EmbeddingSpaceDecl {
                embedding_space_id: FACE_REID_SPACE.to_string(),
                dimension: 512,
                metric: DistanceMetric::Cosine,
                model_family: "vision".to_string(),
                model_name: "face-reid".to_string(),
                model_version: "1".to_string(),
                supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
                default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
                bindings: vec![EmbeddingSpaceBinding {
                    embedding_space_id: FACE_REID_SPACE.to_string(),
                    table_name: "evidence".to_string(),
                    vector_column: "vector_vision".to_string(),
                }],
            },
            EmbeddingSpaceDecl {
                embedding_space_id: FACE_MISMATCH_SPACE.to_string(),
                dimension: 512,
                metric: DistanceMetric::Cosine,
                model_family: "vision".to_string(),
                model_name: "face-reid".to_string(),
                model_version: "2".to_string(),
                supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
                default_for_evidence_kind: None,
                bindings: Vec::new(),
            },
        ],
        ..Default::default()
    })
    .expect("store opens")
}

fn face_producer(version: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "face-local".to_string(),
        model_version: version.to_string(),
        pipeline_version: "face-pipeline-v2".to_string(),
    }
}

fn vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 512];
    vector[axis] = 1.0;
    vector
}

fn scenario(name: &str) -> Scenario {
    let store = store(name);
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "trust Kumar at 8 AM".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: "Kumar".to_string(),
            properties: BTreeMap::from([("role".to_string(), serde_json::json!("guardian"))]),
            tags: vec!["student-gate".to_string()],
            context_id: context.id,
        })
        .expect("entity");
    let detected_entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: "unidentified person at school gate".to_string(),
            properties: BTreeMap::from([("track".to_string(), serde_json::json!("gate-42"))]),
            tags: vec!["school-gate-live-detection".to_string()],
            context_id: context.id,
        })
        .expect("detected person track");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "access-control".to_string(),
            description: "trust Kumar at 8 AM".to_string(),
            confidence: Some(0.91),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            context_id: context.id,
            ..Default::default()
        })
        .expect("decision");
    Scenario {
        store,
        context_id: context.id,
        entity_id: entity.id,
        detected_entity_id: detected_entity.id,
        decision_id: decision.id,
    }
}

fn live_face_observation(scenario: &Scenario, axis: usize) -> RecordObservation {
    let evidence_id = EvidenceId::new_v7();
    RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: scenario.detected_entity_id,
        context_id: scenario.context_id,
        observation_type: "school-gate-camera".to_string(),
        source: "gate-camera".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::ImageFrame,
            source_ref: "vigil-edge://camera/gate/frame/42".to_string(),
            mime_type: Some("image/jpeg".to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: face_producer("detector-v1"),
            ..Default::default()
        }],
        observed_properties: BTreeMap::from([("class".to_string(), serde_json::json!("person"))]),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: FACE_REID_SPACE.to_string(),
            vector: vector(axis),
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: face_producer("live-v2"),
        }],
    }
}

fn invalidation_carries_entity_aware_no_match_payload(
    invalidation: &Invalidation,
    scenario: &Scenario,
) -> bool {
    let expected_override = serde_json::to_value(OverrideFailureReason::EntityAwareNoMatch {
        reference_space: FACE_REID_SPACE.to_string(),
        live_space: FACE_REID_SPACE.to_string(),
    })
    .expect("override json");
    let mut entity_name = false;
    let mut entity_id = false;
    let mut override_failure_reason = false;

    for diff in invalidation
        .basis_diff
        .divergences
        .iter()
        .flat_map(|divergence| &divergence.value_diffs)
    {
        if diff.field == "entity_name" && diff.current_value == Some(serde_json::json!("Kumar")) {
            entity_name = true;
        }
        if diff.field == "entity_id"
            && diff.current_value == Some(serde_json::json!(scenario.entity_id.as_uuid()))
        {
            entity_id = true;
        }
        if diff.field == "override_failure_reason"
            && diff.current_value == Some(expected_override.clone())
        {
            override_failure_reason = true;
        }
    }

    entity_name && entity_id && override_failure_reason
}

#[test]
fn local_entity_face_embedding_match_overrides_class_detection() {
    let scenario = scenario("face-match-overrides");
    let enrollment = scenario.store.enroll_entity_reference(
        scenario.entity_id,
        FACE_REID_SPACE,
        Some("known-person".to_string()),
        vector(0),
        face_producer("reference-v1"),
    );
    let observation = scenario
        .store
        .record_observation(live_face_observation(&scenario, 0));
    let invalidations =
        scenario
            .store
            .list_invalidations(scenario.context_id, Some(scenario.decision_id), None);

    assert!(
        enrollment
            .as_ref()
            .is_ok_and(|id| id.is_valid() && *id != EmbeddingId::default())
            && observation.is_ok()
            && invalidations.as_ref().is_ok_and(Vec::is_empty),
        "matching face reference must suppress class-only invalidation, enrollment={enrollment:?}, observation={observation:?}, invalidations={invalidations:?}"
    );
}

#[test]
fn local_entity_face_embedding_no_match_invalidates_active_decision_and_emits_critical() {
    let scenario = scenario("face-no-match");
    let enrollment = scenario.store.enroll_entity_reference(
        scenario.entity_id,
        FACE_REID_SPACE,
        Some("known-person".to_string()),
        vector(0),
        face_producer("reference-v1"),
    );
    let observation = scenario
        .store
        .record_observation(live_face_observation(&scenario, 1));
    let invalidations = scenario.store.list_invalidations(
        scenario.context_id,
        Some(scenario.decision_id),
        Some(InvalidationStatus::Pending),
    );

    assert!(
        enrollment.is_ok()
            && observation.is_ok()
            && invalidations.as_ref().is_ok_and(|rows| rows.len() == 1
                && rows[0].affected_decision_id == scenario.decision_id
                && rows[0].severity == InvalidationSeverity::Critical
                && rows[0].intention_now_un_served
                && invalidation_carries_entity_aware_no_match_payload(&rows[0], &scenario)),
        "nonmatching face must emit one critical invalidation tied to Kumar's decision with persisted entity-aware override payload, enrollment={enrollment:?}, observation={observation:?}, invalidations={invalidations:?}"
    );
}

#[test]
fn local_entity_face_embedding_match_after_multi_shot_reenrollment_re_overrides() {
    let scenario = scenario("face-reenroll");
    let first = scenario.store.enroll_entity_reference(
        scenario.entity_id,
        FACE_REID_SPACE,
        Some("front".to_string()),
        vector(0),
        face_producer("reference-v1"),
    );
    let second = scenario.store.enroll_entity_reference(
        scenario.entity_id,
        FACE_REID_SPACE,
        Some("profile".to_string()),
        vector(1),
        face_producer("reference-v1"),
    );
    let observation = scenario
        .store
        .record_observation(live_face_observation(&scenario, 1));
    let invalidations =
        scenario
            .store
            .list_invalidations(scenario.context_id, Some(scenario.decision_id), None);

    assert!(
        first
            .as_ref()
            .is_ok_and(|id| id.is_valid() && *id != EmbeddingId::default())
            && second
                .as_ref()
                .is_ok_and(|id| id.is_valid() && Some(id) != first.as_ref().ok())
            && observation.is_ok()
            && invalidations.as_ref().is_ok_and(Vec::is_empty),
        "second enrolled shot must also override a class-only detection, first={first:?}, second={second:?}, observation={observation:?}, invalidations={invalidations:?}"
    );
}

/// @invariant 15a
#[test]
fn local_entity_face_embedding_v1_v2_space_mismatch_refused_with_typed_error_naming_both_spaces() {
    let scenario = scenario("face-space-mismatch");
    let enrollment = scenario.store.enroll_entity_reference(
        scenario.entity_id,
        FACE_REID_SPACE,
        Some("known-person".to_string()),
        vector(0),
        face_producer("face-v1"),
    );
    let result = enrollment
        .as_ref()
        .map(|id| entity_reference_cosine_match(&scenario.store, FACE_MISMATCH_SPACE, *id));

    assert!(
        enrollment.is_ok()
            && matches!(
                result,
                Ok(Err(CgError::CrossSpaceCosineRefused {
                    ref live_space,
                    ref reference_space
                })) if live_space == FACE_MISMATCH_SPACE && reference_space == FACE_REID_SPACE
            ),
        "mismatched live/reference spaces must refuse with typed names from the comparison path, enrollment={enrollment:?}, result={result:?}"
    );
}
