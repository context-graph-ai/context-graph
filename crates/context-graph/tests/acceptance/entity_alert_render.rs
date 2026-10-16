use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::{
    CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId, DistanceMetric,
    EmbeddingId, EmbeddingSpaceBinding, EmbeddingSpaceDecl, EmbeddingTarget, EntityId, EntityType,
    EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, IntentionOrigin, IntentionStatus,
    Invalidation, InvalidationSeverity, InvalidationStatus, MessageBody, MismatchKind,
    OverrideFailureReason, PrecomputedEmbedding, RecordObservation, RetentionStatus, Store,
    StoreConfig,
};

const FACE_REID_SPACE: &str = "vigil_face_reid_v1";

struct Scenario {
    _dir: tempfile::TempDir,
    store: Store,
    context_id: context_graph::ContextId,
    entity_id: EntityId,
    detected_entity_id: EntityId,
    decision_id: DecisionId,
}

fn face_producer(version: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "face-local".to_string(),
        model_version: version.to_string(),
        pipeline_version: "gate-v2".to_string(),
    }
}

fn vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 512];
    vector[axis] = 1.0;
    vector
}

fn scenario() -> Scenario {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("entry.db"),
        extra_embedding_spaces: vec![EmbeddingSpaceDecl {
            embedding_space_id: FACE_REID_SPACE.to_string(),
            dimension: 512,
            metric: DistanceMetric::Cosine,
            model_family: "vision".to_string(),
            model_name: "face-local".to_string(),
            model_version: "v1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
            default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: FACE_REID_SPACE.to_string(),
                table_name: "evidence".to_string(),
                vector_column: "vector_vision".to_string(),
            }],
        }],
        ..Default::default()
    })
    .expect("store opens");
    let context = store
        .create_context(CreateContext {
            name: "entry".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "trust Alex at 8 AM".to_string(),
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
            name: "Alex".to_string(),
            properties: BTreeMap::new(),
            tags: vec!["student:primary-guardian-approved".to_string()],
            context_id: context.id,
        })
        .expect("known child");
    let detected_entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: "unidentified person at entry".to_string(),
            properties: BTreeMap::from([("track".to_string(), serde_json::json!("gate-42"))]),
            tags: vec!["entry-live-detection".to_string()],
            context_id: context.id,
        })
        .expect("live detection track");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "access-control".to_string(),
            description: "trust Alex at 8 AM".to_string(),
            confidence: Some(0.91),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            context_id: context.id,
            ..Default::default()
        })
        .expect("decision");
    Scenario {
        _dir: dir,
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
        observation_type: "entry-camera".to_string(),
        source: "gate-camera".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::ImageFrame,
            source_ref: "s3://camera/gate/frame/42".to_string(),
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
            produced_by: face_producer("live-v1"),
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
        if diff.field == "entity_name" && diff.current_value == Some(serde_json::json!("Alex")) {
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
fn alert_message_body_distinguishes_known_entity_from_unknown_at_entry() {
    let scenario = scenario();
    let enrollment = scenario.store.enroll_entity_reference(
        scenario.entity_id,
        FACE_REID_SPACE,
        Some("student:primary-guardian-approved".to_string()),
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
    let message = invalidations
        .as_ref()
        .ok()
        .and_then(|rows| rows.first())
        .map(|row| scenario.store.render_entity_alert_message_body(row.id));

    assert!(
        enrollment.is_ok()
            && observation.is_ok()
            && invalidations.as_ref().is_ok_and(|rows| rows.len() == 1
                && rows[0].affected_decision_id == scenario.decision_id
                && rows[0].severity == InvalidationSeverity::Critical
                && rows[0].intention_now_un_served
                && invalidation_carries_entity_aware_no_match_payload(&rows[0], &scenario))
            && matches!(
                &message,
                Some(Ok(MessageBody::EntityMismatch {
                    entity_id,
                    entity_name,
                    mismatch_kind: MismatchKind::FaceReferenceNoMatch,
                    override_failure_reason:
                        OverrideFailureReason::EntityAwareNoMatch {
                            reference_space,
                            live_space
                        },
                })) if *entity_id == scenario.entity_id
                    && entity_name == "Alex"
                    && reference_space == FACE_REID_SPACE
                    && live_space == FACE_REID_SPACE
            )
            && !matches!(&message, Some(Ok(MessageBody::ClassOnlyDetection { .. }))),
        "entry alert must come from the persisted no-match invalidation payload and be entity-aware, enrollment={enrollment:?}, observation={observation:?}, invalidations={invalidations:?}, message={message:?}"
    );
}
