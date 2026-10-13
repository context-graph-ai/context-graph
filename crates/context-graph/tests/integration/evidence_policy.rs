use context_graph::*;
use serde_json::json;

use crate::contract_fixture::*;

fn reference_vector(score: f32) -> Vec<f32> {
    let mut vector = vec![0.0; 512];
    vector[0] = score;
    vector[1] = (1.0 - score * score).sqrt();
    vector
}

/// Reference mismatches follow the declared space threshold for every supported evidence kind.
#[test]
fn a_vector_below_the_spaces_declared_mismatch_threshold_invalidates_the_relying_decision() {
    assert_eq!(
        EmbeddingSpaceDecl::default().reference_mismatch_threshold,
        0.8
    );
    for threshold in [0.6, 0.9] {
        let cases = if threshold == 0.6 {
            [(0.5, true), (0.7, false)]
        } else {
            [(0.85, true), (0.95, false)]
        };
        for kind in [EvidenceKind::ImageFrame, EvidenceKind::StructuredSignal] {
            for (score, should_invalidate) in cases {
                let space_id = "reference_identity_v1";
                let (_dir, store, context_id) = fixture(StoreConfig {
                    extra_embedding_spaces: vec![EmbeddingSpaceDecl {
                        embedding_space_id: space_id.into(),
                        dimension: 512,
                        metric: DistanceMetric::Cosine,
                        model_family: "identity".into(),
                        model_name: "reference-identity".into(),
                        model_version: "1".into(),
                        supported_evidence_kinds: vec![
                            EvidenceKind::ImageFrame,
                            EvidenceKind::StructuredSignal,
                        ],
                        reference_mismatch_threshold: threshold,
                        bindings: vec![EmbeddingSpaceBinding {
                            embedding_space_id: space_id.into(),
                            table_name: "evidence".into(),
                            vector_column: "vector_vision".into(),
                        }],
                        ..Default::default()
                    }],
                    ..Default::default()
                });
                let subject = entity(&store, context_id, "reference subject");
                let detected = entity(&store, context_id, "live detection");
                let goal = intention(&store, context_id);
                store
                    .enroll_entity_reference(
                        subject,
                        space_id,
                        None,
                        reference_vector(1.0),
                        EvidenceProducer::default(),
                    )
                    .expect("enroll reference");
                let decision = store
                    .create_decision(CreateDecision {
                        decision_type: "identity".into(),
                        description: "rely on the enrolled identity".into(),
                        intention_ids: vec![goal],
                        based_on_entity_ids: vec![subject],
                        context_id,
                        ..Default::default()
                    })
                    .expect("relying decision");
                let mut input = observation(context_id, detected);
                input.evidence[0].kind = kind;
                input.evidence[0].mime_type = Some(
                    if kind == EvidenceKind::ImageFrame {
                        "image/png"
                    } else {
                        "application/json"
                    }
                    .into(),
                );
                input.embeddings.push(PrecomputedEmbedding {
                    id: EmbeddingId::new_v7(),
                    embedding_space_id: space_id.into(),
                    vector: reference_vector(score),
                    target: EmbeddingTarget::Evidence(input.evidence[0].id),
                    produced_by: EvidenceProducer::default(),
                });
                let observation_id = input.id;
                store
                    .record_observation(input)
                    .expect("record vector evidence");
                let actual_status = store
                    .get_decision(decision.id)
                    .expect("decision lookup")
                    .expect("decision exists")
                    .status;
                assert_eq!(
                    actual_status,
                    if should_invalidate {
                        DecisionStatus::Invalidated
                    } else {
                        DecisionStatus::Active
                    },
                    "space threshold={threshold}, score={score}, kind={kind:?}"
                );
                let invalidations = store
                    .list_invalidations(context_id, Some(decision.id), None)
                    .expect("read invalidations");
                assert_eq!(
                    invalidations.len(),
                    usize::from(should_invalidate),
                    "threshold={threshold}, score={score}, kind={kind:?}"
                );
                if should_invalidate {
                    assert_eq!(invalidations[0].affected_decision_id, decision.id);
                    assert_eq!(invalidations[0].trigger_observation_id, observation_id);
                }
            }
        }
    }
}

/// Ordinary byte counts become state while inline media values are refused atomically.
#[test]
fn an_ordinary_property_named_bytes_is_accepted_while_an_inline_payload_is_refused() {
    let (_dir, store, context_id) = fixture(StoreConfig::default());
    let subject = entity(&store, context_id, "disk");
    let mut ordinary = observation(context_id, subject);
    ordinary
        .state_delta
        .insert("disk_free_bytes".into(), Some(json!(123456789)));
    let ordinary_id = ordinary.id;
    store
        .record_observation(ordinary)
        .expect("ordinary disk byte count");
    let history = store.entity_history(subject).expect("snapshot history");
    assert_eq!(
        history
            .last()
            .expect("current snapshot")
            .properties
            .get("disk_free_bytes"),
        Some(&json!(123456789))
    );
    let payload = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j4ioAAAAASUVORK5CYII=";
    for value in [
        json!(format!("data:image/png;base64,{payload}")),
        json!(payload),
        json!([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82]),
    ] {
        let mut inline = observation(context_id, subject);
        let id = inline.id;
        inline.state_delta.insert("attachment".into(), Some(value));
        assert!(matches!(
            store.record_observation(inline),
            Err(CgError::RawMediaNotStored)
        ));
        assert_eq!(
            store.entity_history(subject).expect("unchanged snapshots"),
            history
        );
        assert_eq!(
            store
                .list_observations(Some(context_id))
                .expect("stored observations")
                .iter()
                .map(|o| o.id)
                .collect::<Vec<_>>(),
            vec![ordinary_id]
        );
        assert_eq!(
            store
                .list_evidence_for_observation(id)
                .expect("refused evidence"),
            Vec::new()
        );
    }
}

/// Evidence schemes come from the store declaration and reject undeclared schemes.
#[test]
fn a_declared_evidence_scheme_is_accepted_and_an_undeclared_one_refused() {
    let (_dir, store, context_id) = fixture(StoreConfig::default());
    let subject = entity(&store, context_id, "source");
    let mut expected = Vec::new();
    for scheme in ["s3", "gs", "azure", "cas", "webhook"] {
        let mut input = observation(context_id, subject);
        input.evidence[0].source_ref = format!("{scheme}://archive/item");
        expected.push(input.id);
        let evidence_id = input.evidence[0].id;
        let source_ref = input.evidence[0].source_ref.clone();
        store
            .record_observation(input)
            .expect("default declared scheme");
        assert_eq!(
            store
                .get_evidence(evidence_id)
                .expect("stored reference")
                .source_ref,
            source_ref
        );
    }
    for scheme in ["vigil-edge", "edge-cam"] {
        let mut input = observation(context_id, subject);
        let id = input.id;
        input.evidence[0].source_ref = format!("{scheme}://camera/frame/42");
        let before = store.entity_history(subject).expect("snapshot history");
        assert!(
            matches!(store.record_observation(input), Err(CgError::InvalidSourceRefScheme { scheme: refused, host: None }) if refused == scheme)
        );
        assert_eq!(
            store.entity_history(subject).expect("unchanged history"),
            before
        );
        assert_eq!(
            store
                .list_evidence_for_observation(id)
                .expect("no refused evidence"),
            Vec::new()
        );
    }
    let mut actual = store
        .list_observations(Some(context_id))
        .expect("accepted observations")
        .iter()
        .map(|o| o.id)
        .collect::<Vec<_>>();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
    let mut config = StoreConfig::default();
    config.accepted_evidence_schemes.push("edge-cam".into());
    let (_extra_dir, extra_store, extra_context) = fixture(config);
    let extra_subject = entity(&extra_store, extra_context, "configured source");
    let mut input = observation(extra_context, extra_subject);
    let evidence_id = input.evidence[0].id;
    input.evidence[0].source_ref = "edge-cam://camera/frame/42".into();
    extra_store
        .record_observation(input)
        .expect("application declared scheme");
    assert_eq!(
        extra_store
            .get_evidence(evidence_id)
            .expect("configured reference")
            .source_ref,
        "edge-cam://camera/frame/42"
    );
    let mut undeclared = observation(extra_context, extra_subject);
    undeclared.evidence[0].source_ref = "other-cam://camera/frame/42".into();
    assert!(
        matches!(extra_store.record_observation(undeclared), Err(CgError::InvalidSourceRefScheme { scheme, host: None }) if scheme == "other-cam")
    );
}
