use chrono::Utc;
use context_graph::{
    CgError, ContextId, CreateContext, CreateDigest, CreateEntity, DigestId, DistanceMetric,
    EmbeddingId, EmbeddingSpaceBinding, EmbeddingSpaceDecl, EmbeddingTarget, EntityId, EntityType,
    EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, PrecomputedEmbedding,
    RecordObservation, ReindexEvent, Store, StoreConfig,
    test_support::embedding::{VectorSearchFilters, dedup_compare_digest_spaces, vector_search},
};
use std::collections::BTreeMap;

const OLD_TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";
const NEW_TEXT_SPACE: &str = "text_default_embeddinggemma_500m_768";

fn store() -> Store {
    Store::open(StoreConfig {
        db_path: std::env::temp_dir().join(format!(
            "screenshot-distinguishability-{}.db",
            uuid::Uuid::now_v7()
        )),
        extra_embedding_spaces: vec![EmbeddingSpaceDecl {
            embedding_space_id: NEW_TEXT_SPACE.to_string(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "text".to_string(),
            model_name: "embeddinggemma-500m".to_string(),
            model_version: "v1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: NEW_TEXT_SPACE.to_string(),
                table_name: "digests".to_string(),
                vector_column: "vector_text".to_string(),
            }],
        }],
        ..Default::default()
    })
    .expect("open")
}

fn context_and_entity(store: &Store) -> (ContextId, EntityId) {
    let context = store
        .create_context(CreateContext {
            name: format!("screenshot-distinguishability-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "screenshot-review".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    (context.id, entity.id)
}

fn producer(model_name: &str, model_version: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: model_name.to_string(),
        model_version: model_version.to_string(),
        pipeline_version: "screenshot-review".to_string(),
    }
}

fn vector(dim: usize, axis: usize) -> Vec<f32> {
    let mut values = vec![0.0; dim];
    values[axis] = 1.0;
    values
}

fn image_frame_evidence(evidence_id: EvidenceId) -> EvidenceRef {
    EvidenceRef {
        id: evidence_id,
        kind: EvidenceKind::ImageFrame,
        source_ref: "cas://sha256:2222222222222222222222222222222222222222222222222222222222222222"
            .to_string(),
        mime_type: Some("image/png".to_string()),
        producer: producer("siglip", "v1"),
        ..Default::default()
    }
}

fn caption_text_evidence(evidence_id: EvidenceId, derived_from: EvidenceId) -> EvidenceRef {
    EvidenceRef {
        id: evidence_id,
        kind: EvidenceKind::Text,
        source_ref: "cas://sha256:3333333333333333333333333333333333333333333333333333333333333333"
            .to_string(),
        mime_type: Some("text/plain".to_string()),
        text_projection: Some("rate limiter architecture screenshot caption".to_string()),
        producer: producer("captioner", "v1"),
        derived_from: Some(derived_from),
        ..Default::default()
    }
}

fn record_caption_text_evidence(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    image: EvidenceRef,
    evidence: EvidenceRef,
) -> context_graph::Result<()> {
    store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "screenshot_review".to_string(),
        source: "edge-screenshot-review".to_string(),
        observed_at: Utc::now(),
        evidence: vec![image, evidence.clone()],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: OLD_TEXT_SPACE.to_string(),
            vector: vector(768, 0),
            target: EmbeddingTarget::Evidence(evidence.id),
            produced_by: producer("embeddinggemma-300m", "v1"),
        }],
    })?;
    Ok(())
}

fn create_digest_in_space(
    store: &Store,
    digest_id: DigestId,
    context_id: ContextId,
    space_id: &str,
    model_name: &str,
) -> context_graph::Result<()> {
    store.create_digest(CreateDigest {
        id: digest_id,
        source: "screenshot-agent".to_string(),
        session_ref: Some("week-later".to_string()),
        summary: "rate limiter architecture".to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: space_id.to_string(),
            vector: vector(768, 0),
            target: EmbeddingTarget::Digest(digest_id),
            produced_by: producer(model_name, "v1"),
        },
        context_id,
        properties: BTreeMap::new(),
    })?;
    Ok(())
}

#[test]
fn screenshot_review_result_row_distinguishes_digest_target_from_evidence_target_via_embedding_target_variant()
 {
    let evidence_id = EvidenceId::new_v7();
    let image_evidence_id = EvidenceId::new_v7();
    let digest_id = DigestId::new_v7();
    let store = store();
    let (context_id, entity_id) = context_and_entity(&store);
    let caption = record_caption_text_evidence(
        &store,
        context_id,
        entity_id,
        image_frame_evidence(image_evidence_id),
        caption_text_evidence(evidence_id, image_evidence_id),
    );
    let digest = create_digest_in_space(
        &store,
        digest_id,
        context_id,
        OLD_TEXT_SPACE,
        "embeddinggemma-300m",
    );
    let hits = vector_search(
        &store,
        &vector(768, 0),
        OLD_TEXT_SPACE,
        VectorSearchFilters {
            top_k: Some(2),
            ..Default::default()
        },
    );

    assert!(
        caption.is_ok()
            && digest.is_ok()
            && hits.as_ref().is_ok_and(|rows| rows.len() >= 2
                && rows
                    .iter()
                    .any(|row| row.target == EmbeddingTarget::Evidence(evidence_id))
                && rows
                    .iter()
                    .any(|row| row.target == EmbeddingTarget::Digest(digest_id))),
        "one text-space retrieval result set must distinguish caption Evidence and Digest targets, caption={caption:?}, digest={digest:?}, hits={hits:?}"
    );
}

/// @invariant 15a
#[test]
fn screenshot_review_dedup_classifier_refuses_cross_space_compare_under_realistic_text_default_version_bump_and_emits_reindex_trigger_signal()
 {
    let store = store();
    let (context_id, _entity_id) = context_and_entity(&store);
    let tuesday_digest = DigestId::new_v7();
    let friday_digest = DigestId::new_v7();
    let tuesday = create_digest_in_space(
        &store,
        tuesday_digest,
        context_id,
        OLD_TEXT_SPACE,
        "embeddinggemma-300m",
    );
    let friday = create_digest_in_space(
        &store,
        friday_digest,
        context_id,
        NEW_TEXT_SPACE,
        "embeddinggemma-500m",
    );
    let result = dedup_compare_digest_spaces(
        &store,
        friday_digest,
        NEW_TEXT_SPACE,
        tuesday_digest,
        OLD_TEXT_SPACE,
    );
    let friday_hits = vector_search(
        &store,
        &vector(768, 0),
        NEW_TEXT_SPACE,
        VectorSearchFilters {
            top_k: Some(2),
            ..Default::default()
        },
    );
    let tuesday_hits = vector_search(
        &store,
        &vector(768, 0),
        OLD_TEXT_SPACE,
        VectorSearchFilters {
            top_k: Some(2),
            ..Default::default()
        },
    );
    let events = store.drain_reindex_events();

    assert!(
        tuesday.is_ok()
            && friday.is_ok()
            && friday_hits.as_ref().is_ok_and(|rows| rows.len() == 1
                && rows[0].target == context_graph::EmbeddingTarget::Digest(friday_digest)
                && rows[0].index_ref == "digests.vector_text")
            && tuesday_hits.as_ref().is_ok_and(|rows| rows.len() == 1
                && rows[0].target == context_graph::EmbeddingTarget::Digest(tuesday_digest)
                && rows[0].index_ref == "digests.vector_text")
            && matches!(
                &result,
                Err(CgError::CrossSpaceQueryRefused {
                    query_space,
                    candidate_space,
                    ..
                }) if query_space == NEW_TEXT_SPACE && candidate_space == OLD_TEXT_SPACE
            )
            && events.iter().any(|event| matches!(
                event,
                ReindexEvent::TriggerSuggested {
                    from_space,
                    to_space
                } if from_space == OLD_TEXT_SPACE && to_space == NEW_TEXT_SPACE
            )),
        "realistic text-default version bump must persist each Digest under its own space-filtered digests.vector_text binding, refuse exact spaces, and emit exact reindex trigger, tuesday={tuesday:?}, friday={friday:?}, friday_hits={friday_hits:?}, tuesday_hits={tuesday_hits:?}, result={result:?}, events={events:?}"
    );
}
