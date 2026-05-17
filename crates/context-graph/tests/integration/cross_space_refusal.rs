use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::test_support::embedding::{
    CompositeVectorClause, FusionPolicy, VectorSearchFilters, VectorSearchResult,
    compare_embedding_targets_across_spaces, composite_vector_search_with_policy,
};
use context_graph::{
    CreateContext, CreateEntity, DistanceMetric, EmbeddingId, EmbeddingSpace,
    EmbeddingSpaceBinding, EmbeddingTarget, EntityType, EvidenceId, EvidenceKind, EvidenceProducer,
    EvidenceRef, PrecomputedEmbedding, RecordObservation, RetentionStatus, Store, StoreConfig,
};

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";
const IMAGE_SPACE: &str = "image_clip_v1_4";

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "test-embedder".to_string(),
        model_version: "v1".to_string(),
        pipeline_version: "cross-space".to_string(),
    }
}

fn image_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: IMAGE_SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "clip-local".to_string(),
        model_version: "v1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: IMAGE_SPACE.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_vision".to_string(),
        }],
    }
}

fn store() -> Store {
    Store::open(StoreConfig {
        db_path: std::env::temp_dir().join(format!("cross-space-{}.db", uuid::Uuid::now_v7())),
        extra_embedding_spaces: vec![image_space()],
        ..Default::default()
    })
    .expect("open")
}

fn axis(dim: usize, idx: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dim];
    vector[idx] = 1.0;
    vector
}

fn mime_for_kind(kind: EvidenceKind) -> Option<String> {
    match kind {
        EvidenceKind::ImageFrame => Some("image/png".to_string()),
        EvidenceKind::StructuredSignal => Some("application/json".to_string()),
        EvidenceKind::Text => Some("text/plain".to_string()),
        _ => None,
    }
}

fn seed_evidence(store: &Store, space: &str, kind: EvidenceKind, vector: Vec<f32>) -> EvidenceId {
    let context = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: format!("entity-{}", uuid::Uuid::now_v7()),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    let evidence_id = EvidenceId::new_v7();
    let recorded = store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: entity.id,
        context_id: context.id,
        observation_type: "embedding-candidate".to_string(),
        source: "test".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            mime_type: mime_for_kind(kind),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer(),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: space.to_string(),
            vector,
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer(),
        }],
    });
    assert!(recorded.is_ok(), "seed evidence failed: {recorded:?}");
    evidence_id
}

fn score_for(hit: &VectorSearchResult, space: &str) -> Option<f32> {
    hit.per_clause_scores
        .iter()
        .find_map(|(candidate_space, score)| (candidate_space == space).then_some(*score))
}

fn close_enough(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1.0e-6
}

/// @invariant 15a
#[test]
fn cross_space_cosine_refused_at_query_surface_when_dimensions_match() {
    let store = store();
    let candidate = seed_evidence(&store, IMAGE_SPACE, EvidenceKind::ImageFrame, axis(512, 0));
    let result = compare_embedding_targets_across_spaces(
        &store,
        EmbeddingTarget::Evidence(EvidenceId::new_v7()),
        "face_live_v2",
        EmbeddingTarget::Evidence(candidate),
        IMAGE_SPACE,
        512,
        512,
    );
    assert!(matches!(
        result,
        Err(context_graph::CgError::CrossSpaceQueryRefused {
            query_space,
            candidate_space,
            query_dim,
            candidate_dim
        }) if query_space == "face_live_v2"
            && candidate_space == IMAGE_SPACE
            && query_dim == Some(512)
            && candidate_dim == Some(512)
    ));
}

/// @invariant 15a
#[test]
fn cross_space_cosine_refused_at_query_surface_when_dimensions_differ() {
    let store = store();
    let text_candidate = seed_evidence(&store, TEXT_SPACE, EvidenceKind::Text, axis(768, 0));
    let _image_candidate =
        seed_evidence(&store, IMAGE_SPACE, EvidenceKind::ImageFrame, axis(512, 0));
    let result = compare_embedding_targets_across_spaces(
        &store,
        EmbeddingTarget::Evidence(EvidenceId::new_v7()),
        "vision_clip_512",
        EmbeddingTarget::Evidence(text_candidate),
        TEXT_SPACE,
        512,
        768,
    );
    assert!(matches!(
        result,
        Err(context_graph::CgError::CrossSpaceQueryRefused {
            query_space,
            candidate_space,
            query_dim,
            candidate_dim
        }) if query_space == "vision_clip_512"
            && candidate_space == "text_default_embeddinggemma_300m_768"
            && query_dim == Some(512)
            && candidate_dim == Some(768)
    ));
}

#[test]
fn composite_vector_search_requires_explicit_fusion_policy_no_default() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/composite_query_without_fusion_policy.rs");
}

#[test]
fn composite_vector_search_keeps_per_clause_scores_when_fusion_policy_provided() {
    let store = store();
    let text_id = seed_evidence(&store, TEXT_SPACE, EvidenceKind::Text, axis(768, 0));
    let image_id = seed_evidence(&store, IMAGE_SPACE, EvidenceKind::ImageFrame, axis(512, 1));
    let hits = composite_vector_search_with_policy(
        &store,
        vec![
            CompositeVectorClause {
                query_vector: axis(768, 0),
                embedding_space_id: TEXT_SPACE.to_string(),
            },
            CompositeVectorClause {
                query_vector: axis(512, 1),
                embedding_space_id: IMAGE_SPACE.to_string(),
            },
        ],
        FusionPolicy::WeightedSum(vec![
            (TEXT_SPACE.to_string(), 0.75),
            (IMAGE_SPACE.to_string(), 0.25),
        ]),
        VectorSearchFilters {
            top_k: Some(10),
            ..Default::default()
        },
    )
    .expect("explicit fusion query");

    let text_hit = hits
        .iter()
        .find(|hit| hit.target == EmbeddingTarget::Evidence(text_id));
    let image_hit = hits
        .iter()
        .find(|hit| hit.target == EmbeddingTarget::Evidence(image_id));
    assert!(
        text_hit.is_some() && image_hit.is_some(),
        "composite fusion must return the exact persisted text and image targets, hits={hits:?}, text_id={text_id:?}, image_id={image_id:?}"
    );
    let text_hit = text_hit.unwrap();
    let image_hit = image_hit.unwrap();

    assert!(
        text_hit.per_clause_scores.len() == 2
            && close_enough(score_for(text_hit, TEXT_SPACE).unwrap_or(-1.0), 1.0)
            && close_enough(score_for(text_hit, IMAGE_SPACE).unwrap_or(-1.0), 0.0)
            && close_enough(text_hit.score, 0.75)
            && image_hit.per_clause_scores.len() == 2
            && close_enough(score_for(image_hit, TEXT_SPACE).unwrap_or(-1.0), 0.0)
            && close_enough(score_for(image_hit, IMAGE_SPACE).unwrap_or(-1.0), 1.0)
            && close_enough(image_hit.score, 0.25)
            && hits
                .first()
                .is_some_and(|hit| hit.target == text_hit.target),
        "explicit fusion must expose both per-clause scores and hand-computed weighted sums; hits={hits:?}"
    );
}
