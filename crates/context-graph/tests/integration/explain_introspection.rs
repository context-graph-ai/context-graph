use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::embedder::embeddinggemma::EmbeddingGemmaTextEmbedder;
use context_graph::embedder::{Embedder, EmbeddingInput};
use context_graph::test_support::embedding::{VectorSearchFilters, vector_search};
use context_graph::{
    CreateContext, CreateEntity, EmbedderConfig, EmbeddingId, EmbeddingTarget, EntityId,
    EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, PrecomputedEmbedding,
    RecordObservation, RetentionStatus, Store, StoreConfig,
};
use contextdb_core::{Value, VectorIndexRef};
use contextdb_engine::Database;

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";

fn vector(axis: usize) -> Vec<f32> {
    let mut value = vec![0.0; 768];
    value[axis % 768] = 1.0;
    value
}

fn blended_vector(primary: usize, secondary: usize, primary_weight: f32) -> Vec<f32> {
    let mut value = vec![0.0; 768];
    value[primary % 768] = primary_weight;
    value[secondary % 768] = 1.0 - primary_weight;
    value
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "embeddinggemma-300m".to_string(),
        model_version: "planner".to_string(),
        pipeline_version: "explain".to_string(),
    }
}

fn seed_candidate(
    store: &Store,
    context_id: context_graph::ContextId,
    entity_id: EntityId,
    evidence_id: EvidenceId,
    vector: Vec<f32>,
) {
    let result = store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id,
        context_id,
        observation_type: "planner-candidate".to_string(),
        source: "explain-introspection".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            mime_type: Some("text/plain".to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer(),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: TEXT_SPACE.to_string(),
            vector,
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer(),
        }],
    });
    assert!(result.is_ok(), "seed candidate failed: {result:?}");
}

#[test]
fn vector_search_with_context_id_in_filter_explain_uses_indexed_scan_filter_intersect_not_post_filter()
 {
    let db_path =
        std::env::temp_dir().join(format!("explain-introspection-{}.db", uuid::Uuid::now_v7()));
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let allowed_context = store
        .create_context(CreateContext {
            name: "allowed-vector-context".to_string(),
            labels: vec!["org:acme".to_string(), "team:platform".to_string()],
            properties: BTreeMap::new(),
        })
        .expect("allowed context");
    let other_context = store
        .create_context(CreateContext {
            name: "other-vector-context".to_string(),
            labels: vec!["org:other".to_string()],
            properties: BTreeMap::new(),
        })
        .expect("other context");
    let allowed_entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "allowed".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: allowed_context.id,
        })
        .expect("allowed entity");
    let other_entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "other".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: other_context.id,
        })
        .expect("other entity");
    let expected = EvidenceId::new_v7();
    let expected_second = EvidenceId::new_v7();
    let expected_third = EvidenceId::new_v7();
    let expected_fourth = EvidenceId::new_v7();
    let expected_fifth = EvidenceId::new_v7();
    seed_candidate(
        &store,
        allowed_context.id,
        allowed_entity.id,
        expected,
        vector(7),
    );
    seed_candidate(
        &store,
        allowed_context.id,
        allowed_entity.id,
        expected_second,
        blended_vector(7, 17, 0.80),
    );
    seed_candidate(
        &store,
        allowed_context.id,
        allowed_entity.id,
        expected_third,
        blended_vector(7, 18, 0.60),
    );
    seed_candidate(
        &store,
        allowed_context.id,
        allowed_entity.id,
        expected_fourth,
        blended_vector(7, 19, 0.40),
    );
    seed_candidate(
        &store,
        allowed_context.id,
        allowed_entity.id,
        expected_fifth,
        blended_vector(7, 20, 0.20),
    );
    for idx in 0..1008 {
        seed_candidate(
            &store,
            other_context.id,
            other_entity.id,
            EvidenceId::new_v7(),
            vector(idx),
        );
    }

    let hits = vector_search(
        &store,
        &vector(7),
        TEXT_SPACE,
        VectorSearchFilters {
            context_ids: Some(vec![allowed_context.id]),
            top_k: Some(5),
            ..Default::default()
        },
    )
    .expect("vector search");
    drop(store);
    let raw_db = Database::open(&db_path).expect("raw db");
    let raw_vector_query = raw_db
        .execute(
            "SELECT id FROM evidence WHERE context_id = $context_id ORDER BY vector_text <=> $q LIMIT 5",
            &std::collections::HashMap::from([
                ("context_id".to_string(), Value::Uuid(allowed_context.id.as_uuid())),
                ("q".to_string(), Value::Vector(vector(7))),
            ]),
        )
        .expect("raw vector query");
    let raw_explain = raw_db
        .explain("SELECT id FROM evidence WHERE context_id = $context_id ORDER BY vector_text <=> $q LIMIT 5")
        .expect("raw vector explain");
    let raw_ids = raw_vector_query
        .rows
        .iter()
        .filter_map(|row| match row.first() {
            Some(Value::Uuid(uuid)) => Some(EvidenceId::from(*uuid)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let hnsw_stats =
        raw_db.__debug_vector_hnsw_stats(VectorIndexRef::new("evidence", "vector_text"));
    let raw_hnsw_rows = raw_db
        .__debug_vector_hnsw_raw_search_for_test(
            VectorIndexRef::new("evidence", "vector_text"),
            &vector(7),
            8,
        )
        .unwrap_or_default();
    let expected_row_id = raw_db
        .scan("evidence", raw_db.snapshot())
        .expect("scan evidence")
        .into_iter()
        .find_map(|row| {
            (row.values.get("id") == Some(&Value::Uuid(expected.as_uuid()))).then_some(row.row_id)
        });
    let hit_ids = hits
        .iter()
        .filter_map(|hit| match hit.target {
            EmbeddingTarget::Evidence(id) => Some(id),
            _ => None,
        })
        .collect::<Vec<_>>();
    let first = hits.first();

    assert!(
        first.is_some_and(|hit| hit.target == EmbeddingTarget::Evidence(expected)
            && hit.context_id == Some(allowed_context.id)
            && hit.index_ref == "evidence.vector_text"
            && hit.planner_trace == format!("{:?}", raw_vector_query.trace)
            && hit.planner_trace.contains("IndexScan -> HNSWSearch")
            && hit.planner_trace.contains("__unique_context_id_id")
            && hit.candidate_count_before_filter >= 1000
            && hit.candidate_count_after_filter == 5)
            && raw_ids.len() == 5
            && raw_ids == hit_ids
            && hnsw_stats.is_some_and(|stats| stats.point_count >= 1013 && stats.dimension == 768)
            && expected_row_id.is_some_and(|expected_row_id| raw_hnsw_rows
                .iter()
                .any(|(row_id, _)| *row_id == expected_row_id))
            && raw_ids
                == vec![
                    expected,
                    expected_second,
                    expected_third,
                    expected_fourth,
                    expected_fifth
                ]
            && raw_vector_query.trace.physical_plan == "IndexScan -> HNSWSearch",
        "vector search must return the same ordered top-k ids as the raw contextdb HNSW query, carry that QueryResult.trace rather than a copied EXPLAIN string, and prove the searched VectorIndexRef has an HNSW graph; first={first:?}, hits={hits:?}, hit_ids={hit_ids:?}, raw_ids={raw_ids:?}, expected_row_id={expected_row_id:?}, raw_hnsw_rows={raw_hnsw_rows:?}, hnsw_stats={hnsw_stats:?}, raw_explain={raw_explain:?}, raw_trace={:?}",
        raw_vector_query.trace
    );
}

#[test]
fn embeddinggemma_disabled_config_returns_typed_error_not_zero_vector() {
    let embedder = EmbeddingGemmaTextEmbedder::new(EmbedderConfig::disabled());
    let first = embedder.embed(EmbeddingInput::Text("same input".to_string()));
    let second = embedder.embed(EmbeddingInput::Text("same input".to_string()));

    assert!(
        matches!(first, Err(context_graph::CgError::TextEmbedderDisabled))
            && matches!(second, Err(context_graph::CgError::TextEmbedderDisabled)),
        "disabled embedder must not synthesize deterministic zero vectors, first={first:?}, second={second:?}"
    );
}
