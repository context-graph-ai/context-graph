use std::collections::{BTreeMap, HashMap};

use chrono::Utc;
use context_graph::test_support::embedding::{VectorSearchFilters, vector_search};
use context_graph::{
    CreateContext, CreateEntity, EmbeddingId, EmbeddingTarget, EntityType, EvidenceId,
    EvidenceKind, EvidenceProducer, EvidenceRef, PrecomputedEmbedding, RecordObservation,
    RetentionStatus, Store, StoreConfig,
};
use contextdb_core::{Error as EngineError, Value};
use contextdb_engine::Database;

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "embeddinggemma-300m".to_string(),
        model_version: "v1".to_string(),
        pipeline_version: "same-space".to_string(),
    }
}

fn vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[axis] = 1.0;
    vector
}

fn mime_for_kind(kind: EvidenceKind) -> Option<String> {
    match kind {
        EvidenceKind::StructuredSignal => Some("application/json".to_string()),
        EvidenceKind::Text => Some("text/plain".to_string()),
        _ => None,
    }
}

fn seed(store: &Store, kind: EvidenceKind, axis: usize) -> EvidenceId {
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
        observation_type: "same-space-modality".to_string(),
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
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: vector(axis),
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer(),
        }],
    });
    assert!(recorded.is_ok(), "seed failed: {recorded:?}");
    evidence_id
}

fn kind_for(db: &Database, evidence_id: EvidenceId) -> Option<String> {
    let result = db
        .execute(
            "SELECT kind FROM evidence WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Uuid(evidence_id.as_uuid()))]),
        )
        .expect("kind query");
    result.rows.first().and_then(|row| match row.first() {
        Some(Value::Text(kind)) => Some(kind.clone()),
        _ => None,
    })
}

fn hit_score(
    hits: &[context_graph::test_support::embedding::VectorSearchResult],
    evidence_id: EvidenceId,
) -> Option<f32> {
    hits.iter()
        .find_map(|hit| (hit.target == EmbeddingTarget::Evidence(evidence_id)).then_some(hit.score))
}

/// @invariant 15b
#[test]
fn evidence_kind_rename_inside_one_embedding_space_does_not_change_vector_search_results() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let db_path =
        std::env::temp_dir().join(format!("intra-space-modality-{}.db", uuid::Uuid::now_v7()));
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let text_id = seed(&store, EvidenceKind::Text, 0);
    let signal_id = seed(&store, EvidenceKind::StructuredSignal, 1);
    let text_hits = vector_search(
        &store,
        &vector(0),
        TEXT_SPACE,
        VectorSearchFilters {
            top_k: Some(2),
            ..Default::default()
        },
    )
    .expect("text-axis vector search");
    let signal_hits = vector_search(
        &store,
        &vector(1),
        TEXT_SPACE,
        VectorSearchFilters {
            top_k: Some(2),
            ..Default::default()
        },
    )
    .expect("structured-axis vector search");
    drop(store);
    let db = Database::open(&db_path).expect("raw db");
    let text_kind = kind_for(&db, text_id);
    let signal_kind = kind_for(&db, signal_id);
    let text_self_score = hit_score(&text_hits, text_id);
    let text_cross_score = hit_score(&text_hits, signal_id);
    let signal_self_score = hit_score(&signal_hits, signal_id);
    let signal_cross_score = hit_score(&signal_hits, text_id);
    let relabel = db.execute(
        "UPDATE evidence SET kind = $kind WHERE id = $id",
        &HashMap::from([
            (
                "kind".to_string(),
                Value::Text("structured_signal".to_string()),
            ),
            ("id".to_string(), Value::Uuid(text_id.as_uuid())),
        ]),
    );

    assert!(
        text_kind.as_deref() == Some("text")
            && signal_kind.as_deref() == Some("structured_signal")
            && matches!(relabel, Err(EngineError::ImmutableTable(ref table)) if table == "evidence")
            && text_hits.iter().map(|hit| &hit.target).collect::<Vec<_>>()
                == vec![
                    &EmbeddingTarget::Evidence(text_id),
                    &EmbeddingTarget::Evidence(signal_id),
                ]
            && signal_hits
                .iter()
                .map(|hit| &hit.target)
                .collect::<Vec<_>>()
                == vec![
                    &EmbeddingTarget::Evidence(signal_id),
                    &EmbeddingTarget::Evidence(text_id),
                ]
            && text_self_score.is_some_and(|score| score > 0.99)
            && signal_self_score.is_some_and(|score| score > 0.99)
            && text_cross_score.is_some_and(|score| score.abs() <= 1.0e-5)
            && signal_cross_score.is_some_and(|score| score.abs() <= 1.0e-5)
            && text_hits
                .iter()
                .chain(signal_hits.iter())
                .all(|hit| hit.embedding_space_id == TEXT_SPACE),
        "evidence kind is immutable, so the product-correct guard is: persisted mixed kinds are engine-observable, relabel is rejected, and same-space vector search ranks both kinds by vector score instead of branching by EvidenceKind or stamping every score to 1.0; text_kind={text_kind:?}, signal_kind={signal_kind:?}, relabel={relabel:?}, text_hits={text_hits:?}, signal_hits={signal_hits:?}, text_self_score={text_self_score:?}, text_cross_score={text_cross_score:?}, signal_self_score={signal_self_score:?}, signal_cross_score={signal_cross_score:?}"
    );
}
