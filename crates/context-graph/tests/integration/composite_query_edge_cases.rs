use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use context_graph::test_support::embedding::{
    CompositeVectorClause, FusionPolicy, VectorSearchFilters, VectorSearchResult,
    composite_vector_search_with_policy, vector_search,
};
use context_graph::{
    CgError, ContextId, CreateContext, CreateEntity, DistanceMetric, EmbeddingId, EmbeddingSpace,
    EmbeddingSpaceBinding, EmbeddingTarget, EntityType, EvidenceId, EvidenceKind, EvidenceProducer,
    EvidenceRef, PrecomputedEmbedding, RecordObservation, RetentionStatus, Store, StoreConfig,
};
use contextdb_core::{Value, VectorIndexRef};
use contextdb_engine::{
    Database,
    plugin::{DatabasePlugin, QueryOutcome},
};

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";
const IMAGE_SPACE: &str = "image_clip_v1_4";

#[derive(Default)]
struct QueryCountingPlugin {
    enabled: AtomicBool,
    sql: Mutex<Vec<String>>,
}

impl QueryCountingPlugin {
    fn enable_for_next_call(&self) {
        self.sql.lock().expect("query log").clear();
        self.enabled.store(true, Ordering::SeqCst);
    }

    fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    fn sql(&self) -> Vec<String> {
        self.sql.lock().expect("query log").clone()
    }
}

impl DatabasePlugin for QueryCountingPlugin {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if self.enabled.load(Ordering::SeqCst) {
            self.sql.lock().expect("query log").push(sql.to_string());
        }
        Ok(())
    }

    fn post_query(&self, _sql: &str, _duration: Duration, _outcome: &QueryOutcome) {}
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "test-embedder".to_string(),
        model_version: "v1".to_string(),
        pipeline_version: "composite".to_string(),
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
        db_path: std::env::temp_dir().join(format!("composite-query-{}.db", uuid::Uuid::now_v7())),
        extra_embedding_spaces: vec![image_space()],
        ..Default::default()
    })
    .expect("open")
}

fn store_with_query_counter(query_counter: Arc<QueryCountingPlugin>) -> Store {
    Store::open_with_database_plugin(
        StoreConfig {
            db_path: std::env::temp_dir()
                .join(format!("composite-query-{}.db", uuid::Uuid::now_v7())),
            extra_embedding_spaces: vec![image_space()],
            ..Default::default()
        },
        query_counter,
    )
    .expect("open")
}

fn axis(dim: usize, idx: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dim];
    vector[idx] = 1.0;
    vector
}

fn unit_with_cosine(dim: usize, cosine_to_axis0: f32) -> Vec<f32> {
    let mut vector = vec![0.0; dim];
    vector[0] = cosine_to_axis0;
    vector[1] = (1.0 - cosine_to_axis0.powi(2)).sqrt();
    vector
}

fn score_close(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() <= 1.0e-4
}

fn mime_for_kind(kind: EvidenceKind) -> Option<String> {
    match kind {
        EvidenceKind::ImageFrame => Some("image/png".to_string()),
        EvidenceKind::StructuredSignal => Some("application/json".to_string()),
        EvidenceKind::Text => Some("text/plain".to_string()),
        _ => None,
    }
}

fn seed_evidence(
    store: &Store,
    space: &str,
    kind: EvidenceKind,
    vector: Vec<f32>,
) -> (ContextId, EvidenceId) {
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
    (context.id, evidence_id)
}

fn seed_text_distractors(store: &Store, count: usize) {
    let context = store
        .create_context(CreateContext {
            name: format!("ctx-distractors-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("distractor context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: format!("entity-distractors-{}", uuid::Uuid::now_v7()),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("distractor entity");

    for idx in 0..count {
        let evidence_id = EvidenceId::new_v7();
        let mut vector = vec![0.0; 768];
        vector[20 + (idx % 700)] = 1.0;
        let recorded = store.record_observation(RecordObservation {
            id: context_graph::ObservationId::new_v7(),
            entity_id: entity.id,
            context_id: context.id,
            observation_type: "embedding-distractor".to_string(),
            source: "test".to_string(),
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
        assert!(recorded.is_ok(), "seed distractor failed: {recorded:?}");
    }
}

fn seed_dual_space_evidence(
    store: &Store,
    label: &str,
    text_vector: Vec<f32>,
    image_vector: Vec<f32>,
) -> EvidenceId {
    let context = store
        .create_context(CreateContext {
            name: format!("dual-ctx-{label}-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: format!("dual-entity-{label}"),
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
        observation_type: "dual-space-candidate".to_string(),
        source: "test".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::ImageFrame,
            source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
            mime_type: Some("image/png".to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer(),
            text_projection: Some(format!("caption {label}")),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![
            PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: text_vector,
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer(),
            },
            PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: IMAGE_SPACE.to_string(),
                vector: image_vector,
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer(),
            },
        ],
    });
    assert!(recorded.is_ok(), "seed dual evidence failed: {recorded:?}");
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

#[test]
fn composite_vector_search_with_two_clauses_same_space_returns_typed_error_or_uses_explicit_fusion_policy()
 {
    let _alongside = crate::runs_alone::alongside_siblings();
    let store = store();
    let result = composite_vector_search_with_policy(
        &store,
        vec![
            CompositeVectorClause {
                query_vector: axis(768, 0),
                embedding_space_id: TEXT_SPACE.to_string(),
            },
            CompositeVectorClause {
                query_vector: axis(768, 1),
                embedding_space_id: TEXT_SPACE.to_string(),
            },
        ],
        FusionPolicy::WeightedSum(vec![(TEXT_SPACE.to_string(), 1.0)]),
        VectorSearchFilters::default(),
    );

    assert!(matches!(
        result,
        Err(context_graph::CgError::CompositeQueryClausesShareSpace { space_id })
            if space_id == TEXT_SPACE
    ));
}

#[test]
fn composite_vector_search_with_single_clause_still_requires_explicit_fusion_policy() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let compile_cases = trybuild::TestCases::new();
    compile_cases.compile_fail("tests/ui/composite_query_single_clause_without_fusion_policy.rs");

    let store = store();
    let (_, text_id) = seed_evidence(&store, TEXT_SPACE, EvidenceKind::Text, axis(768, 0));
    let hits = composite_vector_search_with_policy(
        &store,
        vec![CompositeVectorClause {
            query_vector: axis(768, 0),
            embedding_space_id: TEXT_SPACE.to_string(),
        }],
        FusionPolicy::WeightedSum(vec![(TEXT_SPACE.to_string(), 1.0)]),
        VectorSearchFilters {
            top_k: Some(1),
            ..Default::default()
        },
    )
    .expect("single-clause query with explicit fusion policy");

    assert!(
        hits.first()
            .is_some_and(|hit| hit.target == EmbeddingTarget::Evidence(text_id)
                && hit.embedding_space_id == TEXT_SPACE
                && hit.per_clause_scores == vec![(TEXT_SPACE.to_string(), 1.0)]
                && (hit.score - 1.0).abs() <= f32::EPSILON),
        "single-clause composite queries must still pass through explicit fusion policy math; callers who do not need fusion should use plain vector_search, hits={hits:?}, text_id={text_id:?}"
    );
}

#[test]
fn composite_vector_search_fused_rank_order_matches_explicit_fusion_policy_math() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let store = store();
    let best_id = seed_dual_space_evidence(
        &store,
        "best",
        unit_with_cosine(768, 0.80),
        unit_with_cosine(512, 0.20),
    );
    let middle_id = seed_dual_space_evidence(
        &store,
        "middle",
        unit_with_cosine(768, 0.60),
        unit_with_cosine(512, 0.60),
    );
    let image_heavy_id = seed_dual_space_evidence(
        &store,
        "image-heavy",
        unit_with_cosine(768, 0.40),
        unit_with_cosine(512, 0.90),
    );
    let hits = composite_vector_search_with_policy(
        &store,
        vec![
            CompositeVectorClause {
                query_vector: axis(768, 0),
                embedding_space_id: TEXT_SPACE.to_string(),
            },
            CompositeVectorClause {
                query_vector: axis(512, 0),
                embedding_space_id: IMAGE_SPACE.to_string(),
            },
        ],
        FusionPolicy::WeightedSum(vec![
            (TEXT_SPACE.to_string(), 0.7),
            (IMAGE_SPACE.to_string(), 0.3),
        ]),
        VectorSearchFilters {
            top_k: Some(3),
            ..Default::default()
        },
    )
    .expect("explicit fusion query");

    let best_hit = hits
        .iter()
        .find(|hit| hit.target == EmbeddingTarget::Evidence(best_id));
    let middle_hit = hits
        .iter()
        .find(|hit| hit.target == EmbeddingTarget::Evidence(middle_id));
    let image_heavy_hit = hits
        .iter()
        .find(|hit| hit.target == EmbeddingTarget::Evidence(image_heavy_id));
    assert!(
        best_hit.is_some() && middle_hit.is_some() && image_heavy_hit.is_some(),
        "composite fusion must return the exact persisted mixed-space targets, hits={hits:?}, best_id={best_id:?}, middle_id={middle_id:?}, image_heavy_id={image_heavy_id:?}"
    );
    let best_hit = best_hit.unwrap();
    let middle_hit = middle_hit.unwrap();
    let image_heavy_hit = image_heavy_hit.unwrap();

    assert!(
        hits.windows(2).all(|pair| pair[0].score >= pair[1].score)
            && [best_hit, middle_hit, image_heavy_hit]
                .iter()
                .all(|hit| hit.per_clause_scores.len() == 2)
            && close_enough(score_for(best_hit, TEXT_SPACE).unwrap_or(-1.0), 0.80)
            && close_enough(score_for(best_hit, IMAGE_SPACE).unwrap_or(-1.0), 0.20)
            && close_enough(best_hit.score, 0.62)
            && close_enough(score_for(middle_hit, TEXT_SPACE).unwrap_or(-1.0), 0.60)
            && close_enough(score_for(middle_hit, IMAGE_SPACE).unwrap_or(-1.0), 0.60)
            && close_enough(middle_hit.score, 0.60)
            && close_enough(score_for(image_heavy_hit, TEXT_SPACE).unwrap_or(-1.0), 0.40)
            && close_enough(
                score_for(image_heavy_hit, IMAGE_SPACE).unwrap_or(-1.0),
                0.90
            )
            && close_enough(image_heavy_hit.score, 0.55)
            && hits
                .iter()
                .map(|hit| hit.target.clone())
                .collect::<Vec<_>>()
                == vec![
                    EmbeddingTarget::Evidence(best_id),
                    EmbeddingTarget::Evidence(middle_id),
                    EmbeddingTarget::Evidence(image_heavy_id),
                ],
        "fused rank order must be the explicit weighted-sum math, not a canned or min/max score; hits={hits:?}"
    );
}

#[test]
fn vector_search_with_empty_corpus_returns_empty_results_not_panic() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let store = store();
    let hits = vector_search(
        &store,
        &axis(768, 0),
        TEXT_SPACE,
        VectorSearchFilters {
            top_k: Some(10),
            ..Default::default()
        },
    )
    .expect("empty corpus query");
    let unknown = vector_search(
        &store,
        &[0.0, 0.0],
        "empty_space_2d",
        VectorSearchFilters {
            top_k: Some(10),
            ..Default::default()
        },
    );

    assert!(
        hits.is_empty()
            && matches!(
                unknown,
                Err(CgError::UnknownEmbeddingSpace { ref id }) if id == "empty_space_2d"
            ),
        "registered empty corpus must return no hits, while an undeclared space must reject instead of masquerading as empty; hits={hits:?}, unknown={unknown:?}"
    );
}

#[test]
fn vector_search_with_empty_context_id_filter_returns_empty_not_unfiltered() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let store = store();
    let (context_id, evidence_id) =
        seed_evidence(&store, TEXT_SPACE, EvidenceKind::Text, axis(768, 0));
    let positive_hits = vector_search(
        &store,
        &axis(768, 0),
        TEXT_SPACE,
        VectorSearchFilters {
            context_ids: Some(vec![context_id]),
            top_k: Some(10),
            ..Default::default()
        },
    )
    .expect("positive context filter query");
    assert!(
        positive_hits.iter().any(|hit| {
            hit.target == EmbeddingTarget::Evidence(evidence_id)
                && hit.context_id == Some(context_id)
        }),
        "positive control must find the seeded evidence before the empty-filter check, positive_hits={positive_hits:?}, evidence_id={evidence_id:?}, context_id={context_id:?}"
    );

    let hits = vector_search(
        &store,
        &axis(768, 0),
        TEXT_SPACE,
        VectorSearchFilters {
            context_ids: Some(Vec::new()),
            top_k: Some(10),
            ..Default::default()
        },
    )
    .expect("empty context filter query");

    assert!(hits.is_empty());
}

#[test]
fn vector_search_top_k_ordering_matches_hnsw_documented_ranking() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let query_counter = Arc::new(QueryCountingPlugin::default());
    let store = store_with_query_counter(query_counter.clone());
    let (_, mid) = seed_evidence(
        &store,
        TEXT_SPACE,
        EvidenceKind::Text,
        unit_with_cosine(768, 0.75),
    );
    let (_, weak) = seed_evidence(
        &store,
        TEXT_SPACE,
        EvidenceKind::Text,
        unit_with_cosine(768, 0.10),
    );
    let (_, exact) = seed_evidence(&store, TEXT_SPACE, EvidenceKind::Text, axis(768, 0));
    let (_, far) = seed_evidence(
        &store,
        TEXT_SPACE,
        EvidenceKind::Text,
        unit_with_cosine(768, 0.45),
    );
    let (_, near) = seed_evidence(
        &store,
        TEXT_SPACE,
        EvidenceKind::Text,
        unit_with_cosine(768, 0.95),
    );
    seed_text_distractors(&store, 1008);
    let db_path = store.db_path().to_path_buf();
    drop(store);
    // Graph construction belongs to maintenance; the first query reads its saved result.
    let maintenance_db = Database::open(&db_path).expect("open for graph maintenance");
    let maintenance = maintenance_db
        .run_maintenance_cycle()
        .expect("build the seeded graph");
    assert!(maintenance.vector.built_indexes > 0, "{maintenance:?}");
    drop(maintenance_db);
    let reopened = Store::open_with_database_plugin(
        StoreConfig {
            db_path: db_path.clone(),
            extra_embedding_spaces: vec![image_space()],
            ..Default::default()
        },
        query_counter.clone(),
    )
    .expect("reopen");
    query_counter.enable_for_next_call();
    let hits = vector_search(
        &reopened,
        &axis(768, 0),
        TEXT_SPACE,
        VectorSearchFilters {
            top_k: Some(5),
            include_planner_trace: true,
            ..Default::default()
        },
    )
    .expect("ranked vector query");
    query_counter.disable();
    let public_sql = query_counter.sql();
    let targets = hits
        .iter()
        .map(|hit| hit.target.clone())
        .collect::<Vec<_>>();
    drop(reopened);
    let raw_db = Database::open(&db_path).expect("raw db");
    let raw_vector_query = raw_db
        .execute(
            "SELECT id FROM evidence ORDER BY vector_text <=> $q LIMIT 5",
            &HashMap::from([("q".to_string(), Value::Vector(axis(768, 0)))]),
        )
        .expect("raw vector query");
    let raw_targets = raw_vector_query
        .rows
        .iter()
        .filter_map(|row| match row.first() {
            Some(Value::Uuid(id)) => Some(EmbeddingTarget::Evidence(EvidenceId::from(*id))),
            _ => None,
        })
        .collect::<Vec<_>>();
    let hnsw_stats =
        raw_db.__debug_vector_hnsw_stats(VectorIndexRef::new("evidence", "vector_text"));
    let raw_trace = format!("{:?}", raw_vector_query.trace);

    assert!(
        targets
            == vec![
                EmbeddingTarget::Evidence(exact),
                EmbeddingTarget::Evidence(near),
                EmbeddingTarget::Evidence(mid),
                EmbeddingTarget::Evidence(far),
                EmbeddingTarget::Evidence(weak),
            ]
            && hits.first().is_some_and(|hit| hit.score > 0.99)
            && hits.windows(2).all(|pair| pair[0].score >= pair[1].score)
            && hits
                .iter()
                .zip([1.0, 0.95, 0.75, 0.45, 0.10])
                .all(|(hit, expected)| score_close(hit.score, expected))
            && hits
                .iter()
                .all(|hit| hit.index_ref == "evidence.vector_text"
                    && hit.planner_trace.as_str() == raw_trace.as_str()
                    && hit.planner_trace.contains("HNSWSearch"))
            && raw_targets == targets
            && raw_trace.contains("maintained_graph_ready")
            && public_sql.len() == 1
            && public_sql[0].contains("ORDER BY vector_text <=>")
            && public_sql[0].contains("LIMIT"),
        "top-k ordering must match the raw contextdb HNSW/vector-index query issued by vector_search on a reopened Store, not insertion order, canned exact-match rows, copied trace, or a HashMap full scan; rows were seeded mid/weak/exact/far/near plus 1008 zero-score distractors, hits={hits:?}, targets={targets:?}, raw_targets={raw_targets:?}, hnsw_stats={hnsw_stats:?}, public_sql={public_sql:?}, raw_trace={:?}",
        raw_trace
    );
}
