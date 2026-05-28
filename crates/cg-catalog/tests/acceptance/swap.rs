use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use cg_catalog::{
    Capability, Catalog, CatalogEntry, CgCatalogError, License, ModelKind, Source, SwapOption,
    SwapResult,
};
use context_graph::embedder::mock::MockEmbedder;
use context_graph::{
    CreateContext, CreateEntity, Decision, DistanceMetric, Embedder, EmbedderRegistration,
    EmbeddingId, EmbeddingInput, EmbeddingSpaceBinding, EmbeddingSpaceDecl, EmbeddingTarget,
    EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, ObservationId,
    PrecomputedEmbedding, RecordObservation, ReindexEvent, RetentionStatus, Store, StoreConfig,
    SubscriptionFilter,
};
use contextdb_core::{Value, VectorIndexRef};
use contextdb_engine::Database;

use crate::common::{ALT_1024_ID, NOMIC_ID, nomic_entry};

const CUSTOMER_INTERNAL_ID: &str = "customer-internal-text-v1";
const NOMIC_TEXT_SPACE_ID: &str = "text_default_nomic_v1_5_768";
const ALT_1024_TEXT_SPACE_ID: &str = "text_default_alt_1024_text_v1_1024";
const IMAGE_NOISE_SPACE_ID: &str = "swap-noise-image-v1-512";

#[derive(Clone, Debug)]
struct SeededTextVector {
    evidence_id: EvidenceId,
    text_projection: String,
    original_vector: Vec<f32>,
}

fn catalog_with_swap_entries(root: &std::path::Path) -> Catalog {
    let customer = CatalogEntry {
        id: CUSTOMER_INTERNAL_ID.to_string(),
        kind: ModelKind::TextEmbedder,
        capability: Capability {
            dimension: Some(768),
            output_shape: None,
            input_modality: Some("text".to_string()),
        },
        license: License {
            spdx: "Commercial".to_string(),
            url: "https://customer.example/terms".to_string(),
            requires_manual_acceptance: false,
        },
        source: Source::ApiEndpoint {
            base_url: "https://api.customer.example".to_string(),
            provider: "customer".to_string(),
            model: "text-embedding-v1".to_string(),
            auth_env_var: "CUSTOMER_API_KEY".to_string(),
        },
        byte_size: 0,
        local_path: None,
        artifact_filenames: Vec::new(),
        per_file_sha256: Vec::new(),
        ..CatalogEntry::default()
    };
    let alt = CatalogEntry {
        id: ALT_1024_ID.to_string(),
        capability: Capability {
            dimension: Some(1024),
            output_shape: None,
            input_modality: Some("text".to_string()),
        },
        ..nomic_entry(root)
    };
    Catalog::open_with_entries_against_root(root, "host-A", vec![nomic_entry(root), customer, alt])
        .expect("catalog opens with registered swap entries")
}

fn runtime_config(root: &std::path::Path) -> toml::Value {
    let text = std::fs::read_to_string(root.join("cg/catalog.toml")).unwrap_or_default();
    toml::from_str(&text).unwrap_or_else(|_| toml::Value::Table(Default::default()))
}

fn text_default(config: &toml::Value) -> Option<&str> {
    config
        .get("default_text_embedder")
        .and_then(toml::Value::as_str)
}

fn collect_new_decisions(store: &Store) -> Arc<Mutex<Vec<Decision>>> {
    let decisions = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&decisions);
    store
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), move |payload| {
            seen.lock()
                .expect("decision collector")
                .push(payload.decision.clone());
            Ok(())
        })
        .expect("subscribe new decision");
    decisions
}

fn wait_for_decisions(decisions: &Arc<Mutex<Vec<Decision>>>, expected: usize) -> Vec<Decision> {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let snapshot = decisions.lock().expect("decision collector").clone();
        if snapshot.len() >= expected || Instant::now() >= deadline {
            return snapshot;
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn assert_no_decisions(decisions: &Arc<Mutex<Vec<Decision>>>) {
    let deadline = Instant::now() + Duration::from_millis(250);
    while Instant::now() < deadline {
        assert!(
            decisions.lock().expect("decision collector").is_empty(),
            "refused dimension swaps must not record a model-swap decision in the supplied Store"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn assert_no_hidden_cache_store(root: &std::path::Path) {
    assert!(
        !root.join("context-graph.db").exists(),
        "set_default must use the caller-supplied Store, not create a hidden cache-root Store"
    );
}

fn has_store_swap_decision(
    decisions: &[Decision],
    from: &str,
    to: &str,
    option: Option<&str>,
) -> bool {
    let option_tag = format!("option:{}", option.unwrap_or("same-dimension"));
    decisions.iter().any(|decision| {
        decision.decision_type == "model-swap"
            && decision
                .tags
                .iter()
                .any(|tag| tag == &format!("from:{from}"))
            && decision.tags.iter().any(|tag| tag == &format!("to:{to}"))
            && decision.tags.iter().any(|tag| tag == &option_tag)
            && decision
                .reasoning
                .iter()
                .any(|step| step.details.contains(from) && step.details.contains(to))
    })
}

fn producer(stage: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: "cg-catalog-acceptance".to_string(),
        model_name: stage.to_string(),
        model_version: "test".to_string(),
        pipeline_version: "swap-test".to_string(),
    }
}

fn unit_vector(dimension: usize, axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];
    vector[axis % dimension] = 1.0;
    vector
}

fn image_noise_space() -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: IMAGE_NOISE_SPACE_ID.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "swap-noise-image".to_string(),
        model_version: "test".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: IMAGE_NOISE_SPACE_ID.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_vision".to_string(),
        }],
    }
}

fn separate_store_with_noise_space() -> (tempfile::TempDir, Store) {
    let temp = tempfile::tempdir().expect("store tempdir");
    let config = StoreConfig {
        db_path: temp.path().join("context-graph.db"),
        default_text_embedder: None,
        extra_embedding_spaces: vec![image_noise_space()],
        ..StoreConfig::default()
    };
    let embedder = MockEmbedder::new(NOMIC_TEXT_SPACE_ID, 768);
    let mut space = embedder.embedding_space();
    space.default_for_evidence_kind = Some(EvidenceKind::Text);
    let store = Store::open_with_embedder_registrations(
        config,
        vec![EmbedderRegistration {
            space,
            embedder: Arc::new(embedder),
        }],
    )
    .expect("store opens with real text-space reindex embedder");
    (temp, store)
}

fn seed_text_space_vectors(store: &Store, count: usize) -> Vec<SeededTextVector> {
    let context = store
        .create_context(CreateContext {
            name: format!("swap-vector-count-{count}-{}", ObservationId::new_v7()),
            ..Default::default()
        })
        .expect("context for seeded vectors");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Configuration,
            name: format!("swap-vector-entity-{count}"),
            context_id: context.id,
            ..Default::default()
        })
        .expect("entity for seeded vectors");

    let mut seeded = Vec::new();
    for index in 0..count {
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let text_projection = format!("seeded text vector {index}");
        let original_vector = unit_vector(768, index);
        let recorded = store.record_observation(RecordObservation {
            id: observation_id,
            entity_id: entity.id,
            context_id: context.id,
            observation_type: "swap-vector-count-seed".to_string(),
            source: "cg-catalog-acceptance".to_string(),
            observed_at: chrono::Utc::now(),
            evidence: vec![EvidenceRef {
                id: evidence_id,
                observation_id,
                context_id: context.id,
                kind: EvidenceKind::Text,
                source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
                mime_type: Some("text/plain".to_string()),
                producer: producer("capture"),
                text_projection: Some(text_projection.clone()),
                retention_status: RetentionStatus::RetainedExternal,
                ..EvidenceRef::default()
            }],
            observed_properties: Default::default(),
            state_delta: Default::default(),
            properties: Default::default(),
            embeddings: vec![PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: NOMIC_TEXT_SPACE_ID.to_string(),
                vector: original_vector.clone(),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer("nomic-seed"),
            }],
        });
        assert!(
            recorded.is_ok(),
            "seeding vector {index} through the supplied Store should succeed: {recorded:?}"
        );
        seeded.push(SeededTextVector {
            evidence_id,
            text_projection,
            original_vector,
        });
    }
    seeded
}

fn seed_non_swapped_image_vectors(store: &Store, count: usize) {
    let context = store
        .create_context(CreateContext {
            name: format!("swap-noise-count-{count}-{}", ObservationId::new_v7()),
            ..Default::default()
        })
        .expect("context for non-swapped vectors");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Configuration,
            name: format!("swap-noise-entity-{count}"),
            context_id: context.id,
            ..Default::default()
        })
        .expect("entity for non-swapped vectors");

    for index in 0..count {
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let recorded = store.record_observation(RecordObservation {
            id: observation_id,
            entity_id: entity.id,
            context_id: context.id,
            observation_type: "swap-non-swapped-vector-noise".to_string(),
            source: "cg-catalog-acceptance".to_string(),
            observed_at: chrono::Utc::now(),
            evidence: vec![EvidenceRef {
                id: evidence_id,
                observation_id,
                context_id: context.id,
                kind: EvidenceKind::ImageFrame,
                source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
                mime_type: Some("image/png".to_string()),
                producer: producer("noise-capture"),
                retention_status: RetentionStatus::RetainedExternal,
                ..EvidenceRef::default()
            }],
            observed_properties: Default::default(),
            state_delta: Default::default(),
            properties: Default::default(),
            embeddings: vec![PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: IMAGE_NOISE_SPACE_ID.to_string(),
                vector: unit_vector(512, index),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer("image-noise-seed"),
            }],
        });
        assert!(
            recorded.is_ok(),
            "seeding non-swapped image vector {index} through the supplied Store should succeed: {recorded:?}"
        );
    }
}

fn assert_dimension_gate_reports_live_count(seed_count: u64) {
    let temp = tempfile::tempdir().expect("tempdir");
    let (_store_temp, store) = separate_store_with_noise_space();
    seed_text_space_vectors(&store, seed_count as usize);
    seed_non_swapped_image_vectors(&store, seed_count as usize + 11);
    let decisions = collect_new_decisions(&store);
    let catalog = catalog_with_swap_entries(temp.path());
    let before_config = runtime_config(temp.path());
    let before_store_bytes = std::fs::read(store.db_path()).expect("store bytes readable");

    let result = catalog.set_default(&store, ModelKind::TextEmbedder, ALT_1024_ID, None);
    let after_config = runtime_config(temp.path());
    let after_store_bytes = std::fs::read(store.db_path()).expect("store bytes readable");

    let Err(CgCatalogError::DimensionChangeRequiresConfirmation {
        current_dim,
        proposed_dim,
        options,
    }) = result
    else {
        panic!("dimension-changing swap should require explicit confirmation, got {result:?}");
    };
    assert_eq!(current_dim, 768);
    assert_eq!(proposed_dim, 1024);
    assert!(
        options.contains(&SwapOption::Reindex {
            affected_vectors: seed_count
        }),
        "dimension gate should report the live supplied-Store vector count {seed_count}, got {options:?}"
    );
    assert!(options.contains(&SwapOption::ParallelSpace));
    assert_eq!(
        after_config, before_config,
        "refused dimension swaps must not move the durable default"
    );
    assert_eq!(
        after_store_bytes, before_store_bytes,
        "refused dimension swaps must not rewrite vector storage while counting"
    );
    assert_no_decisions(&decisions);
    assert_no_hidden_cache_store(temp.path());
}

fn assert_raw_1024_entity_reference_query_hits(
    db_path: &std::path::Path,
    embedding_id: EmbeddingId,
    query_vector: &[f32],
) {
    let db = Database::open(db_path).expect("raw database opens for vector query");
    let snapshot = db.snapshot();
    let row_id = db
        .scan("entity_reference_embeddings_1024", snapshot)
        .expect("raw 1024 entity-reference scan")
        .into_iter()
        .find_map(|row| {
            let matches_id = row.values.get("id") == Some(&Value::Uuid(embedding_id.as_uuid()));
            let matches_space = row.values.get("embedding_space_id")
                == Some(&Value::Text(ALT_1024_TEXT_SPACE_ID.to_string()));
            (matches_id && matches_space).then_some(row.row_id)
        })
        .expect("new parallel-space entity-reference row");
    let hits = db
        .query_vector(
            VectorIndexRef::new("entity_reference_embeddings_1024", "vector"),
            query_vector,
            1,
            None,
            snapshot,
        )
        .expect("raw 1024 entity-reference vector query");
    assert!(
        hits.first()
            .is_some_and(|(hit_row, score)| *hit_row == row_id && *score > 0.99),
        "new parallel space must be queryable through the 1024-dimension vector index, row_id={row_id:?}, hits={hits:?}"
    );
}

fn assert_seeded_text_vectors_reindexed(db_path: &std::path::Path, seeded: &[SeededTextVector]) {
    let db = Database::open(db_path).expect("raw database opens for text-vector query");
    let snapshot = db.snapshot();
    let evidence_rows = db
        .scan("evidence", snapshot)
        .expect("raw evidence scan")
        .into_iter()
        .filter_map(|row| match row.values.get("id") {
            Some(Value::Uuid(id)) => Some((*id, row.row_id)),
            _ => None,
        })
        .collect::<std::collections::HashMap<_, _>>();
    let embedder = MockEmbedder::new(NOMIC_TEXT_SPACE_ID, 768);
    for seed in seeded {
        let expected = embedder
            .embed(EmbeddingInput::Text(seed.text_projection.clone()))
            .expect("mock reindex embedding")
            .vector;
        assert_ne!(
            expected, seed.original_vector,
            "fixture must distinguish generated reindex output from the original precomputed vector"
        );
        let row_id = evidence_rows
            .get(&seed.evidence_id.as_uuid())
            .copied()
            .expect("seeded evidence row exists");
        let hits = db
            .query_vector(
                VectorIndexRef::new("evidence", "vector_text"),
                &expected,
                seeded.len(),
                None,
                snapshot,
            )
            .expect("raw text-vector query");
        assert!(
            hits.iter()
                .any(|(hit_row, score)| *hit_row == row_id && *score > 0.99),
            "reindex confirmation must replace seeded evidence vectors through the Store reindex path, evidence_id={}, hits={hits:?}",
            seed.evidence_id
        );
    }
}

fn assert_parallel_space_vector_index_is_queryable(store: Store) {
    let db_path = store.db_path().to_path_buf();
    let context = store
        .create_context(CreateContext {
            name: format!("parallel-space-proof-{}", ObservationId::new_v7()),
            ..Default::default()
        })
        .expect("context for parallel-space proof");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Configuration,
            name: "parallel-space-proof".to_string(),
            context_id: context.id,
            ..Default::default()
        })
        .expect("entity for parallel-space proof");
    let vector = unit_vector(1024, 17);
    let recorded = store.enroll_entity_reference(
        entity.id,
        ALT_1024_TEXT_SPACE_ID,
        Some("parallel-space-ann-proof".to_string()),
        vector.clone(),
        producer("alt-1024-entity-reference"),
    );
    assert!(
        recorded.as_ref().is_ok_and(EmbeddingId::is_valid),
        "new parallel space must accept an actual 1024-dimension vector-index write through the Store substrate: {recorded:?}"
    );
    let embedding_id = recorded.expect("embedding id");
    let rows = store
        .list_entity_reference_embeddings_for_entity(entity.id)
        .expect("entity-reference rows should be listed through Store");
    assert!(
        rows.iter().any(|row| row.id == embedding_id
            && row.embedding_space_id == ALT_1024_TEXT_SPACE_ID
            && row.vector == vector),
        "new parallel space must persist a 1024-dimension entity-reference vector row, rows={rows:?}"
    );
    drop(store);
    assert_raw_1024_entity_reference_query_hits(&db_path, embedding_id, &vector);
}

fn assert_store_has_old_and_parallel_text_spaces(store: &Store) {
    let spaces = store
        .list_embedding_spaces()
        .expect("embedding spaces should be readable from the supplied Store");
    assert!(
        spaces
            .iter()
            .any(|space| space.embedding_space_id == NOMIC_TEXT_SPACE_ID && space.dimension == 768),
        "parallel-space swap should preserve the old text embedding space in the Store substrate"
    );
    assert!(
        spaces
            .iter()
            .any(|space| space.embedding_space_id == ALT_1024_TEXT_SPACE_ID
                && space.dimension == 1024),
        "parallel-space swap should create the requested 1024-dimension space in the Store substrate"
    );
}

#[test]
fn swap_default_same_dimension_completes_without_extra_confirmation_and_records_versioned_decision()
{
    // Same-dimension swaps should be reversible operational choices with an auditable decision.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_store_temp, store) = separate_store_with_noise_space();
    let decisions = collect_new_decisions(&store);
    let catalog = catalog_with_swap_entries(temp.path());
    let result = catalog
        .set_default(&store, ModelKind::TextEmbedder, CUSTOMER_INTERNAL_ID, None)
        .expect("swap returns");
    let reopened = catalog_with_swap_entries(temp.path());
    let config = runtime_config(temp.path());
    assert_eq!(text_default(&config), Some(CUSTOMER_INTERNAL_ID));
    let decision_snapshot = wait_for_decisions(&decisions, 1);
    assert!(
        has_store_swap_decision(&decision_snapshot, NOMIC_ID, CUSTOMER_INTERNAL_ID, None),
        "same-dimension swap should append a structured model-swap decision to the supplied Store"
    );
    assert_no_hidden_cache_store(temp.path());
    assert_eq!(
        runtime_config(temp.path()),
        config,
        "reopening the catalog must preserve the durable default and decision"
    );
    let _ = reopened.list().expect("reopened catalog reads");
    assert_eq!(result, SwapResult::Completed);
}

#[test]
fn swap_default_different_dimension_refuses_without_explicit_user_confirmation_and_surfaces_reindex_vs_parallel_space_gate()
 {
    // Dimension changes must not silently corrupt existing vector storage.
    assert_dimension_gate_reports_live_count(7);
    assert_dimension_gate_reports_live_count(3);
}

#[test]
fn swap_default_different_dimension_with_explicit_reindex_confirmation_proceeds_and_records_decision_with_reindex_option()
 {
    // Choosing reindex should be visible in the recorded model-swap decision.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_store_temp, store) = separate_store_with_noise_space();
    let seeded_text_vectors = seed_text_space_vectors(&store, 7);
    seed_non_swapped_image_vectors(&store, 18);
    let db_path = store.db_path().to_path_buf();
    let decisions = collect_new_decisions(&store);
    let catalog = catalog_with_swap_entries(temp.path());
    let result = catalog
        .set_default(
            &store,
            ModelKind::TextEmbedder,
            ALT_1024_ID,
            Some(SwapOption::Reindex {
                affected_vectors: 999,
            }),
        )
        .expect("swap returns");
    let config = runtime_config(temp.path());
    assert_eq!(text_default(&config), Some(ALT_1024_ID));
    let decision_snapshot = wait_for_decisions(&decisions, 1);
    assert!(
        has_store_swap_decision(&decision_snapshot, NOMIC_ID, ALT_1024_ID, Some("Reindex")),
        "reindex swap should append a structured decision naming the option to the supplied Store"
    );
    assert_no_hidden_cache_store(temp.path());
    let reindex_events = store.drain_reindex_events();
    assert!(
        reindex_events.iter().any(|event| matches!(
            event,
            ReindexEvent::Completed {
                embedding_space_id,
                rows_processed: 7,
                ..
            } if embedding_space_id == NOMIC_TEXT_SPACE_ID
        )),
        "reindex confirmation must drive the Store reindex path and report the live text-space count, not echo the caller-supplied count; events={reindex_events:?}"
    );
    drop(store);
    assert_seeded_text_vectors_reindexed(&db_path, &seeded_text_vectors);
    assert_eq!(
        result,
        SwapResult::CompletedWithReindex {
            reindexed_vectors: 7
        }
    );
}

#[test]
fn swap_default_different_dimension_with_explicit_parallel_space_confirmation_proceeds_and_creates_new_space()
 {
    // Choosing a parallel space should preserve the old vector space and create the new one.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_store_temp, store) = separate_store_with_noise_space();
    seed_text_space_vectors(&store, 7);
    seed_non_swapped_image_vectors(&store, 18);
    let decisions = collect_new_decisions(&store);
    let catalog = catalog_with_swap_entries(temp.path());
    let result = catalog
        .set_default(
            &store,
            ModelKind::TextEmbedder,
            ALT_1024_ID,
            Some(SwapOption::ParallelSpace),
        )
        .expect("swap returns");
    let config = runtime_config(temp.path());
    assert_eq!(text_default(&config), Some(ALT_1024_ID));
    let decision_snapshot = wait_for_decisions(&decisions, 1);
    assert!(
        has_store_swap_decision(
            &decision_snapshot,
            NOMIC_ID,
            ALT_1024_ID,
            Some("ParallelSpace")
        ),
        "parallel-space swap should append a structured decision naming the option to the supplied Store"
    );
    assert_no_hidden_cache_store(temp.path());
    assert_store_has_old_and_parallel_text_spaces(&store);
    assert_parallel_space_vector_index_is_queryable(store);
    assert_eq!(
        result,
        SwapResult::CompletedAsParallelSpace {
            new_space_id: "text_default_alt_1024_text_v1_1024".to_string()
        }
    );
}

#[test]
fn swap_default_back_to_original_after_parallel_space_creation_is_same_dimension_path() {
    // Reverting to the prior default should use the safe same-dimension path.
    let temp = tempfile::tempdir().expect("tempdir");
    let (_store_temp, store) = separate_store_with_noise_space();
    seed_text_space_vectors(&store, 7);
    seed_non_swapped_image_vectors(&store, 18);
    let decisions = collect_new_decisions(&store);
    let catalog = catalog_with_swap_entries(temp.path());
    let _ = catalog
        .set_default(
            &store,
            ModelKind::TextEmbedder,
            ALT_1024_ID,
            Some(SwapOption::ParallelSpace),
        )
        .expect("parallel-space setup returns");
    let result = catalog
        .set_default(&store, ModelKind::TextEmbedder, NOMIC_ID, None)
        .expect("swap returns");
    let config = runtime_config(temp.path());
    assert_eq!(text_default(&config), Some(NOMIC_ID));
    let decision_snapshot = wait_for_decisions(&decisions, 2);
    assert!(
        has_store_swap_decision(&decision_snapshot, ALT_1024_ID, NOMIC_ID, None),
        "reverting should append a same-dimension structured model-swap decision to the supplied Store"
    );
    assert_no_hidden_cache_store(temp.path());
    assert_store_has_old_and_parallel_text_spaces(&store);
    assert_parallel_space_vector_index_is_queryable(store);
    assert_eq!(result, SwapResult::Completed);
}
