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
    EmbeddingId, EmbeddingRole, EmbeddingSpaceBinding, EmbeddingSpaceDecl, EmbeddingTarget,
    EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, ModelIdentity,
    ObservationId, PrecomputedEmbedding, RecordObservation, ReindexEvent, RetentionStatus,
    SpaceState, SpaceStatus, Store, StoreConfig, SubscriptionFilter, VectorPlacement,
};
use contextdb_core::Value;
use contextdb_engine::Database;

use crate::common::{ALT_1024_ID, NOMIC_ID, nomic_entry};

const CUSTOMER_INTERNAL_ID: &str = "customer-internal-text-v1";
const IMAGE_NOISE_SPACE_ID: &str = "swap-noise-image-v1-512";

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
        placement: Some(VectorPlacement::TravelsWithItem),
        media_reachable_on_every_node: false,
        producer: image_noise_identity(),
    }
}

/// The caller-filled noise space's one model: the seed producer's own fields.
fn image_noise_identity() -> ModelIdentity {
    ModelIdentity {
        model: "image-noise-seed".to_string(),
        weights_revision: "test".to_string(),
        preparation: "swap-test".to_string(),
        dimension: 512,
    }
}

/// The space the catalog's model `model_id` embeds into.
fn text_space_of(catalog: &Catalog, model_id: &str) -> String {
    catalog
        .text_embedder_producer(model_id)
        .expect("catalog names the model's identity")
        .space_id()
}

/// A store whose two text models stand in for the catalog's Nomic and alt entries, so the
/// catalog's identities and the store's registered embedders name the same spaces. Nomic's
/// stand-in holds the text role; the alt stand-in claims no role until a swap switches to it.
fn separate_store_with_noise_space(catalog: &Catalog) -> (tempfile::TempDir, Store) {
    let temp = tempfile::tempdir().expect("store tempdir");
    let config = StoreConfig {
        db_path: temp.path().join("context-graph.db"),
        default_text_embedder: None,
        extra_embedding_spaces: vec![image_noise_space()],
        ..StoreConfig::default()
    };
    let registrations = [(NOMIC_ID, Some(EvidenceKind::Text)), (ALT_1024_ID, None)]
        .into_iter()
        .map(|(model_id, default_for)| {
            let embedder = MockEmbedder::with_identity(
                catalog
                    .text_embedder_producer(model_id)
                    .expect("catalog names the model's identity"),
            );
            let mut space = embedder.embedding_space();
            space.default_for_evidence_kind = default_for;
            EmbedderRegistration {
                space,
                embedder: Arc::new(embedder),
            }
        })
        .collect();
    let store = Store::open_with_embedder_registrations(config, registrations)
        .expect("store opens with both text models' embedders");
    (temp, store)
}

fn status_of(store: &Store, space_id: &str) -> SpaceStatus {
    store
        .embedding_space_status()
        .expect("embedding space status")
        .into_iter()
        .find(|status| status.space_id == space_id)
        .unwrap_or_else(|| panic!("status names space {space_id}"))
}

fn active_text_space(store: &Store) -> Option<String> {
    store
        .active_embedding_space(EmbeddingRole::EvidenceText)
        .expect("read the active text space")
}

/// Node-local vector rows each seeded item has in `space_id`.
fn rows_per_item(db_path: &std::path::Path, seeded: &[EvidenceId], space_id: &str) -> Vec<i64> {
    let db = Database::open(db_path).expect("raw database opens for vector rows");
    seeded
        .iter()
        .map(|evidence_id| {
            let result = db
                .execute(
                    "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE target_id = $target_id AND embedding_space_id = $space",
                    &[
                        (
                            "target_id".to_string(),
                            Value::Uuid(evidence_id.as_uuid()),
                        ),
                        ("space".to_string(), Value::Text(space_id.to_string())),
                    ]
                    .into_iter()
                    .collect(),
                )
                .expect("vector-row count");
            match result.rows.first().and_then(|row| row.first()) {
                Some(Value::Int64(n)) => *n,
                other => panic!("expected a count, got {other:?}"),
            }
        })
        .collect()
}

/// Text items the active model embeds at write; returns their ids.
fn seed_text_space_vectors(store: &Store, count: usize) -> Vec<EvidenceId> {
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
                text_projection: Some(text_projection),
                retention_status: RetentionStatus::RetainedExternal,
                ..EvidenceRef::default()
            }],
            observed_properties: Default::default(),
            state_delta: Default::default(),
            properties: Default::default(),
            embeddings: Vec::new(),
        });
        assert!(
            recorded.is_ok(),
            "seeding vector {index} through the supplied Store should succeed: {recorded:?}"
        );
        seeded.push(evidence_id);
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
    let catalog = catalog_with_swap_entries(temp.path());
    let (_store_temp, store) = separate_store_with_noise_space(&catalog);
    seed_text_space_vectors(&store, seed_count as usize);
    seed_non_swapped_image_vectors(&store, seed_count as usize + 11);
    let decisions = collect_new_decisions(&store);
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

/// After a parallel-space swap the new space is the text role's active space: a new text item
/// is embedded into it.
fn assert_parallel_space_vector_index_is_queryable(store: &Store, catalog: &Catalog) {
    let alt_space = text_space_of(catalog, ALT_1024_ID);
    let before = status_of(store, &alt_space).embedded;
    let seeded = seed_text_space_vectors(store, 1);
    assert_eq!(
        status_of(store, &alt_space).embedded,
        before + 1,
        "the new parallel space must take new text items through the store's embedder"
    );
    assert_eq!(
        rows_per_item(store.db_path(), &seeded, &alt_space),
        vec![1],
        "the new item's vector row must land in the new parallel space"
    );
}

fn assert_store_has_old_and_parallel_text_spaces(store: &Store, catalog: &Catalog) {
    let nomic_space = text_space_of(catalog, NOMIC_ID);
    let alt_space = text_space_of(catalog, ALT_1024_ID);
    let spaces = store
        .list_embedding_spaces()
        .expect("embedding spaces should be readable from the supplied Store");
    assert!(
        spaces
            .iter()
            .any(|space| space.embedding_space_id == nomic_space && space.dimension == 768),
        "parallel-space swap should preserve the old text embedding space in the Store substrate"
    );
    assert!(
        spaces
            .iter()
            .any(|space| space.embedding_space_id == alt_space && space.dimension == 1024),
        "parallel-space swap should create the requested 1024-dimension space in the Store substrate"
    );
}

#[test]
fn swap_default_same_dimension_completes_without_extra_confirmation_and_records_versioned_decision()
{
    // Same-dimension swaps should be reversible operational choices with an auditable decision.
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = catalog_with_swap_entries(temp.path());
    let (_store_temp, store) = separate_store_with_noise_space(&catalog);
    let decisions = collect_new_decisions(&store);
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
    let catalog = catalog_with_swap_entries(temp.path());
    let (_store_temp, store) = separate_store_with_noise_space(&catalog);
    let nomic_space = text_space_of(&catalog, NOMIC_ID);
    let alt_space = text_space_of(&catalog, ALT_1024_ID);
    let seeded = seed_text_space_vectors(&store, 7);
    seed_non_swapped_image_vectors(&store, 18);
    let db_path = store.db_path().to_path_buf();
    let decisions = collect_new_decisions(&store);
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
                embedded: 7,
                skipped: 0,
            } if *embedding_space_id == alt_space
        )),
        "reindex confirmation must drive the Store reindex path into the new model's space and report the live embedded count, not echo the caller-supplied count; events={reindex_events:?}"
    );
    assert_eq!(active_text_space(&store), Some(alt_space.clone()));
    assert_eq!(status_of(&store, &alt_space).embedded, 7);
    assert_eq!(status_of(&store, &nomic_space).state, SpaceState::Stale);
    drop(store);
    assert_eq!(
        rows_per_item(&db_path, &seeded, &alt_space),
        vec![1; 7],
        "every seeded item must have its vector rebuilt in the new model's space"
    );
    assert_eq!(
        rows_per_item(&db_path, &seeded, &nomic_space),
        vec![0; 7],
        "each stale vector row must be gone once its replacement is written"
    );
    assert_eq!(
        result,
        SwapResult::CompletedWithReindex {
            reindexed_vectors: 7,
            preserved_elsewhere: 0,
            skipped: 0,
        }
    );
}

#[test]
fn swap_default_different_dimension_with_explicit_parallel_space_confirmation_proceeds_and_creates_new_space()
 {
    // Choosing a parallel space should preserve the old vector space and create the new one.
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = catalog_with_swap_entries(temp.path());
    let (_store_temp, store) = separate_store_with_noise_space(&catalog);
    let nomic_space = text_space_of(&catalog, NOMIC_ID);
    let alt_space = text_space_of(&catalog, ALT_1024_ID);
    seed_text_space_vectors(&store, 7);
    seed_non_swapped_image_vectors(&store, 18);
    let decisions = collect_new_decisions(&store);
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
    assert_store_has_old_and_parallel_text_spaces(&store, &catalog);
    assert_eq!(active_text_space(&store), Some(alt_space.clone()));
    let old = status_of(&store, &nomic_space);
    assert!(
        old.state == SpaceState::Retained && old.embedded == 7,
        "a parallel-space swap keeps the old space retained with its vectors, old={old:?}"
    );
    assert_parallel_space_vector_index_is_queryable(&store, &catalog);
    assert_eq!(
        result,
        SwapResult::CompletedAsParallelSpace {
            new_space_id: alt_space
        }
    );
}

#[test]
fn swap_default_back_to_original_after_parallel_space_creation_is_same_dimension_path() {
    // Reverting to the prior default should use the safe same-dimension path.
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = catalog_with_swap_entries(temp.path());
    let (_store_temp, store) = separate_store_with_noise_space(&catalog);
    let nomic_space = text_space_of(&catalog, NOMIC_ID);
    let alt_space = text_space_of(&catalog, ALT_1024_ID);
    seed_text_space_vectors(&store, 7);
    seed_non_swapped_image_vectors(&store, 18);
    let decisions = collect_new_decisions(&store);
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
    assert_store_has_old_and_parallel_text_spaces(&store, &catalog);
    assert_eq!(
        active_text_space(&store),
        Some(nomic_space.clone()),
        "reverting makes the original model's space active again"
    );
    let original = status_of(&store, &nomic_space);
    assert!(
        original.state == SpaceState::Active && original.embedded == 7,
        "the original space's vectors are intact after the round trip, original={original:?}"
    );
    assert_eq!(status_of(&store, &alt_space).state, SpaceState::Retained);
    assert_eq!(result, SwapResult::Completed);
}
