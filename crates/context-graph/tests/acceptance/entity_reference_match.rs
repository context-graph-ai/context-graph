//! Open-set entity-reference matching through the public consumer surface.
//!
//! An embedded consumer that already holds a vector (a vision pipeline at a camera
//! site) asks "which enrolled entity does this vector match?" and gets back ranked
//! entities with cosine scores — or nothing, honestly, when no enrolled reference
//! clears the caller's threshold. Everything in this module goes through the same
//! public API an external crate sees in a normal build: no test-support cfg, no
//! internal modules, no raw SQL.

use std::collections::BTreeMap;

use context_graph::{
    CgError, CreateContext, CreateEntity, DistanceMetric, EmbedderConfig, EmbeddingInput,
    EmbeddingSpaceDecl, EntityId, EntityMatch, EntityReferenceMatchOptions, EntityType,
    EvidenceProducer, Store, StoreConfig,
};

const VISION_SPACE: &str = "site_vision_768";
const DIM: usize = 768;

fn store(name: &str) -> Store {
    Store::open(StoreConfig {
        db_path: std::env::temp_dir().join(format!("{name}-{}.db", uuid::Uuid::now_v7())),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        extra_embedding_spaces: vec![EmbeddingSpaceDecl {
            embedding_space_id: VISION_SPACE.to_string(),
            dimension: DIM,
            metric: DistanceMetric::Cosine,
            model_family: "vision".to_string(),
            model_name: "joint-image-text".to_string(),
            model_version: "1".to_string(),
            supported_evidence_kinds: vec![context_graph::EvidenceKind::ImageFrame],
            default_for_evidence_kind: None,
            bindings: Vec::new(),
        }],
        ..Default::default()
    })
    .expect("store opens with a registered vision space and no text embedder")
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-vision".to_string(),
        model_name: "joint-image-text".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

/// A unit vector along one axis; cosine between distinct axes is 0.
fn axis_vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[axis] = 1.0;
    v
}

/// A vector close to `axis` with a small off-axis component; cosine to the
/// axis vector is ~0.995, cosine to any other axis is ~0.1.
fn near_axis_vector(axis: usize, other: usize) -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[axis] = 1.0;
    v[other] = 0.1;
    let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

struct Site {
    store: Store,
    context_id: context_graph::ContextId,
}

fn site(name: &str) -> Site {
    let store = store(name);
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-site"),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("site context");
    Site {
        store,
        context_id: context.id,
    }
}

fn enrolled_entity(
    site: &Site,
    name: &str,
    entity_type: EntityType,
    label: &str,
    vector: Vec<f32>,
) -> EntityId {
    let entity = site
        .store
        .create_entity(CreateEntity {
            entity_type,
            name: name.to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: site.context_id,
        })
        .expect("entity");
    site.store
        .enroll_entity_reference(
            entity.id,
            VISION_SPACE,
            Some(label.to_string()),
            vector,
            producer(),
        )
        .expect("enroll reference");
    entity.id
}

fn match_options(site: &Site) -> EntityReferenceMatchOptions {
    EntityReferenceMatchOptions {
        context_id: Some(site.context_id),
        ..Default::default()
    }
}

#[test]
fn match_entity_reference_ranks_enrolled_entities_with_cosine_scores() {
    let site = site("erm-ranks");
    let first = enrolled_entity(&site, "first person", EntityType::Person, "front", {
        axis_vector(0)
    });
    let second = enrolled_entity(
        &site,
        "second person",
        EntityType::Person,
        "front",
        axis_vector(1),
    );

    let matches = site
        .store
        .match_entity_reference(VISION_SPACE, &near_axis_vector(0, 1), match_options(&site))
        .expect("match runs");

    assert!(
        !matches.is_empty(),
        "a probe near an enrolled reference must return matches"
    );
    assert_eq!(
        matches[0].entity_id, first,
        "the nearest enrolled entity ranks first"
    );
    assert!(
        matches[0].score > 0.9,
        "near-probe cosine must be high, got {}",
        matches[0].score
    );
    if let Some(second_match) = matches.iter().find(|m| m.entity_id == second) {
        assert!(
            second_match.score < 0.5,
            "the wrong entity must score clearly lower, got {}",
            second_match.score
        );
        assert!(
            matches[0].score > second_match.score,
            "ranking must follow score"
        );
    }
    assert_eq!(
        matches[0].label.as_deref(),
        Some("front"),
        "the matched reference label rides the result"
    );
}

#[test]
fn match_entity_reference_is_open_set_returning_empty_below_min_score() {
    let site = site("erm-open-set");
    enrolled_entity(
        &site,
        "enrolled person",
        EntityType::Person,
        "front",
        axis_vector(0),
    );

    let matches = site
        .store
        .match_entity_reference(
            VISION_SPACE,
            &axis_vector(7),
            EntityReferenceMatchOptions {
                context_id: Some(site.context_id),
                min_score: Some(0.5),
                ..Default::default()
            },
        )
        .expect("match runs");

    assert!(
        matches.is_empty(),
        "an orthogonal probe with a threshold must match nothing (unknown stays unknown), got {matches:?}"
    );
}

#[test]
fn match_entity_reference_scopes_to_context() {
    let site_a = site("erm-scope-a");
    let entity_a = enrolled_entity(
        &site_a,
        "site-a person",
        EntityType::Person,
        "front",
        axis_vector(0),
    );
    // A second context in the SAME store, enrolled with the same direction.
    let context_b = site_a
        .store
        .create_context(CreateContext {
            name: "erm-scope-b-site".to_string(),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("second context");
    let entity_b = site_a
        .store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: "site-b person".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context_b.id,
        })
        .expect("entity b");
    site_a
        .store
        .enroll_entity_reference(
            entity_b.id,
            VISION_SPACE,
            Some("front".to_string()),
            axis_vector(0),
            producer(),
        )
        .expect("enroll b");

    let matches = site_a
        .store
        .match_entity_reference(VISION_SPACE, &axis_vector(0), match_options(&site_a))
        .expect("match runs");

    assert!(
        matches.iter().any(|m| m.entity_id == entity_a),
        "the in-context entity matches"
    );
    assert!(
        matches.iter().all(|m| m.entity_id != entity_b.id),
        "an entity enrolled in another context must never surface"
    );
}

#[test]
fn match_entity_reference_dedupes_multiple_references_to_best_per_entity() {
    let site = site("erm-dedupe");
    let entity = enrolled_entity(
        &site,
        "multi-reference person",
        EntityType::Person,
        "front",
        axis_vector(0),
    );
    site.store
        .enroll_entity_reference(
            entity,
            VISION_SPACE,
            Some("side".to_string()),
            near_axis_vector(0, 2),
            producer(),
        )
        .expect("second reference");

    let matches = site
        .store
        .match_entity_reference(VISION_SPACE, &axis_vector(0), match_options(&site))
        .expect("match runs");

    let hits: Vec<&EntityMatch> = matches.iter().filter(|m| m.entity_id == entity).collect();
    assert_eq!(
        hits.len(),
        1,
        "one entity with several references returns ONE match (best reference), got {matches:?}"
    );
    assert_eq!(
        hits[0].label.as_deref(),
        Some("front"),
        "the best-scoring reference's label rides the result"
    );
}

#[test]
fn match_entity_reference_respects_top_k() {
    let site = site("erm-top-k");
    for axis in 0..3 {
        enrolled_entity(
            &site,
            &format!("person {axis}"),
            EntityType::Person,
            "front",
            near_axis_vector(0, axis + 1),
        );
    }

    let matches = site
        .store
        .match_entity_reference(
            VISION_SPACE,
            &axis_vector(0),
            EntityReferenceMatchOptions {
                context_id: Some(site.context_id),
                top_k: Some(2),
                ..Default::default()
            },
        )
        .expect("match runs");

    assert_eq!(matches.len(), 2, "top_k bounds the result count");
    assert!(
        matches[0].score >= matches[1].score,
        "results stay ranked under top_k"
    );
}

#[test]
fn match_entity_reference_validates_space_and_dimension() {
    let site = site("erm-validate");
    enrolled_entity(
        &site,
        "enrolled person",
        EntityType::Person,
        "front",
        axis_vector(0),
    );

    let unknown_space =
        site.store
            .match_entity_reference("no_such_space", &axis_vector(0), match_options(&site));
    assert!(
        matches!(unknown_space, Err(CgError::UnknownEmbeddingSpace { .. })),
        "an unregistered space is refused with the typed error, got {unknown_space:?}"
    );

    let wrong_dim =
        site.store
            .match_entity_reference(VISION_SPACE, &[1.0, 0.0, 0.0], match_options(&site));
    assert!(
        matches!(wrong_dim, Err(CgError::EmbeddingDimensionMismatch { .. })),
        "a wrong-dimension probe is refused with the typed error, got {wrong_dim:?}"
    );
}

#[test]
fn entity_type_vocabulary_covers_animal_and_vehicle() {
    let site = site("erm-vocab");
    for (entity_type, name) in [
        (EntityType::Animal, "the dog"),
        (EntityType::Vehicle, "the tractor"),
    ] {
        let created = site
            .store
            .create_entity(CreateEntity {
                entity_type,
                name: name.to_string(),
                properties: BTreeMap::new(),
                tags: Vec::new(),
                context_id: site.context_id,
            })
            .expect("entity with the new type");
        let read_back = site
            .store
            .get_entity(created.id)
            .expect("get")
            .expect("exists");
        assert_eq!(
            read_back.entity_type, entity_type,
            "the entity type round-trips through the store"
        );
    }
    assert!(
        EntityType::ALL.contains(&EntityType::Animal)
            && EntityType::ALL.contains(&EntityType::Vehicle),
        "the vocabulary constant carries the new types"
    );
}

#[test]
fn embedder_input_supports_in_memory_image_bytes() {
    // An external consumer's embedder must be able to receive an in-memory image
    // through the one shared trait — no temp-file URI detour on a hot path.
    struct ByteFaithfulEmbedder;
    impl context_graph::Embedder for ByteFaithfulEmbedder {
        fn embed(
            &self,
            input: EmbeddingInput,
        ) -> context_graph::Result<context_graph::EmbeddingOutput> {
            let bytes = match input {
                EmbeddingInput::ImageBytes(bytes) => bytes,
                other => panic!("this embedder only handles in-memory images, got {other:?}"),
            };
            // Content-faithful: the vector derives from the actual bytes.
            let mut vector = vec![0.0f32; DIM];
            for (i, b) in bytes.iter().enumerate() {
                vector[i % DIM] += f32::from(*b) / 255.0;
            }
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            vector.iter_mut().for_each(|x| *x /= norm);
            Ok(context_graph::EmbeddingOutput {
                embedding_space_id: VISION_SPACE.to_string(),
                dimension: DIM,
                producer: producer(),
                vector,
            })
        }
        fn supported_inputs(&self) -> Vec<context_graph::EvidenceKind> {
            vec![context_graph::EvidenceKind::ImageFrame]
        }
        fn embedding_space(&self) -> context_graph::EmbeddingSpace {
            context_graph::EmbeddingSpace {
                embedding_space_id: VISION_SPACE.to_string(),
                dimension: DIM,
                metric: DistanceMetric::Cosine,
                model_family: "vision".to_string(),
                model_name: "byte-faithful".to_string(),
                model_version: "1".to_string(),
                supported_evidence_kinds: vec![context_graph::EvidenceKind::ImageFrame],
                default_for_evidence_kind: None,
                bindings: Vec::new(),
            }
        }
    }

    let embedder = ByteFaithfulEmbedder;
    let a = context_graph::Embedder::embed(&embedder, EmbeddingInput::ImageBytes(vec![7u8; 4096]))
        .expect("embeds in-memory bytes");
    let b =
        context_graph::Embedder::embed(&embedder, EmbeddingInput::ImageBytes(vec![250u8; 4096]))
            .expect("embeds different bytes");
    assert_eq!(a.vector.len(), DIM);
    assert_ne!(
        a.vector, b.vector,
        "different image bytes produce different vectors (content-faithful, not rigged)"
    );
}

#[test]
fn public_surface_adds_no_free_text_search_reexport() {
    // The consumer door stays vector-only: no free-text search reachable in a
    // normal build. The gated test_support/compile_support doors are allowed.
    // Un-gated crate-root re-exports in lib.rs are top-level `pub use` lines
    // (column 0); the sanctioned gated doors live INSIDE `mod test_support` /
    // `mod compile_support` blocks and are therefore indented.
    let lib_rs = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"),
    )
    .expect("read lib.rs");
    for line in lib_rs.lines() {
        if !line.starts_with("pub use") {
            continue;
        }
        for forbidden in ["semantic_search", "evidence_search", "vector_search"] {
            assert!(
                !line.contains(forbidden),
                "free-text/raw vector search must not be re-exported un-gated: {line}"
            );
        }
    }
}

#[test]
fn removing_an_entitys_references_removes_it_from_matching() {
    // The deletion promise: forgetting an enrolled entity removes its reference
    // vectors, so later probes of the same subject honestly match nothing.
    let site = site("erm-remove");
    let entity = enrolled_entity(
        &site,
        "forgotten person",
        EntityType::Person,
        "front",
        axis_vector(0),
    );
    site.store
        .enroll_entity_reference(
            entity,
            VISION_SPACE,
            Some("side".to_string()),
            near_axis_vector(0, 3),
            producer(),
        )
        .expect("second reference");

    let before = site
        .store
        .match_entity_reference(VISION_SPACE, &axis_vector(0), match_options(&site))
        .expect("match before removal");
    assert!(
        before.iter().any(|m| m.entity_id == entity),
        "the enrolled entity matches before removal"
    );

    let removed = site
        .store
        .remove_entity_references(entity)
        .expect("removal runs");
    assert_eq!(removed, 2, "both enrolled references are removed");

    let after = site
        .store
        .match_entity_reference(VISION_SPACE, &axis_vector(0), match_options(&site))
        .expect("match after removal");
    assert!(
        after.iter().all(|m| m.entity_id != entity),
        "a removed entity never matches again, got {after:?}"
    );

    let repeat = site
        .store
        .remove_entity_references(entity)
        .expect("second removal runs");
    assert_eq!(repeat, 0, "removal is idempotent");
}

#[test]
fn match_entity_reference_survives_a_large_top_k() {
    // A top_k larger than the internal over-fetch cap must return results,
    // never panic (Ord::clamp aborts when min > max).
    let site = site("erm-large-top-k");
    let entity = enrolled_entity(
        &site,
        "enrolled person",
        EntityType::Person,
        "front",
        axis_vector(0),
    );
    let matches = site
        .store
        .match_entity_reference(
            VISION_SPACE,
            &axis_vector(0),
            EntityReferenceMatchOptions {
                context_id: Some(site.context_id),
                top_k: Some(1000),
                ..Default::default()
            },
        )
        .expect("a large top_k is valid input");
    assert_eq!(matches.first().map(|m| m.entity_id), Some(entity));
}
