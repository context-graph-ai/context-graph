use std::collections::BTreeMap;

use context_graph::{
    CgError, CreateContext, CreateEntity, DistanceMetric, EmbeddingId, EmbeddingSpaceDecl,
    EntityId, EntityType, EvidenceKind, EvidenceProducer, Store, StoreConfig,
};

const FACE_REFERENCE_SPACE: &str = "face_reference_v1";

fn store() -> Store {
    fixture("entity-reference-edge").store
}

struct Fixture {
    _dir: tempfile::TempDir,
    store: Store,
}

fn fixture(name: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        extra_embedding_spaces: vec![EmbeddingSpaceDecl {
            embedding_space_id: FACE_REFERENCE_SPACE.to_string(),
            dimension: 512,
            metric: DistanceMetric::Cosine,
            model_family: "vision".to_string(),
            model_name: "face-reference".to_string(),
            model_version: "v1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
            default_for_evidence_kind: None,
            bindings: Vec::new(),
        }],
        ..Default::default()
    })
    .expect("open");
    Fixture { _dir: dir, store }
}

fn seed_entity(store: &Store, name: &str) -> context_graph::Entity {
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: name.to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity")
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: context_graph::CG_EDGE_SYSTEM_TAG.to_string(),
        model_name: "reference".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

#[test]
fn entity_reference_embedding_query_for_unknown_entity_returns_empty_vec_not_error() {
    let fixture = fixture("entity-reference-known-unknown");
    let known = seed_entity(&fixture.store, "Kumar");
    let enrolled = fixture.store.enroll_entity_reference(
        known.id,
        FACE_REFERENCE_SPACE,
        Some("front".to_string()),
        vec![1.0; 512],
        producer(),
    );
    let known_rows = fixture
        .store
        .list_entity_reference_embeddings_for_entity(known.id);
    let rows = fixture
        .store
        .list_entity_reference_embeddings_for_entity(EntityId::new_v7());
    assert!(
        enrolled
            .as_ref()
            .is_ok_and(|id| *id != EmbeddingId::default())
            && known_rows.as_ref().is_ok_and(|items| items.len() == 1
                && items[0].entity_id == known.id
                && Some(&items[0].id) == enrolled.as_ref().ok())
            && rows.as_ref().is_ok_and(Vec::is_empty),
        "known entity must return its reference row and unknown entity reference query should be an empty result, enrolled={enrolled:?}, known_rows={known_rows:?}, unknown_rows={rows:?}"
    );
}

#[test]
fn entity_reference_enrollment_dimension_validated_against_space_at_write_time() {
    let store = store();
    let entity = seed_entity(&store, "Kumar");
    let result = store.enroll_entity_reference(
        entity.id,
        FACE_REFERENCE_SPACE,
        None,
        vec![1.0; 511],
        producer(),
    );
    assert!(
        matches!(
            result,
            Err(CgError::EmbeddingDimensionMismatch {
                expected: 512,
                found: 511,
                ..
            })
        ),
        "expected entity reference dimension mismatch, got {result:?}"
    );
}

#[test]
fn entity_reference_enrollment_for_unknown_entity_id_rejected_with_typed_fk_error() {
    let result = store().enroll_entity_reference(
        EntityId::new_v7(),
        FACE_REFERENCE_SPACE,
        None,
        vec![1.0; 512],
        producer(),
    );
    assert!(
        matches!(
            result,
            Err(CgError::NotFound {
                missing: Some("entity"),
                ..
            })
        ),
        "expected typed missing-entity rejection, got {result:?}"
    );
}

#[test]
fn entity_reference_embedding_parent_entity_archive_does_not_orphan_child_rows() {
    let fixture = fixture("entity-reference-archive");
    let entity = seed_entity(&fixture.store, "Kumar");
    let first_id = fixture.store.enroll_entity_reference(
        entity.id,
        FACE_REFERENCE_SPACE,
        Some("front".to_string()),
        vec![1.0; 512],
        producer(),
    );
    let second_id = fixture.store.enroll_entity_reference(
        entity.id,
        FACE_REFERENCE_SPACE,
        Some("side".to_string()),
        vec![0.5; 512],
        producer(),
    );
    let before_rows = fixture
        .store
        .list_entity_reference_embeddings_for_entity(entity.id);
    let before_created_at = before_rows
        .as_ref()
        .map(|items| {
            items
                .iter()
                .map(|row| (row.id, row.created_at))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let archived = fixture.store.archive_entity(entity.id);
    let rows = fixture
        .store
        .list_entity_reference_embeddings_for_entity(entity.id);
    let after_created_at = rows
        .as_ref()
        .map(|items| {
            items
                .iter()
                .map(|row| (row.id, row.created_at))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let new_enrollment = fixture.store.enroll_entity_reference(
        entity.id,
        FACE_REFERENCE_SPACE,
        Some("post-archive".to_string()),
        vec![0.25; 512],
        producer(),
    );

    assert!(
        first_id
            .as_ref()
            .is_ok_and(|value| value.is_valid() && *value != EmbeddingId::default())
            && second_id
                .as_ref()
                .is_ok_and(|value| value.is_valid() && *value != EmbeddingId::default())
            && before_rows.as_ref().is_ok_and(|items| items.len() == 2)
            && archived.is_ok()
            && rows.as_ref().is_ok_and(|items| items.len() == 2
                && items
                    .iter()
                    .any(|row| Some(&row.id) == first_id.as_ref().ok())
                && items
                    .iter()
                    .any(|row| Some(&row.id) == second_id.as_ref().ok()))
            && before_created_at.len() == 2
            && before_created_at == after_created_at
            && matches!(
                new_enrollment,
                Err(CgError::EntityArchived {
                    entity_id: archived_entity
                }) if archived_entity == entity.id
            ),
        "expected archive to preserve existing child reference rows and txids while rejecting new enrollment, first_id={first_id:?}, second_id={second_id:?}, before_rows={before_rows:?}, archived={archived:?}, rows={rows:?}, before_created_at={before_created_at:?}, after_created_at={after_created_at:?}, new_enrollment={new_enrollment:?}"
    );
}
