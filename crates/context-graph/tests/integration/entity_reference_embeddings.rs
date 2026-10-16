use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use context_graph::{
    CG_EDGE_SYSTEM_TAG, ContextId, CreateContext, CreateEntity, DistanceMetric, EmbeddingId,
    EmbeddingSpaceDecl, EntityId, EntityType, EvidenceKind, EvidenceProducer, Store, StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;

const FACE_REFERENCE_SPACE: &str = "face_reference_v1";

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    store: Store,
    context_id: ContextId,
    entity_id: EntityId,
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
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: "Alex".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    Fixture {
        _dir: dir,
        db_path,
        store,
        context_id: context.id,
        entity_id: entity.id,
    }
}

fn producer(model_version: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: CG_EDGE_SYSTEM_TAG.to_string(),
        model_name: "face-reference".to_string(),
        model_version: model_version.to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn raw_db(path: &Path, store: Store) -> Database {
    drop(store);
    Database::open(path).expect("raw database open")
}

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn count(db: &Database, sql: &str, params: &HashMap<String, Value>) -> i64 {
    let result = db.execute(sql, params).expect("count query");
    assert_eq!(result.rows.len(), 1);
    assert!(
        matches!(result.rows[0].first(), Some(Value::Int64(_))),
        "expected count integer, got {:?}",
        result.rows[0].first()
    );
    match result.rows[0][0] {
        Value::Int64(value) => value,
        _ => 0,
    }
}

fn has_embedding_shadow_key(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => map.iter().any(|(key, value)| {
            let key = key.to_ascii_lowercase();
            key.contains("vector") || key.contains("embedding") || has_embedding_shadow_key(value)
        }),
        serde_json::Value::Array(values) => values.iter().any(has_embedding_shadow_key),
        _ => false,
    }
}

#[test]
fn entity_reference_embedding_single_shot_label_none() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = fixture("single-shot");
    let id = fixture.store.enroll_entity_reference(
        fixture.entity_id,
        FACE_REFERENCE_SPACE,
        None,
        vec![1.0; 512],
        producer("1"),
    );
    let db = raw_db(&fixture.db_path, fixture.store);
    let rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_reference_embeddings_512 WHERE entity_id = $entity_id AND context_id = $context_id AND embedding_space_id = $space AND label IS NULL",
        &params(vec![
            ("entity_id", Value::Uuid(fixture.entity_id.as_uuid())),
            ("context_id", Value::Uuid(fixture.context_id.as_uuid())),
            ("space", Value::Text(FACE_REFERENCE_SPACE.to_string())),
        ]),
    );

    assert!(
        id.as_ref()
            .is_ok_and(|id| id.is_valid() && *id != EmbeddingId::default())
            && rows == 1,
        "expected one FK-backed unlabeled entity reference embedding row, id={id:?}, rows={rows}"
    );
}

#[test]
fn entity_reference_embedding_multi_shot_distinct_labels() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = fixture("multi-shot");
    let first = fixture.store.enroll_entity_reference(
        fixture.entity_id,
        FACE_REFERENCE_SPACE,
        Some("front".to_string()),
        vec![1.0; 512],
        producer("1"),
    );
    let second = fixture.store.enroll_entity_reference(
        fixture.entity_id,
        FACE_REFERENCE_SPACE,
        Some("side".to_string()),
        vec![0.5; 512],
        producer("1"),
    );
    let db = raw_db(&fixture.db_path, fixture.store);
    let rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_reference_embeddings_512 WHERE entity_id = $entity_id",
        &params(vec![(
            "entity_id",
            Value::Uuid(fixture.entity_id.as_uuid()),
        )]),
    );

    assert!(
        first.is_ok() && second.is_ok() && first.as_ref().ok() != second.as_ref().ok() && rows == 2,
        "expected two distinct labeled reference rows, first={first:?}, second={second:?}, rows={rows}"
    );
}

#[test]
fn entity_reference_embedding_re_enrollment_under_same_label_idempotent_overwrite() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = fixture("reenroll");
    let first = fixture.store.enroll_entity_reference(
        fixture.entity_id,
        FACE_REFERENCE_SPACE,
        Some("front".to_string()),
        vec![1.0; 512],
        producer("1"),
    );
    let second = fixture.store.enroll_entity_reference(
        fixture.entity_id,
        FACE_REFERENCE_SPACE,
        Some("front".to_string()),
        vec![0.5; 512],
        producer("2"),
    );
    let rows = fixture
        .store
        .list_entity_reference_embeddings_for_entity(fixture.entity_id);
    let db = raw_db(&fixture.db_path, fixture.store);
    let raw_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_reference_embeddings_512 WHERE entity_id = $entity_id AND label = $label",
        &params(vec![
            ("entity_id", Value::Uuid(fixture.entity_id.as_uuid())),
            ("label", Value::Text("front".to_string())),
        ]),
    );

    assert!(
        first.is_ok()
            && second.is_ok()
            && first == second
            && rows.as_ref().is_ok_and(|rows| rows.len() == 1
                && rows[0].embedding_space_id == FACE_REFERENCE_SPACE
                && rows[0].produced_by.model_version == "2"
                && rows[0].vector == vec![0.5; 512])
            && raw_rows == 1,
        "expected same-label re-enrollment to overwrite one stored row idempotently, first={first:?}, second={second:?}, rows={rows:?}, raw_rows={raw_rows}"
    );
}

#[test]
fn entity_reference_embedding_query_path_uniform_across_shot_count() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = fixture("uniform-query");
    let single = fixture.store.enroll_entity_reference(
        fixture.entity_id,
        FACE_REFERENCE_SPACE,
        None,
        vec![1.0; 512],
        producer("single"),
    );
    let second_entity = fixture
        .store
        .create_entity(CreateEntity {
            entity_type: EntityType::Person,
            name: "Alex alternate".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: fixture.context_id,
        })
        .expect("second entity");
    let multi_a = fixture.store.enroll_entity_reference(
        second_entity.id,
        FACE_REFERENCE_SPACE,
        Some("front".to_string()),
        vec![1.0; 512],
        producer("multi"),
    );
    let multi_b = fixture.store.enroll_entity_reference(
        second_entity.id,
        FACE_REFERENCE_SPACE,
        Some("side".to_string()),
        vec![0.5; 512],
        producer("multi"),
    );
    let single_rows = fixture
        .store
        .list_entity_reference_embeddings_for_entity(fixture.entity_id);
    let multi_rows = fixture
        .store
        .list_entity_reference_embeddings_for_entity(second_entity.id);

    assert!(
        single.is_ok()
            && multi_a.is_ok()
            && multi_b.is_ok()
            && single_rows.as_ref().is_ok_and(|rows| rows.len() == 1)
            && multi_rows.as_ref().is_ok_and(|rows| rows.len() == 2),
        "expected one list path for single-shot and multi-shot rows, single={single:?}, multi_a={multi_a:?}, multi_b={multi_b:?}, single_rows={single_rows:?}, multi_rows={multi_rows:?}"
    );
}

#[test]
fn entity_reference_embedding_writes_to_dedicated_child_table_not_entities_table() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture = fixture("child-table");
    let front_vector = vec![1.0; 512];
    let id = fixture.store.enroll_entity_reference(
        fixture.entity_id,
        FACE_REFERENCE_SPACE,
        Some("front".to_string()),
        front_vector.clone(),
        producer("1"),
    );
    let db = raw_db(&fixture.db_path, fixture.store);
    let reference_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_reference_embeddings_512 WHERE entity_id = $entity_id",
        &params(vec![(
            "entity_id",
            Value::Uuid(fixture.entity_id.as_uuid()),
        )]),
    );
    let vector_result = db
        .execute(
            "SELECT vector FROM entity_reference_embeddings_512 WHERE entity_id = $entity_id AND label = $label",
            &params(vec![
                ("entity_id", Value::Uuid(fixture.entity_id.as_uuid())),
                ("label", Value::Text("front".to_string())),
            ]),
        )
        .expect("reference vector projection");
    let persisted_vector_matches = vector_result.rows.first().is_some_and(
        |row| matches!(row.first(), Some(Value::Vector(vector)) if vector == &front_vector),
    );
    let entity_rows_with_vector_shadow = db
        .table_meta("entities")
        .expect("entities meta")
        .columns
        .iter()
        .filter(|column| matches!(column.column_type, contextdb_core::ColumnType::Vector(_)))
        .count();
    let snapshot_vector_columns = db
        .table_meta("entity_snapshots")
        .expect("entity_snapshots meta")
        .columns
        .iter()
        .filter(|column| matches!(column.column_type, contextdb_core::ColumnType::Vector(_)))
        .count();
    let snapshot_data = db
        .execute(
            "SELECT data FROM entity_snapshots WHERE entity_id = $entity_id",
            &params(vec![(
                "entity_id",
                Value::Uuid(fixture.entity_id.as_uuid()),
            )]),
        )
        .expect("snapshot payloads");
    let snapshot_has_embedding_shadow = snapshot_data.rows.iter().any(
        |row| matches!(row.first(), Some(Value::Json(value)) if has_embedding_shadow_key(value)),
    );

    assert!(
        id.is_ok()
            && reference_rows == 1
            && persisted_vector_matches
            && entity_rows_with_vector_shadow == 0
            && snapshot_vector_columns == 0
            && !snapshot_has_embedding_shadow,
        "expected dedicated reference table row with the exact vector and no vector/embedding shadow on entities or snapshots, id={id:?}, reference_rows={reference_rows}, persisted_vector_matches={persisted_vector_matches}, entity_vector_columns={entity_rows_with_vector_shadow}, snapshot_vector_columns={snapshot_vector_columns}, snapshot_has_embedding_shadow={snapshot_has_embedding_shadow}, vector_rows={:?}",
        vector_result.rows
    );
}

/// @invariant 15a
#[cfg(context_graph_internal_test_support)]
#[test]
fn entity_reference_embedding_cross_space_query_refused_with_typed_error() {
    let _alongside = crate::runs_alone::alongside_siblings();
    use context_graph::{CgError, test_support::embedding::entity_reference_cosine_match};

    let fixture = fixture("cross-space-reference");
    let enrolled = fixture.store.enroll_entity_reference(
        fixture.entity_id,
        FACE_REFERENCE_SPACE,
        Some("front".to_string()),
        vec![1.0; 512],
        producer("1"),
    );
    let result = fixture
        .store
        .list_entity_reference_embeddings_for_entity(fixture.entity_id);
    let comparison = enrolled
        .as_ref()
        .map(|id| entity_reference_cosine_match(&fixture.store, "face_live_v2", *id));

    assert!(
        enrolled.is_ok()
            && result.as_ref().is_ok_and(|rows| rows
                .iter()
                .any(|row| row.entity_id == fixture.entity_id
                    && row.embedding_space_id == FACE_REFERENCE_SPACE))
            && matches!(
                comparison,
                Ok(Err(CgError::CrossSpaceCosineRefused {
                    ref live_space,
                    ref reference_space
                })) if live_space == "face_live_v2" && reference_space == FACE_REFERENCE_SPACE
            ),
        "expected listing to return reference rows and comparison to refuse cross-space match, enrolled={enrolled:?}, result={result:?}, comparison={comparison:?}"
    );
}
