use context_graph::{
    CgError, CreateContext, CreateEntity, DistanceMetric, EmbeddingSpaceDecl, EntityType,
    EvidenceKind, EvidenceProducer, Store, StoreConfig,
};
use contextdb_core::{ColumnType, Value};
use contextdb_engine::Database;
use std::collections::{BTreeMap, HashMap};

fn open_with(space: EmbeddingSpaceDecl) -> context_graph::Result<Store> {
    Store::open(StoreConfig {
        db_path: std::env::temp_dir().join(format!("registry-edge-{}.db", uuid::Uuid::now_v7())),
        extra_embedding_spaces: vec![space],
        ..Default::default()
    })
}

fn space(
    id: &str,
    supported: Vec<EvidenceKind>,
    default: Option<EvidenceKind>,
) -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: id.to_string(),
        dimension: 384,
        metric: DistanceMetric::Cosine,
        model_family: "custom".to_string(),
        model_name: "custom".to_string(),
        model_version: "1".to_string(),
        supported_evidence_kinds: supported,
        default_for_evidence_kind: default,
        bindings: Vec::new(),
    }
}

fn space_with_dim(
    id: &str,
    dimension: usize,
    supported: Vec<EvidenceKind>,
    default: Option<EvidenceKind>,
) -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        dimension,
        ..space(id, supported, default)
    }
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "cg-edge".to_string(),
        model_name: "face-reference".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "registry".to_string(),
    }
}

fn count(db: &Database, sql: &str, params: HashMap<String, Value>) -> i64 {
    let result = db.execute(sql, &params).expect("count query");
    assert_eq!(result.rows.len(), 1);
    let scalar = result.rows[0].first();
    assert!(
        matches!(scalar, Some(Value::Int64(_))),
        "expected integer count, got {scalar:?}"
    );
    if let Some(Value::Int64(value)) = scalar {
        *value
    } else {
        0
    }
}

#[test]
fn embedding_space_declaration_with_empty_string_id_refused_at_store_open() {
    let result = open_with(space("", vec![EvidenceKind::Text], None));
    assert!(matches!(
        result,
        Err(CgError::InvalidEmbeddingSpaceId { reason: "empty_id" })
    ));
}

#[test]
fn embedding_space_id_with_whitespace_or_control_chars_refused_at_store_open() {
    let result = open_with(space("bad id\n", vec![EvidenceKind::Text], None));
    assert!(matches!(
        result,
        Err(CgError::InvalidEmbeddingSpaceId {
            reason: "whitespace_or_control_chars"
        })
    ));
}

#[test]
fn embedding_space_id_exceeding_max_length_refused_at_store_open() {
    let result = open_with(space(&"x".repeat(300), vec![EvidenceKind::Text], None));
    assert!(matches!(
        result,
        Err(CgError::InvalidEmbeddingSpaceId { reason: "too_long" })
    ));
}

#[test]
fn embedding_space_default_for_evidence_kind_not_in_supported_evidence_kinds_refused_at_store_open()
{
    let result = open_with(space(
        "image_default",
        vec![EvidenceKind::ImageFrame],
        Some(EvidenceKind::Text),
    ));
    assert!(matches!(
        result,
        Err(CgError::InconsistentEmbeddingSpaceDeclaration { ref space_id })
            if space_id == "image_default"
    ));
}

#[test]
fn embedder_with_empty_supported_inputs_refused_at_store_open() {
    let result = open_with(space("empty_supported_inputs", Vec::new(), None));
    assert!(matches!(
        result,
        Err(CgError::EmbedderDeclaresNoSupportedInputs {
            ref embedding_space_id
        }) if embedding_space_id == "empty_supported_inputs"
    ));
}

#[test]
fn entity_reference_embeddings_per_dim_table_created_at_store_open_when_new_dimension_declared() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("registry-dim-384.db");
    let opened = Store::open(StoreConfig {
        db_path: db_path.clone(),
        extra_embedding_spaces: vec![space("dim_384", vec![EvidenceKind::Text], None)],
        ..Default::default()
    });
    let opened_ok = opened.is_ok();
    drop(opened);
    let db = Database::open(&db_path).expect("raw database");
    let table = db.table_meta("entity_reference_embeddings_384");
    let old_handle = db.table_meta("embedding_handles");

    assert!(
        opened_ok
            && table.as_ref().is_some_and(|meta| {
                meta.columns.iter().any(|column| {
                    column.name == "vector" && column.column_type == ColumnType::Vector(384)
                }) && meta
                    .columns
                    .iter()
                    .any(|column| column.name == "embedding_space_id")
                    && meta
                        .columns
                        .iter()
                        .any(|column| column.name == "produced_by")
            })
            && old_handle.is_none(),
        "new dimension must create entity_reference_embeddings_384 and remove old polymorphic embedding_handles, opened_ok={opened_ok}, table={table:?}, old_handle={old_handle:?}"
    );
}

#[test]
fn entity_reference_embeddings_per_dim_table_reused_when_existing_dimension_redeclared() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("registry-dim-512.db");
    let first = Store::open(StoreConfig {
        db_path: db_path.clone(),
        extra_embedding_spaces: vec![
            space_with_dim("dim_512_a", 512, vec![EvidenceKind::ImageFrame], None),
            space_with_dim("dim_512_b", 512, vec![EvidenceKind::ImageFrame], None),
        ],
        ..Default::default()
    });
    drop(first);
    let second = Store::open(StoreConfig {
        db_path: db_path.clone(),
        extra_embedding_spaces: vec![
            space_with_dim("dim_512_a", 512, vec![EvidenceKind::ImageFrame], None),
            space_with_dim("dim_512_b", 512, vec![EvidenceKind::ImageFrame], None),
        ],
        ..Default::default()
    });
    let second_ok = second.is_ok();
    if let Ok(store) = &second {
        let context = store
            .create_context(CreateContext {
                name: format!("registry-dim-512-{}", uuid::Uuid::now_v7()),
                labels: Vec::new(),
                properties: BTreeMap::new(),
            })
            .expect("context");
        let entity = store
            .create_entity(CreateEntity {
                entity_type: EntityType::Person,
                name: "same-dim-reference".to_string(),
                properties: BTreeMap::new(),
                tags: Vec::new(),
                context_id: context.id,
            })
            .expect("entity");
        let first_ref = store.enroll_entity_reference(
            entity.id,
            "dim_512_a",
            Some("front".to_string()),
            vec![1.0; 512],
            producer(),
        );
        let second_ref = store.enroll_entity_reference(
            entity.id,
            "dim_512_b",
            Some("side".to_string()),
            vec![0.5; 512],
            producer(),
        );
        assert!(
            first_ref.is_ok() && second_ref.is_ok(),
            "same-dimension reference setup must enroll under both spaces, first_ref={first_ref:?}, second_ref={second_ref:?}"
        );
    }
    drop(second);
    let db = Database::open(&db_path).expect("raw database");
    let handle_tables = [
        "evidence_embedding_handles",
        "entity_reference_embedding_handles",
        "digest_embedding_handles",
    ]
    .into_iter()
    .filter_map(|table| db.table_meta(table).map(|meta| (table, meta)))
    .collect::<Vec<_>>();

    assert!(
        second_ok
            && db.table_meta("entity_reference_embeddings_512").is_some()
            && count(
                &db,
                "SELECT COUNT(*) AS n FROM entity_reference_embeddings_512 WHERE embedding_space_id = $space",
                HashMap::from([("space".to_string(), Value::Text("dim_512_a".to_string()))]),
            ) == 1
            && count(
                &db,
                "SELECT COUNT(*) AS n FROM entity_reference_embeddings_512 WHERE embedding_space_id = $space",
                HashMap::from([("space".to_string(), Value::Text("dim_512_b".to_string()))]),
            ) == 1
            && db.table_meta("entity_reference_embeddings_384").is_none()
            && handle_tables.len() == 3
            && handle_tables.iter().all(|(_, meta)| meta
                .columns
                .iter()
                .all(|column| column.name != "target_kind")),
        "existing dimension must be reused by distinct same-dimension spaces with rows distinguished by embedding_space_id, second_ok={second_ok}, handle_tables={handle_tables:?}"
    );
}
