use std::path::Path;
use std::sync::Arc;

use context_graph::embedder::mock::MockEmbedder;
use context_graph::{
    CgError, DistanceMetric, EmbedderRegistration, EmbeddingSpaceBinding, EmbeddingSpaceDecl,
    EvidenceKind, Store, StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::Database;

fn text_space(id: &str, version: &str, default: Option<EvidenceKind>) -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: id.to_string(),
        dimension: 768,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "consumer-text".to_string(),
        model_version: version.to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: default,
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: id.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_text".to_string(),
        }],
    }
}

fn registry_space(id: &str, version: &str) -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: id.to_string(),
        dimension: 384,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "consumer-registry".to_string(),
        model_version: version.to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
        default_for_evidence_kind: None,
        bindings: Vec::new(),
    }
}

fn open_with(extra_embedding_spaces: Vec<EmbeddingSpaceDecl>) -> Result<Store, CgError> {
    open_path(
        &std::env::temp_dir().join(format!(
            "embedding-space-validation-{}.db",
            uuid::Uuid::now_v7()
        )),
        extra_embedding_spaces,
    )
}

fn open_path(
    db_path: &Path,
    extra_embedding_spaces: Vec<EmbeddingSpaceDecl>,
) -> Result<Store, CgError> {
    Store::open(StoreConfig {
        db_path: db_path.to_path_buf(),
        extra_embedding_spaces,
        ..Default::default()
    })
}

fn embedding_space_row_count(db_path: &Path, space_id: &str) -> i64 {
    let db = Database::open(db_path).expect("raw db");
    let result = db
        .execute(
            "SELECT COUNT(*) AS n FROM embedding_spaces WHERE embedding_space_id = $space",
            &std::collections::HashMap::from([(
                "space".to_string(),
                Value::Text(space_id.to_string()),
            )]),
        )
        .expect("count");
    match result.rows[0][0] {
        Value::Int64(value) => value,
        _ => 0,
    }
}

fn embedding_space_created_at(
    db_path: &Path,
    space_id: &str,
) -> Result<Value, contextdb_core::Error> {
    let db = Database::open(db_path).expect("raw db");
    let result = db.execute(
        "SELECT created_at FROM embedding_spaces WHERE embedding_space_id = $space",
        &std::collections::HashMap::from([(
            "space".to_string(),
            Value::Text(space_id.to_string()),
        )]),
    )?;
    Ok(result
        .rows
        .first()
        .and_then(|row| row.first())
        .cloned()
        .unwrap_or(Value::Null))
}

#[test]
fn embedding_space_collision_with_builtin_id_refused_at_store_open() {
    let result = open_with(vec![text_space(
        "text_default_nomic_v1_5_768",
        "consumer",
        None,
    )]);
    assert!(matches!(
        result,
        Err(CgError::EmbeddingSpaceIdReserved { .. })
    ));
}

#[test]
fn embedding_space_idempotent_redeclaration_succeeds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("registry.db");
    let first = registry_space("consumer_text_a", "1");
    let second = first.clone();
    let first_open = open_path(&db_path, vec![first.clone(), second]);
    let first_open_debug = format!("{first_open:?}");
    drop(first_open);
    let first_created_at = embedding_space_created_at(&db_path, "consumer_text_a");
    let second_open = open_path(&db_path, vec![first]);
    let second_open_ok = second_open
        .as_ref()
        .is_ok_and(|store| store.last_query_trace().as_deref() == Ok("consumer_text_a"));
    let second_open_debug = format!("{second_open:?}");
    drop(second_open);
    let second_created_at = embedding_space_created_at(&db_path, "consumer_text_a");
    let created_at_preserved = matches!(
        (&first_created_at, &second_created_at),
        (Ok(Value::TxId(first)), Ok(Value::TxId(second))) if first == second
    );
    let row_count = embedding_space_row_count(&db_path, "consumer_text_a");
    assert!(
        second_open_ok && row_count == 1 && created_at_preserved,
        "expected idempotent redeclaration to reopen, preserve first-boot created_at TxId, and select consumer_text_a without duplicate rows, first_open={first_open_debug}, first_created_at={first_created_at:?}, second_open={second_open_debug}, second_created_at={second_created_at:?}, row_count={row_count}"
    );
}

#[test]
fn embedding_space_redeclaration_with_field_mismatch_rejects() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("registry-dimension.db");
    let first = registry_space("consumer_text_a", "1");
    let first_open = open_path(&db_path, vec![first]);
    let first_open_debug = format!("{first_open:?}");
    drop(first_open);
    let mut second = registry_space("consumer_text_a", "1");
    second.dimension = 512;
    let result = open_path(&db_path, vec![second]);
    assert!(
        matches!(
            result,
            Err(CgError::EmbeddingSpaceConfigMismatch {
                field: "dimension",
                ..
            })
        ),
        "dimension mismatch must be detected across reopen without replacing the persisted row, first_open={first_open_debug}, result={result:?}"
    );
}

#[test]
fn embedding_space_redeclaration_with_model_version_mismatch_rejects() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("registry-version.db");
    let first_open = open_path(&db_path, vec![registry_space("consumer_text_a", "1")]);
    let first_open_debug = format!("{first_open:?}");
    drop(first_open);
    let result = open_path(&db_path, vec![registry_space("consumer_text_a", "2")]);
    assert!(
        matches!(
            result,
            Err(CgError::EmbeddingSpaceConfigMismatch {
                field: "model_version",
                ..
            })
        ),
        "model_version mismatch must be detected across reopen without delete-and-reinsert, first_open={first_open_debug}, result={result:?}"
    );
}

#[test]
fn embedding_space_two_declarations_default_for_evidence_kind_returns_ambiguous() {
    let result = open_with(vec![
        text_space("text_a", "1", Some(EvidenceKind::Text)),
        text_space("text_b", "1", Some(EvidenceKind::Text)),
    ]);
    assert!(matches!(result, Err(CgError::AmbiguousEmbeddingSpace)));
}

#[test]
fn embedding_space_consumer_can_override_default_with_distinct_id_and_explicit_default_flag() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("override.db");
    let result = open_path(
        &db_path,
        vec![text_space("consumer_text_a", "1", Some(EvidenceKind::Text))],
    );
    let initial_debug = format!("{result:?}");
    drop(result);
    let reopened_without_override = open_path(&db_path, Vec::new());
    assert!(
        reopened_without_override.as_ref().is_ok_and(
            |store| store.last_query_trace().as_deref() == Ok("text_default_nomic_v1_5_768")
        ),
        "consumer default override must be bounded to the boot and fall back to the built-in on reopen without the override, initial={initial_debug}, reopened_without_override={reopened_without_override:?}"
    );
    let override_boot = open_path(
        &db_path,
        vec![text_space("consumer_text_a", "1", Some(EvidenceKind::Text))],
    );
    assert!(
        override_boot
            .as_ref()
            .is_ok_and(|store| store.last_query_trace().as_deref() == Ok("consumer_text_a")),
        "expected explicit consumer default to override built-in default for this boot only, got {override_boot:?}"
    );
}

#[test]
fn consumer_embedder_trait_embedding_space_mismatch_at_registration_rejects_with_typed_error() {
    let registry_declared = text_space("consumer_text_a", "1", None);
    let embedder = Arc::new(MockEmbedder::new("consumer_text_b", 768));
    let result = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: std::env::temp_dir().join(format!(
                "embedding-space-registration-{}.db",
                uuid::Uuid::now_v7()
            )),
            extra_embedding_spaces: vec![registry_declared.clone()],
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: registry_declared,
            embedder,
        }],
    );
    assert!(matches!(
        result,
        Err(CgError::EmbeddingSpaceIdMismatch {
            embedder_declared,
            registry_declared
        }) if embedder_declared == "consumer_text_b" && registry_declared == "consumer_text_a"
    ));
}
