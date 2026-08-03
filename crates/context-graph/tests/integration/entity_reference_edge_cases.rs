use std::collections::BTreeMap;
#[cfg(context_graph_internal_test_support)]
use std::collections::BTreeSet;
use std::path::PathBuf;
#[cfg(context_graph_internal_test_support)]
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use context_graph::{
    CgError, CreateContext, CreateEntity, DistanceMetric, EmbeddingId, EmbeddingSpaceDecl,
    EntityId, EntityType, EvidenceKind, EvidenceProducer, Store, StoreConfig, TxId,
};
#[cfg(context_graph_internal_test_support)]
use contextdb_core::{Error as EngineError, RowId};
use contextdb_core::{Value, VectorIndexRef};
use contextdb_engine::Database;
#[cfg(context_graph_internal_test_support)]
use contextdb_engine::plugin::{CommitSource, DatabasePlugin};
#[cfg(context_graph_internal_test_support)]
use contextdb_tx::WriteSet;

const FACE_REFERENCE_SPACE: &str = "face_reference_v1";

fn store() -> Store {
    fixture("entity-reference-edge").store
}

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    store: Store,
}

fn store_config(db_path: PathBuf) -> StoreConfig {
    StoreConfig {
        db_path,
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
    }
}

fn fixture(name: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let store = Store::open(store_config(db_path.clone())).expect("open");
    Fixture {
        _dir: dir,
        db_path,
        store,
    }
}

#[cfg(context_graph_internal_test_support)]
fn fixture_with_database_plugin(name: &str, plugin: Arc<dyn DatabasePlugin>) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let store = Store::open_with_database_plugin(store_config(db_path.clone()), plugin)
        .expect("open with plugin");
    Fixture {
        _dir: dir,
        db_path,
        store,
    }
}

#[cfg(context_graph_internal_test_support)]
#[derive(Default)]
struct EntityReferenceFkProbe {
    target: Mutex<Option<EntityReferenceFkTarget>>,
    saw_failed_engine_fk_write: AtomicBool,
}

#[cfg(context_graph_internal_test_support)]
#[derive(Clone)]
struct EntityReferenceFkTarget {
    entity_id: uuid::Uuid,
    context_id: uuid::Uuid,
    vector: Vec<f32>,
}

#[cfg(context_graph_internal_test_support)]
impl EntityReferenceFkProbe {
    fn arm(&self, entity_id: uuid::Uuid, context_id: uuid::Uuid, vector: Vec<f32>) {
        *self.target.lock().expect("target") = Some(EntityReferenceFkTarget {
            entity_id,
            context_id,
            vector,
        });
        self.saw_failed_engine_fk_write
            .store(false, Ordering::SeqCst);
    }

    fn saw_engine_fk_path(&self) -> bool {
        self.saw_failed_engine_fk_write.load(Ordering::SeqCst)
    }
}

#[cfg(context_graph_internal_test_support)]
impl DatabasePlugin for EntityReferenceFkProbe {
    fn commit_failed(&self, ws: &WriteSet, source: CommitSource, error: &EngineError) {
        let Some(target) = self.target.lock().expect("target").clone() else {
            return;
        };
        let reference_row = write_set_row_for_entity_reference(ws, &target);
        let handle_row = write_set_handle_for_entity_reference(ws, &target);
        let reference_vector_tx = reference_row.and_then(|row_id| {
            write_set_vector_tx(
                ws,
                row_id,
                "entity_reference_embeddings_512",
                "vector",
                &target.vector,
            )
        });
        let txs = [
            reference_row.and_then(|row_id| write_set_row_tx(ws, row_id)),
            handle_row.and_then(|row_id| write_set_row_tx(ws, row_id)),
            reference_vector_tx,
        ]
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>();
        if reference_row.is_some()
            && handle_row.is_some()
            && reference_vector_tx.is_some()
            && txs.len() == 1
            && source == CommitSource::User
            && entity_reference_fk_error(error)
        {
            self.saw_failed_engine_fk_write
                .store(true, Ordering::SeqCst);
        }
    }
}

#[cfg(context_graph_internal_test_support)]
fn entity_reference_fk_error(error: &EngineError) -> bool {
    matches!(
        error,
        EngineError::ForeignKeyViolation {
            child_table,
            child_columns,
            parent_table,
            ..
        } if parent_table == "entities"
            && (child_table == "entity_reference_embeddings_512"
                || child_table == "entity_reference_embedding_handles")
            && child_columns.iter().any(|column| column == "context_id")
            && child_columns.iter().any(|column| column == "entity_id" || column == "target_id")
    )
}

#[cfg(context_graph_internal_test_support)]
fn write_set_row_for_entity_reference(
    ws: &WriteSet,
    target: &EntityReferenceFkTarget,
) -> Option<RowId> {
    ws.relational_inserts
        .iter()
        .filter(|(table_name, _)| table_name == "entity_reference_embeddings_512")
        .find_map(|(_, row)| {
            (row.values.get("entity_id") == Some(&Value::Uuid(target.entity_id))
                && row.values.get("context_id") == Some(&Value::Uuid(target.context_id))
                && row.values.get("embedding_space_id")
                    == Some(&Value::Text(FACE_REFERENCE_SPACE.to_string())))
            .then_some(row.row_id)
        })
}

#[cfg(context_graph_internal_test_support)]
fn write_set_handle_for_entity_reference(
    ws: &WriteSet,
    target: &EntityReferenceFkTarget,
) -> Option<RowId> {
    ws.relational_inserts
        .iter()
        .filter(|(table_name, _)| table_name == "entity_reference_embedding_handles")
        .find_map(|(_, row)| {
            (row.values.get("target_id") == Some(&Value::Uuid(target.entity_id))
                && row.values.get("context_id") == Some(&Value::Uuid(target.context_id))
                && row.values.get("embedding_space_id")
                    == Some(&Value::Text(FACE_REFERENCE_SPACE.to_string())))
            .then_some(row.row_id)
        })
}

#[cfg(context_graph_internal_test_support)]
fn write_set_row_tx(ws: &WriteSet, row_id: RowId) -> Option<u64> {
    ws.relational_inserts
        .iter()
        .find_map(|(_, row)| (row.row_id == row_id).then_some(row.created_tx.0))
}

#[cfg(context_graph_internal_test_support)]
fn write_set_vector_tx(
    ws: &WriteSet,
    row_id: RowId,
    table: &str,
    column: &str,
    vector: &[f32],
) -> Option<u64> {
    ws.vector_inserts.iter().find_map(|entry| {
        (entry.row_id == row_id
            && entry.index.table == table
            && entry.index.column == column
            && entry.vector == vector)
            .then_some(entry.created_tx.0)
    })
}

#[cfg(context_graph_internal_test_support)]
fn count_from_store(
    store: &Store,
    sql: &str,
    params: &std::collections::HashMap<String, Value>,
) -> i64 {
    let result = store
        .execute_sql_for_test(sql, params)
        .expect("count query");
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

#[cfg(context_graph_internal_test_support)]
fn params(pairs: Vec<(&str, Value)>) -> std::collections::HashMap<String, Value> {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

fn leaked_entity_reference_vector_count(db_path: &PathBuf, vector: &[f32]) -> usize {
    let db = Database::open(db_path).expect("raw database open");
    db.query_vector(
        VectorIndexRef::new("entity_reference_embeddings_512", "vector"),
        vector,
        10,
        None,
        db.snapshot(),
    )
    .expect("raw entity reference vector query")
    .into_iter()
    .filter(|(_, score)| *score > 0.99)
    .count()
}

fn raw_entity_reference_rows(db_path: &PathBuf, entity_id: EntityId) -> Vec<(EmbeddingId, TxId)> {
    let db = Database::open(db_path).expect("raw database open");
    let mut rows = db
        .scan("entity_reference_embeddings_512", db.snapshot())
        .expect("raw entity reference scan")
        .into_iter()
        .filter_map(|row| {
            if row.values.get("entity_id") != Some(&Value::Uuid(entity_id.as_uuid())) {
                return None;
            }
            let id = match row.values.get("id") {
                Some(Value::Uuid(id)) => EmbeddingId::from(*id),
                _ => return None,
            };
            let created_at = match row.values.get("created_at") {
                Some(Value::TxId(txid)) => *txid,
                _ => return None,
            };
            Some((id, created_at))
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    rows
}

fn raw_entity_reference_handle_count(db_path: &PathBuf, entity_id: EntityId) -> usize {
    let db = Database::open(db_path).expect("raw database open");
    db.scan("entity_reference_embedding_handles", db.snapshot())
        .expect("raw entity reference handle scan")
        .into_iter()
        .filter(|row| row.values.get("target_id") == Some(&Value::Uuid(entity_id.as_uuid())))
        .count()
}

fn axis_vector(dimension: usize, axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];
    vector[axis] = 1.0;
    vector
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
    #[cfg(context_graph_internal_test_support)]
    {
        assert_unknown_entity_reference_enrollment_reaches_engine_fk();
    }
    #[cfg(not(context_graph_internal_test_support))]
    {
        let unknown_entity_id = EntityId::new_v7();
        let result = store().enroll_entity_reference(
            unknown_entity_id,
            FACE_REFERENCE_SPACE,
            None,
            vec![1.0; 512],
            producer(),
        );
        assert!(
            matches!(
                result,
                Err(CgError::EntityNotFound { entity_id }) if entity_id == unknown_entity_id
            ),
            "expected typed missing-entity rejection, got {result:?}"
        );
    }
}

#[cfg(context_graph_internal_test_support)]
fn assert_unknown_entity_reference_enrollment_reaches_engine_fk() {
    let probe = Arc::new(EntityReferenceFkProbe::default());
    let fixture = fixture_with_database_plugin("entity-reference-unknown-fk", probe.clone());
    let Fixture {
        _dir,
        db_path,
        store,
    } = fixture;
    let context = store
        .create_context(CreateContext {
            name: format!("unknown-fk-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let unknown_entity_id = EntityId::new_v7();
    let vector = vec![1.0; 512];
    probe.arm(
        unknown_entity_id.as_uuid(),
        context.id.as_uuid(),
        vector.clone(),
    );
    let result = store.enroll_entity_reference(
        unknown_entity_id,
        FACE_REFERENCE_SPACE,
        None,
        vector.clone(),
        producer(),
    );
    let reference_rows = count_from_store(
        &store,
        "SELECT COUNT(*) AS n FROM entity_reference_embeddings_512 WHERE entity_id = $entity_id AND context_id = $context_id",
        &params(vec![
            ("entity_id", Value::Uuid(unknown_entity_id.as_uuid())),
            ("context_id", Value::Uuid(context.id.as_uuid())),
        ]),
    );
    let handle_rows = count_from_store(
        &store,
        "SELECT COUNT(*) AS n FROM entity_reference_embedding_handles WHERE target_id = $entity_id AND context_id = $context_id",
        &params(vec![
            ("entity_id", Value::Uuid(unknown_entity_id.as_uuid())),
            ("context_id", Value::Uuid(context.id.as_uuid())),
        ]),
    );
    let engine_fk_path_seen = probe.saw_engine_fk_path();
    drop(store);
    let leaked_vectors = leaked_entity_reference_vector_count(&db_path, &vector);
    assert!(
        matches!(
            result,
            Err(CgError::EntityNotFound { entity_id }) if entity_id == unknown_entity_id
        ),
        "expected typed missing-entity rejection from engine FK path with no durable row or vector leak, result={result:?}, engine_fk_path_seen={engine_fk_path_seen}, reference_rows={reference_rows}, handle_rows={handle_rows}, leaked_vectors={leaked_vectors}"
    );
    assert!(
        engine_fk_path_seen && reference_rows == 0 && handle_rows == 0 && leaked_vectors == 0,
        "unknown entity enrollment must reach a failed contextdb WriteSet with entity-reference row, handle, and row-bound vector in one tx, then roll all of it back; result={result:?}, engine_fk_path_seen={engine_fk_path_seen}, reference_rows={reference_rows}, handle_rows={handle_rows}, leaked_vectors={leaked_vectors}"
    );
}

#[test]
fn entity_reference_embedding_parent_entity_archive_does_not_orphan_child_rows() {
    let fixture = fixture("entity-reference-archive");
    let db_path = fixture.db_path.clone();
    let entity = seed_entity(&fixture.store, "Kumar");
    let first_vector = axis_vector(512, 0);
    let second_vector = axis_vector(512, 1);
    let third_vector = axis_vector(512, 2);
    let rejected_vector = axis_vector(512, 3);
    let first_id = fixture.store.enroll_entity_reference(
        entity.id,
        FACE_REFERENCE_SPACE,
        Some("front".to_string()),
        first_vector,
        producer(),
    );
    let second_id = fixture.store.enroll_entity_reference(
        entity.id,
        FACE_REFERENCE_SPACE,
        Some("side".to_string()),
        second_vector,
        producer(),
    );
    let third_id = fixture.store.enroll_entity_reference(
        entity.id,
        FACE_REFERENCE_SPACE,
        Some("overhead".to_string()),
        third_vector,
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
    let expected_archived_at =
        fixture
            .store
            .entity_history_graph(entity.id)
            .ok()
            .and_then(|graph| {
                let archived_heads = graph.archived_heads;
                graph
                    .snapshots
                    .into_iter()
                    .filter(|snapshot| archived_heads.contains(&snapshot.id))
                    .map(|snapshot| snapshot.valid_from)
                    .max_by_key(|txid| txid.0)
            });
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
        rejected_vector.clone(),
        producer(),
    );
    let rows_after_rejected_write = fixture
        .store
        .list_entity_reference_embeddings_for_entity(entity.id);
    let after_rejected_write_created_at = rows_after_rejected_write
        .as_ref()
        .map(|items| {
            items
                .iter()
                .map(|row| (row.id, row.created_at))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    drop(fixture.store);
    let raw_rows_after_rejected_write = raw_entity_reference_rows(&db_path, entity.id);
    let raw_handle_count_after_rejected_write =
        raw_entity_reference_handle_count(&db_path, entity.id);
    let rejected_vector_hits_after_rejected_write =
        leaked_entity_reference_vector_count(&db_path, &rejected_vector);
    let reopened_store = Store::open(store_config(db_path.clone())).expect("reopen after archive");
    let rows_after_reopen = reopened_store.list_entity_reference_embeddings_for_entity(entity.id);
    let after_reopen_created_at = rows_after_reopen
        .as_ref()
        .map(|items| {
            items
                .iter()
                .map(|row| (row.id, row.created_at))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    drop(reopened_store);
    let raw_rows_after_reopen = raw_entity_reference_rows(&db_path, entity.id);
    let raw_handle_count_after_reopen = raw_entity_reference_handle_count(&db_path, entity.id);
    let rejected_vector_hits_after_reopen =
        leaked_entity_reference_vector_count(&db_path, &rejected_vector);

    assert!(
        first_id
            .as_ref()
            .is_ok_and(|value| value.is_valid() && *value != EmbeddingId::default())
            && second_id
                .as_ref()
                .is_ok_and(|value| value.is_valid() && *value != EmbeddingId::default())
            && third_id
                .as_ref()
                .is_ok_and(|value| value.is_valid() && *value != EmbeddingId::default())
            && before_rows.as_ref().is_ok_and(|items| items.len() == 3)
            && archived.is_ok()
            && rows.as_ref().is_ok_and(|items| items.len() == 3
                && items
                    .iter()
                    .any(|row| Some(&row.id) == first_id.as_ref().ok())
                && items
                    .iter()
                    .any(|row| Some(&row.id) == second_id.as_ref().ok())
                && items
                    .iter()
                    .any(|row| Some(&row.id) == third_id.as_ref().ok()))
            && rows_after_rejected_write
                .as_ref()
                .is_ok_and(|items| items.len() == 3
                    && items
                        .iter()
                        .any(|row| Some(&row.id) == first_id.as_ref().ok())
                    && items
                        .iter()
                        .any(|row| Some(&row.id) == second_id.as_ref().ok())
                    && items
                        .iter()
                        .any(|row| Some(&row.id) == third_id.as_ref().ok()))
            && before_created_at.len() == 3
            && before_created_at == after_created_at
            && before_created_at == after_rejected_write_created_at
            && before_created_at == after_reopen_created_at
            && raw_rows_after_rejected_write == before_created_at
            && raw_rows_after_reopen == before_created_at
            && raw_handle_count_after_rejected_write == 3
            && raw_handle_count_after_reopen == 3
            && rejected_vector_hits_after_rejected_write == 0
            && rejected_vector_hits_after_reopen == 0
            && matches!(
                new_enrollment,
                Err(CgError::EntityArchived {
                    entity_id: archived_entity,
                    archived_at
                }) if archived_entity == entity.id
                    && Some(archived_at) == expected_archived_at
            ),
        "expected archive to preserve exact persisted child rows/handles/vectors while rejecting new enrollment with the archive TxId and no fourth row, first_id={first_id:?}, second_id={second_id:?}, third_id={third_id:?}, before_rows={before_rows:?}, archived={archived:?}, expected_archived_at={expected_archived_at:?}, rows={rows:?}, rows_after_rejected_write={rows_after_rejected_write:?}, rows_after_reopen={rows_after_reopen:?}, before_created_at={before_created_at:?}, after_created_at={after_created_at:?}, after_rejected_write_created_at={after_rejected_write_created_at:?}, after_reopen_created_at={after_reopen_created_at:?}, raw_rows_after_rejected_write={raw_rows_after_rejected_write:?}, raw_rows_after_reopen={raw_rows_after_reopen:?}, raw_handle_count_after_rejected_write={raw_handle_count_after_rejected_write}, raw_handle_count_after_reopen={raw_handle_count_after_reopen}, rejected_vector_hits_after_rejected_write={rejected_vector_hits_after_rejected_write}, rejected_vector_hits_after_reopen={rejected_vector_hits_after_reopen}, new_enrollment={new_enrollment:?}"
    );
}
