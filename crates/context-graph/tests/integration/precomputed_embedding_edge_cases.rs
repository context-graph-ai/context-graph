use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::{
    CG_EDGE_SYSTEM_TAG, CgError, ContextId, CreateContext, CreateDigest, CreateEntity, DigestId,
    DistanceMetric, EmbeddingId, EmbeddingSpaceDecl, EmbeddingTarget, EntityId, EntityType,
    EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, PrecomputedEmbedding,
    RecordObservation, RetentionStatus, Store, StoreConfig,
};
use contextdb_core::{Error as EngineError, Value, VectorIndexRef};
use contextdb_engine::{Database, QueryResult};

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";
const FACE_SPACE: &str = "face_reference_v1";

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    store: Store,
    context_id: ContextId,
    entity_id: EntityId,
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "external".to_string(),
        model_name: "encoder".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn face_space_decl() -> EmbeddingSpaceDecl {
    EmbeddingSpaceDecl {
        embedding_space_id: FACE_SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "face-reference".to_string(),
        model_version: "v1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: None,
        bindings: Vec::new(),
    }
}

fn fixture(name: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        extra_embedding_spaces: vec![face_space_decl()],
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
            entity_type: EntityType::Service,
            name: "screen-event".to_string(),
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

fn text_vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[axis] = 1.0;
    vector
}

fn embedding(vector: Vec<f32>, target: EmbeddingTarget) -> PrecomputedEmbedding {
    PrecomputedEmbedding {
        id: EmbeddingId::new_v7(),
        embedding_space_id: TEXT_SPACE.to_string(),
        vector,
        target,
        produced_by: producer(),
    }
}

fn evidence(evidence_id: EvidenceId) -> EvidenceRef {
    EvidenceRef {
        id: evidence_id,
        kind: EvidenceKind::Text,
        source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
        mime_type: Some("text/plain".to_string()),
        retention_status: RetentionStatus::RetainedExternal,
        ..Default::default()
    }
}

fn record(
    fixture: &Fixture,
    embedding: PrecomputedEmbedding,
) -> context_graph::Result<context_graph::RecordObservationResult> {
    let evidence_id = match embedding.target {
        EmbeddingTarget::Evidence(id) => id,
        EmbeddingTarget::EntityReference(_, _) | EmbeddingTarget::Digest(_) => EvidenceId::new_v7(),
    };
    fixture.store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: fixture.entity_id,
        context_id: fixture.context_id,
        observation_type: "precomputed_embedding_edge".to_string(),
        source: "edge".to_string(),
        observed_at: Utc::now(),
        evidence: vec![evidence(evidence_id)],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![embedding],
    })
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

fn stored_vector(db: &Database, table: &str, column: &str, id: uuid::Uuid) -> Option<Vec<f32>> {
    let sql = format!("SELECT {column} FROM {table} WHERE id = $id");
    let result = db
        .execute(&sql, &params(vec![("id", Value::Uuid(id))]))
        .expect("vector select");
    result.rows.first().and_then(|row| match row.first() {
        Some(Value::Vector(vector)) => Some(vector.clone()),
        _ => None,
    })
}

fn vector_close_enough(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| (left - right).abs() <= 1.0e-6)
}

fn immutable_result(result: std::result::Result<QueryResult, EngineError>, table: &str) -> bool {
    matches!(result, Err(EngineError::ImmutableTable(ref actual)) if actual == table)
}

fn immutable_mutations_rejected_and_row_unchanged(
    db: &Database,
    table: &str,
    id: EmbeddingId,
    target_id: uuid::Uuid,
) -> bool {
    let row_before = count(
        db,
        &format!("SELECT COUNT(*) AS n FROM {table} WHERE id = $id AND target_id = $target_id"),
        &params(vec![
            ("id", Value::Uuid(id.as_uuid())),
            ("target_id", Value::Uuid(target_id)),
        ]),
    );
    let update_id = db.execute(
        &format!("UPDATE {table} SET id = $new_id WHERE id = $id"),
        &params(vec![
            ("id", Value::Uuid(id.as_uuid())),
            ("new_id", Value::Uuid(EmbeddingId::new_v7().as_uuid())),
        ]),
    );
    let update_producer = db.execute(
        &format!("UPDATE {table} SET produced_by = $producer WHERE id = $id"),
        &params(vec![
            ("id", Value::Uuid(id.as_uuid())),
            (
                "producer",
                Value::Json(serde_json::to_value(producer()).expect("producer json")),
            ),
        ]),
    );
    let update_target = db.execute(
        &format!("UPDATE {table} SET target_id = $new_target WHERE id = $id"),
        &params(vec![
            ("id", Value::Uuid(id.as_uuid())),
            ("new_target", Value::Uuid(uuid::Uuid::now_v7())),
        ]),
    );
    let delete = db.execute(
        &format!("DELETE FROM {table} WHERE id = $id"),
        &params(vec![("id", Value::Uuid(id.as_uuid()))]),
    );
    let row_after = count(
        db,
        &format!("SELECT COUNT(*) AS n FROM {table} WHERE id = $id AND target_id = $target_id"),
        &params(vec![
            ("id", Value::Uuid(id.as_uuid())),
            ("target_id", Value::Uuid(target_id)),
        ]),
    );
    row_before == 1
        && row_after == 1
        && immutable_result(update_id, table)
        && immutable_result(update_producer, table)
        && immutable_result(update_target, table)
        && immutable_result(delete, table)
}

fn cross_context_handle_insert_rejected(
    db: &Database,
    table: &str,
    wrong_context: ContextId,
    target_id: uuid::Uuid,
    embedding_space_id: &str,
) -> bool {
    let sql = format!(
        "INSERT INTO {table} (id, context_id, target_id, embedding_space_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $target_id, $space, $producer, $producer_system, $created_at)"
    );
    let insert = db.execute(
        &sql,
        &params(vec![
            ("id", Value::Uuid(EmbeddingId::new_v7().as_uuid())),
            ("context_id", Value::Uuid(wrong_context.as_uuid())),
            ("target_id", Value::Uuid(target_id)),
            ("space", Value::Text(embedding_space_id.to_string())),
            (
                "producer",
                Value::Json(serde_json::to_value(producer()).expect("producer json")),
            ),
            ("producer_system", Value::Text(producer().system)),
            ("created_at", Value::TxId(context_graph::TxId(1))),
        ]),
    );
    matches!(
        insert,
        Err(EngineError::ForeignKeyViolation {
            child_table,
            child_columns,
            parent_table,
            ..
        }) if child_table == table
            && child_columns == vec!["context_id".to_string(), "target_id".to_string()]
            && matches!(parent_table.as_str(), "evidence" | "digests" | "entities")
    )
}

#[test]
fn precomputed_embedding_zero_length_vector_rejected_with_typed_empty_vector_error() {
    let fixture = fixture("zero-vector");
    let result = record(
        &fixture,
        embedding(Vec::new(), EmbeddingTarget::Evidence(EvidenceId::new_v7())),
    );
    assert!(
        matches!(result, Err(CgError::EmptyVector)),
        "expected EmptyVector, got {result:?}"
    );
}

#[test]
fn precomputed_embedding_with_nan_or_inf_floats_rejected_at_record_observation() {
    for (case, actual, value) in [
        ("nan", "NaN", f32::NAN),
        ("inf", "inf", f32::INFINITY),
        ("neg_inf", "-inf", f32::NEG_INFINITY),
        ("subnormal", "subnormal", f32::from_bits(1)),
    ] {
        let fixture = fixture(&format!("invalid-float-{case}"));
        let mut vector = text_vector(0);
        vector[17] = value;
        let evidence_id = EvidenceId::new_v7();
        let result = record(
            &fixture,
            embedding(vector, EmbeddingTarget::Evidence(evidence_id)),
        );
        let db = raw_db(&fixture.db_path, fixture.store);
        let handle_rows = count(
            &db,
            "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id",
            &params(vec![("target_id", Value::Uuid(evidence_id.as_uuid()))]),
        );
        assert!(
            matches!(
                result,
                Err(CgError::InvalidVectorFloat {
                    field: "vector",
                    index: 17,
                    actual: got,
                }) if got == actual
            ) && handle_rows == 0,
            "expected typed InvalidVectorFloat with rollback for {case}, result={result:?}, handle_rows={handle_rows}"
        );
    }
}

#[test]
fn precomputed_embedding_target_evidence_in_different_context_rejected_at_record_observation() {
    let fixture = fixture("cross-context-source");
    let target_evidence_id = EvidenceId::new_v7();
    let inserted = record(
        &fixture,
        embedding(
            text_vector(0),
            EmbeddingTarget::Evidence(target_evidence_id),
        ),
    );
    assert!(
        inserted.is_ok(),
        "source evidence setup failed: {inserted:?}"
    );

    let target_context = fixture
        .store
        .create_context(CreateContext {
            name: format!("target-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("target context");
    let target_entity = fixture
        .store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "target".to_string(),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: target_context.id,
        })
        .expect("target entity");
    let result = fixture.store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: target_entity.id,
        context_id: target_context.id,
        observation_type: "cross_context_embedding_target".to_string(),
        source: "edge".to_string(),
        observed_at: Utc::now(),
        evidence: vec![evidence(EvidenceId::new_v7())],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![embedding(
            text_vector(1),
            EmbeddingTarget::Evidence(target_evidence_id),
        )],
    });
    let db = raw_db(&fixture.db_path, fixture.store);
    let leaked_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $id AND context_id = $context_id",
        &params(vec![
            ("id", Value::Uuid(target_evidence_id.as_uuid())),
            ("context_id", Value::Uuid(target_context.id.as_uuid())),
        ]),
    );

    assert!(
        matches!(
            result,
            Err(CgError::CrossContextEmbeddingTargetRejected {
                handle_context,
                target_context: actual_target_context,
            }) if handle_context == target_context.id && actual_target_context == fixture.context_id
        ) && leaked_rows == 0,
        "expected cross-context target rejection naming both contexts and no target-store leak, result={result:?}, leaked_rows={leaked_rows}"
    );
}

#[test]
fn precomputed_embedding_caller_supplied_reserved_producer_system_tag_rejected() {
    let fixture = fixture("reserved-producer");
    let evidence_id = EvidenceId::new_v7();
    let mut input = embedding(text_vector(0), EmbeddingTarget::Evidence(evidence_id));
    input.produced_by.system = CG_EDGE_SYSTEM_TAG.to_string();
    let query_vector = input.vector.clone();
    let Fixture {
        _dir,
        db_path,
        store,
        context_id,
        entity_id,
    } = fixture;
    let snapshot_params = params(vec![
        ("context_id", Value::Uuid(context_id.as_uuid())),
        ("entity_id", Value::Uuid(entity_id.as_uuid())),
    ]);
    let baseline_history = store
        .entity_history(entity_id)
        .expect("baseline entity history");
    let baseline_snapshot = baseline_history
        .first()
        .expect("baseline entity snapshot")
        .clone();
    drop(store);
    let baseline_db = Database::open(&db_path).expect("baseline raw database open");
    let context_params = params(vec![("context_id", Value::Uuid(context_id.as_uuid()))]);
    let pre_observation_rows = count(
        &baseline_db,
        "SELECT COUNT(*) AS n FROM observations WHERE context_id = $context_id",
        &context_params,
    );
    let pre_snapshot_rows = count(
        &baseline_db,
        "SELECT COUNT(*) AS n FROM entity_snapshots WHERE context_id = $context_id AND entity_id = $entity_id",
        &snapshot_params,
    );
    let pre_snapshot_observation_rows = count(
        &baseline_db,
        "SELECT COUNT(*) AS n FROM entity_snapshot_observations WHERE context_id = $context_id AND entity_id = $entity_id",
        &snapshot_params,
    );
    let pre_evidence_rows = count(
        &baseline_db,
        "SELECT COUNT(*) AS n FROM evidence WHERE context_id = $context_id",
        &context_params,
    );
    let pre_evidence_handle_rows = count(
        &baseline_db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE context_id = $context_id",
        &context_params,
    );
    let pre_entity_reference_handle_rows = count(
        &baseline_db,
        "SELECT COUNT(*) AS n FROM entity_reference_embedding_handles WHERE context_id = $context_id",
        &context_params,
    );
    let pre_digest_handle_rows = count(
        &baseline_db,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE context_id = $context_id",
        &context_params,
    );
    let pre_legacy_handle_rows = if baseline_db.table_meta("embedding_handles").is_some() {
        count(
            &baseline_db,
            "SELECT COUNT(*) AS n FROM embedding_handles WHERE context_id = $context_id",
            &context_params,
        )
    } else {
        0
    };
    let pre_raw_vector_hits = baseline_db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &query_vector,
            8,
            None,
            baseline_db.snapshot(),
        )
        .expect("baseline reserved-tag vector query")
        .len();
    drop(baseline_db);
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        extra_embedding_spaces: vec![face_space_decl()],
        ..Default::default()
    })
    .expect("reopen store for reserved producer attempt");
    let fixture = Fixture {
        _dir,
        db_path,
        store,
        context_id,
        entity_id,
    };
    let result = record(&fixture, input);
    let db = raw_db(&fixture.db_path, fixture.store);
    let observation_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM observations WHERE context_id = $context_id",
        &context_params,
    );
    let snapshot_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_snapshots WHERE context_id = $context_id AND entity_id = $entity_id",
        &snapshot_params,
    );
    let snapshot_observation_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_snapshot_observations WHERE context_id = $context_id AND entity_id = $entity_id",
        &snapshot_params,
    );
    let baseline_snapshot_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_snapshots WHERE id = $snapshot_id AND context_id = $context_id AND entity_id = $entity_id AND valid_from = $valid_from AND valid_to IS NULL",
        &params(vec![
            ("snapshot_id", Value::Uuid(baseline_snapshot.id.as_uuid())),
            ("context_id", Value::Uuid(context_id.as_uuid())),
            ("entity_id", Value::Uuid(entity_id.as_uuid())),
            ("valid_from", Value::TxId(baseline_snapshot.valid_from)),
        ]),
    );
    let evidence_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE context_id = $context_id",
        &context_params,
    );
    let evidence_handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE context_id = $context_id",
        &context_params,
    );
    let entity_reference_handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_reference_embedding_handles WHERE context_id = $context_id",
        &context_params,
    );
    let digest_handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE context_id = $context_id",
        &context_params,
    );
    let legacy_handle_rows = if db.table_meta("embedding_handles").is_some() {
        count(
            &db,
            "SELECT COUNT(*) AS n FROM embedding_handles WHERE context_id = $context_id",
            &context_params,
        )
    } else {
        0
    };
    let raw_vector_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &query_vector,
            8,
            None,
            db.snapshot(),
        )
        .expect("reserved-tag vector query")
        .len();
    assert!(
        matches!(result, Err(CgError::ReservedProducerTag { ref tag }) if tag == CG_EDGE_SYSTEM_TAG)
            && observation_rows == pre_observation_rows
            && snapshot_rows == pre_snapshot_rows
            && baseline_snapshot_rows == 1
            && snapshot_observation_rows == pre_snapshot_observation_rows
            && evidence_rows == pre_evidence_rows
            && evidence_handle_rows == pre_evidence_handle_rows
            && entity_reference_handle_rows == pre_entity_reference_handle_rows
            && digest_handle_rows == pre_digest_handle_rows
            && legacy_handle_rows == pre_legacy_handle_rows
            && raw_vector_hits == pre_raw_vector_hits,
        "expected reserved producer tag rejection before any observation or new snapshot/evidence/handle/vector write, got result={result:?}, pre_observation_rows={pre_observation_rows}, observation_rows={observation_rows}, pre_snapshot_rows={pre_snapshot_rows}, snapshot_rows={snapshot_rows}, baseline_snapshot={baseline_snapshot:?}, baseline_snapshot_rows={baseline_snapshot_rows}, pre_snapshot_observation_rows={pre_snapshot_observation_rows}, snapshot_observation_rows={snapshot_observation_rows}, pre_evidence_rows={pre_evidence_rows}, evidence_rows={evidence_rows}, pre_evidence_handle_rows={pre_evidence_handle_rows}, evidence_handle_rows={evidence_handle_rows}, pre_entity_reference_handle_rows={pre_entity_reference_handle_rows}, entity_reference_handle_rows={entity_reference_handle_rows}, pre_digest_handle_rows={pre_digest_handle_rows}, digest_handle_rows={digest_handle_rows}, pre_legacy_handle_rows={pre_legacy_handle_rows}, legacy_handle_rows={legacy_handle_rows}, pre_raw_vector_hits={pre_raw_vector_hits}, raw_vector_hits={raw_vector_hits}"
    );
}

#[test]
fn embedding_handle_id_update_attempt_rejected_with_typed_immutable_error_single_thread() {
    let fixture = fixture("immutable-handle");
    let evidence_id = EvidenceId::new_v7();
    let evidence_embedding_id = EmbeddingId::new_v7();
    let evidence_result = record(
        &fixture,
        PrecomputedEmbedding {
            id: evidence_embedding_id,
            ..embedding(text_vector(0), EmbeddingTarget::Evidence(evidence_id))
        },
    );
    let digest_id = DigestId::new_v7();
    let digest_embedding_id = EmbeddingId::new_v7();
    let digest_result = fixture.store.create_digest(CreateDigest {
        id: digest_id,
        source: "immutable-digest".to_string(),
        session_ref: None,
        summary: "digest basis".to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: digest_embedding_id,
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: text_vector(1),
            target: EmbeddingTarget::Digest(digest_id),
            produced_by: producer(),
        },
        context_id: fixture.context_id,
        properties: BTreeMap::new(),
    });
    let reference_result = fixture.store.enroll_entity_reference(
        fixture.entity_id,
        FACE_SPACE,
        Some("front".to_string()),
        vec![1.0; 512],
        producer(),
    );
    let reference_embedding_id = reference_result
        .as_ref()
        .map_or(EmbeddingId::default(), |id| *id);
    let other_context = fixture
        .store
        .create_context(CreateContext {
            name: format!("other-context-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("other context");
    let db = raw_db(&fixture.db_path, fixture.store);
    let evidence_handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id",
        &params(vec![("id", Value::Uuid(evidence_embedding_id.as_uuid()))]),
    );
    let digest_handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE id = $id",
        &params(vec![("id", Value::Uuid(digest_embedding_id.as_uuid()))]),
    );
    let reference_handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_reference_embedding_handles WHERE id = $id",
        &params(vec![("id", Value::Uuid(reference_embedding_id.as_uuid()))]),
    );
    let immutable_updates = [
        (
            "evidence_embedding_handles",
            evidence_embedding_id,
            evidence_id.as_uuid(),
        ),
        (
            "digest_embedding_handles",
            digest_embedding_id,
            digest_id.as_uuid(),
        ),
        (
            "entity_reference_embedding_handles",
            reference_embedding_id,
            fixture.entity_id.as_uuid(),
        ),
    ]
    .iter()
    .all(|(table, id, target_id)| {
        immutable_mutations_rejected_and_row_unchanged(&db, table, *id, *target_id)
    });
    let same_context_fks = [
        (
            "evidence_embedding_handles",
            evidence_id.as_uuid(),
            TEXT_SPACE,
        ),
        ("digest_embedding_handles", digest_id.as_uuid(), TEXT_SPACE),
        (
            "entity_reference_embedding_handles",
            fixture.entity_id.as_uuid(),
            FACE_SPACE,
        ),
    ]
    .iter()
    .all(|(table, target_id, space)| {
        cross_context_handle_insert_rejected(&db, table, other_context.id, *target_id, space)
    });

    assert!(
        evidence_result.is_ok()
            && digest_result.is_ok()
            && reference_result.is_ok()
            && evidence_handle_rows == 1
            && digest_handle_rows == 1
            && reference_handle_rows == 1
            && immutable_updates
            && same_context_fks,
        "expected all variant handle rows to be immutable and guarded by same-context composite FKs, evidence_result={evidence_result:?}, digest_result={digest_result:?}, reference_result={reference_result:?}, evidence_handle_rows={evidence_handle_rows}, digest_handle_rows={digest_handle_rows}, reference_handle_rows={reference_handle_rows}, immutable_updates={immutable_updates}, same_context_fks={same_context_fks}"
    );
}

#[test]
fn embedding_target_routing_evidence_variant_writes_to_evidence_vector_column_digest_variant_writes_to_digest_vector_column()
 {
    let fixture = fixture("target-routing");
    let evidence_id = EvidenceId::new_v7();
    let evidence_embedding_id = EmbeddingId::new_v7();
    let evidence_result = record(
        &fixture,
        PrecomputedEmbedding {
            id: evidence_embedding_id,
            ..embedding(text_vector(0), EmbeddingTarget::Evidence(evidence_id))
        },
    );
    let digest_id = DigestId::new_v7();
    let digest_embedding_id = EmbeddingId::new_v7();
    let digest_result = fixture.store.create_digest(CreateDigest {
        id: digest_id,
        source: "screen-summary".to_string(),
        session_ref: None,
        summary: "evidence summary".to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: digest_embedding_id,
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: text_vector(1),
            target: EmbeddingTarget::Digest(digest_id),
            produced_by: producer(),
        },
        context_id: fixture.context_id,
        properties: BTreeMap::new(),
    });
    let reference_embedding_id = EmbeddingId::new_v7();
    let reference_vector = vec![0.25; 512];
    let reference_result = record(
        &fixture,
        PrecomputedEmbedding {
            id: reference_embedding_id,
            embedding_space_id: FACE_SPACE.to_string(),
            vector: reference_vector.clone(),
            target: EmbeddingTarget::EntityReference(fixture.entity_id, Some("front".to_string())),
            produced_by: producer(),
        },
    );
    let db = raw_db(&fixture.db_path, fixture.store);
    let evidence_handles = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND target_id = $target_id",
        &params(vec![
            ("id", Value::Uuid(evidence_embedding_id.as_uuid())),
            ("target_id", Value::Uuid(evidence_id.as_uuid())),
        ]),
    );
    let digest_handles = count(
        &db,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE id = $id AND target_id = $target_id",
        &params(vec![
            ("id", Value::Uuid(digest_embedding_id.as_uuid())),
            ("target_id", Value::Uuid(digest_id.as_uuid())),
        ]),
    );
    let reference_handles = count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_reference_embedding_handles WHERE id = $id AND target_id = $target_id",
        &params(vec![
            ("id", Value::Uuid(reference_embedding_id.as_uuid())),
            ("target_id", Value::Uuid(fixture.entity_id.as_uuid())),
        ]),
    );
    let evidence_vector = stored_vector(&db, "evidence", "vector_text", evidence_id.as_uuid());
    let digest_vector = stored_vector(&db, "digests", "vector_text", digest_id.as_uuid());
    let reference_vector_row = stored_vector(
        &db,
        "entity_reference_embeddings_512",
        "vector",
        reference_embedding_id.as_uuid(),
    );
    let wrong_evidence_in_digest = count(
        &db,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE target_id = $target_id",
        &params(vec![("target_id", Value::Uuid(evidence_id.as_uuid()))]),
    );
    let wrong_digest_in_evidence = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id",
        &params(vec![("target_id", Value::Uuid(digest_id.as_uuid()))]),
    );
    let wrong_reference_in_evidence_or_digest = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id",
        &params(vec![(
            "target_id",
            Value::Uuid(fixture.entity_id.as_uuid()),
        )]),
    ) + count(
        &db,
        "SELECT COUNT(*) AS n FROM digest_embedding_handles WHERE target_id = $target_id",
        &params(vec![(
            "target_id",
            Value::Uuid(fixture.entity_id.as_uuid()),
        )]),
    );

    assert!(
        evidence_result.is_ok()
            && digest_result.is_ok()
            && reference_result.is_ok()
            && evidence_handles == 1
            && digest_handles == 1
            && reference_handles == 1
            && evidence_vector
                .as_ref()
                .is_some_and(|vector| vector_close_enough(vector, &text_vector(0)))
            && digest_vector
                .as_ref()
                .is_some_and(|vector| vector_close_enough(vector, &text_vector(1)))
            && reference_vector_row
                .as_ref()
                .is_some_and(|vector| vector_close_enough(vector, &reference_vector))
            && wrong_evidence_in_digest == 0
            && wrong_digest_in_evidence == 0
            && wrong_reference_in_evidence_or_digest == 0,
        "expected evidence, digest, and entity-reference variants to write only their intended tables/columns with exact ids/vectors, evidence_result={evidence_result:?}, digest_result={digest_result:?}, reference_result={reference_result:?}, evidence_handles={evidence_handles}, digest_handles={digest_handles}, reference_handles={reference_handles}, evidence_vector={evidence_vector:?}, digest_vector={digest_vector:?}, reference_vector_row={reference_vector_row:?}, wrong_evidence_in_digest={wrong_evidence_in_digest}, wrong_digest_in_evidence={wrong_digest_in_evidence}, wrong_reference_in_evidence_or_digest={wrong_reference_in_evidence_or_digest}"
    );
}

#[test]
fn producer_metadata_model_name_persists_full_string_up_to_documented_max() {
    let fixture = fixture("producer-long-name");
    let long_name = "encoder-".repeat(31);
    let embedding_id = EmbeddingId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let mut input = embedding(text_vector(0), EmbeddingTarget::Evidence(evidence_id));
    input.id = embedding_id;
    input.produced_by.model_name = long_name.clone();
    let result = record(&fixture, input);
    let db = raw_db(&fixture.db_path, fixture.store);
    let rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND produced_by = $producer",
        &params(vec![
            ("id", Value::Uuid(embedding_id.as_uuid())),
            (
                "producer",
                Value::Json(serde_json::json!({
                    "system": "external",
                    "model_name": long_name,
                    "model_version": "1",
                    "pipeline_version": "1"
                })),
            ),
        ]),
    );

    assert!(
        result.is_ok() && rows == 1,
        "expected full producer model_name to persist in handle JSON, result={result:?}, rows={rows}"
    );
}

#[test]
fn producer_metadata_field_exceeding_max_rejected_at_record_observation() {
    for field in ["model_name", "model_version", "pipeline_version"] {
        let fixture = fixture(&format!("producer-too-long-{field}"));
        let evidence_id = EvidenceId::new_v7();
        let mut input = embedding(text_vector(0), EmbeddingTarget::Evidence(evidence_id));
        match field {
            "model_name" => input.produced_by.model_name = "x".repeat(4096),
            "model_version" => input.produced_by.model_version = "x".repeat(4096),
            "pipeline_version" => input.produced_by.pipeline_version = "x".repeat(4096),
            _ => unreachable!(),
        }
        let result = record(&fixture, input);
        let db = raw_db(&fixture.db_path, fixture.store);
        let handle_rows = count(
            &db,
            "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $target_id",
            &params(vec![("target_id", Value::Uuid(evidence_id.as_uuid()))]),
        );
        assert!(
            matches!(
                result,
                Err(CgError::ProducerFieldExceedsMaxLength {
                    field: got_field,
                    max: 255,
                    actual: 4096,
                }) if got_field == field
            ) && handle_rows == 0,
            "expected typed producer length error and rollback for {field}, got result={result:?}, handle_rows={handle_rows}"
        );
    }
}
