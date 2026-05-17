use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::{
    CgError, CreateContext, CreateEntity, EmbeddingId, EmbeddingTarget, EntityType, EvidenceId,
    EvidenceKind, EvidenceProducer, EvidenceRef, PrecomputedEmbedding, RecordObservation,
    RetentionStatus, Store, StoreConfig,
};
use contextdb_core::{RowId, Value, VectorIndexRef};
use contextdb_engine::Database;

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    store: Store,
    context_id: context_graph::ContextId,
    entity_id: context_graph::EntityId,
}

fn producer(system: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: system.to_string(),
        model_name: "external-encoder".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("precomputed.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let context = store
        .create_context(CreateContext {
            name: format!("precomputed-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "camera-event".to_string(),
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

fn observation(
    fixture: &Fixture,
    evidence_id: EvidenceId,
    embedding_id: EmbeddingId,
    vector: Vec<f32>,
    space: &str,
) -> RecordObservation {
    RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: fixture.entity_id,
        context_id: fixture.context_id,
        observation_type: "camera-frame".to_string(),
        source: "edge".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: evidence_id,
            kind: EvidenceKind::Text,
            source_ref:
                "cas://sha256:1111111111111111111111111111111111111111111111111111111111111111"
                    .to_string(),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer("capture-pipeline"),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: embedding_id,
            embedding_space_id: space.to_string(),
            vector,
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer("external"),
        }],
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

fn hit_contains_id(db: &Database, table: &str, hits: &[(RowId, f32)], id: uuid::Uuid) -> bool {
    let rows = db.scan(table, db.snapshot()).expect("scan table");
    hits.iter().any(|(row_id, _)| {
        rows.iter()
            .find(|row| row.row_id == *row_id)
            .is_some_and(|row| row.values.get("id") == Some(&Value::Uuid(id)))
    })
}

#[test]
fn ingest_observation_with_precomputed_embedding() {
    let fixture = fixture();
    let evidence_id = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let vector = text_vector(0);
    let result = fixture.store.record_observation(observation(
        &fixture,
        evidence_id,
        embedding_id,
        vector.clone(),
        TEXT_SPACE,
    ));
    let db = raw_db(&fixture.db_path, fixture.store);

    let handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND target_id = $target_id AND embedding_space_id = $space",
        &params(vec![
            ("id", Value::Uuid(embedding_id.as_uuid())),
            ("target_id", Value::Uuid(evidence_id.as_uuid())),
            ("space", Value::Text(TEXT_SPACE.to_string())),
        ]),
    );
    let producer_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $id AND produced_by = $producer",
        &params(vec![
            ("id", Value::Uuid(embedding_id.as_uuid())),
            (
                "producer",
                Value::Json(serde_json::to_value(producer("external")).expect("producer json")),
            ),
        ]),
    );
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &vector,
            1,
            None,
            db.snapshot(),
        )
        .expect("vector query");

    assert!(
        result
            .as_ref()
            .is_ok_and(|output| output.embedding_ids == vec![embedding_id])
            && handle_rows == 1
            && producer_rows == 1
            && hit_contains_id(&db, "evidence", &hits, evidence_id.as_uuid()),
        "expected precomputed evidence embedding to write a variant handle, producer JSON, and queryable vector; result={result:?}, handle_rows={handle_rows}, producer_rows={producer_rows}, hits={hits:?}"
    );
}

#[test]
fn ingest_observation_rejects_dimension_mismatch_for_known_space() {
    let fixture = fixture();
    let evidence_id = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let result = fixture.store.record_observation(observation(
        &fixture,
        evidence_id,
        embedding_id,
        vec![1.0; 767],
        TEXT_SPACE,
    ));
    let db = raw_db(&fixture.db_path, fixture.store);
    let evidence_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $id",
        &params(vec![("id", Value::Uuid(evidence_id.as_uuid()))]),
    );

    assert!(
        matches!(
            result,
            Err(CgError::EmbeddingDimensionMismatch {
                ref space_id,
                expected: 768,
                found: 767,
            }) if space_id == TEXT_SPACE
        ) && evidence_rows == 0,
        "expected typed dimension mismatch before evidence write, result={result:?}, evidence_rows={evidence_rows}"
    );
}

#[test]
fn precomputed_embedding_dimension_validated_at_record_observation_before_db_write() {
    let fixture = fixture();
    let before_observations = 0;

    let evidence_id = EvidenceId::new_v7();
    let result = fixture.store.record_observation(observation(
        &fixture,
        evidence_id,
        EmbeddingId::new_v7(),
        vec![1.0; 767],
        TEXT_SPACE,
    ));
    let db = raw_db(&fixture.db_path, fixture.store);
    let after_observations = count(
        &db,
        "SELECT COUNT(*) AS n FROM observations",
        &HashMap::new(),
    );
    let leaked_vectors = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &text_vector(0),
            10,
            None,
            db.snapshot(),
        )
        .expect("vector query");

    assert!(
        matches!(result, Err(CgError::EmbeddingDimensionMismatch { .. }))
            && after_observations == before_observations
            && leaked_vectors.is_empty(),
        "dimension validation must happen before observation/evidence/vector writes, result={result:?}, before={before_observations}, after={after_observations}, leaked_vectors={leaked_vectors:?}"
    );
}

#[test]
fn precomputed_embedding_unknown_space_rejected_at_record_observation() {
    let fixture = fixture();
    let result = fixture.store.record_observation(observation(
        &fixture,
        EvidenceId::new_v7(),
        EmbeddingId::new_v7(),
        vec![1.0; 32],
        "unknown_space",
    ));

    assert!(
        matches!(result, Err(CgError::UnknownEmbeddingSpace { ref id }) if id == "unknown_space"),
        "expected unknown-space typed error, got {result:?}"
    );
}

#[test]
fn precomputed_embedding_atomic_with_observation_partial_failure_rolls_back_all() {
    let fixture = fixture();
    let good_evidence = EvidenceId::new_v7();
    let bad_evidence = EvidenceId::new_v7();
    let good_embedding = EmbeddingId::new_v7();
    let bad_embedding = EmbeddingId::new_v7();
    let mut input = observation(
        &fixture,
        good_evidence,
        good_embedding,
        text_vector(1),
        TEXT_SPACE,
    );
    input.evidence.push(EvidenceRef {
        id: bad_evidence,
        kind: EvidenceKind::Text,
        source_ref: "cas://sha256:2222222222222222222222222222222222222222222222222222222222222222"
            .to_string(),
        retention_status: RetentionStatus::RetainedExternal,
        ..Default::default()
    });
    input.embeddings.push(PrecomputedEmbedding {
        id: bad_embedding,
        embedding_space_id: TEXT_SPACE.to_string(),
        vector: vec![1.0; 767],
        target: EmbeddingTarget::Evidence(bad_evidence),
        produced_by: producer("external"),
    });

    let result = fixture.store.record_observation(input);
    let db = raw_db(&fixture.db_path, fixture.store);
    let evidence_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $good OR id = $bad",
        &params(vec![
            ("good", Value::Uuid(good_evidence.as_uuid())),
            ("bad", Value::Uuid(bad_evidence.as_uuid())),
        ]),
    );
    let handle_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE id = $good OR id = $bad",
        &params(vec![
            ("good", Value::Uuid(good_embedding.as_uuid())),
            ("bad", Value::Uuid(bad_embedding.as_uuid())),
        ]),
    );
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &text_vector(1),
            10,
            None,
            db.snapshot(),
        )
        .expect("vector query");

    assert!(
        matches!(result, Err(CgError::Rollback { .. }))
            && evidence_rows == 0
            && handle_rows == 0
            && !hit_contains_id(&db, "evidence", &hits, good_evidence.as_uuid()),
        "partial embedding failure must roll back evidence, handle, and vector writes; result={result:?}, evidence_rows={evidence_rows}, handle_rows={handle_rows}, hits={hits:?}"
    );
}
