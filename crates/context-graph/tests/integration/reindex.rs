use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use chrono::Utc;
use context_graph::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use context_graph::{
    CgError, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId,
    DistanceMetric, EmbedderRegistration, EmbeddingId, EmbeddingSpace, EmbeddingSpaceBinding,
    EmbeddingTarget, EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef,
    IntentionOrigin, IntentionStatus, PrecomputedEmbedding, RecordObservation, ReindexOptions,
    RetentionStatus, Store, StoreConfig,
};
use contextdb_core::{TxId, Value, VectorIndexRef};
use contextdb_engine::Database;

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";
const IMAGE_SPACE: &str = "vigil_face_reid_v1";

struct Fixture {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
    store: Store,
    context_id: context_graph::ContextId,
    entity_id: context_graph::EntityId,
    decision_id: DecisionId,
    evidence_ids: Vec<EvidenceId>,
    source_texts: Vec<String>,
    old_vectors: Vec<Vec<f32>>,
    reindex_calls: Arc<Mutex<Vec<String>>>,
    embed_calls: Arc<Mutex<Vec<RecordedEmbedCall>>>,
    other_space_evidence_id: EvidenceId,
    other_space_vector: Vec<f32>,
}

#[derive(Clone, Debug)]
struct RecordedEmbedCall {
    text: String,
    thread_name: Option<String>,
}

#[derive(Clone)]
struct RecordingReindexEmbedder {
    calls: Arc<Mutex<Vec<String>>>,
    embed_calls: Arc<Mutex<Vec<RecordedEmbedCall>>>,
}

impl Embedder for RecordingReindexEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        match input {
            EmbeddingInput::Text(text) => {
                let thread_name = thread::current().name().map(ToString::to_string);
                self.calls
                    .lock()
                    .expect("recording calls")
                    .push(text.clone());
                self.embed_calls
                    .lock()
                    .expect("recording detailed calls")
                    .push(RecordedEmbedCall {
                        text: text.clone(),
                        thread_name,
                    });
                Ok(EmbeddingOutput {
                    embedding_space_id: TEXT_SPACE.to_string(),
                    dimension: 768,
                    producer: producer("new"),
                    vector: expected_reindexed_vector(&text),
                })
            }
            EmbeddingInput::ImageFrame { .. } | EmbeddingInput::ImageBytes(..) => {
                Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::ImageFrame,
                })
            }
            EmbeddingInput::AudioSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::AudioSegment,
            }),
            EmbeddingInput::VideoSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::VideoSegment,
            }),
            EmbeddingInput::StructuredSignal(_) => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::StructuredSignal,
            }),
        }
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        text_space()
    }
}

fn text_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: TEXT_SPACE.to_string(),
        dimension: 768,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "embeddinggemma-300m".to_string(),
        model_version: "default".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text, EvidenceKind::StructuredSignal],
        default_for_evidence_kind: Some(EvidenceKind::Text),
        bindings: vec![
            EmbeddingSpaceBinding {
                embedding_space_id: TEXT_SPACE.to_string(),
                table_name: "evidence".to_string(),
                vector_column: "vector_text".to_string(),
            },
            EmbeddingSpaceBinding {
                embedding_space_id: TEXT_SPACE.to_string(),
                table_name: "digests".to_string(),
                vector_column: "vector_text".to_string(),
            },
        ],
    }
}

fn image_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: IMAGE_SPACE.to_string(),
        dimension: 512,
        metric: DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "face-reid".to_string(),
        model_version: "v1".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
        default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: IMAGE_SPACE.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_vision".to_string(),
        }],
    }
}

fn producer(version: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: "edge-memory".to_string(),
        model_name: "embeddinggemma-300m".to_string(),
        model_version: version.to_string(),
        pipeline_version: "reindex".to_string(),
    }
}

fn vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[axis] = 1.0;
    vector
}

fn expected_reindexed_vector(source_text: &str) -> Vec<f32> {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in source_text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let primary = 100 + (hash as usize % 500);
    let secondary = 650 + ((hash >> 16) as usize % 100);
    let mut vector = vec![0.0; 768];
    vector[primary] = 0.8;
    vector[secondary] = 0.6;
    vector
}

fn vision_vector(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 512];
    vector[axis] = 1.0;
    vector
}

fn fixture(name: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join(format!("{name}.db"));
    let reindex_calls = Arc::new(Mutex::new(Vec::new()));
    let embed_calls = Arc::new(Mutex::new(Vec::new()));
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: db_path.clone(),
            extra_embedding_spaces: vec![image_space()],
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: text_space(),
            embedder: Arc::new(RecordingReindexEmbedder {
                calls: Arc::clone(&reindex_calls),
                embed_calls: Arc::clone(&embed_calls),
            }),
        }],
    )
    .expect("open");
    let context = store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "keep screen evidence searchable".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "screen".to_string(),
            properties: BTreeMap::from([("state".to_string(), serde_json::json!("known"))]),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "memory".to_string(),
            description: "remember screen".to_string(),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            context_id: context.id,
            ..Default::default()
        })
        .expect("decision");
    let mut evidence_ids = Vec::new();
    let nonce = uuid::Uuid::now_v7();
    let source_texts = vec![
        format!("screen alpha {nonce}"),
        format!("screen beta {nonce}"),
        format!("screen alpha {nonce}"),
    ];
    let mut old_vectors = Vec::new();
    for (idx, axis) in [0, 2, 4].into_iter().enumerate() {
        let evidence_id = EvidenceId::new_v7();
        let embedding_id = EmbeddingId::new_v7();
        let old_vector = vector(axis);
        let source_text = source_texts[idx].clone();
        let recorded = store.record_observation(RecordObservation {
            id: context_graph::ObservationId::new_v7(),
            entity_id: entity.id,
            context_id: context.id,
            observation_type: "screen-capture".to_string(),
            source: "edge".to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: evidence_id,
                kind: EvidenceKind::Text,
                source_ref: format!("webhook://screen/reindex/{idx}"),
                retention_status: RetentionStatus::RetainedExternal,
                producer: producer("capture"),
                text_projection: Some(source_text),
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: vec![PrecomputedEmbedding {
                id: embedding_id,
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: old_vector.clone(),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer("old"),
            }],
        });
        assert!(recorded.is_ok(), "reindex setup failed: {recorded:?}");
        evidence_ids.push(evidence_id);
        old_vectors.push(old_vector);
    }
    let other_space_evidence_id = EvidenceId::new_v7();
    let other_space_vector = vision_vector(0);
    let other_recorded = store.record_observation(RecordObservation {
        id: context_graph::ObservationId::new_v7(),
        entity_id: entity.id,
        context_id: context.id,
        observation_type: "face-reference".to_string(),
        source: "edge".to_string(),
        observed_at: Utc::now(),
        evidence: vec![EvidenceRef {
            id: other_space_evidence_id,
            kind: EvidenceKind::ImageFrame,
            source_ref: format!(
                "cas://sha256:{:064x}",
                other_space_evidence_id.as_uuid().as_u128()
            ),
            mime_type: Some("image/png".to_string()),
            retention_status: RetentionStatus::RetainedExternal,
            producer: producer("capture"),
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: vec![PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: IMAGE_SPACE.to_string(),
            vector: other_space_vector.clone(),
            target: EmbeddingTarget::Evidence(other_space_evidence_id),
            produced_by: producer("other-space"),
        }],
    });
    assert!(
        other_recorded.is_ok(),
        "other-space setup failed: {other_recorded:?}"
    );
    Fixture {
        _dir: dir,
        db_path,
        store,
        context_id: context.id,
        entity_id: entity.id,
        decision_id: decision.id,
        evidence_ids,
        source_texts,
        old_vectors,
        reindex_calls,
        embed_calls,
        other_space_evidence_id,
        other_space_vector,
    }
}

fn raw_db(path: &Path) -> Database {
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

fn single_i64(db: &Database, sql: &str, params: &HashMap<String, Value>) -> Option<i64> {
    db.execute(sql, params)
        .ok()
        .and_then(|result| result.rows.first().and_then(|row| row.first()).cloned())
        .and_then(|value| match value {
            Value::Int64(value) => Some(value),
            _ => None,
        })
}

fn single_txid(db: &Database, sql: &str, params: &HashMap<String, Value>) -> Option<TxId> {
    db.execute(sql, params)
        .ok()
        .and_then(|result| result.rows.first().and_then(|row| row.first()).cloned())
        .and_then(|value| match value {
            Value::TxId(value) => Some(value),
            _ => None,
        })
}

fn single_uuid(db: &Database, sql: &str, params: &HashMap<String, Value>) -> Option<uuid::Uuid> {
    db.execute(sql, params)
        .ok()
        .and_then(|result| result.rows.first().and_then(|row| row.first()).cloned())
        .and_then(|value| match value {
            Value::Uuid(value) => Some(value),
            _ => None,
        })
}

fn stored_evidence_vision_vector(db: &Database, evidence_id: EvidenceId) -> Option<Vec<f32>> {
    let result = db
        .execute(
            "SELECT vector_vision FROM evidence WHERE id = $id",
            &params(vec![("id", Value::Uuid(evidence_id.as_uuid()))]),
        )
        .expect("evidence vision vector query");
    result.rows.first().and_then(|row| match row.first() {
        Some(Value::Vector(vector)) => Some(vector.clone()),
        _ => None,
    })
}

fn hit_ids(db: &Database, query: Vec<f32>, top_k: usize) -> Vec<uuid::Uuid> {
    hit_scores(db, query, top_k)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

fn hit_scores(db: &Database, query: Vec<f32>, top_k: usize) -> Vec<(uuid::Uuid, f32)> {
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &query,
            top_k,
            None,
            db.snapshot(),
        )
        .expect("vector query");
    let rows = db.scan("evidence", db.snapshot()).expect("scan evidence");
    hits.into_iter()
        .filter_map(|(row_id, score)| {
            rows.iter().find_map(|row| {
                (row.row_id == row_id)
                    .then(|| row.values.get("id").cloned())
                    .flatten()
                    .and_then(|value| match value {
                        Value::Uuid(id) => Some((id, score)),
                        _ => None,
                    })
            })
        })
        .collect()
}

fn score_for(db: &Database, evidence_id: EvidenceId, query: Vec<f32>) -> Option<f32> {
    hit_scores(db, query, 32)
        .into_iter()
        .find_map(|(id, score)| (id == evidence_id.as_uuid()).then_some(score))
}

fn vector_index_matches_sources(
    db: &Database,
    evidence_ids: &[EvidenceId],
    source_texts: &[String],
) -> bool {
    evidence_ids.len() == source_texts.len()
        && evidence_ids
            .iter()
            .zip(source_texts)
            .all(|(evidence_id, text)| {
                score_for(db, *evidence_id, expected_reindexed_vector(text))
                    .is_some_and(|score| score > 0.99)
            })
}

fn old_vectors_no_longer_match(
    db: &Database,
    evidence_ids: &[EvidenceId],
    old_vectors: &[Vec<f32>],
) -> bool {
    evidence_ids.len() == old_vectors.len()
        && evidence_ids
            .iter()
            .zip(old_vectors)
            .all(|(evidence_id, old)| {
                score_for(db, *evidence_id, old.clone()).is_none_or(|score| score < 0.5)
            })
}

fn calls_include_all(calls: &Arc<Mutex<Vec<String>>>, expected_texts: &[String]) -> bool {
    let calls = calls.lock().expect("reindex calls");
    expected_texts.iter().all(|text| calls.contains(text))
}

fn writer_texts_embedded_only_by_writer_threads(
    calls: &Arc<Mutex<Vec<RecordedEmbedCall>>>,
    writer_texts: &[String],
) -> bool {
    let calls = calls.lock().expect("recording detailed calls");
    writer_texts.iter().all(|text| {
        let matching = calls
            .iter()
            .filter(|call| call.text == *text)
            .collect::<Vec<_>>();
        !matching.is_empty()
            && matching.iter().all(|call| {
                call.thread_name
                    .as_deref()
                    .is_some_and(|name| name.starts_with("concurrent-writer-"))
            })
    })
}

fn observation_created_txids(db: &Database, evidence_ids: &[EvidenceId]) -> Vec<Option<TxId>> {
    evidence_ids
        .iter()
        .map(|evidence_id| {
            let observation_id = single_uuid(
                db,
                "SELECT observation_id FROM evidence WHERE id = $evidence_id",
                &params(vec![("evidence_id", Value::Uuid(evidence_id.as_uuid()))]),
            )?;
            single_txid(
                db,
                "SELECT created_at FROM observations WHERE id = $observation_id",
                &params(vec![("observation_id", Value::Uuid(observation_id))]),
            )
        })
        .collect()
}

#[test]
fn reindex_updates_vectors_and_preserves_searchability() {
    let fixture = fixture("reindex-searchability");
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let db_path = fixture.db_path.clone();
    let evidence_ids = fixture.evidence_ids.clone();
    let old_vectors = fixture.old_vectors.clone();
    let source_texts = fixture.source_texts.clone();
    let reindex_calls = Arc::clone(&fixture.reindex_calls);
    drop(fixture.store);
    let db = raw_db(&db_path);
    let after_hits = evidence_ids
        .iter()
        .zip(source_texts.iter())
        .map(|(evidence_id, vector)| {
            let hits = hit_ids(&db, expected_reindexed_vector(vector), 3);
            (evidence_id.as_uuid(), hits)
        })
        .collect::<Vec<_>>();

    assert!(
        result.is_ok()
            && old_vectors_no_longer_match(&db, &evidence_ids, &old_vectors)
            && vector_index_matches_sources(&db, &evidence_ids, &source_texts)
            && calls_include_all(&reindex_calls, &source_texts)
            && after_hits
                .iter()
                .all(|(expected, hits)| hits.contains(expected)),
        "reindex must call the registered embedder on each row's original source text, replace the engine vector index with that output, and preserve searchability, result={result:?}, source_texts={source_texts:?}, reindex_calls={:?}, after_hits={after_hits:?}",
        reindex_calls.lock().expect("calls")
    );
}

#[test]
fn reindex_changes_stored_vectors() {
    let fixture = fixture("reindex-changes-vector");
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let db_path = fixture.db_path.clone();
    let evidence_ids = fixture.evidence_ids.clone();
    let old_vectors = fixture.old_vectors.clone();
    let source_texts = fixture.source_texts.clone();
    drop(fixture.store);
    let db_after = raw_db(&db_path);
    let duplicate_rows_match_same_source = score_for(
        &db_after,
        evidence_ids[0],
        expected_reindexed_vector(&source_texts[0]),
    )
    .is_some_and(|score| score > 0.99)
        && score_for(
            &db_after,
            evidence_ids[2],
            expected_reindexed_vector(&source_texts[2]),
        )
        .is_some_and(|score| score > 0.99);
    let distinct_row_matches_own_source = score_for(
        &db_after,
        evidence_ids[1],
        expected_reindexed_vector(&source_texts[1]),
    )
    .is_some_and(|score| score > 0.99);
    let new_producer_rows = count(
        &db_after,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE produced_by = $producer",
        &params(vec![(
            "producer",
            Value::Json(serde_json::to_value(producer("new")).expect("json")),
        )]),
    );
    let old_producer_rows = count(
        &db_after,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE produced_by = $producer",
        &params(vec![(
            "producer",
            Value::Json(serde_json::to_value(producer("old")).expect("json")),
        )]),
    );

    assert!(
        result.is_ok()
            && old_vectors_no_longer_match(&db_after, &evidence_ids, &old_vectors)
            && vector_index_matches_sources(&db_after, &evidence_ids, &source_texts)
            && duplicate_rows_match_same_source
            && distinct_row_matches_own_source
            && new_producer_rows == evidence_ids.len() as i64
            && old_producer_rows == evidence_ids.len() as i64,
        "engine vector index should be replaced with the registered embedder output for source content, duplicate source content should search against the same expected vector, and producer lineage should be preserved, result={result:?}, source_texts={source_texts:?}, duplicate_rows_match_same_source={duplicate_rows_match_same_source}, distinct_row_matches_own_source={distinct_row_matches_own_source}, new_producer_rows={new_producer_rows}, old_producer_rows={old_producer_rows}"
    );
}

#[test]
fn reindex_scoped_by_embedding_space() {
    let fixture = fixture("reindex-scope");
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let db_path = fixture.db_path.clone();
    let other_space_evidence_id = fixture.other_space_evidence_id;
    let other_space_before = fixture.other_space_vector.clone();
    drop(fixture.store);
    let db = raw_db(&db_path);
    let other_space_after = stored_evidence_vision_vector(&db, other_space_evidence_id);
    let text_cursor = count(
        &db,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space",
        &params(vec![("space", Value::Text(TEXT_SPACE.to_string()))]),
    );
    let other_cursor = count(
        &db,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space",
        &params(vec![(
            "space",
            Value::Text("vigil_face_reid_v1".to_string()),
        )]),
    );

    assert!(
        result.is_ok()
            && text_cursor == 1
            && other_cursor == 0
            && other_space_after.as_ref() == Some(&other_space_before),
        "reindex cursor and vector mutation must be scoped to the requested embedding space, result={result:?}, text_cursor={text_cursor}, other_cursor={other_cursor}, other_space_after={other_space_after:?}"
    );
}

#[test]
fn reindex_does_not_trigger_impact_analysis_invalidation_cascade() {
    let fixture = fixture("reindex-no-invalidation");
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let invalidations =
        fixture
            .store
            .list_invalidations(fixture.context_id, Some(fixture.decision_id), None);
    let db_path = fixture.db_path.clone();
    drop(fixture.store);
    let db = raw_db(&db_path);
    let cursor_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space",
        &params(vec![("space", Value::Text(TEXT_SPACE.to_string()))]),
    );

    assert!(
        result.is_ok() && invalidations.as_ref().is_ok_and(Vec::is_empty) && cursor_rows == 1,
        "reindex vector replacement must bypass observation-trigger invalidation cascade while recording completion, result={result:?}, invalidations={invalidations:?}, cursor_rows={cursor_rows}"
    );
}

#[test]
fn reindex_preserves_pre_reindex_producer_lineage_via_original_write_txid() {
    let fixture = fixture("reindex-provenance");
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let db_path = fixture.db_path.clone();
    let evidence_ids = fixture.evidence_ids.clone();
    drop(fixture.store);
    let db = raw_db(&db_path);
    let old_producer_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE produced_by = $producer",
        &params(vec![(
            "producer",
            Value::Json(serde_json::to_value(producer("old")).expect("json")),
        )]),
    );
    let cursor_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space AND from_producer = $old_producer AND to_producer = $new_producer",
        &params(vec![
            ("space", Value::Text(TEXT_SPACE.to_string())),
            (
                "old_producer",
                Value::Json(serde_json::to_value(producer("old")).expect("json")),
            ),
            (
                "new_producer",
                Value::Json(serde_json::to_value(producer("new")).expect("json")),
            ),
        ]),
    );

    assert!(
        result.is_ok() && old_producer_rows == evidence_ids.len() as i64 && cursor_rows == 1,
        "reindex must preserve original producer lineage and record old/new producer cursor metadata, result={result:?}, old_producer_rows={old_producer_rows}, cursor_rows={cursor_rows}"
    );
}

#[test]
fn reindex_resumable_after_interruption_picks_up_at_last_committed_cursor() {
    let fixture = fixture("reindex-resume");
    let interrupted = fixture.store.reindex_space(
        TEXT_SPACE,
        ReindexOptions {
            interrupt_after: Some(1),
        },
    );
    let resumed = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let db_path = fixture.db_path.clone();
    drop(fixture.store);
    let db = raw_db(&db_path);
    let cursor_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space AND last_committed_cursor >= 1",
        &params(vec![("space", Value::Text(TEXT_SPACE.to_string()))]),
    );

    assert!(
        matches!(
            interrupted,
            Err(CgError::ReindexPaused {
                ref embedding_space_id,
                last_committed_cursor: 1,
            }) if embedding_space_id == TEXT_SPACE
        ) && resumed.is_ok()
            && cursor_rows == 1,
        "interrupted reindex must persist cursor and resume from it, interrupted={interrupted:?}, resumed={resumed:?}, cursor_rows={cursor_rows}"
    );
}

#[test]
fn reindex_concurrent_writes_during_reindex_window_land_in_target_space_deterministically() {
    let fixture = fixture("reindex-concurrent-write");
    let expected_scan_rows = fixture.evidence_ids.len() + 96;
    for idx in 0..96 {
        let evidence_id = EvidenceId::new_v7();
        let seeded = fixture.store.record_observation(RecordObservation {
            id: context_graph::ObservationId::new_v7(),
            entity_id: fixture.entity_id,
            context_id: fixture.context_id,
            observation_type: "screen-capture-reindex-padding".to_string(),
            source: format!("edge-padding-{idx}"),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: evidence_id,
                kind: EvidenceKind::Text,
                source_ref: format!("cas://sha256:{:064x}", evidence_id.as_uuid().as_u128()),
                retention_status: RetentionStatus::RetainedExternal,
                producer: producer("capture"),
                text_projection: Some(format!("padding text {idx}")),
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: vec![PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: vector(idx % 3),
                target: EmbeddingTarget::Evidence(evidence_id),
                produced_by: producer("old"),
            }],
        });
        assert!(
            seeded.is_ok(),
            "reindex concurrent padding seed failed: {seeded:?}"
        );
    }
    let writer_count = 24;
    let writer_nonce = uuid::Uuid::now_v7();
    let new_evidence_ids = (0..writer_count)
        .map(|_| EvidenceId::new_v7())
        .collect::<Vec<_>>();
    let writer_source_texts = (0..writer_count)
        .map(|idx| format!("writer text {writer_nonce} {idx}"))
        .collect::<Vec<_>>();
    let barrier = Arc::new(Barrier::new(writer_count + 2));
    let reindex_started = Arc::new(AtomicBool::new(false));
    let reindex_done = Arc::new(AtomicBool::new(false));
    let writers_entered_before_reindex_done = Arc::new(AtomicUsize::new(0));
    let entered_before_done = Arc::new(Mutex::new(Vec::<EvidenceId>::new()));
    let reindex_store = fixture.store.clone();
    let reindex_barrier = Arc::clone(&barrier);
    let reindex_started_flag = Arc::clone(&reindex_started);
    let reindex_done_flag = Arc::clone(&reindex_done);
    let reindex_thread = thread::Builder::new()
        .name("bounded-reindex-scan".to_string())
        .spawn(move || {
            reindex_barrier.wait();
            reindex_started_flag.store(true, Ordering::SeqCst);
            let result = reindex_store.reindex_space(TEXT_SPACE, ReindexOptions::default());
            reindex_done_flag.store(true, Ordering::SeqCst);
            result
        })
        .expect("spawn reindex thread");
    let writer_threads = new_evidence_ids
        .iter()
        .zip(writer_source_texts.iter())
        .enumerate()
        .map(|(idx, (evidence_id, source_text))| {
            let writer = fixture.store.clone();
            let barrier = Arc::clone(&barrier);
            let reindex_calls = Arc::clone(&fixture.reindex_calls);
            let reindex_started = Arc::clone(&reindex_started);
            let reindex_done = Arc::clone(&reindex_done);
            let writers_entered_before_reindex_done =
                Arc::clone(&writers_entered_before_reindex_done);
            let entered_before_done = Arc::clone(&entered_before_done);
            let context_id = fixture.context_id;
            let entity_id = fixture.entity_id;
            let evidence_id = *evidence_id;
            let source_text = source_text.clone();
            thread::Builder::new()
                .name(format!("concurrent-writer-{idx}"))
                .spawn(move || {
                    barrier.wait();
                    while !reindex_started.load(Ordering::SeqCst) {
                        thread::yield_now();
                    }
                    while !reindex_done.load(Ordering::SeqCst)
                        && reindex_calls.lock().expect("reindex calls").is_empty()
                    {
                        thread::yield_now();
                    }
                    if !reindex_done.load(Ordering::SeqCst) {
                        writers_entered_before_reindex_done.fetch_add(1, Ordering::SeqCst);
                        entered_before_done
                            .lock()
                            .expect("entered ids")
                            .push(evidence_id);
                    }
                    writer.record_observation(RecordObservation {
                        id: context_graph::ObservationId::new_v7(),
                        entity_id,
                        context_id,
                        observation_type: "screen-capture".to_string(),
                        source: format!("edge-concurrent-{idx}"),
                        observed_at: Utc::now(),
                        evidence: vec![EvidenceRef {
                            id: evidence_id,
                            kind: EvidenceKind::Text,
                            source_ref: format!("webhook://screen/concurrent/{idx}"),
                            retention_status: RetentionStatus::RetainedExternal,
                            producer: producer("capture"),
                            text_projection: Some(source_text),
                            ..Default::default()
                        }],
                        observed_properties: BTreeMap::new(),
                        state_delta: BTreeMap::new(),
                        properties: BTreeMap::new(),
                        embeddings: Vec::new(),
                    })
                })
                .expect("spawn writer thread")
        })
        .collect::<Vec<_>>();
    barrier.wait();
    let reindex_result = reindex_thread.join().expect("reindex thread");
    let entered_before_reindex_done = writers_entered_before_reindex_done.load(Ordering::SeqCst);
    let writer_results = writer_threads
        .into_iter()
        .map(|handle| handle.join().expect("writer thread"))
        .collect::<Vec<_>>();
    let db_path = fixture.db_path.clone();
    let reindex_calls = Arc::clone(&fixture.reindex_calls);
    let embed_calls = Arc::clone(&fixture.embed_calls);
    drop(fixture.store);
    let db = raw_db(&db_path);
    let cursor_rows = count(
        &db,
        "SELECT COUNT(*) AS n FROM reindex_cursors WHERE embedding_space_id = $space",
        &params(vec![("space", Value::Text(TEXT_SPACE.to_string()))]),
    );
    let cursor_last_committed = single_i64(
        &db,
        "SELECT last_committed_cursor FROM reindex_cursors WHERE embedding_space_id = $space",
        &params(vec![("space", Value::Text(TEXT_SPACE.to_string()))]),
    );
    let scan_started_at = single_txid(
        &db,
        "SELECT started_at FROM reindex_cursors WHERE embedding_space_id = $space",
        &params(vec![("space", Value::Text(TEXT_SPACE.to_string()))]),
    );
    let entered_ids = entered_before_done.lock().expect("entered ids").clone();
    let entered_texts = entered_ids
        .iter()
        .filter_map(|evidence_id| {
            new_evidence_ids
                .iter()
                .position(|id| id == evidence_id)
                .map(|idx| writer_source_texts[idx].clone())
        })
        .collect::<Vec<_>>();
    let entered_created_txids = observation_created_txids(&db, &entered_ids);
    let entered_after_scan_start = scan_started_at.is_some_and(|started_at| {
        entered_created_txids
            .iter()
            .all(|created_at| created_at.is_some_and(|created_at| created_at > started_at))
    });
    let new_producer_rows = entered_ids
        .iter()
        .map(|evidence_id| {
            count(
                &db,
                "SELECT COUNT(*) AS n FROM evidence_embedding_handles WHERE target_id = $evidence_id AND produced_by = $producer",
                &params(vec![
                    ("evidence_id", Value::Uuid(evidence_id.as_uuid())),
                    (
                        "producer",
                        Value::Json(serde_json::to_value(producer("new")).expect("json")),
                    ),
                ]),
            )
        })
        .sum::<i64>();

    assert!(
        reindex_result.is_ok()
            && writer_results.iter().all(Result::is_ok)
            && entered_before_reindex_done > 0
            && cursor_rows == 1
            && cursor_last_committed == Some(expected_scan_rows as i64)
            && entered_after_scan_start
            && vector_index_matches_sources(&db, &entered_ids, &entered_texts)
            && writer_texts_embedded_only_by_writer_threads(&embed_calls, &entered_texts)
            && new_producer_rows == entered_ids.len() as i64,
        "writes that entered before reindex completed must commit after the scan-start cursor, be embedded on the writer path from their original source text, and never be embedded by the bounded reindex scan or a reindex worker, reindex_result={reindex_result:?}, writer_results={writer_results:?}, entered_before_reindex_done={entered_before_reindex_done}, expected_scan_rows={expected_scan_rows}, cursor_rows={cursor_rows}, cursor_last_committed={cursor_last_committed:?}, scan_started_at={scan_started_at:?}, entered_created_txids={entered_created_txids:?}, entered_after_scan_start={entered_after_scan_start}, entered_ids={entered_ids:?}, entered_texts={entered_texts:?}, reindex_calls={:?}, embed_calls={:?}, new_producer_rows={new_producer_rows}",
        reindex_calls.lock().expect("calls"),
        embed_calls.lock().expect("detailed calls")
    );
}
