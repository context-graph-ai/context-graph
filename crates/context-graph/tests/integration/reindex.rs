use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use chrono::Utc;
use context_graph::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use context_graph::{
    CgError, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId,
    DistanceMetric, EmbedderRegistration, EmbeddingId, EmbeddingRole, EmbeddingSpace,
    EmbeddingSpaceBinding, EmbeddingTarget, EntityType, EvidenceId, EvidenceKind, EvidenceProducer,
    EvidenceRef, IntentionOrigin, IntentionStatus, ModelIdentity, OldSpace, PrecomputedEmbedding,
    RecordObservation, ReindexOptions, RetentionStatus, SpaceState, Store, StoreConfig,
    SwitchResult, VectorPlacement,
};
use contextdb_core::{TxId, Value, VectorIndexRef};
use contextdb_engine::Database;

/// The space reindex fills: the new text model's own space.
const TEXT_SPACE: &str = "reindex_test_embeddinggemma_300m_new_768";
/// The space the evidence was embedded in before the text role switched models.
const OLD_TEXT_SPACE: &str = "reindex_test_embeddinggemma_300m_old_768";
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

impl Fixture {
    /// Switch the text role to the new model; the old space turns stale and reindex fills the new one.
    fn switch_to_new_model(mut self) -> (Self, SwitchResult) {
        let switched = self
            .store
            .switch_embedding_space(EmbeddingRole::EvidenceText, text_space(), OldSpace::Reindex)
            .expect("switch the text role to the new model");
        // The switch records both generations before a reopen registers their models together.
        // Their declared defaults stay immutable; no registration order chooses the active one.
        drop(self.store);
        self.store = Store::open_with_embedder_registrations(
            StoreConfig {
                db_path: self.db_path.clone(),
                extra_embedding_spaces: vec![image_space()],
                ..Default::default()
            },
            vec![
                EmbedderRegistration {
                    space: OldTextEmbedder.embedding_space(),
                    embedder: Arc::new(OldTextEmbedder),
                },
                EmbedderRegistration {
                    space: text_space(),
                    embedder: Arc::new(RecordingReindexEmbedder {
                        calls: Arc::clone(&self.reindex_calls),
                        embed_calls: Arc::clone(&self.embed_calls),
                    }),
                },
            ],
        )
        .expect("reopen with both models after the text role's recorded switch");
        (self, switched)
    }
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
                    producer: EvidenceProducer::from(&new_identity()),
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

    fn producer(&self) -> ModelIdentity {
        new_identity()
    }
}

fn new_identity() -> ModelIdentity {
    ModelIdentity {
        model: "embeddinggemma-300m".to_string(),
        weights_revision: "new".to_string(),
        preparation: "reindex".to_string(),
        dimension: 768,
    }
}

fn old_identity() -> ModelIdentity {
    ModelIdentity {
        model: "embeddinggemma-300m".to_string(),
        weights_revision: "old".to_string(),
        preparation: "reindex".to_string(),
        dimension: 768,
    }
}

/// The text model the evidence was embedded with before the switch.
struct OldTextEmbedder;

impl Embedder for OldTextEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        match input {
            EmbeddingInput::Text(text) => Ok(EmbeddingOutput {
                embedding_space_id: OLD_TEXT_SPACE.to_string(),
                dimension: 768,
                producer: EvidenceProducer::from(&old_identity()),
                vector: old_vector_for(&text),
            }),
            _ => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::ImageFrame,
            }),
        }
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: OLD_TEXT_SPACE.to_string(),
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: OLD_TEXT_SPACE.to_string(),
                table_name: "evidence".to_string(),
                vector_column: "vector_text".to_string(),
            }],
            model_version: "old".to_string(),
            producer: old_identity(),
            ..text_space()
        }
    }

    fn producer(&self) -> ModelIdentity {
        old_identity()
    }
}

fn text_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: TEXT_SPACE.to_string(),
        dimension: 768,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: "embeddinggemma-300m".to_string(),
        model_version: "new".to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text],
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
        placement: Some(VectorPlacement::NodeLocal),
        media_reachable_on_every_node: false,
        producer: new_identity(),
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
        placement: Some(VectorPlacement::TravelsWithItem),
        media_reachable_on_every_node: false,
        producer: image_identity(),
    }
}

fn image_identity() -> ModelIdentity {
    ModelIdentity {
        model: "face-reid".to_string(),
        weights_revision: "v1".to_string(),
        preparation: "face".to_string(),
        dimension: 512,
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

/// The old model's vector for a source text: a different direction from the new model's.
fn old_vector_for(source_text: &str) -> Vec<f32> {
    let hash = source_text.bytes().fold(7_usize, |hash, byte| {
        hash.wrapping_mul(31).wrapping_add(byte as usize)
    });
    vector(hash % 90)
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
            space: OldTextEmbedder.embedding_space(),
            embedder: Arc::new(OldTextEmbedder),
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
    for idx in 0..source_texts.len() {
        let evidence_id = EvidenceId::new_v7();
        // Embedded at write by the old model, the text role's active space.
        let old_vector = old_vector_for(&source_texts[idx]);
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
            embeddings: Vec::new(),
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
            produced_by: EvidenceProducer {
                system: "edge-memory".to_string(),
                ..EvidenceProducer::from(&image_identity())
            },
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

/// The first column of the first row, or `None` when the query returns no row. A failed query
/// or a value of another type fails the test instead of reading as an absent row.
fn single_value(db: &Database, sql: &str, params: &HashMap<String, Value>) -> Option<Value> {
    let result = db
        .execute(sql, params)
        .unwrap_or_else(|error| panic!("readback query failed: {sql}: {error:?}"));
    result.rows.first().map(|row| {
        row.first()
            .cloned()
            .unwrap_or_else(|| panic!("readback row has no column: {sql}"))
    })
}

fn single_txid(db: &Database, sql: &str, params: &HashMap<String, Value>) -> Option<TxId> {
    single_value(db, sql, params).map(|value| match value {
        Value::TxId(value) => value,
        other => panic!("expected a transaction id from {sql}, got {other:?}"),
    })
}

fn single_uuid(db: &Database, sql: &str, params: &HashMap<String, Value>) -> Option<uuid::Uuid> {
    single_value(db, sql, params).map(|value| match value {
        Value::Uuid(value) => value,
        other => panic!("expected a UUID from {sql}, got {other:?}"),
    })
}

fn stored_evidence_vision_vector(db: &Database, evidence_id: EvidenceId) -> Option<Vec<f32>> {
    let result = db
        .execute(
            "SELECT vector_vision FROM evidence_embeddings_travelling WHERE target_id = $id",
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
            VectorIndexRef::new("evidence_embeddings", "vector_text"),
            &query,
            top_k,
            None,
            db.snapshot(),
        )
        .expect("vector query");
    let rows = db
        .scan("evidence_embeddings", db.snapshot())
        .expect("scan node-local evidence vectors");
    hits.into_iter()
        .filter_map(|(row_id, score)| {
            rows.iter().find_map(|row| {
                (row.row_id == row_id)
                    .then(|| row.values.get("target_id").cloned())
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
                let expected = expected_reindexed_vector(text);
                let stored = db
                    .execute(
                        "SELECT vector_text FROM evidence_embeddings WHERE target_id = $id AND embedding_space_id = $space",
                        &params(vec![
                            ("id", Value::Uuid(evidence_id.as_uuid())),
                            ("space", Value::Text(TEXT_SPACE.to_string())),
                        ]),
                    )
                    .expect("read the new model's exact stored vector");
                stored.rows == vec![vec![Value::Vector(expected.clone())]]
                    && score_for(db, *evidence_id, expected)
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

/// Node-local evidence vector rows of one space produced by one model.
fn produced_rows(db: &Database, space: &str, identity: &ModelIdentity) -> i64 {
    count(
        db,
        "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE embedding_space_id = $space AND produced_by = $producer",
        &params(vec![
            ("space", Value::Text(space.to_string())),
            (
                "producer",
                Value::Json(serde_json::to_value(EvidenceProducer::from(identity)).expect("json")),
            ),
        ]),
    )
}

fn calls_include_all(calls: &Arc<Mutex<Vec<String>>>, expected_texts: &[String]) -> bool {
    let calls = calls.lock().expect("reindex calls");
    calls.len() == expected_texts.len()
        && expected_texts.iter().all(|text| {
            calls.iter().filter(|call| *call == text).count()
                == expected_texts
                    .iter()
                    .filter(|expected| *expected == text)
                    .count()
        })
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
    let (fixture, _) = fixture.switch_to_new_model();
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
        "reindex must call the new model on each row's original source text, fill the new space's vector index with that output, and preserve searchability, result={result:?}, source_texts={source_texts:?}, reindex_calls={:?}, after_hits={after_hits:?}",
        reindex_calls.lock().expect("calls")
    );
}

#[test]
fn reindex_changes_stored_vectors() {
    let fixture = fixture("reindex-changes-vector");
    let (fixture, _) = fixture.switch_to_new_model();
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let old_space_state = fixture
        .store
        .embedding_space_status()
        .expect("space status")
        .into_iter()
        .find(|status| status.space_id == OLD_TEXT_SPACE)
        .map(|status| status.state);
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
    let new_producer_rows = produced_rows(&db_after, TEXT_SPACE, &new_identity());
    let old_producer_rows = produced_rows(&db_after, OLD_TEXT_SPACE, &old_identity());

    assert!(
        result.is_ok()
            && old_vectors_no_longer_match(&db_after, &evidence_ids, &old_vectors)
            && vector_index_matches_sources(&db_after, &evidence_ids, &source_texts)
            && duplicate_rows_match_same_source
            && distinct_row_matches_own_source
            && new_producer_rows == evidence_ids.len() as i64
            && old_producer_rows == 0
            && old_space_state == Some(SpaceState::Stale),
        "the new space's vector index should hold the new model's output for each source, duplicate source content should search against the same expected vector, and each stale old-model row is replaced in the batch that writes its new row, result={result:?}, source_texts={source_texts:?}, duplicate_rows_match_same_source={duplicate_rows_match_same_source}, distinct_row_matches_own_source={distinct_row_matches_own_source}, new_producer_rows={new_producer_rows}, old_producer_rows={old_producer_rows}, old_space_state={old_space_state:?}"
    );
}

#[test]
fn reindex_scoped_by_embedding_space() {
    let fixture = fixture("reindex-scope");
    let (fixture, _) = fixture.switch_to_new_model();
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let statuses = fixture
        .store
        .embedding_space_status()
        .expect("space status");
    let db_path = fixture.db_path.clone();
    let other_space_evidence_id = fixture.other_space_evidence_id;
    let other_space_before = fixture.other_space_vector.clone();
    let text_rows = fixture.evidence_ids.len() as u64;
    drop(fixture.store);
    let db = raw_db(&db_path);
    let other_space_after = stored_evidence_vision_vector(&db, other_space_evidence_id);
    let text_embedded = statuses
        .iter()
        .find(|status| status.space_id == TEXT_SPACE)
        .map(|status| status.embedded);
    let other_embedded = statuses
        .iter()
        .find(|status| status.space_id == IMAGE_SPACE)
        .map(|status| status.embedded);

    assert!(
        result
            .as_ref()
            .is_ok_and(|report| report.embedded == text_rows)
            && text_embedded == Some(text_rows)
            && other_embedded == Some(1)
            && other_space_after.as_ref() == Some(&other_space_before),
        "reindex work and vector mutation must be scoped to the requested embedding space, result={result:?}, text_embedded={text_embedded:?}, other_embedded={other_embedded:?}, other_space_after={other_space_after:?}"
    );
}

#[test]
fn reindex_does_not_trigger_impact_analysis_invalidation_cascade() {
    let fixture = fixture("reindex-no-invalidation");
    let (fixture, _) = fixture.switch_to_new_model();
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let invalidations =
        fixture
            .store
            .list_invalidations(fixture.context_id, Some(fixture.decision_id), None);

    assert!(
        result
            .as_ref()
            .is_ok_and(|report| report.remaining == 0 && report.embedded > 0)
            && invalidations.as_ref().is_ok_and(Vec::is_empty),
        "reindex vector replacement must bypass observation-trigger invalidation cascade while completing its work, result={result:?}, invalidations={invalidations:?}"
    );
}

#[test]
fn reindex_preserves_pre_reindex_producer_lineage_via_original_write_txid() {
    let fixture = fixture("reindex-provenance");
    let db_path = fixture.db_path.clone();
    let evidence_ids = fixture.evidence_ids.clone();
    let written_before = {
        let db = fixture.store.sync_database();
        evidence_ids
            .iter()
            .map(|evidence_id| {
                single_value(
                    &db,
                    "SELECT producer FROM evidence WHERE id = $id",
                    &params(vec![("id", Value::Uuid(evidence_id.as_uuid()))]),
                )
            })
            .collect::<Vec<_>>()
    };
    let created_txids_before =
        observation_created_txids(&fixture.store.sync_database(), &evidence_ids);
    let (fixture, switched) = fixture.switch_to_new_model();
    let result = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    drop(fixture.store);
    let db = raw_db(&db_path);
    let created_txids = observation_created_txids(&db, &evidence_ids);
    let written_after = evidence_ids
        .iter()
        .map(|evidence_id| {
            single_value(
                &db,
                "SELECT producer FROM evidence WHERE id = $id",
                &params(vec![("id", Value::Uuid(evidence_id.as_uuid()))]),
            )
        })
        .collect::<Vec<_>>();
    let new_producer_rows = produced_rows(&db, TEXT_SPACE, &new_identity());

    assert!(
        result.is_ok()
            && written_after == written_before
            && created_txids == created_txids_before
            && created_txids.iter().all(Option::is_some)
            && new_producer_rows == evidence_ids.len() as i64
            && switched.from_space == OLD_TEXT_SPACE
            && switched.to_space == TEXT_SPACE,
        "reindex must leave each item's original capture lineage and write transaction untouched, record which model produced each new vector, and report the model switch it serves, result={result:?}, switched={switched:?}, written_before={written_before:?}, written_after={written_after:?}, created_txids_before={created_txids_before:?}, created_txids={created_txids:?}, new_producer_rows={new_producer_rows}"
    );
}

#[test]
fn reindex_resumable_after_interruption_picks_up_at_last_committed_cursor() {
    let fixture = fixture("reindex-resume");
    let (fixture, _) = fixture.switch_to_new_model();
    let interrupted = fixture.store.reindex_space(
        TEXT_SPACE,
        ReindexOptions {
            interrupt_after: Some(1),
        },
    );
    let calls_before_resume = fixture.reindex_calls.lock().expect("calls").len();
    let vector_snapshot_sql = "SELECT id, target_id, created_at, vector_text, produced_by FROM evidence_embeddings WHERE embedding_space_id = $space";
    let vector_snapshot_params = params(vec![("space", Value::Text(TEXT_SPACE.to_string()))]);
    let committed_before_resume = fixture
        .store
        .sync_database()
        .execute(vector_snapshot_sql, &vector_snapshot_params)
        .expect("snapshot the committed vector before resume")
        .rows;
    let resumed = fixture
        .store
        .reindex_space(TEXT_SPACE, ReindexOptions::default());
    let calls_after_resume = fixture.reindex_calls.lock().expect("calls").len();
    let total = fixture.evidence_ids.len() as u64;
    let db = fixture.store.sync_database();
    let committed_after_resume = db
        .execute(vector_snapshot_sql, &vector_snapshot_params)
        .expect("snapshot the completed vectors after resume")
        .rows;

    assert!(
        matches!(
            interrupted,
            Err(CgError::ReindexPaused {
                ref embedding_space_id,
                remaining,
            }) if embedding_space_id == TEXT_SPACE && remaining == total - 1
        ) && resumed
            .as_ref()
            .is_ok_and(|report| report.embedded == total - 1 && report.remaining == 0)
            && calls_after_resume - calls_before_resume == (total - 1) as usize
            && committed_before_resume.len() == 1
            && committed_before_resume
                .iter()
                .all(|row| committed_after_resume.contains(row))
            && vector_index_matches_sources(&db, &fixture.evidence_ids, &fixture.source_texts),
        "interrupted reindex must keep its committed batch and resume from it, embedding only what remained, interrupted={interrupted:?}, resumed={resumed:?}, calls_before_resume={calls_before_resume}, calls_after_resume={calls_after_resume}, committed_before_resume={committed_before_resume:?}, committed_after_resume={committed_after_resume:?}"
    );
}

#[test]
fn reindex_concurrent_writes_during_reindex_window_land_in_target_space_deterministically() {
    let fixture = fixture("reindex-concurrent-write");
    let expected_scan_rows = fixture.evidence_ids.len() + 96;
    // Padding is written under the old model, before the switch, so it is reindex work.
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
            embeddings: Vec::new(),
        });
        assert!(
            seeded.is_ok(),
            "reindex concurrent padding seed failed: {seeded:?}"
        );
    }
    let (fixture, _) = fixture.switch_to_new_model();
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
    // A writer whose record_observation returned Err committed nothing; only acknowledged writes
    // are owed a landing in the target space.
    let acknowledged = new_evidence_ids
        .iter()
        .zip(writer_results.iter())
        .filter(|(_, result)| result.is_ok())
        .map(|(id, _)| *id)
        .collect::<Vec<_>>();
    let errored = new_evidence_ids
        .iter()
        .zip(writer_results.iter())
        .filter(|(_, result)| result.is_err())
        .map(|(id, _)| *id)
        .collect::<Vec<_>>();
    let errored_writers_committed_nothing = errored.iter().all(|evidence_id| {
        count(
            &db,
            "SELECT COUNT(*) AS n FROM evidence WHERE id = $evidence_id",
            &params(vec![("evidence_id", Value::Uuid(evidence_id.as_uuid()))]),
        ) == 0
    });
    let entered_ids = entered_before_done
        .lock()
        .expect("entered ids")
        .iter()
        .copied()
        .filter(|evidence_id| acknowledged.contains(evidence_id))
        .collect::<Vec<_>>();
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
    let new_producer_rows = entered_ids
        .iter()
        .map(|evidence_id| {
            count(
                &db,
                "SELECT COUNT(*) AS n FROM evidence_embeddings WHERE target_id = $evidence_id AND embedding_space_id = $space AND produced_by = $producer",
                &params(vec![
                    ("evidence_id", Value::Uuid(evidence_id.as_uuid())),
                    ("space", Value::Text(TEXT_SPACE.to_string())),
                    (
                        "producer",
                        Value::Json(
                            serde_json::to_value(EvidenceProducer::from(&new_identity()))
                                .expect("json"),
                        ),
                    ),
                ]),
            )
        })
        .sum::<i64>();

    assert!(
        reindex_result.is_ok()
            && errored_writers_committed_nothing
            && entered_before_reindex_done > 0
            && !entered_ids.is_empty()
            && reindex_result
                .as_ref()
                .is_ok_and(|report| report.embedded == expected_scan_rows as u64)
            && entered_created_txids.iter().all(Option::is_some)
            && vector_index_matches_sources(&db, &entered_ids, &entered_texts)
            && writer_texts_embedded_only_by_writer_threads(&embed_calls, &entered_texts)
            && new_producer_rows == entered_ids.len() as i64,
        "acknowledged writes that entered before reindex completed must commit, be embedded on the writer path into the active space from their original source text, and never be embedded by the reindex, which embeds exactly the items written before the switch, reindex_result={reindex_result:?}, writer_results={writer_results:?}, errored={errored:?}, errored_writers_committed_nothing={errored_writers_committed_nothing}, entered_before_reindex_done={entered_before_reindex_done}, expected_scan_rows={expected_scan_rows}, entered_created_txids={entered_created_txids:?}, entered_ids={entered_ids:?}, entered_texts={entered_texts:?}, reindex_calls={:?}, embed_calls={:?}, new_producer_rows={new_producer_rows}",
        reindex_calls.lock().expect("calls"),
        embed_calls.lock().expect("detailed calls")
    );
}
