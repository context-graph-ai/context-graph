use chrono::Utc;
use context_graph::*;
use contextdb_core::{Error, RowId, Value, VectorIndexRef};
use contextdb_engine::Database;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use tempfile::TempDir;

fn shared_config() -> (TempDir, StoreConfig) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        extra_embedding_spaces: vec![EmbeddingSpaceDecl {
            embedding_space_id: "vigil_face_reid_v1".to_string(),
            dimension: 512,
            metric: DistanceMetric::Cosine,
            model_family: "vision".to_string(),
            model_name: "vigil-face".to_string(),
            model_version: "1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::ImageFrame, EvidenceKind::Text],
            default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: "vigil_face_reid_v1".to_string(),
                table_name: "evidence".to_string(),
                vector_column: "vector_vision".to_string(),
            }],
        }],
        ..Default::default()
    };
    (dir, config)
}

fn raw_params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn table_count(db: &Database, table: &str) -> i64 {
    let result = db
        .execute(
            &format!("SELECT COUNT(*) AS n FROM {table}"),
            &HashMap::new(),
        )
        .expect("count table");
    assert_eq!(result.rows.len(), 1);
    match result.rows[0][0] {
        Value::Int64(n) => n,
        ref other => panic!("expected count integer, got {other:?}"),
    }
}

fn count_where_uuid(db: &Database, table: &str, column: &str, id: uuid::Uuid) -> i64 {
    let result = db
        .execute(
            &format!("SELECT COUNT(*) AS n FROM {table} WHERE {column} = $id"),
            &raw_params(vec![("id", Value::Uuid(id))]),
        )
        .expect("count row");
    assert_eq!(result.rows.len(), 1);
    match result.rows[0][0] {
        Value::Int64(n) => n,
        ref other => panic!("expected count integer, got {other:?}"),
    }
}

fn vector_hits_contain_id(
    db: &Database,
    table: &str,
    hits: &[(RowId, f32)],
    expected: uuid::Uuid,
) -> bool {
    let rows = db
        .scan(table, db.snapshot())
        .expect("scan vector hit table");
    hits.iter().any(|(row_id, _)| {
        rows.iter()
            .find(|row| row.row_id == *row_id)
            .is_some_and(|row| row.values.get("id") == Some(&Value::Uuid(expected)))
    })
}

fn producer(system: &str) -> EvidenceProducer {
    EvidenceProducer {
        system: system.to_string(),
        model_name: "fixture".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn text_vector(seed: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[seed % 768] = 1.0;
    vector
}

fn vision_vector(seed: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 512];
    vector[seed % 512] = 1.0;
    vector
}

fn text_observation(
    observation_id: ObservationId,
    evidence_id: EvidenceId,
    entity: EntityId,
    context: ContextId,
    state_delta: BTreeMap<String, Option<serde_json::Value>>,
    vector: Option<Vec<f32>>,
) -> RecordObservation {
    let evidence = EvidenceRef {
        id: evidence_id,
        observation_id,
        context_id: context,
        kind: EvidenceKind::Text,
        source_ref: "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        mime_type: None,
        content_hash: Some(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        ),
        captured_at: Some(Utc::now()),
        byte_range: None,
        time_range_ms: None,
        frame_index: None,
        region: None,
        producer: producer("probe"),
        text_projection: None,
        derived_from: None,
        retention_status: RetentionStatus::RetainedExternal,
        media_retention_until: None,
        properties: BTreeMap::new(),
    };
    let embeddings = vector
        .map(|vector| PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: "text_default_embeddinggemma_300m_768".to_string(),
            vector,
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer("probe"),
        })
        .into_iter()
        .collect();
    RecordObservation {
        id: observation_id,
        entity_id: entity,
        context_id: context,
        observation_type: "probe".to_string(),
        source: "probe".to_string(),
        observed_at: Utc::now(),
        evidence: vec![evidence],
        observed_properties: BTreeMap::new(),
        state_delta,
        properties: BTreeMap::new(),
        embeddings,
    }
}

fn seed_context_entity(store: &Store) -> (ContextId, EntityId) {
    let context = store
        .create_context(CreateContext {
            name: format!("probe-{}", ContextId::new_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: format!("entity-{}", EntityId::new_v7()),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    (context.id, entity.id)
}

fn seed_decision(store: &Store) -> (DecisionId, ObservationId) {
    let (context, entity) = seed_context_entity(store);
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "probe intention".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context,
            properties: BTreeMap::new(),
            tags: Vec::new(),
        })
        .expect("intention");
    let observation_id = ObservationId::new_v7();
    let record = store
        .record_observation(text_observation(
            observation_id,
            EvidenceId::new_v7(),
            entity,
            context,
            BTreeMap::from([("basis".to_string(), Some(json!("v1")))]),
            None,
        ))
        .expect("record");
    let snapshot_id = record
        .snapshot_id
        .or_else(|| {
            store
                .current_entity_snapshot(entity)
                .expect("current snapshot")
                .map(|snapshot| snapshot.id)
        })
        .expect("basis snapshot");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "probe".to_string(),
            description: "probe decision".to_string(),
            context_id: context,
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity],
            basis_fields: None,
            based_on_snapshots: vec![snapshot_id],
            ..Default::default()
        })
        .expect("decision");
    (decision.id, observation_id)
}

fn decision_context(store: &Store, decision: DecisionId) -> ContextId {
    store
        .get_decision(decision)
        .expect("decision")
        .expect("decision row")
        .context_id
}

#[test]
fn probe_record_observation_observation_and_evidence_visible_atomically() {
    let (_dir, config) = shared_config();
    let seed = Store::open(config.clone()).expect("open");
    let (context, entity) = seed_context_entity(&seed);
    drop(seed);
    let pairs: Vec<_> = (0..2)
        .map(|_| (ObservationId::new_v7(), EvidenceId::new_v7()))
        .collect();
    let barrier = Arc::new(Barrier::new(3));
    let done = Arc::new(AtomicBool::new(false));
    let torn = Arc::new(AtomicBool::new(false));

    let reader_config = config.clone();
    let reader_pairs = pairs.clone();
    let reader_barrier = Arc::clone(&barrier);
    let reader_done = Arc::clone(&done);
    let reader_torn = Arc::clone(&torn);
    let reader = std::thread::spawn(move || {
        let reader = Store::open(reader_config).expect("reader open");
        reader_barrier.wait();
        while !reader_done.load(Ordering::SeqCst) {
            for (observation_id, _) in &reader_pairs {
                if let Ok(observation) = reader.get_observation(*observation_id)
                    && observation.evidence.is_empty()
                {
                    reader_torn.store(true, Ordering::SeqCst);
                }
            }
        }
    });

    let mut writers = Vec::new();
    for (writer_idx, (observation_id, evidence_id)) in pairs.iter().copied().enumerate() {
        let writer_config = config.clone();
        let writer_barrier = Arc::clone(&barrier);
        writers.push(std::thread::spawn(move || {
            let writer = Store::open(writer_config).expect("writer open");
            writer_barrier.wait();
            writer.record_observation(text_observation(
                observation_id,
                evidence_id,
                entity,
                context,
                BTreeMap::from([(format!("writer_{writer_idx}"), Some(json!(writer_idx)))]),
                Some(text_vector(writer_idx)),
            ))
        }));
    }

    for writer in writers {
        writer.join().expect("writer joined").expect("record");
    }
    done.store(true, Ordering::SeqCst);
    reader.join().expect("reader joined");
    assert!(!torn.load(Ordering::SeqCst));

    let reader = Store::open(config).expect("final reader");
    for (observation_id, evidence_id) in pairs {
        assert_eq!(
            reader
                .list_evidence_for_observation(observation_id)
                .expect("evidence")
                .iter()
                .map(|e| e.id)
                .collect::<Vec<_>>(),
            vec![evidence_id]
        );
    }
}

#[test]
fn probe_invalidate_decision_conditional_where_exactly_one_winner_under_4_writer_barrier() {
    let (_dir, config) = shared_config();
    let seed = Store::open(config.clone()).expect("open");
    let (decision_id, trigger_id) = seed_decision(&seed);
    drop(seed);
    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let writer_config = config.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let writer = Store::open(writer_config).expect("writer open");
                barrier.wait();
                writer.invalidate_decision(decision_id, trigger_id, "same basis changed")
            })
        })
        .collect();
    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer joined"))
        .collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    let reader = Store::open(config).expect("reader");
    assert_eq!(
        reader
            .list_invalidations(
                decision_context(&reader, decision_id),
                Some(decision_id),
                None
            )
            .expect("invalidations")
            .len(),
        1
    );
}

#[test]
fn probe_evidence_availability_log_concurrent_inserts_get_distinct_monotonic_txids() {
    let (_dir, config) = shared_config();
    let seed = Store::open(config.clone()).expect("open");
    let (context, entity) = seed_context_entity(&seed);
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    seed.record_observation(text_observation(
        observation_id,
        evidence_id,
        entity,
        context,
        BTreeMap::from([("seed".to_string(), Some(json!(true)))]),
        None,
    ))
    .expect("seed evidence");
    drop(seed);
    let statuses = [
        RetentionStatus::NotStored,
        RetentionStatus::Ephemeral,
        RetentionStatus::RetainedExternal,
        RetentionStatus::Expired,
        RetentionStatus::Unavailable,
        RetentionStatus::Expired,
    ];
    let barrier = Arc::new(Barrier::new(statuses.len()));
    let handles: Vec<_> = statuses
        .into_iter()
        .map(|status| {
            let writer_config = config.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let writer = Store::open(writer_config).expect("writer open");
                barrier.wait();
                writer.update_evidence_retention_status(evidence_id, status)
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("writer joined").expect("retention");
    }
    let rows = Store::open(config)
        .expect("reader")
        .evidence_availability_log(evidence_id)
        .expect("log");
    assert_eq!(rows.len(), 6);
    assert_eq!(
        rows.iter()
            .map(|row| row.recorded_at_txid)
            .collect::<BTreeSet<_>>()
            .len(),
        rows.len()
    );
    assert!(
        rows.windows(2)
            .all(|pair| pair[0].recorded_at_txid < pair[1].recorded_at_txid)
    );
}

#[test]
fn probe_record_observation_vector_write_atomic_rolls_back_on_validation_error() {
    let (_dir, config) = shared_config();
    let seed = Store::open(config.clone()).expect("open");
    let (context, entity) = seed_context_entity(&seed);
    drop(seed);
    let pre_db = Database::open(&config.db_path).expect("open raw contextdb");
    let pre_observations = table_count(&pre_db, "observations");
    let pre_evidence = table_count(&pre_db, "evidence");
    let pre_entity_snapshots = table_count(&pre_db, "entity_snapshots");
    let pre_entity_snapshot_observations = table_count(&pre_db, "entity_snapshot_observations");
    let pre_embedding_handles = table_count(&pre_db, "evidence_embedding_handles");
    let pre_evidence_availability_log = table_count(&pre_db, "evidence_availability_log");
    drop(pre_db);

    let attempts = (0..2)
        .map(|writer_idx| {
            (
                writer_idx,
                ObservationId::new_v7(),
                EvidenceId::new_v7(),
                EmbeddingId::new_v7(),
                text_vector(writer_idx),
            )
        })
        .collect::<Vec<_>>();
    let barrier = Arc::new(Barrier::new(2));
    let thread_attempts = attempts.clone();
    let handles: Vec<_> = thread_attempts
        .into_iter()
        .map(
            |(writer_idx, observation_id, evidence_id, embedding_id, vector)| {
                let writer_config = config.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let writer = Store::open(writer_config).expect("writer open");
                    barrier.wait();
                    let mut input = text_observation(
                        observation_id,
                        evidence_id,
                        entity,
                        context,
                        BTreeMap::from([(format!("writer_{writer_idx}"), Some(json!(writer_idx)))]),
                        Some(vector),
                    );
                    input.embeddings[0].id = embedding_id;
                    input.evidence[0].region =
                        Some(json!({"x": -0.1, "y": 0.1, "w": 0.2, "h": 0.2}));
                    writer.record_observation(input)
                })
            },
        )
        .collect();
    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer joined"))
        .collect();
    assert!(
        results
            .iter()
            .all(|result| matches!(result, Err(CgError::InvalidEvidenceRegionBounds { .. })))
    );
    let db = Database::open(&config.db_path).expect("open raw contextdb");
    assert_eq!(table_count(&db, "observations"), pre_observations);
    assert_eq!(table_count(&db, "evidence"), pre_evidence);
    assert_eq!(table_count(&db, "entity_snapshots"), pre_entity_snapshots);
    assert_eq!(
        table_count(&db, "entity_snapshot_observations"),
        pre_entity_snapshot_observations
    );
    assert_eq!(
        table_count(&db, "evidence_embedding_handles"),
        pre_embedding_handles
    );
    assert_eq!(
        table_count(&db, "evidence_availability_log"),
        pre_evidence_availability_log
    );
    for (_, observation_id, evidence_id, embedding_id, vector) in attempts {
        assert_eq!(
            count_where_uuid(&db, "observations", "id", observation_id.as_uuid()),
            0
        );
        assert_eq!(
            count_where_uuid(&db, "evidence", "id", evidence_id.as_uuid()),
            0
        );
        assert_eq!(
            count_where_uuid(
                &db,
                "evidence_embedding_handles",
                "id",
                embedding_id.as_uuid()
            ),
            0
        );
        let hits = db
            .query_vector(
                VectorIndexRef::new("evidence", "vector_text"),
                &vector,
                10,
                None,
                db.snapshot(),
            )
            .expect("query");
        assert!(
            !vector_hits_contain_id(&db, "evidence", &hits, evidence_id.as_uuid()),
            "failed record_observation leaked vector_text entry for {evidence_id:?}"
        );
    }
}

#[test]
fn probe_evidence_derived_from_fk_delete_race_rejects_child_insert() {
    let (_dir, config) = shared_config();
    drop(Store::open(config.clone()).expect("open"));
    let db = Arc::new(Database::open(&config.db_path).expect("open raw contextdb"));
    db.execute(
        "CREATE TABLE evidence_delete_race_probe (id UUID PRIMARY KEY, derived_from UUID REFERENCES evidence_delete_race_probe(id))",
        &HashMap::new(),
    )
    .expect("mutable fk fixture");
    let parent = EvidenceId::new_v7();
    let child = EvidenceId::new_v7();
    db.execute(
        "INSERT INTO evidence_delete_race_probe (id) VALUES ($id)",
        &raw_params(vec![("id", Value::Uuid(parent.as_uuid()))]),
    )
    .expect("parent row");

    let (child_tx_started_send, child_tx_started_recv) = mpsc::channel();
    let (delete_committed_send, delete_committed_recv) = mpsc::channel();
    let delete_parent = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || {
            child_tx_started_recv.recv().expect("child tx started");
            let tx = db.begin().expect("begin");
            let deleted = db.execute_in_tx(
                tx,
                "DELETE FROM evidence_delete_race_probe WHERE id = $id",
                &raw_params(vec![("id", Value::Uuid(parent.as_uuid()))]),
            );
            let result = match deleted {
                Ok(_) => db.commit(tx),
                Err(err) => {
                    let _ = db.rollback(tx);
                    Err(err)
                }
            };
            delete_committed_send
                .send(())
                .expect("notify delete committed");
            result
        })
    };
    let child_insert = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || {
            let tx = db.begin().expect("begin");
            child_tx_started_send.send(()).expect("notify child tx");
            delete_committed_recv.recv().expect("delete committed");
            let inserted = db.execute_in_tx(
                tx,
                "INSERT INTO evidence_delete_race_probe (id, derived_from) VALUES ($id, $derived_from)",
                &raw_params(vec![
                    ("id", Value::Uuid(child.as_uuid())),
                    ("derived_from", Value::Uuid(parent.as_uuid())),
                ]),
            );
            match inserted {
                Ok(_) => db.commit(tx),
                Err(err) => {
                    let _ = db.rollback(tx);
                    Err(err)
                }
            }
        })
    };
    let delete_result = delete_parent.join().expect("delete writer joined");
    let result = child_insert.join().expect("child writer joined");
    assert!(delete_result.is_ok(), "parent delete must win the race");
    assert!(matches!(result, Err(Error::ForeignKeyViolation { .. })));
}

#[test]
fn probe_record_observation_concurrent_same_entity_snapshot_timeline_contiguous() {
    let (_dir, config) = shared_config();
    let seed = Store::open(config.clone()).expect("open");
    let (context, entity) = seed_context_entity(&seed);
    drop(seed);
    let barrier = Arc::new(Barrier::new(6));
    let handles: Vec<_> = (0..6)
        .map(|writer| {
            let writer_config = config.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let writer_store = Store::open(writer_config).expect("writer open");
                barrier.wait();
                writer_store.record_observation(text_observation(
                    ObservationId::new_v7(),
                    EvidenceId::new_v7(),
                    entity,
                    context,
                    BTreeMap::from([(format!("writer_{writer}"), Some(json!(writer)))]),
                    None,
                ))
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("writer joined").expect("record");
    }
    let history = Store::open(config)
        .expect("reader")
        .entity_history(entity)
        .expect("history");
    assert_eq!(history.len(), 7);
    assert!(
        history
            .windows(2)
            .all(|pair| pair[0].valid_to == Some(pair[1].valid_from))
    );
    assert_eq!(
        history
            .iter()
            .map(|snapshot| snapshot.captured_at)
            .collect::<BTreeSet<_>>()
            .len(),
        history.len()
    );
}

#[test]
fn probe_basis_diff_under_concurrent_writer_returns_mvcc_consistent_horizon() {
    let (_dir, config) = shared_config();
    let seed = Store::open(config.clone()).expect("open");
    let (decision_id, observation_id) = seed_decision(&seed);
    let seeded_observation = seed.get_observation(observation_id).expect("observation");
    let context = seeded_observation.context_id;
    let entity = seeded_observation.entity_id;
    drop(seed);
    let barrier = Arc::new(Barrier::new(5));
    let stop = Arc::new(AtomicBool::new(false));
    let committed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (diff_tx, diff_rx) = mpsc::channel();
    let writers = (0..4)
        .map(|_| {
            let writer_config = config.clone();
            let writer_barrier = Arc::clone(&barrier);
            let writer_stop = Arc::clone(&stop);
            let writer_committed = Arc::clone(&committed);
            let writer_attempts = Arc::clone(&attempts);
            std::thread::spawn(move || {
                let writer = Store::open(writer_config).expect("writer open");
                writer_barrier.wait();
                while !writer_stop.load(Ordering::SeqCst) {
                    let seq = writer_attempts.fetch_add(1, Ordering::SeqCst);
                    if writer
                        .record_observation(text_observation(
                            ObservationId::new_v7(),
                            EvidenceId::new_v7(),
                            entity,
                            context,
                            BTreeMap::from([("basis".to_string(), Some(json!(seq)))]),
                            None,
                        ))
                        .is_ok()
                    {
                        writer_committed.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    let reader_config = config.clone();
    let reader_barrier = Arc::clone(&barrier);
    let reader_committed = Arc::clone(&committed);
    std::thread::spawn(move || {
        let reader = Store::open(reader_config).expect("reader open");
        reader_barrier.wait();
        while reader_committed.load(Ordering::SeqCst) < 50 {
            std::thread::yield_now();
        }
        let before = reader_committed.load(Ordering::SeqCst);
        let diff = reader.basis_diff(decision_id);
        let after = reader_committed.load(Ordering::SeqCst);
        diff_tx.send((before, after, diff)).expect("send diff");
    });
    let (before_diff, after_diff, diff) = diff_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("basis_diff returned while writer flood was still active");
    stop.store(true, Ordering::SeqCst);
    for writer in writers {
        writer.join().expect("writer joined");
    }
    assert!(before_diff >= 50);
    assert!(after_diff >= before_diff);
    let reader = Store::open(config).expect("reader");
    let history = reader.entity_history(entity).expect("history");
    assert!(history.len() >= 51);
    let diff = diff.expect("basis diff");
    assert!(diff.is_invalidated_candidate);
    assert_eq!(diff.read_txid, diff.current_snapshot_read_txid);
    let snapshot_at_diff = history
        .iter()
        .find(|snapshot| {
            snapshot.valid_from <= diff.read_txid
                && snapshot
                    .valid_to
                    .map(|valid_to| diff.read_txid < valid_to)
                    .unwrap_or(true)
        })
        .expect("snapshot visible at basis_diff read txid");
    assert!(
        diff.divergences.iter().any(|divergence| {
            divergence.value_diffs.iter().any(|value_diff| {
                value_diff.field == "basis"
                    && value_diff.current_value.as_ref() == snapshot_at_diff.properties.get("basis")
            })
        }),
        "basis_diff must report the current value from one snapshot at its read txid"
    );
}

#[test]
fn probe_evidence_two_vectors_text_and_vision_isolated_per_column_under_concurrent_writers() {
    let (_dir, config) = shared_config();
    let seed = Store::open(config.clone()).expect("open");
    let (context, entity) = seed_context_entity(&seed);
    drop(seed);
    let barrier = Arc::new(Barrier::new(4));
    let evidence_ids: Vec<_> = (0..4).map(|_| EvidenceId::new_v7()).collect();
    let handles: Vec<_> = evidence_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(writer_idx, evidence_id)| {
            let writer_config = config.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let writer = Store::open(writer_config).expect("writer open");
                let mut input = text_observation(
                    ObservationId::new_v7(),
                    evidence_id,
                    entity,
                    context,
                    BTreeMap::from([(format!("writer_{writer_idx}"), Some(json!(writer_idx)))]),
                    Some(text_vector(writer_idx)),
                );
                input.embeddings.push(PrecomputedEmbedding {
                    id: EmbeddingId::new_v7(),
                    embedding_space_id: "vigil_face_reid_v1".to_string(),
                    vector: vision_vector(writer_idx),
                    target: EmbeddingTarget::Evidence(evidence_id),
                    produced_by: producer("probe"),
                });
                barrier.wait();
                writer.record_observation(input)
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("writer joined").expect("record");
    }
    let db = Database::open(&config.db_path).expect("open raw contextdb");
    for (writer_idx, expected_id) in evidence_ids.iter().copied().enumerate() {
        let text_hits = db
            .execute(
                "SELECT id FROM evidence ORDER BY vector_text <=> $query LIMIT 1",
                &raw_params(vec![("query", Value::Vector(text_vector(writer_idx)))]),
            )
            .expect("text query");
        assert_eq!(text_hits.rows.len(), 1);
        assert_eq!(
            text_hits.rows[0][0],
            Value::Uuid(expected_id.as_uuid()),
            "text query must return the row whose text vector was written to vector_text"
        );

        let vision_hits = db
            .execute(
                "SELECT id FROM evidence ORDER BY vector_vision <=> $query LIMIT 1",
                &raw_params(vec![("query", Value::Vector(vision_vector(writer_idx)))]),
            )
            .expect("vision query");
        assert_eq!(vision_hits.rows.len(), 1);
        assert_eq!(
            vision_hits.rows[0][0],
            Value::Uuid(expected_id.as_uuid()),
            "vision query must return the row whose vision vector was written to vector_vision"
        );

        assert!(matches!(
            db.execute(
                "SELECT id FROM evidence ORDER BY vector_text <=> $query LIMIT 1",
                &raw_params(vec![("query", Value::Vector(vision_vector(writer_idx)))])
            ),
            Err(Error::VectorIndexDimensionMismatch {
                expected: 768,
                actual: 512,
                ..
            })
        ));
        assert!(matches!(
            db.execute(
                "SELECT id FROM evidence ORDER BY vector_vision <=> $query LIMIT 1",
                &raw_params(vec![("query", Value::Vector(text_vector(writer_idx)))])
            ),
            Err(Error::VectorIndexDimensionMismatch {
                expected: 512,
                actual: 768,
                ..
            })
        ));
    }
}

#[test]
fn probe_extra_embedding_space_concurrent_idempotent_seed_collapses_to_one_row() {
    let (_dir, config) = shared_config();
    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let writer_config = config.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                Store::open(writer_config)
            })
        })
        .collect();
    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer joined"))
        .collect();
    assert!(results.iter().all(Result::is_ok));
    drop(results);
    let count = Store::open(config)
        .expect("reader")
        .list_embedding_spaces()
        .expect("spaces")
        .iter()
        .filter(|space| space.embedding_space_id == "vigil_face_reid_v1")
        .count();
    assert_eq!(count, 1);
}

#[test]
fn probe_decisions_pk_update_concurrent_writers_collapse_to_one_visible_row() {
    let (_dir, config) = shared_config();
    let seed = Store::open(config.clone()).expect("open");
    let (decision_id, _) = seed_decision(&seed);
    let decision_context = seed
        .get_decision(decision_id)
        .expect("decision")
        .expect("visible decision")
        .context_id;
    let supplied_bases: Vec<Vec<EntitySnapshotId>> = (0..6)
        .map(|writer_idx| {
            (0..2)
                .map(|basis_idx| {
                    let entity = seed
                        .create_entity(CreateEntity {
                            entity_type: EntityType::Service,
                            name: format!("basis-{writer_idx}-{basis_idx}"),
                            properties: BTreeMap::from([("writer".to_string(), json!(writer_idx))]),
                            tags: Vec::new(),
                            context_id: decision_context,
                        })
                        .expect("basis entity");
                    seed.current_entity_snapshot(entity.id)
                        .expect("current snapshot")
                        .expect("basis snapshot")
                        .id
                })
                .collect()
        })
        .collect();
    drop(seed);
    let barrier = Arc::new(Barrier::new(6));
    let handles: Vec<_> = supplied_bases
        .clone()
        .into_iter()
        .map(|basis| {
            let writer_config = config.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let writer = Store::open(writer_config).expect("writer open");
                barrier.wait();
                writer.update_decision_basis(decision_id, basis)
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("writer joined").expect("update");
    }
    let decision = Store::open(config)
        .expect("reader")
        .get_decision(decision_id)
        .expect("decision")
        .expect("visible decision");
    assert!(supplied_bases.contains(&decision.based_on_snapshots));
}
