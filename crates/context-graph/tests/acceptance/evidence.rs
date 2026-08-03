use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{Duration, Utc};
use context_graph::*;
use contextdb_core::{ColumnType, Error, RowId, Value, VectorIndexRef};
use contextdb_engine::Database;

const TEXT_SPACE: &str = "text_default_nomic_v1_5_768";

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open");
    (dir, store)
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "acceptance".to_string(),
        model_name: "fixture".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn text_vector() -> Vec<f32> {
    let mut v = vec![0.0; 768];
    v[0] = 1.0;
    v
}

fn evidence_ref(observation_id: ObservationId, evidence_id: EvidenceId) -> EvidenceRef {
    EvidenceRef {
        id: evidence_id,
        observation_id,
        context_id: ContextId::default(),
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
        region: Some(serde_json::json!({"line_start": 1, "line_end": 2})),
        producer: producer(),
        text_projection: None,
        derived_from: None,
        retention_status: RetentionStatus::RetainedExternal,
        media_retention_until: None,
        properties: BTreeMap::new(),
    }
}

fn image_evidence_ref(observation_id: ObservationId, evidence_id: EvidenceId) -> EvidenceRef {
    EvidenceRef {
        id: evidence_id,
        observation_id,
        context_id: ContextId::default(),
        kind: EvidenceKind::ImageFrame,
        source_ref: "vigil-edge://camera-7/frame/42".to_string(),
        mime_type: Some("image/jpeg".to_string()),
        content_hash: Some(
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        ),
        captured_at: Some(Utc::now()),
        byte_range: None,
        time_range_ms: None,
        frame_index: Some(42),
        region: Some(serde_json::json!({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4})),
        producer: EvidenceProducer {
            system: "vigil-edge".to_string(),
            model_name: "yolov8s".to_string(),
            model_version: "0.42.1".to_string(),
            pipeline_version: "1.7".to_string(),
        },
        text_projection: None,
        derived_from: None,
        retention_status: RetentionStatus::RetainedExternal,
        media_retention_until: None,
        properties: BTreeMap::new(),
    }
}

fn structured_evidence_ref(observation_id: ObservationId, evidence_id: EvidenceId) -> EvidenceRef {
    EvidenceRef {
        id: evidence_id,
        observation_id,
        context_id: ContextId::default(),
        kind: EvidenceKind::StructuredSignal,
        source_ref: "webhook://terraform/apply/42".to_string(),
        mime_type: Some("application/json".to_string()),
        content_hash: Some(
            "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
        ),
        captured_at: Some(Utc::now()),
        byte_range: None,
        time_range_ms: None,
        frame_index: None,
        region: None,
        producer: producer(),
        text_projection: None,
        derived_from: None,
        retention_status: RetentionStatus::RetainedExternal,
        media_retention_until: None,
        properties: BTreeMap::new(),
    }
}

fn set_evidence_context(evidence: &mut [EvidenceRef], context_id: ContextId) {
    for row in evidence {
        row.context_id = context_id;
    }
}

fn record_input_with(
    observation_id: ObservationId,
    entity_id: EntityId,
    context_id: ContextId,
    mut evidence: Vec<EvidenceRef>,
    state_delta: BTreeMap<String, Option<serde_json::Value>>,
    embeddings: Vec<PrecomputedEmbedding>,
) -> RecordObservation {
    set_evidence_context(&mut evidence, context_id);
    RecordObservation {
        id: observation_id,
        entity_id,
        context_id,
        observation_type: "acceptance".to_string(),
        source: "acceptance".to_string(),
        observed_at: Utc::now(),
        evidence,
        observed_properties: BTreeMap::from([("class".to_string(), serde_json::json!("person"))]),
        state_delta,
        properties: BTreeMap::new(),
        embeddings,
    }
}

fn record_input(observation_id: ObservationId, evidence_id: EvidenceId) -> RecordObservation {
    let evidence = evidence_ref(observation_id, evidence_id);
    record_input_with(
        observation_id,
        EntityId::new_v7(),
        ContextId::new_v7(),
        vec![evidence],
        BTreeMap::from([("region".to_string(), Some(serde_json::json!("eu")))]),
        vec![PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: text_vector(),
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer(),
        }],
    )
}

fn assert_evidence_ids_round_trip(
    label: &str,
    input: RecordObservation,
    expected: Vec<EvidenceId>,
) {
    let (_dir, store) = fresh_store();
    let observation_id = input.id;
    let result = store.record_observation(input).expect("record");
    assert_eq!(result.evidence_ids, expected, "{label}: result order");
    let returned = store
        .list_evidence_for_observation(observation_id)
        .expect("evidence");
    assert_eq!(
        returned.iter().map(|e| e.id).collect::<Vec<_>>(),
        result.evidence_ids,
        "{label}: persisted order"
    );
}

fn replay_evidence_refs(replay: &ReplayResult) -> Vec<EvidenceRef> {
    replay
        .evidence_lineage
        .iter()
        .flat_map(|lineage| lineage.evidence.iter().cloned())
        .collect()
}

fn replay_evidence_ids(replay: &ReplayResult) -> Vec<EvidenceId> {
    replay_evidence_refs(replay)
        .into_iter()
        .map(|evidence| evidence.id)
        .collect()
}

fn replay_evidence_ref(replay: &ReplayResult, id: EvidenceId) -> EvidenceRef {
    let found = replay_evidence_refs(replay)
        .into_iter()
        .find(|evidence| evidence.id == id);
    assert!(
        found.is_some(),
        "replay evidence lineage must contain evidence id {id}"
    );
    found.unwrap()
}

fn assert_record_error<F>(label: &str, mut input: RecordObservation, predicate: F)
where
    F: FnOnce(&CgError) -> bool,
{
    input.id = ObservationId::new_v7();
    assert_record_failure_zero_rollback(
        label,
        input,
        |result| matches!(result, Err(err) if predicate(err)),
    );
}

fn assert_record_zero_rollback(label: &str, input: RecordObservation) {
    assert_record_failure_zero_rollback(label, input, |result| result.is_err());
}

fn assert_record_failure_zero_rollback<F>(label: &str, input: RecordObservation, predicate: F)
where
    F: FnOnce(&std::result::Result<RecordObservationResult, CgError>) -> bool,
{
    let (_dir, store) = fresh_store();
    assert_record_failure_zero_rollback_inner(label, store, input, predicate, true);
}

fn assert_record_failure_zero_rollback_inner<F>(
    label: &str,
    store: Store,
    input: RecordObservation,
    predicate: F,
    expect_empty_tables: bool,
) where
    F: FnOnce(&std::result::Result<RecordObservationResult, CgError>) -> bool,
{
    let observation_id = input.id;
    let entity_id = input.entity_id;
    let evidence_ids = input.evidence.iter().map(|e| e.id).collect::<Vec<_>>();
    let embedding_ids = input.embeddings.iter().map(|e| e.id).collect::<Vec<_>>();
    let embedding_vectors = input
        .embeddings
        .iter()
        .map(|embedding| {
            (
                embedding.embedding_space_id.clone(),
                embedding.target.clone(),
                embedding.vector.clone(),
            )
        })
        .collect::<Vec<_>>();

    let result = store.record_observation(input);
    assert!(predicate(&result), "{label}: unexpected result {result:?}");

    let db = reopen_database_for_store(store);
    if expect_empty_tables {
        for table in [
            "observations",
            "evidence",
            "entity_snapshots",
            "entity_snapshot_observations",
            "evidence_availability_log",
        ] {
            assert_eq!(table_count(&db, table), 0, "{label}: {table} table changed");
        }
        for table in EMBEDDING_HANDLE_TABLES {
            assert_eq!(table_count(&db, table), 0, "{label}: {table} table changed");
        }
    }
    assert_eq!(
        count_where(
            &db,
            "observations",
            "id = $id",
            db_params(vec![("id", Value::Uuid(observation_id.as_uuid()))])
        ),
        0,
        "{label}: observation row leaked"
    );
    assert_eq!(
        count_where(
            &db,
            "entity_snapshots",
            "entity_id = $entity_id",
            db_params(vec![("entity_id", Value::Uuid(entity_id.as_uuid()))])
        ),
        0,
        "{label}: snapshot row leaked"
    );
    for evidence_id in evidence_ids {
        assert_eq!(
            count_where(
                &db,
                "evidence",
                "id = $id",
                db_params(vec![("id", Value::Uuid(evidence_id.as_uuid()))])
            ),
            0,
            "{label}: evidence row leaked for {evidence_id:?}"
        );
    }
    for embedding_id in embedding_ids {
        assert_eq!(
            embedding_handle_count_by_id(&db, embedding_id),
            0,
            "{label}: embedding handle leaked for {embedding_id:?}"
        );
    }
    for (space, target, vector) in embedding_vectors {
        let column_and_evidence_id = match (space.as_str(), target) {
            (TEXT_SPACE, EmbeddingTarget::Evidence(id)) => Some(("vector_text", id.as_uuid())),
            ("vigil_face_reid_v1", EmbeddingTarget::Evidence(id)) => {
                Some(("vector_vision", id.as_uuid()))
            }
            _ => None,
        };
        if let Some((column, evidence_id)) = column_and_evidence_id {
            let expected_dimension = match column {
                "vector_text" => 768,
                "vector_vision" => 512,
                _ => vector.len(),
            };
            if vector.len() != expected_dimension {
                continue;
            }
            let hits = db
                .query_vector(
                    VectorIndexRef::new("evidence", column),
                    &vector,
                    10,
                    None,
                    db.snapshot(),
                )
                .expect("rollback vector query");
            assert!(
                !vector_hits_contain_id(&db, "evidence", &hits, evidence_id),
                "{label}: vector index entry leaked for evidence.{column} id {evidence_id}"
            );
        }
    }
}

fn seed_decision(store: &Store) -> (DecisionId, ObservationId, EvidenceId) {
    seed_decision_with_source(store, "acceptance")
}

fn decision_context(store: &Store, decision: DecisionId) -> ContextId {
    store
        .get_decision(decision)
        .expect("decision")
        .expect("decision row")
        .context_id
}

fn seed_decision_with_source(
    store: &Store,
    source: &str,
) -> (DecisionId, ObservationId, EvidenceId) {
    let ctx = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "preserve evidence".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "rule".to_string(),
            properties: BTreeMap::from([("region".to_string(), serde_json::json!("us-east-1"))]),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let mut observation = record_input_with(
        observation_id,
        entity.id,
        ctx,
        vec![evidence_ref(observation_id, evidence_id)],
        BTreeMap::from([("region".to_string(), Some(serde_json::json!("eu-west-1")))]),
        Vec::new(),
    );
    observation.source = source.to_string();
    let observation_result = store.record_observation(observation).expect("observation");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "evidence".to_string(),
            description: "use evidence-backed basis".to_string(),
            reasoning: Vec::new(),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: None,
            based_on_snapshots: observation_result.snapshot_id.into_iter().collect(),
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx,
            precedent_ids: vec![],
            agent_id: None,
        })
        .expect("decision");
    (decision.id, observation_id, evidence_id)
}

fn seed_decision_with_evidence(
    store: &Store,
    mut evidence: Vec<EvidenceRef>,
    state_delta: BTreeMap<String, Option<serde_json::Value>>,
) -> (DecisionId, ObservationId, Vec<EvidenceRef>) {
    let observation_id = evidence
        .first()
        .expect("at least one evidence ref")
        .observation_id;
    let ctx = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "preserve evidence".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "rule".to_string(),
            properties: BTreeMap::from([("region".to_string(), serde_json::json!("us-east-1"))]),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    set_evidence_context(&mut evidence, ctx);
    let expected_evidence = evidence.clone();
    let observation_result = store
        .record_observation(record_input_with(
            observation_id,
            entity.id,
            ctx,
            evidence,
            state_delta,
            Vec::new(),
        ))
        .expect("observation");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "evidence".to_string(),
            description: "use evidence-backed basis".to_string(),
            reasoning: Vec::new(),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: None,
            based_on_snapshots: observation_result.snapshot_id.into_iter().collect(),
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx,
            precedent_ids: vec![],
            agent_id: None,
        })
        .expect("decision");
    (decision.id, observation_id, expected_evidence)
}

fn vision_config() -> StoreConfig {
    StoreConfig {
        extra_embedding_spaces: vec![EmbeddingSpaceDecl {
            embedding_space_id: "vigil_face_reid_v1".to_string(),
            dimension: 512,
            metric: DistanceMetric::Cosine,
            model_family: "vision".to_string(),
            model_name: "vigil-face".to_string(),
            model_version: "1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::ImageFrame],
            default_for_evidence_kind: Some(EvidenceKind::ImageFrame),
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: "vigil_face_reid_v1".to_string(),
                table_name: "evidence".to_string(),
                vector_column: "vector_vision".to_string(),
            }],
        }],
        ..Default::default()
    }
}

fn reopen_database_for_store(store: Store) -> Database {
    let db_path = store.db_path().to_path_buf();
    drop(store);
    Database::open(&db_path).expect("open raw contextdb")
}

fn db_params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

const EMBEDDING_HANDLE_TABLES: [&str; 3] = [
    "evidence_embedding_handles",
    "entity_reference_embedding_handles",
    "digest_embedding_handles",
];

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

fn embedding_handle_count(db: &Database) -> i64 {
    EMBEDDING_HANDLE_TABLES
        .iter()
        .map(|table| table_count(db, table))
        .sum()
}

fn embedding_handle_count_by_id(db: &Database, id: EmbeddingId) -> i64 {
    EMBEDDING_HANDLE_TABLES
        .iter()
        .map(|table| {
            count_where(
                db,
                table,
                "id = $id",
                db_params(vec![("id", Value::Uuid(id.as_uuid()))]),
            )
        })
        .sum()
}

fn count_where(
    db: &Database,
    table: &str,
    where_clause: &str,
    params: HashMap<String, Value>,
) -> i64 {
    let result = db
        .execute(
            &format!("SELECT COUNT(*) AS n FROM {table} WHERE {where_clause}"),
            &params,
        )
        .expect("count table");
    assert_eq!(result.rows.len(), 1);
    match result.rows[0][0] {
        Value::Int64(n) => n,
        ref other => panic!("expected count integer, got {other:?}"),
    }
}

fn raw_evidence_rows_for_observation(
    db: &Database,
    observation_id: ObservationId,
) -> Vec<HashMap<String, Value>> {
    let result = db
        .execute(
            "SELECT id, observation_id, context_id, kind, source_ref, mime_type, content_hash, captured_at, byte_range, time_range_ms, frame_index, region, producer, text_projection, derived_from, retention_status, media_retention_until, properties, supply_order FROM evidence WHERE observation_id = $observation_id ORDER BY supply_order",
            &db_params(vec![(
                "observation_id",
                Value::Uuid(observation_id.as_uuid()),
            )]),
        )
        .expect("raw evidence rows");
    result
        .rows
        .into_iter()
        .map(|row| result.columns.iter().cloned().zip(row).collect())
        .collect()
}

fn value_uuid(value: &Value) -> uuid::Uuid {
    match value {
        Value::Uuid(uuid) => *uuid,
        other => panic!("expected uuid value, got {other:?}"),
    }
}

fn value_text(value: &Value) -> &str {
    match value {
        Value::Text(text) => text,
        other => panic!("expected text value, got {other:?}"),
    }
}

fn value_json_from_properties(
    properties: &BTreeMap<String, serde_json::Value>,
) -> serde_json::Value {
    serde_json::Value::Object(
        properties
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

fn evidence_kind_value(kind: EvidenceKind) -> Value {
    Value::Text(
        match kind {
            EvidenceKind::Text => "text",
            EvidenceKind::ImageFrame => "image_frame",
            EvidenceKind::AudioSegment => "audio_segment",
            EvidenceKind::VideoSegment => "video_segment",
            EvidenceKind::StructuredSignal => "structured_signal",
        }
        .to_string(),
    )
}

fn retention_status_value(status: RetentionStatus) -> Value {
    Value::Text(
        match status {
            RetentionStatus::NotStored => "not_stored",
            RetentionStatus::Ephemeral => "ephemeral",
            RetentionStatus::RetainedExternal => "retained_external",
            RetentionStatus::Expired => "expired",
            RetentionStatus::Unavailable => "unavailable",
        }
        .to_string(),
    )
}

fn optional_text_value(value: &Option<String>) -> Value {
    value
        .as_ref()
        .map(|value| Value::Text(value.clone()))
        .unwrap_or(Value::Null)
}

fn optional_uuid_value(value: Option<EvidenceId>) -> Value {
    value
        .map(|value| Value::Uuid(value.as_uuid()))
        .unwrap_or(Value::Null)
}

fn optional_timestamp_value(value: &Option<chrono::DateTime<Utc>>) -> Value {
    value
        .map(|value| Value::Timestamp(value.timestamp_millis()))
        .unwrap_or(Value::Null)
}

fn optional_u64_value(value: Option<u64>) -> Value {
    value
        .map(|value| Value::Int64(value as i64))
        .unwrap_or(Value::Null)
}

fn optional_pair_json_value(value: Option<(u64, u64)>) -> Value {
    value
        .map(|(start, end)| Value::Json(serde_json::json!([start, end])))
        .unwrap_or(Value::Null)
}

fn optional_json_value(value: &Option<serde_json::Value>) -> Value {
    value
        .as_ref()
        .map(|value| Value::Json(value.clone()))
        .unwrap_or(Value::Null)
}

fn assert_raw_evidence_matches(
    row: &HashMap<String, Value>,
    expected: &EvidenceRef,
    supply_order: i64,
) {
    assert_eq!(row.get("id"), Some(&Value::Uuid(expected.id.as_uuid())));
    assert_eq!(
        row.get("observation_id"),
        Some(&Value::Uuid(expected.observation_id.as_uuid()))
    );
    assert_eq!(
        row.get("context_id"),
        Some(&Value::Uuid(expected.context_id.as_uuid()))
    );
    assert_eq!(row.get("kind"), Some(&evidence_kind_value(expected.kind)));
    assert_eq!(
        row.get("source_ref"),
        Some(&Value::Text(expected.source_ref.clone()))
    );
    assert_eq!(
        row.get("mime_type"),
        Some(&optional_text_value(&expected.mime_type))
    );
    assert_eq!(
        row.get("content_hash"),
        Some(&optional_text_value(&expected.content_hash))
    );
    assert_eq!(
        row.get("captured_at"),
        Some(&optional_timestamp_value(&expected.captured_at))
    );
    assert_eq!(
        row.get("byte_range"),
        Some(&optional_pair_json_value(expected.byte_range))
    );
    assert_eq!(
        row.get("time_range_ms"),
        Some(&optional_pair_json_value(expected.time_range_ms))
    );
    assert_eq!(
        row.get("frame_index"),
        Some(&optional_u64_value(expected.frame_index))
    );
    assert_eq!(
        row.get("region"),
        Some(&optional_json_value(&expected.region))
    );
    assert_eq!(
        row.get("producer"),
        Some(&Value::Json(
            serde_json::to_value(&expected.producer).unwrap()
        ))
    );
    assert_eq!(
        row.get("text_projection"),
        Some(&optional_text_value(&expected.text_projection))
    );
    assert_eq!(
        row.get("derived_from"),
        Some(&optional_uuid_value(expected.derived_from))
    );
    assert_eq!(
        row.get("retention_status"),
        Some(&retention_status_value(expected.retention_status))
    );
    assert_eq!(
        row.get("media_retention_until"),
        Some(&optional_timestamp_value(&expected.media_retention_until))
    );
    assert_eq!(
        row.get("properties"),
        Some(&Value::Json(value_json_from_properties(
            &expected.properties
        )))
    );
    assert_eq!(row.get("supply_order"), Some(&Value::Int64(supply_order)));
}

fn collect_rs_files(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read source dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

fn assert_no_unsanctioned_app_side_locks_or_global_state_in_core_sources(context: &str) {
    let crate_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = crate_dir.join("src");
    let allowed_lines = [
        "use std::sync::{Arc, Mutex, OnceLock, Weak};",
        "trace: Arc<Mutex<String>>,",
        "outcome_mutex: Arc<Mutex<()>>,",
        "reindex_events: Arc<Mutex<Vec<ReindexEvent>>>,",
        "trace: Arc::new(Mutex::new(String::new())),",
        "outcome_mutex: Arc::new(Mutex::new(())),",
        "reindex_events: Arc::new(Mutex::new(Vec::new())),",
        "next_seq: Mutex<u64>,",
        "next_seq: Mutex::new(0),",
        "let seq = match self.next_seq.lock() {",
        "let mut guard = self.trace.lock().expect(\"trace lock\");",
        "let guard = self.trace.lock().expect(\"trace lock\");",
        "let mut events = self.reindex_events.lock().expect(\"reindex events\");",
        "static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Weak<Database>>>> = OnceLock::new();",
        "let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));",
        "let mut guard = registry.lock().expect(\"store registry lock\");",
        "let _outcome_guard = self.outcome_mutex.lock().expect(\"outcome mutex poisoned\");",
    ];
    let forbidden = [
        "Mutex",
        "RwLock",
        "OnceLock",
        "LazyLock",
        ".lock(",
        "parking_lot",
        "Atomic",
        "std::sync::atomic",
        "compare_exchange",
        "fetch_update",
        "spin_loop",
        "thread::park",
        "thread_local!",
        "static mut",
        "Box::leak",
    ];
    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);
    for file in files {
        if file == crate_dir.join("src/embedder/embeddinggemma.rs")
            || file == crate_dir.join("src/embedder/nomic.rs")
            || file == crate_dir.join("src/control_transport.rs")
        {
            continue;
        }
        let text = std::fs::read_to_string(&file).expect("core source");
        for line in text.lines() {
            let trimmed = line.trim();
            let global_hash_state = trimmed.contains("static ")
                && (trimmed.contains("HashMap") || trimmed.contains("BTreeMap"));
            if (global_hash_state || forbidden.iter().any(|forbidden| line.contains(forbidden)))
                && !allowed_lines.iter().any(|allowed| line.trim() == *allowed)
            {
                panic!(
                    "{context}: core source must not serialize evidence/decision contention or hold product state with app-side globals in {}: {line}",
                    file.display()
                );
            }
        }
    }
}

fn assert_core_source_has_no_raw_byte_fetch_primitives() {
    let crate_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = crate_dir.join("src");
    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);
    for file in files {
        if file == crate_dir.join("src/embedder/embeddinggemma.rs")
            || file == crate_dir.join("src/embedder/nomic.rs")
            || file == crate_dir.join("src/control_transport.rs")
        {
            continue;
        }
        let text = std::fs::read_to_string(&file).expect("core source");
        for line in text.lines() {
            let trimmed = line.trim();
            let allowed_store_rebuild = file == crate_dir.join("src/store/mod.rs")
                && (trimmed == "if config.db_path.exists() {"
                    || trimmed == "let _ = std::fs::remove_file(&config.db_path);"
                    || trimmed == "let _ = std::fs::create_dir_all(parent);"
                    || trimmed == "if let Ok(mut file) = std::fs::OpenOptions::new()");
            for forbidden in [
                "std::fs",
                "fs::read",
                "fs::File",
                "File::open",
                "OpenOptions",
                "std::io::Read",
                "tokio::fs",
                "read_to_end",
                "read_to_string",
                ".read(",
                "::read(",
                "metadata(",
                ".exists()",
                "remove_file",
                "remove_dir",
                "rename(",
                "copy(",
            ] {
                assert!(
                    !trimmed.contains(forbidden) || allowed_store_rebuild,
                    "core source must not fetch or probe evidence bytes in {}: {line}",
                    file.display()
                );
            }
        }
    }
}

fn assert_evidence_vector_schema_has_no_json_shadow(
    db: &Database,
    required_vectors: &[(&str, usize)],
) {
    let meta = db.table_meta("evidence").expect("evidence table meta");
    let column_type = |name: &str| {
        meta.columns
            .iter()
            .find(|column| column.name == name)
            .map(|column| &column.column_type)
    };
    for (column, dimension) in required_vectors {
        assert_eq!(
            column_type(column),
            Some(&ColumnType::Vector(*dimension)),
            "evidence.{column} must be an engine vector column"
        );
    }
    for forbidden in [
        "embedding",
        "embedding_json",
        "vector",
        "vector_json",
        "vector_text_json",
        "vector_vision_json",
    ] {
        assert!(
            column_type(forbidden).is_none(),
            "evidence.{forbidden} must not exist as a JSON shadow column"
        );
    }
}

fn assert_embedding_handles_have_no_vector_json_shadow(db: &Database) {
    for table in EMBEDDING_HANDLE_TABLES {
        let meta = db.table_meta(table).expect("embedding handle table meta");
        for forbidden in [
            "embedding",
            "embedding_json",
            "vector",
            "vector_json",
            "vector_text_json",
            "vector_vision_json",
        ] {
            assert!(
                meta.columns.iter().all(|column| column.name != forbidden),
                "{table}.{forbidden} must not exist as a vector JSON shadow column"
            );
        }
    }
}

fn assert_first_vector_hit_id(
    db: &Database,
    table: &str,
    hits: &[(RowId, f32)],
    expected: uuid::Uuid,
) {
    assert!(
        !hits.is_empty(),
        "expected at least one vector hit for {table}.id {expected}"
    );
    let row_id = hits[0].0;
    let rows = db
        .scan(table, db.snapshot())
        .expect("scan vector hit table");
    let row = rows
        .into_iter()
        .find(|row| row.row_id == row_id)
        .expect("vector hit row must still be visible");
    assert_eq!(row.values.get("id"), Some(&Value::Uuid(expected)));
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

#[test]
fn record_observation_text_evidence_lands_in_vector_text_hnsw_index_queryable_via_engine_no_json_shim()
 {
    let (_dir, store) = fresh_store();
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let result = store
        .record_observation(record_input(observation_id, evidence_id))
        .expect("record");
    assert_eq!(result.evidence_ids, vec![evidence_id]);
    let db = reopen_database_for_store(store);
    assert_evidence_vector_schema_has_no_json_shadow(&db, &[("vector_text", 768)]);
    assert_embedding_handles_have_no_vector_json_shadow(&db);
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &text_vector(),
            1,
            None,
            db.snapshot(),
        )
        .expect("query");
    assert_first_vector_hit_id(&db, "evidence", &hits, evidence_id.as_uuid());
}

#[test]
fn record_observation_image_evidence_lands_in_vector_vision_hnsw_index_queryable_via_engine_no_json_shim()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..vision_config()
    };
    let store = Store::open(config.clone()).expect("open");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        ContextId::new_v7(),
        vec![image_evidence_ref(observation_id, evidence_id)],
        BTreeMap::new(),
        vec![PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: "vigil_face_reid_v1".to_string(),
            vector: vec![1.0; 512],
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer(),
        }],
    );
    let result = store.record_observation(input).expect("record");
    assert_eq!(result.evidence_ids, vec![evidence_id]);
    let db = reopen_database_for_store(store);
    assert_evidence_vector_schema_has_no_json_shadow(&db, &[("vector_vision", 512)]);
    assert_embedding_handles_have_no_vector_json_shadow(&db);
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_vision"),
            &[1.0; 512],
            1,
            None,
            db.snapshot(),
        )
        .expect("query");
    assert_first_vector_hit_id(&db, "evidence", &hits, evidence_id.as_uuid());
}

#[test]
fn record_observation_vigil_edge_camera_frame_atomic_observation_evidence_snapshot_embedding_one_tx()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..vision_config()
    };
    let store = Store::open(config).expect("open");
    let context_id = store
        .create_context(CreateContext {
            name: "vigil-frame-atomic".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let entity_id = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "camera target".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id,
        })
        .expect("entity")
        .id;
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let input = record_input_with(
        observation_id,
        entity_id,
        context_id,
        vec![image_evidence_ref(observation_id, evidence_id)],
        BTreeMap::from([("seen".to_string(), Some(serde_json::json!(true)))]),
        vec![PrecomputedEmbedding {
            id: embedding_id,
            embedding_space_id: "vigil_face_reid_v1".to_string(),
            vector: vec![1.0; 512],
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer(),
        }],
    );
    let result = store.record_observation(input).expect("record");
    assert_eq!(result.observation_id, observation_id);
    assert_eq!(result.evidence_ids, vec![evidence_id]);
    assert_eq!(result.embedding_ids, vec![embedding_id]);
    let snapshot_id = result
        .snapshot_id
        .expect("state delta must create a snapshot");

    let db = reopen_database_for_store(store);
    assert_eq!(
        count_where(
            &db,
            "observations",
            "id = $id",
            db_params(vec![("id", Value::Uuid(observation_id.as_uuid()))])
        ),
        1
    );
    assert_eq!(
        count_where(
            &db,
            "entity_snapshot_observations",
            "snapshot_id = $snapshot_id AND observation_id = $observation_id AND observation_order = $observation_order",
            db_params(vec![
                ("snapshot_id", Value::Uuid(snapshot_id.as_uuid())),
                ("observation_id", Value::Uuid(observation_id.as_uuid())),
                ("observation_order", Value::Int64(0))
            ])
        ),
        1
    );
    assert_eq!(
        count_where(
            &db,
            "evidence",
            "id = $id AND observation_id = $observation_id",
            db_params(vec![
                ("id", Value::Uuid(evidence_id.as_uuid())),
                ("observation_id", Value::Uuid(observation_id.as_uuid()))
            ])
        ),
        1
    );
    assert_eq!(
        count_where(
            &db,
            "entity_snapshots",
            "id = $id AND entity_id = $entity_id",
            db_params(vec![
                ("id", Value::Uuid(snapshot_id.as_uuid())),
                ("entity_id", Value::Uuid(entity_id.as_uuid()))
            ])
        ),
        1
    );
    assert_eq!(
        count_where(
            &db,
            "evidence_embedding_handles",
            "id = $id AND target_id = $target_id AND embedding_space_id = $embedding_space_id",
            db_params(vec![
                ("id", Value::Uuid(embedding_id.as_uuid())),
                ("target_id", Value::Uuid(evidence_id.as_uuid())),
                (
                    "embedding_space_id",
                    Value::Text("vigil_face_reid_v1".to_string())
                )
            ])
        ),
        1
    );
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_vision"),
            &[1.0; 512],
            1,
            None,
            db.snapshot(),
        )
        .expect("vision query");
    assert_first_vector_hit_id(&db, "evidence", &hits, evidence_id.as_uuid());
}

#[test]
fn record_observation_rejects_inline_data_uri_with_raw_media_not_stored_before_any_db_write() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let mut input = record_input(observation_id, evidence_id);
    input.embeddings[0].id = embedding_id;
    input.evidence[0].source_ref = "data:image/jpeg;base64,AAAA".to_string();
    let result = store.record_observation(input);
    assert!(
        matches!(result, Err(CgError::RawMediaNotStored)),
        "data uri must be rejected before storage, got {result:?}"
    );
    drop(store);
    let db = Database::open(&db_path).expect("raw db");
    assert_eq!(
        count_where(
            &db,
            "observations",
            "id = $id",
            db_params(vec![("id", Value::Uuid(observation_id.as_uuid()))])
        ),
        0
    );
    assert_eq!(
        count_where(
            &db,
            "evidence",
            "id = $id OR source_ref = $source_ref",
            db_params(vec![
                ("id", Value::Uuid(evidence_id.as_uuid())),
                (
                    "source_ref",
                    Value::Text("data:image/jpeg;base64,AAAA".to_string())
                )
            ])
        ),
        0
    );
    assert_eq!(embedding_handle_count_by_id(&db, embedding_id), 0);
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &text_vector(),
            10,
            None,
            db.snapshot(),
        )
        .expect("data uri rollback vector query");
    assert!(
        hits.is_empty(),
        "data uri rejection must not leave a vector index entry"
    );

    let mut evidence_payload = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    evidence_payload.evidence[0].properties.insert(
        "media_payload_base64".to_string(),
        serde_json::json!("AAAA"),
    );
    assert_record_error("inline evidence payload", evidence_payload, |err| {
        matches!(err, CgError::RawMediaNotStored)
    });

    let mut observation_payload = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    observation_payload.properties.insert(
        "raw_media_bytes".to_string(),
        serde_json::json!([0, 1, 2, 3]),
    );
    assert_record_error("inline observation payload", observation_payload, |err| {
        matches!(err, CgError::RawMediaNotStored)
    });

    let mut nested_payload = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    nested_payload.observed_properties.insert(
        "camera".to_string(),
        serde_json::json!({
            "frames": [
                {"metadata": {"raw_media": "data:image/jpeg;base64,BBBB"}}
            ]
        }),
    );
    assert_record_error("nested inline observation payload", nested_payload, |err| {
        matches!(err, CgError::RawMediaNotStored)
    });

    let mut nested_delta_payload = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    nested_delta_payload.state_delta.insert(
        "camera".to_string(),
        Some(serde_json::json!({
            "frames": [
                {"metadata": {"media_payload": "data:image/jpeg;base64,CCCC"}}
            ]
        })),
    );
    assert_record_error(
        "nested inline state delta payload",
        nested_delta_payload,
        |err| matches!(err, CgError::RawMediaNotStored),
    );

    let mut upper_data_uri_payload = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    upper_data_uri_payload.observed_properties.insert(
        "thumbnail".to_string(),
        serde_json::json!("DATA:image/jpeg;base64,DDDD"),
    );
    assert_record_error(
        "uppercase inline data uri payload",
        upper_data_uri_payload,
        |err| matches!(err, CgError::RawMediaNotStored),
    );

    let mut broad_base64_key_payload = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    broad_base64_key_payload.evidence[0]
        .properties
        .insert("image_base64".to_string(), serde_json::json!("EEEE"));
    assert_record_error(
        "broad inline base64 key payload",
        broad_base64_key_payload,
        |err| matches!(err, CgError::RawMediaNotStored),
    );
}

#[test]
fn record_observation_rejects_blob_javascript_vbscript_about_schemes_with_invalid_source_ref_scheme()
 {
    for scheme in [
        "blob:abc",
        "javascript:alert(1)",
        "vbscript:msgbox",
        "about:blank",
    ] {
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let mut input = record_input(observation_id, evidence_id);
        input.evidence[0].source_ref = scheme.to_string();
        assert_record_error(scheme, input, |err| {
            matches!(err, CgError::InvalidSourceRefScheme { .. })
        });
    }
}

#[test]
fn record_observation_text_evidence_with_cas_sha256_source_ref_and_line_range_region_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        https_allowed_hosts: vec!["github.com".to_string(), "slack.com".to_string()],
        ..Default::default()
    };
    let db_path = config.db_path.clone();
    let store = Store::open(config.clone()).expect("open");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let context_id = ContextId::new_v7();
    let mut supplied = evidence_ref(observation_id, evidence_id);
    supplied.context_id = context_id;
    supplied.source_ref =
        "cas://sha256:abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd".to_string();
    supplied.region = Some(serde_json::json!({"line_start": 100, "line_end": 142}));
    supplied.byte_range = Some((25, 91));
    supplied.time_range_ms = Some((1_200, 2_400));
    supplied.media_retention_until = supplied
        .captured_at
        .map(|captured| captured + Duration::days(30));
    supplied
        .properties
        .insert("case_id".to_string(), serde_json::json!("agent-7c4a"));
    supplied.producer = EvidenceProducer {
        system: "agent-session-7c4a".to_string(),
        model_name: "claude-code".to_string(),
        model_version: "1.0".to_string(),
        pipeline_version: "session/2026-05-05".to_string(),
    };
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        context_id,
        vec![supplied.clone()],
        BTreeMap::new(),
        Vec::new(),
    );
    let result = store.record_observation(input).expect("record");
    assert_eq!(result.evidence_ids, vec![evidence_id]);
    drop(store);
    let raw_db = Database::open(&db_path).expect("raw db");
    let raw = raw_evidence_rows_for_observation(&raw_db, observation_id);
    assert_eq!(raw.len(), 1);
    assert_raw_evidence_matches(&raw[0], &supplied, 0);
    drop(raw_db);

    let reopened = Store::open(config).expect("reopen");
    let returned = reopened
        .list_evidence_for_observation(observation_id)
        .expect("evidence");
    assert_eq!(returned.len(), 1);
    assert_eq!(returned[0], supplied);
    assert_eq!(returned[0].id, supplied.id);
    assert_eq!(returned[0].source_ref, supplied.source_ref);
    assert_eq!(returned[0].region, supplied.region);
    assert_eq!(returned[0].producer, supplied.producer);
    assert_eq!(returned[0].retention_status, supplied.retention_status);
}

#[test]
fn record_observation_evidence_supply_order_preserved_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        https_allowed_hosts: vec!["github.com".to_string(), "slack.com".to_string()],
        ..Default::default()
    };
    let db_path = config.db_path.clone();
    let store = Store::open(config.clone()).expect("open");
    let observation_id = ObservationId::new_v7();
    let ids = vec![
        EvidenceId::new_v7(),
        EvidenceId::new_v7(),
        EvidenceId::new_v7(),
    ];
    let mut evidence = ids
        .iter()
        .map(|id| evidence_ref(observation_id, *id))
        .collect::<Vec<_>>();
    evidence[0].source_ref = "https://github.com/acme/repo/pull/42".to_string();
    evidence[1].source_ref = "https://slack.com/archives/C456/p789".to_string();
    evidence[2] = image_evidence_ref(observation_id, ids[2]);
    evidence[2].source_ref = "s3://wiki/whiteboard.png".to_string();
    let context_id = ContextId::new_v7();
    set_evidence_context(&mut evidence, context_id);
    let expected_evidence = evidence.clone();
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        context_id,
        evidence,
        BTreeMap::new(),
        Vec::new(),
    );
    let result = store.record_observation(input).expect("record");
    assert_eq!(result.evidence_ids, ids);
    drop(store);
    let raw_db = Database::open(&db_path).expect("raw db");
    let raw = raw_evidence_rows_for_observation(&raw_db, observation_id);
    assert_eq!(raw.len(), 3);
    assert_eq!(
        raw.iter()
            .map(|row| value_uuid(&row["id"]))
            .collect::<Vec<_>>(),
        ids.iter().map(|id| id.as_uuid()).collect::<Vec<_>>()
    );
    assert_eq!(
        raw.iter()
            .map(|row| value_text(&row["source_ref"]).to_string())
            .collect::<Vec<_>>(),
        vec![
            "https://github.com/acme/repo/pull/42".to_string(),
            "https://slack.com/archives/C456/p789".to_string(),
            "s3://wiki/whiteboard.png".to_string(),
        ]
    );
    for (idx, row) in raw.iter().enumerate() {
        assert_raw_evidence_matches(row, &expected_evidence[idx], idx as i64);
    }
    drop(raw_db);

    let returned = Store::open(config)
        .expect("reopen")
        .list_evidence_for_observation(observation_id)
        .expect("evidence");
    assert_eq!(returned.iter().map(|e| e.id).collect::<Vec<_>>(), ids);
    assert_eq!(
        returned
            .iter()
            .map(|e| e.source_ref.clone())
            .collect::<Vec<_>>(),
        vec![
            "https://github.com/acme/repo/pull/42".to_string(),
            "https://slack.com/archives/C456/p789".to_string(),
            "s3://wiki/whiteboard.png".to_string(),
        ]
    );
}

#[test]
fn record_observation_derived_from_links_within_same_context_chain_preserved() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    };
    let db_path = config.db_path.clone();
    let store = Store::open(config.clone()).expect("open");
    let observation_id = ObservationId::new_v7();
    let image_id = EvidenceId::new_v7();
    let caption_id = EvidenceId::new_v7();
    let mut caption = evidence_ref(observation_id, caption_id);
    caption.derived_from = Some(image_id);
    caption.text_projection = Some("whiteboard decision caption".to_string());
    caption.properties.insert(
        "projection_kind".to_string(),
        serde_json::json!("ocr-caption"),
    );
    let mut image = image_evidence_ref(observation_id, image_id);
    let context_id = ContextId::new_v7();
    set_evidence_context(std::slice::from_mut(&mut image), context_id);
    set_evidence_context(std::slice::from_mut(&mut caption), context_id);
    let expected_caption = caption.clone();
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        context_id,
        vec![image.clone(), expected_caption.clone()],
        BTreeMap::new(),
        Vec::new(),
    );
    let result = store.record_observation(input).expect("record");
    assert_eq!(result.evidence_ids, vec![image_id, caption_id]);
    let returned = store
        .list_evidence_for_observation(observation_id)
        .expect("evidence");
    assert_eq!(
        returned.iter().map(|e| e.id).collect::<Vec<_>>(),
        vec![image_id, caption_id]
    );
    let caption = returned
        .iter()
        .find(|e| e.id == caption_id)
        .expect("caption evidence");
    assert_eq!(caption.derived_from, Some(image_id));
    drop(store);
    let raw_db = Database::open(&db_path).expect("raw db");
    let raw = raw_evidence_rows_for_observation(&raw_db, observation_id);
    let raw_caption = raw
        .iter()
        .find(|row| value_uuid(&row["id"]) == caption_id.as_uuid());
    assert!(
        raw_caption.is_some(),
        "raw evidence table must contain the caption row"
    );
    assert_eq!(
        raw_caption.unwrap().get("derived_from"),
        Some(&Value::Uuid(image_id.as_uuid()))
    );
    assert_raw_evidence_matches(&raw[0], &image, 0);
    assert_raw_evidence_matches(raw_caption.unwrap(), &expected_caption, 1);
    drop(raw_db);
    let reopened = Store::open(config).expect("reopen");
    let reopened_rows = reopened
        .list_evidence_for_observation(observation_id)
        .expect("reopened evidence");
    let reopened_caption = reopened_rows
        .iter()
        .find(|e| e.id == caption_id)
        .expect("reopened caption evidence");
    assert_eq!(reopened_caption.derived_from, Some(image_id));
}

#[test]
fn record_observation_derived_from_cross_context_rejected_with_invalid_derived_from_reference_before_db_write()
 {
    let (_dir, store) = fresh_store();
    let ctx_a = store
        .create_context(CreateContext {
            name: "derived-parent-context".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("ctx a")
        .id;
    let entity_a = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "parent".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx_a,
        })
        .expect("entity a")
        .id;
    let parent_observation = ObservationId::new_v7();
    let parent = EvidenceId::new_v7();
    store
        .record_observation(record_input_with(
            parent_observation,
            entity_a,
            ctx_a,
            vec![image_evidence_ref(parent_observation, parent)],
            BTreeMap::new(),
            Vec::new(),
        ))
        .expect("parent evidence");

    let ctx_b = store
        .create_context(CreateContext {
            name: "derived-child-context".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("ctx b")
        .id;
    let entity_b = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "child".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx_b,
        })
        .expect("entity b")
        .id;
    let observation_id = ObservationId::new_v7();
    let child = EvidenceId::new_v7();
    let mut evidence = evidence_ref(observation_id, child);
    evidence.derived_from = Some(parent);
    let input = record_input_with(
        observation_id,
        entity_b,
        ctx_b,
        vec![evidence],
        BTreeMap::new(),
        Vec::new(),
    );
    let result = store.record_observation(input);
    assert!(
        matches!(result, Err(CgError::InvalidDerivedFromReference)),
        "cross-context derived_from must be rejected, got {result:?}"
    );
    assert!(store.get_observation(observation_id).is_err());
    assert!(
        store
            .list_evidence_for_observation(observation_id)
            .expect("child evidence")
            .is_empty()
    );
}

#[test]
fn record_observation_derived_from_within_batch_a_b_a_cycle_rejected_with_invalid_derived_from_reference()
 {
    let observation_id = ObservationId::new_v7();
    let a = EvidenceId::new_v7();
    let b = EvidenceId::new_v7();
    let mut ea = evidence_ref(observation_id, a);
    let mut eb = evidence_ref(observation_id, b);
    ea.derived_from = Some(b);
    eb.derived_from = Some(a);
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        ContextId::new_v7(),
        vec![ea, eb],
        BTreeMap::new(),
        Vec::new(),
    );
    assert_record_error("derived_from cycle", input, |err| {
        matches!(err, CgError::InvalidDerivedFromReference)
    });
}

#[test]
fn record_observation_self_cycle_derived_from_rejected() {
    let observation_id = ObservationId::new_v7();
    let id = EvidenceId::new_v7();
    let mut evidence = evidence_ref(observation_id, id);
    evidence.derived_from = Some(id);
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        ContextId::new_v7(),
        vec![evidence],
        BTreeMap::new(),
        Vec::new(),
    );
    assert_record_error("self-cycle", input, |err| {
        matches!(err, CgError::InvalidDerivedFromReference)
    });
}

#[test]
fn record_observation_invalid_region_bounds_rejected_with_invalid_evidence_region_bounds_valid_region_accepted()
 {
    for (label, region, expected_field) in [
        (
            "negative x",
            serde_json::json!({"x": -0.1, "y": 0.1, "w": 0.3, "h": 0.3}),
            Some("x"),
        ),
        (
            "zero width",
            serde_json::json!({"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.3}),
            Some("w"),
        ),
        (
            "non numeric width",
            serde_json::json!({"x": 0.1, "y": 0.1, "w": "NaN", "h": 0.3}),
            Some("w"),
        ),
        (
            "out of bounds",
            serde_json::json!({"x": 0.6, "y": 0.1, "w": 0.5, "h": 0.3}),
            Some("x"),
        ),
        (
            "inverted",
            serde_json::json!({"x": 0.5, "y": 0.5, "w": -0.3, "h": -0.3}),
            Some("w"),
        ),
        (
            "valid image box",
            serde_json::json!({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}),
            None,
        ),
    ] {
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let mut input = record_input(observation_id, evidence_id);
        input.evidence[0].region = Some(region);
        match expected_field {
            Some(field) => assert_record_failure_zero_rollback(label, input, |result| {
                matches!(
                    result,
                    Err(CgError::InvalidEvidenceRegionBounds { field: got }) if *got == field
                )
            }),
            None => {
                let (_dir, store) = fresh_store();
                assert_eq!(
                    store.record_observation(input).expect(label).evidence_ids,
                    vec![evidence_id]
                );
            }
        }
    }
}

#[test]
fn record_observation_invalid_line_range_region_rejected_before_db_write() {
    let mut inverted = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    inverted.evidence[0].region = Some(serde_json::json!({
        "line_start": 142,
        "line_end": 100,
    }));
    assert_record_error("inverted line range", inverted, |err| {
        matches!(
            err,
            CgError::InvalidEvidenceRegionBounds { field: "line_end" }
        )
    });

    let mut negative = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    negative.evidence[0].region = Some(serde_json::json!({
        "line_start": -1,
        "line_end": 2,
    }));
    assert_record_error("negative line range", negative, |err| {
        matches!(
            err,
            CgError::InvalidEvidenceRegionBounds {
                field: "line_start"
            }
        )
    });
}

#[test]
fn record_observation_embedding_validation_failure_does_not_auto_create_context_or_entity() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let mut input = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    let context_id = input.context_id;
    let entity_id = input.entity_id;
    input.embeddings[0].vector = vec![0.0; 384];
    let result = store.record_observation(input);
    assert!(matches!(
        result,
        Err(CgError::EmbeddingDimensionMismatch {
            space_id,
            expected: 768,
            found: 384
        }) if space_id == TEXT_SPACE
    ));
    drop(store);

    let db = Database::open(&db_path).expect("raw db");
    assert_eq!(
        count_where(
            &db,
            "contexts",
            "id = $id",
            db_params(vec![("id", Value::Uuid(context_id.as_uuid()))])
        ),
        0,
        "failed observation must not auto-create context"
    );
    assert_eq!(
        count_where(
            &db,
            "entities",
            "id = $id",
            db_params(vec![("id", Value::Uuid(entity_id.as_uuid()))])
        ),
        0,
        "failed observation must not auto-create entity"
    );
}

#[test]
fn record_observation_duplicate_observation_id_rolls_back_auto_created_context_and_entity() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let committed_observation = ObservationId::new_v7();
    let committed_evidence = EvidenceId::new_v7();
    store
        .record_observation(record_input(committed_observation, committed_evidence))
        .expect("seed observation");

    let leaked_context = ContextId::new_v7();
    let leaked_entity = EntityId::new_v7();
    let result = store.record_observation(record_input_with(
        committed_observation,
        leaked_entity,
        leaked_context,
        vec![evidence_ref(committed_observation, EvidenceId::new_v7())],
        BTreeMap::from([("region".to_string(), Some(serde_json::json!("eu-west-1")))]),
        Vec::new(),
    ));
    assert!(matches!(result, Err(CgError::UniqueViolation { .. })));
    drop(store);

    let db = Database::open(&db_path).expect("raw db");
    assert_eq!(
        count_where(
            &db,
            "contexts",
            "id = $id",
            db_params(vec![("id", Value::Uuid(leaked_context.as_uuid()))])
        ),
        0,
        "failed observation must roll back auto-created context"
    );
    assert_eq!(
        count_where(
            &db,
            "entities",
            "id = $id",
            db_params(vec![("id", Value::Uuid(leaked_entity.as_uuid()))])
        ),
        0,
        "failed observation must roll back auto-created entity"
    );
}

#[test]
fn record_observation_invalid_content_hash_rejected_with_invalid_evidence_hash_valid_hash_accepted()
{
    for (hash, expected_ok) in [
        (
            "sha512:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            false,
        ),
        (
            "sha256:NOTHEX_NOTHEX_NOTHEX_NOTHEX_NOTHEX_NOTHEX_NOTHEX_NOTHEX_NOTHEXX0",
            false,
        ),
        ("sha256:abc", false),
        ("", false),
        (
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            true,
        ),
    ] {
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let mut input = record_input(observation_id, evidence_id);
        input.evidence[0].content_hash = Some(hash.to_string());
        if expected_ok {
            let (_dir, store) = fresh_store();
            let result = store.record_observation(input);
            assert_eq!(result.expect(hash).evidence_ids, vec![evidence_id]);
        } else {
            assert_record_error(
                hash,
                input,
                |err| matches!(err, CgError::InvalidEvidenceHash { hash: got } if got == hash),
            );
        }
    }
}

#[test]
fn record_observation_empty_evidence_vec_rejected_with_missing_evidence_ref() {
    let mut input = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    input.evidence.clear();
    assert_record_error("empty evidence", input, |err| {
        matches!(err, CgError::MissingEvidenceRef)
    });
}

#[test]
fn record_observation_mime_matrix_per_evidence_kind_enforced() {
    for (kind, mime, expected_ok) in [
        (EvidenceKind::Text, None, true),
        (EvidenceKind::Text, Some("application/json"), true),
        (EvidenceKind::StructuredSignal, None, true),
        (
            EvidenceKind::StructuredSignal,
            Some("application/json"),
            true,
        ),
        (EvidenceKind::ImageFrame, None, false),
        (EvidenceKind::ImageFrame, Some("image/png"), true),
        (EvidenceKind::ImageFrame, Some("text/plain"), false),
        (EvidenceKind::AudioSegment, None, false),
        (EvidenceKind::AudioSegment, Some("audio/wav"), true),
        (EvidenceKind::VideoSegment, None, false),
        (EvidenceKind::VideoSegment, Some("video/mp4"), true),
        (EvidenceKind::Text, Some("application/octet-stream"), false),
        (
            EvidenceKind::ImageFrame,
            Some("application/octet-stream"),
            false,
        ),
        (
            EvidenceKind::StructuredSignal,
            Some("application/octet-stream"),
            false,
        ),
    ] {
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let mut evidence = match kind {
            EvidenceKind::StructuredSignal => structured_evidence_ref(observation_id, evidence_id),
            EvidenceKind::ImageFrame => image_evidence_ref(observation_id, evidence_id),
            _ => evidence_ref(observation_id, evidence_id),
        };
        evidence.kind = kind;
        evidence.mime_type = mime.map(str::to_string);
        let input = record_input_with(
            observation_id,
            EntityId::new_v7(),
            ContextId::new_v7(),
            vec![evidence],
            BTreeMap::new(),
            Vec::new(),
        );
        if expected_ok {
            let (_dir, store) = fresh_store();
            let result = store.record_observation(input);
            assert_eq!(
                result.expect("accepted mime").evidence_ids,
                vec![evidence_id]
            );
        } else {
            assert_record_error(&format!("{kind:?}/{mime:?}"), input, |err| {
                matches!(err, CgError::InvalidMimeForEvidenceKind)
            });
        }
    }
}

#[test]
fn record_observation_structured_signal_terraform_webhook_state_delta_round_trips_losslessly() {
    let (_dir, store) = fresh_store();
    let context_id = store
        .create_context(CreateContext {
            name: "terraform-state".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let entity_id = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "terraform resource".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id,
        })
        .expect("entity")
        .id;
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let input = record_input_with(
        observation_id,
        entity_id,
        context_id,
        vec![structured_evidence_ref(observation_id, evidence_id)],
        BTreeMap::from([("region".to_string(), Some(serde_json::json!("eu-west-1")))]),
        Vec::new(),
    );
    let result = store.record_observation(input).expect("record");
    assert_eq!(result.evidence_ids, vec![evidence_id]);
    assert_eq!(
        store
            .get_observation(observation_id)
            .expect("observation")
            .state_delta
            .get("region"),
        Some(&Some(serde_json::json!("eu-west-1")))
    );
    let snapshot_id = result
        .snapshot_id
        .expect("structured state delta must create snapshot");
    let snapshot = store
        .entity_history(entity_id)
        .expect("history")
        .into_iter()
        .find(|snapshot| snapshot.id == snapshot_id)
        .expect("snapshot");
    assert_eq!(
        snapshot.properties.get("region"),
        Some(&serde_json::json!("eu-west-1"))
    );
}

#[test]
fn record_observation_state_delta_no_op_merge_does_not_create_snapshot_real_change_does() {
    let (_dir, store) = fresh_store();
    let context_id = store
        .create_context(CreateContext {
            name: "state-delta-no-op".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let entity_id = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "stateful service".to_string(),
            properties: BTreeMap::from([("region".to_string(), serde_json::json!("us-east-1"))]),
            tags: vec![],
            context_id,
        })
        .expect("entity")
        .id;
    let history_before = store.entity_history(entity_id).expect("history").len();

    let no_op_observation = ObservationId::new_v7();
    let no_op_evidence = EvidenceId::new_v7();
    let no_op = record_input_with(
        no_op_observation,
        entity_id,
        context_id,
        vec![evidence_ref(no_op_observation, no_op_evidence)],
        BTreeMap::from([("region".to_string(), Some(serde_json::json!("us-east-1")))]),
        Vec::new(),
    );
    let no_op_result = store.record_observation(no_op).expect("no-op record");
    assert_eq!(no_op_result.snapshot_id, None);
    assert_eq!(
        store.entity_history(entity_id).expect("history").len(),
        history_before,
        "no-op state merge must not create a snapshot"
    );

    let changed_observation = ObservationId::new_v7();
    let changed_evidence = EvidenceId::new_v7();
    let changed = record_input_with(
        changed_observation,
        entity_id,
        context_id,
        vec![evidence_ref(changed_observation, changed_evidence)],
        BTreeMap::from([("region".to_string(), Some(serde_json::json!("eu-west-1")))]),
        Vec::new(),
    );
    let changed_result = store.record_observation(changed).expect("changed record");
    assert!(
        changed_result.snapshot_id.is_some(),
        "real state change must create a snapshot"
    );
    let history_after = store.entity_history(entity_id).expect("history");
    assert_eq!(
        history_after.len(),
        history_before + 1,
        "real state change must add exactly one snapshot"
    );
    assert_eq!(
        history_after
            .last()
            .expect("latest snapshot")
            .properties
            .get("region"),
        Some(&serde_json::json!("eu-west-1"))
    );
}

#[test]
fn record_observation_state_delta_real_change_creates_snapshot_with_evidenced_by_pointing_back() {
    let (_dir, store) = fresh_store();
    let context_id = store
        .create_context(CreateContext {
            name: "state-delta-provenance".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let entity_id = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "stateful service".to_string(),
            properties: BTreeMap::from([("region".to_string(), serde_json::json!("us-east-1"))]),
            tags: vec![],
            context_id,
        })
        .expect("entity")
        .id;
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let result = store
        .record_observation(record_input_with(
            observation_id,
            entity_id,
            context_id,
            vec![evidence_ref(observation_id, evidence_id)],
            BTreeMap::from([("region".to_string(), Some(serde_json::json!("eu-west-1")))]),
            Vec::new(),
        ))
        .expect("record");
    let snapshot_id = result
        .snapshot_id
        .expect("real state change must create snapshot");
    let snapshot = store
        .entity_history(entity_id)
        .expect("history")
        .into_iter()
        .find(|snapshot| snapshot.id == snapshot_id)
        .expect("new snapshot");
    assert_eq!(snapshot.entity_id, entity_id);
    assert_eq!(snapshot.evidenced_by, vec![observation_id]);
}

#[test]
fn record_observation_partial_evidence_failure_rolls_back_observation_and_snapshot_and_embedding() {
    let mut input = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    input.evidence.push(EvidenceRef {
        id: EvidenceId::new_v7(),
        observation_id: input.id,
        source_ref: "javascript:alert(1)".to_string(),
        ..input.evidence[0].clone()
    });
    assert_record_zero_rollback("partial evidence rollback", input);
}

#[test]
fn record_observation_concurrent_cross_handle_appends_immutable_snapshot_chain() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let seed = Store::open(config.clone()).expect("open");
    let context_id = seed
        .create_context(CreateContext {
            name: "a20-concurrent".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let entity = seed
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "shared entity".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id,
        })
        .expect("entity")
        .id;
    drop(seed);

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
    let writers: Vec<_> = (0..4)
        .map(|writer| {
            let config = config.clone();
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                let store = Store::open(config).expect("writer open");
                let obs = ObservationId::new_v7();
                let eid = EvidenceId::new_v7();
                let input = record_input_with(
                    obs,
                    entity,
                    context_id,
                    vec![evidence_ref(obs, eid)],
                    BTreeMap::from([(format!("writer_{writer}"), Some(serde_json::json!(writer)))]),
                    Vec::new(),
                );
                barrier.wait();
                store.record_observation(input)
            })
        })
        .collect();
    let results = writers
        .into_iter()
        .map(|writer| writer.join().expect("writer joined"))
        .collect::<Vec<_>>();
    assert!(
        results.iter().all(Result::is_ok),
        "all cross-handle writers must commit"
    );
    assert!(
        results
            .iter()
            .all(|result| result.as_ref().expect("record").snapshot_id.is_some()),
        "each writer must create one snapshot"
    );

    let reader = Store::open(config).expect("reader open");
    let history = reader.entity_history(entity).expect("history");
    assert_eq!(history.len(), 5);
    assert!(history.iter().all(|snapshot| snapshot.valid_to.is_none()));
    let current = history.last().expect("current snapshot");
    let graph = reader.entity_history_graph(entity).expect("history graph");
    assert_eq!(graph.links.len(), 4);
    assert_eq!(graph.branch_heads, vec![current.id]);
    for writer in 0..4 {
        assert_eq!(
            current.properties.get(&format!("writer_{writer}")),
            Some(&serde_json::json!(writer)),
            "the serialized snapshot chain retains writer {writer}'s delta"
        );
    }
    let txids = history
        .iter()
        .map(|snapshot| snapshot.valid_from.0)
        .collect::<HashSet<_>>();
    assert_eq!(txids.len(), history.len());
    let source = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/store/observation.rs"),
    )
    .expect("observation source");
    let start = source
        .find("pub fn record_observation")
        .expect("record_observation source");
    let end = source[start..]
        .find("\n    pub fn ")
        .map(|offset| start + offset)
        .unwrap_or(source.len());
    let record_source = &source[start..end];
    for forbidden in ["Mutex", "OnceLock", "LazyLock", ".lock(", "outcome_mutex"] {
        assert!(
            !record_source.contains(forbidden),
            "record_observation must not serialize contention with app-side lock {forbidden}"
        );
    }
    assert_no_unsanctioned_app_side_locks_or_global_state_in_core_sources(
        "record_observation contention",
    );
}

#[test]
fn record_observation_evidence_text_projection_does_not_replace_original_kind() {
    let (_dir, store) = fresh_store();
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let mut evidence = image_evidence_ref(observation_id, evidence_id);
    let source_ref = evidence.source_ref.clone();
    let text_projection = Some("whiteboard caption".to_string());
    evidence.text_projection = text_projection.clone();
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        ContextId::new_v7(),
        vec![evidence],
        BTreeMap::new(),
        Vec::new(),
    );
    let result = store.record_observation(input).expect("record");
    assert_eq!(result.evidence_ids, vec![evidence_id]);
    let returned = store
        .list_evidence_for_observation(observation_id)
        .expect("evidence");
    assert_eq!(returned.len(), 1);
    assert_eq!(returned[0].kind, EvidenceKind::ImageFrame);
    assert_eq!(returned[0].text_projection, text_projection);
    assert_eq!(returned[0].source_ref, source_ref);
}

#[test]
fn record_observation_returns_recordobservationresult_with_evidence_ids_in_supply_order() {
    let observation_id = ObservationId::new_v7();
    let first = EvidenceId::new_v7();
    let second = EvidenceId::new_v7();
    let third = EvidenceId::new_v7();
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        ContextId::new_v7(),
        vec![
            evidence_ref(observation_id, first),
            evidence_ref(observation_id, second),
            evidence_ref(observation_id, third),
        ],
        BTreeMap::new(),
        Vec::new(),
    );
    assert_evidence_ids_round_trip("result evidence_ids", input, vec![first, second, third]);
}

#[test]
fn record_observation_embedding_target_evidence_id_not_in_batch_rejected_with_embedding_target_not_found()
 {
    let unknown = EvidenceId::new_v7();
    let mut input = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    input.embeddings[0].target = EmbeddingTarget::Evidence(unknown);
    assert_record_error(
        "embedding target",
        input,
        |err| matches!(err, CgError::EmbeddingTargetNotFound { id } if *id == unknown),
    );
}

#[test]
fn record_observation_embedding_dimension_mismatch_rejected_with_embedding_space_mismatch_naming_field()
 {
    let mut input = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    input.embeddings[0].vector = vec![0.0; 384];
    assert_record_error(
        "embedding dimension",
        input,
        |err| matches!(err, CgError::EmbeddingDimensionMismatch { space_id, expected: 768, found: 384 } if space_id == TEXT_SPACE),
    );
}

#[test]
fn record_observation_per_table_vector_column_routing_image_does_not_pollute_text_index_no_json_shim()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..vision_config()
    };
    let store = Store::open(config.clone()).expect("open");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        ContextId::new_v7(),
        vec![image_evidence_ref(observation_id, evidence_id)],
        BTreeMap::new(),
        vec![PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: "vigil_face_reid_v1".to_string(),
            vector: vec![1.0; 512],
            target: EmbeddingTarget::Evidence(evidence_id),
            produced_by: producer(),
        }],
    );
    let result = store.record_observation(input).expect("record");
    assert_eq!(result.evidence_ids, vec![evidence_id]);
    let db = reopen_database_for_store(store);
    assert_evidence_vector_schema_has_no_json_shadow(
        &db,
        &[("vector_text", 768), ("vector_vision", 512)],
    );
    let vision_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_vision"),
            &[1.0; 512],
            1,
            None,
            db.snapshot(),
        )
        .expect("vision query");
    assert_first_vector_hit_id(&db, "evidence", &vision_hits, evidence_id.as_uuid());
    let text_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &text_vector(),
            10,
            None,
            db.snapshot(),
        )
        .expect("text query");
    assert!(
        text_hits.is_empty(),
        "image evidence must not pollute evidence.vector_text"
    );
}

#[test]
fn replay_returns_original_image_frame_evidence_ref_byte_equal_when_text_projection_is_set() {
    let (_dir, store) = fresh_store();
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let mut supplied = image_evidence_ref(observation_id, evidence_id);
    supplied.source_ref = "s3://wiki/whiteboard.png".to_string();
    supplied.content_hash =
        Some("sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string());
    supplied.region = Some(serde_json::json!({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}));
    supplied.text_projection = Some("a whiteboard with retry logic diagram".to_string());
    let (decision, observation_id, expected_evidence) = seed_decision_with_evidence(
        &store,
        vec![supplied.clone()],
        BTreeMap::from([("seen".to_string(), Some(serde_json::json!(true)))]),
    );
    let entity = store
        .get_observation(observation_id)
        .expect("observation")
        .entity_id;
    store
        .update_entity(
            entity,
            EntityPatch {
                properties: Some(BTreeMap::from([
                    ("region".to_string(), serde_json::json!("us-east-1")),
                    ("seen".to_string(), serde_json::json!(false)),
                ])),
                ..Default::default()
            },
        )
        .expect("update current");
    let supplied = expected_evidence
        .into_iter()
        .next()
        .expect("expected evidence");
    let replay = store.replay(decision).expect("replay");
    assert_eq!(replay.decision_id, decision);
    assert!(replay.input_diffs.iter().any(|diff| diff.changed));
    let returned = replay_evidence_ref(&replay, evidence_id);
    assert_eq!(returned, supplied);
    assert_eq!(returned.kind, EvidenceKind::ImageFrame);
    assert_eq!(returned.source_ref, "s3://wiki/whiteboard.png");
    assert_eq!(returned.content_hash, supplied.content_hash);
    assert_eq!(returned.region, supplied.region);
    assert_eq!(returned.producer, supplied.producer);
    assert_eq!(
        returned.text_projection,
        Some("a whiteboard with retry logic diagram".to_string())
    );
}

#[test]
fn replay_lineage_walk_visits_entity_snapshot_evidenced_by_observation_evidence_for_three_evidence_observation()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        https_allowed_hosts: vec!["github.com".to_string(), "slack.com".to_string()],
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open");
    let observation = ObservationId::new_v7();
    let pr = EvidenceId::new_v7();
    let slack = EvidenceId::new_v7();
    let whiteboard = EvidenceId::new_v7();
    let mut pr_text = evidence_ref(observation, pr);
    pr_text.source_ref = "https://github.com/acme/repo/pull/42".to_string();
    pr_text.content_hash =
        Some("sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string());
    pr_text.byte_range = Some((4, 128));
    pr_text.retention_status = RetentionStatus::NotStored;
    pr_text.producer.system = "github-pr-ingester".to_string();
    let mut slack_text = evidence_ref(observation, slack);
    slack_text.source_ref = "https://slack.com/archives/C456/p789".to_string();
    slack_text.content_hash =
        Some("sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string());
    slack_text.text_projection = Some("incident summary from slack".to_string());
    slack_text.retention_status = RetentionStatus::Ephemeral;
    slack_text.producer.system = "slack-export".to_string();
    let mut whiteboard_image = image_evidence_ref(observation, whiteboard);
    whiteboard_image.source_ref = "s3://wiki/whiteboard.png".to_string();
    whiteboard_image.region = Some(serde_json::json!({"x": 0.2, "y": 0.3, "w": 0.4, "h": 0.5}));
    whiteboard_image.frame_index = Some(8);
    whiteboard_image.producer.system = "whiteboard-capture".to_string();
    let (decision, _, expected) = seed_decision_with_evidence(
        &store,
        vec![pr_text, slack_text, whiteboard_image],
        BTreeMap::from([("region".to_string(), Some(serde_json::json!("eu-west-1")))]),
    );
    let replay = store.replay(decision).expect("replay");
    assert_eq!(replay.decision_id, decision);
    let evidence = replay_evidence_refs(&replay);
    assert_eq!(
        evidence.iter().map(|e| e.id).collect::<Vec<_>>(),
        vec![pr, slack, whiteboard]
    );
    assert_eq!(replay.evidence_lineage.len(), 1);
    assert_eq!(replay.evidence_lineage[0].observation_id, Some(observation));
    assert_eq!(
        evidence.iter().map(|e| e.kind).collect::<Vec<_>>(),
        vec![
            EvidenceKind::Text,
            EvidenceKind::Text,
            EvidenceKind::ImageFrame
        ]
    );
    assert_eq!(evidence, expected);
}

#[test]
fn replay_survives_media_expiry_returns_lineage_with_retention_status_expired() {
    let (_dir, store) = fresh_store();
    let (decision, _obs, evidence) = seed_decision(&store);
    store
        .update_evidence_retention_status(evidence, RetentionStatus::Expired)
        .expect("retention");
    let replay = store.replay(decision).expect("replay");
    assert!(!replay.input_diffs.is_empty());
    assert_eq!(
        replay_evidence_ref(&replay, evidence).retention_status,
        RetentionStatus::Expired
    );
}

#[test]
fn replay_survives_media_unavailable_and_not_stored_returns_lineage_with_status() {
    for status in [
        RetentionStatus::NotStored,
        RetentionStatus::Ephemeral,
        RetentionStatus::RetainedExternal,
        RetentionStatus::Expired,
        RetentionStatus::Unavailable,
    ] {
        let (_dir, store) = fresh_store();
        let (decision, _obs, evidence) = seed_decision(&store);
        store
            .update_evidence_retention_status(evidence, status)
            .expect("retention");
        let replay = store.replay(decision).expect("replay");
        assert_eq!(
            replay_evidence_ref(&replay, evidence).retention_status,
            status
        );
    }
}

#[test]
fn replay_returns_evidence_producer_and_region_byte_equal_to_record_observation_input() {
    let (_dir, store) = fresh_store();
    let (decision, _obs, evidence) = seed_decision(&store);
    let replay = store.replay(decision).expect("replay");
    let returned = replay_evidence_ref(&replay, evidence);
    assert_eq!(
        returned.region,
        Some(serde_json::json!({"line_start": 1, "line_end": 2}))
    );
    assert_eq!(returned.producer, producer());
}

#[test]
fn replay_returns_cas_sha256_source_ref_and_line_range_byte_equal() {
    let (_dir, store) = fresh_store();
    let (decision, _obs, evidence) = seed_decision(&store);
    let replay = store.replay(decision).expect("replay");
    let returned = replay_evidence_ref(&replay, evidence);
    assert_eq!(
        returned.source_ref,
        "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
    assert_eq!(
        returned.region,
        Some(serde_json::json!({"line_start": 1, "line_end": 2}))
    );
}

#[test]
fn replay_preserves_multiple_basis_snapshots_for_same_entity_without_map_collapse() {
    let (_dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "same entity basis history".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "same-entity".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");

    let obs_a = ObservationId::new_v7();
    let ev_a = EvidenceId::new_v7();
    let snapshot_a = store
        .record_observation(record_input_with(
            obs_a,
            entity.id,
            ctx,
            vec![evidence_ref(obs_a, ev_a)],
            BTreeMap::from([("phase".to_string(), Some(serde_json::json!("a")))]),
            Vec::new(),
        ))
        .expect("obs a")
        .snapshot_id
        .expect("snapshot a");
    let obs_b = ObservationId::new_v7();
    let ev_b = EvidenceId::new_v7();
    let snapshot_b = store
        .record_observation(record_input_with(
            obs_b,
            entity.id,
            ctx,
            vec![evidence_ref(obs_b, ev_b)],
            BTreeMap::from([("phase".to_string(), Some(serde_json::json!("b")))]),
            Vec::new(),
        ))
        .expect("obs b")
        .snapshot_id
        .expect("snapshot b");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "same-entity-basis".to_string(),
            description: "pin two historical basis snapshots".to_string(),
            intention_ids: vec![intention.id],
            based_on_snapshots: vec![snapshot_a, snapshot_b],
            context_id: ctx,
            ..Default::default()
        })
        .expect("decision");

    let replay = store.replay(decision.id).expect("replay");
    assert_eq!(
        replay
            .evidence_lineage
            .iter()
            .map(|lineage| lineage.basis_snapshot_id)
            .collect::<Vec<_>>(),
        vec![snapshot_a, snapshot_b]
    );
    assert_eq!(replay_evidence_ids(&replay), vec![ev_a, ev_b]);
    assert_eq!(replay.input_diffs.len(), 2);
}

#[test]
fn snapshot_only_decision_basis_persists_based_on_graph_edges_for_create_and_supersede() {
    let (_dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "snapshot-only graph basis".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "snapshot-only-entity".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    let basis = store
        .current_entity_snapshot(entity.id)
        .expect("current")
        .expect("basis");

    let decision = store
        .create_decision(CreateDecision {
            decision_type: "snapshot-only".to_string(),
            description: "create from snapshot id only".to_string(),
            intention_ids: vec![intention.id],
            based_on_snapshots: vec![basis.id],
            context_id: ctx,
            ..Default::default()
        })
        .expect("decision");
    assert!(
        store
            .decision_targets_entity(decision.id, RelationshipType::BasedOn, entity.id)
            .expect("based_on edge")
    );
    assert_eq!(
        store
            .by_entity_decisions(entity.id, DiscoveryFilter::default())
            .expect("decisions by entity")
            .iter()
            .map(|decision| decision.id)
            .collect::<Vec<_>>(),
        vec![decision.id]
    );

    let replacement = store
        .supersede_decision(
            decision.id,
            CreateDecision {
                decision_type: "snapshot-only".to_string(),
                description: "supersede from snapshot id only".to_string(),
                intention_ids: vec![intention.id],
                based_on_snapshots: vec![basis.id],
                context_id: ctx,
                ..Default::default()
            },
        )
        .expect("supersede");
    assert!(
        store
            .decision_targets_entity(replacement.id, RelationshipType::BasedOn, entity.id)
            .expect("replacement based_on edge")
    );
}

#[test]
fn replay_under_concurrent_cross_handle_writer_against_same_entity_returns_single_txid_consistent_basis()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let seed = Store::open(config.clone()).expect("open");
    let (decision, obs, _evidence) = seed_decision(&seed);
    let seeded_observation = seed.get_observation(obs).expect("observation");
    let entity = seeded_observation.entity_id;
    let context_id = seeded_observation.context_id;
    drop(seed);
    const WRITER_COUNT: usize = 4;
    const REQUIRED_COMMITS: usize = 50;
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(WRITER_COUNT + 1));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let committed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (replay_tx, replay_rx) = std::sync::mpsc::channel();
    let writers = (0..WRITER_COUNT)
        .map(|writer| {
            let config = config.clone();
            let barrier = std::sync::Arc::clone(&barrier);
            let stop = std::sync::Arc::clone(&stop);
            let committed = std::sync::Arc::clone(&committed);
            let attempts = std::sync::Arc::clone(&attempts);
            std::thread::spawn(move || {
                let store = Store::open(config).expect("writer");
                barrier.wait();
                while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                    let seq = attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let observation_id = ObservationId::new_v7();
                    let evidence_id = EvidenceId::new_v7();
                    if store
                        .record_observation(record_input_with(
                            observation_id,
                            entity,
                            context_id,
                            vec![evidence_ref(observation_id, evidence_id)],
                            BTreeMap::from([(
                                "writer".to_string(),
                                Some(serde_json::json!({"thread": writer, "seq": seq})),
                            )]),
                            Vec::new(),
                        ))
                        .is_ok()
                    {
                        committed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    let progress_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let progress_reached = loop {
        if committed.load(std::sync::atomic::Ordering::SeqCst) >= REQUIRED_COMMITS {
            break true;
        }
        if std::time::Instant::now() >= progress_deadline {
            break false;
        }
        std::thread::yield_now();
    };
    match progress_reached {
        true => {}
        false => {
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
            for writer in writers {
                writer.join().expect("writer joined after progress timeout");
            }
            panic!(
                "writer flood did not create {REQUIRED_COMMITS} durable observations before replay; committed={}, attempts={}",
                committed.load(std::sync::atomic::Ordering::SeqCst),
                attempts.load(std::sync::atomic::Ordering::SeqCst)
            );
        }
    }
    let reader_config = config.clone();
    let reader_committed = std::sync::Arc::clone(&committed);
    std::thread::spawn(move || {
        let reader = Store::open(reader_config).expect("reader");
        let before = reader_committed.load(std::sync::atomic::Ordering::SeqCst);
        let replay = reader.replay(decision);
        let after = reader_committed.load(std::sync::atomic::Ordering::SeqCst);
        replay_tx
            .send((before, after, replay))
            .expect("send replay result");
    });
    let (before_replay, after_replay, replay) = replay_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("replay returned while writer flood was still active");
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    for writer in writers {
        writer.join().expect("writer joined");
    }
    assert!(before_replay >= REQUIRED_COMMITS);
    assert!(after_replay >= before_replay);
    let reader = Store::open(config).expect("reader");
    let history = reader.entity_history(entity).expect("history");
    assert!(
        history.len() > REQUIRED_COMMITS,
        "writer flood must create durable snapshots"
    );
    let replay = replay.expect("replay");
    assert_eq!(
        replay
            .evidence_lineage
            .iter()
            .map(|lineage| lineage.read_txid)
            .collect::<HashSet<_>>(),
        HashSet::from([replay.read_txid])
    );
    assert_eq!(replay.input_diffs.len(), 1);
    assert_eq!(replay.changed_input_count, 1);
    assert_eq!(replay.unchanged_input_count, 0);
    let writer_change = replay.input_diffs[0]
        .field_changes
        .iter()
        .find(|change| change.field == "writer")
        .expect("writer field change");
    assert_eq!(writer_change.old_value, None);
    let snapshot_at_replay = reader
        .state_at(entity, replay.read_txid)
        .expect("resolve immutable history at replay read txid")
        .expect("snapshot visible at replay read txid");
    assert_eq!(
        writer_change.new_value.as_ref(),
        snapshot_at_replay.properties.get("writer")
    );
    let store_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/store");
    for (file, method) in [
        ("decision.rs", "pub fn replay"),
        ("evidence.rs", "pub fn basis_diff"),
    ] {
        let source = std::fs::read_to_string(store_dir.join(file)).expect("store source");
        let start = source.find(method).expect("method source");
        let end = source[start + method.len()..]
            .find("\n    pub fn ")
            .map(|offset| start + method.len() + offset)
            .unwrap_or(source.len());
        let method_source = &source[start..end];
        assert!(
            !method_source.contains("execute_in_tx"),
            "{method} must use one read horizon, not per-query tx execution"
        );
        assert!(
            !method_source.contains("\"execute_at_snapshot\""),
            "{method} must call execute_at_snapshot, not mention it in a string literal"
        );
        assert!(
            method_source.lines().any(|line| {
                let trimmed = line.trim_start();
                !trimmed.starts_with("//") && trimmed.contains(".execute_at_snapshot(")
            }),
            "{method} must read through one execute_at_snapshot MVCC horizon"
        );
        for forbidden in [
            "self.db.execute(",
            ".current_entity_snapshot(",
            ".entity_history(",
            ".get_decision(",
            ".get_observation(",
            ".list_evidence_for_observation(",
            ".list_invalidations(",
        ] {
            assert!(
                !method_source.contains(forbidden),
                "{method} must not perform snapshot-sensitive reads through {forbidden}"
            );
        }
    }
}

#[test]
fn replay_per_context_isolation_consultant_b_does_not_see_client_a_evidence() {
    let (_dir, store) = fresh_store();
    let (decision_a, obs_a, evidence_a) = seed_decision_with_source(&store, "client-a");
    let (decision_b, _obs_b, evidence_b) = seed_decision_with_source(&store, "client-b");
    let ctx_a = store
        .get_decision(decision_a)
        .expect("client-a decision")
        .expect("client-a decision row")
        .context_id;
    let ctx_b = store
        .get_decision(decision_b)
        .expect("client-b decision")
        .expect("client-b decision row")
        .context_id;
    let replay = store.replay(decision_b).expect("replay");
    assert_eq!(replay_evidence_ids(&replay), vec![evidence_b]);
    assert!(!replay_evidence_ids(&replay).contains(&evidence_a));
    assert!(
        replay_evidence_refs(&replay)
            .iter()
            .all(|row| row.context_id == ctx_b)
    );
    assert_eq!(
        store
            .get_evidence(evidence_a)
            .expect("client-a evidence")
            .context_id,
        ctx_a
    );
    assert_eq!(
        store
            .get_evidence(evidence_b)
            .expect("client-b evidence")
            .context_id,
        ctx_b
    );
    assert_eq!(
        store
            .list_evidence_for_observation(obs_a)
            .expect("client-a evidence")
            .iter()
            .map(|row| row.id)
            .collect::<Vec<_>>(),
        vec![evidence_a]
    );
}

#[test]
fn replay_per_context_isolation_combined_with_expired_retention_consultant_b_returns_lineage_with_expired_status_no_client_a_leak()
 {
    let (_dir, store) = fresh_store();
    let (decision_a, obs_a, evidence_a) = seed_decision_with_source(&store, "client-a");
    let (decision, _obs_b, evidence) = seed_decision_with_source(&store, "client-b");
    let ctx_a = store
        .get_decision(decision_a)
        .expect("client-a decision")
        .expect("client-a decision row")
        .context_id;
    let ctx_b = store
        .get_decision(decision)
        .expect("client-b decision")
        .expect("client-b decision row")
        .context_id;
    store
        .update_evidence_retention_status(evidence, RetentionStatus::Expired)
        .expect("retention");
    let replay = store.replay(decision).expect("replay");
    assert_eq!(replay.decision_id, decision);
    let client_b_evidence = replay_evidence_refs(&replay);
    assert_eq!(
        client_b_evidence
            .iter()
            .map(|row| row.id)
            .collect::<Vec<_>>(),
        vec![evidence]
    );
    assert_eq!(
        client_b_evidence[0].retention_status,
        RetentionStatus::Expired
    );
    assert_eq!(client_b_evidence[0].context_id, ctx_b);
    assert!(!client_b_evidence.iter().any(|row| row.id == evidence_a));
    assert_eq!(
        store
            .get_evidence(evidence_a)
            .expect("client-a evidence")
            .context_id,
        ctx_a
    );
    assert_eq!(
        store
            .list_evidence_for_observation(obs_a)
            .expect("client-a")
            .iter()
            .map(|row| row.id)
            .collect::<Vec<_>>(),
        vec![evidence_a]
    );
}

#[test]
fn replay_when_entity_snapshot_evidenced_by_is_empty_returns_snapshot_state_with_empty_evidence_vec_and_real_state()
 {
    let (_dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: "empty-evidence-snapshot".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "empty evidence basis".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "manual snapshot".to_string(),
            properties: BTreeMap::from([("region".to_string(), serde_json::json!("us-east-1"))]),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    let snapshot = store
        .current_entity_snapshot(entity.id)
        .expect("snapshot")
        .expect("current snapshot");
    assert!(snapshot.evidenced_by.is_empty());
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "manual".to_string(),
            description: "manual basis".to_string(),
            reasoning: Vec::new(),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: None,
            based_on_snapshots: vec![snapshot.id],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx,
            precedent_ids: vec![],
            agent_id: None,
        })
        .expect("decision")
        .id;
    let replay = store.replay(decision).expect("replay");
    assert_eq!(replay.input_diffs[0].field_changes.len(), 0);
    assert_eq!(replay.evidence_lineage.len(), 1);
    assert_eq!(replay.evidence_lineage[0].basis_snapshot_id, snapshot.id);
    assert_eq!(replay.evidence_lineage[0].observation_id, None);
    assert!(replay.evidence_lineage[0].evidence.is_empty());
}

#[test]
fn invalidate_decision_concurrent_against_same_active_decision_produces_exactly_one_invalidation_row_one_ok_rest_illegal_transition()
 {
    let (_dir, store) = fresh_store();
    let (decision, obs, _evidence) = seed_decision(&store);
    let context_id = store
        .get_decision(decision)
        .expect("decision")
        .expect("decision row")
        .context_id;
    let mut triggers = vec![obs];
    for writer in 1..4 {
        let entity = store
            .create_entity(CreateEntity {
                entity_type: EntityType::Service,
                name: format!("trigger-entity-{writer}"),
                properties: BTreeMap::new(),
                tags: vec![],
                context_id,
            })
            .expect("trigger entity");
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let mut input = record_input_with(
            observation_id,
            entity.id,
            context_id,
            vec![evidence_ref(observation_id, evidence_id)],
            BTreeMap::from([(format!("trigger_{writer}"), Some(serde_json::json!(writer)))]),
            Vec::new(),
        );
        input.source = format!("concurrent-trigger-{writer}");
        store
            .record_observation(input)
            .expect("trigger observation");
        triggers.push(observation_id);
    }
    let db_path = store.db_path().to_path_buf();
    drop(store);
    let pre_db = Database::open(&db_path).expect("raw pre-count");
    let pre_observations = table_count(&pre_db, "observations");
    let pre_evidence = table_count(&pre_db, "evidence");
    let pre_snapshots = table_count(&pre_db, "entity_snapshots");
    let pre_embeddings = embedding_handle_count(&pre_db);
    let pre_invalidations = table_count(&pre_db, "invalidations");
    drop(pre_db);

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
    let handles: Vec<_> = triggers
        .into_iter()
        .enumerate()
        .map(|(writer, trigger)| {
            let db_path = db_path.clone();
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                let store = Store::open(StoreConfig {
                    db_path,
                    ..Default::default()
                })
                .expect("writer open");
                barrier.wait();
                store.invalidate_decision(decision, trigger, format!("reason_{writer}"))
            })
        })
        .collect();
    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer joined"))
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(
                result,
                Err(CgError::IllegalTransition {
                    from: DecisionStatus::Invalidated,
                    to: DecisionStatus::Invalidated
                })
            ))
            .count(),
        3
    );
    let reader = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("reader open");
    assert_eq!(
        reader
            .list_invalidations(decision_context(&reader, decision), Some(decision), None)
            .expect("invalidations")
            .len(),
        1
    );
    drop(reader);
    let db = Database::open(&db_path).expect("raw post-count");
    assert_eq!(table_count(&db, "observations"), pre_observations);
    assert_eq!(table_count(&db, "evidence"), pre_evidence);
    assert_eq!(table_count(&db, "entity_snapshots"), pre_snapshots);
    assert_eq!(embedding_handle_count(&db), pre_embeddings);
    assert_eq!(table_count(&db, "invalidations"), pre_invalidations + 1);
    let source = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/store/evidence.rs"),
    )
    .expect("evidence source");
    let start = source
        .find("pub fn invalidate_decision")
        .expect("invalidate source");
    let end = source[start..]
        .find("\n    pub fn ")
        .map(|offset| start + offset)
        .unwrap_or(source.len());
    let invalidate_source = &source[start..end];
    for forbidden in ["Mutex", "OnceLock", "LazyLock", ".lock(", "outcome_mutex"] {
        assert!(
            !invalidate_source.contains(forbidden),
            "invalidate_decision must not serialize contention with app-side lock {forbidden}"
        );
    }
    assert_no_unsanctioned_app_side_locks_or_global_state_in_core_sources(
        "invalidate_decision contention",
    );
    let self_edge = db.execute(
        "UPDATE decisions SET status = $status WHERE id = $id",
        &db_params(vec![
            ("status", Value::Text("invalidated".to_string())),
            ("id", Value::Uuid(decision.as_uuid())),
        ]),
    );
    assert!(matches!(self_edge, Err(Error::InvalidStateTransition(_))));
}

#[test]
fn invalidate_decision_active_to_invalidated_returns_result_with_previous_status_active_new_status_invalidated_reason_invalidation_id_transitioned_at()
 {
    let (_dir, store) = fresh_store();
    let (decision, obs, _evidence) = seed_decision(&store);
    let original = store
        .get_decision(decision)
        .expect("decision")
        .expect("decision row");
    let result = store
        .invalidate_decision(decision, obs, "basis changed")
        .expect("invalidate");
    assert_eq!(result.previous_status, DecisionStatus::Active);
    assert_eq!(result.new_status, DecisionStatus::Invalidated);
    assert_eq!(result.reason.as_deref(), Some("basis changed"));
    assert!(result.invalidation_id.is_valid());
    assert_ne!(result.invalidation_id.as_uuid(), uuid::Uuid::nil());
    assert!(result.transitioned_at.0 > 0);

    let rows = store
        .list_invalidations(original.context_id, Some(decision), None)
        .expect("invalidations");
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.reason.as_deref(), Some("basis changed"));
    assert_eq!(row.threatened_intention_ids, original.intention_ids);
    assert!(row.intention_now_un_served);
    assert_eq!(row.severity, InvalidationSeverity::Critical);
}

#[test]
fn invalidate_decision_actor_recoverable_through_one_hop_observation_source_walk() {
    let (_dir, store) = fresh_store();
    let (decision, obs, _evidence) =
        seed_decision_with_source(&store, "github-pr-merge:user@acme.com");
    store
        .invalidate_decision(decision, obs, "basis changed")
        .expect("invalidate");
    let invalidations = store
        .list_invalidations(decision_context(&store, decision), Some(decision), None)
        .expect("invalidations");
    let decision_context = store
        .get_decision(decision)
        .expect("decision")
        .expect("decision row")
        .context_id;
    assert_eq!(invalidations.len(), 1);
    assert_eq!(invalidations[0].context_id, decision_context);
    assert_eq!(invalidations[0].trigger_observation_id, obs);
    assert_eq!(invalidations[0].affected_decision_id, decision);
    assert_eq!(
        store.get_observation(obs).expect("observation").source,
        "github-pr-merge:user@acme.com"
    );
}

#[test]
fn invalidate_decision_terminal_status_from_superseded_archived_invalidated_returns_illegal_transition_naming_current()
 {
    let (_dir, store) = fresh_store();
    let (archived, archived_obs, _evidence) = seed_decision(&store);
    store
        .archive_decision(archived, "done".to_string())
        .expect("archive");
    let result = store.invalidate_decision(archived, archived_obs, "basis changed");
    assert!(matches!(
        result,
        Err(CgError::IllegalTransition {
            from: DecisionStatus::Archived,
            to: DecisionStatus::Invalidated
        })
    ));

    let (superseded, superseded_obs, _evidence) = seed_decision(&store);
    let current = store
        .get_decision(superseded)
        .expect("decision")
        .expect("decision row");
    store
        .supersede_decision(
            superseded,
            CreateDecision {
                decision_type: current.decision_type.clone(),
                description: "replacement decision".to_string(),
                reasoning: current.reasoning.clone(),
                confidence: current.confidence,
                intention_ids: current.intention_ids.clone(),
                based_on_entity_ids: Vec::new(),
                basis_fields: None,
                based_on_snapshots: current.based_on_snapshots.clone(),
                tags: current.tags.clone(),
                properties: current.properties.clone(),
                context_id: current.context_id,
                precedent_ids: current.precedent_ids.clone(),
                agent_id: current.agent_id,
            },
        )
        .expect("supersede");
    let result = store.invalidate_decision(superseded, superseded_obs, "basis changed");
    assert!(matches!(
        result,
        Err(CgError::IllegalTransition {
            from: DecisionStatus::Superseded,
            to: DecisionStatus::Invalidated
        })
    ));

    let (invalidated, invalidated_obs, _evidence) = seed_decision(&store);
    let first = store
        .invalidate_decision(invalidated, invalidated_obs, "first invalidation")
        .expect("first invalidation");
    assert_eq!(first.new_status, DecisionStatus::Invalidated);
    let result = store.invalidate_decision(invalidated, invalidated_obs, "basis changed");
    assert!(matches!(
        result,
        Err(CgError::IllegalTransition {
            from: DecisionStatus::Invalidated,
            to: DecisionStatus::Invalidated
        })
    ));
}

#[test]
fn invalidate_decision_with_terraform_hook_observation_round_trips_reason_and_walks_trigger_to_observation_source()
 {
    let (_dir, store) = fresh_store();
    let (decision, obs, _evidence) = seed_decision_with_source(&store, "terraform-webhook");
    let result = store
        .invalidate_decision(decision, obs, "terraform drift changed subnet cidr")
        .expect("invalidate");
    assert_eq!(result.new_status, DecisionStatus::Invalidated);
    assert_eq!(
        result.reason.as_deref(),
        Some("terraform drift changed subnet cidr")
    );
    assert_eq!(
        store.get_observation(obs).expect("observation").source,
        "terraform-webhook"
    );
}

#[test]
fn invalidate_decision_under_basis_file_renamed_observation_records_reason_byte_equal() {
    let (_dir, store) = fresh_store();
    let (decision, obs, _evidence) = seed_decision(&store);
    let result = store
        .invalidate_decision(
            decision,
            obs,
            "basis file renamed src/auth.py -> src/authentication.py",
        )
        .expect("invalidate");
    assert_eq!(
        result.reason.as_deref(),
        Some("basis file renamed src/auth.py -> src/authentication.py")
    );
    assert_eq!(result.new_status, DecisionStatus::Invalidated);
}

#[test]
fn invalidate_decision_per_context_isolation_consultant_b_invalidation_does_not_appear_in_client_a()
{
    let (_dir, store) = fresh_store();
    let (decision_a, _obs_a, _evidence_a) = seed_decision_with_source(&store, "client-a");
    let (decision, obs, _evidence) = seed_decision_with_source(&store, "client-b");
    store
        .invalidate_decision(decision, obs, "race condition was fixed in 2024")
        .expect("invalidate");
    assert!(
        store
            .list_invalidations(decision_context(&store, decision_a), Some(decision_a), None,)
            .expect("client-a invalidations")
            .is_empty()
    );
    assert_eq!(
        store
            .list_invalidations(decision_context(&store, decision), Some(decision), None)
            .expect("invalidations")
            .len(),
        1
    );
}

#[test]
fn list_invalidations_status_only_is_scoped_to_requested_context() {
    let (_dir, store) = fresh_store();
    let (decision_a, obs_a, _evidence_a) = seed_decision_with_source(&store, "client-a");
    let (decision_b, obs_b, _evidence_b) = seed_decision_with_source(&store, "client-b");
    let ctx_a = decision_context(&store, decision_a);
    let ctx_b = decision_context(&store, decision_b);
    store
        .invalidate_decision(decision_a, obs_a, "client-a drift")
        .expect("invalidate a");
    store
        .invalidate_decision(decision_b, obs_b, "client-b drift")
        .expect("invalidate b");

    let client_a_pending = store
        .list_invalidations(ctx_a, None, Some(InvalidationStatus::Pending))
        .expect("client-a pending");
    let client_b_pending = store
        .list_invalidations(ctx_b, None, Some(InvalidationStatus::Pending))
        .expect("client-b pending");
    assert_eq!(
        client_a_pending
            .iter()
            .map(|row| row.affected_decision_id)
            .collect::<Vec<_>>(),
        vec![decision_a]
    );
    assert_eq!(
        client_b_pending
            .iter()
            .map(|row| row.affected_decision_id)
            .collect::<Vec<_>>(),
        vec![decision_b]
    );
}

#[test]
fn invalidate_decision_concurrent_per_client_does_not_produce_cross_context_invalidation_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let store = Store::open(config.clone()).expect("open");
    let (decision_a, obs_a, _evidence_a) = seed_decision_with_source(&store, "client-a");
    let (decision_b, obs_b, _evidence_b) = seed_decision_with_source(&store, "client-b");

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let barrier_a = std::sync::Arc::clone(&barrier);
    let config_a = config.clone();
    let writer_a = std::thread::spawn(move || {
        let writer = Store::open(config_a).expect("writer a open");
        barrier_a.wait();
        writer.invalidate_decision(decision_a, obs_a, "client-a invalidation")
    });
    let barrier_b = std::sync::Arc::clone(&barrier);
    let config_b = config.clone();
    let writer_b = std::thread::spawn(move || {
        let writer = Store::open(config_b).expect("writer b open");
        barrier_b.wait();
        writer.invalidate_decision(decision_b, obs_b, "client-b invalidation")
    });
    barrier.wait();
    let result_a = writer_a.join().expect("writer a joined");
    let result_b = writer_b.join().expect("writer b joined");
    assert!(
        result_a.is_ok(),
        "client-a invalidation failed: {result_a:?}"
    );
    assert!(
        result_b.is_ok(),
        "client-b invalidation failed: {result_b:?}"
    );

    let invalidations_a = store
        .list_invalidations(decision_context(&store, decision_a), Some(decision_a), None)
        .expect("client-a invalidations");
    let invalidations_b = store
        .list_invalidations(decision_context(&store, decision_b), Some(decision_b), None)
        .expect("client-b invalidations");
    assert_eq!(invalidations_a.len(), 1);
    assert_eq!(invalidations_b.len(), 1);
    assert_eq!(
        invalidations_a[0].context_id,
        store
            .get_decision(decision_a)
            .expect("client-a decision")
            .expect("client-a decision row")
            .context_id
    );
    assert_eq!(
        invalidations_b[0].context_id,
        store
            .get_decision(decision_b)
            .expect("client-b decision")
            .expect("client-b decision row")
            .context_id
    );
    assert_eq!(invalidations_a[0].affected_decision_id, decision_a);
    assert_eq!(invalidations_a[0].trigger_observation_id, obs_a);
    assert_eq!(
        invalidations_a[0].reason.as_deref(),
        Some("client-a invalidation")
    );
    assert_eq!(invalidations_b[0].affected_decision_id, decision_b);
    assert_eq!(invalidations_b[0].trigger_observation_id, obs_b);
    assert_eq!(
        invalidations_b[0].reason.as_deref(),
        Some("client-b invalidation")
    );
}

#[test]
fn basis_diff_returns_is_invalidated_candidate_true_iff_any_basis_property_diverges_from_current() {
    let (_dir, store) = fresh_store();
    let (decision, obs, _evidence) = seed_decision(&store);
    let entity = store.get_observation(obs).expect("observation").entity_id;
    store
        .update_entity(
            entity,
            EntityPatch {
                properties: Some(BTreeMap::from([(
                    "region".to_string(),
                    serde_json::json!("eu-west-2"),
                )])),
                ..Default::default()
            },
        )
        .expect("update current");
    assert!(
        store
            .basis_diff(decision)
            .expect("diff")
            .is_invalidated_candidate
    );
}

#[test]
fn basis_diff_returns_is_invalidated_candidate_false_when_no_property_diverges() {
    let (_dir, store) = fresh_store();
    let (decision, _obs, _evidence) = seed_decision(&store);
    let diff = store.basis_diff(decision).expect("diff");
    assert!(!diff.is_invalidated_candidate);
    assert_eq!(diff.divergences.len(), 0);
}

#[test]
fn basis_diff_under_concurrent_writer_returns_one_consistent_basis_to_current_draw_never_a_torn_intermediate()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let store = Store::open(config.clone()).expect("open");
    let (decision, obs, _evidence) = seed_decision(&store);
    let seeded_observation = store.get_observation(obs).expect("observation");
    let entity = seeded_observation.entity_id;
    let context_id = seeded_observation.context_id;
    let second_entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "basis-diff-second-entity".to_string(),
            properties: BTreeMap::from([("region".to_string(), serde_json::json!("us-east-1"))]),
            tags: vec![],
            context_id,
        })
        .expect("second entity")
        .id;
    let first_basis = store
        .current_entity_snapshot(entity)
        .expect("first basis")
        .expect("first basis snapshot");
    let second_basis = store
        .current_entity_snapshot(second_entity)
        .expect("second basis")
        .expect("second basis snapshot");
    store
        .update_decision_basis(decision, vec![first_basis.id, second_basis.id])
        .expect("update basis");
    store
        .update_entity(
            second_entity,
            EntityPatch {
                properties: Some(BTreeMap::from([(
                    "region".to_string(),
                    serde_json::json!("ap-south-1"),
                )])),
                ..Default::default()
            },
        )
        .expect("second entity current change");
    drop(store);
    let no_writer_a = Store::open(config.clone())
        .expect("reader a")
        .basis_diff(decision)
        .expect("diff a");
    let no_writer_b = Store::open(config.clone())
        .expect("reader b")
        .basis_diff(decision)
        .expect("diff b");
    assert_eq!(no_writer_a.divergences, no_writer_b.divergences);

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(5));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let committed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (diff_tx, diff_rx) = std::sync::mpsc::channel();
    let writers = (0..4)
        .map(|_| {
            let writer_config = config.clone();
            let writer_barrier = std::sync::Arc::clone(&barrier);
            let writer_stop = std::sync::Arc::clone(&stop);
            let writer_committed = std::sync::Arc::clone(&committed);
            let writer_attempts = std::sync::Arc::clone(&attempts);
            std::thread::spawn(move || {
                let store = Store::open(writer_config).expect("writer");
                writer_barrier.wait();
                while !writer_stop.load(std::sync::atomic::Ordering::SeqCst) {
                    let seq = writer_attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let target_entity = if seq.is_multiple_of(2) {
                        entity
                    } else {
                        second_entity
                    };
                    let observation_id = ObservationId::new_v7();
                    let evidence_id = EvidenceId::new_v7();
                    if store
                        .record_observation(record_input_with(
                            observation_id,
                            target_entity,
                            context_id,
                            vec![evidence_ref(observation_id, evidence_id)],
                            BTreeMap::from([(
                                "region".to_string(),
                                Some(serde_json::json!(format!("eu-west-{seq}"))),
                            )]),
                            Vec::new(),
                        ))
                        .is_ok()
                    {
                        writer_committed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    let progress_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let progress_reached = loop {
        if committed.load(std::sync::atomic::Ordering::SeqCst) >= 50 {
            break true;
        }
        if std::time::Instant::now() >= progress_deadline {
            break false;
        }
        std::thread::yield_now();
    };
    match progress_reached {
        true => {}
        false => {
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
            for writer in writers {
                writer.join().expect("writer joined after progress timeout");
            }
            panic!(
                "writer flood did not create 50 durable snapshots before basis_diff; committed={}, attempts={}",
                committed.load(std::sync::atomic::Ordering::SeqCst),
                attempts.load(std::sync::atomic::Ordering::SeqCst)
            );
        }
    }
    let diff_config = config.clone();
    let diff_committed = std::sync::Arc::clone(&committed);
    std::thread::spawn(move || {
        let reader = Store::open(diff_config).expect("reader");
        let before = diff_committed.load(std::sync::atomic::Ordering::SeqCst);
        let diff = reader.basis_diff(decision);
        let after = diff_committed.load(std::sync::atomic::Ordering::SeqCst);
        diff_tx.send((before, after, diff)).expect("send diff");
    });
    let (before_diff, after_diff, diff) = diff_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("basis_diff returned while writer flood was still active");
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    for writer in writers {
        writer.join().expect("writer joined");
    }
    assert!(before_diff >= 50);
    assert!(after_diff >= before_diff);
    let reader = Store::open(config).expect("reader");
    let diff = diff.expect("diff");
    assert!(diff.is_invalidated_candidate);
    assert_eq!(
        diff.divergences.len(),
        2,
        "multi-basis decisions must report each divergent basis at one read horizon"
    );
    let first_history = reader.entity_history(entity).expect("first history");
    let second_history = reader
        .entity_history(second_entity)
        .expect("second history");
    assert!(
        first_history.len() + second_history.len() >= 52,
        "writer flood must create durable snapshots across both basis entities"
    );
    assert!(
        first_history.len() >= 26 && second_history.len() >= 26,
        "writer flood must create durable snapshots for both basis entities"
    );
    for divergence in &diff.divergences {
        assert_eq!(divergence.read_txid, diff.read_txid);
        let snapshot_at_diff = reader
            .state_at(divergence.entity_id, diff.read_txid)
            .expect("resolve immutable history at basis_diff read txid")
            .expect("snapshot visible at basis_diff read txid");
        assert!(
            divergence.value_diffs.iter().any(|value_diff| {
                value_diff.field == "region"
                    && value_diff.current_value.as_ref()
                        == snapshot_at_diff.properties.get("region")
            }),
            "basis_diff current_value must match one snapshot at its read txid"
        );
    }
    let source = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/store/evidence.rs"),
    )
    .expect("store evidence source");
    let start = source.find("pub fn basis_diff").expect("basis_diff source");
    let end = source[start..]
        .find("\n    pub fn ")
        .map(|offset| start + offset)
        .unwrap_or(source.len());
    let basis_source = &source[start..end];
    assert!(!basis_source.contains("execute_in_tx"));
    assert!(
        !basis_source.contains("\"execute_at_snapshot\""),
        "basis_diff must call execute_at_snapshot, not mention it in a string literal"
    );
    assert!(
        basis_source.lines().any(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("//") && trimmed.contains(".execute_at_snapshot(")
        }),
        "basis_diff must read through one execute_at_snapshot MVCC horizon"
    );
    for forbidden in [
        "self.db.execute(",
        ".current_entity_snapshot(",
        ".entity_history(",
        ".get_decision(",
        ".get_observation(",
        ".list_evidence_for_observation(",
        ".list_invalidations(",
    ] {
        assert!(
            !basis_source.contains(forbidden),
            "basis_diff must not perform snapshot-sensitive reads through {forbidden}"
        );
    }
}

#[test]
fn basis_diff_value_diff_serializes_with_basis_value_and_current_value_field_names_no_old_new_drift()
 {
    let (_dir, store) = fresh_store();
    let (decision, obs, _evidence) = seed_decision(&store);
    let entity = store.get_observation(obs).expect("observation").entity_id;
    store
        .update_entity(
            entity,
            EntityPatch {
                properties: Some(BTreeMap::from([(
                    "region".to_string(),
                    serde_json::json!("eu-west-2"),
                )])),
                ..Default::default()
            },
        )
        .expect("update current");
    let diff = store.basis_diff(decision).expect("diff");
    assert!(diff.divergences.iter().any(|d| {
        d.value_diffs.iter().any(|v| {
            v.field == "region"
                && v.basis_value == Some(serde_json::json!("eu-west-1"))
                && v.current_value == Some(serde_json::json!("eu-west-2"))
        })
    }));
    let serialized = serde_json::to_value(&diff).expect("serialize diff");
    let text = serialized.to_string();
    assert!(text.contains("basis_value"));
    assert!(text.contains("current_value"));
    assert!(!text.contains("old_value"));
    assert!(!text.contains("new_value"));
}

#[test]
fn decisions_based_on_snapshots_round_trip_from_normalized_db_table() {
    let (_dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "snapshot basis".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "payments".to_string(),
            properties: BTreeMap::from([("region".to_string(), serde_json::json!("us-east-1"))]),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    let expected_snapshot = store
        .entity_history(entity.id)
        .expect("history")
        .into_iter()
        .last()
        .expect("snapshot");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "basis".to_string(),
            description: "pin entity snapshot".to_string(),
            reasoning: Vec::new(),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: None,
            based_on_snapshots: vec![],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx,
            precedent_ids: vec![],
            agent_id: None,
        })
        .expect("decision");
    let fetched = store
        .get_decision(decision.id)
        .expect("decision")
        .expect("decision row");
    assert_eq!(fetched.based_on_snapshots, vec![expected_snapshot.id]);
    assert_eq!(expected_snapshot.entity_id, entity.id);
    let db_path = store.db_path().to_path_buf();
    drop(store);
    let db = Database::open(&db_path).expect("raw db");
    assert_eq!(
        count_where(
            &db,
            "decision_basis_snapshots",
            "decision_id = $decision_id AND snapshot_id = $snapshot_id AND basis_order = $basis_order",
            db_params(vec![
                ("decision_id", Value::Uuid(decision.id.as_uuid())),
                ("snapshot_id", Value::Uuid(expected_snapshot.id.as_uuid())),
                ("basis_order", Value::Int64(0)),
            ]),
        ),
        1,
        "decision basis must be a constrained row, not a JSON array on decisions"
    );
    let decisions_meta = db.table_meta("decisions").expect("decisions meta");
    assert!(
        decisions_meta
            .columns
            .iter()
            .all(|column| column.name != "based_on_snapshots"),
        "decisions.based_on_snapshots JSON column must not exist"
    );
}

#[test]
fn replay_uses_entity_snapshot_observations_table_not_nested_data_copy() {
    let (dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "column lineage".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "column-lineage-entity".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    store
        .record_observation(record_input_with(
            observation_id,
            entity.id,
            ctx,
            vec![evidence_ref(observation_id, evidence_id)],
            BTreeMap::new(),
            Vec::new(),
        ))
        .expect("observation with evidence");
    let db_path = store.db_path().to_path_buf();
    drop(store);

    let snapshot_id = EntitySnapshotId::new_v7();
    let db = Database::open(&db_path).expect("raw db");
    let tx = db.begin().expect("begin");
    db.execute_in_tx(
        tx,
        "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
        &db_params(vec![
            ("id", Value::Uuid(snapshot_id.as_uuid())),
            ("context_id", Value::Uuid(ctx.as_uuid())),
            ("entity_id", Value::Uuid(entity.id.as_uuid())),
            (
                "data",
                Value::Json(serde_json::json!({
                    "entity_id": entity.id.as_uuid().to_string(),
                    "entity_type": "SERVICE",
                    "name": "column-lineage-entity",
                    "properties": {"from": "raw-normalized"},
                    "evidenced_by": []
                })),
            ),
            ("valid_from", Value::TxId(tx)),
            ("valid_to", Value::Null),
        ]),
    )
    .expect("raw snapshot");
    db.execute_in_tx(
        tx,
        "INSERT INTO entity_snapshot_observations (id, context_id, entity_id, snapshot_id, observation_id, observation_order, created_at) VALUES ($id, $context_id, $entity_id, $snapshot_id, $observation_id, $observation_order, $created_at)",
        &db_params(vec![
            ("id", Value::Uuid(uuid::Uuid::now_v7())),
            ("context_id", Value::Uuid(ctx.as_uuid())),
            ("entity_id", Value::Uuid(entity.id.as_uuid())),
            ("snapshot_id", Value::Uuid(snapshot_id.as_uuid())),
            ("observation_id", Value::Uuid(observation_id.as_uuid())),
            ("observation_order", Value::Int64(0)),
            ("created_at", Value::TxId(tx)),
        ]),
    )
    .expect("raw normalized snapshot observation");
    db.commit(tx).expect("commit raw snapshot");
    drop(db);

    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    })
    .expect("reopen");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "column-lineage".to_string(),
            description: "replay from normalized evidenced_by rows".to_string(),
            intention_ids: vec![intention.id],
            based_on_snapshots: vec![snapshot_id],
            context_id: ctx,
            ..Default::default()
        })
        .expect("decision");
    let replay = store.replay(decision.id).expect("replay");
    assert_eq!(replay.evidence_lineage.len(), 1);
    assert_eq!(
        replay.evidence_lineage[0].observation_id,
        Some(observation_id)
    );
    assert_eq!(replay_evidence_ids(&replay), vec![evidence_id]);
}

#[test]
fn replay_ignores_nested_data_evidenced_by_when_normalized_table_has_no_lineage() {
    let (dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "column divergence".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "column-divergence-entity".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    let nested_observation_id = ObservationId::new_v7();
    let db_path = store.db_path().to_path_buf();
    drop(store);

    let snapshot_id = EntitySnapshotId::new_v7();
    let db = Database::open(&db_path).expect("raw db");
    let tx = db.begin().expect("begin");
    db.execute_in_tx(
        tx,
        "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
        &db_params(vec![
            ("id", Value::Uuid(snapshot_id.as_uuid())),
            ("context_id", Value::Uuid(ctx.as_uuid())),
            ("entity_id", Value::Uuid(entity.id.as_uuid())),
            (
                "data",
                Value::Json(serde_json::json!({
                    "entity_id": entity.id.as_uuid().to_string(),
                    "entity_type": "SERVICE",
                    "name": "column-divergence-entity",
                    "properties": {},
                    "evidenced_by": [nested_observation_id]
                })),
            ),
            ("valid_from", Value::TxId(tx)),
            ("valid_to", Value::Null),
        ]),
    )
    .expect("raw snapshot with stale nested lineage");
    db.commit(tx).expect("commit raw snapshot");
    drop(db);

    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    })
    .expect("reopen");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "nested-lineage-ignored".to_string(),
            description: "normalized table is the only lineage authority".to_string(),
            intention_ids: vec![intention.id],
            based_on_snapshots: vec![snapshot_id],
            context_id: ctx,
            ..Default::default()
        })
        .expect("decision");
    let replay = store.replay(decision.id).expect("replay");
    assert_eq!(replay.evidence_lineage.len(), 1);
    assert_eq!(replay.evidence_lineage[0].basis_snapshot_id, snapshot_id);
    assert_eq!(replay.evidence_lineage[0].observation_id, None);
    assert!(
        replay.evidence_lineage[0].evidence.is_empty(),
        "nested JSON evidenced_by must not drive replay lineage"
    );
}

#[test]
fn record_observation_state_delta_merges_from_entity_current_snapshot_not_orphan_open_snapshot() {
    let (dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "current-pointer-entity".to_string(),
            properties: BTreeMap::from([(
                "region".to_string(),
                serde_json::json!("authoritative"),
            )]),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    let current_before = store
        .current_entity_snapshot(entity.id)
        .expect("current")
        .expect("snapshot");
    let db_path = store.db_path().to_path_buf();
    drop(store);

    let orphan_snapshot = EntitySnapshotId::new_v7();
    let db = Database::open(&db_path).expect("raw db");
    let tx = db.begin().expect("begin");
    db.execute_in_tx(
        tx,
        "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
        &db_params(vec![
            ("id", Value::Uuid(orphan_snapshot.as_uuid())),
            ("context_id", Value::Uuid(ctx.as_uuid())),
            ("entity_id", Value::Uuid(entity.id.as_uuid())),
            (
                "data",
                Value::Json(serde_json::json!({
                    "entity_id": entity.id.as_uuid().to_string(),
                    "entity_type": "SERVICE",
                    "name": "current-pointer-entity",
                    "properties": {"region": "orphan", "orphan_only": true}
                })),
            ),
            ("valid_from", Value::TxId(tx)),
            ("valid_to", Value::Null),
        ]),
    )
    .expect("orphan open snapshot");
    db.commit(tx).expect("commit orphan open snapshot");
    drop(db);

    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    })
    .expect("reopen");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let result = store
        .record_observation(record_input_with(
            observation_id,
            entity.id,
            ctx,
            vec![evidence_ref(observation_id, evidence_id)],
            BTreeMap::from([("safe".to_string(), Some(serde_json::json!(true)))]),
            Vec::new(),
        ))
        .expect("record");
    assert_ne!(result.snapshot_id, Some(orphan_snapshot));
    let current_after = store
        .current_entity_snapshot(entity.id)
        .expect("current")
        .expect("snapshot");
    assert_ne!(current_after.id, current_before.id);
    assert_eq!(
        current_after.properties.get("region"),
        Some(&serde_json::json!("authoritative"))
    );
    assert_eq!(
        current_after.properties.get("safe"),
        Some(&serde_json::json!(true))
    );
    assert!(
        !current_after.properties.contains_key("orphan_only"),
        "merge must not read state from an open row outside entity_current_snapshots"
    );
}

#[test]
fn update_decision_basis_preserves_decision_created_at_for_intention_replay_horizon() {
    let (_dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "intention v1".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "basis-created-at".to_string(),
            properties: BTreeMap::from([("version".to_string(), serde_json::json!(1))]),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "basis-created-at".to_string(),
            description: "preserve decision horizon".to_string(),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: None,
            context_id: ctx,
            ..Default::default()
        })
        .expect("decision");
    let original_created_at = decision.created_at;

    store
        .update_intention(
            intention.id,
            IntentionPatch {
                description: Some("intention v2".to_string()),
                ..Default::default()
            },
        )
        .expect("update intention");
    store
        .update_entity(
            entity.id,
            EntityPatch {
                properties: Some(BTreeMap::from([(
                    "version".to_string(),
                    serde_json::json!(2),
                )])),
                ..Default::default()
            },
        )
        .expect("update entity");
    let new_basis = store
        .current_entity_snapshot(entity.id)
        .expect("current")
        .expect("snapshot");
    store
        .update_decision_basis(decision.id, vec![new_basis.id])
        .expect("update basis");

    let updated = store
        .get_decision(decision.id)
        .expect("decision")
        .expect("decision row");
    assert_eq!(updated.created_at, original_created_at);
    let replay = store.replay(decision.id).expect("replay");
    assert_eq!(replay.cited_intention_snapshots.len(), 1);
    assert_eq!(
        replay.cited_intention_snapshots[0].description,
        "intention v1"
    );
}

#[test]
fn update_decision_basis_reorders_existing_snapshots_without_unique_conflict() {
    let (_dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: format!("basis-reorder-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "basis reorder".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let first_entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "basis-first".to_string(),
            properties: BTreeMap::from([("order".to_string(), serde_json::json!("first"))]),
            tags: vec![],
            context_id: ctx,
        })
        .expect("first entity");
    let second_entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "basis-second".to_string(),
            properties: BTreeMap::from([("order".to_string(), serde_json::json!("second"))]),
            tags: vec![],
            context_id: ctx,
        })
        .expect("second entity");
    let first = store
        .current_entity_snapshot(first_entity.id)
        .expect("first snapshot")
        .expect("first snapshot");
    let second = store
        .current_entity_snapshot(second_entity.id)
        .expect("second snapshot")
        .expect("second snapshot");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "basis-reorder".to_string(),
            description: "basis reorder".to_string(),
            intention_ids: vec![intention.id],
            based_on_snapshots: vec![first.id, second.id],
            context_id: ctx,
            ..Default::default()
        })
        .expect("decision");

    store
        .update_decision_basis(decision.id, vec![second.id, first.id])
        .expect("reorder basis");
    let updated = store
        .get_decision(decision.id)
        .expect("decision")
        .expect("decision");
    assert_eq!(updated.based_on_snapshots, vec![second.id, first.id]);
    let replay = store.replay(decision.id).expect("replay");
    assert_eq!(
        replay
            .evidence_lineage
            .iter()
            .map(|lineage| lineage.basis_snapshot_id)
            .collect::<Vec<_>>(),
        vec![second.id, first.id]
    );
}

#[test]
fn update_decision_basis_rejects_missing_and_cross_context_snapshots() {
    let (_dir, store) = fresh_store();
    let ctx_a = store
        .create_context(CreateContext {
            name: format!("ctx-a-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("ctx a")
        .id;
    let ctx_b = store
        .create_context(CreateContext {
            name: format!("ctx-b-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("ctx b")
        .id;
    let intention_a = store
        .create_intention(CreateIntention {
            id: None,
            description: "basis a".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx_a,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention a");
    let entity_a = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "entity-a".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx_a,
        })
        .expect("entity a");
    let entity_b = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "entity-b".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx_b,
        })
        .expect("entity b");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "basis-validation".to_string(),
            description: "pin only same-context basis".to_string(),
            intention_ids: vec![intention_a.id],
            based_on_entity_ids: vec![entity_a.id],
            basis_fields: None,
            context_id: ctx_a,
            ..Default::default()
        })
        .expect("decision");
    let missing = EntitySnapshotId::new_v7();
    let missing_result = store.update_decision_basis(decision.id, vec![missing]);
    assert!(matches!(
        missing_result,
        Err(CgError::NotFound {
            missing: Some("entity_snapshot"),
            ..
        })
    ));
    let cross_snapshot = store
        .current_entity_snapshot(entity_b.id)
        .expect("current b")
        .expect("snapshot b");
    let cross_result = store.update_decision_basis(decision.id, vec![cross_snapshot.id]);
    assert!(matches!(cross_result, Err(CgError::CrossContextViolation)));
}

#[test]
fn decision_basis_snapshot_missing_target_rejected_by_db_fk() {
    let (_dir, store) = fresh_store();
    let decision = seed_decision(&store).0;
    let decision_row = store
        .get_decision(decision)
        .expect("decision")
        .expect("decision row");
    let basis_id = decision_row.based_on_snapshots[0];
    let basis_entity = store
        .entity_history(store.replay(decision).expect("replay").input_diffs[0].entity_id)
        .expect("history")
        .into_iter()
        .find(|snapshot| snapshot.id == basis_id)
        .expect("basis snapshot")
        .entity_id;
    let db_path = store.db_path().to_path_buf();
    drop(store);
    let missing = EntitySnapshotId::new_v7();
    let db = Database::open(&db_path).expect("raw db");
    let tx = db.begin().expect("begin");
    let insert = db.execute_in_tx(
        tx,
        "INSERT INTO decision_basis_snapshots (id, context_id, decision_id, entity_id, snapshot_id, basis_order, created_at) VALUES ($id, $context_id, $decision_id, $entity_id, $snapshot_id, $basis_order, $created_at)",
        &db_params(vec![
            ("id", Value::Uuid(uuid::Uuid::now_v7())),
            ("context_id", Value::Uuid(decision_row.context_id.as_uuid())),
            ("decision_id", Value::Uuid(decision.as_uuid())),
            ("entity_id", Value::Uuid(basis_entity.as_uuid())),
            ("snapshot_id", Value::Uuid(missing.as_uuid())),
            ("basis_order", Value::Int64(99)),
            ("created_at", Value::TxId(tx)),
        ]),
    );
    let result = match insert {
        Ok(_) => db.commit(tx),
        Err(err) => {
            let _ = db.rollback(tx);
            Err(err)
        }
    };
    assert!(
        matches!(result, Err(Error::ForeignKeyViolation { .. })),
        "unexpected raw basis insert result: {result:?}"
    );
}

#[test]
fn raw_sql_cannot_update_removed_decision_basis_json_column() {
    let (_dir, store) = fresh_store();
    let decision = seed_decision(&store).0;
    let db_path = store.db_path().to_path_buf();
    drop(store);
    let db = Database::open(&db_path).expect("raw db");
    let decisions_meta = db.table_meta("decisions").expect("decisions meta");
    assert!(
        decisions_meta
            .columns
            .iter()
            .all(|column| column.name != "based_on_snapshots"),
        "decision basis must not be stored as a JSON column"
    );
    let result = db.execute(
        "UPDATE decisions SET based_on_snapshots = $basis WHERE id = $id",
        &db_params(vec![
            ("id", Value::Uuid(decision.as_uuid())),
            ("basis", Value::Json(serde_json::json!({"not": "an array"}))),
        ]),
    );
    assert!(result.is_err());
}

#[test]
fn replay_ignores_raw_cross_context_serves_edge_for_same_decision_source() {
    let (dir, store) = fresh_store();
    let ctx_a = store
        .create_context(CreateContext {
            name: format!("ctx-a-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("ctx a")
        .id;
    let ctx_b = store
        .create_context(CreateContext {
            name: format!("ctx-b-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("ctx b")
        .id;
    let intention_a = store
        .create_intention(CreateIntention {
            id: None,
            description: "ctx a intention".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx_a,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention a");
    let intention_b = store
        .create_intention(CreateIntention {
            id: None,
            description: "ctx b leaked intention".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx_b,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention b");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "entity-a".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx_a,
        })
        .expect("entity");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "replay-context".to_string(),
            description: "cite only same context".to_string(),
            intention_ids: vec![intention_a.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: None,
            context_id: ctx_a,
            ..Default::default()
        })
        .expect("decision");
    let db_path = store.db_path().to_path_buf();
    drop(store);
    let db = Database::open(&db_path).expect("raw db");
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &db_params(vec![
            ("id", Value::Uuid(uuid::Uuid::now_v7())),
            ("source_id", Value::Uuid(decision.id.as_uuid())),
            ("target_id", Value::Uuid(intention_b.id.as_uuid())),
            ("edge_type", Value::Text("SERVES".to_string())),
            ("context_id", Value::Uuid(ctx_b.as_uuid())),
            ("valid_from", Value::TxId(TxId(1))),
            ("valid_to", Value::Null),
            ("properties", Value::Json(serde_json::json!({}))),
        ]),
    )
    .expect("raw cross-context edge");
    drop(db);
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    })
    .expect("reopen");
    let replay = store.replay(decision.id).expect("replay");
    assert_eq!(
        replay
            .cited_intention_snapshots
            .iter()
            .map(|snapshot| snapshot.intention_id)
            .collect::<Vec<_>>(),
        vec![intention_a.id]
    );
}

#[test]
fn store_open_default_storeconfig_on_empty_path_routes_to_initialize_fresh_seeds_default_text_embedding_space()
 {
    let (_dir, store) = fresh_store();
    let spaces = store.list_embedding_spaces().expect("spaces");
    assert_eq!(spaces.len(), 1);
    let space = &spaces[0];
    assert_eq!(space.embedding_space_id, TEXT_SPACE);
    assert_eq!(space.dimension, 768);
    assert_eq!(space.metric, DistanceMetric::Cosine);
    let bindings = space
        .bindings
        .iter()
        .map(|binding| (binding.table_name.as_str(), binding.vector_column.as_str()))
        .collect::<HashSet<_>>();
    assert_eq!(
        bindings,
        HashSet::from([("evidence", "vector_text"), ("digests", "vector_text")])
    );
}

#[test]
fn store_open_freshly_initialized_schema_introspection_shows_no_observations_data_column_no_digests_embedding_vector384()
 {
    let (_dir, store) = fresh_store();
    let spaces = store.list_embedding_spaces().expect("spaces");
    assert!(spaces.iter().any(|space| space.dimension == 768));
    let db = reopen_database_for_store(store);
    let tables = db.table_names();
    assert!(tables.iter().any(|table| table == "evidence"));
    let observations = db.table_meta("observations").expect("observations meta");
    assert!(
        observations
            .columns
            .iter()
            .all(|column| column.name != "data"),
        "observations.data must not exist"
    );
    let digests = db.table_meta("digests").expect("digests meta");
    assert!(
        digests.columns.iter().any(|column| {
            column.name == "vector_text" && column.column_type == ColumnType::Vector(768)
        }),
        "digests.vector_text VECTOR(768) must exist"
    );
    assert!(
        digests.columns.iter().all(|column| {
            !(column.name == "embedding" && column.column_type == ColumnType::Vector(384))
        }),
        "digests.embedding VECTOR(384) must not exist"
    );
}

#[test]
fn store_open_extra_embedding_spaces_validates_binding_against_schema() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = vision_config();
    config.db_path = dir.path().join("cg.db");
    config.extra_embedding_spaces[0].bindings[0].vector_column = "missing_vector".to_string();
    let result = Store::open(config);
    assert!(matches!(
        result,
        Err(CgError::MissingVectorColumnForSpace { table, column })
            if table == "evidence" && column == "missing_vector"
    ));
}

#[test]
fn store_open_invalid_extra_embedding_space_does_not_poison_fresh_db_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let mut invalid = vision_config();
    invalid.db_path = db_path.clone();
    invalid.extra_embedding_spaces[0].bindings[0].vector_column = "missing_vector".to_string();
    let result = Store::open(invalid);
    assert!(matches!(
        result,
        Err(CgError::MissingVectorColumnForSpace { table, column })
            if table == "evidence" && column == "missing_vector"
    ));

    let recovered = Store::open(StoreConfig {
        db_path,
        ..Default::default()
    })
    .expect("fresh path must recover after failed invalid open");
    assert!(
        recovered
            .list_embedding_spaces()
            .expect("spaces")
            .iter()
            .any(|space| space.embedding_space_id == TEXT_SPACE)
    );
}

#[test]
fn store_open_same_id_model_drift_extra_embedding_space_does_not_poison_fresh_db_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let invalid = StoreConfig {
        db_path: db_path.clone(),
        extra_embedding_spaces: vec![EmbeddingSpaceDecl {
            embedding_space_id: TEXT_SPACE.to_string(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "text".to_string(),
            model_name: "different-model".to_string(),
            model_version: "default".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::Text, EvidenceKind::StructuredSignal],
            default_for_evidence_kind: None,
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
        }],
        ..Default::default()
    };
    let result = Store::open(invalid);
    assert!(matches!(
        result,
        Err(CgError::EmbeddingSpaceIdReserved { id }) if id == TEXT_SPACE
    ));

    Store::open(StoreConfig {
        db_path,
        ..Default::default()
    })
    .expect("fresh path must recover after failed same-id model drift open");
}

#[test]
fn store_open_extra_embedding_spaces_idempotent_redeclaration_succeeds_no_duplicate_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = vision_config();
    config.db_path = dir.path().join("cg.db");
    Store::open(config.clone()).expect("open 1");
    Store::open(config.clone()).expect("open 2");
    let spaces = Store::open(config)
        .expect("open 3")
        .list_embedding_spaces()
        .expect("spaces");
    assert_eq!(
        spaces
            .iter()
            .filter(|space| space.embedding_space_id == "vigil_face_reid_v1")
            .count(),
        1
    );
}

#[test]
fn store_open_extra_embedding_spaces_conflicting_model_identity_returns_embedding_space_mismatch_naming_field()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = vision_config();
    config.db_path = dir.path().join("cg.db");
    Store::open(config.clone()).expect("initial open");
    config.extra_embedding_spaces[0].model_name = "different-model".to_string();
    let result = Store::open(config);
    assert!(matches!(
        result,
        Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "model_name",
            ..
        })
    ));
}

#[test]
fn store_open_extra_embedding_spaces_conflicting_dimension_returns_embedding_space_mismatch_naming_field()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = vision_config();
    config.db_path = dir.path().join("cg.db");
    config.extra_embedding_spaces[0].dimension = 768;
    let result = Store::open(config);
    assert!(matches!(
        result,
        Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "dimension",
            ..
        })
    ));
}

#[test]
fn store_open_extra_embedding_spaces_conflicting_metric_returns_embedding_space_mismatch_naming_field()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = vision_config();
    config.db_path = dir.path().join("cg.db");
    Store::open(config.clone()).expect("initial open");
    config.extra_embedding_spaces[0].metric = DistanceMetric::Euclidean;
    let result = Store::open(config);
    assert!(matches!(
        result,
        Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "metric",
            ..
        })
    ));
}

#[test]
fn store_open_extra_embedding_space_cannot_claim_existing_vector_column_owned_by_another_space() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        extra_embedding_spaces: vec![EmbeddingSpaceDecl {
            embedding_space_id: "other_text_space".to_string(),
            dimension: 768,
            metric: DistanceMetric::Cosine,
            model_family: "text".to_string(),
            model_name: "other-text".to_string(),
            model_version: "1".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: "other_text_space".to_string(),
                table_name: "evidence".to_string(),
                vector_column: "vector_text".to_string(),
            }],
        }],
        ..Default::default()
    };
    let result = Store::open(config.clone());
    assert!(matches!(
        result,
        Err(CgError::EmbeddingSpaceConfigMismatch {
            field: "bindings",
            ..
        })
    ));

    config.extra_embedding_spaces[0].bindings[0].vector_column = "vector_vision".to_string();
    config.extra_embedding_spaces[0].dimension = 512;
    Store::open(config).expect("unclaimed vision column can be owned");
}

#[test]
fn store_open_two_declarations_default_for_evidence_kind_text_returns_ambiguous_embedding_space() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = StoreConfig {
        extra_embedding_spaces: vec![
            EmbeddingSpaceDecl {
                embedding_space_id: "first_text_default".to_string(),
                dimension: 768,
                metric: DistanceMetric::Cosine,
                model_family: "text".to_string(),
                model_name: "first".to_string(),
                model_version: "1".to_string(),
                supported_evidence_kinds: vec![EvidenceKind::Text],
                default_for_evidence_kind: Some(EvidenceKind::Text),
                bindings: vec![],
            },
            EmbeddingSpaceDecl {
                embedding_space_id: "second_text_default".to_string(),
                dimension: 768,
                metric: DistanceMetric::Cosine,
                model_family: "text".to_string(),
                model_name: "second".to_string(),
                model_version: "1".to_string(),
                supported_evidence_kinds: vec![EvidenceKind::Text],
                default_for_evidence_kind: Some(EvidenceKind::Text),
                bindings: vec![],
            },
        ],
        ..Default::default()
    };
    config.db_path = dir.path().join("cg.db");
    let result = Store::open(config);
    assert!(matches!(result, Err(CgError::AmbiguousEmbeddingSpace)));
}

#[test]
fn store_open_legacy_digests_embedding_vector384_column_returns_legacy_vector_store_detected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("cg.db");
    Store::admin_raw_install_legacy_schema_for_tests(&path, LegacySchemaFixture::LegacyDigest384)
        .expect("legacy");
    let result = Store::open(StoreConfig {
        db_path: path,
        ..Default::default()
    });
    assert!(matches!(result, Err(CgError::LegacyVectorStoreDetected)));
}

#[test]
fn store_open_no_cg_schema_version_row_with_existing_cg_tables_returns_legacy_schema_detected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("cg.db");
    Store::admin_raw_install_legacy_schema_for_tests(
        &path,
        LegacySchemaFixture::NoVersionRowButTablesExist,
    )
    .expect("legacy");
    let result = Store::open(StoreConfig {
        db_path: path,
        ..Default::default()
    });
    assert!(matches!(result, Err(CgError::LegacySchemaDetected { .. })));
}

#[test]
fn store_open_https_scheme_accepted_only_when_host_in_allowlist_case_insensitive_per_rfc3986() {
    let mut default_denied = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    default_denied.evidence[0].source_ref = "https://wiki.client-a.com/page".to_string();
    assert_record_error("https default deny", default_denied, |err| {
        matches!(
            err,
            CgError::InvalidSourceRefScheme {
                scheme,
                host: Some(host)
            } if scheme == "https" && host == "wiki.client-a.com"
        )
    });

    for (host, expected_ok) in [
        ("wiki.client-a.com", true),
        ("WIKI.client-a.com", true),
        ("wiki.client-b.com", false),
        ("wiki.client-a.com.attacker.com", false),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Store::open(StoreConfig {
            db_path: dir.path().join("cg.db"),
            https_allowed_hosts: vec!["wiki.client-a.com".to_string()],
            ..Default::default()
        })
        .expect("open");
        let mut input = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
        input.evidence[0].source_ref = format!("https://{host}/page");
        let evidence_id = input.evidence[0].id;
        let result = store.record_observation(input);
        if expected_ok {
            assert_eq!(
                result.expect("accepted host").evidence_ids,
                vec![evidence_id]
            );
        } else {
            assert!(matches!(
                result,
                Err(CgError::InvalidSourceRefScheme { scheme, host: Some(rejected) })
                    if scheme == "https" && rejected.eq_ignore_ascii_case(host)
            ));
        }
    }
}

#[test]
fn store_open_file_uri_accepted_only_when_file_uri_allowed_true() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        file_uri_allowed: true,
        ..Default::default()
    })
    .expect("open");
    let mut accepted = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    accepted.evidence[0].source_ref = "file:///tmp/evidence.txt".to_string();
    let accepted_id = accepted.evidence[0].id;
    assert_eq!(
        store
            .record_observation(accepted)
            .expect("file uri allowed")
            .evidence_ids,
        vec![accepted_id]
    );

    let mut input = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    input.evidence[0].source_ref = "file:///tmp/evidence.txt".to_string();
    assert_record_error(
        "file uri blocked",
        input,
        |err| matches!(err, CgError::InvalidSourceRefScheme { scheme, .. } if scheme == "file"),
    );
}

#[test]
fn store_open_vigil_edge_s3_gs_azure_cas_webhook_schemes_accepted_unconditionally() {
    let schemes = [
        "vigil-edge://camera/1",
        "s3://bucket/key",
        "gs://bucket/key",
        "azure://container/blob",
        "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "webhook://terraform/1",
    ];
    for source_ref in schemes {
        let observation_id = ObservationId::new_v7();
        let evidence_id = EvidenceId::new_v7();
        let mut input = record_input(observation_id, evidence_id);
        input.evidence[0].source_ref = source_ref.to_string();
        assert_evidence_ids_round_trip(source_ref, input, vec![evidence_id]);
    }
}

#[test]
fn decision_status_invalidated_variant_round_trips() {
    let (_dir, store) = fresh_store();
    let (decision, obs, _evidence) = seed_decision(&store);
    store
        .invalidate_decision(decision, obs, "basis drift")
        .expect("invalidate");
    let loaded = store
        .get_decision(decision)
        .expect("load")
        .expect("decision");
    assert_eq!(loaded.status, DecisionStatus::Invalidated);
    let encoded = serde_json::to_value(loaded.status).expect("status json");
    assert_eq!(encoded, serde_json::json!("invalidated"));
    let decoded: DecisionStatus = serde_json::from_value(encoded).expect("status round trip");
    assert_eq!(decoded, DecisionStatus::Invalidated);
}

#[test]
fn decision_status_enum_has_invalidated_variant_and_all_constant_lists_four_variants_in_stable_order()
 {
    assert_eq!(
        DecisionStatus::ALL,
        [
            DecisionStatus::Active,
            DecisionStatus::Superseded,
            DecisionStatus::Archived,
            DecisionStatus::Invalidated
        ]
    );
}

#[test]
fn evidence_availability_log_concurrent_cross_handle_appends_each_persist_distinct_strictly_ordered_txids_last_wins_by_txid_not_wallclock()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    store
        .record_observation(record_input(observation_id, evidence_id))
        .expect("seed evidence");

    let statuses = vec![
        RetentionStatus::NotStored,
        RetentionStatus::Ephemeral,
        RetentionStatus::RetainedExternal,
        RetentionStatus::Expired,
        RetentionStatus::Unavailable,
    ];
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(statuses.len() + 1));
    let mut writers = Vec::new();
    for status in statuses.iter().copied() {
        let path = db_path.clone();
        let barrier = std::sync::Arc::clone(&barrier);
        writers.push(std::thread::spawn(move || {
            let writer = Store::open(StoreConfig {
                db_path: path,
                ..Default::default()
            })
            .expect("writer open");
            barrier.wait();
            writer.update_evidence_retention_status(evidence_id, status)
        }));
    }
    barrier.wait();
    for writer in writers {
        writer.join().expect("writer joined").expect("retention");
    }

    let rows = store.evidence_availability_log(evidence_id).expect("log");
    assert_eq!(rows.len(), statuses.len());
    let row_statuses = rows.iter().map(|row| row.status).collect::<Vec<_>>();
    for status in statuses {
        assert!(
            row_statuses.contains(&status),
            "availability log must retain status {status:?} byte-equal"
        );
    }
    let distinct_txids = rows
        .iter()
        .map(|row| row.recorded_at_txid)
        .collect::<HashSet<_>>();
    assert_eq!(distinct_txids.len(), rows.len());
    let latest = rows
        .iter()
        .max_by_key(|row| row.recorded_at_txid)
        .expect("latest log row");
    let evidence = store
        .list_evidence_for_observation(observation_id)
        .expect("evidence");
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence[0].retention_status, latest.status);
}

#[test]
fn dereference_evidence_with_unavailable_retention_returns_err_evidence_unavailable_no_test_only_seam()
 {
    let (_dir, store) = fresh_store();
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    store
        .record_observation(record_input(observation_id, evidence_id))
        .expect("record");
    store
        .update_evidence_retention_status(evidence_id, RetentionStatus::Unavailable)
        .expect("retention");
    fn collect_rs(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("source dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                collect_rs(&path, files);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
    }
    let source_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut source_files = Vec::new();
    collect_rs(&source_root, &mut source_files);
    let test_only_symbols = source_files
        .iter()
        .flat_map(|path| {
            std::fs::read_to_string(path)
                .expect("source")
                .lines()
                .filter(|line| line.contains("_for_tests"))
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(test_only_symbols.len(), 1);
    assert!(test_only_symbols[0].contains("admin_raw_install_legacy_schema_for_tests"));
    assert!(matches!(
        store.dereference_evidence(evidence_id),
        Err(CgError::EvidenceUnavailable)
    ));
}

#[test]
fn dereference_evidence_with_retained_external_returns_ok_opaque_handle_without_outbound_fetch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file_uri = format!(
        "file://{}",
        dir.path()
            .join("must-not-be-opened-by-context-graph")
            .display()
    );
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        file_uri_allowed: true,
        ..Default::default()
    })
    .expect("open");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let mut evidence = image_evidence_ref(observation_id, evidence_id);
    evidence.source_ref = file_uri.clone();
    store
        .record_observation(record_input_with(
            observation_id,
            EntityId::new_v7(),
            ContextId::new_v7(),
            vec![evidence],
            BTreeMap::new(),
            Vec::new(),
        ))
        .expect("record");
    let handle = store.dereference_evidence(evidence_id).expect("handle");
    assert_eq!(handle.source_ref, file_uri);
    let source = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/store/evidence.rs"),
    )
    .expect("store evidence source");
    let start = source
        .find("pub fn dereference_evidence")
        .expect("dereference source");
    let end = source[start..]
        .find("\n    pub fn ")
        .map(|offset| start + offset)
        .unwrap_or(source.len());
    let dereference_source = &source[start..end];
    for forbidden in [
        "reqwest",
        "ureq",
        "hyper",
        "std::process",
        "Command::new",
        "std::fs::read",
        "std::fs::File",
        "File::open",
        "OpenOptions",
        "tokio::fs",
    ] {
        assert!(!dereference_source.contains(forbidden));
    }
    for line in dereference_source.lines() {
        assert!(!(line.contains("read") && line.contains("source_ref")));
    }
    assert_core_source_has_no_raw_byte_fetch_primitives();
}

#[test]
fn create_digest_with_non_text_evidence_kind_rejected_with_unsupported_evidence_kind() {
    let (_dir, store) = fresh_store();
    for kind in [
        EvidenceKind::ImageFrame,
        EvidenceKind::AudioSegment,
        EvidenceKind::VideoSegment,
        EvidenceKind::StructuredSignal,
    ] {
        let digest_id = DigestId::new_v7();
        let result = store.create_digest(CreateDigest {
            id: digest_id,
            source: "acceptance".to_string(),
            session_ref: None,
            summary: format!("{kind:?} digest"),
            kind,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: text_vector(),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: producer(),
            },
            context_id: ContextId::new_v7(),
            properties: BTreeMap::new(),
        });
        assert!(matches!(
            result,
            Err(CgError::UnsupportedEvidenceKind { kind: rejected }) if rejected == kind
        ));
    }
}

#[test]
fn create_digest_with_text_evidence_kind_lands_in_digests_vector_text_hnsw_index_queryable_via_engine_no_json_shim()
 {
    let (_dir, store) = fresh_store();
    let digest_id = DigestId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let decoy_digest_id = DigestId::new_v7();
    let mut decoy_vector = vec![0.0; 768];
    decoy_vector[1] = 1.0;
    store
        .create_digest(CreateDigest {
            id: digest_id,
            source: "acceptance".to_string(),
            session_ref: None,
            summary: "text digest".to_string(),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: embedding_id,
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: text_vector(),
                target: EmbeddingTarget::Digest(digest_id),
                produced_by: producer(),
            },
            context_id: ContextId::new_v7(),
            properties: BTreeMap::new(),
        })
        .expect("digest");
    store
        .create_digest(CreateDigest {
            id: decoy_digest_id,
            source: "acceptance-decoy".to_string(),
            session_ref: None,
            summary: "decoy text digest".to_string(),
            kind: EvidenceKind::Text,
            embedding: PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: TEXT_SPACE.to_string(),
                vector: decoy_vector.clone(),
                target: EmbeddingTarget::Digest(decoy_digest_id),
                produced_by: producer(),
            },
            context_id: ContextId::new_v7(),
            properties: BTreeMap::new(),
        })
        .expect("decoy digest");
    let db = reopen_database_for_store(store);
    let hits = db
        .query_vector(
            VectorIndexRef::new("digests", "vector_text"),
            &text_vector(),
            1,
            None,
            db.snapshot(),
        )
        .expect("query");
    assert_first_vector_hit_id(&db, "digests", &hits, digest_id.as_uuid());
    let decoy_hits = db
        .query_vector(
            VectorIndexRef::new("digests", "vector_text"),
            &decoy_vector,
            1,
            None,
            db.snapshot(),
        )
        .expect("decoy query");
    assert_first_vector_hit_id(&db, "digests", &decoy_hits, decoy_digest_id.as_uuid());
    assert_eq!(
        count_where(
            &db,
            "digest_embedding_handles",
            "id = $id AND embedding_space_id = $embedding_space_id AND target_id = $target_id",
            db_params(vec![
                ("id", Value::Uuid(embedding_id.as_uuid())),
                ("embedding_space_id", Value::Text(TEXT_SPACE.to_string())),
                ("target_id", Value::Uuid(digest_id.as_uuid())),
            ])
        ),
        1,
        "create_digest must persist embedding handle lineage for the digest vector"
    );
    let meta = db.table_meta("digests").expect("digests meta");
    assert!(
        meta.columns.iter().any(|column| {
            column.name == "vector_text" && column.column_type == ColumnType::Vector(768)
        }),
        "digests.vector_text must be an engine vector column"
    );
    for forbidden in [
        "embedding",
        "embedding_json",
        "vector_text_json",
        "vector_json",
    ] {
        assert!(
            meta.columns.iter().all(|column| column.name != forbidden),
            "digests.{forbidden} must not exist as a JSON/vector shadow column"
        );
    }
}

#[test]
fn create_digest_with_dimension_mismatch_rejected_with_embedding_space_mismatch_naming_field() {
    let (_dir, store) = fresh_store();
    let digest_id = DigestId::new_v7();
    let result = store.create_digest(CreateDigest {
        id: digest_id,
        source: "acceptance".to_string(),
        session_ref: None,
        summary: "short vector".to_string(),
        kind: EvidenceKind::Text,
        embedding: PrecomputedEmbedding {
            id: EmbeddingId::new_v7(),
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: vec![0.0; 384],
            target: EmbeddingTarget::Digest(digest_id),
            produced_by: producer(),
        },
        context_id: ContextId::new_v7(),
        properties: BTreeMap::new(),
    });
    assert!(matches!(
        result,
        Err(CgError::EmbeddingDimensionMismatch { space_id, expected: 768, found: 384 }) if space_id == TEXT_SPACE
    ));
}

#[test]
fn record_observation_vigil_edge_multi_camera_evidence_each_carries_distinct_region_and_frame_index_routes_to_vision_index()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..vision_config()
    };
    let store = Store::open(config.clone()).expect("open");
    let observation_id = ObservationId::new_v7();
    let first = EvidenceId::new_v7();
    let second = EvidenceId::new_v7();
    let mut cam_1 = image_evidence_ref(observation_id, first);
    cam_1.source_ref = "vigil-edge://camera-1/frame/17".to_string();
    cam_1.frame_index = Some(17);
    cam_1.region = Some(serde_json::json!({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}));
    cam_1.producer.system = "vigil-edge-cam-1".to_string();
    let mut cam_3 = image_evidence_ref(observation_id, second);
    cam_3.source_ref = "vigil-edge://camera-3/frame/42".to_string();
    cam_3.frame_index = Some(42);
    cam_3.region = Some(serde_json::json!({"x": 0.5, "y": 0.6, "w": 0.2, "h": 0.2}));
    cam_3.producer.system = "vigil-edge-cam-3".to_string();
    let context_id = ContextId::new_v7();
    set_evidence_context(std::slice::from_mut(&mut cam_1), context_id);
    set_evidence_context(std::slice::from_mut(&mut cam_3), context_id);
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        context_id,
        vec![cam_1.clone(), cam_3.clone()],
        BTreeMap::new(),
        vec![
            PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: "vigil_face_reid_v1".to_string(),
                vector: vec![1.0; 512],
                target: EmbeddingTarget::Evidence(first),
                produced_by: cam_1.producer.clone(),
            },
            PrecomputedEmbedding {
                id: EmbeddingId::new_v7(),
                embedding_space_id: "vigil_face_reid_v1".to_string(),
                vector: {
                    let mut vector = vec![0.0; 512];
                    vector[1] = 1.0;
                    vector
                },
                target: EmbeddingTarget::Evidence(second),
                produced_by: cam_3.producer.clone(),
            },
        ],
    );
    let result = store.record_observation(input).expect("record");
    assert_eq!(result.evidence_ids, vec![first, second]);
    let returned = store
        .list_evidence_for_observation(observation_id)
        .expect("evidence");
    assert_eq!(
        returned.iter().map(|e| e.id).collect::<Vec<_>>(),
        vec![first, second]
    );
    assert_eq!(returned[0].region, cam_1.region);
    assert_eq!(returned[0].frame_index, Some(17));
    assert_eq!(returned[0].producer, cam_1.producer);
    assert_eq!(returned[1].region, cam_3.region);
    assert_eq!(returned[1].frame_index, Some(42));
    assert_eq!(returned[1].producer, cam_3.producer);
    drop(store);

    let reopened = Store::open(config).expect("reopen");
    let returned = reopened
        .list_evidence_for_observation(observation_id)
        .expect("reopened evidence");
    assert_eq!(
        returned.iter().map(|e| e.id).collect::<Vec<_>>(),
        vec![first, second]
    );
    assert_eq!(returned[0], cam_1);
    assert_eq!(returned[0].source_ref, "vigil-edge://camera-1/frame/17");
    assert_eq!(returned[0].region, cam_1.region);
    assert_eq!(returned[0].frame_index, Some(17));
    assert_eq!(returned[0].producer, cam_1.producer);
    assert_eq!(returned[1], cam_3);
    assert_eq!(returned[1].source_ref, "vigil-edge://camera-3/frame/42");
    assert_eq!(returned[1].region, cam_3.region);
    assert_eq!(returned[1].frame_index, Some(42));
    assert_eq!(returned[1].producer, cam_3.producer);
    let db = reopen_database_for_store(reopened);
    let first_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_vision"),
            &[1.0; 512],
            1,
            None,
            db.snapshot(),
        )
        .expect("first vision query");
    assert_first_vector_hit_id(&db, "evidence", &first_hits, first.as_uuid());
    let mut second_vector = vec![0.0; 512];
    second_vector[1] = 1.0;
    let second_hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_vision"),
            &second_vector,
            1,
            None,
            db.snapshot(),
        )
        .expect("second vision query");
    assert_first_vector_hit_id(&db, "evidence", &second_hits, second.as_uuid());
}

#[test]
fn replay_returns_evidence_frame_index_byte_equal_to_record_observation_input() {
    let (_dir, store) = fresh_store();
    let observation_id = ObservationId::new_v7();
    let evidence = EvidenceId::new_v7();
    let mut supplied = image_evidence_ref(observation_id, evidence);
    supplied.frame_index = Some(70_000);
    let (decision, _, _) = seed_decision_with_evidence(
        &store,
        vec![supplied],
        BTreeMap::from([("seen".to_string(), Some(serde_json::json!(true)))]),
    );
    let replay = store.replay(decision).expect("replay");
    assert_eq!(
        replay_evidence_ref(&replay, evidence).frame_index,
        Some(70_000)
    );
}

#[test]
fn basis_diff_returns_divergence_for_enterprise_constraint_change_with_multi_modal_basis_evidence()
{
    let (_dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: format!("ctx-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "enterprise constraint".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "abstraction".to_string(),
            properties: BTreeMap::from([(
                "abstraction_layer".to_string(),
                serde_json::json!("v1"),
            )]),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    let obs = ObservationId::new_v7();
    let text_a = EvidenceId::new_v7();
    let text_b = EvidenceId::new_v7();
    let image = EvidenceId::new_v7();
    let multi_modal = record_input_with(
        obs,
        entity.id,
        ctx,
        vec![
            evidence_ref(obs, text_a),
            evidence_ref(obs, text_b),
            image_evidence_ref(obs, image),
        ],
        BTreeMap::from([(
            "abstraction_layer".to_string(),
            Some(serde_json::json!("v2")),
        )]),
        Vec::new(),
    );
    let result = store.record_observation(multi_modal).expect("multi-modal");
    assert_eq!(result.evidence_ids, vec![text_a, text_b, image]);
    let basis_snapshot = result.snapshot_id.expect("basis snapshot");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: "pin abstraction layer".to_string(),
            reasoning: Vec::new(),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: None,
            based_on_snapshots: vec![basis_snapshot],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx,
            precedent_ids: vec![],
            agent_id: None,
        })
        .expect("decision");
    store
        .update_entity(
            entity.id,
            EntityPatch {
                properties: Some(BTreeMap::from([(
                    "abstraction_layer".to_string(),
                    serde_json::json!("v3"),
                )])),
                ..Default::default()
            },
        )
        .expect("post-decision change");
    let diff = store.basis_diff(decision.id).expect("diff");
    assert!(diff.is_invalidated_candidate);
    assert_eq!(diff.divergences.len(), 1);
    assert_eq!(diff.divergences[0].basis_snapshot_id, basis_snapshot);
    let basis = store
        .entity_history(entity.id)
        .expect("history")
        .into_iter()
        .find(|snapshot| snapshot.id == basis_snapshot)
        .expect("basis");
    assert_eq!(basis.evidenced_by, vec![obs]);
    let returned = store.list_evidence_for_observation(obs).expect("evidence");
    assert_eq!(
        basis
            .evidenced_by
            .iter()
            .flat_map(|observation| store
                .list_evidence_for_observation(*observation)
                .expect("basis evidence"))
            .map(|e| e.id)
            .collect::<Vec<_>>(),
        vec![text_a, text_b, image]
    );
    assert_eq!(
        returned.iter().map(|e| e.kind).collect::<Vec<_>>(),
        vec![
            EvidenceKind::Text,
            EvidenceKind::Text,
            EvidenceKind::ImageFrame,
        ]
    );
}

#[test]
fn invalidate_decision_with_human_github_pr_observation_source_round_trips_byte_equal_unicode_safe()
{
    for source in [
        "github-pr-merge:user@acme.com",
        "engineer-用户@acme.com",
        "JIRA-XYZ-1234@release/2026.05",
    ] {
        let (_dir, store) = fresh_store();
        let (decision, obs, _evidence) = seed_decision_with_source(&store, source);
        let result = store
            .invalidate_decision(decision, obs, "constraint changed")
            .expect("invalidate");
        assert_eq!(result.new_status, DecisionStatus::Invalidated);
        assert_eq!(
            store.get_observation(obs).expect("observation").source,
            source
        );
    }
}

#[test]
fn replay_under_concurrent_invalidator_returns_consistent_decision_status_across_basis_walk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    for iteration in 0..50 {
        let seed = Store::open(config.clone()).expect("open");
        let (decision, obs, _evidence) = seed_decision(&seed);
        drop(seed);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let writer_config = config.clone();
        let writer_barrier = std::sync::Arc::clone(&barrier);
        let writer = std::thread::spawn(move || {
            let store = Store::open(writer_config).expect("writer");
            writer_barrier.wait();
            store.invalidate_decision(decision, obs, "basis changed")
        });
        let reader = Store::open(config.clone()).expect("reader");
        barrier.wait();
        let replay = reader.replay(decision).expect("replay");
        writer.join().expect("writer joined").expect("invalidate");
        assert_eq!(replay.decision_id, decision);
        let invalidations_for_decision = replay
            .invalidations_at_read
            .iter()
            .filter(|invalidation| invalidation.affected_decision_id == decision)
            .count();
        match replay.decision_status {
            DecisionStatus::Active => assert_eq!(
                invalidations_for_decision, 0,
                "iteration {iteration}: active replay read must not include a visible invalidation"
            ),
            DecisionStatus::Invalidated => assert_eq!(
                invalidations_for_decision, 1,
                "iteration {iteration}: invalidated replay read must include exactly one visible invalidation"
            ),
            status => panic!(
                "iteration {iteration}: replay status {status:?} is outside active/invalidation race"
            ),
        }
        let final_reader = Store::open(config.clone()).expect("final reader");
        assert_eq!(
            final_reader
                .get_decision(decision)
                .expect("decision")
                .expect("decision row")
                .status,
            DecisionStatus::Invalidated
        );
        assert_eq!(
            final_reader
                .list_invalidations(
                    decision_context(&final_reader, decision),
                    Some(decision),
                    None,
                )
                .expect("invalidations")
                .len(),
            1
        );
    }
}

#[test]
fn basis_diff_for_nested_property_divergence_returns_full_nested_value_diff_byte_equal() {
    let (_dir, store) = fresh_store();
    let ctx = store
        .create_context(CreateContext {
            name: "nested-basis-diff".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "nested slo".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("intention");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "api".to_string(),
            properties: BTreeMap::from([(
                "slo".to_string(),
                serde_json::json!({"p99_threshold_ms": 200, "region": "us-east-1"}),
            )]),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity");
    let basis_snapshot = store
        .current_entity_snapshot(entity.id)
        .expect("snapshot")
        .expect("current");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "ops".to_string(),
            description: "pin nested slo".to_string(),
            reasoning: Vec::new(),
            confidence: Some(0.8),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: None,
            based_on_snapshots: vec![basis_snapshot.id],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx,
            precedent_ids: vec![],
            agent_id: None,
        })
        .expect("decision");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    store
        .record_observation(record_input_with(
            observation_id,
            entity.id,
            ctx,
            vec![structured_evidence_ref(observation_id, evidence_id)],
            BTreeMap::from([(
                "slo".to_string(),
                Some(serde_json::json!({"p99_threshold_ms": 500, "region": "us-east-1"})),
            )]),
            Vec::new(),
        ))
        .expect("nested change");
    let diff = store.basis_diff(decision.id).expect("diff");
    assert!(diff.is_invalidated_candidate);
    assert_eq!(diff.divergences.len(), 1);
    assert_eq!(diff.divergences[0].basis_snapshot_id, basis_snapshot.id);
    assert!(
        diff.divergences
            .iter()
            .any(|d| d.value_diffs.iter().any(|v| {
                v.field == "slo"
                    && v.basis_value
                        == Some(serde_json::json!({"p99_threshold_ms": 200, "region": "us-east-1"}))
                    && v.current_value
                        == Some(serde_json::json!({"p99_threshold_ms": 500, "region": "us-east-1"}))
            }))
    );
}

#[test]
fn record_observation_per_store_instance_isolation_store_a_writes_do_not_appear_in_store_b_at_different_db_path()
 {
    let dir_a = tempfile::tempdir().expect("dir a");
    let dir_b = tempfile::tempdir().expect("dir b");
    let store_a = Store::open(StoreConfig {
        db_path: dir_a.path().join("cg.db"),
        ..Default::default()
    })
    .expect("store a");
    let db_b_path = dir_b.path().join("cg.db");
    let store_b = Store::open(StoreConfig {
        db_path: db_b_path.clone(),
        ..Default::default()
    })
    .expect("store b");
    let observation_a = ObservationId::new_v7();
    let evidence_a = EvidenceId::new_v7();
    let input_a = record_input(observation_a, evidence_a);
    let context_a = input_a.context_id;
    let result = store_a.record_observation(input_a).expect("record a");
    assert_eq!(result.evidence_ids, vec![evidence_a]);
    assert!(store_a.get_observation(observation_a).is_ok());
    assert!(store_b.get_observation(observation_a).is_err());
    assert!(
        store_b
            .list_observations(Some(context_a))
            .expect("store b observations for context a")
            .is_empty()
    );

    let observation_b = ObservationId::new_v7();
    let evidence_b = EvidenceId::new_v7();
    let input_b = record_input_with(
        observation_b,
        EntityId::new_v7(),
        ContextId::new_v7(),
        vec![evidence_ref(observation_b, evidence_b)],
        BTreeMap::new(),
        Vec::new(),
    );
    let context_b = input_b.context_id;
    store_b.record_observation(input_b).expect("record b");
    assert!(store_b.get_observation(observation_b).is_ok());
    assert!(store_a.get_observation(observation_b).is_err());
    assert!(
        store_a
            .list_observations(Some(context_b))
            .expect("store a observations for context b")
            .is_empty()
    );

    drop(store_a);
    drop(store_b);
    let db_b = Database::open(&db_b_path).expect("raw store b");
    let hits = db_b
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &text_vector(),
            10,
            None,
            db_b.snapshot(),
        )
        .expect("store b vector query");
    assert!(
        hits.is_empty(),
        "store b vector query must not expose store a evidence embeddings"
    );
}

#[test]
fn invalidate_decision_with_consultant_session_source_records_reason_byte_equal_in_per_client_context()
 {
    let (_dir, store) = fresh_store();
    let (decision_a, _obs_a, _evidence_a) = seed_decision_with_source(&store, "client-a");
    let (decision_b, obs_b, _evidence_b) =
        seed_decision_with_source(&store, "consultant-session-12ab");
    let ctx_b = store
        .get_decision(decision_b)
        .expect("client-b decision")
        .expect("decision exists")
        .context_id;
    let result = store
        .invalidate_decision(decision_b, obs_b, "race condition was fixed in 2024")
        .expect("invalidate");
    assert_eq!(result.previous_status, DecisionStatus::Active);
    assert_eq!(
        result.reason.as_deref(),
        Some("race condition was fixed in 2024")
    );
    assert_eq!(result.new_status, DecisionStatus::Invalidated);
    assert!(
        store
            .list_invalidations(decision_context(&store, decision_a), Some(decision_a), None,)
            .expect("client-a invalidations")
            .is_empty()
    );
    let invalidations = store
        .list_invalidations(decision_context(&store, decision_b), Some(decision_b), None)
        .expect("client-b invalidations");
    assert_eq!(invalidations.len(), 1);
    assert_eq!(invalidations[0].context_id, ctx_b);
    assert_eq!(invalidations[0].affected_decision_id, decision_b);
    assert_eq!(invalidations[0].trigger_observation_id, obs_b);
    assert_eq!(
        invalidations[0].reason.as_deref(),
        Some("race condition was fixed in 2024")
    );
    assert_eq!(
        store
            .get_observation(invalidations[0].trigger_observation_id)
            .expect("trigger observation")
            .source,
        "consultant-session-12ab"
    );
}

#[test]
fn record_observation_duplicate_evidence_id_within_batch_rejected_with_duplicate_evidence_id() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let observation_id = ObservationId::new_v7();
    let duplicate = EvidenceId::new_v7();
    let embedding_id = EmbeddingId::new_v7();
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        ContextId::new_v7(),
        vec![
            evidence_ref(observation_id, duplicate),
            evidence_ref(observation_id, duplicate),
        ],
        BTreeMap::new(),
        vec![PrecomputedEmbedding {
            id: embedding_id,
            embedding_space_id: TEXT_SPACE.to_string(),
            vector: text_vector(),
            target: EmbeddingTarget::Evidence(duplicate),
            produced_by: producer(),
        }],
    );
    let result = store.record_observation(input);
    assert!(
        matches!(result, Err(CgError::DuplicateEvidenceId { id }) if id == duplicate),
        "duplicate batch id: unexpected result {result:?}"
    );
    drop(store);
    let db = Database::open(&db_path).expect("raw db");
    assert_eq!(
        count_where(
            &db,
            "observations",
            "id = $id",
            db_params(vec![("id", Value::Uuid(observation_id.as_uuid()))])
        ),
        0
    );
    assert_eq!(
        count_where(
            &db,
            "evidence",
            "id = $id",
            db_params(vec![("id", Value::Uuid(duplicate.as_uuid()))])
        ),
        0
    );
    assert_eq!(embedding_handle_count_by_id(&db, embedding_id), 0);
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &text_vector(),
            10,
            None,
            db.snapshot(),
        )
        .expect("duplicate rollback vector query");
    assert!(
        hits.is_empty(),
        "duplicate evidence batch must not leave a vector index entry"
    );
}

#[test]
fn record_observation_evidence_id_collides_with_existing_committed_row_rejected_with_duplicate_evidence_id()
 {
    let (_dir, store) = fresh_store();
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let first = record_input(observation_id, evidence_id);
    let original_source_ref = first.evidence[0].source_ref.clone();
    store.record_observation(first).expect("first");
    let second_observation_id = ObservationId::new_v7();
    let mut second = record_input(second_observation_id, evidence_id);
    second.evidence[0].source_ref =
        "cas://sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string();
    second.evidence[0].content_hash =
        Some("sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string());
    let result = store.record_observation(second);
    assert!(matches!(
        result,
        Err(CgError::DuplicateEvidenceId { id }) if id == evidence_id
    ));
    let persisted = store
        .list_evidence_for_observation(observation_id)
        .expect("original evidence");
    assert_eq!(persisted.len(), 1);
    assert_eq!(persisted[0].id, evidence_id);
    assert_eq!(persisted[0].source_ref, original_source_ref);
    assert!(store.get_observation(second_observation_id).is_err());
    assert!(
        store
            .list_evidence_for_observation(second_observation_id)
            .expect("second observation evidence")
            .is_empty()
    );
}

#[test]
fn record_observation_concurrent_same_evidence_id_returns_duplicate_evidence_id_for_loser() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let seed = Store::open(config.clone()).expect("open");
    let ctx = seed
        .create_context(CreateContext {
            name: format!("duplicate-evidence-{}", uuid::Uuid::now_v7()),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let entity_a = seed
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "duplicate-evidence-a".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity a")
        .id;
    let entity_b = seed
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "duplicate-evidence-b".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ctx,
        })
        .expect("entity b")
        .id;
    drop(seed);

    let duplicate = EvidenceId::new_v7();
    let obs_a = ObservationId::new_v7();
    let obs_b = ObservationId::new_v7();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let writers = [(obs_a, entity_a, "a"), (obs_b, entity_b, "b")]
        .into_iter()
        .map(|(observation_id, entity_id, label)| {
            let config = config.clone();
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                let store = Store::open(config).expect("writer open");
                let mut evidence = evidence_ref(observation_id, duplicate);
                evidence.source_ref = format!(
                    "cas://sha256:{hash}",
                    hash = if label == "a" {
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    } else {
                        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    }
                );
                evidence.content_hash = Some(format!(
                    "sha256:{hash}",
                    hash = if label == "a" {
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    } else {
                        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    }
                ));
                let input = record_input_with(
                    observation_id,
                    entity_id,
                    ctx,
                    vec![evidence],
                    BTreeMap::from([("writer".to_string(), Some(serde_json::json!(label)))]),
                    Vec::new(),
                );
                barrier.wait();
                store.record_observation(input)
            })
        })
        .collect::<Vec<_>>();
    let results = writers
        .into_iter()
        .map(|writer| writer.join().expect("writer joined"))
        .collect::<Vec<_>>();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(CgError::DuplicateEvidenceId { id }) if *id == duplicate))
            .count(),
        1
    );

    let reader = Store::open(config).expect("reader");
    let visible = [obs_a, obs_b]
        .into_iter()
        .map(|observation_id| {
            reader
                .list_evidence_for_observation(observation_id)
                .expect("evidence list")
                .len()
        })
        .sum::<usize>();
    assert_eq!(visible, 1);
}

#[test]
fn record_observation_derived_from_fan_out_two_children_pointing_at_same_parent_evidence_accepted()
{
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let store = Store::open(config.clone()).expect("open");
    let observation_id = ObservationId::new_v7();
    let image = EvidenceId::new_v7();
    let text_1 = EvidenceId::new_v7();
    let text_2 = EvidenceId::new_v7();
    let mut e1 = evidence_ref(observation_id, text_1);
    let mut e2 = evidence_ref(observation_id, text_2);
    e1.derived_from = Some(image);
    e2.derived_from = Some(image);
    let input = record_input_with(
        observation_id,
        EntityId::new_v7(),
        ContextId::new_v7(),
        vec![image_evidence_ref(observation_id, image), e1, e2],
        BTreeMap::new(),
        Vec::new(),
    );
    let result = store.record_observation(input).expect("derived fanout");
    assert_eq!(result.evidence_ids, vec![image, text_1, text_2]);
    drop(store);
    let reopened = Store::open(config).expect("reopen");
    let returned = reopened
        .list_evidence_for_observation(observation_id)
        .expect("reopened evidence");
    assert_eq!(
        returned.iter().map(|e| e.id).collect::<Vec<_>>(),
        vec![image, text_1, text_2]
    );
    let parent_by_child = returned
        .iter()
        .filter(|e| e.id == text_1 || e.id == text_2)
        .map(|e| (e.id, e.derived_from))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(parent_by_child.get(&text_1), Some(&Some(image)));
    assert_eq!(parent_by_child.get(&text_2), Some(&Some(image)));
}

#[test]
fn record_observation_state_delta_some_null_sets_json_null_distinct_from_none_which_deletes_property()
 {
    let (dir, store) = fresh_store();
    let db_path = dir.path().join("cg.db");
    let context_id = store
        .create_context(CreateContext {
            name: "null-vs-delete".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("context")
        .id;
    let entity_id = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "stateful service".to_string(),
            properties: BTreeMap::from([
                ("region".to_string(), serde_json::json!("us-east-1")),
                ("tier".to_string(), serde_json::json!("gold")),
            ]),
            tags: vec![],
            context_id,
        })
        .expect("entity")
        .id;
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let input = record_input_with(
        observation_id,
        entity_id,
        context_id,
        vec![evidence_ref(observation_id, evidence_id)],
        BTreeMap::from([
            ("region".to_string(), Some(serde_json::Value::Null)),
            ("tier".to_string(), None),
        ]),
        Vec::new(),
    );
    let result = store.record_observation(input).expect("record");
    let observation = store.get_observation(observation_id).expect("observation");
    assert_eq!(
        observation.state_delta.get("region"),
        Some(&Some(serde_json::Value::Null))
    );
    assert_eq!(observation.state_delta.get("tier"), Some(&None));
    let snapshot = store
        .entity_history(entity_id)
        .expect("history")
        .into_iter()
        .find(|snapshot| Some(snapshot.id) == result.snapshot_id)
        .expect("snapshot");
    assert_eq!(
        snapshot.properties.get("region"),
        Some(&serde_json::Value::Null)
    );
    assert!(
        !snapshot.properties.contains_key("tier"),
        "None must delete, distinct from Some(Null)"
    );
    drop(store);

    let reopened = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("reopen");
    let reopened_observation = reopened
        .get_observation(observation_id)
        .expect("reopened observation");
    assert_eq!(
        reopened_observation.state_delta.get("region"),
        Some(&Some(serde_json::Value::Null))
    );
    assert_eq!(reopened_observation.state_delta.get("tier"), Some(&None));
    drop(reopened);

    let db = Database::open(&db_path).expect("raw db");
    let row = db
        .execute(
            "SELECT state_delta FROM observations WHERE id = $id",
            &db_params(vec![("id", Value::Uuid(observation_id.as_uuid()))]),
        )
        .expect("raw state_delta");
    assert_eq!(row.rows.len(), 1);
    let Value::Json(raw) = &row.rows[0][0] else {
        panic!("state_delta must be JSON");
    };
    assert_eq!(raw["format"], serde_json::json!("cg_state_delta_v1"));
    assert_eq!(raw["ops"]["region"]["op"], serde_json::json!("set"));
    assert!(raw["ops"]["region"]["value"].is_null());
    assert_eq!(raw["ops"]["tier"]["op"], serde_json::json!("delete"));
    assert!(raw["ops"]["tier"].get("value").is_none());
}

#[test]
fn record_observation_embedding_target_evidence_id_exists_in_db_but_not_in_current_batch_rejected_with_embedding_target_not_found()
 {
    let (_dir, store) = fresh_store();
    let old_observation = ObservationId::new_v7();
    let old_evidence = EvidenceId::new_v7();
    let old_input = record_input(old_observation, old_evidence);
    let old_context = old_input.context_id;
    store.record_observation(old_input).expect("old record");
    let mut input = record_input(ObservationId::new_v7(), EvidenceId::new_v7());
    input.context_id = old_context;
    for evidence in &mut input.evidence {
        evidence.context_id = old_context;
    }
    let rejected_observation = input.id;
    let rejected_entity = input.entity_id;
    let rejected_evidence = input.evidence[0].id;
    input.embeddings[0].target = EmbeddingTarget::Evidence(old_evidence);
    let result = store.record_observation(input);
    assert!(matches!(
        result,
        Err(CgError::EmbeddingTargetNotFound { id }) if id == old_evidence
    ));
    assert_eq!(
        store
            .list_evidence_for_observation(old_observation)
            .expect("old evidence")
            .iter()
            .map(|row| row.id)
            .collect::<Vec<_>>(),
        vec![old_evidence]
    );
    let db_path = store.db_path().to_path_buf();
    drop(store);
    let db = Database::open(&db_path).expect("open raw contextdb");
    assert_eq!(
        count_where(
            &db,
            "observations",
            "id = $id",
            db_params(vec![("id", Value::Uuid(rejected_observation.as_uuid()))])
        ),
        0,
        "embedding target exists outside batch: observation row leaked"
    );
    assert_eq!(
        count_where(
            &db,
            "entity_snapshots",
            "entity_id = $entity_id",
            db_params(vec![("entity_id", Value::Uuid(rejected_entity.as_uuid()))])
        ),
        0,
        "embedding target exists outside batch: snapshot row leaked"
    );
    assert_eq!(
        count_where(
            &db,
            "evidence",
            "id = $id",
            db_params(vec![("id", Value::Uuid(rejected_evidence.as_uuid()))])
        ),
        0,
        "embedding target exists outside batch: evidence row leaked"
    );
}

#[test]
fn embedding_space_bindings_data_shape_pins_evidence_vector_text_to_text_default_not_to_extra_vision_space()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..vision_config()
    })
    .expect("open");
    let spaces = store.list_embedding_spaces().expect("spaces");
    let text_space = spaces
        .iter()
        .find(|space| space.embedding_space_id == TEXT_SPACE);
    assert!(
        text_space.is_some(),
        "default text embedding space must be listed"
    );
    let text_space = text_space.unwrap();
    assert_eq!(text_space.dimension, 768);
    assert_eq!(text_space.metric, DistanceMetric::Cosine);
    let text_bindings = text_space
        .bindings
        .iter()
        .map(|binding| (binding.table_name.as_str(), binding.vector_column.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(
        text_bindings
            .iter()
            .filter(|binding| **binding == ("evidence", "vector_text"))
            .count(),
        1
    );
    let mut owners = HashMap::<(&str, &str), usize>::new();
    for space in &spaces {
        for binding in &space.bindings {
            *owners
                .entry((binding.table_name.as_str(), binding.vector_column.as_str()))
                .or_insert(0) += 1;
        }
    }
    assert_eq!(owners.get(&("evidence", "vector_text")), Some(&1));
    assert_eq!(owners.get(&("evidence", "vector_vision")), Some(&1));
    assert_eq!(
        spaces
            .iter()
            .filter(|space| {
                space.bindings.iter().any(|binding| {
                    binding.table_name == "evidence" && binding.vector_column == "vector_text"
                })
            })
            .count(),
        1,
        "extra vision space must not claim evidence.vector_text"
    );
}

#[test]
fn store_initialize_fresh_against_non_empty_path_drops_all_rows_and_reseeds_default_text_embedding_space()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    let input = record_input(observation_id, evidence_id);
    let entity_id = input.entity_id;
    let embedding_id = input.embeddings[0].id;
    let vector = input.embeddings[0].vector.clone();
    store.record_observation(input).expect("record");
    let reopened = Store::initialize_fresh(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("fresh");
    assert!(
        reopened
            .list_observations(None)
            .expect("observations")
            .is_empty()
    );
    assert!(
        reopened
            .list_embedding_spaces()
            .expect("spaces")
            .iter()
            .any(|space| space.embedding_space_id == TEXT_SPACE)
    );
    let db = reopen_database_for_store(reopened);
    for table in [
        "observations",
        "evidence",
        "entity_snapshots",
        "entity_snapshot_observations",
        "entity_current_snapshots",
        "decision_basis_snapshots",
        "evidence_availability_log",
        "decisions",
        "invalidations",
        "outcomes",
    ] {
        assert_eq!(
            table_count(&db, table),
            0,
            "{table} rows survived fresh init"
        );
    }
    for table in EMBEDDING_HANDLE_TABLES {
        assert_eq!(
            table_count(&db, table),
            0,
            "{table} rows survived fresh init"
        );
    }
    assert_eq!(
        count_where(
            &db,
            "observations",
            "id = $id",
            db_params(vec![("id", Value::Uuid(observation_id.as_uuid()))])
        ),
        0
    );
    assert_eq!(
        count_where(
            &db,
            "evidence",
            "id = $id",
            db_params(vec![("id", Value::Uuid(evidence_id.as_uuid()))])
        ),
        0
    );
    assert_eq!(
        count_where(
            &db,
            "entity_snapshots",
            "entity_id = $entity_id",
            db_params(vec![("entity_id", Value::Uuid(entity_id.as_uuid()))])
        ),
        0
    );
    assert_eq!(embedding_handle_count_by_id(&db, embedding_id), 0);
    let hits = db
        .query_vector(
            VectorIndexRef::new("evidence", "vector_text"),
            &vector,
            10,
            None,
            db.snapshot(),
        )
        .expect("fresh init vector query");
    assert!(
        hits.is_empty(),
        "initialize_fresh must drop prior evidence vector entries"
    );
}

#[test]
fn cg_library_crate_context_graph_carries_no_outbound_http_or_subprocess_or_filesystem_source_ref_dereference_path()
 {
    let crate_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(crate_dir.join("Cargo.toml")).expect("manifest");
    // The promise is about THIS library crate's own resolved dependency
    // graph: a workspace-lock scan would also trip on sibling BINARIES'
    // transitives (the cg CLI legitimately carries the dial-by-key sync
    // transport, whose internals use an HTTP client). Resolve the library's
    // real production graph instead, through the allowlisted dependency-audit
    // helper.
    let library_graph = crate::cg_main_dep_audit::default_build_tree("context-graph");
    assert!(
        library_graph.contains("contextdb-engine"),
        "library graph resolution must actually list dependencies"
    );

    for dep in [
        ["req", "west"].concat(),
        ["hyper-", "client"].concat(),
        ["s", "urf"].concat(),
        ["is", "ahc"].concat(),
        ["atto", "httpc"].concat(),
        ["tonic-", "web"].concat(),
    ] {
        assert!(
            !library_graph.contains(&format!("{dep} v")) && !manifest.contains(&dep),
            "outbound dependency present: {dep}"
        );
    }

    let mut files = Vec::new();
    collect_rs_files(&crate_dir.join("src"), &mut files);
    collect_rs_files(&crate_dir.join("tests"), &mut files);
    let direct_forbidden = [
        ["Tcp", "Stream::connect"].concat(),
        ["tokio::net::Tcp", "Stream"].concat(),
        ["std::process::Com", "mand"].concat(),
    ];
    for file in files {
        let text = std::fs::read_to_string(&file).expect("source file");
        let dependency_audit_test = file == crate_dir.join("tests/acceptance/cg_main_dep_audit.rs")
            || file == crate_dir.join("tests/acceptance/host_session_substrate_guards.rs");
        for forbidden in &direct_forbidden {
            assert!(
                !text.contains(forbidden) || dependency_audit_test,
                "forbidden dereference call in {}: {forbidden}",
                file.display()
            );
        }
        if file.starts_with(crate_dir.join("src")) {
            if file == crate_dir.join("src/embedder/embeddinggemma.rs")
                || file == crate_dir.join("src/embedder/nomic.rs")
                || file == crate_dir.join("src/control_transport.rs")
            {
                continue;
            }
            for line in text.lines() {
                let trimmed = line.trim();
                let allowed_store_recovery_log = file == crate_dir.join("src/store/mod.rs")
                    && trimmed == "if let Ok(mut file) = std::fs::OpenOptions::new()";
                for forbidden in [
                    "std::fs::read",
                    "std::fs::File",
                    "File::open",
                    "OpenOptions",
                    "tokio::fs",
                ] {
                    assert!(
                        !line.contains(forbidden) || allowed_store_recovery_log,
                        "core source must not fetch evidence bytes in {}: {forbidden}",
                        file.display()
                    );
                }
            }
        }
        for line in text.lines() {
            assert!(
                !(line.contains(&["std::fs::", "read"].concat()) && line.contains("source_ref")),
                "source_ref-keyed filesystem read in {}: {line}",
                file.display()
            );
            assert!(
                !(line.contains(&["tokio::fs::", "read"].concat()) && line.contains("source_ref")),
                "source_ref-keyed async filesystem read in {}: {line}",
                file.display()
            );
        }
    }
    assert_core_source_has_no_raw_byte_fetch_primitives();
}
