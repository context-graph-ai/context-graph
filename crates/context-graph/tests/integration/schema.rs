use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::mpsc;
use std::sync::{Arc, Barrier};

use context_graph::schema::{DIGEST_VECTOR_DIMENSION, TABLES};
use contextdb_core::{Error, TxId, Value};
use contextdb_engine::{Database, QueryResult};
use serde_json::json;
use uuid::Uuid;

const TEXT_SPACE: &str = "text_default_embeddinggemma_300m_768";

fn initialize_schema(db: &Database) {
    let existing: HashSet<_> = db.table_names().into_iter().collect();
    for table in TABLES {
        if existing.contains(table.name) {
            continue;
        }
        db.execute(table.ddl, &Default::default()).unwrap();
    }
}

fn fresh_db() -> Database {
    let db = Database::open_memory();
    initialize_schema(&db);
    db
}

fn cg_observation(
    observation_id: context_graph::ObservationId,
    evidence_id: context_graph::EvidenceId,
    entity_id: context_graph::EntityId,
    context_id: context_graph::ContextId,
    writer: usize,
) -> context_graph::RecordObservation {
    context_graph::RecordObservation {
        id: observation_id,
        entity_id,
        context_id,
        observation_type: format!("writer_{writer}"),
        source: "schema".to_string(),
        observed_at: chrono::Utc::now(),
        evidence: vec![context_graph::EvidenceRef {
            id: evidence_id,
            observation_id,
            context_id,
            kind: context_graph::EvidenceKind::Text,
            source_ref:
                "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
            content_hash: Some(
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
            ),
            retention_status: context_graph::RetentionStatus::RetainedExternal,
            ..Default::default()
        }],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::from([(format!("writer_{writer}"), Some(json!(writer)))]),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    }
}

fn cg_observation_with_text_embedding(
    observation_id: context_graph::ObservationId,
    evidence_id: context_graph::EvidenceId,
    embedding_id: context_graph::EmbeddingId,
    entity_id: context_graph::EntityId,
    context_id: context_graph::ContextId,
    vector: Vec<f32>,
) -> context_graph::RecordObservation {
    let mut observation = cg_observation(observation_id, evidence_id, entity_id, context_id, 0);
    observation.state_delta = BTreeMap::from([("risk".to_string(), Some(json!("elevated")))]);
    observation.embeddings = vec![context_graph::PrecomputedEmbedding {
        id: embedding_id,
        embedding_space_id: TEXT_SPACE.to_string(),
        vector,
        target: context_graph::EmbeddingTarget::Evidence(evidence_id),
        produced_by: context_graph::EvidenceProducer {
            system: "schema".to_string(),
            model_name: "embeddinggemma-300m".to_string(),
            model_version: "default".to_string(),
            pipeline_version: "1".to_string(),
        },
    }];
    observation
}

fn text_vector_with_axis(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 768];
    vector[axis % 768] = 1.0;
    vector
}

fn vision_vector_with_axis(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 512];
    vector[axis % 512] = 1.0;
    vector
}

fn vision_embedding_space_decl(id: &str) -> context_graph::EmbeddingSpaceDecl {
    context_graph::EmbeddingSpaceDecl {
        embedding_space_id: id.to_string(),
        dimension: 512,
        metric: context_graph::DistanceMetric::Cosine,
        model_family: "vision".to_string(),
        model_name: "vigil-face".to_string(),
        model_version: "1".to_string(),
        supported_evidence_kinds: vec![context_graph::EvidenceKind::ImageFrame],
        default_for_evidence_kind: Some(context_graph::EvidenceKind::ImageFrame),
        bindings: vec![context_graph::EmbeddingSpaceBinding {
            embedding_space_id: id.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_vision".to_string(),
        }],
    }
}

fn seed_store_context_entity(
    store: &context_graph::Store,
) -> (context_graph::ContextId, context_graph::EntityId) {
    let context = store
        .create_context(context_graph::CreateContext {
            name: format!("schema-{}", uuid::Uuid::now_v7()),
            labels: Vec::new(),
            properties: BTreeMap::new(),
        })
        .expect("context");
    let entity = store
        .create_entity(context_graph::CreateEntity {
            entity_type: context_graph::EntityType::Service,
            name: format!("entity-{}", uuid::Uuid::now_v7()),
            properties: BTreeMap::new(),
            tags: Vec::new(),
            context_id: context.id,
        })
        .expect("entity");
    (context.id, entity.id)
}

fn seed_store_decision(
    config: &context_graph::StoreConfig,
) -> (context_graph::DecisionId, context_graph::ObservationId) {
    let store = context_graph::Store::open(config.clone()).expect("open");
    let (context_id, entity_id) = seed_store_context_entity(&store);
    let intention = store
        .create_intention(context_graph::CreateIntention {
            id: None,
            description: "schema intention".to_string(),
            status: context_graph::IntentionStatus::Active,
            origin: context_graph::IntentionOrigin::Human,
            context_id,
            properties: BTreeMap::new(),
            tags: Vec::new(),
        })
        .expect("intention");
    let observation_id = context_graph::ObservationId::new_v7();
    let record = store
        .record_observation(cg_observation(
            observation_id,
            context_graph::EvidenceId::new_v7(),
            entity_id,
            context_id,
            0,
        ))
        .expect("observation");
    let snapshot_id = record
        .snapshot_id
        .or_else(|| {
            store
                .current_entity_snapshot(entity_id)
                .expect("current snapshot")
                .map(|snapshot| snapshot.id)
        })
        .expect("basis snapshot");
    let decision = store
        .create_decision(context_graph::CreateDecision {
            decision_type: "schema".to_string(),
            description: "schema decision".to_string(),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity_id],
            basis_fields: None,
            based_on_snapshots: vec![snapshot_id],
            context_id,
            ..Default::default()
        })
        .expect("decision");
    (decision.id, observation_id)
}

fn store_decision_context(
    store: &context_graph::Store,
    decision_id: context_graph::DecisionId,
) -> context_graph::ContextId {
    store
        .get_decision(decision_id)
        .expect("decision")
        .expect("decision row")
        .context_id
}

fn uuid(n: u128) -> String {
    Uuid::from_u128(n).to_string()
}

fn now_ts() -> String {
    "2026-01-02T03:04:05Z".to_string()
}

fn params(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn row_value<'a>(result: &'a QueryResult, row: usize, col: &str) -> &'a Value {
    let idx = result
        .columns
        .iter()
        .position(|name| name == col)
        .unwrap_or_else(|| panic!("missing column {col}"));
    &result.rows[row][idx]
}

fn assert_one_row(result: &QueryResult) {
    assert_eq!(result.rows.len(), 1, "expected exactly one row");
}

fn assert_error_is_fk(result: contextdb_core::Result<QueryResult>) {
    assert!(matches!(result, Err(Error::ForeignKeyViolation { .. })));
}

fn scalar_count(db: &Database, sql: &str, params: &HashMap<String, Value>) -> i64 {
    let result = db.execute(sql, params).expect("count query");
    assert_one_row(&result);
    match row_value(&result, 0, "n") {
        Value::Int64(n) => *n,
        other => panic!("expected integer count, got {other:?}"),
    }
}

fn insert_context(db: &Database, id: Uuid, name: &str) {
    db.execute(
        "INSERT INTO contexts (id, name, labels, properties, created_at) VALUES ($id, $name, $labels, $properties, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("name", Value::Text(name.to_string())),
            ("labels", Value::Json(json!({"org": "acme"}))),
            ("properties", Value::Json(json!({"tier": "gold"}))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();
}

fn insert_entity(db: &Database, id: Uuid, context_id: Uuid, name: &str) {
    db.execute(
        "INSERT INTO entities (id, entity_type, name, status, context_id, properties, tags, created_at) VALUES ($id, $entity_type, $name, $status, $context_id, $properties, $tags, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("entity_type", Value::Text("SERVICE".into())),
            ("name", Value::Text(name.to_string())),
            ("status", Value::Text("active".into())),
            ("context_id", Value::Uuid(context_id)),
            ("properties", Value::Json(json!({"owner": "team-a"}))),
            ("tags", Value::Json(json!([]))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();
}

fn insert_intention(db: &Database, id: Uuid, context_id: Uuid, status: &str) {
    db.execute(
        "INSERT INTO intentions (id, description, status, context_id, origin, origin_ref, blueprint_id, bindings, blueprint_proposal, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $origin_ref, $blueprint_id, $bindings, $blueprint_proposal, $properties, $tags, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("description", Value::Text("Protect children from unauthorized pickup".into())),
            ("status", Value::Text(status.to_string())),
            ("context_id", Value::Uuid(context_id)),
            ("origin", Value::Text("human".into())),
            ("origin_ref", Value::Null),
            ("blueprint_id", Value::Null),
            ("bindings", Value::Json(json!({"slot": "value"}))),
            ("blueprint_proposal", Value::Text("proposal text".into())),
            ("properties", Value::Json(json!({"priority": "high"}))),
            ("tags", Value::Json(json!(["safety", "family"]))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();
}

fn insert_decision(db: &Database, id: Uuid, context_id: Uuid, status: &str) {
    db.execute(
        "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("decision_type", Value::Text("architecture".into())),
            ("description", Value::Text("Use abstraction layer for fault isolation".into())),
            ("reasoning", Value::Json(json!([{
                "operation": "analyzed",
                "details": "compared latency distributions",
                "result": "threshold too low",
                "agent_id": "agent-1",
                "timestamp": "2026-01-02T03:04:05Z"
            }]))),
            ("confidence", Value::Float64(0.7)),
            ("status", Value::Text(status.to_string())),
            ("agent_id", Value::Text("agent-1".into())),
            ("context_id", Value::Uuid(context_id)),
            ("properties", Value::Json(json!({"owner": "platform"}))),
            ("tags", Value::Json(json!(["decision", "ops"]))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();
}

fn insert_observation(db: &Database, id: Uuid, entity_id: Uuid, context_id: Uuid) {
    db.execute(
        "INSERT INTO observations (id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at) VALUES ($id, $observation_type, $observed_properties, $state_delta, $properties, $source, $entity_id, $context_id, $timestamp, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("observation_type", Value::Text("config_change".into())),
            (
                "observed_properties",
                Value::Json(json!({"version": "2.0"})),
            ),
            (
                "state_delta",
                Value::Json(json!({"version": {"old": "1.0", "new": "2.0"}})),
            ),
            ("properties", Value::Json(json!({"field": "version"}))),
            ("source", Value::Text("git-hook".into())),
            ("entity_id", Value::Uuid(entity_id)),
            ("context_id", Value::Uuid(context_id)),
            ("timestamp", Value::Timestamp(1_700_000_004)),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();
}

fn insert_evidence(
    db: &Database,
    id: Uuid,
    observation_id: Uuid,
    context_id: Uuid,
    supply_order: i64,
    derived_from: Option<Uuid>,
) {
    db.execute(
        "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, derived_from, retention_status, supply_order) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $derived_from, $retention_status, $supply_order)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("observation_id", Value::Uuid(observation_id)),
            ("context_id", Value::Uuid(context_id)),
            ("kind", Value::Text("text".into())),
            (
                "source_ref",
                Value::Text("cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()),
            ),
            (
                "derived_from",
                derived_from.map(Value::Uuid).unwrap_or(Value::Null),
            ),
            ("retention_status", Value::Text("retained_external".into())),
            ("supply_order", Value::Int64(supply_order)),
        ]),
    )
    .unwrap();
}

fn insert_outcome(db: &Database, id: Uuid, context_id: Uuid, decision_id: Uuid) {
    db.execute(
        "INSERT INTO outcomes (id, context_id, decision_id, success, notes, recorded_at) VALUES ($id, $context_id, $decision_id, $success, $notes, $recorded_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("context_id", Value::Uuid(context_id)),
            ("decision_id", Value::Uuid(decision_id)),
            ("success", Value::Bool(true)),
            ("notes", Value::Text("works as expected".into())),
            ("recorded_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();
}

fn insert_invalidation(
    db: &Database,
    id: Uuid,
    context_id: Uuid,
    status: &str,
    affected_decision_id: Uuid,
    trigger_observation_id: Uuid,
) {
    db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("status", Value::Text(status.to_string())),
            ("context_id", Value::Uuid(context_id)),
            ("affected_decision_id", Value::Uuid(affected_decision_id)),
            ("trigger_observation_id", Value::Uuid(trigger_observation_id)),
            ("basis_diff", Value::Json(json!({"field": "version", "old": "1.0", "new": "2.0"}))),
            ("severity", Value::Text("warning".into())),
            ("detected_at", Value::TxId(TxId(1_u64))),
            ("resolved_at", Value::TxId(TxId(2_u64))),
            ("resolution_action", Value::Text("new_decision_created".into())),
            ("resolution_decision_id", Value::Null),
        ]),
    )
    .unwrap();
}

fn insert_invalidation_status(db: &Database, id: Uuid, context_id: Uuid, status: &str) {
    let decision_id = Uuid::from_u128(604);
    let observation_id = Uuid::from_u128(605);
    insert_decision(db, decision_id, context_id, "active");
    insert_entity(db, Uuid::from_u128(606), context_id, "entity");
    insert_observation(db, observation_id, Uuid::from_u128(606), context_id);
    insert_invalidation(db, id, context_id, status, decision_id, observation_id);
}

fn insert_signature(db: &Database, id: Uuid, status: &str) {
    db.execute(
        "INSERT INTO signatures (id, signature_type, description, match_criteria, scope_labels, confidence, origin_context_ids, status, created_at, last_matched_at) VALUES ($id, $signature_type, $description, $match_criteria, $scope_labels, $confidence, $origin_context_ids, $status, $created_at, $last_matched_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("signature_type", Value::Text("frequency_anomaly".into())),
            ("description", Value::Text("Same person at multiple locations".into())),
            ("match_criteria", Value::Json(json!({"observation_types": ["person_detected"]}))),
            ("scope_labels", Value::Json(json!({"org": "bank-x"}))),
            ("confidence", Value::Float64(0.75)),
            ("origin_context_ids", Value::Json(json!([uuid(1)]))),
            ("status", Value::Text(status.to_string())),
            ("created_at", Value::TxId(TxId(1_u64))),
            ("last_matched_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();
}

fn insert_digest(db: &Database, id: Uuid, context_id: Uuid, embedding: Vec<f32>) {
    db.execute(
        "INSERT INTO digests (id, source, session_ref, summary, kind, vector_text, dimension, context_id, properties, created_at) VALUES ($id, $source, $session_ref, $summary, $kind, $vector_text, $dimension, $context_id, $properties, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("source", Value::Text("claude-cli".into())),
            ("session_ref", Value::Text("session-1".into())),
            ("summary", Value::Text("Discussed auth module refactoring".into())),
            ("kind", Value::Text("Text".into())),
            ("dimension", Value::Int64(embedding.len() as i64)),
            ("vector_text", Value::Vector(embedding)),
            ("context_id", Value::Uuid(context_id)),
            ("properties", Value::Json(json!({"modality": "text"}))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();
}

fn insert_blueprint(db: &Database, id: Uuid) {
    db.execute(
        "INSERT INTO blueprints (id, canonical_description, category, slots, default_config_shape, version, status, created_at) VALUES ($id, $canonical_description, $category, $slots, $default_config_shape, $version, $status, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("canonical_description", Value::Text("Detect entity at location during time window".into())),
            ("category", Value::Text("detection".into())),
            ("slots", Value::Json(json!({"entity_type": "SERVICE"}))),
            ("default_config_shape", Value::Json(json!({"window": "PT5M"}))),
            ("version", Value::Text("1.0.0".into())),
            ("status", Value::Text("active".into())),
            ("created_at", Value::Timestamp(1_700_000_011)),
        ]),
    )
    .unwrap();
}

fn insert_edge(
    db: &Database,
    id: Uuid,
    source: Uuid,
    target: Uuid,
    edge_type: &str,
    context_id: Uuid,
) {
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("source_id", Value::Uuid(source)),
            ("target_id", Value::Uuid(target)),
            ("edge_type", Value::Text(edge_type.to_string())),
            ("context_id", Value::Uuid(context_id)),
            ("valid_from", Value::TxId(TxId(1_u64))),
            ("valid_to", Value::Null),
            ("properties", Value::Json(json!({"weight": 1}))),
        ]),
    )
    .unwrap();
}

fn insert_snapshot(
    db: &Database,
    id: Uuid,
    context_id: Uuid,
    entity_id: Uuid,
    valid_from: i64,
    valid_to: i64,
) {
    db.execute(
        "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("context_id", Value::Uuid(context_id)),
            ("entity_id", Value::Uuid(entity_id)),
            ("data", Value::Json(json!({"version": valid_from}))),
            ("valid_from", Value::TxId(TxId(valid_from as u64))),
            ("valid_to", Value::TxId(TxId(valid_to as u64))),
        ]),
    )
    .unwrap();
}

fn insert_approval(db: &Database, id: Uuid, context_id: Uuid, decision_id: Uuid) {
    db.execute(
        "INSERT INTO approvals (id, context_id, decision_id, approver_id, level, rationale, created_at) VALUES ($id, $context_id, $decision_id, $approver_id, $level, $rationale, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("context_id", Value::Uuid(context_id)),
            ("decision_id", Value::Uuid(decision_id)),
            ("approver_id", Value::Text("approver-1".into())),
            ("level", Value::Text("team".into())),
            ("rationale", Value::Text("looks good".into())),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();
}

// contexts table round-trip: id, name, labels JSON, properties JSON, and created_at survive insert + select byte-exact.
#[test]
fn test_contexts_table_insert() {
    let db = fresh_db();
    let _ = now_ts();
    let id = Uuid::from_u128(1);
    db.execute(
        "INSERT INTO contexts (id, name, labels, properties, created_at) VALUES ($id, $name, $labels, $properties, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("name", Value::Text("project-alpha".into())),
            ("labels", Value::Json(json!({"org": "acme", "tier": "prod"}))),
            ("properties", Value::Json(json!({"retention_days": 30}))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();

    let result = db
        .execute(
            "SELECT * FROM contexts WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "name"),
        &Value::Text("project-alpha".into())
    );
    assert_eq!(
        row_value(&result, 0, "labels"),
        &Value::Json(json!({"org": "acme", "tier": "prod"}))
    );
    assert_eq!(
        row_value(&result, 0, "properties"),
        &Value::Json(json!({"retention_days": 30}))
    );
    assert_eq!(
        row_value(&result, 0, "created_at"),
        &Value::TxId(TxId(1_u64))
    );
}

// intentions table round-trip: every r3 §1 column (origin, origin_ref, blueprint_id, bindings, blueprint_proposal, tags) persists as declared.
#[test]
fn test_intentions_table_insert() {
    let db = fresh_db();
    let id = Uuid::from_u128(2);
    let context_id = Uuid::from_u128(20);
    insert_context(&db, context_id, "ctx");
    db.execute(
        "INSERT INTO intentions (id, description, status, context_id, origin, origin_ref, blueprint_id, bindings, blueprint_proposal, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $origin_ref, $blueprint_id, $bindings, $blueprint_proposal, $properties, $tags, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("description", Value::Text("Protect children from unauthorized pickup".into())),
            ("status", Value::Text("active".into())),
            ("context_id", Value::Uuid(context_id)),
            ("origin", Value::Text("human".into())),
            ("origin_ref", Value::Uuid(Uuid::from_u128(21))),
            ("blueprint_id", Value::Uuid(Uuid::from_u128(22))),
            ("bindings", Value::Json(json!({"slot": "value"}))),
            ("blueprint_proposal", Value::Text("proposal text".into())),
            ("properties", Value::Json(json!({"priority": "high"}))),
            ("tags", Value::Json(json!(["safety", "family"]))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();

    let result = db
        .execute(
            "SELECT * FROM intentions WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "description"),
        &Value::Text("Protect children from unauthorized pickup".into())
    );
    assert_eq!(
        row_value(&result, 0, "status"),
        &Value::Text("active".into())
    );
    assert_eq!(
        row_value(&result, 0, "context_id"),
        &Value::Uuid(context_id)
    );
    assert_eq!(
        row_value(&result, 0, "origin"),
        &Value::Text("human".into())
    );
    assert_eq!(
        row_value(&result, 0, "origin_ref"),
        &Value::Uuid(Uuid::from_u128(21))
    );
    assert_eq!(
        row_value(&result, 0, "blueprint_id"),
        &Value::Uuid(Uuid::from_u128(22))
    );
    assert_eq!(
        row_value(&result, 0, "bindings"),
        &Value::Json(json!({"slot": "value"}))
    );
    assert_eq!(
        row_value(&result, 0, "blueprint_proposal"),
        &Value::Text("proposal text".into())
    );
    assert_eq!(
        row_value(&result, 0, "properties"),
        &Value::Json(json!({"priority": "high"}))
    );
    assert_eq!(
        row_value(&result, 0, "tags"),
        &Value::Json(json!(["safety", "family"]))
    );
    assert_eq!(
        row_value(&result, 0, "created_at"),
        &Value::TxId(TxId(1_u64))
    );
}

// decisions table round-trip: decision_type, reasoning JSON, confidence, status, agent_id, tags, and context_id FK all persist.
#[test]
fn test_decisions_table_insert() {
    let db = fresh_db();
    let id = Uuid::from_u128(3);
    let context_id = Uuid::from_u128(30);
    insert_context(&db, context_id, "ctx");
    db.execute(
        "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("decision_type", Value::Text("architecture".into())),
            ("description", Value::Text("Use abstraction layer for fault isolation".into())),
            ("reasoning", Value::Json(json!([{ "operation": "analyzed", "details": "compared", "result": "threshold too low", "agent_id": "agent-1", "timestamp": "2026-01-02T03:04:05Z" }]))),
            ("confidence", Value::Float64(0.7)),
            ("status", Value::Text("active".into())),
            ("agent_id", Value::Text("agent-1".into())),
            ("context_id", Value::Uuid(context_id)),
            ("properties", Value::Json(json!({"owner": "platform"}))),
            ("tags", Value::Json(json!(["decision", "ops"]))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    )
    .unwrap();

    let result = db
        .execute(
            "SELECT * FROM decisions WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "decision_type"),
        &Value::Text("architecture".into())
    );
    assert_eq!(
        row_value(&result, 0, "description"),
        &Value::Text("Use abstraction layer for fault isolation".into())
    );
    assert_eq!(row_value(&result, 0, "confidence"), &Value::Float64(0.7));
    assert_eq!(
        row_value(&result, 0, "status"),
        &Value::Text("active".into())
    );
    assert_eq!(
        row_value(&result, 0, "agent_id"),
        &Value::Text("agent-1".into())
    );
    assert_eq!(
        row_value(&result, 0, "context_id"),
        &Value::Uuid(context_id)
    );
    assert_eq!(
        row_value(&result, 0, "properties"),
        &Value::Json(json!({"owner": "platform"}))
    );
    assert_eq!(
        row_value(&result, 0, "tags"),
        &Value::Json(json!(["decision", "ops"]))
    );
    assert_eq!(
        row_value(&result, 0, "created_at"),
        &Value::TxId(TxId(1_u64))
    );
}

// entities table round-trip: entity_type, name, context_id, properties JSON, and created_at survive insert + select.
#[test]
fn test_entities_table_insert() {
    let db = fresh_db();
    let id = Uuid::from_u128(4);
    let context_id = Uuid::from_u128(40);
    insert_context(&db, context_id, "ctx");
    insert_entity(&db, id, context_id, "payment-service");

    let result = db
        .execute(
            "SELECT * FROM entities WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "entity_type"),
        &Value::Text("SERVICE".into())
    );
    assert_eq!(
        row_value(&result, 0, "name"),
        &Value::Text("payment-service".into())
    );
    assert_eq!(
        row_value(&result, 0, "context_id"),
        &Value::Uuid(context_id)
    );
    assert_eq!(
        row_value(&result, 0, "properties"),
        &Value::Json(json!({"owner": "team-a"}))
    );
    assert_eq!(
        row_value(&result, 0, "created_at"),
        &Value::TxId(TxId(1_u64))
    );
}

// observations table round-trip: observation_type, property JSON, source, entity_id and context_id FKs, and timestamp persist.
#[test]
fn test_observations_table_insert() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(50);
    let entity_id = Uuid::from_u128(51);
    insert_context(&db, context_id, "ctx");
    insert_entity(&db, entity_id, context_id, "config");
    let id = Uuid::from_u128(5);
    insert_observation(&db, id, entity_id, context_id);

    let result = db
        .execute(
            "SELECT * FROM observations WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "observation_type"),
        &Value::Text("config_change".into())
    );
    assert_eq!(
        row_value(&result, 0, "observed_properties"),
        &Value::Json(json!({"version": "2.0"}))
    );
    assert_eq!(
        row_value(&result, 0, "state_delta"),
        &Value::Json(json!({"version": {"old": "1.0", "new": "2.0"}}))
    );
    assert_eq!(
        row_value(&result, 0, "source"),
        &Value::Text("git-hook".into())
    );
    assert_eq!(row_value(&result, 0, "entity_id"), &Value::Uuid(entity_id));
    assert_eq!(
        row_value(&result, 0, "context_id"),
        &Value::Uuid(context_id)
    );
    assert_eq!(
        row_value(&result, 0, "timestamp"),
        &Value::Timestamp(1_700_000_004)
    );
}

// outcomes table round-trip: decision_id FK, success bool, notes, and recorded_at persist.
#[test]
fn test_outcomes_table_insert() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(60);
    insert_context(&db, context_id, "ctx");
    let decision_id = Uuid::from_u128(61);
    insert_decision(&db, decision_id, context_id, "active");
    let id = Uuid::from_u128(6);
    insert_outcome(&db, id, context_id, decision_id);

    let result = db
        .execute(
            "SELECT * FROM outcomes WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "decision_id"),
        &Value::Uuid(decision_id)
    );
    assert_eq!(row_value(&result, 0, "success"), &Value::Bool(true));
    assert_eq!(
        row_value(&result, 0, "notes"),
        &Value::Text("works as expected".into())
    );
    assert_eq!(
        row_value(&result, 0, "recorded_at"),
        &Value::TxId(TxId(1_u64))
    );
}

// invalidations table round-trip: status, affected_decision_id, trigger_observation_id, basis_diff JSON, severity, and resolution columns persist.
#[test]
fn test_invalidations_table_insert() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(70);
    insert_context(&db, context_id, "ctx");
    let decision_id = Uuid::from_u128(71);
    let observation_id = Uuid::from_u128(72);
    insert_decision(&db, decision_id, context_id, "active");
    insert_entity(&db, Uuid::from_u128(73), context_id, "entity");
    insert_observation(&db, observation_id, Uuid::from_u128(73), context_id);
    let id = Uuid::from_u128(7);
    insert_invalidation(&db, id, context_id, "pending", decision_id, observation_id);

    let result = db
        .execute(
            "SELECT * FROM invalidations WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "status"),
        &Value::Text("pending".into())
    );
    assert_eq!(
        row_value(&result, 0, "context_id"),
        &Value::Uuid(context_id)
    );
    assert_eq!(
        row_value(&result, 0, "affected_decision_id"),
        &Value::Uuid(decision_id)
    );
    assert_eq!(
        row_value(&result, 0, "trigger_observation_id"),
        &Value::Uuid(observation_id)
    );
    assert_eq!(
        row_value(&result, 0, "basis_diff"),
        &Value::Json(json!({"field": "version", "old": "1.0", "new": "2.0"}))
    );
    assert_eq!(
        row_value(&result, 0, "severity"),
        &Value::Text("warning".into())
    );
    assert_eq!(
        row_value(&result, 0, "detected_at"),
        &Value::TxId(TxId(1_u64))
    );
    assert_eq!(
        row_value(&result, 0, "resolved_at"),
        &Value::TxId(TxId(2_u64))
    );
    assert_eq!(
        row_value(&result, 0, "resolution_action"),
        &Value::Text("new_decision_created".into())
    );
    assert_eq!(
        row_value(&result, 0, "resolution_decision_id"),
        &Value::Null
    );
}

#[test]
fn combination_invalidations_unique_per_context_affected_decision_collapses_raw_duplicate() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(20_060);
    let entity_id = Uuid::from_u128(20_061);
    let decision_id = Uuid::from_u128(20_062);
    let observation_id = Uuid::from_u128(20_063);
    insert_context(&db, context_id, "ctx-invalidation-unique");
    insert_entity(&db, entity_id, context_id, "entity-invalidation-unique");
    insert_decision(&db, decision_id, context_id, "active");
    insert_observation(&db, observation_id, entity_id, context_id);
    insert_invalidation(
        &db,
        Uuid::from_u128(20_064),
        context_id,
        "pending",
        decision_id,
        observation_id,
    );

    let duplicate = db
        .execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(20_065))),
            ("status", Value::Text("pending".into())),
            ("context_id", Value::Uuid(context_id)),
            ("affected_decision_id", Value::Uuid(decision_id)),
            ("trigger_observation_id", Value::Uuid(observation_id)),
            ("basis_diff", Value::Json(json!({"field": "risk"}))),
            ("severity", Value::Text("critical".into())),
            ("detected_at", Value::TxId(TxId(3_u64))),
        ]),
    )
    .expect("duplicate insert is a contextdb unique no-op");
    assert_eq!(duplicate.rows_affected, 0);
    assert_eq!(
        scalar_count(
            &db,
            "SELECT COUNT(*) AS n FROM invalidations WHERE affected_decision_id = $id",
            &params(vec![("id", Value::Uuid(decision_id))]),
        ),
        1
    );
}

// signatures table round-trip: signature_type, match_criteria JSON, scope_labels, origin_context_ids, confidence, and status persist.
#[test]
fn test_signatures_table_insert() {
    let db = fresh_db();
    let id = Uuid::from_u128(8);
    insert_signature(&db, id, "active");

    let result = db
        .execute(
            "SELECT * FROM signatures WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "signature_type"),
        &Value::Text("frequency_anomaly".into())
    );
    assert_eq!(
        row_value(&result, 0, "description"),
        &Value::Text("Same person at multiple locations".into())
    );
    assert_eq!(
        row_value(&result, 0, "match_criteria"),
        &Value::Json(json!({"observation_types": ["person_detected"]}))
    );
    assert_eq!(
        row_value(&result, 0, "scope_labels"),
        &Value::Json(json!({"org": "bank-x"}))
    );
    assert_eq!(row_value(&result, 0, "confidence"), &Value::Float64(0.75));
    assert_eq!(
        row_value(&result, 0, "origin_context_ids"),
        &Value::Json(json!([uuid(1)]))
    );
    assert_eq!(
        row_value(&result, 0, "status"),
        &Value::Text("active".into())
    );
    assert_eq!(
        row_value(&result, 0, "created_at"),
        &Value::TxId(TxId(1_u64))
    );
    assert_eq!(
        row_value(&result, 0, "last_matched_at"),
        &Value::TxId(TxId(1_u64))
    );
}

// digests table round-trip: source, session_ref, summary, text vector, context_id FK, properties, and created_at persist.
#[test]
fn test_digests_table_insert() {
    let db = fresh_db();
    let id = Uuid::from_u128(9);
    let context_id = Uuid::from_u128(90);
    insert_context(&db, context_id, "ctx");
    let embedding = vec![0.1_f32; DIGEST_VECTOR_DIMENSION];
    insert_digest(&db, id, context_id, embedding.clone());

    let result = db
        .execute(
            "SELECT * FROM digests WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "source"),
        &Value::Text("claude-cli".into())
    );
    assert_eq!(
        row_value(&result, 0, "session_ref"),
        &Value::Text("session-1".into())
    );
    assert_eq!(
        row_value(&result, 0, "summary"),
        &Value::Text("Discussed auth module refactoring".into())
    );
    assert_eq!(
        row_value(&result, 0, "vector_text"),
        &Value::Vector(embedding)
    );
    assert_eq!(
        row_value(&result, 0, "dimension"),
        &Value::Int64(DIGEST_VECTOR_DIMENSION as i64)
    );
    assert_eq!(
        row_value(&result, 0, "context_id"),
        &Value::Uuid(context_id)
    );
    assert_eq!(
        row_value(&result, 0, "properties"),
        &Value::Json(json!({"modality": "text"}))
    );
    assert_eq!(
        row_value(&result, 0, "created_at"),
        &Value::TxId(TxId(1_u64))
    );
}

// blueprints table round-trip: canonical_description, category, slots JSON, default_config_shape, version, and status persist.
#[test]
fn test_blueprints_table_insert() {
    let db = fresh_db();
    let id = Uuid::from_u128(10);
    insert_blueprint(&db, id);

    let result = db
        .execute(
            "SELECT * FROM blueprints WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "canonical_description"),
        &Value::Text("Detect entity at location during time window".into())
    );
    assert_eq!(
        row_value(&result, 0, "category"),
        &Value::Text("detection".into())
    );
    assert_eq!(
        row_value(&result, 0, "slots"),
        &Value::Json(json!({"entity_type": "SERVICE"}))
    );
    assert_eq!(
        row_value(&result, 0, "default_config_shape"),
        &Value::Json(json!({"window": "PT5M"}))
    );
    assert_eq!(
        row_value(&result, 0, "version"),
        &Value::Text("1.0.0".into())
    );
    assert_eq!(
        row_value(&result, 0, "status"),
        &Value::Text("active".into())
    );
    assert_eq!(
        row_value(&result, 0, "created_at"),
        &Value::Timestamp(1_700_000_011)
    );
}

// edges table accepts every r3 relationship label (SERVES, BASED_ON, ABOUT, CITES, SUPERSEDES, RELATES_TO, OBSERVED_ON, HAS_OUTCOME, INVALIDATES, TRIGGERED_BY, MATCHED_BY, GENERATED, EXTRACTED_FROM, INSTANTIATES, REFINES); removing a label at the schema layer silently drops a relationship kind.
#[test]
fn test_edges_table_all_relationship_types() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(99);
    insert_context(&db, context_id, "ctx");
    let source = Uuid::from_u128(100);
    let mut expected_types = Vec::new();

    for (i, edge_type) in [
        "SERVES",
        "BASED_ON",
        "ABOUT",
        "CITES",
        "SUPERSEDES",
        "RELATES_TO",
        "OBSERVED_ON",
        "HAS_OUTCOME",
        "INVALIDATES",
        "TRIGGERED_BY",
        "MATCHED_BY",
        "GENERATED",
        "EXTRACTED_FROM",
        "INSTANTIATES",
        "REFINES",
    ]
    .iter()
    .enumerate()
    {
        expected_types.push((*edge_type).to_string());
        db.execute(
            "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
            &params(vec![
                ("id", Value::Uuid(Uuid::from_u128(1000 + i as u128))),
                ("source_id", Value::Uuid(source)),
                ("target_id", Value::Uuid(Uuid::from_u128(200 + i as u128))),
                ("edge_type", Value::Text(edge_type.to_string())),
                ("context_id", Value::Uuid(context_id)),
                ("valid_from", Value::TxId(TxId(1_u64))),
                ("valid_to", Value::Null),
                ("properties", Value::Json(json!({"rank": i}))),
            ]),
        )
        .unwrap();
    }

    let result = db.execute("SELECT * FROM edges", &HashMap::new()).unwrap();
    assert_eq!(result.rows.len(), 15);
    for edge_type in expected_types {
        assert!(result.rows.iter().any(|row| {
            row_value(
                &result,
                result
                    .rows
                    .iter()
                    .position(|candidate| candidate == row)
                    .unwrap(),
                "edge_type",
            ) == &Value::Text(edge_type.clone())
        }));
    }
}

// entity_snapshots table round-trip: entity_id, data JSON, and valid_from/valid_to txid integers persist as declared.
#[test]
fn test_entity_snapshots_table_insert() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(299);
    let entity_id = Uuid::from_u128(300);
    let snapshot_id = Uuid::from_u128(301);
    insert_context(&db, context_id, "ctx");
    insert_entity(&db, entity_id, context_id, "entity");
    insert_snapshot(&db, snapshot_id, context_id, entity_id, 1, 2);

    let result = db
        .execute(
            "SELECT * FROM entity_snapshots WHERE id = $id",
            &params(vec![("id", Value::Uuid(snapshot_id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(row_value(&result, 0, "entity_id"), &Value::Uuid(entity_id));
    assert_eq!(
        row_value(&result, 0, "valid_from"),
        &Value::TxId(TxId(1_u64))
    );
    assert_eq!(row_value(&result, 0, "valid_to"), &Value::TxId(TxId(2_u64)));
}

#[test]
fn combination_entity_snapshot_links_reject_cross_entity_successor_by_composite_fk() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(20_070);
    let entity_a = Uuid::from_u128(20_071);
    let entity_b = Uuid::from_u128(20_072);
    let snapshot_a = Uuid::from_u128(20_073);
    let snapshot_b = Uuid::from_u128(20_074);
    insert_context(&db, context_id, "ctx-snapshot-link-composite");
    insert_entity(&db, entity_a, context_id, "entity-a");
    insert_entity(&db, entity_b, context_id, "entity-b");
    insert_snapshot(&db, snapshot_a, context_id, entity_a, 1, 2);
    insert_snapshot(&db, snapshot_b, context_id, entity_b, 2, 3);

    let result = db.execute(
        "INSERT INTO entity_snapshot_links (id, context_id, entity_id, predecessor_snapshot_id, successor_snapshot_id, created_at) VALUES ($id, $context_id, $entity_id, $predecessor_snapshot_id, $successor_snapshot_id, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(20_075))),
            ("context_id", Value::Uuid(context_id)),
            ("entity_id", Value::Uuid(entity_a)),
            ("predecessor_snapshot_id", Value::Uuid(snapshot_a)),
            ("successor_snapshot_id", Value::Uuid(snapshot_b)),
            ("created_at", Value::TxId(TxId(3_u64))),
        ]),
    );
    assert_error_is_fk(result);
}

#[test]
fn combination_entity_current_snapshots_reject_cross_entity_snapshot_by_composite_fk() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(20_076);
    let entity_a = Uuid::from_u128(20_077);
    let entity_b = Uuid::from_u128(20_078);
    let snapshot_b = Uuid::from_u128(20_079);
    insert_context(&db, context_id, "ctx-current-snapshot-composite");
    insert_entity(&db, entity_a, context_id, "entity-a");
    insert_entity(&db, entity_b, context_id, "entity-b");
    insert_snapshot(&db, snapshot_b, context_id, entity_b, 1, 2);

    let result = db.execute(
        "INSERT INTO entity_current_snapshots (context_id, entity_id, snapshot_id, updated_at) VALUES ($context_id, $entity_id, $snapshot_id, $updated_at)",
        &params(vec![
            ("context_id", Value::Uuid(context_id)),
            ("entity_id", Value::Uuid(entity_a)),
            ("snapshot_id", Value::Uuid(snapshot_b)),
            ("updated_at", Value::TxId(TxId(2_u64))),
        ]),
    );
    assert_error_is_fk(result);
}

// intention_snapshots round-trip: intention_id FK, description, status, origin, properties JSON, tags JSON, valid_from and valid_to TXIDs survive insert + select.
#[test]
fn test_intention_snapshots_table_insert() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(310);
    let intention_id = Uuid::from_u128(311);
    let snapshot_id = Uuid::from_u128(312);
    insert_context(&db, context_id, "ctx");
    insert_intention(&db, intention_id, context_id, "active");
    db.execute(
        "INSERT INTO intention_snapshots (id, context_id, intention_id, description, status, origin, properties, tags, valid_from, valid_to) VALUES ($id, $context_id, $intention_id, $description, $status, $origin, $properties, $tags, $valid_from, $valid_to)",
        &params(vec![
            ("id", Value::Uuid(snapshot_id)),
            ("context_id", Value::Uuid(context_id)),
            ("intention_id", Value::Uuid(intention_id)),
            ("description", Value::Text("initial framing".into())),
            ("status", Value::Text("active".into())),
            ("origin", Value::Text("human".into())),
            ("properties", Value::Json(json!({"priority": "high"}))),
            ("tags", Value::Json(json!(["safety"]))),
            ("valid_from", Value::TxId(TxId(1_u64))),
            ("valid_to", Value::TxId(TxId(2_u64))),
        ]),
    )
    .unwrap();

    let result = db
        .execute(
            "SELECT * FROM intention_snapshots WHERE id = $id",
            &params(vec![("id", Value::Uuid(snapshot_id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "intention_id"),
        &Value::Uuid(intention_id)
    );
    assert_eq!(
        row_value(&result, 0, "description"),
        &Value::Text("initial framing".into())
    );
    assert_eq!(
        row_value(&result, 0, "status"),
        &Value::Text("active".into())
    );
    assert_eq!(
        row_value(&result, 0, "origin"),
        &Value::Text("human".into())
    );
    assert_eq!(
        row_value(&result, 0, "properties"),
        &Value::Json(json!({"priority": "high"}))
    );
    assert_eq!(
        row_value(&result, 0, "tags"),
        &Value::Json(json!(["safety"]))
    );
    assert_eq!(
        row_value(&result, 0, "valid_from"),
        &Value::TxId(TxId(1_u64))
    );
    assert_eq!(row_value(&result, 0, "valid_to"), &Value::TxId(TxId(2_u64)));
}

// intention_snapshots.intention_id FK into intentions.id must hold; a row pointing at a non-existent intention must fail with ForeignKeyViolation.
#[test]
fn test_fk_intention_snapshot_invalid_intention_id_rejected() {
    let db = fresh_db();
    assert_error_is_fk(db.execute(
        "INSERT INTO intention_snapshots (id, context_id, intention_id, description, status, origin, properties, tags, valid_from, valid_to) VALUES ($id, $context_id, $intention_id, $description, $status, $origin, $properties, $tags, $valid_from, $valid_to)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(320))),
            ("context_id", Value::Uuid(Uuid::from_u128(319))),
            ("intention_id", Value::Uuid(Uuid::from_u128(321))),
            ("description", Value::Text("stranded".into())),
            ("status", Value::Text("active".into())),
            ("origin", Value::Text("human".into())),
            ("properties", Value::Json(json!({}))),
            ("tags", Value::Json(json!([]))),
            ("valid_from", Value::TxId(TxId(1_u64))),
            ("valid_to", Value::Null),
        ]),
    ));
}

// approvals table round-trip: decision_id FK, approver_id, level, rationale, and created_at persist.
#[test]
fn test_approvals_table_insert() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(400);
    insert_context(&db, context_id, "ctx");
    let decision_id = Uuid::from_u128(401);
    insert_decision(&db, decision_id, context_id, "active");
    let id = Uuid::from_u128(11);
    insert_approval(&db, id, context_id, decision_id);

    let result = db
        .execute(
            "SELECT * FROM approvals WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    assert_eq!(
        row_value(&result, 0, "decision_id"),
        &Value::Uuid(decision_id)
    );
    assert_eq!(
        row_value(&result, 0, "approver_id"),
        &Value::Text("approver-1".into())
    );
    assert_eq!(row_value(&result, 0, "level"), &Value::Text("team".into()));
    assert_eq!(
        row_value(&result, 0, "rationale"),
        &Value::Text("looks good".into())
    );
    assert_eq!(
        row_value(&result, 0, "created_at"),
        &Value::TxId(TxId(1_u64))
    );
}

// observations carries TableMeta.immutable=true at the engine layer (Invariant 4); a raw UPDATE must fail with Error::ImmutableTable, not merely be blocked higher up.
#[test]
fn test_observation_immutable_rejects_update() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(500);
    let entity_id = Uuid::from_u128(501);
    let obs_id = Uuid::from_u128(502);
    insert_context(&db, context_id, "ctx");
    insert_entity(&db, entity_id, context_id, "entity");
    insert_observation(&db, obs_id, entity_id, context_id);

    let result = db.execute(
        "UPDATE observations SET source = $source WHERE id = $id",
        &params(vec![
            ("source", Value::Text("other".into())),
            ("id", Value::Uuid(obs_id)),
        ]),
    );
    assert!(matches!(result, Err(Error::ImmutableTable(_))));
}

// observations rows cannot be physically removed (Invariant 3 + Invariant 4); a raw DELETE must return Error::ImmutableTable from the engine.
#[test]
fn test_observation_immutable_rejects_delete() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(503);
    let entity_id = Uuid::from_u128(504);
    let obs_id = Uuid::from_u128(505);
    insert_context(&db, context_id, "ctx");
    insert_entity(&db, entity_id, context_id, "entity");
    insert_observation(&db, obs_id, entity_id, context_id);

    let result = db.execute(
        "DELETE FROM observations WHERE id = $id",
        &params(vec![("id", Value::Uuid(obs_id))]),
    );
    assert!(matches!(result, Err(Error::ImmutableTable(_))));
}

// digests table is immutable at the schema layer (Invariant 16); a raw UPDATE of summary must fail with Error::ImmutableTable.
#[test]
fn test_digest_immutable_rejects_update() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(506);
    let id = Uuid::from_u128(507);
    insert_context(&db, context_id, "ctx");
    insert_digest(&db, id, context_id, vec![0.1; DIGEST_VECTOR_DIMENSION]);

    let result = db.execute(
        "UPDATE digests SET summary = $summary WHERE id = $id",
        &params(vec![
            ("summary", Value::Text("changed".into())),
            ("id", Value::Uuid(id)),
        ]),
    );
    assert!(matches!(result, Err(Error::ImmutableTable(_))));
}

// digests rows cannot be deleted because downstream extractions hold EXTRACTED_FROM edges back to them; the engine must refuse raw DELETE with Error::ImmutableTable.
#[test]
fn test_digest_immutable_rejects_delete() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(508);
    let id = Uuid::from_u128(509);
    insert_context(&db, context_id, "ctx");
    insert_digest(&db, id, context_id, vec![0.1; DIGEST_VECTOR_DIMENSION]);

    let result = db.execute(
        "DELETE FROM digests WHERE id = $id",
        &params(vec![("id", Value::Uuid(id))]),
    );
    assert!(matches!(result, Err(Error::ImmutableTable(_))));
}

macro_rules! allowed_status_test {
    ($name:ident, $insert_fn:ident, $table:literal, $start:literal, $next:literal) => {
        #[test]
        fn $name() {
            let db = fresh_db();
            assert!(
                db.table_meta($table).unwrap().state_machine.is_some(),
                "expected {} to define a state machine",
                $table
            );
            let context_id = Uuid::from_u128(600);
            let id = Uuid::from_u128(601);
            insert_context(&db, context_id, "ctx");
            $insert_fn(&db, id, context_id, $start);
            let update = db.execute(
                &format!("UPDATE {} SET status = $status WHERE id = $id", $table),
                &params(vec![
                    ("status", Value::Text($next.into())),
                    ("id", Value::Uuid(id)),
                ]),
            );
            assert!(update.is_ok());
            let result = db
                .execute(
                    &format!("SELECT status FROM {} WHERE id = $id", $table),
                    &params(vec![("id", Value::Uuid(id))]),
                )
                .unwrap();
            assert_one_row(&result);
            assert_eq!(row_value(&result, 0, "status"), &Value::Text($next.into()));
        }
    };
}

macro_rules! rejected_status_test {
    ($name:ident, $insert_fn:ident, $table:literal, $start:literal, $next:literal) => {
        #[test]
        fn $name() {
            let db = fresh_db();
            let context_id = Uuid::from_u128(602);
            let id = Uuid::from_u128(603);
            insert_context(&db, context_id, "ctx");
            $insert_fn(&db, id, context_id, $start);
            let update = db.execute(
                &format!("UPDATE {} SET status = $status WHERE id = $id", $table),
                &params(vec![
                    ("status", Value::Text($next.into())),
                    ("id", Value::Uuid(id)),
                ]),
            );
            assert!(update.is_err());
        }
    };
}

// intentions state machine permits Active -> Paused so a goal can be put on hold without losing SERVES edges.
allowed_status_test!(
    test_intention_status_active_to_paused,
    insert_intention,
    "intentions",
    "active",
    "paused"
);
// intentions state machine permits Active -> Archived so concluded goals leave the active listing without hard-delete.
allowed_status_test!(
    test_intention_status_active_to_archived,
    insert_intention,
    "intentions",
    "active",
    "archived"
);
// intentions state machine permits Active -> Superseded so a refined Intention can replace a broader one.
allowed_status_test!(
    test_intention_status_active_to_superseded,
    insert_intention,
    "intentions",
    "active",
    "superseded"
);
// intentions state machine permits Paused -> Active so a resumed goal re-enters the active listing.
allowed_status_test!(
    test_intention_status_paused_to_active,
    insert_intention,
    "intentions",
    "paused",
    "active"
);
// intentions state machine permits Paused -> Superseded so a hold can be resolved by a refinement instead of resumed.
allowed_status_test!(
    test_intention_status_paused_to_superseded,
    insert_intention,
    "intentions",
    "paused",
    "superseded"
);
// intentions state machine permits Paused -> Archived so on-hold goals can be retired without resuming first.
allowed_status_test!(
    test_intention_status_paused_to_archived,
    insert_intention,
    "intentions",
    "paused",
    "archived"
);
// intentions state machine refuses Archived -> Active so retired goals cannot silently come back to life.
rejected_status_test!(
    test_intention_status_archived_to_active_rejected,
    insert_intention,
    "intentions",
    "archived",
    "active"
);
// intentions state machine refuses Superseded -> Active so a superseded goal cannot reclaim SERVES edges that its replacement owns.
rejected_status_test!(
    test_intention_status_superseded_to_active_rejected,
    insert_intention,
    "intentions",
    "superseded",
    "active"
);
// intentions state machine refuses Superseded -> Paused so a replaced goal cannot re-enter the held pool.
rejected_status_test!(
    test_intention_status_superseded_to_paused_rejected,
    insert_intention,
    "intentions",
    "superseded",
    "paused"
);
// intentions state machine refuses Archived -> Superseded so archived goals stay archived and cannot be rewritten as supersession targets.
rejected_status_test!(
    test_intention_status_archived_to_superseded_rejected,
    insert_intention,
    "intentions",
    "archived",
    "superseded"
);

// decisions state machine permits Active -> Superseded so supersede_decision can flip the old row atomically with the new edge.
allowed_status_test!(
    test_decision_status_active_to_superseded,
    insert_decision,
    "decisions",
    "active",
    "superseded"
);
// decisions state machine permits Active -> Archived so archive_decision has a legal destination state.
allowed_status_test!(
    test_decision_status_active_to_archived,
    insert_decision,
    "decisions",
    "active",
    "archived"
);
// decisions state machine refuses Superseded -> Active so a replaced Decision cannot re-enter the active set via a status flip.
rejected_status_test!(
    test_decision_status_superseded_to_active_rejected,
    insert_decision,
    "decisions",
    "superseded",
    "active"
);
// decisions state machine refuses Archived -> Active so archived Decisions cannot be quietly reactivated without a fresh replacement.
rejected_status_test!(
    test_decision_status_archived_to_active_rejected,
    insert_decision,
    "decisions",
    "archived",
    "active"
);
// decisions state machine refuses Archived -> Superseded so an archived Decision is a terminal state, not a starting point for supersession.
rejected_status_test!(
    test_decision_status_archived_to_superseded_rejected,
    insert_decision,
    "decisions",
    "archived",
    "superseded"
);
// decisions state machine refuses Superseded -> Archived so a replaced Decision keeps its supersession provenance instead of being reclassified.
rejected_status_test!(
    test_decision_status_superseded_to_archived_rejected,
    insert_decision,
    "decisions",
    "superseded",
    "archived"
);

// invalidations state machine permits Pending -> Acknowledged so a reviewer can stage a response without resolving it yet.
allowed_status_test!(
    test_invalidation_status_pending_to_acknowledged,
    insert_invalidation_status,
    "invalidations",
    "pending",
    "acknowledged"
);
// invalidations state machine permits Acknowledged -> Resolved so a reviewer closes the invalidation by linking a new Decision.
allowed_status_test!(
    test_invalidation_status_acknowledged_to_resolved,
    insert_invalidation_status,
    "invalidations",
    "acknowledged",
    "resolved"
);
// invalidations state machine permits Acknowledged -> Dismissed so a reviewer can judge an invalidation irrelevant after inspection.
allowed_status_test!(
    test_invalidation_status_acknowledged_to_dismissed,
    insert_invalidation_status,
    "invalidations",
    "acknowledged",
    "dismissed"
);
// invalidations state machine refuses Resolved -> Pending so a closed-out invalidation cannot silently reopen.
rejected_status_test!(
    test_invalidation_status_resolved_to_pending_rejected,
    insert_invalidation_status,
    "invalidations",
    "resolved",
    "pending"
);
// invalidations state machine refuses Dismissed -> Acknowledged so a dismissed finding cannot be quietly re-queued for review.
rejected_status_test!(
    test_invalidation_status_dismissed_to_acknowledged_rejected,
    insert_invalidation_status,
    "invalidations",
    "dismissed",
    "acknowledged"
);
// invalidations state machine refuses Pending -> Resolved so resolution requires passing through Acknowledged, keeping the review checkpoint mandatory.
rejected_status_test!(
    test_invalidation_status_pending_to_resolved_rejected,
    insert_invalidation_status,
    "invalidations",
    "pending",
    "resolved"
);
// invalidations state machine refuses Pending -> Dismissed so dismissal requires explicit acknowledgement, preventing silent suppression of findings.
rejected_status_test!(
    test_invalidation_status_pending_to_dismissed_rejected,
    insert_invalidation_status,
    "invalidations",
    "pending",
    "dismissed"
);
// invalidations state machine refuses Dismissed -> Resolved so dismissed findings cannot be relabeled as resolutions and inflate the resolved count.
rejected_status_test!(
    test_invalidation_status_dismissed_to_resolved_rejected,
    insert_invalidation_status,
    "invalidations",
    "dismissed",
    "resolved"
);

// edges of type CITES must form a DAG (Invariant 5); a direct back-edge that would close a 2-node cycle must be rejected at insert.
#[test]
fn test_cites_dag_rejects_cycle() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(699);
    insert_context(&db, context_id, "ctx");
    let source_a = Uuid::from_u128(700);
    let source_b = Uuid::from_u128(701);
    insert_edge(
        &db,
        Uuid::from_u128(702),
        source_a,
        source_b,
        "CITES",
        context_id,
    );
    let result = db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(703))),
            ("source_id", Value::Uuid(source_b)),
            ("target_id", Value::Uuid(source_a)),
            ("edge_type", Value::Text("CITES".into())),
            ("context_id", Value::Uuid(context_id)),
            ("valid_from", Value::TxId(TxId(1_u64))),
            ("valid_to", Value::Null),
            ("properties", Value::Json(json!({}))),
        ]),
    );
    assert!(result.is_err());
}

// a CITES edge from a node to itself is a degenerate cycle and must be refused at insert so precedent_chain cannot spin forever.
#[test]
fn test_cites_dag_rejects_self_cycle() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(7030);
    insert_context(&db, context_id, "ctx");
    let node = Uuid::from_u128(704);
    let result = db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(705))),
            ("source_id", Value::Uuid(node)),
            ("target_id", Value::Uuid(node)),
            ("edge_type", Value::Text("CITES".into())),
            ("context_id", Value::Uuid(context_id)),
            ("valid_from", Value::TxId(TxId(1_u64))),
            ("valid_to", Value::Null),
            ("properties", Value::Json(json!({}))),
        ]),
    );
    assert!(result.is_err());
}

// the CITES DAG guard must detect transitive cycles (A->B->C->A), not merely direct back-edges, so reachability cannot create a loop.
#[test]
fn test_cites_dag_rejects_transitive_cycle() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(7050);
    insert_context(&db, context_id, "ctx");
    let a = Uuid::from_u128(706);
    let b = Uuid::from_u128(707);
    let c = Uuid::from_u128(708);
    insert_edge(&db, Uuid::from_u128(709), a, b, "CITES", context_id);
    insert_edge(&db, Uuid::from_u128(710), b, c, "CITES", context_id);
    let result = db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(711))),
            ("source_id", Value::Uuid(c)),
            ("target_id", Value::Uuid(a)),
            ("edge_type", Value::Text("CITES".into())),
            ("context_id", Value::Uuid(context_id)),
            ("valid_from", Value::TxId(TxId(1_u64))),
            ("valid_to", Value::Null),
            ("properties", Value::Json(json!({}))),
        ]),
    );
    assert!(result.is_err());
}

// the DAG guard is scoped to CITES only; typed edges such as RELATES_TO must accept both directions between the same pair without being misclassified as cycles.
#[test]
fn test_non_cites_edge_allows_bidirectional() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(7110);
    insert_context(&db, context_id, "ctx");
    let a = Uuid::from_u128(712);
    let b = Uuid::from_u128(713);
    insert_edge(&db, Uuid::from_u128(714), a, b, "RELATES_TO", context_id);
    insert_edge(&db, Uuid::from_u128(715), b, a, "RELATES_TO", context_id);

    let result = db
        .execute(
            "SELECT * FROM edges WHERE edge_type = $edge_type",
            &params(vec![("edge_type", Value::Text("RELATES_TO".into()))]),
        )
        .unwrap();
    assert_eq!(result.rows.len(), 2);
}

// edges carry a UNIQUE constraint on (source_id, target_id, edge_type); a duplicate insert succeeds as a no-op so add_relationship idempotency is engine-native (Invariant 10).
#[test]
fn test_edge_duplicate_insert_noop() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(7150);
    insert_context(&db, context_id, "ctx");
    let a = Uuid::from_u128(716);
    let b = Uuid::from_u128(717);
    insert_edge(&db, Uuid::from_u128(718), a, b, "SERVES", context_id);
    let second = db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(719))),
            ("source_id", Value::Uuid(a)),
            ("target_id", Value::Uuid(b)),
            ("edge_type", Value::Text("SERVES".into())),
            ("context_id", Value::Uuid(context_id)),
            ("valid_from", Value::TxId(TxId(1_u64))),
            ("valid_to", Value::Null),
            ("properties", Value::Json(json!({"weight": 1}))),
        ]),
    );
    assert!(
        second.is_ok(),
        "duplicate edge insert should no-op: {second:?}"
    );
    let result = db.execute(
        "SELECT * FROM edges WHERE source_id = $source_id AND target_id = $target_id AND edge_type = $edge_type",
        &params(vec![
            ("source_id", Value::Uuid(a)),
            ("target_id", Value::Uuid(b)),
            ("edge_type", Value::Text("SERVES".into())),
        ]),
    )
    .unwrap();
    assert_eq!(result.rows.len(), 1);
}

// entities.context_id column supports engine-side equality filtering so list_entities scoped to one context cannot bleed rows from another (Invariant 7).
#[test]
fn test_entities_filter_by_context_id() {
    let db = fresh_db();
    let ctx_a = Uuid::from_u128(800);
    let ctx_b = Uuid::from_u128(801);
    insert_context(&db, ctx_a, "client-a");
    insert_context(&db, ctx_b, "client-b");
    insert_entity(&db, Uuid::from_u128(802), ctx_a, "api");
    insert_entity(&db, Uuid::from_u128(803), ctx_b, "api");

    let result = db
        .execute(
            "SELECT * FROM entities WHERE context_id = $context_id",
            &params(vec![("context_id", Value::Uuid(ctx_b))]),
        )
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(row_value(&result, 0, "context_id"), &Value::Uuid(ctx_b));
}

// edges.context_id column supports engine-side equality filtering so traverse and impact-analysis scopes never see edges authored under a sibling context.
#[test]
fn test_edges_filter_by_context_id() {
    let db = fresh_db();
    let ctx_a = Uuid::from_u128(804);
    let ctx_b = Uuid::from_u128(805);
    let entity_a = Uuid::from_u128(806);
    let entity_b = Uuid::from_u128(807);
    insert_context(&db, ctx_a, "client-a");
    insert_context(&db, ctx_b, "client-b");
    insert_entity(&db, entity_a, ctx_a, "api");
    insert_entity(&db, entity_b, ctx_b, "api");
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(808))),
            ("source_id", Value::Uuid(entity_a)),
            ("target_id", Value::Uuid(Uuid::from_u128(809))),
            ("edge_type", Value::Text("RELATES_TO".into())),
            ("context_id", Value::Uuid(ctx_a)),
            ("valid_from", Value::TxId(TxId(1_u64))),
            ("valid_to", Value::Null),
            ("properties", Value::Json(json!({}))),
        ]),
    )
    .unwrap();
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(810))),
            ("source_id", Value::Uuid(entity_b)),
            ("target_id", Value::Uuid(Uuid::from_u128(811))),
            ("edge_type", Value::Text("RELATES_TO".into())),
            ("context_id", Value::Uuid(ctx_b)),
            ("valid_from", Value::TxId(TxId(1_u64))),
            ("valid_to", Value::Null),
            ("properties", Value::Json(json!({}))),
        ]),
    )
    .unwrap();

    let result = db
        .execute(
            "SELECT * FROM edges WHERE context_id = $context_id",
            &params(vec![("context_id", Value::Uuid(ctx_b))]),
        )
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(row_value(&result, 0, "source_id"), &Value::Uuid(entity_b));
}

// entity_snapshots rows whose valid_to equals the next row's valid_from are contiguous and permitted (Invariant 11); the engine must not mistake them for overlaps.
#[test]
fn test_entity_snapshots_adjacent_rows_insert() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(899);
    let entity_id = Uuid::from_u128(900);
    let snapshot_1 = Uuid::from_u128(901);
    let snapshot_2 = Uuid::from_u128(902);
    let snapshot_3 = Uuid::from_u128(903);
    insert_context(&db, context_id, "ctx");
    insert_entity(&db, entity_id, context_id, "entity");
    insert_snapshot(&db, snapshot_1, context_id, entity_id, 1, 2);
    insert_snapshot(&db, snapshot_2, context_id, entity_id, 2, 3);
    let result = db.execute(
        "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
        &params(vec![
            ("id", Value::Uuid(snapshot_3)),
            ("context_id", Value::Uuid(context_id)),
            ("entity_id", Value::Uuid(entity_id)),
            ("data", Value::Json(json!({"version": 4}))),
            ("valid_from", Value::TxId(TxId(4_u64))),
            ("valid_to", Value::TxId(TxId(5_u64))),
        ]),
    );
    assert!(result.is_ok());
}

// the engine itself does not reject overlapping entity_snapshots rows; overlap prevention is the update_entity operation's contract, not a schema-level constraint, and this test pins that division.
#[test]
fn test_entity_snapshots_engine_permits_overlapping_rows() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(9030);
    let entity_id = Uuid::from_u128(904);
    insert_context(&db, context_id, "ctx");
    insert_entity(&db, entity_id, context_id, "entity");
    insert_snapshot(&db, Uuid::from_u128(905), context_id, entity_id, 1, 3);
    let result = db.execute(
        "INSERT INTO entity_snapshots (id, context_id, entity_id, data, valid_from, valid_to) VALUES ($id, $context_id, $entity_id, $data, $valid_from, $valid_to)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(906))),
            ("context_id", Value::Uuid(context_id)),
            ("entity_id", Value::Uuid(entity_id)),
            ("data", Value::Json(json!({"version": 2}))),
            ("valid_from", Value::TxId(TxId(2_u64))),
            ("valid_to", Value::TxId(TxId(4_u64))),
        ]),
    );
    assert!(result.is_ok());
}

// outcomes.decision_id is a declared FK into decisions.id; an insert naming a non-existent decision must fail with ForeignKeyViolation so no HAS_OUTCOME edge can dangle.
#[test]
fn test_fk_outcome_invalid_decision_id_rejected() {
    let db = fresh_db();
    assert_error_is_fk(db.execute(
        "INSERT INTO outcomes (id, context_id, decision_id, success, notes, recorded_at) VALUES ($id, $context_id, $decision_id, $success, $notes, $recorded_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1000))),
            ("context_id", Value::Uuid(Uuid::from_u128(999))),
            ("decision_id", Value::Uuid(Uuid::from_u128(1001))),
            ("success", Value::Bool(true)),
            ("notes", Value::Text("n".into())),
            ("recorded_at", Value::TxId(TxId(1_u64))),
        ]),
    ));
}

// observations.entity_id is a declared FK into entities.id; observations pointing at a non-existent entity must be refused at the engine, not silently accepted.
#[test]
fn test_fk_observation_invalid_entity_id_rejected() {
    let db = fresh_db();
    let ctx = Uuid::from_u128(1002);
    insert_context(&db, ctx, "ctx");
    assert_error_is_fk(db.execute(
        "INSERT INTO observations (id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at) VALUES ($id, $observation_type, $observed_properties, $state_delta, $properties, $source, $entity_id, $context_id, $timestamp, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1003))),
            ("observation_type", Value::Text("config_change".into())),
            ("observed_properties", Value::Json(json!({"field": "x"}))),
            ("state_delta", Value::Json(json!({}))),
            ("properties", Value::Json(json!({}))),
            ("source", Value::Text("source".into())),
            ("entity_id", Value::Uuid(Uuid::from_u128(1004))),
            ("context_id", Value::Uuid(ctx)),
            ("timestamp", Value::Timestamp(1_700_000_061)),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    ));
}

// invalidations.affected_decision_id FK must resolve to a real decisions row; a phantom target is rejected so INVALIDATES edges cannot dangle.
#[test]
fn test_fk_invalidation_invalid_decision_id_rejected() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(1004);
    let entity_id = Uuid::from_u128(1005);
    let observation_id = Uuid::from_u128(1007);
    insert_context(&db, context_id, "ctx");
    insert_entity(&db, entity_id, context_id, "entity");
    insert_observation(&db, observation_id, entity_id, context_id);
    assert_error_is_fk(db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1008))),
            ("status", Value::Text("pending".into())),
            ("context_id", Value::Uuid(context_id)),
            ("affected_decision_id", Value::Uuid(Uuid::from_u128(1006))),
            ("trigger_observation_id", Value::Uuid(observation_id)),
            ("basis_diff", Value::Json(json!({"field": "x"}))),
            ("severity", Value::Text("warning".into())),
            ("detected_at", Value::TxId(TxId(1_u64))),
            ("resolved_at", Value::Null),
            ("resolution_action", Value::Null),
            ("resolution_decision_id", Value::Null),
        ]),
    ));
}

// approvals.decision_id FK must resolve; an approval pointing at no Decision would leave the governance record untethered.
#[test]
fn test_fk_approval_invalid_decision_id_rejected() {
    let db = fresh_db();
    assert_error_is_fk(db.execute(
        "INSERT INTO approvals (id, context_id, decision_id, approver_id, level, rationale, created_at) VALUES ($id, $context_id, $decision_id, $approver_id, $level, $rationale, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1008))),
            ("context_id", Value::Uuid(Uuid::from_u128(1007))),
            ("decision_id", Value::Uuid(Uuid::from_u128(1009))),
            ("approver_id", Value::Text("approver".into())),
            ("level", Value::Text("team".into())),
            ("rationale", Value::Text("no".into())),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    ));
}

// intentions.context_id FK into contexts.id must hold; an intention with a non-existent context would break the isolation boundary (Invariant 7).
#[test]
fn test_fk_intention_invalid_context_id_rejected() {
    let db = fresh_db();
    assert_error_is_fk(db.execute(
        "INSERT INTO intentions (id, description, status, context_id, origin, origin_ref, blueprint_id, bindings, blueprint_proposal, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $origin_ref, $blueprint_id, $bindings, $blueprint_proposal, $properties, $tags, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1010))),
            ("description", Value::Text("test".into())),
            ("status", Value::Text("active".into())),
            ("context_id", Value::Uuid(Uuid::from_u128(1011))),
            ("origin", Value::Text("human".into())),
            ("origin_ref", Value::Null),
            ("blueprint_id", Value::Null),
            ("bindings", Value::Json(json!({}))),
            ("blueprint_proposal", Value::Null),
            ("properties", Value::Json(json!({}))),
            ("tags", Value::Json(json!([]))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    ));
}

// decisions.context_id FK into contexts.id must hold; a decision outside any context would escape the per-tenant isolation surface.
#[test]
fn test_fk_decision_invalid_context_id_rejected() {
    let db = fresh_db();
    assert_error_is_fk(db.execute(
        "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1012))),
            ("decision_type", Value::Text("architecture".into())),
            ("description", Value::Text("test".into())),
            ("reasoning", Value::Json(json!([]))),
            ("confidence", Value::Float64(0.5)),
            ("status", Value::Text("active".into())),
            ("agent_id", Value::Text("agent".into())),
            ("context_id", Value::Uuid(Uuid::from_u128(1013))),
            ("properties", Value::Json(json!({}))),
            ("tags", Value::Json(json!([]))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    ));
}

// entities.context_id FK into contexts.id must hold; an entity with a dangling context would be invisible to every scoped list_entities call.
#[test]
fn test_fk_entity_invalid_context_id_rejected() {
    let db = fresh_db();
    assert_error_is_fk(db.execute(
        "INSERT INTO entities (id, entity_type, name, status, context_id, properties, tags, created_at) VALUES ($id, $entity_type, $name, $status, $context_id, $properties, $tags, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1014))),
            ("entity_type", Value::Text("SERVICE".into())),
            ("name", Value::Text("entity".into())),
            ("status", Value::Text("active".into())),
            ("context_id", Value::Uuid(Uuid::from_u128(1015))),
            ("properties", Value::Json(json!({}))),
            ("tags", Value::Json(json!([]))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    ));
}

// digests.context_id FK into contexts.id must hold; a digest attached to no context would make its provenance edges leak across tenancy.
#[test]
fn test_fk_digest_invalid_context_id_rejected() {
    let db = fresh_db();
    assert_error_is_fk(db.execute(
        "INSERT INTO digests (id, source, session_ref, summary, kind, vector_text, dimension, context_id, properties, created_at) VALUES ($id, $source, $session_ref, $summary, $kind, $vector_text, $dimension, $context_id, $properties, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1016))),
            ("source", Value::Text("source".into())),
            ("session_ref", Value::Null),
            ("summary", Value::Text("summary".into())),
            ("kind", Value::Text("Text".into())),
            (
                "vector_text",
                Value::Vector(vec![0.0; DIGEST_VECTOR_DIMENSION]),
            ),
            ("dimension", Value::Int64(DIGEST_VECTOR_DIMENSION as i64)),
            ("context_id", Value::Uuid(Uuid::from_u128(1017))),
            ("properties", Value::Json(json!({}))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    ));
}

// invalidations.trigger_observation_id FK into observations.id must hold; without it, TRIGGERED_BY provenance would point at nothing.
#[test]
fn test_fk_invalidation_invalid_observation_id_rejected() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(1018);
    let decision_id = Uuid::from_u128(1019);
    insert_context(&db, context_id, "ctx");
    insert_decision(&db, decision_id, context_id, "active");
    assert_error_is_fk(db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1021))),
            ("status", Value::Text("pending".into())),
            ("context_id", Value::Uuid(context_id)),
            ("affected_decision_id", Value::Uuid(decision_id)),
            ("trigger_observation_id", Value::Uuid(Uuid::from_u128(1020))),
            ("basis_diff", Value::Json(json!({"field": "x"}))),
            ("severity", Value::Text("warning".into())),
            ("detected_at", Value::TxId(TxId(1_u64))),
            ("resolved_at", Value::Null),
            ("resolution_action", Value::Null),
            ("resolution_decision_id", Value::Null),
        ]),
    ));
}

// intentions.description is NOT NULL at the schema layer; a null description would make the Intention semantically empty and is refused at insert.
#[test]
fn test_not_null_intention_description_rejected() {
    let db = fresh_db();
    let ctx = Uuid::from_u128(1030);
    insert_context(&db, ctx, "ctx");
    let result = db.execute(
        "INSERT INTO intentions (id, description, status, context_id, origin, origin_ref, blueprint_id, bindings, blueprint_proposal, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $origin_ref, $blueprint_id, $bindings, $blueprint_proposal, $properties, $tags, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1033))),
            ("description", Value::Null),
            ("status", Value::Text("active".into())),
            ("context_id", Value::Uuid(ctx)),
            ("origin", Value::Text("human".into())),
            ("origin_ref", Value::Null),
            ("blueprint_id", Value::Null),
            ("bindings", Value::Json(json!({}))),
            ("blueprint_proposal", Value::Null),
            ("properties", Value::Json(json!({}))),
            ("tags", Value::Json(json!([]))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    );
    assert!(
        result.is_err(),
        "null intentions.description accepted: {result:?}"
    );
}

// decisions.decision_type is NOT NULL; Discovery and audit queries key off this column, so a null value must be rejected at insert.
#[test]
fn test_not_null_decision_decision_type_rejected() {
    let db = fresh_db();
    let ctx = Uuid::from_u128(1130);
    insert_context(&db, ctx, "ctx");
    let result = db.execute(
        "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1034))),
            ("decision_type", Value::Null),
            ("description", Value::Text("test".into())),
            ("reasoning", Value::Json(json!([]))),
            ("confidence", Value::Float64(0.2)),
            ("status", Value::Text("active".into())),
            ("agent_id", Value::Null),
            ("context_id", Value::Uuid(ctx)),
            ("properties", Value::Json(json!({}))),
            ("tags", Value::Json(json!([]))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    );
    assert!(
        result.is_err(),
        "null decisions.decision_type accepted: {result:?}"
    );
}

// entities.name is NOT NULL so list_entities name_contains filters always have a string to match on; null names are refused at insert.
#[test]
fn test_not_null_entity_name_rejected() {
    let db = fresh_db();
    let ctx = Uuid::from_u128(1230);
    insert_context(&db, ctx, "ctx");
    let result = db.execute(
        "INSERT INTO entities (id, entity_type, name, context_id, properties, created_at) VALUES ($id, $entity_type, $name, $context_id, $properties, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1035))),
            ("entity_type", Value::Text("SERVICE".into())),
            ("name", Value::Null),
            ("context_id", Value::Uuid(ctx)),
            ("properties", Value::Json(json!({}))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    );
    assert!(result.is_err(), "null entities.name accepted: {result:?}");
}

// observations.source is NOT NULL so every captured fact carries a traceable origin; null source is refused at insert.
#[test]
fn test_not_null_observation_source_rejected() {
    let db = fresh_db();
    let ctx = Uuid::from_u128(1330);
    insert_context(&db, ctx, "ctx");
    let entity_id = Uuid::from_u128(1331);
    insert_entity(&db, entity_id, ctx, "entity");
    let result = db.execute(
        "INSERT INTO observations (id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at) VALUES ($id, $observation_type, $observed_properties, $state_delta, $properties, $source, $entity_id, $context_id, $timestamp, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1036))),
            ("observation_type", Value::Text("config_change".into())),
            ("observed_properties", Value::Json(json!({}))),
            ("state_delta", Value::Json(json!({}))),
            ("properties", Value::Json(json!({}))),
            ("source", Value::Null),
            ("entity_id", Value::Uuid(entity_id)),
            ("context_id", Value::Uuid(ctx)),
            ("timestamp", Value::Timestamp(1_700_000_073)),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    );
    assert!(
        result.is_err(),
        "null observations.source accepted: {result:?}"
    );
}

// contexts.name is NOT NULL because the unique-name contract underpins tenant isolation; null name is refused at insert.
#[test]
fn test_not_null_context_name_rejected() {
    let db = fresh_db();
    let result = db.execute(
        "INSERT INTO contexts (id, name, labels, properties, created_at) VALUES ($id, $name, $labels, $properties, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1037))),
            ("name", Value::Null),
            ("labels", Value::Json(json!({}))),
            ("properties", Value::Json(json!({}))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    );
    assert!(result.is_err(), "null contexts.name accepted: {result:?}");
}

// contexts.name carries a UNIQUE constraint; a duplicate insert succeeds as a no-op at the engine, so the application layer (Store) detects the collision and surfaces CgError::UniqueViolation.
#[test]
fn test_unique_context_name_noop() {
    let db = fresh_db();
    let id1 = Uuid::from_u128(1040);
    let id2 = Uuid::from_u128(1041);
    insert_context(&db, id1, "project-alpha");
    let result = db.execute(
        "INSERT INTO contexts (id, name, labels, properties, created_at) VALUES ($id, $name, $labels, $properties, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id2)),
            ("name", Value::Text("project-alpha".into())),
            ("labels", Value::Json(json!({"org": "acme"}))),
            ("properties", Value::Json(json!({"tier": "gold"}))),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    );
    assert!(
        result.is_ok(),
        "duplicate context name insert should no-op: {result:?}"
    );
    let rows = db
        .execute(
            "SELECT * FROM contexts WHERE name = $name",
            &params(vec![("name", Value::Text("project-alpha".into()))]),
        )
        .unwrap();
    assert_eq!(rows.rows.len(), 1);
}

// digests.vector_text is a fixed-dimension vector column (DIGEST_VECTOR_DIMENSION); the engine must round-trip a full-dimension embedding without truncation or dimension drift.
#[test]
fn test_digest_vector_column_accepts_embedding() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(1050);
    insert_context(&db, context_id, "ctx");
    let id = Uuid::from_u128(1051);
    let embedding = vec![0.0_f32; DIGEST_VECTOR_DIMENSION];
    insert_digest(&db, id, context_id, embedding.clone());

    let result = db
        .execute(
            "SELECT * FROM digests WHERE id = $id",
            &params(vec![("id", Value::Uuid(id))]),
        )
        .unwrap();
    assert_one_row(&result);
    match row_value(&result, 0, "vector_text") {
        Value::Vector(v) => assert_eq!(v.len(), DIGEST_VECTOR_DIMENSION),
        other => panic!("expected vector embedding, got {other:?}"),
    }
}

// the cosine-distance operator on digests.vector_text must honor context_id scoping during ANN ordering so semantic search never surfaces a closer digest from a sibling context.
#[test]
fn test_digest_vector_cosine_distance_query() {
    let db = fresh_db();
    let ctx_a = Uuid::from_u128(1060);
    let ctx_b = Uuid::from_u128(1061);
    insert_context(&db, ctx_a, "ctx-a");
    insert_context(&db, ctx_b, "ctx-b");

    let close_other_context = Uuid::from_u128(1062);
    let farther_right_context = Uuid::from_u128(1063);
    let mut close_vec = vec![0.0_f32; DIGEST_VECTOR_DIMENSION];
    close_vec[0] = 0.99;
    close_vec[1] = 0.01;
    let mut far_vec = vec![0.0_f32; DIGEST_VECTOR_DIMENSION];
    far_vec[0] = 0.5;
    far_vec[1] = 0.5;
    insert_digest(&db, close_other_context, ctx_b, close_vec);
    insert_digest(&db, farther_right_context, ctx_a, far_vec);

    let mut query = vec![0.0_f32; DIGEST_VECTOR_DIMENSION];
    query[0] = 1.0;
    let result = db
        .execute(
            "SELECT id, context_id FROM digests WHERE context_id = $context_id ORDER BY vector_text <=> $query LIMIT 2",
            &params(vec![
                ("context_id", Value::Uuid(ctx_a)),
                ("query", Value::Vector(query)),
            ]),
        )
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(
        row_value(&result, 0, "id"),
        &Value::Uuid(farther_right_context)
    );
    assert_eq!(row_value(&result, 0, "context_id"), &Value::Uuid(ctx_a));
}

// schema initialization is idempotent: calling it twice must neither duplicate tables nor error, so Store::open on an already-initialized database stays safe.
#[test]
fn test_initialize_schema_idempotent() {
    let db = Database::open_memory();
    initialize_schema(&db);
    initialize_schema(&db);

    insert_context(&db, Uuid::from_u128(1070), "project-alpha");
    let result = db
        .execute(
            "SELECT * FROM contexts WHERE name = $name",
            &params(vec![("name", Value::Text("project-alpha".into()))]),
        )
        .unwrap();
    assert_one_row(&result);
}

fn assert_table_present(db: &Database, table: &str) {
    assert!(
        db.table_names().iter().any(|name| name == table),
        "{table} table must be present in the evidence schema"
    );
}

#[test]
fn combination_record_observation_observation_evidence_snapshot_embedding_atomic_one_tx() {
    let _dir = tempfile::tempdir().expect("tempdir");
    let config = context_graph::StoreConfig {
        db_path: _dir.path().join("cg.db"),
        ..Default::default()
    };
    let store = context_graph::Store::open(config.clone()).expect("open");
    let (context_id, entity_id) = seed_store_context_entity(&store);
    let observation_id = context_graph::ObservationId::new_v7();
    let evidence_id = context_graph::EvidenceId::new_v7();
    let embedding_id = context_graph::EmbeddingId::new_v7();
    let vector = text_vector_with_axis(0);
    let recorded = store
        .record_observation(cg_observation_with_text_embedding(
            observation_id,
            evidence_id,
            embedding_id,
            entity_id,
            context_id,
            vector.clone(),
        ))
        .expect("record observation");
    assert_eq!(recorded.observation_id, observation_id);
    assert_eq!(
        recorded.evidence_ids,
        vec![evidence_id],
        "record_observation must surface the evidence ids it committed"
    );
    assert_eq!(
        recorded.embedding_ids,
        vec![embedding_id],
        "record_observation must surface the embedding ids it committed"
    );
    let snapshot_id = recorded
        .snapshot_id
        .expect("non-empty state_delta must commit an entity snapshot");
    drop(store);

    let db = Database::open(&config.db_path).expect("raw open");
    let observation = db
        .execute(
            "SELECT id, created_at FROM observations WHERE id = $id",
            &params(vec![("id", Value::Uuid(observation_id.as_uuid()))]),
        )
        .expect("observation row");
    assert_one_row(&observation);
    let created_at = row_value(&observation, 0, "created_at").clone();

    let evidence = db
        .execute(
            "SELECT id, observation_id FROM evidence WHERE id = $id",
            &params(vec![("id", Value::Uuid(evidence_id.as_uuid()))]),
        )
        .expect("evidence row");
    assert_one_row(&evidence);
    assert_eq!(
        row_value(&evidence, 0, "observation_id"),
        &Value::Uuid(observation_id.as_uuid())
    );

    let snapshot = db
        .execute(
            "SELECT id, entity_id, valid_from FROM entity_snapshots WHERE id = $id",
            &params(vec![("id", Value::Uuid(snapshot_id.as_uuid()))]),
        )
        .expect("snapshot row");
    assert_one_row(&snapshot);
    assert_eq!(
        row_value(&snapshot, 0, "entity_id"),
        &Value::Uuid(entity_id.as_uuid())
    );
    assert_eq!(row_value(&snapshot, 0, "valid_from"), &created_at);
    let snapshot_observation = db
        .execute(
            "SELECT context_id, entity_id, snapshot_id, observation_id, observation_order, created_at FROM entity_snapshot_observations WHERE snapshot_id = $snapshot_id",
            &params(vec![("snapshot_id", Value::Uuid(snapshot_id.as_uuid()))]),
        )
        .expect("snapshot observation row");
    assert_one_row(&snapshot_observation);
    assert_eq!(
        row_value(&snapshot_observation, 0, "context_id"),
        &Value::Uuid(context_id.as_uuid())
    );
    assert_eq!(
        row_value(&snapshot_observation, 0, "entity_id"),
        &Value::Uuid(entity_id.as_uuid())
    );
    assert_eq!(
        row_value(&snapshot_observation, 0, "observation_id"),
        &Value::Uuid(observation_id.as_uuid())
    );
    assert_eq!(
        row_value(&snapshot_observation, 0, "observation_order"),
        &Value::Int64(0)
    );
    assert_eq!(
        row_value(&snapshot_observation, 0, "created_at"),
        &created_at
    );

    let embedding = db
        .execute(
            "SELECT id, embedding_space_id, target_kind, target_id, dimension, created_at FROM embedding_handles WHERE id = $id",
            &params(vec![("id", Value::Uuid(embedding_id.as_uuid()))]),
        )
        .expect("embedding handle row");
    assert_one_row(&embedding);
    assert_eq!(
        row_value(&embedding, 0, "embedding_space_id"),
        &Value::Text(TEXT_SPACE.to_string())
    );
    assert_eq!(
        row_value(&embedding, 0, "target_kind"),
        &Value::Text("evidence".to_string())
    );
    assert_eq!(
        row_value(&embedding, 0, "target_id"),
        &Value::Uuid(evidence_id.as_uuid())
    );
    assert_eq!(row_value(&embedding, 0, "dimension"), &Value::Int64(768));
    assert_eq!(row_value(&embedding, 0, "created_at"), &created_at);

    let vector_hits = db
        .execute(
            "SELECT id FROM evidence ORDER BY vector_text <=> $query LIMIT 1",
            &params(vec![("query", Value::Vector(vector))]),
        )
        .expect("vector search");
    assert_one_row(&vector_hits);
    assert_eq!(
        row_value(&vector_hits, 0, "id"),
        &Value::Uuid(evidence_id.as_uuid())
    );
}

#[test]
fn combination_record_observation_rollback_on_validation_error_after_tx_open_leaves_zero_rows() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = context_graph::StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let store = context_graph::Store::open(config.clone()).expect("open");
    let (context_id, entity_id) = seed_store_context_entity(&store);
    let observation_id = context_graph::ObservationId::new_v7();
    let valid_evidence_id = context_graph::EvidenceId::new_v7();
    let invalid_evidence_id = context_graph::EvidenceId::new_v7();
    let embedding_id = context_graph::EmbeddingId::new_v7();
    let vector = text_vector_with_axis(5);
    let mut input = cg_observation_with_text_embedding(
        observation_id,
        valid_evidence_id,
        embedding_id,
        entity_id,
        context_id,
        vector.clone(),
    );
    input.evidence.push(context_graph::EvidenceRef {
        id: invalid_evidence_id,
        observation_id,
        context_id,
        kind: context_graph::EvidenceKind::ImageFrame,
        source_ref: "vigil-edge://camera-1/frame/invalid".to_string(),
        mime_type: Some("image/jpeg".to_string()),
        region: Some(json!({"x": 1.4, "y": 0.0, "w": 0.2, "h": 0.2})),
        retention_status: context_graph::RetentionStatus::RetainedExternal,
        ..Default::default()
    });

    let pre_snapshot_rows = context_graph::Store::open(config.clone())
        .expect("pre reader")
        .entity_history(entity_id)
        .expect("pre history")
        .len() as i64;
    let result = store.record_observation(input);
    drop(store);
    let db = Database::open(&config.db_path).expect("raw open");
    let observation_rows = scalar_count(
        &db,
        "SELECT COUNT(*) AS n FROM observations WHERE id = $id",
        &params(vec![("id", Value::Uuid(observation_id.as_uuid()))]),
    );
    let evidence_rows = scalar_count(
        &db,
        "SELECT COUNT(*) AS n FROM evidence WHERE id = $valid OR id = $invalid",
        &params(vec![
            ("valid", Value::Uuid(valid_evidence_id.as_uuid())),
            ("invalid", Value::Uuid(invalid_evidence_id.as_uuid())),
        ]),
    );
    let snapshot_rows = scalar_count(
        &db,
        "SELECT COUNT(*) AS n FROM entity_snapshots WHERE entity_id = $id",
        &params(vec![("id", Value::Uuid(entity_id.as_uuid()))]),
    );
    let embedding_rows = scalar_count(
        &db,
        "SELECT COUNT(*) AS n FROM embedding_handles WHERE id = $id",
        &params(vec![("id", Value::Uuid(embedding_id.as_uuid()))]),
    );
    let vector_hits = db
        .query_vector(
            contextdb_core::VectorIndexRef::new("evidence", "vector_text"),
            &vector,
            10,
            None,
            db.snapshot(),
        )
        .expect("vector query");

    assert!(matches!(
        result,
        Err(context_graph::CgError::InvalidEvidenceRegionBounds { .. })
    ));
    assert_eq!(observation_rows, 0);
    assert_eq!(evidence_rows, 0);
    assert_eq!(snapshot_rows, pre_snapshot_rows);
    assert_eq!(embedding_rows, 0);
    assert!(
        vector_hits.is_empty(),
        "failed record_observation must not leave vector_text entries behind"
    );
}

#[test]
fn combination_evidence_table_immutable_rejects_update_and_delete() {
    let db = fresh_db();
    assert_table_present(&db, "evidence");
    let update = db.execute(
        "UPDATE evidence SET source_ref = $source_ref WHERE id = $id",
        &params(vec![
            ("source_ref", Value::Text("s3://mutated".into())),
            ("id", Value::Uuid(Uuid::from_u128(20_003))),
        ]),
    );
    let delete = db.execute(
        "DELETE FROM evidence WHERE id = $id",
        &params(vec![("id", Value::Uuid(Uuid::from_u128(20_003)))]),
    );
    assert!(matches!(
        update,
        Err(Error::ImmutableTable { .. } | Error::ImmutableColumn { .. })
    ));
    assert!(matches!(
        delete,
        Err(Error::ImmutableTable { .. } | Error::ImmutableColumn { .. })
    ));
}

#[test]
fn combination_evidence_availability_log_immutable_rejects_update_and_delete() {
    let db = fresh_db();
    assert_table_present(&db, "evidence_availability_log");
    let update = db.execute(
        "UPDATE evidence_availability_log SET status = $status WHERE evidence_id = $id",
        &params(vec![
            ("status", Value::Text("expired".into())),
            ("id", Value::Uuid(Uuid::from_u128(20_004))),
        ]),
    );
    let delete = db.execute(
        "DELETE FROM evidence_availability_log WHERE evidence_id = $id",
        &params(vec![("id", Value::Uuid(Uuid::from_u128(20_004)))]),
    );
    assert!(matches!(
        update,
        Err(Error::ImmutableTable { .. } | Error::ImmutableColumn { .. })
    ));
    assert!(matches!(
        delete,
        Err(Error::ImmutableTable { .. } | Error::ImmutableColumn { .. })
    ));
}

#[test]
fn combination_evidence_availability_log_concurrent_appends_persist_distinct_rows_last_writer_wins_by_txid()
 {
    let db = Arc::new(fresh_db());
    let context_id = Uuid::from_u128(20_105);
    let entity_id = Uuid::from_u128(20_205);
    let observation_id = Uuid::from_u128(20_305);
    let evidence_id = Uuid::from_u128(20_005);
    insert_context(&db, context_id, "availability-log");
    insert_entity(&db, entity_id, context_id, "availability-log-entity");
    insert_observation(&db, observation_id, entity_id, context_id);
    insert_evidence(&db, evidence_id, observation_id, context_id, 0, None);
    let barrier = Arc::new(Barrier::new(2));
    let handles: Vec<_> = ["expired", "unavailable"]
        .into_iter()
        .map(|status| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let tx = db.begin().expect("begin");
                barrier.wait();
                let result = db.execute_in_tx(
                    tx,
                    "INSERT INTO evidence_availability_log (id, context_id, evidence_id, status, recorded_at_txid) VALUES ($id, $context_id, $evidence_id, $status, $recorded_at_txid)",
                    &params(vec![
                        ("id", Value::Uuid(Uuid::now_v7())),
                        ("context_id", Value::Uuid(context_id)),
                        ("evidence_id", Value::Uuid(evidence_id)),
                        ("status", Value::Text(status.into())),
                        ("recorded_at_txid", Value::TxId(tx)),
                    ]),
                );
                match result {
                    Ok(_) => db.commit(tx).map(|_| tx),
                    Err(err) => {
                        let _ = db.rollback(tx);
                        Err(err)
                    }
                }
            })
        })
        .collect();
    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer joined"))
        .collect();
    assert!(
        results.iter().all(|result| result.is_ok()),
        "both append-only log writes must persist"
    );
    let txids: BTreeSet<_> = results
        .iter()
        .filter_map(|result| result.as_ref().ok().copied())
        .collect();
    assert_eq!(txids.len(), 2, "log TXIDs must be distinct");
    let rows = db
        .execute(
            "SELECT status, recorded_at_txid FROM evidence_availability_log WHERE evidence_id = $id",
            &params(vec![("id", Value::Uuid(evidence_id))]),
        )
        .expect("select log");
    assert_eq!(rows.rows.len(), 2);
    let latest_txid = txids.iter().next_back().copied().expect("latest txid");
    assert!(
        rows.rows
            .iter()
            .any(|row| row.contains(&Value::TxId(latest_txid)))
    );
}

#[test]
fn combination_decision_status_state_machine_active_to_invalidated_accepted() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(20_006);
    insert_context(&db, context_id, "ctx");
    let decision_id = Uuid::from_u128(20_007);
    insert_decision(&db, decision_id, context_id, "active");
    let result = db
        .execute(
            "UPDATE decisions SET status = $status WHERE id = $id AND status = $from",
            &params(vec![
                ("status", Value::Text("invalidated".into())),
                ("from", Value::Text("active".into())),
                ("id", Value::Uuid(decision_id)),
            ]),
        )
        .expect("active to invalidated");
    assert_eq!(result.rows_affected, 1);
}

#[test]
fn combination_decision_status_state_machine_invalidated_to_active_rejected() {
    assert_invalidated_terminal_transition_rejected("active");
}

#[test]
fn combination_decision_status_state_machine_invalidated_to_archived_rejected() {
    assert_invalidated_terminal_transition_rejected("archived");
}

#[test]
fn combination_decision_status_state_machine_invalidated_to_superseded_rejected() {
    assert_invalidated_terminal_transition_rejected("superseded");
}

fn assert_invalidated_terminal_transition_rejected(to: &str) {
    let db = fresh_db();
    let context_id = Uuid::from_u128(20_008);
    insert_context(&db, context_id, "ctx");
    let decision_id = Uuid::from_u128(20_009);
    insert_decision(&db, decision_id, context_id, "invalidated");
    let result = db.execute(
        "UPDATE decisions SET status = $status WHERE id = $id",
        &params(vec![
            ("status", Value::Text(to.into())),
            ("id", Value::Uuid(decision_id)),
        ]),
    );
    assert!(
        matches!(result, Err(Error::InvalidStateTransition(_))),
        "invalidated -> {to} must be a typed state-machine error"
    );
}

#[test]
fn combination_invalidate_decision_with_invalidation_row_atomic_or_neither_visible() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = context_graph::StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let (decision_id, trigger_id) = seed_store_decision(&config);
    let writer = context_graph::Store::open(config).expect("writer open");
    let invalidation = writer
        .invalidate_decision(decision_id, trigger_id, "basis drift")
        .expect("invalidate");
    assert_eq!(
        invalidation.previous_status,
        context_graph::DecisionStatus::Active
    );
    assert_eq!(
        invalidation.new_status,
        context_graph::DecisionStatus::Invalidated
    );
    assert_eq!(
        writer
            .get_decision(decision_id)
            .expect("decision")
            .expect("decision row")
            .status,
        context_graph::DecisionStatus::Invalidated
    );
    assert_eq!(
        writer
            .list_invalidations(
                store_decision_context(&writer, decision_id),
                Some(decision_id),
                None,
            )
            .expect("invalidations")
            .len(),
        1
    );
}

#[test]
fn combination_invalidate_decision_concurrent_atomic_one_winner_count_invalidations_equals_one() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = context_graph::StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let (decision_id, trigger_id) = seed_store_decision(&config);
    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let config = config.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let store = context_graph::Store::open(config).expect("writer open");
                barrier.wait();
                store.invalidate_decision(decision_id, trigger_id, "basis drift")
            })
        })
        .collect();
    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer joined"))
        .collect();
    let ok_count = results.iter().filter(|result| result.is_ok()).count();
    let illegal_count = results
        .iter()
        .filter(|result| {
            matches!(
                result,
                Err(context_graph::CgError::IllegalTransition { .. })
            )
        })
        .count();
    let reader = context_graph::Store::open(config).expect("reader");
    let invalidation_rows = reader
        .list_invalidations(
            store_decision_context(&reader, decision_id),
            Some(decision_id),
            None,
        )
        .expect("invalidations");
    assert_eq!(
        ok_count, 1,
        "conditional invalidation must have exactly one winner"
    );
    assert_eq!(illegal_count, 3);
    assert_eq!(invalidation_rows.len(), 1);
}

#[test]
fn combination_decision_basis_snapshots_reject_wrong_context_and_wrong_entity_by_composite_fk() {
    let db = fresh_db();
    let ctx_a = Uuid::from_u128(20_011);
    let ctx_b = Uuid::from_u128(20_012);
    let decision_a = Uuid::from_u128(20_013);
    let entity_a = Uuid::from_u128(20_014);
    let entity_b = Uuid::from_u128(20_015);
    let snapshot_a = Uuid::from_u128(20_016);
    let snapshot_b = Uuid::from_u128(20_017);
    insert_context(&db, ctx_a, "basis-ctx-a");
    insert_context(&db, ctx_b, "basis-ctx-b");
    insert_decision(&db, decision_a, ctx_a, "active");
    insert_entity(&db, entity_a, ctx_a, "entity-a");
    insert_entity(&db, entity_b, ctx_b, "entity-b");
    insert_snapshot(&db, snapshot_a, ctx_a, entity_a, 1, 2);
    insert_snapshot(&db, snapshot_b, ctx_b, entity_b, 1, 2);

    let wrong_context = db.execute(
        "INSERT INTO decision_basis_snapshots (id, context_id, decision_id, entity_id, snapshot_id, basis_order, created_at) VALUES ($id, $context_id, $decision_id, $entity_id, $snapshot_id, $basis_order, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(20_018))),
            ("context_id", Value::Uuid(ctx_a)),
            ("decision_id", Value::Uuid(decision_a)),
            ("entity_id", Value::Uuid(entity_b)),
            ("snapshot_id", Value::Uuid(snapshot_b)),
            ("basis_order", Value::Int64(0)),
            ("created_at", Value::TxId(TxId(3))),
        ]),
    );
    assert_error_is_fk(wrong_context);

    let wrong_entity = db.execute(
        "INSERT INTO decision_basis_snapshots (id, context_id, decision_id, entity_id, snapshot_id, basis_order, created_at) VALUES ($id, $context_id, $decision_id, $entity_id, $snapshot_id, $basis_order, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(20_019))),
            ("context_id", Value::Uuid(ctx_a)),
            ("decision_id", Value::Uuid(decision_a)),
            ("entity_id", Value::Uuid(entity_a)),
            ("snapshot_id", Value::Uuid(snapshot_b)),
            ("basis_order", Value::Int64(0)),
            ("created_at", Value::TxId(TxId(3))),
        ]),
    );
    assert_error_is_fk(wrong_entity);
}

#[test]
fn combination_entity_snapshot_observations_reject_wrong_context_and_wrong_entity_by_composite_fk()
{
    let db = fresh_db();
    let ctx_a = Uuid::from_u128(20_020);
    let ctx_b = Uuid::from_u128(20_021);
    let entity_a = Uuid::from_u128(20_022);
    let entity_b = Uuid::from_u128(20_023);
    let entity_c = Uuid::from_u128(20_030);
    let snapshot_a = Uuid::from_u128(20_024);
    let snapshot_b = Uuid::from_u128(20_025);
    let observation_a = Uuid::from_u128(20_026);
    let observation_b = Uuid::from_u128(20_027);
    let observation_c = Uuid::from_u128(20_031);
    insert_context(&db, ctx_a, "snapshot-observation-ctx-a");
    insert_context(&db, ctx_b, "snapshot-observation-ctx-b");
    insert_entity(&db, entity_a, ctx_a, "entity-a");
    insert_entity(&db, entity_b, ctx_b, "entity-b");
    insert_entity(&db, entity_c, ctx_a, "entity-c");
    insert_snapshot(&db, snapshot_a, ctx_a, entity_a, 1, 2);
    insert_snapshot(&db, snapshot_b, ctx_b, entity_b, 1, 2);
    insert_observation(&db, observation_a, entity_a, ctx_a);
    insert_observation(&db, observation_b, entity_b, ctx_b);
    insert_observation(&db, observation_c, entity_c, ctx_a);

    let wrong_context_observation = db.execute(
        "INSERT INTO entity_snapshot_observations (id, context_id, entity_id, snapshot_id, observation_id, observation_order, created_at) VALUES ($id, $context_id, $entity_id, $snapshot_id, $observation_id, $observation_order, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(20_028))),
            ("context_id", Value::Uuid(ctx_a)),
            ("entity_id", Value::Uuid(entity_a)),
            ("snapshot_id", Value::Uuid(snapshot_a)),
            ("observation_id", Value::Uuid(observation_b)),
            ("observation_order", Value::Int64(0)),
            ("created_at", Value::TxId(TxId(3))),
        ]),
    );
    assert_error_is_fk(wrong_context_observation);

    let wrong_entity_snapshot = db.execute(
        "INSERT INTO entity_snapshot_observations (id, context_id, entity_id, snapshot_id, observation_id, observation_order, created_at) VALUES ($id, $context_id, $entity_id, $snapshot_id, $observation_id, $observation_order, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(20_029))),
            ("context_id", Value::Uuid(ctx_a)),
            ("entity_id", Value::Uuid(entity_a)),
            ("snapshot_id", Value::Uuid(snapshot_b)),
            ("observation_id", Value::Uuid(observation_a)),
            ("observation_order", Value::Int64(0)),
            ("created_at", Value::TxId(TxId(3))),
        ]),
    );
    assert_error_is_fk(wrong_entity_snapshot);

    let wrong_same_context_observation_entity = db.execute(
        "INSERT INTO entity_snapshot_observations (id, context_id, entity_id, snapshot_id, observation_id, observation_order, created_at) VALUES ($id, $context_id, $entity_id, $snapshot_id, $observation_id, $observation_order, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(20_032))),
            ("context_id", Value::Uuid(ctx_a)),
            ("entity_id", Value::Uuid(entity_a)),
            ("snapshot_id", Value::Uuid(snapshot_a)),
            ("observation_id", Value::Uuid(observation_c)),
            ("observation_order", Value::Int64(0)),
            ("created_at", Value::TxId(TxId(3))),
        ]),
    );
    assert_error_is_fk(wrong_same_context_observation_entity);
}

#[test]
fn combination_embedding_spaces_table_unique_on_embedding_space_id_concurrent_seed_collapses() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = context_graph::StoreConfig {
        db_path: dir.path().join("cg.db"),
        extra_embedding_spaces: vec![vision_embedding_space_decl("vision_v1")],
        ..Default::default()
    };
    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let config = config.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                context_graph::Store::open(config)
            })
        })
        .collect();
    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer joined"))
        .collect();
    assert!(
        results.iter().all(Result::is_ok),
        "idempotent concurrent Store::open declarations must not surface unique violations: {results:?}"
    );
    drop(results);
    let db = Database::open(&config.db_path).expect("raw open");
    let count = scalar_count(
        &db,
        "SELECT COUNT(*) AS n FROM embedding_spaces WHERE embedding_space_id = $id",
        &params(vec![("id", Value::Text("vision_v1".to_string()))]),
    );
    assert_eq!(count, 1);
}

#[test]
fn combination_embedding_space_bindings_unique_composite_table_column_space() {
    let db = fresh_db();
    assert_table_present(&db, "embedding_space_bindings");
    for (space_id, model_name) in [
        (
            "text_default_embeddinggemma_300m_768",
            "embeddinggemma-300m",
        ),
        ("other_text_space", "other-text"),
    ] {
        db.execute(
        "INSERT INTO embedding_spaces (embedding_space_id, dimension, metric, model_family, model_name, model_version, supported_evidence_kinds, default_for_evidence_kind, bindings) VALUES ($embedding_space_id, $dimension, $metric, $model_family, $model_name, $model_version, $supported_evidence_kinds, $default_for_evidence_kind, $bindings)",
        &params(vec![
            (
                "embedding_space_id",
                Value::Text(space_id.into()),
            ),
            ("dimension", Value::Int64(768)),
            ("metric", Value::Text("cosine".into())),
            ("model_family", Value::Text("embeddinggemma".into())),
            ("model_name", Value::Text(model_name.into())),
            ("model_version", Value::Text("default".into())),
            ("supported_evidence_kinds", Value::Json(json!(["text"]))),
            ("default_for_evidence_kind", Value::Null),
            (
                "bindings",
                Value::Json(json!([{"table_name": "evidence", "vector_column": "vector_text"}])),
            ),
        ]),
    )
    .expect("seed embedding space");
    }
    let binding_params = params(vec![
        (
            "embedding_space_id",
            Value::Text("text_default_embeddinggemma_300m_768".into()),
        ),
        ("table_name", Value::Text("evidence".into())),
        ("vector_column", Value::Text("vector_text".into())),
    ]);
    let first = db.execute(
        "INSERT INTO embedding_space_bindings (embedding_space_id, table_name, vector_column) VALUES ($embedding_space_id, $table_name, $vector_column)",
        &binding_params,
    );
    let second = db.execute(
        "INSERT INTO embedding_space_bindings (embedding_space_id, table_name, vector_column) VALUES ($embedding_space_id, $table_name, $vector_column)",
        &binding_params,
    );
    assert!(first.is_ok());
    assert!(second.is_ok());
    let other_owner = db.execute(
        "INSERT INTO embedding_space_bindings (embedding_space_id, table_name, vector_column) VALUES ($embedding_space_id, $table_name, $vector_column)",
        &params(vec![
            ("embedding_space_id", Value::Text("other_text_space".into())),
            ("table_name", Value::Text("evidence".into())),
            ("vector_column", Value::Text("vector_text".into())),
        ]),
    );
    assert!(other_owner.is_ok());
    assert_eq!(
        scalar_count(
            &db,
            "SELECT COUNT(*) AS n FROM embedding_space_bindings WHERE table_name = $table_name AND vector_column = $vector_column",
            &params(vec![
                ("table_name", Value::Text("evidence".into())),
                ("vector_column", Value::Text("vector_text".into())),
            ]),
        ),
        1
    );
}

#[test]
fn combination_evidence_multi_vector_columns_per_row_text_and_vision_independently_indexed() {
    let db = fresh_db();
    let context_id = Uuid::from_u128(20_050);
    let entity_id = Uuid::from_u128(20_051);
    let observation_id = Uuid::from_u128(20_052);
    let target_id = Uuid::from_u128(20_053);
    let decoy_id = Uuid::from_u128(20_054);
    insert_context(&db, context_id, "ctx-multi-vector");
    insert_entity(&db, entity_id, context_id, "entity-multi-vector");
    insert_observation(&db, observation_id, entity_id, context_id);

    let target_text = text_vector_with_axis(0);
    let target_vision = vision_vector_with_axis(1);
    let decoy_text = text_vector_with_axis(1);
    let decoy_vision = vision_vector_with_axis(0);
    for (supply_order, id, vector_text, vector_vision) in [
        (0_i64, target_id, target_text.clone(), target_vision.clone()),
        (1_i64, decoy_id, decoy_text, decoy_vision),
    ] {
        db.execute(
            "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, retention_status, supply_order, vector_text, vector_vision) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $retention_status, $supply_order, $vector_text, $vector_vision)",
            &params(vec![
                ("id", Value::Uuid(id)),
                ("observation_id", Value::Uuid(observation_id)),
                ("context_id", Value::Uuid(context_id)),
                ("kind", Value::Text("image_frame".into())),
                ("source_ref", Value::Text(format!("vigil-edge://camera-1/frame/{supply_order}"))),
                ("retention_status", Value::Text("retained_external".into())),
                ("supply_order", Value::Int64(supply_order)),
                ("vector_text", Value::Vector(vector_text)),
                ("vector_vision", Value::Vector(vector_vision)),
            ]),
        )
        .expect("insert vector-backed evidence");
    }

    let text_hit = db
        .execute(
            "SELECT id FROM evidence ORDER BY vector_text <=> $query LIMIT 1",
            &params(vec![("query", Value::Vector(target_text))]),
        )
        .expect("text vector query");
    assert_one_row(&text_hit);
    assert_eq!(row_value(&text_hit, 0, "id"), &Value::Uuid(target_id));

    let vision_hit = db
        .execute(
            "SELECT id FROM evidence ORDER BY vector_vision <=> $query LIMIT 1",
            &params(vec![("query", Value::Vector(target_vision))]),
        )
        .expect("vision vector query");
    assert_one_row(&vision_hit);
    assert_eq!(row_value(&vision_hit, 0, "id"), &Value::Uuid(target_id));

    let text_wrong_dimension = db.execute(
        "SELECT id FROM evidence ORDER BY vector_text <=> $query LIMIT 1",
        &params(vec![("query", Value::Vector(vec![0.0; 512]))]),
    );
    assert!(matches!(
        text_wrong_dimension,
        Err(Error::VectorIndexDimensionMismatch {
            expected: 768,
            actual: 512,
            ..
        })
    ));
    let vision_wrong_dimension = db.execute(
        "SELECT id FROM evidence ORDER BY vector_vision <=> $query LIMIT 1",
        &params(vec![("query", Value::Vector(vec![0.0; 768]))]),
    );
    assert!(matches!(
        vision_wrong_dimension,
        Err(Error::VectorIndexDimensionMismatch {
            expected: 512,
            actual: 768,
            ..
        })
    ));
}

#[test]
fn combination_cg_schema_version_table_single_row_invariant_multiple_rows_returns_legacy_schema_detected()
 {
    let dir = tempfile::tempdir().expect("tempdir");
    context_graph::Store::admin_raw_install_legacy_schema_for_tests(
        dir.path().join("cg.db"),
        context_graph::LegacySchemaFixture::MultipleVersionRows,
    )
    .expect("install legacy fixture");
    let result = context_graph::Store::open(context_graph::StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    });
    assert!(matches!(
        result,
        Err(context_graph::CgError::LegacySchemaDetected { .. })
    ));
}

#[test]
fn combination_observation_with_evidence_atomic_insert_under_fk_violation_rolls_back_at_commit() {
    let db = fresh_db();
    assert_table_present(&db, "evidence");
    let evidence_id = Uuid::from_u128(20_016);
    let observation_id = Uuid::from_u128(20_017);
    let entity_id = Uuid::from_u128(20_018);
    let context_id = Uuid::from_u128(20_019);
    let invalid_evidence_id = Uuid::from_u128(20_020);
    let missing_parent_evidence = Uuid::from_u128(20_021);
    insert_context(&db, context_id, "ctx-evidence-fk-rollback");
    insert_entity(&db, entity_id, context_id, "rollback-entity");

    let tx = db.begin().expect("begin");
    let observation = db.execute_in_tx(
        tx,
        "INSERT INTO observations (id, entity_id, context_id, observation_type, observed_properties, state_delta, properties, source, timestamp, created_at) VALUES ($id, $entity_id, $context_id, $observation_type, $observed_properties, $state_delta, $properties, $source, $timestamp, $created_at)",
        &params(vec![
            ("id", Value::Uuid(observation_id)),
            ("entity_id", Value::Uuid(entity_id)),
            ("context_id", Value::Uuid(context_id)),
            ("observation_type", Value::Text("config_change".into())),
            ("observed_properties", Value::Json(json!({}))),
            ("state_delta", Value::Json(json!({}))),
            ("properties", Value::Json(json!({}))),
            ("source", Value::Text("git-hook".into())),
            ("timestamp", Value::Timestamp(1_700_000_004)),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    );
    assert!(observation.is_ok());
    let evidence = db.execute_in_tx(
        tx,
        "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, retention_status, supply_order) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $retention_status, $supply_order)",
        &params(vec![
            ("id", Value::Uuid(evidence_id)),
            ("observation_id", Value::Uuid(observation_id)),
            ("context_id", Value::Uuid(context_id)),
            ("kind", Value::Text("text".into())),
            (
                "source_ref",
                Value::Text("cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()),
            ),
            ("retention_status", Value::Text("retained_external".into())),
            ("supply_order", Value::Int64(0)),
        ]),
    );
    assert!(evidence.is_ok());
    let invalid_evidence = db.execute_in_tx(
        tx,
        "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, derived_from, retention_status, supply_order) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $derived_from, $retention_status, $supply_order)",
        &params(vec![
            ("id", Value::Uuid(invalid_evidence_id)),
            ("observation_id", Value::Uuid(observation_id)),
            ("context_id", Value::Uuid(context_id)),
            ("kind", Value::Text("text_projection".into())),
            (
                "source_ref",
                Value::Text("cas://sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into()),
            ),
            ("derived_from", Value::Uuid(missing_parent_evidence)),
            ("retention_status", Value::Text("retained_external".into())),
            ("supply_order", Value::Int64(1)),
        ]),
    );
    assert!(invalid_evidence.is_ok());
    assert!(matches!(
        db.commit(tx),
        Err(Error::ForeignKeyViolation { .. })
    ));
    let observation_rows = db
        .execute(
            "SELECT id FROM observations WHERE id = $id",
            &params(vec![("id", Value::Uuid(observation_id))]),
        )
        .expect("select observation");
    assert!(observation_rows.rows.is_empty());
    let evidence_rows = db
        .execute(
            "SELECT id FROM evidence WHERE id = $id OR id = $invalid_id",
            &params(vec![
                ("id", Value::Uuid(evidence_id)),
                ("invalid_id", Value::Uuid(invalid_evidence_id)),
            ]),
        )
        .expect("select evidence");
    assert!(evidence_rows.rows.is_empty());
}

#[test]
fn combination_evidence_self_fk_derived_from_validated_at_commit_against_committed_snapshot() {
    let db = Arc::new(fresh_db());
    db.execute(
        "CREATE TABLE evidence_delete_race (id UUID PRIMARY KEY, derived_from UUID REFERENCES evidence_delete_race(id))",
        &HashMap::new(),
    )
    .expect("create mutable fk fixture");
    let parent = Uuid::from_u128(20_020);
    let child = Uuid::from_u128(20_021);
    db.execute(
        "INSERT INTO evidence_delete_race (id) VALUES ($id)",
        &params(vec![("id", Value::Uuid(parent))]),
    )
    .expect("seed parent");

    let (tx2_started_send, tx2_started_recv) = mpsc::channel();
    let (delete_committed_send, delete_committed_recv) = mpsc::channel();
    let deleter_db = Arc::clone(&db);
    let deleter = std::thread::spawn(move || {
        tx2_started_recv.recv().expect("child tx started");
        let tx = deleter_db.begin().expect("begin");
        let deleted = deleter_db.execute_in_tx(
            tx,
            "DELETE FROM evidence_delete_race WHERE id = $id",
            &params(vec![("id", Value::Uuid(parent))]),
        );
        let result = match deleted {
            Ok(_) => deleter_db.commit(tx),
            Err(err) => {
                let _ = deleter_db.rollback(tx);
                Err(err)
            }
        };
        delete_committed_send
            .send(())
            .expect("notify delete committed");
        result
    });

    let child_db = Arc::clone(&db);
    let child_writer = std::thread::spawn(move || {
        let tx = child_db.begin().expect("begin");
        tx2_started_send.send(()).expect("notify tx started");
        delete_committed_recv
            .recv()
            .expect("delete commit notification");
        let inserted = child_db.execute_in_tx(
            tx,
            "INSERT INTO evidence_delete_race (id, derived_from) VALUES ($id, $derived_from)",
            &params(vec![
                ("id", Value::Uuid(child)),
                ("derived_from", Value::Uuid(parent)),
            ]),
        );
        match inserted {
            Ok(_) => child_db.commit(tx),
            Err(err) => {
                let _ = child_db.rollback(tx);
                Err(err)
            }
        }
    });

    assert!(
        deleter.join().expect("deleter joined").is_ok(),
        "parent delete must win the overlapping transaction"
    );
    assert!(matches!(
        child_writer.join().expect("child writer joined"),
        Err(Error::ForeignKeyViolation { .. })
    ));
}

#[test]
fn combination_observation_entity_cross_context_raw_insert_rejected_by_schema() {
    let db = fresh_db();
    let ctx_a = Uuid::from_u128(20_022);
    let ctx_b = Uuid::from_u128(20_023);
    let entity_a = Uuid::from_u128(20_024);
    let observation = Uuid::from_u128(20_025);
    insert_context(&db, ctx_a, "ctx-observation-parent-a");
    insert_context(&db, ctx_b, "ctx-observation-child-b");
    insert_entity(&db, entity_a, ctx_a, "entity-a");

    let result = db.execute(
        "INSERT INTO observations (id, observation_type, observed_properties, state_delta, properties, source, entity_id, context_id, timestamp, created_at) VALUES ($id, $observation_type, $observed_properties, $state_delta, $properties, $source, $entity_id, $context_id, $timestamp, $created_at)",
        &params(vec![
            ("id", Value::Uuid(observation)),
            ("observation_type", Value::Text("config_change".into())),
            ("observed_properties", Value::Json(json!({"version": "2.0"}))),
            ("state_delta", Value::Json(json!({"version": "2.0"}))),
            ("properties", Value::Json(json!({}))),
            ("source", Value::Text("git-hook".into())),
            ("entity_id", Value::Uuid(entity_a)),
            ("context_id", Value::Uuid(ctx_b)),
            ("timestamp", Value::Timestamp(1_700_000_004)),
            ("created_at", Value::TxId(TxId(1_u64))),
        ]),
    );

    assert_error_is_fk(result);
    assert_eq!(
        scalar_count(
            &db,
            "SELECT count(*) AS n FROM observations WHERE id = $id",
            &params(vec![("id", Value::Uuid(observation))]),
        ),
        0
    );
}

#[test]
fn combination_evidence_observation_cross_context_raw_insert_rejected_by_schema() {
    let db = fresh_db();
    let ctx_a = Uuid::from_u128(20_026);
    let ctx_b = Uuid::from_u128(20_027);
    let entity_a = Uuid::from_u128(20_028);
    let observation_a = Uuid::from_u128(20_029);
    let evidence = Uuid::from_u128(20_030);
    insert_context(&db, ctx_a, "ctx-evidence-observation-a");
    insert_context(&db, ctx_b, "ctx-evidence-child-b");
    insert_entity(&db, entity_a, ctx_a, "entity-a");
    insert_observation(&db, observation_a, entity_a, ctx_a);

    let result = db.execute(
        "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, retention_status, supply_order) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $retention_status, $supply_order)",
        &params(vec![
            ("id", Value::Uuid(evidence)),
            ("observation_id", Value::Uuid(observation_a)),
            ("context_id", Value::Uuid(ctx_b)),
            ("kind", Value::Text("text".into())),
            (
                "source_ref",
                Value::Text("cas://sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into()),
            ),
            ("retention_status", Value::Text("retained_external".into())),
            ("supply_order", Value::Int64(0)),
        ]),
    );

    assert_error_is_fk(result);
    assert_eq!(
        scalar_count(
            &db,
            "SELECT count(*) AS n FROM evidence WHERE id = $id",
            &params(vec![("id", Value::Uuid(evidence))]),
        ),
        0
    );
}

#[test]
fn combination_evidence_derived_from_cross_context_raw_insert_rejected_by_schema() {
    let db = fresh_db();
    let ctx_a = Uuid::from_u128(20_031);
    let ctx_b = Uuid::from_u128(20_032);
    let entity_a = Uuid::from_u128(20_033);
    let entity_b = Uuid::from_u128(20_034);
    let observation_a = Uuid::from_u128(20_035);
    let observation_b = Uuid::from_u128(20_036);
    let parent_evidence = Uuid::from_u128(20_037);
    let child_evidence = Uuid::from_u128(20_038);
    insert_context(&db, ctx_a, "ctx-derived-parent-a");
    insert_context(&db, ctx_b, "ctx-derived-child-b");
    insert_entity(&db, entity_a, ctx_a, "entity-a");
    insert_entity(&db, entity_b, ctx_b, "entity-b");
    insert_observation(&db, observation_a, entity_a, ctx_a);
    insert_observation(&db, observation_b, entity_b, ctx_b);
    insert_evidence(&db, parent_evidence, observation_a, ctx_a, 0, None);

    let result = db.execute(
        "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, derived_from, retention_status, supply_order) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $derived_from, $retention_status, $supply_order)",
        &params(vec![
            ("id", Value::Uuid(child_evidence)),
            ("observation_id", Value::Uuid(observation_b)),
            ("context_id", Value::Uuid(ctx_b)),
            ("kind", Value::Text("text_projection".into())),
            (
                "source_ref",
                Value::Text("cas://sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into()),
            ),
            ("derived_from", Value::Uuid(parent_evidence)),
            ("retention_status", Value::Text("retained_external".into())),
            ("supply_order", Value::Int64(0)),
        ]),
    );

    assert_error_is_fk(result);
    assert_eq!(
        scalar_count(
            &db,
            "SELECT count(*) AS n FROM evidence WHERE id = $id",
            &params(vec![("id", Value::Uuid(child_evidence))]),
        ),
        0
    );
}

#[test]
fn combination_invalidation_cross_context_raw_insert_rejected_by_schema() {
    let db = fresh_db();
    let ctx_a = Uuid::from_u128(20_039);
    let ctx_b = Uuid::from_u128(20_040);
    let entity_a = Uuid::from_u128(20_041);
    let entity_b = Uuid::from_u128(20_042);
    let observation_a = Uuid::from_u128(20_043);
    let observation_b = Uuid::from_u128(20_044);
    let decision_a = Uuid::from_u128(20_045);
    let decision_b = Uuid::from_u128(20_046);
    let mismatch_decision = Uuid::from_u128(20_047);
    let mismatch_observation = Uuid::from_u128(20_048);
    let mismatch_resolution = Uuid::from_u128(20_049);
    insert_context(&db, ctx_a, "ctx-invalidation-a");
    insert_context(&db, ctx_b, "ctx-invalidation-b");
    insert_entity(&db, entity_a, ctx_a, "entity-a");
    insert_entity(&db, entity_b, ctx_b, "entity-b");
    insert_observation(&db, observation_a, entity_a, ctx_a);
    insert_observation(&db, observation_b, entity_b, ctx_b);
    insert_decision(&db, decision_a, ctx_a, "active");
    insert_decision(&db, decision_b, ctx_b, "active");

    let affected_mismatch = db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &params(vec![
            ("id", Value::Uuid(mismatch_decision)),
            ("status", Value::Text("pending".into())),
            ("context_id", Value::Uuid(ctx_b)),
            ("affected_decision_id", Value::Uuid(decision_a)),
            ("trigger_observation_id", Value::Uuid(observation_b)),
            ("basis_diff", Value::Json(json!({"field": "risk"}))),
            ("severity", Value::Text("warning".into())),
            ("detected_at", Value::TxId(TxId(1_u64))),
            ("resolved_at", Value::Null),
            ("resolution_action", Value::Null),
            ("resolution_decision_id", Value::Null),
        ]),
    );
    assert_error_is_fk(affected_mismatch);

    let trigger_mismatch = db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &params(vec![
            ("id", Value::Uuid(mismatch_observation)),
            ("status", Value::Text("pending".into())),
            ("context_id", Value::Uuid(ctx_b)),
            ("affected_decision_id", Value::Uuid(decision_b)),
            ("trigger_observation_id", Value::Uuid(observation_a)),
            ("basis_diff", Value::Json(json!({"field": "risk"}))),
            ("severity", Value::Text("warning".into())),
            ("detected_at", Value::TxId(TxId(1_u64))),
            ("resolved_at", Value::Null),
            ("resolution_action", Value::Null),
            ("resolution_decision_id", Value::Null),
        ]),
    );
    assert_error_is_fk(trigger_mismatch);

    let resolution_mismatch = db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &params(vec![
            ("id", Value::Uuid(mismatch_resolution)),
            ("status", Value::Text("acknowledged".into())),
            ("context_id", Value::Uuid(ctx_b)),
            ("affected_decision_id", Value::Uuid(decision_b)),
            ("trigger_observation_id", Value::Uuid(observation_b)),
            ("basis_diff", Value::Json(json!({"field": "risk"}))),
            ("severity", Value::Text("warning".into())),
            ("detected_at", Value::TxId(TxId(1_u64))),
            ("resolved_at", Value::TxId(TxId(2_u64))),
            ("resolution_action", Value::Text("new_decision_created".into())),
            ("resolution_decision_id", Value::Uuid(decision_a)),
        ]),
    );
    assert_error_is_fk(resolution_mismatch);

    for id in [mismatch_decision, mismatch_observation, mismatch_resolution] {
        assert_eq!(
            scalar_count(
                &db,
                "SELECT count(*) AS n FROM invalidations WHERE id = $id",
                &params(vec![("id", Value::Uuid(id))]),
            ),
            0
        );
    }
}

#[test]
fn combination_record_observation_concurrent_against_same_entity_snapshot_timeline_contiguous() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = context_graph::StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let store = context_graph::Store::open(config.clone()).expect("open");
    let (context, entity) = seed_store_context_entity(&store);
    drop(store);
    let barrier = Arc::new(Barrier::new(6));
    let handles: Vec<_> = (0..6)
        .map(|writer| {
            let config = config.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let store = context_graph::Store::open(config).expect("writer open");
                barrier.wait();
                store.record_observation(cg_observation(
                    context_graph::ObservationId::new_v7(),
                    context_graph::EvidenceId::new_v7(),
                    entity,
                    context,
                    writer,
                ))
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("writer joined").expect("record");
    }
    let snapshots = context_graph::Store::open(config)
        .expect("reader")
        .entity_history(entity)
        .expect("history");
    assert_eq!(snapshots.len(), 7);
    assert_eq!(
        snapshots
            .iter()
            .map(|snapshot| snapshot.captured_at)
            .collect::<BTreeSet<_>>()
            .len(),
        snapshots.len()
    );
    for pair in snapshots.windows(2) {
        assert_eq!(pair[0].valid_to, Some(pair[1].valid_from));
        assert!(pair[0].valid_from < pair[1].valid_from);
    }
}

#[test]
fn combination_invalidate_decision_observation_atomic_visibility_no_torn_reader() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = context_graph::StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    };
    let (decision_id, trigger_id) = seed_store_decision(&config);
    let writer = context_graph::Store::open(config).expect("writer open");
    writer
        .invalidate_decision(decision_id, trigger_id, "basis drift")
        .expect("invalidate");
    assert_eq!(
        writer
            .list_invalidations(
                store_decision_context(&writer, decision_id),
                Some(decision_id),
                None,
            )
            .expect("invalidations")
            .len(),
        1,
        "decision status and invalidation row must become visible at one horizon"
    );
}

#[test]
fn combination_record_observation_text_evidence_vector_text_hnsw_atomic_with_row_insert_rollback_reverts_index()
 {
    let db = fresh_db();
    assert_table_present(&db, "evidence");
    let context_id = Uuid::from_u128(20_032);
    let entity_id = Uuid::from_u128(20_033);
    let evidence_id = Uuid::from_u128(20_030);
    let observation_id = Uuid::from_u128(20_031);
    insert_context(&db, context_id, "ctx-vector-rollback");
    insert_entity(&db, entity_id, context_id, "entity-vector-rollback");
    insert_observation(&db, observation_id, entity_id, context_id);
    let tx = db.begin().expect("begin");
    let row_id = db
        .insert_row(
            tx,
            "evidence",
            params(vec![
                ("id", Value::Uuid(evidence_id)),
                ("observation_id", Value::Uuid(observation_id)),
                ("context_id", Value::Uuid(context_id)),
                ("kind", Value::Text("text".into())),
                ("source_ref", Value::Text("cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into())),
                ("retention_status", Value::Text("retained_external".into())),
                ("supply_order", Value::Int64(0)),
            ]),
        )
        .expect("insert evidence row");
    let query = vec![0.0; 768];
    let vector_insert = db.insert_vector(
        tx,
        contextdb_core::VectorIndexRef::new("evidence", "vector_text"),
        row_id,
        query.clone(),
    );
    assert!(vector_insert.is_ok());
    let violation = db.execute_in_tx(
        tx,
        "INSERT INTO evidence (id, observation_id, context_id, kind, source_ref, retention_status, supply_order) VALUES ($id, $observation_id, $context_id, $kind, $source_ref, $retention_status, $supply_order)",
        &params(vec![
            ("id", Value::Uuid(evidence_id)),
            ("observation_id", Value::Uuid(observation_id)),
            ("context_id", Value::Uuid(context_id)),
            ("kind", Value::Text("text".into())),
            ("source_ref", Value::Text("cas://sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into())),
            ("retention_status", Value::Text("retained_external".into())),
            ("supply_order", Value::Int64(1)),
        ]),
    );
    assert!(violation.is_err());
    db.rollback(tx).expect("rollback");
    let hits = db
        .query_vector(
            contextdb_core::VectorIndexRef::new("evidence", "vector_text"),
            &query,
            10,
            None,
            db.snapshot(),
        )
        .expect("query");
    assert!(
        hits.is_empty(),
        "vector writes must be committed and rolled back with their evidence row"
    );
}
