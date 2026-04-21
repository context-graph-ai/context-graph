use std::collections::{HashMap, HashSet};

use context_graph::schema::{DIGEST_VECTOR_DIMENSION, TABLES};
use contextdb_core::{Error, TxId, Value};
use contextdb_engine::{Database, QueryResult};
use serde_json::json;
use uuid::Uuid;

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
        "INSERT INTO observations (id, observation_type, data, source, entity_id, context_id, timestamp) VALUES ($id, $observation_type, $data, $source, $entity_id, $context_id, $timestamp)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("observation_type", Value::Text("config_change".into())),
            ("data", Value::Json(json!({"field": "version", "old": "1.0", "new": "2.0"}))),
            ("source", Value::Text("git-hook".into())),
            ("entity_id", Value::Uuid(entity_id)),
            ("context_id", Value::Uuid(context_id)),
            ("timestamp", Value::Timestamp(1_700_000_004)),
        ]),
    )
    .unwrap();
}

fn insert_outcome(db: &Database, id: Uuid, decision_id: Uuid) {
    db.execute(
        "INSERT INTO outcomes (id, decision_id, success, notes, recorded_at) VALUES ($id, $decision_id, $success, $notes, $recorded_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
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
    status: &str,
    affected_decision_id: Uuid,
    trigger_observation_id: Uuid,
) {
    db.execute(
        "INSERT INTO invalidations (id, status, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("status", Value::Text(status.to_string())),
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
    insert_invalidation(db, id, status, decision_id, observation_id);
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
        "INSERT INTO digests (id, source, session_ref, summary, embedding, context_id, properties, created_at) VALUES ($id, $source, $session_ref, $summary, $embedding, $context_id, $properties, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("source", Value::Text("claude-cli".into())),
            ("session_ref", Value::Text("session-1".into())),
            ("summary", Value::Text("Discussed auth module refactoring".into())),
            ("embedding", Value::Vector(embedding)),
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

fn insert_snapshot(db: &Database, id: Uuid, entity_id: Uuid, valid_from: i64, valid_to: i64) {
    db.execute(
        "INSERT INTO entity_snapshots (id, entity_id, data, valid_from, valid_to) VALUES ($id, $entity_id, $data, $valid_from, $valid_to)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("entity_id", Value::Uuid(entity_id)),
            ("data", Value::Json(json!({"version": valid_from}))),
            ("valid_from", Value::TxId(TxId(valid_from as u64))),
            ("valid_to", Value::TxId(TxId(valid_to as u64))),
        ]),
    )
    .unwrap();
}

fn insert_approval(db: &Database, id: Uuid, decision_id: Uuid) {
    db.execute(
        "INSERT INTO approvals (id, decision_id, approver_id, level, rationale, created_at) VALUES ($id, $decision_id, $approver_id, $level, $rationale, $created_at)",
        &params(vec![
            ("id", Value::Uuid(id)),
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

// observations table round-trip: observation_type, data JSON, source, entity_id and context_id FKs, and timestamp persist.
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
        row_value(&result, 0, "data"),
        &Value::Json(json!({"field": "version", "old": "1.0", "new": "2.0"}))
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
    insert_outcome(&db, id, decision_id);

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
    insert_invalidation(&db, id, "pending", decision_id, observation_id);

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

// digests table round-trip: source, session_ref, summary, embedding vector, context_id FK, properties, and created_at persist.
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
        row_value(&result, 0, "embedding"),
        &Value::Vector(embedding)
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
    insert_snapshot(&db, snapshot_id, entity_id, 1, 2);

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
        "INSERT INTO intention_snapshots (id, intention_id, description, status, origin, properties, tags, valid_from, valid_to) VALUES ($id, $intention_id, $description, $status, $origin, $properties, $tags, $valid_from, $valid_to)",
        &params(vec![
            ("id", Value::Uuid(snapshot_id)),
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
        "INSERT INTO intention_snapshots (id, intention_id, description, status, origin, properties, tags, valid_from, valid_to) VALUES ($id, $intention_id, $description, $status, $origin, $properties, $tags, $valid_from, $valid_to)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(320))),
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
    insert_approval(&db, id, decision_id);

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
    insert_snapshot(&db, snapshot_1, entity_id, 1, 2);
    insert_snapshot(&db, snapshot_2, entity_id, 2, 3);
    let result = db.execute(
        "INSERT INTO entity_snapshots (id, entity_id, data, valid_from, valid_to) VALUES ($id, $entity_id, $data, $valid_from, $valid_to)",
        &params(vec![
            ("id", Value::Uuid(snapshot_3)),
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
    insert_snapshot(&db, Uuid::from_u128(905), entity_id, 1, 3);
    let result = db.execute(
        "INSERT INTO entity_snapshots (id, entity_id, data, valid_from, valid_to) VALUES ($id, $entity_id, $data, $valid_from, $valid_to)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(906))),
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
        "INSERT INTO outcomes (id, decision_id, success, notes, recorded_at) VALUES ($id, $decision_id, $success, $notes, $recorded_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1000))),
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
        "INSERT INTO observations (id, observation_type, data, source, entity_id, context_id, timestamp) VALUES ($id, $observation_type, $data, $source, $entity_id, $context_id, $timestamp)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1003))),
            ("observation_type", Value::Text("config_change".into())),
            ("data", Value::Json(json!({"field": "x"}))),
            ("source", Value::Text("source".into())),
            ("entity_id", Value::Uuid(Uuid::from_u128(1004))),
            ("context_id", Value::Uuid(ctx)),
            ("timestamp", Value::Timestamp(1_700_000_061)),
        ]),
    ));
}

// invalidations.affected_decision_id FK must resolve to a real decisions row; a phantom target is rejected so INVALIDATES edges cannot dangle.
#[test]
fn test_fk_invalidation_invalid_decision_id_rejected() {
    let db = fresh_db();
    assert_error_is_fk(db.execute(
        "INSERT INTO invalidations (id, status, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1005))),
            ("status", Value::Text("pending".into())),
            ("affected_decision_id", Value::Uuid(Uuid::from_u128(1006))),
            ("trigger_observation_id", Value::Uuid(Uuid::from_u128(1007))),
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
        "INSERT INTO approvals (id, decision_id, approver_id, level, rationale, created_at) VALUES ($id, $decision_id, $approver_id, $level, $rationale, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1008))),
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
        "INSERT INTO digests (id, source, session_ref, summary, embedding, context_id, properties, created_at) VALUES ($id, $source, $session_ref, $summary, $embedding, $context_id, $properties, $created_at)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1016))),
            ("source", Value::Text("source".into())),
            ("session_ref", Value::Null),
            ("summary", Value::Text("summary".into())),
            ("embedding", Value::Vector(vec![0.0; DIGEST_VECTOR_DIMENSION])),
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
    assert_error_is_fk(db.execute(
        "INSERT INTO invalidations (id, status, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1018))),
            ("status", Value::Text("pending".into())),
            ("affected_decision_id", Value::Uuid(Uuid::from_u128(1019))),
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

// decisions.decision_type is NOT NULL; list_decisions and audit queries key off this column, so a null value must be rejected at insert.
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
        "INSERT INTO observations (id, observation_type, data, source, entity_id, context_id, timestamp) VALUES ($id, $observation_type, $data, $source, $entity_id, $context_id, $timestamp)",
        &params(vec![
            ("id", Value::Uuid(Uuid::from_u128(1036))),
            ("observation_type", Value::Text("config_change".into())),
            ("data", Value::Json(json!({}))),
            ("source", Value::Null),
            ("entity_id", Value::Uuid(entity_id)),
            ("context_id", Value::Uuid(ctx)),
            ("timestamp", Value::Timestamp(1_700_000_073)),
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

// digests.embedding is a fixed-dimension vector column (DIGEST_VECTOR_DIMENSION); the engine must round-trip a full-dimension embedding without truncation or dimension drift.
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
    match row_value(&result, 0, "embedding") {
        Value::Vector(v) => assert_eq!(v.len(), DIGEST_VECTOR_DIMENSION),
        other => panic!("expected vector embedding, got {other:?}"),
    }
}

// the cosine-distance operator on digests.embedding must honor context_id scoping during ANN ordering so semantic search never surfaces a closer digest from a sibling context.
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

    let query = vec![1.0_f32, 0.0];
    let result = db
        .execute(
            "SELECT id, context_id FROM digests WHERE context_id = $context_id ORDER BY embedding <=> $query LIMIT 2",
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
