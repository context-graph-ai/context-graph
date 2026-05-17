use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Barrier,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

use chrono::Utc;
use context_graph::*;
use contextdb_core::{Result as EngineResult, TxId, Value};
use contextdb_engine::Database;
use serde_json::{Value as JsonValue, json};
use tempfile::TempDir;

macro_rules! ok {
    ($expr:expr, $step:literal) => {{
        let result = $expr;
        assert!(result.is_ok(), "{}: expected Ok, got {:?}", $step, result);
        result.unwrap()
    }};
}

fn fixture() -> (TempDir, PathBuf, Store) {
    let tempdir = TempDir::new().expect("tempdir");
    let db_path = tempdir.path().join("context-graph.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open");
    (tempdir, db_path, store)
}

fn reopen(db_path: &Path) -> Store {
    Store::open(StoreConfig {
        db_path: db_path.to_path_buf(),
        ..Default::default()
    })
    .expect("open")
}

fn current_tx(db_path: &Path) -> TxId {
    let db = raw_db(db_path);
    TxId::from_snapshot(db.snapshot())
}

fn raw_db(db_path: &Path) -> Database {
    let db = Database::open(db_path).expect("raw db");
    let registered = db.registered_trigger_callbacks();
    for trigger in db.list_triggers() {
        if !registered.iter().any(|name| name == &trigger.name) {
            db.register_trigger_callback(&trigger.name, |_db, _ctx| -> EngineResult<()> { Ok(()) })
                .expect("register noop trigger");
        }
    }
    db.complete_initialization().expect("raw db triggers ready");
    db
}

fn scoped(db_path: &Path, context_id: ContextId) -> Store {
    Store::open_with_contexts(
        StoreConfig {
            db_path: db_path.to_path_buf(),
            ..Default::default()
        },
        BTreeSet::from([context_id]),
    )
    .expect("open scoped")
}

fn context(store: &Store, name: &str) -> Context {
    ok!(
        store.create_context(CreateContext {
            name: name.to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        }),
        "create context"
    )
}

fn intention(store: &Store, context_id: ContextId, description: &str) -> Intention {
    intention_with_status(store, context_id, description, IntentionStatus::Active)
}

fn intention_with_status(
    store: &Store,
    context_id: ContextId,
    description: &str,
    status: IntentionStatus,
) -> Intention {
    ok!(
        store.create_intention(CreateIntention {
            id: None,
            description: description.to_string(),
            status,
            origin: IntentionOrigin::Human,
            context_id,
            properties: BTreeMap::from([(
                "fixture_marker".to_string(),
                json!(format!("intention-{description}")),
            )]),
            tags: vec![format!("intent-{}", description.replace(' ', "-"))],
        }),
        "create intention"
    )
}

fn untagged_intention(store: &Store, context_id: ContextId, description: &str) -> Intention {
    ok!(
        store.create_intention(CreateIntention {
            id: None,
            description: description.to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            properties: BTreeMap::from([(
                "fixture_marker".to_string(),
                json!(format!("intention-{description}")),
            )]),
            tags: Vec::new(),
        }),
        "create untagged intention"
    )
}

fn entity(store: &Store, context_id: ContextId, name: &str) -> Entity {
    ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: name.to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id,
        }),
        "create entity"
    )
}

fn entity_with_properties(
    store: &Store,
    context_id: ContextId,
    name: &str,
    properties: BTreeMap<String, JsonValue>,
) -> Entity {
    ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: name.to_string(),
            properties,
            tags: vec![],
            context_id,
        }),
        "create entity"
    )
}

fn reasoning(label: &str) -> Vec<ReasoningStep> {
    vec![ReasoningStep {
        operation: format!("op-{label}"),
        details: format!("details-{label}"),
        result: format!("result-{label}"),
        agent_id: Some("acceptance-agent".to_string()),
        timestamp: Utc::now(),
    }]
}

fn decision_input(
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
) -> CreateDecision {
    CreateDecision {
        decision_type: format!("architecture-{}", description.replace(' ', "-")),
        description: description.to_string(),
        reasoning: reasoning(description),
        confidence: Some(0.9),
        intention_ids,
        based_on_entity_ids,
        basis_fields: None,
        based_on_snapshots: vec![],
        tags: vec![format!("decision-{}", description.replace(' ', "-"))],
        properties: BTreeMap::from([
            (
                "fixture_marker".to_string(),
                json!(format!("decision-{description}")),
            ),
            ("non_default".to_string(), json!(true)),
        ]),
        context_id,
        precedent_ids: vec![],
        agent_id: Some(format!("agent-{}", description.replace(' ', "-"))),
    }
}

fn untagged_decision(
    store: &Store,
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
) -> Decision {
    let mut input = decision_input(context_id, description, intention_ids, based_on_entity_ids);
    input.tags = Vec::new();
    ok!(store.create_decision(input), "create untagged decision")
}

fn decision(
    store: &Store,
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
) -> Decision {
    ok!(
        store.create_decision(decision_input(
            context_id,
            description,
            intention_ids,
            based_on_entity_ids,
        )),
        "create decision"
    )
}

fn decision_with_basis_fields(
    store: &Store,
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
    basis_fields: Option<HashMap<EntityId, Vec<String>>>,
) -> Decision {
    let mut input = decision_input(context_id, description, intention_ids, based_on_entity_ids);
    input.basis_fields = basis_fields;
    ok!(store.create_decision(input), "create decision")
}

fn insert_raw_edges(
    db_path: &Path,
    context_id: ContextId,
    relationship: &str,
    edges: &[(uuid::Uuid, uuid::Uuid)],
) {
    let db = raw_db(db_path);
    for (source_id, target_id) in edges {
        db.execute(
            "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
            &HashMap::from([
                ("id".to_string(), Value::Uuid(uuid::Uuid::now_v7())),
                ("source_id".to_string(), Value::Uuid(*source_id)),
                ("target_id".to_string(), Value::Uuid(*target_id)),
                ("edge_type".to_string(), Value::Text(relationship.to_string())),
                ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
                ("valid_from".to_string(), Value::TxId(TxId(1))),
                ("valid_to".to_string(), Value::Null),
                ("properties".to_string(), Value::Json(json!({}))),
            ]),
        )
        .expect("insert raw edge");
    }
}

fn insert_about_edges(db_path: &Path, context_id: ContextId, edges: &[(DecisionId, EntityId)]) {
    let raw_edges = edges
        .iter()
        .map(|(decision_id, entity_id)| (decision_id.as_uuid(), entity_id.as_uuid()))
        .collect::<Vec<_>>();
    insert_raw_edges(db_path, context_id, "ABOUT", &raw_edges);
}

fn insert_raw_decision(
    db_path: &Path,
    context_id: ContextId,
    id: DecisionId,
    description: &str,
    created_at: TxId,
) {
    let db = raw_db(db_path);
    db.execute(
        "INSERT INTO decisions (id, decision_type, description, reasoning, confidence, status, agent_id, context_id, properties, tags, created_at) VALUES ($id, $decision_type, $description, $reasoning, $confidence, $status, $agent_id, $context_id, $properties, $tags, $created_at)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            (
                "decision_type".to_string(),
                Value::Text(format!("raw-{}", description.replace(' ', "-"))),
            ),
            ("description".to_string(), Value::Text(description.to_string())),
            (
                "reasoning".to_string(),
                Value::Json(json!([{
                    "operation": format!("raw-op-{description}"),
                    "details": format!("raw-details-{description}"),
                    "result": format!("raw-result-{description}"),
                    "agent_id": "raw-agent",
                    "timestamp": "2026-01-02T03:04:05Z"
                }])),
            ),
            ("confidence".to_string(), Value::Float64(0.7)),
            ("status".to_string(), Value::Text("active".to_string())),
            ("agent_id".to_string(), Value::Text("raw-agent".to_string())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            (
                "properties".to_string(),
                Value::Json(json!({"fixture_marker": format!("raw-{description}")})),
            ),
            ("tags".to_string(), Value::Json(json!([]))),
            ("created_at".to_string(), Value::TxId(created_at)),
        ]),
    )
    .expect("insert raw decision");
}

fn insert_raw_intention(
    db_path: &Path,
    context_id: ContextId,
    id: IntentionId,
    description: &str,
    created_at: TxId,
) {
    let db = Database::open(db_path).expect("raw db");
    db.execute(
        "INSERT INTO intentions (id, description, status, context_id, origin, origin_ref, blueprint_id, bindings, blueprint_proposal, properties, tags, created_at) VALUES ($id, $description, $status, $context_id, $origin, $origin_ref, $blueprint_id, $bindings, $blueprint_proposal, $properties, $tags, $created_at)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(id.as_uuid())),
            ("description".to_string(), Value::Text(description.to_string())),
            ("status".to_string(), Value::Text("active".to_string())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("origin".to_string(), Value::Text("human".to_string())),
            ("origin_ref".to_string(), Value::Null),
            ("blueprint_id".to_string(), Value::Null),
            ("bindings".to_string(), Value::Null),
            ("blueprint_proposal".to_string(), Value::Null),
            (
                "properties".to_string(),
                Value::Json(json!({"fixture_marker": format!("raw-{description}")})),
            ),
            ("tags".to_string(), Value::Json(json!([]))),
            ("created_at".to_string(), Value::TxId(created_at)),
        ]),
    )
    .expect("insert raw intention");
}

fn engine_contexts(contexts: &[uuid::Uuid]) -> BTreeSet<contextdb_core::ContextId> {
    contexts
        .iter()
        .copied()
        .map(contextdb_core::ContextId::new)
        .collect()
}

fn engine_uuid_set(result: &contextdb_engine::QueryResult, column: &str) -> BTreeSet<uuid::Uuid> {
    let idx = result
        .columns
        .iter()
        .position(|candidate| candidate == column || candidate.rsplit('.').next() == Some(column));
    assert!(
        idx.is_some(),
        "column {column} not found in {:?}",
        result.columns
    );
    let idx = idx.unwrap();
    result
        .rows
        .iter()
        .map(|row| match &row[idx] {
            Value::Uuid(id) => *id,
            other => {
                assert!(
                    matches!(other, Value::Uuid(_)),
                    "expected UUID in column {column}, got {other:?}"
                );
                uuid::Uuid::nil()
            }
        })
        .collect()
}

fn assert_contextdb_explicit_anchor_scope_violation() {
    let db = Database::open_memory();
    db.execute(
        "CREATE TABLE t (id UUID PRIMARY KEY, context_id UUID CONTEXT_ID, data TEXT)",
        &HashMap::new(),
    )
    .expect("create scoped table");
    let ctx_a = uuid::Uuid::from_u128(0xA);
    let ctx_b = uuid::Uuid::from_u128(0xB);
    let hidden = uuid::Uuid::from_u128(0xB01);
    db.execute(
        "INSERT INTO t (id, context_id, data) VALUES ($id, $context_id, $data)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(hidden)),
            ("context_id".to_string(), Value::Uuid(ctx_b)),
            ("data".to_string(), Value::Text("hidden".to_string())),
        ]),
    )
    .expect("insert hidden row");

    let scoped = db.scoped_with_contexts(engine_contexts(&[ctx_a]));
    let err = scoped
        .execute(
            "SELECT data FROM t WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Uuid(hidden))]),
        )
        .expect_err("hidden explicit anchor must be refused by contextdb");
    assert!(
        matches!(err, contextdb_core::Error::ContextScopeViolation { .. }),
        "expected ContextScopeViolation, got {err:?}"
    );
    if let contextdb_core::Error::ContextScopeViolation { requested, allowed } = err {
        assert_eq!(requested, contextdb_core::ContextId::new(ctx_b));
        assert_eq!(allowed, engine_contexts(&[ctx_a]));
    }
}

fn insert_engine_node(db: &Database, id: uuid::Uuid) {
    db.execute(
        "INSERT INTO nodes (id) VALUES ($id)",
        &HashMap::from([("id".to_string(), Value::Uuid(id))]),
    )
    .expect("insert engine node");
}

fn insert_engine_edge(db: &Database, source: uuid::Uuid, target: uuid::Uuid, edge_type: &str) {
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type) VALUES ($id, $source_id, $target_id, $edge_type)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(uuid::Uuid::now_v7())),
            ("source_id".to_string(), Value::Uuid(source)),
            ("target_id".to_string(), Value::Uuid(target)),
            ("edge_type".to_string(), Value::Text(edge_type.to_string())),
        ]),
    )
    .expect("insert engine edge");
}

fn assert_contextdb_pinned_discovery_graph_composition() {
    let db = Database::open_memory();
    db.execute("CREATE TABLE nodes (id UUID PRIMARY KEY)", &HashMap::new())
        .expect("create nodes");
    db.execute(
        "CREATE TABLE edges (id UUID PRIMARY KEY, source_id UUID, target_id UUID, edge_type TEXT)",
        &HashMap::new(),
    )
    .expect("create edges");

    let entity = uuid::Uuid::from_u128(0xE);
    let based_pre = uuid::Uuid::from_u128(0xB01);
    let about_pre = uuid::Uuid::from_u128(0xA01);
    let based_post = uuid::Uuid::from_u128(0xB02);
    let about_post = uuid::Uuid::from_u128(0xA02);
    let served_based_pre = uuid::Uuid::from_u128(0x501);
    let served_about_pre = uuid::Uuid::from_u128(0x502);
    let served_based_post = uuid::Uuid::from_u128(0x503);
    let served_about_post = uuid::Uuid::from_u128(0x504);
    for id in [
        entity,
        based_pre,
        about_pre,
        based_post,
        about_post,
        served_based_pre,
        served_about_pre,
        served_based_post,
        served_about_post,
    ] {
        insert_engine_node(&db, id);
    }

    insert_engine_edge(&db, based_pre, entity, "BASED_ON");
    insert_engine_edge(&db, about_pre, entity, "ABOUT");
    insert_engine_edge(&db, based_pre, served_based_pre, "SERVES");
    insert_engine_edge(&db, about_pre, served_about_pre, "SERVES");
    let snapshot = db.snapshot();
    insert_engine_edge(&db, based_post, entity, "BASED_ON");
    insert_engine_edge(&db, about_post, entity, "ABOUT");
    insert_engine_edge(&db, based_post, served_based_post, "SERVES");
    insert_engine_edge(&db, about_post, served_about_post, "SERVES");

    let sql = "WITH based AS (
            SELECT d_id FROM GRAPH_TABLE (edges MATCH (e)<-[:BASED_ON]-(d) WHERE e.id = $entity COLUMNS (d.id AS d_id))
        ), about AS (
            SELECT d_id FROM GRAPH_TABLE (edges MATCH (e)<-[:ABOUT]-(d) WHERE e.id = $entity COLUMNS (d.id AS d_id))
        ), served_by_based AS (
            SELECT i_id FROM GRAPH_TABLE (edges MATCH (d)-[:SERVES]->(i) WHERE d.id IN (SELECT d_id FROM based) COLUMNS (i.id AS i_id))
        ), served_by_about AS (
            SELECT i_id FROM GRAPH_TABLE (edges MATCH (d)-[:SERVES]->(i) WHERE d.id IN (SELECT d_id FROM about) COLUMNS (i.id AS i_id))
        )
        SELECT id FROM nodes
        WHERE id IN (SELECT i_id FROM served_by_based)
           OR id IN (SELECT i_id FROM served_by_about)";
    let result = db
        .execute_at_snapshot(
            sql,
            &HashMap::from([("entity".to_string(), Value::Uuid(entity))]),
            snapshot,
        )
        .expect("pinned discovery graph query");
    assert_eq!(
        engine_uuid_set(&result, "id"),
        BTreeSet::from([served_based_pre, served_about_pre])
    );
}

fn producer() -> EvidenceProducer {
    EvidenceProducer {
        system: "acceptance".to_string(),
        model_name: "fixture".to_string(),
        model_version: "1".to_string(),
        pipeline_version: "1".to_string(),
    }
}

fn evidence_ref(
    observation_id: ObservationId,
    evidence_id: EvidenceId,
    context_id: ContextId,
) -> EvidenceRef {
    EvidenceRef {
        id: evidence_id,
        observation_id,
        context_id,
        kind: EvidenceKind::Text,
        source_ref: "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        mime_type: Some("text/plain".to_string()),
        content_hash: Some(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
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

fn observation(store: &Store, context_id: ContextId, entity_id: EntityId) -> ObservationId {
    let observation_id = ObservationId::new_v7();
    let evidence_id = EvidenceId::new_v7();
    ok!(
        store.record_observation(RecordObservation {
            id: observation_id,
            entity_id,
            context_id,
            observation_type: "acceptance".to_string(),
            source: "acceptance".to_string(),
            observed_at: Utc::now(),
            evidence: vec![evidence_ref(observation_id, evidence_id, context_id)],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::from([("marker".to_string(), Some(json!("changed")))]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        }),
        "record observation"
    );
    observation_id
}

fn invalidate_decision_for_test(
    store: &Store,
    context_id: ContextId,
    decision_id: DecisionId,
) -> Decision {
    let marker = entity(store, context_id, "invalidation-marker");
    let observation_id = observation(store, context_id, marker.id);
    ok!(
        store.invalidate_decision(decision_id, observation_id, "basis changed"),
        "invalidate decision"
    );
    ok!(store.get_decision(decision_id), "get invalidated decision").expect("decision")
}

fn supersede_with_side_replacement(
    store: &Store,
    context_id: ContextId,
    decision_id: DecisionId,
) -> Decision {
    let side = intention(store, context_id, "replacement side intention");
    ok!(
        store.supersede_decision(
            decision_id,
            decision_input(
                context_id,
                "replacement outside discovery anchor",
                vec![side.id],
                vec![],
            ),
        ),
        "supersede decision"
    );
    ok!(store.get_decision(decision_id), "get superseded decision").expect("decision")
}

fn archive_decision_for_test(store: &Store, decision_id: DecisionId, reason: &str) -> Decision {
    ok!(
        store.archive_decision(decision_id, reason.to_string()),
        "archive decision"
    )
}

fn decision_filter(statuses: Vec<DecisionStatus>) -> DiscoveryFilter {
    DiscoveryFilter {
        include_decision_statuses: Some(statuses),
        include_intention_statuses: None,
    }
}

fn discovery_filter(
    decision_statuses: Option<Vec<DecisionStatus>>,
    intention_statuses: Option<Vec<IntentionStatus>>,
) -> DiscoveryFilter {
    DiscoveryFilter {
        include_decision_statuses: decision_statuses,
        include_intention_statuses: intention_statuses,
    }
}

fn decision_ids(rows: &[Decision]) -> BTreeSet<DecisionId> {
    rows.iter().map(|row| row.id).collect()
}

fn decision_id_order(rows: &[Decision]) -> Vec<DecisionId> {
    rows.iter().map(|row| row.id).collect()
}

fn intention_ids(rows: &[Intention]) -> BTreeSet<IntentionId> {
    rows.iter().map(|row| row.id).collect()
}

fn intention_id_order(rows: &[Intention]) -> Vec<IntentionId> {
    rows.iter().map(|row| row.id).collect()
}

fn assert_decision_ids(rows: &[Decision], expected: &[DecisionId]) {
    assert_eq!(
        rows.len(),
        expected.len(),
        "duplicate or missing decision rows: {rows:?}"
    );
    assert_eq!(
        decision_ids(rows),
        expected.iter().copied().collect::<BTreeSet<_>>()
    );
}

fn assert_decision_order(rows: &[Decision], expected: &[DecisionId]) {
    assert_eq!(decision_id_order(rows), expected);
}

fn assert_intention_order(rows: &[Intention], expected: &[IntentionId]) {
    assert_eq!(intention_id_order(rows), expected);
}

fn assert_intention_ids(rows: &[Intention], expected: &[IntentionId]) {
    assert_eq!(
        rows.len(),
        expected.len(),
        "duplicate or missing intention rows: {rows:?}"
    );
    assert_eq!(
        intention_ids(rows),
        expected.iter().copied().collect::<BTreeSet<_>>(),
        "rows: {rows:?}"
    );
}

fn assert_decisions_sorted(rows: &[Decision]) {
    assert!(rows.windows(2).all(|pair| {
        pair[0].created_at.0 > pair[1].created_at.0
            || (pair[0].created_at == pair[1].created_at && pair[0].id <= pair[1].id)
    }));
}

fn assert_decision_rows_match_get(store: &Store, rows: &[Decision]) {
    for row in rows {
        let current = ok!(store.get_decision(row.id), "get discovered decision")
            .expect("discovered decision exists");
        assert_eq!(row, &current);
    }
}

fn assert_intention_rows_match_get(store: &Store, rows: &[Intention]) {
    for row in rows {
        let current = ok!(store.get_intention(row.id), "get discovered intention")
            .expect("discovered intention exists");
        assert_eq!(row, &current);
    }
}

fn child_module_path(base_dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let file_path = base_dir.join(format!("{name}.rs"));
    if file_path.exists() {
        return file_path;
    }
    base_dir.join(name).join("mod.rs")
}

fn collect_compiled_store_sources(
    path: std::path::PathBuf,
    module_dir: std::path::PathBuf,
    seen: &mut BTreeSet<std::path::PathBuf>,
    sources: &mut Vec<(std::path::PathBuf, String)>,
) {
    let canonical = std::fs::canonicalize(&path).expect("compiled store module path");
    if !seen.insert(canonical) {
        return;
    }
    let source = std::fs::read_to_string(&path).expect("compiled store module source");
    sources.push((path.clone(), source.clone()));
    let code = code_mask_for_function_scan(&source);
    for line in code.lines().map(str::trim_start) {
        let rest = line
            .strip_prefix("mod ")
            .or_else(|| line.strip_prefix("pub mod "))
            .or_else(|| line.strip_prefix("pub(crate) mod "))
            .or_else(|| line.strip_prefix("pub(super) mod "));
        let Some(rest) = rest else {
            continue;
        };
        let name = rest
            .trim()
            .trim_end_matches(';')
            .split_whitespace()
            .next()
            .unwrap_or("");
        if name.is_empty() || rest.contains('{') {
            continue;
        }
        let child = child_module_path(&module_dir, name);
        assert!(
            child.exists(),
            "compiled store module declaration missing source: {name}"
        );
        let child_module_dir = if child.file_name().and_then(|name| name.to_str()) == Some("mod.rs")
        {
            child.parent().expect("module parent").to_path_buf()
        } else {
            module_dir.join(name)
        };
        collect_compiled_store_sources(child, child_module_dir, seen, sources);
    }
}

fn store_source_files() -> Vec<(std::path::PathBuf, String)> {
    let store_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/store");
    let mut sources = Vec::new();
    let mut seen = BTreeSet::new();
    collect_compiled_store_sources(store_dir.join("mod.rs"), store_dir, &mut seen, &mut sources);
    sources.sort_by(|a, b| a.0.cmp(&b.0));
    sources
}

fn mask_byte(out: &mut Vec<u8>, byte: u8) {
    if byte == b'\n' {
        out.push(b'\n');
    } else {
        out.push(b' ');
    }
}

fn raw_string_at(bytes: &[u8], index: usize) -> Option<(usize, usize)> {
    let mut raw_start = index;
    if bytes.get(index) == Some(&b'b') && bytes.get(index + 1) == Some(&b'r') {
        raw_start = index + 1;
    } else if bytes.get(index) != Some(&b'r') {
        return None;
    }

    let mut cursor = raw_start + 1;
    let mut hashes = 0;
    while bytes.get(cursor) == Some(&b'#') {
        hashes += 1;
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return None;
    }

    let prefix_len = cursor - index + 1;
    Some((prefix_len, hashes))
}

fn transform_source(source: &str, mask_strings: bool) -> String {
    let bytes = source.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        if bytes.get(index) == Some(&b'/') && bytes.get(index + 1) == Some(&b'/') {
            while index < bytes.len() {
                let byte = bytes[index];
                mask_byte(&mut out, byte);
                index += 1;
                if byte == b'\n' {
                    break;
                }
            }
        } else if bytes.get(index) == Some(&b'/') && bytes.get(index + 1) == Some(&b'*') {
            let mut depth = 0usize;
            while index < bytes.len() {
                if bytes.get(index) == Some(&b'/') && bytes.get(index + 1) == Some(&b'*') {
                    mask_byte(&mut out, bytes[index]);
                    mask_byte(&mut out, bytes[index + 1]);
                    index += 2;
                    depth += 1;
                } else if bytes.get(index) == Some(&b'*') && bytes.get(index + 1) == Some(&b'/') {
                    mask_byte(&mut out, bytes[index]);
                    mask_byte(&mut out, bytes[index + 1]);
                    index += 2;
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                } else {
                    mask_byte(&mut out, bytes[index]);
                    index += 1;
                }
            }
        } else if let Some((prefix_len, hashes)) = raw_string_at(bytes, index) {
            for _ in 0..prefix_len {
                if mask_strings {
                    mask_byte(&mut out, bytes[index]);
                } else {
                    out.push(bytes[index]);
                }
                index += 1;
            }

            while index < bytes.len() {
                let terminator = bytes[index] == b'"'
                    && (0..hashes).all(|offset| bytes.get(index + 1 + offset) == Some(&b'#'));
                if terminator {
                    for _ in 0..=hashes {
                        if mask_strings {
                            mask_byte(&mut out, bytes[index]);
                        } else {
                            out.push(bytes[index]);
                        }
                        index += 1;
                    }
                    break;
                }

                if mask_strings {
                    mask_byte(&mut out, bytes[index]);
                } else {
                    out.push(bytes[index]);
                }
                index += 1;
            }
        } else if (bytes.get(index) == Some(&b'"'))
            || (bytes.get(index) == Some(&b'b') && bytes.get(index + 1) == Some(&b'"'))
        {
            let prefix_len = if bytes[index] == b'b' { 2 } else { 1 };
            for _ in 0..prefix_len {
                if mask_strings {
                    mask_byte(&mut out, bytes[index]);
                } else {
                    out.push(bytes[index]);
                }
                index += 1;
            }
            let mut escaped = false;
            while index < bytes.len() {
                let byte = bytes[index];
                if mask_strings {
                    mask_byte(&mut out, byte);
                } else {
                    out.push(byte);
                }
                index += 1;
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    break;
                }
            }
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }

    String::from_utf8(out).expect("source transform preserves utf-8")
}

fn code_mask_for_function_scan(source: &str) -> String {
    transform_source(source, true)
}

fn source_without_comments(source: &str) -> String {
    transform_source(source, false)
}

fn function_start_in_code(code: &str, method_name: &str) -> Option<usize> {
    let mut offset = 0usize;
    for line in code.split_inclusive('\n') {
        let trim_offset = line.len() - line.trim_start().len();
        let trimmed = &line[trim_offset..];
        for prefix in ["fn ", "pub fn ", "pub(super) fn ", "pub(crate) fn "] {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                let name = rest
                    .split(|ch: char| ch == '(' || ch == '<' || ch.is_whitespace())
                    .next()
                    .unwrap_or("");
                if name == method_name {
                    return Some(offset + trim_offset);
                }
            }
        }
        offset += line.len();
    }
    None
}

fn store_function_names() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for (_, source) in store_source_files() {
        let code = code_mask_for_function_scan(&source);
        for line in code.lines().map(str::trim_start) {
            for prefix in ["fn ", "pub fn ", "pub(super) fn ", "pub(crate) fn "] {
                if let Some(rest) = line.strip_prefix(prefix) {
                    let name = rest
                        .split(|ch: char| ch == '(' || ch == '<' || ch.is_whitespace())
                        .next()
                        .unwrap_or("");
                    if !name.is_empty() {
                        names.insert(name.to_string());
                    }
                }
            }
        }
    }
    names
}

fn store_method_source(method_name: &str) -> String {
    let mut matches = Vec::new();
    for (path, source) in store_source_files() {
        let code = code_mask_for_function_scan(&source);
        if let Some(start) = function_start_in_code(&code, method_name) {
            let body_start = code[start..]
                .find('{')
                .map(|offset| start + offset)
                .expect("method body missing");
            let mut depth = 0usize;
            let mut end = None;
            for (offset, byte) in code[body_start..].bytes().enumerate() {
                if byte == b'{' {
                    depth += 1;
                } else if byte == b'}' {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(body_start + offset + 1);
                        break;
                    }
                }
            }
            let end = end.expect("method body unterminated");
            matches.push((path, source_without_comments(&source[start..end])));
        }
    }
    assert_eq!(
        matches.len(),
        1,
        "{method_name} source must appear exactly once in compiled store modules: {:?}",
        matches
            .iter()
            .map(|(path, _)| path.display().to_string())
            .collect::<Vec<_>>()
    );
    matches.pop().unwrap().1
}

fn source_lines(source: &str) -> impl Iterator<Item = &str> {
    source
        .lines()
        .map(str::trim_start)
        .filter(|line| !line.starts_with("//"))
}

fn assert_method_source_lacks(method: &str, source: &str, forbidden: &[&str]) {
    let code = code_mask_for_function_scan(source);
    for needle in forbidden {
        assert!(
            !source_lines(&code).any(|line| line.contains(needle)),
            "{method} must not use cg-side discovery compensation through {needle}"
        );
    }
}

fn assert_no_unapproved_store_helper_calls(method: &str, source: &str, allowed_helpers: &[&str]) {
    let allowed = allowed_helpers.iter().copied().collect::<BTreeSet<&str>>();
    let code = code_mask_for_function_scan(source);
    let calls = call_names_in_function(&code);
    for name in store_function_names() {
        if allowed.contains(name.as_str()) {
            continue;
        }
        assert!(
            !calls.iter().any(|call| call == &name),
            "{method} must not delegate Discovery behavior to unscanned helper {name}"
        );
    }
    for line in source_lines(&code).filter(|line| line.contains("self.")) {
        assert!(
            line.contains("self.db.")
                || line.contains("self.decode_discovery_decisions_at_read(")
                || line.contains("self.intentions_for_decision_at_read(")
                || line.contains("self.basis_snapshot_ids_for_decision_at_read(")
                || line.contains("self.outcome_for_decision_at_read(")
                || line.contains("self.precedents_for_decision_at_read("),
            "{method} must not delegate Discovery behavior through {line}"
        );
    }
}

fn call_names_before_result(source: &str) -> Vec<String> {
    let code = code_mask_for_function_scan(source);
    let body_start = code.find('{').expect("method body missing");
    let result_start = code
        .find("let result =")
        .expect("method must bind discovery query result");
    let mut calls = Vec::new();
    let mut index = body_start;
    let bytes = code.as_bytes();
    while index < result_start {
        let byte = bytes[index];
        if byte.is_ascii_alphabetic() || byte == b'_' {
            let name_start = index;
            index += 1;
            while index < result_start
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            let name = &code[name_start..index];
            let mut lookahead = index;
            while lookahead < result_start && bytes[lookahead].is_ascii_whitespace() {
                lookahead += 1;
            }
            if lookahead < result_start && bytes[lookahead] == b'!' {
                continue;
            }
            if lookahead < result_start && bytes[lookahead] == b'(' {
                calls.push(name.to_string());
            }
        } else {
            index += 1;
        }
    }
    calls
}

fn call_names_in_function(source: &str) -> Vec<String> {
    let code = code_mask_for_function_scan(source);
    let body_start = code.find('{').expect("method body missing");
    let mut calls = Vec::new();
    let mut index = body_start;
    let bytes = code.as_bytes();
    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_alphabetic() || byte == b'_' {
            let name_start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            let name = &code[name_start..index];
            let mut lookahead = index;
            while lookahead < bytes.len() && bytes[lookahead].is_ascii_whitespace() {
                lookahead += 1;
            }
            if lookahead < bytes.len() && bytes[lookahead] == b'!' {
                continue;
            }
            if lookahead < bytes.len() && bytes[lookahead] == b'(' {
                calls.push(name.to_string());
            }
        } else {
            index += 1;
        }
    }
    calls
}

fn assert_only_allowed_calls(method: &str, source: &str, allowed_calls: &[&str]) {
    let allowed = allowed_calls.iter().copied().collect::<BTreeSet<&str>>();
    for call in call_names_in_function(source) {
        assert!(
            allowed.contains(call.as_str()),
            "{method} must not delegate Discovery behavior through {call}(...)"
        );
    }
}

fn assert_no_candidate_work_before_result(method: &str, source: &str) {
    let allowed = [
        "as_uuid",
        "decision_statuses_for_discovery",
        "Err",
        "execute_at_snapshot",
        "format",
        "from",
        "intention_statuses_for_discovery",
        "is_empty",
        "is_valid",
        "map_err",
        "new",
        "Ok",
        "snapshot",
        "sql_string_list",
        "to_string",
        "Uuid",
    ]
    .into_iter()
    .collect::<BTreeSet<&str>>();
    for call in call_names_before_result(source) {
        assert!(
            allowed.contains(call.as_str()),
            "{method} must not compute Discovery candidates before the inspected query via {call}(...)"
        );
    }
}

fn assert_anchor_scope_check_shape(method: &str, source: &str) {
    let sql_start = source
        .find("let sql = format!(")
        .expect("method must bind discovery SQL");
    let prefix = &source[..sql_start];
    assert_eq!(
        prefix.matches(".execute_at_snapshot(").count(),
        1,
        "{method} must execute exactly one anchor-scope check before Discovery SQL"
    );
    if method.contains("by_intention") {
        assert_eq!(
            prefix
                .matches("SELECT id FROM intentions WHERE id = $intention_id")
                .count(),
            1,
            "{method} anchor scope check must contain exactly one Intention row lookup"
        );
        assert!(
            prefix.contains("SELECT id FROM intentions WHERE id = $intention_id"),
            "{method} anchor scope check must read only the requested Intention row"
        );
    } else {
        assert_eq!(
            prefix
                .matches("SELECT id FROM entities WHERE id = $entity_id")
                .count(),
            1,
            "{method} anchor scope check must contain exactly one Entity row lookup"
        );
        assert!(
            prefix.contains("SELECT id FROM entities WHERE id = $entity_id"),
            "{method} anchor scope check must read only the requested Entity row"
        );
    }
    let prefix_lower = prefix.to_ascii_lowercase();
    for forbidden in [
        "graph_table",
        "edges",
        " match ",
        " join ",
        " with ",
        "context_id",
    ] {
        assert!(
            !prefix_lower.contains(forbidden),
            "{method} anchor scope check must not smuggle Discovery traversal through {forbidden}"
        );
    }
}

fn assert_no_context_predicates(method: &str, source: &str) {
    let compact = compact_source(source).to_ascii_lowercase();
    let no_space = source
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    for forbidden in [
        "context_id =",
        "context_id=",
        "context_id in",
        "context_id!=",
        "context_id !=",
        "context_id <>",
        "context_id>",
        "context_id >",
        "context_id<",
        "context_id <",
        ".context_id ==",
        ".context_id==",
    ] {
        assert!(
            !compact.contains(forbidden),
            "{method} must rely on engine scope refusal, not app-side context predicate {forbidden}"
        );
    }
    for forbidden in [
        "context_id=",
        "context_idin",
        "context_id!=",
        "context_id<>",
        "context_id>",
        "context_id<",
        ".context_id==",
    ] {
        assert!(
            !no_space.contains(forbidden),
            "{method} must rely on engine scope refusal, not app-side context predicate {forbidden}"
        );
    }
}

fn compact_source(source: &str) -> String {
    source.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn discovery_sql_source(method: &str, source: &str) -> String {
    let code = code_mask_for_function_scan(source);
    let start = code.find("let sql = format!(");
    assert!(
        start.is_some(),
        "{method} must build the executed discovery query in `let sql = format!(...)`"
    );
    let start = start.unwrap();
    let after_sql = &code[start..];
    let end = after_sql.find("let result =");
    assert!(
        end.is_some(),
        "{method} must execute the discovery query after building `sql`"
    );
    let end = end.unwrap();
    source[start..start + end].to_string()
}

fn assert_discovery_result_return_shape(method: &str, source: &str) {
    let code = code_mask_for_function_scan(source);
    let start = code.find("let sql = format!(").unwrap();
    let after_sql = &code[start..];
    let end = after_sql.find("let result =").unwrap();
    let result_source = compact_source(&after_sql[end..]);
    let expected = if method.contains("by_entity_intentions") {
        "let result = self .db .execute_at_snapshot(&sql, &params, snapshot) .map_err(map_engine_err)?; result .rows .iter() .map(|row| decode_intention(&result.columns, row)) .collect()"
    } else {
        "let result = self .db .execute_at_snapshot(&sql, &params, snapshot) .map_err(map_engine_err)?; self.decode_discovery_decisions_at_read(&result.columns, &result.rows, snapshot)"
    };
    let expected = format!("{expected} }}");
    assert_eq!(
        result_source, expected,
        "{method} must return rows decoded directly from execute_at_snapshot(&sql, &params, snapshot)"
    );
}

fn assert_single_snapshot_reused(method: &str, code: &str) {
    let compact = compact_source(code);
    let snapshot_captures = compact
        .matches("let snapshot = self .db .snapshot();")
        .count()
        + compact
            .matches("let snapshot = self.db.snapshot();")
            .count();
    assert_eq!(
        snapshot_captures, 1,
        "{method} must capture exactly one snapshot for the whole Discovery read"
    );
    let snapshot_uses = compact
        .matches("snapshot) .map_err(map_engine_err)?;")
        .count()
        + compact
            .matches("snapshot).map_err(map_engine_err)?;")
            .count();
    assert_eq!(
        snapshot_uses, 2,
        "{method} must pass the same captured snapshot to both execute_at_snapshot calls"
    );
}

fn assert_discovery_query_shape(method: &str, source: &str) {
    let sql_source = discovery_sql_source(method, source);
    let compact = compact_source(&sql_source);
    if method.contains("by_intention") {
        for needle in [
            "WITH serving AS",
            "GRAPH_TABLE",
            "edges MATCH (i)<-[:SERVES]-(d)",
            "SELECT DISTINCT d.id, d.decision_type, d.description, d.reasoning, d.confidence, d.status, d.agent_id, d.context_id, d.properties, d.tags, d.created_at FROM decisions d",
            "WHERE d.id IN (SELECT d_id FROM serving)",
            "ORDER BY d.created_at DESC, d.id ASC",
        ] {
            assert!(
                compact.contains(needle),
                "{method} missing query shape: {needle}"
            );
        }
        assert_eq!(
            compact.matches("edges MATCH").count(),
            1,
            "{method} must use exactly one GRAPH_TABLE edge match"
        );
        assert_eq!(
            compact.to_ascii_lowercase().matches("edges").count(),
            1,
            "{method} must not add raw edges-table traversal beside GRAPH_TABLE"
        );
    } else if method.contains("by_entity_decisions") {
        for needle in [
            "WITH based AS",
            "about AS",
            "edges MATCH (e)<-[:BASED_ON]-(d)",
            "edges MATCH (e)<-[:ABOUT]-(d)",
            "SELECT DISTINCT d.id, d.decision_type, d.description, d.reasoning, d.confidence, d.status, d.agent_id, d.context_id, d.properties, d.tags, d.created_at FROM decisions d",
            "WHERE (d.id IN (SELECT d_id FROM based) OR d.id IN (SELECT d_id FROM about))",
            "ORDER BY d.created_at DESC, d.id ASC",
        ] {
            assert!(
                compact.contains(needle),
                "{method} missing query shape: {needle}"
            );
        }
        assert_eq!(
            compact.matches("edges MATCH").count(),
            2,
            "{method} must use exactly two GRAPH_TABLE edge matches"
        );
        assert_eq!(
            compact.to_ascii_lowercase().matches("edges").count(),
            2,
            "{method} must not add raw edges-table traversal beside GRAPH_TABLE"
        );
    } else if method.contains("by_entity_intentions") {
        for needle in [
            "WITH based AS",
            "about AS",
            "edges MATCH (e)<-[:BASED_ON]-(d)",
            "edges MATCH (e)<-[:ABOUT]-(d)",
            "SELECT DISTINCT i.id, i.description, i.status, i.origin, i.context_id, i.properties, i.tags, i.created_at FROM intentions i",
            "INNER JOIN edges s ON i.id = s.target_id",
            "s.edge_type = 'SERVES'",
            "s.source_id IN (SELECT d_id FROM based)",
            "s.source_id IN (SELECT d_id FROM about)",
            "s.source_id IN (SELECT id FROM decisions WHERE status IN",
            "ORDER BY i.created_at DESC, i.id ASC",
        ] {
            assert!(
                compact.contains(needle),
                "{method} missing query shape: {needle}"
            );
        }
        assert_eq!(
            compact.matches("edges MATCH").count(),
            2,
            "{method} must use GRAPH_TABLE only for the Entity BASED_ON/ABOUT walks"
        );
        assert_eq!(
            compact
                .matches("INNER JOIN edges s ON i.id = s.target_id")
                .count(),
            1,
            "{method} permits exactly one top-level SERVES join through edges"
        );
        assert_eq!(
            compact.to_ascii_lowercase().matches("edges").count(),
            3,
            "{method} must not add raw edges-table traversal beyond the allowed SERVES join"
        );
    }
}

fn assert_discovery_method_uses_engine_snapshot(method: &str, min_graph_tables: usize) {
    let source = store_method_source(method);
    let code = code_mask_for_function_scan(&source);
    assert!(
        !source.contains("\"execute_at_snapshot\""),
        "{method} must call execute_at_snapshot, not mention it in a string literal"
    );
    assert!(
        !source.contains("\"GRAPH_TABLE\"") && !source.contains("\"edges MATCH"),
        "{method} must put GRAPH_TABLE traversal in the executed SQL, not a decoy string literal"
    );
    assert!(
        source_lines(&code).any(|line| line.contains("self.db.snapshot()")),
        "{method} must capture one engine snapshot for the discovery read"
    );
    assert!(
        source_lines(&code).any(|line| line.contains(".execute_at_snapshot(")),
        "{method} must run discovery reads through execute_at_snapshot"
    );
    assert_eq!(
        code.matches(".execute_at_snapshot(").count(),
        2,
        "{method} must only execute the anchor scope check and the inspected Discovery query"
    );
    assert_single_snapshot_reused(method, &code);
    assert_eq!(
        code.matches("let sql = format!(").count(),
        1,
        "{method} must build exactly one executable Discovery SQL binding"
    );
    assert_eq!(
        code.matches("let sql =").count(),
        1,
        "{method} must not shadow the inspected Discovery SQL binding"
    );
    assert_eq!(
        code.matches("let result =").count(),
        1,
        "{method} must execute exactly one Discovery result query"
    );
    assert_method_source_lacks(
        method,
        &source,
        &[
            "self.db.execute(",
            "execute_in_tx",
            ".lock(",
            ".read(",
            ".write(",
            "Mutex",
            "RwLock",
            "OnceLock",
            "OnceCell",
            "LazyLock",
            "Condvar",
            "thread::",
            "sleep(",
            "Duration::",
            "Instant",
            "SystemTime",
            "std::time",
            "elapsed(",
            "spin_loop",
            "park(",
            "yield_now(",
            "list_decisions(",
            "edge_targets(",
            "decision_targets_entity(",
            "decisions_serving_intention(",
            "resource_owner",
            "authorized_contexts",
            "Database::open",
            "scoped_with_contexts",
            "CrossContextNotSupported",
            "context_id = $context_id",
            ".context_id ==",
            ".retain(",
        ],
    );
    assert_no_unapproved_store_helper_calls(
        method,
        &source,
        &[
            method,
            "decision_statuses_for_discovery",
            "intention_statuses_for_discovery",
            "sql_string_list",
            "map_engine_err",
            "decode_intention",
            "decode_discovery_decisions_at_read",
        ],
    );
    assert_no_candidate_work_before_result(method, &source);
    assert_anchor_scope_check_shape(method, &source);
    assert_no_context_predicates(method, &source);
    assert_discovery_query_shape(method, &source);
    assert_discovery_result_return_shape(method, &source);
    let graph_table_count = discovery_sql_source(method, &source)
        .matches("GRAPH_TABLE")
        .count();
    assert!(
        graph_table_count >= min_graph_tables,
        "{method} must use contextdb GRAPH_TABLE for discovery graph traversal"
    );
    assert_only_allowed_calls(
        method,
        &source,
        &[
            "as_uuid",
            "decode_discovery_decisions_at_read",
            "decode_intention",
            "decision_statuses_for_discovery",
            "Err",
            "execute_at_snapshot",
            "from",
            "intention_statuses_for_discovery",
            "iter",
            "is_empty",
            "is_valid",
            "collect",
            "map",
            "map_err",
            "new",
            "Ok",
            "Some",
            "snapshot",
            "sql_string_list",
            "to_string",
            "Uuid",
        ],
    );
}

fn assert_snapshot_hydration_helper_uses_snapshot(method: &str) {
    let source = store_method_source(method);
    let code = code_mask_for_function_scan(&source);
    let compact = compact_source(&code);
    assert!(
        compact.contains("snapshot: SnapshotId"),
        "{method} must accept the caller's pinned snapshot"
    );
    assert!(
        compact.contains(".execute_at_snapshot("),
        "{method} must hydrate through execute_at_snapshot"
    );
    assert!(
        compact.contains("snapshot) .map_err(map_engine_err)?")
            || compact.contains("snapshot).map_err(map_engine_err)?"),
        "{method} must pass the caller's pinned snapshot into execute_at_snapshot"
    );
    assert_method_source_lacks(
        method,
        &source,
        &[
            "self.db.execute(",
            "self.db.snapshot()",
            "execute_in_tx",
            ".lock(",
            ".read(",
            ".write(",
            "edge_targets(",
            "get_decision(",
            "get_intention(",
            "intentions_for_decision(",
            "basis_snapshot_ids_for_decision(",
            "basis_snapshots_for_decision(",
            "outcome_for_decision(",
            "precedents_for_decision(",
            "self.edge_targets",
            "self.get_decision",
            "self.get_intention",
            "self.intentions_for_decision",
            "self.basis_snapshot_ids_for_decision",
            "self.basis_snapshots_for_decision",
            "self.outcome_for_decision",
            "self.precedents_for_decision",
            "Self::edge_targets",
            "Self::get_decision",
            "Self::get_intention",
            "Self::intentions_for_decision",
            "Self::basis_snapshot_ids_for_decision",
            "Self::basis_snapshots_for_decision",
            "Self::outcome_for_decision",
            "Self::precedents_for_decision",
            "Store::edge_targets",
            "Store::get_decision",
            "Store::get_intention",
            "Store::intentions_for_decision",
            "Store::basis_snapshot_ids_for_decision",
            "Store::basis_snapshots_for_decision",
            "Store::outcome_for_decision",
            "Store::precedents_for_decision",
        ],
    );
    assert_only_allowed_calls(
        method,
        &source,
        &[
            "as_uuid",
            "decode_outcome",
            "execute_at_snapshot",
            "first",
            "from",
            "len",
            "map",
            "map_err",
            "Ok",
            "ordered_snapshot_ids",
            "push",
            "sort",
            "Some",
            "Text",
            "to_string",
            "transpose",
            "Uuid",
            "with_capacity",
        ],
    );
}

fn assert_discovery_decode_helper_is_only_row_hydration() {
    let source = store_method_source("decode_discovery_decisions_at_read");
    let code = code_mask_for_function_scan(&source);
    let compact = compact_source(&code);
    assert!(
        compact.contains("snapshot: SnapshotId"),
        "decode_discovery_decisions_at_read must receive the Discovery snapshot"
    );
    assert_method_source_lacks(
        "decode_discovery_decisions_at_read",
        &source,
        &[
            "GRAPH_TABLE",
            "self.db.execute(",
            "execute_in_tx",
            ".lock(",
            ".read(",
            ".write(",
            "Mutex",
            "RwLock",
            "OnceLock",
            "OnceCell",
            "LazyLock",
            "Condvar",
            "thread::",
            "sleep(",
            "Duration::",
            "CrossContextNotSupported",
            "authorized_contexts",
            "resource_owner",
            "Database::open",
            "scoped_with_contexts",
            ".context_id ==",
            ".retain(",
            "edge_targets(",
            "decision_targets_entity(",
            "decisions_serving_intention(",
            "list_decisions(",
        ],
    );
    for needle in [
        "intentions_for_decision_at_read(",
        "basis_snapshot_ids_for_decision_at_read(",
        "outcome_for_decision_at_read(",
        "precedents_for_decision_at_read(",
    ] {
        assert!(
            compact.contains(needle),
            "decode_discovery_decisions_at_read must hydrate Decision field through {needle}"
        );
    }
    assert_no_unapproved_store_helper_calls(
        "decode_discovery_decisions_at_read",
        &source,
        &[
            "decode_discovery_decisions_at_read",
            "decode_decision",
            "intentions_for_decision_at_read",
            "basis_snapshot_ids_for_decision_at_read",
            "outcome_for_decision_at_read",
            "precedents_for_decision_at_read",
        ],
    );
    assert_only_allowed_calls(
        "decode_discovery_decisions_at_read",
        &source,
        &[
            "basis_snapshot_ids_for_decision_at_read",
            "decode_decision",
            "intentions_for_decision_at_read",
            "len",
            "Ok",
            "outcome_for_decision_at_read",
            "precedents_for_decision_at_read",
            "push",
            "with_capacity",
        ],
    );
    for method in [
        "intentions_for_decision_at_read",
        "basis_snapshot_ids_for_decision_at_read",
        "outcome_for_decision_at_read",
        "precedents_for_decision_at_read",
    ] {
        assert_snapshot_hydration_helper_uses_snapshot(method);
    }
}

fn assert_discovery_source_contract() {
    assert_discovery_method_uses_engine_snapshot("by_intention", 1);
    for method in ["by_entity_decisions", "by_entity_intentions"] {
        let source = store_method_source(method);
        assert!(
            source.contains("BASED_ON") && source.contains("ABOUT"),
            "{method} must walk BASED_ON union ABOUT through the engine"
        );
        assert_discovery_method_uses_engine_snapshot(method, 2);
    }
    assert_discovery_decode_helper_is_only_row_hydration();
}

// An agent resuming a goal needs only the current active Decisions serving that goal.
#[test]
fn by_intention_returns_serving_decisions_active_only() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "client-a");
    let i = intention(&store, ctx.id, "keep references current");
    let side = intention(&store, ctx.id, "side goal");
    let d1 = untagged_decision(&store, ctx.id, "current path", vec![i.id], vec![]);
    let d2 = decision(&store, ctx.id, "old path", vec![i.id], vec![]);
    let d3 = decision(&store, ctx.id, "drifted path", vec![i.id], vec![]);
    let d_side = decision(&store, ctx.id, "other goal", vec![side.id], vec![]);
    supersede_with_side_replacement(&store, ctx.id, d2.id);
    invalidate_decision_for_test(&store, ctx.id, d3.id);

    let rows = ok!(
        store.by_intention(i.id, DiscoveryFilter::default()),
        "by intention"
    );

    assert_decision_ids(&rows, &[d1.id]);
    assert_eq!(rows[0].status, DecisionStatus::Active);
    assert!(!rows.iter().any(|row| row.id == d_side.id));
    assert_decision_rows_match_get(&store, &rows);
}

// An auditor can opt in to the full graveyard and still receive the original reasoning chain.
#[test]
fn by_intention_include_statuses_returns_filtered_set() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "audit");
    let i = intention(&store, ctx.id, "document framework history");
    let d1 = decision(&store, ctx.id, "active choice", vec![i.id], vec![]);
    let d2 = decision(&store, ctx.id, "superseded choice", vec![i.id], vec![]);
    let d3 = decision(&store, ctx.id, "invalidated choice", vec![i.id], vec![]);
    let d4 = decision(&store, ctx.id, "archived choice", vec![i.id], vec![]);
    supersede_with_side_replacement(&store, ctx.id, d2.id);
    invalidate_decision_for_test(&store, ctx.id, d3.id);
    archive_decision_for_test(&store, d4.id, "retired");

    let all = ok!(
        store.by_intention(
            i.id,
            decision_filter(vec![
                DecisionStatus::Active,
                DecisionStatus::Superseded,
                DecisionStatus::Invalidated,
                DecisionStatus::Archived,
            ]),
        ),
        "by intention all statuses"
    );
    assert_decision_ids(&all, &[d1.id, d2.id, d3.id, d4.id]);
    assert_decisions_sorted(&all);
    assert_decision_rows_match_get(&store, &all);
    for row in &all {
        assert!(!row.reasoning.is_empty());
        assert_eq!(
            row.reasoning[0].details,
            format!("details-{}", row.description)
        );
    }

    let superseded = ok!(
        store.by_intention(i.id, decision_filter(vec![DecisionStatus::Superseded])),
        "by intention superseded"
    );
    assert_decision_ids(&superseded, &[d2.id]);
    assert_decision_rows_match_get(&store, &superseded);

    let empty = ok!(
        store.by_intention(i.id, decision_filter(vec![])),
        "by intention empty status list"
    );
    assert!(empty.is_empty());
}

// A consultant scoped to one client must get an explicit refusal for another client's anchor.
#[test]
fn by_intention_context_isolation() {
    assert_contextdb_explicit_anchor_scope_violation();

    let (_tempdir, db_path, store) = fixture();
    let ctx_a = context(&store, "client-a");
    let ctx_b = context(&store, "client-b");
    drop(store);

    let store_a = scoped(&db_path, ctx_a.id);
    let ia = intention(&store_a, ctx_a.id, "client a goal");
    let da = decision(&store_a, ctx_a.id, "client a decision", vec![ia.id], vec![]);
    let ia_other = intention(&store_a, ctx_a.id, "client a other goal");
    let da_wrong_edge = decision(
        &store_a,
        ctx_a.id,
        "wrong edge context",
        vec![ia_other.id],
        vec![],
    );
    drop(store_a);

    let store_b = scoped(&db_path, ctx_b.id);
    let ib = intention(&store_b, ctx_b.id, "client b goal");
    let db = decision(&store_b, ctx_b.id, "client b decision", vec![ib.id], vec![]);
    drop(store_b);
    insert_raw_edges(
        &db_path,
        ctx_b.id,
        "SERVES",
        &[(da_wrong_edge.id.as_uuid(), ia.id.as_uuid())],
    );
    let store_b = scoped(&db_path, ctx_b.id);
    let store_a = scoped(&db_path, ctx_a.id);

    let denied = store_b.by_intention(ia.id, DiscoveryFilter::default());
    assert!(matches!(denied, Err(CgError::CrossContextNotSupported)));
    assert!(
        denied
            .unwrap_err()
            .to_string()
            .contains("cross-client retrospective")
    );

    let rows = ok!(
        store_a.by_intention(ia.id, DiscoveryFilter::default()),
        "same-context by intention"
    );
    assert_decision_ids(&rows, &[da.id]);
    assert!(rows.iter().all(|row| row.context_id == ctx_a.id));
    assert!(!rows.iter().any(|row| row.id == db.id));
    assert!(!rows.iter().any(|row| row.id == da_wrong_edge.id));

    drop(store_a);
    let store_a = scoped(&db_path, ctx_a.id);
    let rows = ok!(
        store_a.by_intention(ia.id, DiscoveryFilter::default()),
        "same-context by intention after reopen"
    );
    assert_decision_ids(&rows, &[da.id]);
    assert_discovery_source_contract();
}

// A reviewer asking what touches an Entity needs BASED_ON and ABOUT Decisions, with duplicates collapsed.
#[test]
fn by_entity_returns_connected_decisions_active_only() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = context(&store, "review");
    let e = entity(&store, ctx.id, "abstraction-layer");
    let i = intention(&store, ctx.id, "review architecture");
    let d1 = decision(&store, ctx.id, "based on entity", vec![i.id], vec![e.id]);
    let d2 = untagged_decision(&store, ctx.id, "about entity", vec![i.id], vec![]);
    let d3 = decision(&store, ctx.id, "based and about", vec![i.id], vec![e.id]);
    let d4 = decision(&store, ctx.id, "retired touch", vec![i.id], vec![e.id]);
    let d5 = decision(&store, ctx.id, "unrelated", vec![i.id], vec![]);
    supersede_with_side_replacement(&store, ctx.id, d4.id);
    drop(store);
    insert_about_edges(&db_path, ctx.id, &[(d2.id, e.id), (d3.id, e.id)]);
    let store = reopen(&db_path);

    let rows = ok!(
        store.by_entity_decisions(e.id, DiscoveryFilter::default()),
        "by entity decisions"
    );

    assert_decision_ids(&rows, &[d1.id, d2.id, d3.id]);
    assert_decision_order(&rows, &[d3.id, d2.id, d1.id]);
    assert_eq!(rows.iter().filter(|row| row.id == d3.id).count(), 1);
    assert!(!rows.iter().any(|row| row.id == d4.id || row.id == d5.id));
    assert_decision_rows_match_get(&store, &rows);
    for row in &rows {
        assert!(!row.reasoning.is_empty());
        assert_eq!(
            row.reasoning[0].details,
            format!("details-{}", row.description)
        );
    }
}

// A returning consultant needs the in-play Intentions around an Entity, not duplicated or retired goals.
#[test]
fn by_entity_returns_connected_intentions() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "client-resume");
    let e = entity(&store, ctx.id, "payments-service");
    let i_active = untagged_intention(&store, ctx.id, "active goal");
    let i_paused = intention_with_status(&store, ctx.id, "paused goal", IntentionStatus::Paused);
    let i_archived =
        intention_with_status(&store, ctx.id, "archived goal", IntentionStatus::Archived);
    let i_superseded = intention_with_status(
        &store,
        ctx.id,
        "superseded goal",
        IntentionStatus::Superseded,
    );
    let i_unreachable = intention(&store, ctx.id, "behind retired decision");
    untagged_decision(
        &store,
        ctx.id,
        "active route",
        vec![i_active.id],
        vec![e.id],
    );
    decision(
        &store,
        ctx.id,
        "active route duplicate",
        vec![i_active.id],
        vec![e.id],
    );
    decision(
        &store,
        ctx.id,
        "paused route",
        vec![i_paused.id],
        vec![e.id],
    );
    decision(
        &store,
        ctx.id,
        "archived route",
        vec![i_archived.id],
        vec![e.id],
    );
    decision(
        &store,
        ctx.id,
        "superseded route",
        vec![i_superseded.id],
        vec![e.id],
    );
    let hidden = decision(
        &store,
        ctx.id,
        "retired intermediate",
        vec![i_unreachable.id],
        vec![e.id],
    );
    supersede_with_side_replacement(&store, ctx.id, hidden.id);

    let rows = ok!(
        store.by_entity_intentions(e.id, DiscoveryFilter::default()),
        "by entity intentions"
    );

    assert_intention_ids(&rows, &[i_active.id, i_paused.id]);
    assert_intention_order(&rows, &[i_paused.id, i_active.id]);
    assert_intention_rows_match_get(&store, &rows);
    assert_eq!(
        rows.iter()
            .find(|row| row.id == i_paused.id)
            .expect("paused")
            .status,
        IntentionStatus::Paused
    );
}

// A retrospective caller can widen the Decision side without widening the Intention graveyard.
#[test]
fn by_entity_include_statuses_returns_filtered_set() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "retrospective");
    let e = entity(&store, ctx.id, "gateway");
    let i_active = intention(&store, ctx.id, "active goal");
    let i_paused = intention_with_status(&store, ctx.id, "paused goal", IntentionStatus::Paused);
    let i_superseded =
        intention_with_status(&store, ctx.id, "old goal", IntentionStatus::Superseded);
    let i_behind = intention(&store, ctx.id, "behind superseded decision");
    let i_behind_archived = intention(&store, ctx.id, "behind archived decision");
    let i_behind_invalidated = intention(&store, ctx.id, "behind invalidated decision");
    let d_active = decision(
        &store,
        ctx.id,
        "active touch",
        vec![i_active.id],
        vec![e.id],
    );
    let d_paused = decision(
        &store,
        ctx.id,
        "paused touch",
        vec![i_paused.id],
        vec![e.id],
    );
    let d_superseded_intention = decision(
        &store,
        ctx.id,
        "touches superseded intention",
        vec![i_superseded.id],
        vec![e.id],
    );
    let d_old = decision(&store, ctx.id, "old touch", vec![i_behind.id], vec![e.id]);
    let d_archived = decision(
        &store,
        ctx.id,
        "archived touch",
        vec![i_behind_archived.id],
        vec![e.id],
    );
    let d_invalidated = decision(
        &store,
        ctx.id,
        "invalidated touch",
        vec![i_behind_invalidated.id],
        vec![e.id],
    );
    supersede_with_side_replacement(&store, ctx.id, d_old.id);
    archive_decision_for_test(&store, d_archived.id, "retired");
    invalidate_decision_for_test(&store, ctx.id, d_invalidated.id);

    let decisions = ok!(
        store.by_entity_decisions(
            e.id,
            decision_filter(vec![
                DecisionStatus::Active,
                DecisionStatus::Superseded,
                DecisionStatus::Archived,
                DecisionStatus::Invalidated,
            ]),
        ),
        "by entity decisions widened"
    );
    assert_decision_ids(
        &decisions,
        &[
            d_active.id,
            d_paused.id,
            d_superseded_intention.id,
            d_old.id,
            d_archived.id,
            d_invalidated.id,
        ],
    );
    assert_decision_order(
        &decisions,
        &[
            d_invalidated.id,
            d_archived.id,
            d_old.id,
            d_superseded_intention.id,
            d_paused.id,
            d_active.id,
        ],
    );
    assert!(
        decisions
            .iter()
            .any(|row| row.status == DecisionStatus::Superseded)
    );
    assert!(
        decisions
            .iter()
            .any(|row| row.status == DecisionStatus::Archived)
    );
    assert!(
        decisions
            .iter()
            .any(|row| row.status == DecisionStatus::Invalidated)
    );
    assert_decision_rows_match_get(&store, &decisions);

    let intentions = ok!(
        store.by_entity_intentions(
            e.id,
            decision_filter(vec![
                DecisionStatus::Active,
                DecisionStatus::Superseded,
                DecisionStatus::Archived,
                DecisionStatus::Invalidated,
            ]),
        ),
        "by entity intentions widened decision side"
    );
    assert_intention_ids(
        &intentions,
        &[
            i_active.id,
            i_paused.id,
            i_behind.id,
            i_behind_archived.id,
            i_behind_invalidated.id,
        ],
    );
    assert_intention_order(
        &intentions,
        &[
            i_behind_invalidated.id,
            i_behind_archived.id,
            i_behind.id,
            i_paused.id,
            i_active.id,
        ],
    );
    assert_intention_rows_match_get(&store, &intentions);
    assert!(!intentions.iter().any(|row| row.id == i_superseded.id));
}

// A client-scoped Entity read refuses another client's anchor on both typed result surfaces.
#[test]
fn by_entity_context_isolation() {
    let (_tempdir, db_path, store) = fixture();
    let ctx_a = context(&store, "client-a");
    let ctx_b = context(&store, "client-b");
    drop(store);

    let store_a = scoped(&db_path, ctx_a.id);
    let ea = entity(&store_a, ctx_a.id, "client-a-service");
    let ia = intention(&store_a, ctx_a.id, "client a goal");
    let da = decision(
        &store_a,
        ctx_a.id,
        "client a decision",
        vec![ia.id],
        vec![ea.id],
    );
    let ia_wrong_edge = intention(&store_a, ctx_a.id, "client a wrong edge goal");
    let da_wrong_edge = decision(
        &store_a,
        ctx_a.id,
        "wrong edge context",
        vec![ia_wrong_edge.id],
        vec![],
    );
    drop(store_a);

    let store_b = scoped(&db_path, ctx_b.id);
    let eb = entity(&store_b, ctx_b.id, "client-b-service");
    let ib = intention(&store_b, ctx_b.id, "client b goal");
    decision(
        &store_b,
        ctx_b.id,
        "client b decision",
        vec![ib.id],
        vec![eb.id],
    );
    drop(store_b);
    insert_raw_edges(
        &db_path,
        ctx_b.id,
        "BASED_ON",
        &[(da_wrong_edge.id.as_uuid(), ea.id.as_uuid())],
    );
    insert_raw_edges(
        &db_path,
        ctx_b.id,
        "ABOUT",
        &[(da_wrong_edge.id.as_uuid(), ea.id.as_uuid())],
    );
    let store_b = scoped(&db_path, ctx_b.id);
    let store_a = scoped(&db_path, ctx_a.id);

    let widened = decision_filter(vec![
        DecisionStatus::Active,
        DecisionStatus::Superseded,
        DecisionStatus::Archived,
        DecisionStatus::Invalidated,
    ]);
    assert!(matches!(
        store_b.by_entity_decisions(ea.id, widened.clone()),
        Err(CgError::CrossContextNotSupported)
    ));
    assert!(matches!(
        store_b.by_entity_intentions(ea.id, widened),
        Err(CgError::CrossContextNotSupported)
    ));

    let decisions = ok!(
        store_a.by_entity_decisions(ea.id, DiscoveryFilter::default()),
        "same-context decisions"
    );
    assert_decision_ids(&decisions, &[da.id]);
    assert!(decisions.iter().all(|row| row.context_id == ctx_a.id));
    assert!(!decisions.iter().any(|row| row.id == da_wrong_edge.id));

    let intentions = ok!(
        store_a.by_entity_intentions(ea.id, DiscoveryFilter::default()),
        "same-context intentions"
    );
    assert_intention_ids(&intentions, &[ia.id]);
    assert!(intentions.iter().all(|row| row.context_id == ctx_a.id));
    assert!(!intentions.iter().any(|row| row.id == ia_wrong_edge.id));
    assert_discovery_source_contract();
}

// The default filter protects prompt injection from stale Decision rows while keeping paused goals in play.
#[test]
fn discovery_default_active_filter_excludes_invalidated_superseded_archived() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "default-wedge");
    let e = entity(&store, ctx.id, "auth-module");
    let intentions = [
        intention_with_status(&store, ctx.id, "active goal", IntentionStatus::Active),
        intention_with_status(&store, ctx.id, "paused goal", IntentionStatus::Paused),
        intention_with_status(
            &store,
            ctx.id,
            "superseded goal",
            IntentionStatus::Superseded,
        ),
        intention_with_status(&store, ctx.id, "archived goal", IntentionStatus::Archived),
    ];
    let mut active_for_active = None;
    let mut active_for_paused = None;
    let mut active_decision_ids = Vec::new();
    for status in DecisionStatus::ALL {
        for intention in &intentions {
            let row = decision(
                &store,
                ctx.id,
                &format!("cell-{status:?}-{:?}", intention.status),
                vec![intention.id],
                vec![e.id],
            );
            match status {
                DecisionStatus::Active => {
                    active_decision_ids.push(row.id);
                    if intention.status == IntentionStatus::Active {
                        active_for_active = Some(row.id);
                    } else if intention.status == IntentionStatus::Paused {
                        active_for_paused = Some(row.id);
                    }
                }
                DecisionStatus::Superseded => {
                    supersede_with_side_replacement(&store, ctx.id, row.id);
                }
                DecisionStatus::Archived => {
                    archive_decision_for_test(&store, row.id, "graveyard");
                }
                DecisionStatus::Invalidated => {
                    invalidate_decision_for_test(&store, ctx.id, row.id);
                }
            }
        }
    }

    let by_intention = ok!(
        store.by_intention(intentions[0].id, DiscoveryFilter::default()),
        "by intention default"
    );
    assert!(
        by_intention
            .iter()
            .all(|row| row.status == DecisionStatus::Active)
    );
    assert_decision_ids(
        &by_intention,
        &[active_for_active.expect("active active decision")],
    );
    assert_decision_rows_match_get(&store, &by_intention);

    let decisions = ok!(
        store.by_entity_decisions(e.id, DiscoveryFilter::default()),
        "by entity decisions default"
    );
    assert!(
        decisions
            .iter()
            .all(|row| row.status == DecisionStatus::Active)
    );
    assert_decision_ids(&decisions, &active_decision_ids);
    assert_decision_rows_match_get(&store, &decisions);

    let result_intentions = ok!(
        store.by_entity_intentions(e.id, DiscoveryFilter::default()),
        "by entity intentions default"
    );
    assert_eq!(
        ok!(
            store.get_decision(active_for_paused.expect("active paused decision")),
            "get active paused decision"
        )
        .expect("active paused decision")
        .status,
        DecisionStatus::Active
    );
    assert!(result_intentions.iter().all(|row| matches!(
        row.status,
        IntentionStatus::Active | IntentionStatus::Paused
    )));
    assert_intention_ids(&result_intentions, &[intentions[0].id, intentions[1].id]);
    assert_intention_rows_match_get(&store, &result_intentions);
}

// An informational Decision that is only ABOUT an Entity is still part of what we know about it.
#[test]
fn by_entity_about_only_decision_surfaces() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = context(&store, "about-only");
    let e = entity(&store, ctx.id, "abstraction-layer");
    let i_about = untagged_intention(&store, ctx.id, "review abstraction about");
    let i_based = intention(&store, ctx.id, "review abstraction basis");
    let d_about = untagged_decision(&store, ctx.id, "add monitoring", vec![i_about.id], vec![]);
    let d_based = decision(
        &store,
        ctx.id,
        "fault isolation",
        vec![i_based.id],
        vec![e.id],
    );
    drop(store);
    insert_about_edges(&db_path, ctx.id, &[(d_about.id, e.id)]);
    let store = reopen(&db_path);

    let rows = ok!(
        store.by_entity_decisions(e.id, DiscoveryFilter::default()),
        "by entity decisions"
    );

    assert_decision_ids(&rows, &[d_about.id, d_based.id]);
    assert_decision_order(&rows, &[d_based.id, d_about.id]);
    assert_decision_rows_match_get(&store, &rows);
    let about = rows.iter().find(|row| row.id == d_about.id).expect("about");
    assert!(about.based_on_snapshots.is_empty());
    let based = rows.iter().find(|row| row.id == d_based.id).expect("based");
    assert!(!based.based_on_snapshots.is_empty());

    let intentions = ok!(
        store.by_entity_intentions(e.id, DiscoveryFilter::default()),
        "by entity intentions"
    );
    assert_intention_ids(&intentions, &[i_about.id, i_based.id]);
    assert_intention_order(&intentions, &[i_based.id, i_about.id]);
    assert_intention_rows_match_get(&store, &intentions);
    assert_discovery_source_contract();
}

// Multiple Decisions around one Entity can serve the same goal, but the goal should appear once.
#[test]
fn by_entity_result_type_intention_deduplicates_across_serves_edges() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "dedup-intention");
    let e = entity(&store, ctx.id, "cache");
    let i = intention(&store, ctx.id, "optimize cache");
    decision(&store, ctx.id, "cache ttl", vec![i.id], vec![e.id]);
    decision(&store, ctx.id, "cache key", vec![i.id], vec![e.id]);
    decision(&store, ctx.id, "cache eviction", vec![i.id], vec![e.id]);

    let rows = ok!(
        store.by_entity_intentions(e.id, DiscoveryFilter::default()),
        "by entity intentions"
    );

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, i.id);
}

// Stable ordering keeps repeated prompt construction byte-stable for the same graph state.
#[test]
fn discovery_ordering_created_at_desc_id_asc_deterministic() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = context(&store, "ordering");
    let i = intention(&store, ctx.id, "stable prompt order");
    let d1 = decision(&store, ctx.id, "oldest", vec![i.id], vec![]);
    let d2 = decision(&store, ctx.id, "middle", vec![i.id], vec![]);
    let d3 = decision(&store, ctx.id, "newest", vec![i.id], vec![]);

    let first = ok!(
        store.by_intention(i.id, DiscoveryFilter::default()),
        "by intention first"
    );
    let second = ok!(
        store.by_intention(i.id, DiscoveryFilter::default()),
        "by intention second"
    );
    assert_eq!(first, second);
    assert_eq!(
        first.iter().map(|row| row.id).collect::<Vec<_>>(),
        vec![d3.id, d2.id, d1.id]
    );
    drop(store);

    let store = reopen(&db_path);
    let reopened = ok!(
        store.by_intention(i.id, DiscoveryFilter::default()),
        "by intention reopened"
    );
    assert_eq!(first, reopened);

    supersede_with_side_replacement(&store, ctx.id, d2.id);
    let widened = decision_filter(vec![DecisionStatus::Active, DecisionStatus::Superseded]);
    let graveyard_first = ok!(
        store.by_intention(i.id, widened.clone()),
        "by intention graveyard first"
    );
    let graveyard_second = ok!(
        store.by_intention(i.id, widened.clone()),
        "by intention graveyard second"
    );
    assert_eq!(graveyard_first, graveyard_second);
    assert_decisions_sorted(&graveyard_first);
    drop(store);

    let store = reopen(&db_path);
    let graveyard_reopened = ok!(
        store.by_intention(i.id, widened),
        "by intention graveyard reopened"
    );
    assert_eq!(graveyard_first, graveyard_reopened);

    let i_tie = intention(&store, ctx.id, "same tx tie order");
    let low = DecisionId::from(uuid::Uuid::from_u128(
        0x0000_0000_0000_0000_0000_0000_0000_0011,
    ));
    let mid = DecisionId::from(uuid::Uuid::from_u128(
        0x0000_0000_0000_0000_0000_0000_0000_0012,
    ));
    let high = DecisionId::from(uuid::Uuid::from_u128(
        0x0000_0000_0000_0000_0000_0000_0000_0013,
    ));
    drop(store);
    let tie_tx = current_tx(&db_path);
    insert_raw_decision(&db_path, ctx.id, high, "tie high", tie_tx);
    insert_raw_decision(&db_path, ctx.id, low, "tie low", tie_tx);
    insert_raw_decision(&db_path, ctx.id, mid, "tie mid", tie_tx);
    insert_raw_edges(
        &db_path,
        ctx.id,
        "SERVES",
        &[
            (high.as_uuid(), i_tie.id.as_uuid()),
            (low.as_uuid(), i_tie.id.as_uuid()),
            (mid.as_uuid(), i_tie.id.as_uuid()),
        ],
    );
    let store = reopen(&db_path);
    let tied = ok!(
        store.by_intention(i_tie.id, DiscoveryFilter::default()),
        "by intention same tx"
    );
    assert_decision_order(&tied, &[low, mid, high]);
    assert!(tied.iter().all(|row| row.created_at == tie_tx));

    let e_tie = entity(&store, ctx.id, "same tx entity tie");
    let decision_low = DecisionId::from(uuid::Uuid::from_u128(
        0x0000_0000_0000_0000_0000_0000_0000_0021,
    ));
    let decision_mid = DecisionId::from(uuid::Uuid::from_u128(
        0x0000_0000_0000_0000_0000_0000_0000_0022,
    ));
    let decision_high = DecisionId::from(uuid::Uuid::from_u128(
        0x0000_0000_0000_0000_0000_0000_0000_0023,
    ));
    let intention_low = IntentionId::from(uuid::Uuid::from_u128(
        0x0000_0000_0000_0000_0000_0000_0000_0031,
    ));
    let intention_mid = IntentionId::from(uuid::Uuid::from_u128(
        0x0000_0000_0000_0000_0000_0000_0000_0032,
    ));
    let intention_high = IntentionId::from(uuid::Uuid::from_u128(
        0x0000_0000_0000_0000_0000_0000_0000_0033,
    ));
    drop(store);
    let entity_tie_tx = current_tx(&db_path);
    insert_raw_intention(
        &db_path,
        ctx.id,
        intention_high,
        "entity tie high",
        entity_tie_tx,
    );
    insert_raw_intention(
        &db_path,
        ctx.id,
        intention_low,
        "entity tie low",
        entity_tie_tx,
    );
    insert_raw_intention(
        &db_path,
        ctx.id,
        intention_mid,
        "entity tie mid",
        entity_tie_tx,
    );
    insert_raw_decision(
        &db_path,
        ctx.id,
        decision_high,
        "entity tie high",
        entity_tie_tx,
    );
    insert_raw_decision(
        &db_path,
        ctx.id,
        decision_low,
        "entity tie low",
        entity_tie_tx,
    );
    insert_raw_decision(
        &db_path,
        ctx.id,
        decision_mid,
        "entity tie mid",
        entity_tie_tx,
    );
    insert_raw_edges(
        &db_path,
        ctx.id,
        "BASED_ON",
        &[
            (decision_high.as_uuid(), e_tie.id.as_uuid()),
            (decision_low.as_uuid(), e_tie.id.as_uuid()),
            (decision_mid.as_uuid(), e_tie.id.as_uuid()),
        ],
    );
    insert_raw_edges(
        &db_path,
        ctx.id,
        "SERVES",
        &[
            (decision_high.as_uuid(), intention_high.as_uuid()),
            (decision_low.as_uuid(), intention_low.as_uuid()),
            (decision_mid.as_uuid(), intention_mid.as_uuid()),
        ],
    );
    let store = reopen(&db_path);
    let entity_decisions_first = ok!(
        store.by_entity_decisions(e_tie.id, DiscoveryFilter::default()),
        "by entity decisions same tx first"
    );
    let entity_decisions_second = ok!(
        store.by_entity_decisions(e_tie.id, DiscoveryFilter::default()),
        "by entity decisions same tx second"
    );
    assert_eq!(entity_decisions_first, entity_decisions_second);
    assert_decision_order(
        &entity_decisions_first,
        &[decision_low, decision_mid, decision_high],
    );
    assert!(
        entity_decisions_first
            .iter()
            .all(|row| row.created_at == entity_tie_tx)
    );

    let entity_intentions_first = ok!(
        store.by_entity_intentions(e_tie.id, DiscoveryFilter::default()),
        "by entity intentions same tx first"
    );
    let entity_intentions_second = ok!(
        store.by_entity_intentions(e_tie.id, DiscoveryFilter::default()),
        "by entity intentions same tx second"
    );
    assert_eq!(entity_intentions_first, entity_intentions_second);
    assert_intention_order(
        &entity_intentions_first,
        &[intention_low, intention_mid, intention_high],
    );
    assert!(
        entity_intentions_first
            .iter()
            .all(|row| row.created_at == entity_tie_tx)
    );
    drop(store);

    let store = reopen(&db_path);
    let entity_decisions_reopened = ok!(
        store.by_entity_decisions(e_tie.id, DiscoveryFilter::default()),
        "by entity decisions same tx reopened"
    );
    assert_eq!(entity_decisions_first, entity_decisions_reopened);
    let entity_intentions_reopened = ok!(
        store.by_entity_intentions(e_tie.id, DiscoveryFilter::default()),
        "by entity intentions same tx reopened"
    );
    assert_eq!(entity_intentions_first, entity_intentions_reopened);
}

// Empty is a typed answer when an existing anchor has no live connected rows.
#[test]
fn discovery_empty_result_is_typed_information_not_error() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "empty");
    let i_unserved = intention(&store, ctx.id, "currently unserved");
    let e_untouched = entity(&store, ctx.id, "untouched");
    let i_served_by_grave = intention(&store, ctx.id, "served only by graveyard");
    let old = decision(
        &store,
        ctx.id,
        "old service",
        vec![i_served_by_grave.id],
        vec![],
    );
    supersede_with_side_replacement(&store, ctx.id, old.id);

    assert!(
        ok!(
            store.by_intention(i_unserved.id, DiscoveryFilter::default()),
            "unserved intention"
        )
        .is_empty()
    );
    assert!(
        ok!(
            store.by_intention(i_served_by_grave.id, DiscoveryFilter::default()),
            "grave-only intention"
        )
        .is_empty()
    );
    assert!(
        ok!(
            store.by_entity_decisions(e_untouched.id, DiscoveryFilter::default()),
            "untouched entity decisions"
        )
        .is_empty()
    );
    assert!(
        ok!(
            store.by_entity_intentions(e_untouched.id, DiscoveryFilter::default()),
            "untouched entity intentions"
        )
        .is_empty()
    );
}

// Narrow basis fields affect invalidation, not whether the Decision touches the Entity.
#[test]
fn by_entity_basis_fields_narrow_basis_does_not_narrow_reachability() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "basis-fields");
    let e = entity_with_properties(
        &store,
        ctx.id,
        "instance",
        BTreeMap::from([
            ("region".to_string(), json!("us-east-1")),
            ("instance_type".to_string(), json!("m7g.large")),
        ]),
    );
    let i = intention(&store, ctx.id, "review infrastructure");
    let d_narrow = decision_with_basis_fields(
        &store,
        ctx.id,
        "region-specific policy",
        vec![i.id],
        vec![e.id],
        Some(HashMap::from([(e.id, vec!["region".to_string()])])),
    );
    let d_whole = decision_with_basis_fields(
        &store,
        ctx.id,
        "whole-instance policy",
        vec![i.id],
        vec![e.id],
        None,
    );

    let rows = ok!(
        store.by_entity_decisions(e.id, DiscoveryFilter::default()),
        "by entity decisions"
    );

    assert_decision_ids(&rows, &[d_narrow.id, d_whole.id]);
}

// Paused Intentions remain visible by default, and archived Intentions require explicit opt-in.
#[test]
fn by_entity_result_type_intention_paused_surfaces_archived_does_not() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "intention-axis");
    let e = entity(&store, ctx.id, "seasonal-service");
    let i_active = intention(&store, ctx.id, "active goal");
    let i_paused = intention_with_status(&store, ctx.id, "paused goal", IntentionStatus::Paused);
    let i_archived =
        intention_with_status(&store, ctx.id, "archived goal", IntentionStatus::Archived);
    let i_superseded = intention_with_status(
        &store,
        ctx.id,
        "superseded goal",
        IntentionStatus::Superseded,
    );
    let i_behind = intention(&store, ctx.id, "behind retired decision");
    decision(&store, ctx.id, "active", vec![i_active.id], vec![e.id]);
    decision(&store, ctx.id, "paused", vec![i_paused.id], vec![e.id]);
    decision(&store, ctx.id, "archived", vec![i_archived.id], vec![e.id]);
    decision(
        &store,
        ctx.id,
        "superseded",
        vec![i_superseded.id],
        vec![e.id],
    );
    let old = decision(&store, ctx.id, "old active", vec![i_behind.id], vec![e.id]);
    supersede_with_side_replacement(&store, ctx.id, old.id);

    let default = ok!(
        store.by_entity_intentions(e.id, DiscoveryFilter::default()),
        "default"
    );
    assert_intention_ids(&default, &[i_active.id, i_paused.id]);
    assert_intention_order(&default, &[i_paused.id, i_active.id]);
    assert_intention_rows_match_get(&store, &default);

    let decision_widened = ok!(
        store.by_entity_intentions(
            e.id,
            discovery_filter(
                Some(vec![DecisionStatus::Active, DecisionStatus::Superseded]),
                None,
            ),
        ),
        "decision widened"
    );
    assert_intention_ids(&decision_widened, &[i_active.id, i_paused.id, i_behind.id]);
    assert_intention_order(&decision_widened, &[i_behind.id, i_paused.id, i_active.id]);
    assert_intention_rows_match_get(&store, &decision_widened);

    let intention_widened = ok!(
        store.by_entity_intentions(
            e.id,
            discovery_filter(
                None,
                Some(vec![
                    IntentionStatus::Active,
                    IntentionStatus::Paused,
                    IntentionStatus::Archived,
                    IntentionStatus::Superseded,
                ]),
            ),
        ),
        "intention widened"
    );
    assert_intention_ids(
        &intention_widened,
        &[i_active.id, i_paused.id, i_archived.id, i_superseded.id],
    );
    assert_intention_order(
        &intention_widened,
        &[i_superseded.id, i_archived.id, i_paused.id, i_active.id],
    );
    assert_intention_rows_match_get(&store, &intention_widened);

    let both_widened = ok!(
        store.by_entity_intentions(
            e.id,
            discovery_filter(
                Some(vec![DecisionStatus::Active, DecisionStatus::Superseded]),
                Some(vec![
                    IntentionStatus::Active,
                    IntentionStatus::Paused,
                    IntentionStatus::Archived,
                    IntentionStatus::Superseded,
                ]),
            ),
        ),
        "both widened"
    );
    assert_intention_ids(
        &both_widened,
        &[
            i_active.id,
            i_paused.id,
            i_archived.id,
            i_superseded.id,
            i_behind.id,
        ],
    );
    assert_intention_order(
        &both_widened,
        &[
            i_behind.id,
            i_superseded.id,
            i_archived.id,
            i_paused.id,
            i_active.id,
        ],
    );
    assert_intention_rows_match_get(&store, &both_widened);

    let default_again = ok!(
        store.by_entity_intentions(e.id, DiscoveryFilter::default()),
        "default again"
    );
    assert_intention_ids(&default_again, &[i_active.id, i_paused.id]);
    assert_intention_order(&default_again, &[i_paused.id, i_active.id]);
}

// Superseded opt-in is the graveyard for this Entity, not a chain walk to another anchor.
#[test]
fn by_entity_supersedes_chain_does_not_cross_anchor() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "supersedes-boundary");
    let e_old = entity(&store, ctx.id, "old-service");
    let e_new = entity(&store, ctx.id, "new-service");
    let i = intention(&store, ctx.id, "move service");
    let d_old = decision(&store, ctx.id, "old basis", vec![i.id], vec![e_old.id]);
    let d_new = ok!(
        store.supersede_decision(
            d_old.id,
            decision_input(ctx.id, "new basis", vec![i.id], vec![e_new.id]),
        ),
        "supersede decision"
    );
    assert_eq!(
        ok!(store.get_decision(d_old.id), "get old")
            .expect("old")
            .status,
        DecisionStatus::Superseded
    );

    let default = ok!(
        store.by_entity_decisions(e_old.id, DiscoveryFilter::default()),
        "default old entity"
    );
    assert!(default.is_empty());

    let widened = ok!(
        store.by_entity_decisions(e_old.id, decision_filter(vec![DecisionStatus::Superseded])),
        "widened old entity"
    );
    assert_decision_ids(&widened, &[d_old.id]);
    assert!(!widened.iter().any(|row| row.id == d_new.id));
}

// A reader racing a supersede sees either the old active set or the new active set, never both.
#[test]
fn discovery_pull_side_returns_consistent_view_under_concurrent_writer() {
    assert_contextdb_pinned_discovery_graph_composition();

    let (_tempdir, db_path, store) = fixture();
    let ctx = context(&store, "concurrent");
    let ctx_id = ctx.id;
    let e = entity(&store, ctx_id, "concurrent service");
    let entity_id = e.id;
    let i = intention(&store, ctx_id, "serve concurrent reader");
    let main_intention_id = i.id;
    let stable = decision(
        &store,
        ctx_id,
        "stable one",
        vec![main_intention_id],
        vec![entity_id],
    );
    let mut replaceable_pairs = Vec::new();
    for n in 0..32 {
        let old_intention = intention(&store, ctx_id, &format!("old lineage {n:02}"));
        let new_intention = intention(&store, ctx_id, &format!("new lineage {n:02}"));
        let row = decision(
            &store,
            ctx_id,
            &format!("replaceable {n:02}"),
            vec![main_intention_id, old_intention.id],
            vec![entity_id],
        );
        replaceable_pairs.push((row.id, old_intention.id, new_intention.id));
    }
    let writer_pairs = replaceable_pairs.clone();
    drop(store);

    let barrier = Arc::new(Barrier::new(2));
    let writer_started = Arc::new(AtomicBool::new(false));
    let writer_done = Arc::new(AtomicBool::new(false));
    let reader_in_call = Arc::new(AtomicBool::new(false));
    let reader_store = reopen(&db_path);
    let writer_store = reopen(&db_path);
    let reader_barrier = Arc::clone(&barrier);
    let writer_barrier = Arc::clone(&barrier);
    let reader_started = Arc::clone(&writer_started);
    let reader_done = Arc::clone(&writer_done);
    let writer_started_flag = Arc::clone(&writer_started);
    let writer_done_flag = Arc::clone(&writer_done);
    let reader_in_call_flag = Arc::clone(&reader_in_call);
    let writer_observed_reader = Arc::clone(&reader_in_call);

    #[derive(Debug)]
    struct ObservedRead {
        by_intention_ids: BTreeSet<DecisionId>,
        by_entity_decision_ids: BTreeSet<DecisionId>,
        by_entity_intention_ids: BTreeSet<IntentionId>,
        by_intention_len: usize,
        by_entity_decision_len: usize,
        by_entity_intention_len: usize,
        by_intention_statuses: Vec<(DecisionId, DecisionStatus)>,
        by_entity_decision_statuses: Vec<(DecisionId, DecisionStatus)>,
        writer_active_at_start: bool,
        writer_done_before_read: bool,
        writer_done_after_read: bool,
    }

    let reader = thread::spawn(move || {
        reader_barrier.wait();
        let mut reads = Vec::new();
        while reads.len() < 150 || !reader_done.load(Ordering::SeqCst) {
            let writer_active_at_start =
                reader_started.load(Ordering::SeqCst) && !reader_done.load(Ordering::SeqCst);
            let writer_done_before_read = reader_done.load(Ordering::SeqCst);
            reader_in_call_flag.store(true, Ordering::SeqCst);
            let by_intention = reader_store
                .by_intention(main_intention_id, DiscoveryFilter::default())
                .expect("reader by_intention");
            let by_entity_decisions = reader_store
                .by_entity_decisions(entity_id, DiscoveryFilter::default())
                .expect("reader by_entity_decisions");
            let by_entity_intentions = reader_store
                .by_entity_intentions(entity_id, DiscoveryFilter::default())
                .expect("reader by_entity_intentions");
            reader_in_call_flag.store(false, Ordering::SeqCst);
            let writer_done_after_read = reader_done.load(Ordering::SeqCst);
            reads.push(ObservedRead {
                by_intention_ids: decision_ids(&by_intention),
                by_entity_decision_ids: decision_ids(&by_entity_decisions),
                by_entity_intention_ids: intention_ids(&by_entity_intentions),
                by_intention_len: by_intention.len(),
                by_entity_decision_len: by_entity_decisions.len(),
                by_entity_intention_len: by_entity_intentions.len(),
                by_intention_statuses: by_intention
                    .iter()
                    .map(|row| (row.id, row.status))
                    .collect(),
                by_entity_decision_statuses: by_entity_decisions
                    .iter()
                    .map(|row| (row.id, row.status))
                    .collect(),
                writer_active_at_start,
                writer_done_before_read,
                writer_done_after_read,
            });
            if !writer_done_before_read {
                thread::yield_now();
            }
        }
        reads
    });
    let writer = thread::spawn(move || {
        writer_barrier.wait();
        while !writer_observed_reader.load(Ordering::SeqCst) {
            thread::yield_now();
        }
        writer_started_flag.store(true, Ordering::SeqCst);
        let mut replacements = Vec::new();
        for (n, (decision_id, old_intention_id, new_intention_id)) in
            writer_pairs.into_iter().enumerate()
        {
            let replacement = writer_store
                .supersede_decision(
                    decision_id,
                    decision_input(
                        ctx_id,
                        &format!("replacement {n:02}"),
                        vec![main_intention_id, new_intention_id],
                        vec![entity_id],
                    ),
                )
                .expect("writer supersede");
            replacements.push((
                decision_id,
                replacement.id,
                old_intention_id,
                new_intention_id,
            ));
            thread::sleep(Duration::from_millis(2));
            thread::yield_now();
        }
        writer_done_flag.store(true, Ordering::SeqCst);
        replacements
    });

    let replacements = writer.join().expect("writer join");
    let reads = reader.join().expect("reader join");
    let old_ids = replacements
        .iter()
        .map(|(old_id, _, _, _)| *old_id)
        .collect::<BTreeSet<_>>();
    let replacement_ids = replacements
        .iter()
        .map(|(_, replacement_id, _, _)| *replacement_id)
        .collect::<BTreeSet<_>>();
    let old_intention_ids = replacements
        .iter()
        .map(|(_, _, old_intention_id, _)| *old_intention_id)
        .collect::<BTreeSet<_>>();
    let new_intention_ids = replacements
        .iter()
        .map(|(_, _, _, new_intention_id)| *new_intention_id)
        .collect::<BTreeSet<_>>();
    let expected_len = 1 + replacements.len();
    assert!(reads.len() >= 150);
    assert!(reads.iter().any(|read| read.writer_active_at_start));
    assert!(
        reads
            .iter()
            .any(|read| read.writer_active_at_start && !read.writer_done_after_read)
    );
    assert!(
        reads
            .iter()
            .any(|read| replacement_ids.is_subset(&read.by_intention_ids))
    );
    assert!(
        reads
            .iter()
            .any(|read| replacement_ids.is_subset(&read.by_entity_decision_ids))
    );
    assert!(
        reads
            .iter()
            .any(|read| new_intention_ids.is_subset(&read.by_entity_intention_ids))
    );
    assert!(reads.iter().all(|read| {
        read.by_intention_statuses
            .iter()
            .all(|(_, status)| *status == DecisionStatus::Active)
    }));
    assert!(reads.iter().all(|read| {
        read.by_entity_decision_statuses
            .iter()
            .all(|(_, status)| *status == DecisionStatus::Active)
    }));
    for read in &reads {
        for (len, ids) in [
            (read.by_intention_len, &read.by_intention_ids),
            (read.by_entity_decision_len, &read.by_entity_decision_ids),
        ] {
            assert!(ids.contains(&stable.id));
            assert_eq!(
                len, expected_len,
                "duplicate or missing decision rows: {read:?}"
            );
            assert_eq!(ids.len(), expected_len, "torn decision read: {read:?}");
            for (old_id, replacement_id, _, _) in &replacements {
                assert_ne!(
                    ids.contains(old_id),
                    ids.contains(replacement_id),
                    "read must contain exactly one row from a supersede pair: {read:?}"
                );
            }
            if read.writer_done_before_read {
                assert!(
                    old_ids.is_disjoint(ids),
                    "post-writer read still included superseded rows: {read:?}"
                );
                assert!(
                    replacement_ids.is_subset(ids),
                    "post-writer read missed replacements: {read:?}"
                );
            }
        }
        assert!(read.by_entity_intention_ids.contains(&main_intention_id));
        assert_eq!(
            read.by_entity_intention_len, expected_len,
            "duplicate or missing intention rows: {read:?}"
        );
        assert_eq!(
            read.by_entity_intention_ids.len(),
            expected_len,
            "torn intention read: {read:?}"
        );
        for (_, _, old_intention_id, new_intention_id) in &replacements {
            assert_ne!(
                read.by_entity_intention_ids.contains(old_intention_id),
                read.by_entity_intention_ids.contains(new_intention_id),
                "read must contain exactly one Intention from a supersede pair: {read:?}"
            );
        }
        if read.writer_done_before_read {
            assert!(
                old_intention_ids.is_disjoint(&read.by_entity_intention_ids),
                "post-writer read still included old served Intentions: {read:?}"
            );
            assert!(
                new_intention_ids.is_subset(&read.by_entity_intention_ids),
                "post-writer read missed new served Intentions: {read:?}"
            );
        }
    }
    assert_discovery_source_contract();
}

// One Decision with both touch edges should still yield its served Intention once.
#[test]
fn by_entity_intention_dedups_when_decision_has_multi_edges_to_same_entity() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = context(&store, "multi-edge");
    let e = entity(&store, ctx.id, "service");
    let i = intention(&store, ctx.id, "understand service");
    let d1 = decision(&store, ctx.id, "based and about", vec![i.id], vec![e.id]);
    let unrelated_entity = entity(&store, ctx.id, "other-service");
    decision(
        &store,
        ctx.id,
        "unrelated",
        vec![i.id],
        vec![unrelated_entity.id],
    );
    drop(store);
    insert_about_edges(&db_path, ctx.id, &[(d1.id, e.id)]);
    let store = reopen(&db_path);

    let rows = ok!(
        store.by_entity_intentions(e.id, DiscoveryFilter::default()),
        "by entity intentions"
    );

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, i.id);
}

// A reviewer iterating changed Entities should see a multi-Entity Decision for each anchor.
#[test]
fn by_entity_decision_state_bleed_across_entity_queries() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = context(&store, "state-bleed");
    let e1 = entity(&store, ctx.id, "service-one");
    let e2 = entity(&store, ctx.id, "service-two");
    let i = intention(&store, ctx.id, "review coupled services");
    let d = decision(
        &store,
        ctx.id,
        "shared decision",
        vec![i.id],
        vec![e1.id, e2.id],
    );

    let first = ok!(
        store.by_entity_decisions(e1.id, DiscoveryFilter::default()),
        "first entity"
    );
    assert_decision_ids(&first, &[d.id]);
    let second = ok!(
        store.by_entity_decisions(e2.id, DiscoveryFilter::default()),
        "second entity"
    );
    assert_decision_ids(&second, &[d.id]);
    drop(store);

    let store = reopen(&db_path);
    let second_reopened = ok!(
        store.by_entity_decisions(e2.id, DiscoveryFilter::default()),
        "second reopened"
    );
    assert_decision_ids(&second_reopened, &[d.id]);
    let first_reopened = ok!(
        store.by_entity_decisions(e1.id, DiscoveryFilter::default()),
        "first reopened"
    );
    assert_decision_ids(&first_reopened, &[d.id]);
}

// Explicit empty status lists mean return nothing, distinct from the default live set.
#[test]
fn by_entity_include_statuses_empty_vec_returns_nothing_symmetric_with_by_intention() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "empty-filter");
    let e = entity(&store, ctx.id, "service");
    let i = intention(&store, ctx.id, "serve service");
    let d = decision(&store, ctx.id, "active decision", vec![i.id], vec![e.id]);

    let empty_decisions = ok!(
        store.by_entity_decisions(e.id, decision_filter(vec![])),
        "empty decisions"
    );
    assert!(empty_decisions.is_empty());
    let empty_intentions = ok!(
        store.by_entity_intentions(e.id, discovery_filter(Some(vec![]), Some(vec![]))),
        "empty intentions"
    );
    assert!(empty_intentions.is_empty());
    let empty_decision_axis_only = ok!(
        store.by_entity_intentions(e.id, discovery_filter(Some(vec![]), None)),
        "empty decision axis only"
    );
    assert!(empty_decision_axis_only.is_empty());
    let empty_intention_axis_only = ok!(
        store.by_entity_intentions(e.id, discovery_filter(None, Some(vec![]))),
        "empty intention axis only"
    );
    assert!(empty_intention_axis_only.is_empty());

    let default_decisions = ok!(
        store.by_entity_decisions(e.id, DiscoveryFilter::default()),
        "default decisions"
    );
    assert_decision_ids(&default_decisions, &[d.id]);
    let default_intentions = ok!(
        store.by_entity_intentions(e.id, DiscoveryFilter::default()),
        "default intentions"
    );
    assert_intention_ids(&default_intentions, &[i.id]);
}

// A Decision serving two goals is still one touching Decision for this Entity.
#[test]
fn by_entity_decision_serving_multiple_intentions_returns_once_not_per_intention() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "multi-intention");
    let e = entity(&store, ctx.id, "service");
    let i1 = intention(&store, ctx.id, "first goal");
    let i2 = intention(&store, ctx.id, "second goal");
    let d = decision(
        &store,
        ctx.id,
        "one decision",
        vec![i1.id, i2.id],
        vec![e.id],
    );

    let rows = ok!(
        store.by_entity_decisions(e.id, DiscoveryFilter::default()),
        "by entity decisions"
    );

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, d.id);
    assert_decision_rows_match_get(&store, &rows);

    let by_first = ok!(
        store.by_intention(i1.id, DiscoveryFilter::default()),
        "by intention first"
    );
    assert_decision_ids(&by_first, &[d.id]);
    assert_decision_rows_match_get(&store, &by_first);
    let by_second = ok!(
        store.by_intention(i2.id, DiscoveryFilter::default()),
        "by intention second"
    );
    assert_decision_ids(&by_second, &[d.id]);
    assert_decision_rows_match_get(&store, &by_second);

    let intentions = ok!(
        store.by_entity_intentions(e.id, DiscoveryFilter::default()),
        "by entity intentions"
    );
    assert_intention_ids(&intentions, &[i1.id, i2.id]);
    assert_intention_order(&intentions, &[i2.id, i1.id]);
    assert_intention_rows_match_get(&store, &intentions);
}

// A paused goal is still a valid anchor; its answer depends on served Decisions, not input status.
#[test]
fn by_intention_with_paused_intention_input_is_queryable() {
    let (_tempdir, _db_path, store) = fixture();
    let ctx = context(&store, "paused-input");
    let i_paused_empty =
        intention_with_status(&store, ctx.id, "seasonal goal", IntentionStatus::Paused);
    let i_paused_served = intention_with_status(
        &store,
        ctx.id,
        "paused served goal",
        IntentionStatus::Paused,
    );
    let served = decision(
        &store,
        ctx.id,
        "active service for paused goal",
        vec![i_paused_served.id],
        vec![],
    );

    let default = ok!(
        store.by_intention(i_paused_empty.id, DiscoveryFilter::default()),
        "paused empty intention default"
    );
    assert!(default.is_empty());

    let served_default = ok!(
        store.by_intention(i_paused_served.id, DiscoveryFilter::default()),
        "paused served intention default"
    );
    assert_decision_ids(&served_default, &[served.id]);
    assert_decision_rows_match_get(&store, &served_default);

    let widened = ok!(
        store.by_intention(
            i_paused_empty.id,
            decision_filter(vec![
                DecisionStatus::Active,
                DecisionStatus::Superseded,
                DecisionStatus::Archived,
                DecisionStatus::Invalidated,
            ]),
        ),
        "paused intention widened"
    );
    assert!(widened.is_empty());
}
