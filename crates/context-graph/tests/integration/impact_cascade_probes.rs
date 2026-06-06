use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Barrier};
use std::thread;

use chrono::Utc;
use context_graph::*;
use contextdb_core::{TxId, Value};
use contextdb_engine::{Database, QueryResult};
use serde_json::json;
use uuid::Uuid;

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        ..Default::default()
    })
    .expect("open");
    (dir, store)
}

fn reopen_store(dir: &tempfile::TempDir) -> Store {
    Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        ..Default::default()
    })
    .expect("reopen")
}

fn open_raw_database(dir: &tempfile::TempDir) -> Database {
    let db = Database::open(dir.path().join("cg.db")).expect("raw open");
    let registered = db.registered_trigger_callbacks();
    for name in [
        "impact_analysis_trigger",
        "conflict_detection_on_decisions",
        "conflict_detection_on_outcomes",
        "precedent_match_on_decisions",
    ] {
        if db
            .list_triggers()
            .iter()
            .any(|trigger| trigger.name == name)
            && !registered.iter().any(|registered| registered == name)
        {
            db.register_trigger_callback(name, |_db, _ctx| Ok(()))
                .expect("register trigger callback");
        }
    }
    if !db.list_triggers().is_empty() {
        db.complete_initialization().expect("complete init");
    }
    db
}

fn props(entries: &[(&str, serde_json::Value)]) -> BTreeMap<String, serde_json::Value> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn delta(entries: &[(&str, serde_json::Value)]) -> BTreeMap<String, Option<serde_json::Value>> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), Some(value.clone())))
        .collect()
}

fn evidence_ref(observation_id: ObservationId) -> EvidenceRef {
    EvidenceRef {
        id: EvidenceId::new_v7(),
        observation_id,
        context_id: ContextId::default(),
        kind: EvidenceKind::StructuredSignal,
        source_ref: "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        content_hash: Some(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        ),
        captured_at: Some(Utc::now()),
        region: Some(json!({"line_start": 1, "line_end": 2})),
        producer: EvidenceProducer::default(),
        retention_status: RetentionStatus::RetainedExternal,
        ..Default::default()
    }
}

fn context(store: &Store, name: &str) -> Context {
    store
        .create_context(CreateContext {
            name: format!("{name}-{}", Uuid::now_v7()),
            ..Default::default()
        })
        .expect("context")
}

fn entity(store: &Store, context_id: ContextId, name: &str) -> Entity {
    store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: name.to_string(),
            properties: props(&[("version", json!("1"))]),
            context_id,
            ..Default::default()
        })
        .expect("entity")
}

fn intention(store: &Store, context_id: ContextId, description: &str) -> Intention {
    store
        .create_intention(CreateIntention {
            description: description.to_string(),
            context_id,
            ..Default::default()
        })
        .expect("intention")
}

fn decision(
    store: &Store,
    context_id: ContextId,
    description: &str,
    intention_id: IntentionId,
    based_on: Vec<EntityId>,
    precedent_ids: Vec<DecisionId>,
) -> Decision {
    store
        .create_decision(CreateDecision {
            decision_type: "integration".to_string(),
            description: description.to_string(),
            confidence: Some(0.9),
            intention_ids: vec![intention_id],
            based_on_entity_ids: based_on,
            basis_fields: None,
            precedent_ids,
            context_id,
            ..Default::default()
        })
        .expect("decision")
}

fn observe(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    version: &str,
) -> std::result::Result<RecordObservationResult, CgError> {
    let observation_id = ObservationId::new_v7();
    store.record_observation(RecordObservation {
        id: observation_id,
        entity_id,
        context_id,
        observation_type: "change".to_string(),
        source: "integration".to_string(),
        observed_at: Utc::now(),
        evidence: vec![evidence_ref(observation_id)],
        observed_properties: BTreeMap::new(),
        state_delta: delta(&[("version", json!(version))]),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    })
}

fn uuid_column_set(result: &QueryResult, column: &str) -> BTreeSet<Uuid> {
    let index = result
        .columns
        .iter()
        .position(|name| name == column)
        .unwrap_or_else(|| panic!("missing column {column}"));
    result
        .rows
        .iter()
        .map(|row| match row.get(index) {
            Some(Value::Uuid(id)) => *id,
            other => panic!("{column}: expected uuid, got {other:?}"),
        })
        .collect()
}

fn insert_engine_node(db: &Database, id: Uuid) {
    db.execute(
        "INSERT INTO nodes (id) VALUES ($id)",
        &HashMap::from([("id".to_string(), Value::Uuid(id))]),
    )
    .expect("insert node");
}

fn insert_engine_cites(db: &Database, source: Uuid, target: Uuid) {
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type) VALUES ($id, $source_id, $target_id, $edge_type)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(Uuid::now_v7())),
            ("source_id".to_string(), Value::Uuid(source)),
            ("target_id".to_string(), Value::Uuid(target)),
            ("edge_type".to_string(), Value::Text("CITES".to_string())),
        ]),
    )
    .expect("insert cites");
}

fn insert_raw_cites(
    dir: &tempfile::TempDir,
    context_id: ContextId,
    source: DecisionId,
    target: DecisionId,
) {
    let db = open_raw_database(dir);
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, valid_to, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $valid_to, $properties)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(Uuid::now_v7())),
            ("source_id".to_string(), Value::Uuid(source.as_uuid())),
            ("target_id".to_string(), Value::Uuid(target.as_uuid())),
            ("edge_type".to_string(), Value::Text("CITES".to_string())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("valid_from".to_string(), Value::TxId(TxId(1))),
            ("valid_to".to_string(), Value::Null),
            ("properties".to_string(), Value::Json(json!({}))),
        ]),
    )
    .expect("insert raw cites");
}

fn update_decision_status(dir: &tempfile::TempDir, decision_id: DecisionId, status: &str) {
    let db = open_raw_database(dir);
    db.execute(
        "UPDATE decisions SET status = $status WHERE id = $id",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(decision_id.as_uuid())),
            ("status".to_string(), Value::Text(status.to_string())),
        ]),
    )
    .expect("update status");
}

fn affected_set(rows: &[Invalidation]) -> BTreeSet<DecisionId> {
    rows.iter().map(|row| row.affected_decision_id).collect()
}

fn invalidations_by_decision(rows: &[Invalidation]) -> BTreeMap<DecisionId, Vec<&Invalidation>> {
    let mut out: BTreeMap<DecisionId, Vec<&Invalidation>> = BTreeMap::new();
    for row in rows {
        out.entry(row.affected_decision_id).or_default().push(row);
    }
    out
}

#[test]
fn closure_incoming_anchored_depth_equiv_per_hop() {
    let db = Database::open_memory();
    db.execute("CREATE TABLE nodes (id UUID PRIMARY KEY)", &HashMap::new())
        .expect("create nodes");
    db.execute(
        "CREATE TABLE edges (id UUID PRIMARY KEY, source_id UUID, target_id UUID, edge_type TEXT) DAG('CITES')",
        &HashMap::new(),
    )
    .expect("create edges");

    let root = Uuid::from_u128(0x1000);
    let chain = (1..=11)
        .map(|index| Uuid::from_u128(0x1100 + index))
        .collect::<Vec<_>>();
    let diamond_a = Uuid::from_u128(0x1201);
    let diamond_b = Uuid::from_u128(0x1202);
    let merge = Uuid::from_u128(0x1203);
    let branch = (1..=11)
        .map(|index| Uuid::from_u128(0x1300 + index))
        .collect::<Vec<_>>();

    let all_nodes = std::iter::once(root)
        .chain(chain.iter().copied())
        .chain([diamond_a, diamond_b, merge])
        .chain(branch.iter().copied())
        .collect::<Vec<_>>();
    for id in all_nodes {
        insert_engine_node(&db, id);
    }

    insert_engine_cites(&db, chain[0], root);
    for window in chain.windows(2) {
        insert_engine_cites(&db, window[1], window[0]);
    }
    insert_engine_cites(&db, diamond_a, root);
    insert_engine_cites(&db, diamond_b, root);
    insert_engine_cites(&db, merge, diamond_a);
    insert_engine_cites(&db, merge, diamond_b);
    insert_engine_cites(&db, branch[0], root);
    for window in branch.windows(2) {
        insert_engine_cites(&db, window[1], window[0]);
    }

    let mut truth = chain.iter().take(10).copied().collect::<BTreeSet<_>>();
    truth.extend([diamond_a, diamond_b, merge]);
    truth.extend(branch.iter().take(10).copied());

    let bounded = db
        .execute(
            "SELECT d_id FROM GRAPH_TABLE(edges MATCH (root)<-[:CITES]-{1,10}(d) WHERE root.id = $root COLUMNS (d.id AS d_id))",
            &HashMap::from([("root".to_string(), Value::Uuid(root))]),
        )
        .expect("bounded incoming closure");
    let bounded_set = uuid_column_set(&bounded, "d_id");

    let mut visited = BTreeSet::new();
    let mut frontier = BTreeSet::from([root]);
    for _ in 0..10 {
        let mut next = BTreeSet::new();
        for current in &frontier {
            let hop = db
                .execute(
                    "SELECT d_id FROM GRAPH_TABLE(edges MATCH (cited)<-[:CITES]-(d) WHERE cited.id = $current COLUMNS (d.id AS d_id))",
                    &HashMap::from([("current".to_string(), Value::Uuid(*current))]),
                )
                .expect("single hop");
            for id in uuid_column_set(&hop, "d_id") {
                if visited.insert(id) {
                    next.insert(id);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    assert_eq!(bounded_set, truth);
    assert_eq!(visited, truth);
    assert!(bounded_set.contains(&chain[9]));
    assert!(!bounded_set.contains(&chain[10]));
    assert!(visited.contains(&chain[9]));
    assert!(!visited.contains(&chain[10]));
}

#[test]
fn closure_equiv_with_non_active_midchain_matches_invalidation_semantics() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "non-active-midchain");
    let ent = entity(&store, ctx.id, "router");
    let root_intent = intention(&store, ctx.id, "Keep router version current");
    let mid_intent = intention(&store, ctx.id, "Document replacement route");
    let tail_intent = intention(&store, ctx.id, "Audit replacement route");
    let root = decision(
        &store,
        ctx.id,
        "router version 1 is approved",
        root_intent.id,
        vec![ent.id],
        Vec::new(),
    );
    let mid = decision(
        &store,
        ctx.id,
        "old migration note cites router approval",
        mid_intent.id,
        Vec::new(),
        vec![root.id],
    );
    let tail = decision(
        &store,
        ctx.id,
        "audit note cites old migration note",
        tail_intent.id,
        Vec::new(),
        vec![mid.id],
    );
    drop(store);
    update_decision_status(&dir, mid.id, "superseded");
    let store = reopen_store(&dir);

    let result = observe(&store, ctx.id, ent.id, "2").expect("observe");

    assert_eq!(
        affected_set(&result.invalidations),
        BTreeSet::from([root.id])
    );
    assert!(!affected_set(&result.invalidations).contains(&mid.id));
    assert!(!affected_set(&result.invalidations).contains(&tail.id));
    let row = result
        .invalidations
        .iter()
        .find(|row| row.affected_decision_id == root.id)
        .expect("root row");
    assert_eq!(row.cascade_from, None);
    assert_eq!(row.severity, InvalidationSeverity::Critical);
}

#[test]
fn cascade_does_not_leak_across_context_via_cites() {
    let (dir, store) = fresh_store();
    let ctx_a = context(&store, "context-a");
    let ctx_b = context(&store, "context-b");
    let ent = entity(&store, ctx_a.id, "billing-api");
    let root_intent = intention(&store, ctx_a.id, "Keep billing API current");
    let a1_intent = intention(&store, ctx_a.id, "Keep billing runbook current");
    let a2_intent = intention(&store, ctx_a.id, "Keep billing audit current");
    let b_intent = intention(&store, ctx_b.id, "Keep regional runbook current");
    let root = decision(
        &store,
        ctx_a.id,
        "billing API version 1 is approved",
        root_intent.id,
        vec![ent.id],
        Vec::new(),
    );
    let a1 = decision(
        &store,
        ctx_a.id,
        "billing runbook cites API approval",
        a1_intent.id,
        Vec::new(),
        vec![root.id],
    );
    let a2 = decision(
        &store,
        ctx_a.id,
        "billing audit cites runbook",
        a2_intent.id,
        Vec::new(),
        vec![a1.id],
    );
    let b1 = decision(
        &store,
        ctx_b.id,
        "regional runbook cites billing audit",
        b_intent.id,
        Vec::new(),
        Vec::new(),
    );
    drop(store);
    insert_raw_cites(&dir, ctx_b.id, b1.id, a2.id);
    let store = reopen_store(&dir);

    let result = observe(&store, ctx_a.id, ent.id, "2").expect("observe");

    assert_eq!(
        affected_set(&result.invalidations),
        BTreeSet::from([root.id, a1.id, a2.id])
    );
    assert!(!affected_set(&result.invalidations).contains(&b1.id));
    assert!(
        store
            .list_invalidations(ctx_b.id, None, Some(InvalidationStatus::Pending))
            .expect("context-b invalidations")
            .is_empty()
    );
}

#[test]
fn diamond_cites_attribution_through_public_surface() {
    let (_dir, store) = fresh_store();
    let ctx = context(&store, "diamond-cites");
    let ent = entity(&store, ctx.id, "checkout-api");
    let root_intent = intention(&store, ctx.id, "Keep checkout API current");
    let a_intent = intention(&store, ctx.id, "Keep rollback route current");
    let b_intent = intention(&store, ctx.id, "Keep rollout route current");
    let merge_intent = intention(&store, ctx.id, "Keep operator guide current");
    let root = decision(
        &store,
        ctx.id,
        "checkout API version 1 is approved",
        root_intent.id,
        vec![ent.id],
        Vec::new(),
    );
    let a = decision(
        &store,
        ctx.id,
        "rollback route cites checkout approval",
        a_intent.id,
        Vec::new(),
        vec![root.id],
    );
    let b = decision(
        &store,
        ctx.id,
        "rollout route cites checkout approval",
        b_intent.id,
        Vec::new(),
        vec![root.id],
    );
    let merge = decision(
        &store,
        ctx.id,
        "operator guide cites both routes",
        merge_intent.id,
        Vec::new(),
        vec![a.id, b.id],
    );

    let result = observe(&store, ctx.id, ent.id, "2").expect("observe");
    let by_decision = invalidations_by_decision(&result.invalidations);
    let merge_rows = by_decision.get(&merge.id).expect("merge row");

    assert_eq!(merge_rows.len(), 1);
    assert_eq!(merge_rows[0].cascade_from, Some(root.id));
    assert_eq!(merge_rows[0].severity, InvalidationSeverity::Info);
    assert!(merge_rows[0].basis_diff.is_invalidated_candidate);
    assert!(merge_rows[0].basis_diff.divergences.is_empty());
}

#[test]
fn concurrent_record_observation_two_root_serializes_with_typed_error_on_refusal() {
    let (_dir, store) = fresh_store();
    let ctx = context(&store, "concurrent-two-root");
    let ent = entity(&store, ctx.id, "search-api");
    let first_intent = intention(&store, ctx.id, "Keep search API primary route current");
    let second_intent = intention(&store, ctx.id, "Keep search API secondary route current");
    let cascade_intent = intention(&store, ctx.id, "Keep search API operations current");
    let root_a = decision(
        &store,
        ctx.id,
        "primary route uses search API version 1",
        first_intent.id,
        vec![ent.id],
        Vec::new(),
    );
    let root_b = decision(
        &store,
        ctx.id,
        "secondary route uses search API version 1",
        second_intent.id,
        vec![ent.id],
        Vec::new(),
    );
    let mut previous = root_a.id;
    for depth in 1..=10 {
        let node = decision(
            &store,
            ctx.id,
            &format!("operations note {depth} cites previous route"),
            cascade_intent.id,
            Vec::new(),
            vec![previous],
        );
        previous = node.id;
    }
    store
        .admin_raw_insert_cites(previous, root_b.id)
        .expect("link secondary root into cascade");
    let store = Arc::new(store);
    let barrier = Arc::new(Barrier::new(2));

    let mut handles = Vec::new();
    for version in ["2", "3"] {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            observe(&store, ctx.id, ent.id, version)
        }));
    }

    let results = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer thread"))
        .collect::<Vec<_>>();
    for result in &results {
        match result {
            Ok(result) => assert!(
                result.invalidations.is_empty() || result.invalidations.len() == 12,
                "unexpected committed invalidation count: {}",
                result.invalidations.len()
            ),
            Err(CgError::Engine(_)) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    let committed_rows = results
        .into_iter()
        .filter_map(Result::ok)
        .map(|result| result.invalidations.len())
        .sum::<usize>();
    let stored_rows = store
        .list_invalidations(ctx.id, None, Some(InvalidationStatus::Pending))
        .expect("invalidations");
    assert_eq!(stored_rows.len(), committed_rows);
    assert_eq!(stored_rows.len(), 12);
}
