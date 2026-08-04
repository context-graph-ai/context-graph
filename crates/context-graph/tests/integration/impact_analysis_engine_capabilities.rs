use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Barrier};
use std::thread;

use chrono::Utc;
use context_graph::*;
use contextdb_core::{TxId, Value};
use contextdb_engine::Database;
use serde_json::json;

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
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
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

fn intention(store: &Store, context_id: ContextId) -> Intention {
    store
        .create_intention(CreateIntention {
            description: "Keep service state current".to_string(),
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

// Concurrent observation writers should serialize or surface the engine's typed refusal without torn derived rows.
#[test]
fn engine_serializes_concurrent_record_observation_with_typed_error_on_refusal() {
    let (_dir, store) = fresh_store();
    let ctx = context(&store, "concurrent-impact");
    let ent = entity(&store, ctx.id, "service");
    let intent = intention(&store, ctx.id);
    let _decision = decision(
        &store,
        ctx.id,
        "service is at version 1",
        intent.id,
        vec![ent.id],
        Vec::new(),
    );
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
    let successes = results.iter().filter(|result| result.is_ok()).count();
    let typed_refusals = results
        .iter()
        .filter(|result| {
            matches!(
                result,
                Err(CgError::Engine(message))
                    if message.contains("transaction control is not allowed from another thread")
                        || message.contains("trigger callback is active")
            )
        })
        .count();

    assert_eq!(successes + typed_refusals, 2);
    let invalidations = store
        .list_invalidations(ctx.id, None, Some(InvalidationStatus::Pending))
        .expect("invalidations");
    assert_eq!(invalidations.len(), 1);
    let returned_count = results
        .into_iter()
        .filter_map(Result::ok)
        .map(|result| result.invalidations.len())
        .sum::<usize>();
    assert_eq!(returned_count, 1);
}

// The cascade cap should be the product's ten-hop boundary, not the engine ceiling.
#[test]
fn cg_cascade_depth_cap_below_engine_cap() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "depth-cap");
    let ent = entity(&store, ctx.id, "root-service");
    let intent = intention(&store, ctx.id);
    let mut decisions = Vec::new();
    let d1 = decision(
        &store,
        ctx.id,
        "root decision",
        intent.id,
        vec![ent.id],
        Vec::new(),
    );
    decisions.push(d1);
    for index in 2..=12 {
        let previous = decisions.last().expect("previous").id;
        decisions.push(decision(
            &store,
            ctx.id,
            &format!("cascade decision {index}"),
            intent.id,
            Vec::new(),
            vec![previous],
        ));
    }
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(&store, ctx.id, ent.id, "2").expect("observe");
    let affected = result
        .invalidations
        .iter()
        .map(|row| row.affected_decision_id)
        .collect::<HashSet<_>>();
    let expected = decisions
        .iter()
        .take(11)
        .map(|decision| decision.id)
        .collect::<HashSet<_>>();

    assert_eq!(result.invalidations.len(), 11);
    assert_eq!(affected, expected);
    assert!(!affected.contains(&decisions[11].id));
}

// A callback write failure should roll back the firing observation rather than leave partial state.
#[test]
fn trigger_callback_failure_rolls_back_observation() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "rollback");
    let ent = entity(&store, ctx.id, "rollback-service");
    let intent = intention(&store, ctx.id);
    let dec = decision(
        &store,
        ctx.id,
        "service is at version 1",
        intent.id,
        vec![ent.id],
        Vec::new(),
    );
    let seed_observation = observe(&store, ctx.id, ent.id, "1").expect("seed observation");
    drop(store);

    let db = open_raw_database(&dir);
    db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(uuid::Uuid::now_v7())),
            ("status".to_string(), Value::Text("pending".to_string())),
            ("context_id".to_string(), Value::Uuid(ctx.id.as_uuid())),
            ("affected_decision_id".to_string(), Value::Uuid(dec.id.as_uuid())),
            (
                "trigger_observation_id".to_string(),
                Value::Uuid(seed_observation.observation_id.as_uuid()),
            ),
            (
                "basis_diff".to_string(),
                Value::Json(json!({
                    "is_invalidated_candidate": true,
                    "divergences": [],
                    "read_txid": 1,
                    "current_snapshot_read_txid": 1
                })),
            ),
            ("severity".to_string(), Value::Text("warning".to_string())),
            ("detected_at".to_string(), Value::TxId(TxId(1))),
        ]),
    )
    .expect("seed invalidation");
    drop(db);

    let store = reopen_store(&dir);
    let attempted_id = ObservationId::new_v7();
    let attempted = RecordObservation {
        id: attempted_id,
        entity_id: ent.id,
        context_id: ctx.id,
        observation_type: "change".to_string(),
        source: "integration".to_string(),
        observed_at: Utc::now(),
        evidence: vec![evidence_ref(attempted_id)],
        observed_properties: BTreeMap::new(),
        state_delta: delta(&[("version", json!("2"))]),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    };
    let result = store.record_observation(attempted);

    assert!(
        matches!(&result, Err(CgError::UniqueViolation { entity, .. }) if *entity == "invalidation"),
        "the failing callback surfaces the duplicate invalidation that stopped it: {result:?}"
    );
    drop(store);
    let db = open_raw_database(&dir);
    let observations = db
        .scan("observations", db.snapshot())
        .expect("observations");
    assert!(
        !observations
            .iter()
            .any(|row| row.values.get("id") == Some(&Value::Uuid(attempted_id.as_uuid())))
    );
    let invalidations = db
        .scan("invalidations", db.snapshot())
        .expect("invalidations");
    assert_eq!(invalidations.len(), 1);
}
