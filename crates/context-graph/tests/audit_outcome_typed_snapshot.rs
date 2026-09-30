#![cfg(feature = "test-support")]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use chrono::Utc;
use context_graph::schema::TABLES;
use context_graph::{
    AuditFilter, AuditTarget, ConsumerDeclaration, ConsumerReader, ConsumerSchema, ConsumerTable,
    ContextId, CreateContext, CreateDecision, CreateIntention, DecisionId, IntentionOrigin,
    IntentionStatus, ReasoningStep, Store, StoreConfig,
};
use contextdb_core::Value;
use contextdb_engine::DatabasePlugin;
use serde_json::Value as JsonValue;
use tempfile::TempDir;

const OUTCOME_LOOKUP: &str = "SELECT id, decision_id, success, notes, recorded_at FROM outcomes WHERE decision_id = $decision_id";

struct PauseBeforeOutcomeWrite {
    armed: AtomicBool,
    entered: Barrier,
    release: Barrier,
}

impl DatabasePlugin for PauseBeforeOutcomeWrite {
    fn on_query(&self, sql: &str) -> contextdb_core::Result<()> {
        if sql == OUTCOME_LOOKUP && self.armed.swap(false, Ordering::SeqCst) {
            self.entered.wait();
            self.release.wait();
        }
        Ok(())
    }
}

fn store_config(path: std::path::PathBuf) -> StoreConfig {
    StoreConfig {
        db_path: path,
        default_text_embedder: None,
        ..Default::default()
    }
}

fn audit_declaration() -> ConsumerDeclaration {
    let audit_log = TABLES
        .iter()
        .find(|table| table.name == "audit_log")
        .expect("published audit_log table");
    ConsumerDeclaration {
        schema: ConsumerSchema {
            namespace: "audit-snapshot-proof".to_string(),
            tables: vec![ConsumerTable {
                name: audit_log.name.to_string(),
                ddl: audit_log.ddl.to_string(),
            }],
        },
        scope_labels: BTreeSet::new(),
    }
}

fn create_decision(store: &Store) -> (ContextId, DecisionId) {
    let context = store
        .create_context(CreateContext {
            name: "typed audit snapshot".to_string(),
            labels: vec![],
            properties: BTreeMap::new(),
        })
        .expect("create Context");
    let intention = store
        .create_intention(CreateIntention {
            id: None,
            description: "prove typed audit Outcome timestamps".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: context.id,
            properties: BTreeMap::new(),
            tags: vec![],
            blueprint_catalog_id: None,
        })
        .expect("create Intention");
    let decision = store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: "store Outcome audit snapshots by typed timestamp".to_string(),
            reasoning: vec![ReasoningStep {
                operation: "record".to_string(),
                details: "audit timestamp".to_string(),
                result: "typed storage".to_string(),
                agent_id: None,
                timestamp: chrono::DateTime::<Utc>::from_timestamp_millis(1_700_000_000_000)
                    .expect("fixed fixture time"),
            }],
            confidence: None,
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![],
            basis_fields: None,
            based_on_snapshots: vec![],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: context.id,
            precedent_ids: vec![],
            agent_id: None,
        })
        .expect("create Decision");
    (context.id, decision.id)
}

fn commit_between_staged_and_outcome_commit(
    store: &Store,
    context_id: ContextId,
) -> context_graph::TxId {
    let db = store.sync_database();
    let tx = db.begin().expect("begin intervening transaction");
    let id = uuid::Uuid::now_v7();
    db.execute_in_tx(
        tx,
        "INSERT INTO audit_log (id, txid, operation, actor, reason, target_kind, target_id, context_id, outcome_recorded_at, payload) VALUES ($id, $txid, $operation, $actor, $reason, $target_kind, $target_id, $context_id, $outcome_recorded_at, $payload)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(id)),
            ("txid".to_string(), Value::TxId(tx)),
            ("operation".to_string(), Value::Text("record_outcome".to_string())),
            ("actor".to_string(), Value::Text("system".to_string())),
            ("reason".to_string(), Value::Null),
            ("target_kind".to_string(), Value::Text("context".to_string())),
            ("target_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
            ("outcome_recorded_at".to_string(), Value::Null),
            ("payload".to_string(), Value::Json(JsonValue::Null)),
        ]),
    )
    .expect("stage intervening audit row");
    db.commit(tx).expect("commit intervening transaction");
    let committed = db
        .execute(
            "SELECT txid FROM audit_log WHERE id = $id",
            &HashMap::from([("id".to_string(), Value::Uuid(id))]),
        )
        .expect("read intervening audit row");
    match committed.rows.as_slice() {
        [row] => match row.as_slice() {
            [Value::TxId(tx)] => *tx,
            other => panic!("intervening audit row must expose its committed TXID: {other:?}"),
        },
        other => panic!("intervening audit query must return exactly one row: {other:?}"),
    }
}

#[test]
fn outcome_audit_snapshot_keeps_typed_committed_time_across_store_reopen_and_reader() {
    let root = TempDir::new().expect("tempdir");
    let path = root.path().join("context-graph.db");
    let pause = Arc::new(PauseBeforeOutcomeWrite {
        armed: AtomicBool::new(false),
        entered: Barrier::new(2),
        release: Barrier::new(2),
    });
    let store = Store::open_with_database_plugin(store_config(path.clone()), pause.clone())
        .expect("open Store with observer");
    let (context_id, decision_id) = create_decision(&store);

    pause.armed.store(true, Ordering::SeqCst);
    let caller = store.clone();
    let first =
        thread::spawn(move || caller.record_outcome(decision_id, true, Some("first".to_string())));
    pause.entered.wait();
    let intervening = commit_between_staged_and_outcome_commit(&store, context_id);
    pause.release.wait();
    let first = first.join().expect("first caller").expect("first Outcome");
    assert!(
        first.recorded_at > intervening,
        "the observer forced the Outcome transaction to commit after its staged TXID"
    );

    let second = store
        .record_outcome(decision_id, false, Some("replacement".to_string()))
        .expect("replacement Outcome");
    let live = store
        .get_decision(decision_id)
        .expect("live Decision")
        .expect("Decision exists")
        .outcome
        .expect("live Outcome");
    assert_eq!(live, second);
    let store_audit = store
        .audit_query(AuditFilter {
            target: Some(AuditTarget::Outcome(decision_id)),
            ..AuditFilter::default()
        })
        .expect("Store audit");
    assert!(
        store_audit
            .iter()
            .any(|entry| entry.outcome.as_ref() == Some(&first))
    );
    assert!(
        store_audit
            .iter()
            .any(|entry| entry.outcome.as_ref() == Some(&second))
    );

    let raw = store
        .sync_database()
        .execute(
            "SELECT outcome_recorded_at, payload FROM audit_log WHERE target_kind = $target_kind AND target_id = $target_id",
            &HashMap::from([
                ("target_kind".to_string(), Value::Text("outcome".to_string())),
                ("target_id".to_string(), Value::Uuid(decision_id.as_uuid())),
            ]),
    )
    .expect("raw Outcome audit storage");
    assert_eq!(raw.rows.len(), 3);
    let mut raw_times = Vec::with_capacity(raw.rows.len());
    for row in &raw.rows {
        let Some(Value::TxId(recorded_at)) = row.first() else {
            panic!("Outcome audit row must store its timestamp as TXID: {row:?}");
        };
        raw_times.push(*recorded_at);
        let Value::Json(JsonValue::Object(snapshot)) = &row[1] else {
            panic!("Outcome audit snapshot must remain JSON object: {row:?}");
        };
        assert!(!snapshot.contains_key("recorded_at"));
    }
    raw_times.sort();
    let mut expected_times = vec![first.recorded_at, first.recorded_at, second.recorded_at];
    expected_times.sort();
    assert_eq!(raw_times, expected_times);

    drop(store);
    let reopened = Store::open(store_config(path.clone())).expect("reopen Store");
    let reopened_live = reopened
        .get_decision(decision_id)
        .expect("reopened Decision")
        .expect("Decision persists")
        .outcome
        .expect("reopened Outcome");
    assert_eq!(reopened_live, second);
    let reopened_audit = reopened
        .audit_query(AuditFilter {
            target: Some(AuditTarget::Outcome(decision_id)),
            ..AuditFilter::default()
        })
        .expect("reopened Store audit");
    assert_eq!(reopened_audit, store_audit);
    drop(reopened);

    let reader = ConsumerReader::open(&path, audit_declaration()).expect("open read-only consumer");
    assert_eq!(
        reader
            .graph()
            .audit_query(AuditFilter {
                target: Some(AuditTarget::Outcome(decision_id)),
                ..AuditFilter::default()
            })
            .expect("read-only audit"),
        store_audit
    );
}
