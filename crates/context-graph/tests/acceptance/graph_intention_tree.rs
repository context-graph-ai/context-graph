use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId, EntityId,
    EntityType, EvidenceId, EvidenceKind, EvidenceProducer, EvidenceRef, IntentionId,
    IntentionOrigin, IntentionStatus, InvalidationSeverity, InvalidationStatus, ObservationId,
    ReasoningStep, RecordObservation, RetentionStatus, Store, StoreConfig,
};
use contextdb_core::{TxId, Value};
use contextdb_engine::Database;
use tempfile::TempDir;
use uuid::Uuid;

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

fn context_input(name: &str) -> CreateContext {
    CreateContext {
        name: name.to_string(),
        labels: vec![],
        properties: BTreeMap::new(),
    }
}

fn create_intention_input(context_id: ContextId, description: &str) -> CreateIntention {
    CreateIntention {
        id: None,
        description: description.to_string(),
        status: IntentionStatus::Active,
        origin: IntentionOrigin::Human,
        context_id,
        properties: BTreeMap::new(),
        tags: vec![],
    }
}

fn create_entity_input(context_id: ContextId, entity_type: EntityType, name: &str) -> CreateEntity {
    CreateEntity {
        entity_type,
        name: name.to_string(),
        properties: BTreeMap::new(),
        tags: vec![],
        context_id,
    }
}

fn reasoning(label: &str) -> Vec<ReasoningStep> {
    vec![ReasoningStep {
        operation: format!("op-{label}"),
        details: format!("details-{label}"),
        result: format!("result-{label}"),
        agent_id: Some("claude-code-instance-dev".to_string()),
        timestamp: Utc::now(),
    }]
}

fn text_evidence(observation_id: ObservationId, label: &str) -> EvidenceRef {
    EvidenceRef {
        id: EvidenceId::new_v7(),
        observation_id,
        context_id: ContextId::default(),
        kind: EvidenceKind::Text,
        source_ref: format!("webhook://seed/{label}"),
        mime_type: Some("text/plain".to_string()),
        content_hash: None,
        captured_at: Some(Utc::now()),
        byte_range: None,
        time_range_ms: None,
        frame_index: None,
        region: None,
        producer: EvidenceProducer {
            system: "acceptance".to_string(),
            model_name: "fixture".to_string(),
            model_version: "1".to_string(),
            pipeline_version: "1".to_string(),
        },
        text_projection: None,
        derived_from: None,
        retention_status: RetentionStatus::RetainedExternal,
        media_retention_until: None,
        properties: BTreeMap::new(),
    }
}

fn create_decision_input(
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
    precedent_ids: Vec<DecisionId>,
) -> CreateDecision {
    CreateDecision {
        decision_type: "security".to_string(),
        description: description.to_string(),
        reasoning: reasoning(description),
        confidence: Some(0.9),
        intention_ids,
        based_on_entity_ids,
        basis_fields: None,
        based_on_snapshots: vec![],
        tags: vec![],
        properties: BTreeMap::new(),
        context_id,
        precedent_ids,
        agent_id: None,
    }
}

fn seed_observation(
    store: &Store,
    ctx: ContextId,
    entity_id: EntityId,
    label: &str,
) -> ObservationId {
    let observation_id = ObservationId::new_v7();
    ok!(
        store.record_observation(RecordObservation {
            id: observation_id,
            entity_id,
            context_id: ctx,
            observation_type: format!("seed-{label}"),
            source: "test-seed".to_string(),
            observed_at: Utc::now(),
            evidence: vec![text_evidence(observation_id, label)],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::new(),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        }),
        "record_observation seed"
    )
    .observation_id
}

// The `Store` facade exposes no invalidation-write method. Invalidations are
// produced by engine-emission on observation basis-diffs; these tests exercise
// only the read path, so rows are seeded via a direct contextdb INSERT.
fn seed_invalidation(
    db_path: &Path,
    context_id: ContextId,
    affected_decision: DecisionId,
    trigger_observation: ObservationId,
    status: &str,
    severity: &str,
    detected_at: u64,
) {
    let db = Database::open(db_path).expect("open contextdb for invalidation seed");
    let params: HashMap<String, Value> = [
        ("id".to_string(), Value::Uuid(Uuid::now_v7())),
        ("status".to_string(), Value::Text(status.to_string())),
        ("context_id".to_string(), Value::Uuid(context_id.as_uuid())),
        (
            "affected".to_string(),
            Value::Uuid(affected_decision.as_uuid()),
        ),
        (
            "trigger".to_string(),
            Value::Uuid(trigger_observation.as_uuid()),
        ),
        ("basis_diff".to_string(), Value::Json(serde_json::json!({}))),
        ("severity".to_string(), Value::Text(severity.to_string())),
        ("detected_at".to_string(), Value::TxId(TxId(detected_at))),
    ]
    .into_iter()
    .collect();
    db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at) VALUES ($id, $status, $context_id, $affected, $trigger, $basis_diff, $severity, $detected_at)",
        &params,
    )
    .expect("seed invalidation row");
}

// Use case: A consultant returning to a month-old engagement asks "show me the whole picture under this Intention" — every Decision that serves it, the Entities those Decisions rest on, how each one turned out, and any pending invalidations. The engine returns the complete picture in one call from the on-disk store.
#[test]
fn intention_tree_expands_decisions_entities_outcomes_invalidations() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("test-intention-tree")),
        "tx0: context"
    )
    .id;
    let i = ok!(
        store.create_intention(create_intention_input(ctx, "Secure user authentication")),
        "tx1: I"
    );
    let e1 = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Module, "auth-module")),
        "tx2: E1"
    );
    let e2 = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Service,
            "identity-provider"
        )),
        "tx3: E2"
    );
    let d1 = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "Use OAuth 2.0",
            vec![i.id],
            vec![e1.id, e2.id],
            vec![],
        )),
        "tx4: D1"
    );
    let d2 = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "Use PKCE flow for mobile",
            vec![i.id],
            vec![e1.id],
            vec![d1.id],
        )),
        "tx5: D2 cites D1"
    );
    ok!(
        store.record_outcome(d1.id, true, Some("rolled out cleanly".to_string())),
        "tx6: record outcome on D1"
    );

    // Adversarial secondary intention: D4 created first, then D3, then CITES D4->D3 via admin write.
    let i_prime = ok!(
        store.create_intention(create_intention_input(ctx, "adversarial ordering")),
        "tx7: I'"
    );
    let d4 = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "D4 first-created",
            vec![i_prime.id],
            vec![],
            vec![],
        )),
        "tx8: D4"
    );
    let d3 = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "D3 created after D4 but becomes root via post-facto CITES",
            vec![i_prime.id],
            vec![],
            vec![],
        )),
        "tx9: D3"
    );
    ok!(
        store.admin_raw_insert_cites(d4.id, d3.id),
        "tx10: post-facto CITES D4 -> D3"
    );

    let obs_for_d2 = seed_observation(&store, ctx, e1.id, "d2-pending");
    let obs_for_d1 = seed_observation(&store, ctx, e1.id, "d1-acknowledged");

    drop(store);
    seed_invalidation(&db_path, ctx, d2.id, obs_for_d2, "pending", "warning", 1);
    seed_invalidation(&db_path, ctx, d1.id, obs_for_d1, "acknowledged", "info", 2);
    let store = reopen(&db_path);
    let tree = ok!(store.intention_tree(i.id), "tx11: intention_tree I");
    assert_eq!(tree.intention.id, i.id);
    let decision_ids: Vec<DecisionId> = tree.decisions.iter().map(|d| d.id).collect();
    assert_eq!(decision_ids, vec![d1.id, d2.id]);
    let entity_ids: HashSet<EntityId> = tree.entities.iter().map(|e| e.id).collect();
    let expected_entities: HashSet<EntityId> = [e1.id, e2.id].into_iter().collect();
    assert_eq!(entity_ids, expected_entities);
    assert_eq!(tree.outcomes.len(), 1);
    assert_eq!(tree.outcomes[0].decision_id, d1.id);
    assert_eq!(tree.invalidations.len(), 1);
    assert_eq!(tree.invalidations[0].context_id, ctx);
    assert_eq!(tree.invalidations[0].affected_decision_id, d2.id);
    assert_eq!(tree.invalidations[0].status, InvalidationStatus::Pending);

    let tree2 = ok!(store.intention_tree(i_prime.id), "tx12: intention_tree I'");
    let decision_ids2: Vec<DecisionId> = tree2.decisions.iter().map(|d| d.id).collect();
    assert_eq!(decision_ids2, vec![d3.id, d4.id]);
}

// Use case: A developer watching a refresh-token decision sees both a direct Info flag from a routine config observation and a Critical cascade flag from a DPoP change. The intention tree returns them Critical-first so the urgent signal never hides behind a routine one.
#[test]
fn intention_tree_orders_invalidations_by_severity() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("refresh-token")),
        "tx0: context"
    )
    .id;
    let i = ok!(
        store.create_intention(create_intention_input(ctx, "secure token exchange")),
        "tx1: I"
    );
    let e1 = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Service,
            "identity-provider"
        )),
        "tx2: E1"
    );
    let d1 = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "refresh-token policy",
            vec![i.id],
            vec![e1.id],
            vec![],
        )),
        "tx3: D1"
    );
    let d2 = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "dpop-bound refresh-token policy",
            vec![i.id],
            vec![e1.id],
            vec![],
        )),
        "tx3b: D2"
    );

    let obs_info = seed_observation(&store, ctx, e1.id, "d1-info");
    let obs_critical = seed_observation(&store, ctx, e1.id, "d2-critical");

    drop(store);
    seed_invalidation(&db_path, ctx, d1.id, obs_info, "pending", "info", 1);
    seed_invalidation(&db_path, ctx, d2.id, obs_critical, "pending", "critical", 2);
    let store = reopen(&db_path);
    let tree = ok!(store.intention_tree(i.id), "tx4: intention_tree");
    assert_eq!(tree.invalidations.len(), 2);
    assert_eq!(
        tree.invalidations[0].severity,
        InvalidationSeverity::Critical
    );
    assert_eq!(tree.invalidations[1].severity, InvalidationSeverity::Info);
    assert_eq!(tree.invalidations[0].context_id, ctx);
    assert_eq!(tree.invalidations[1].context_id, ctx);
    assert_eq!(tree.invalidations[0].affected_decision_id, d2.id);
    assert_eq!(tree.invalidations[1].affected_decision_id, d1.id);
}
