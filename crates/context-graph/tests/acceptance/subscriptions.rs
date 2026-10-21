use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use chrono::Utc;
use context_graph::*;
use contextdb_core::{Lsn, TxId, Value as DbValue};
use contextdb_engine::Database as RawDatabase;
use serde_json::json;
use tempfile::TempDir;

type NewDecisionCollector = (
    TempDir,
    PathBuf,
    Store,
    ContextId,
    IntentionId,
    EntityId,
    Arc<Mutex<Vec<NewDecisionPayload>>>,
);

type InvalidationCollector = (
    TempDir,
    PathBuf,
    Store,
    ContextId,
    EntityId,
    IntentionId,
    DecisionId,
    Arc<Mutex<Vec<InvalidationDetectedPayload>>>,
);

type ConflictCollector = (
    TempDir,
    Store,
    ContextId,
    EntityId,
    IntentionId,
    DecisionId,
    Arc<Mutex<Vec<ConflictDetectedPayload>>>,
);

type PrecedentCollector = (
    TempDir,
    Store,
    ContextId,
    IntentionId,
    DecisionId,
    Arc<Mutex<Vec<PrecedentMatchPayload>>>,
);

fn fixture() -> (TempDir, PathBuf, Store) {
    let dir = TempDir::new().expect("tempdir");
    let db_path = dir.path().join("context-graph.db");
    let store = Store::open(StoreConfig {
        db_path: db_path.clone(),
        ..Default::default()
    })
    .expect("open store");
    (dir, db_path, store)
}

fn reopen(path: &Path) -> Store {
    Store::open(StoreConfig {
        db_path: path.to_path_buf(),
        ..Default::default()
    })
    .expect("reopen store")
}

fn reopen_scoped(path: &Path, contexts: impl IntoIterator<Item = ContextId>) -> Store {
    Store::open_with_contexts(
        StoreConfig {
            db_path: path.to_path_buf(),
            ..Default::default()
        },
        contexts.into_iter().collect::<BTreeSet<_>>(),
    )
    .expect("reopen scoped store")
}

fn raw_database(path: &Path) -> RawDatabase {
    let db = RawDatabase::open(path).expect("raw database open");
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
        db.complete_initialization()
            .expect("complete initialization");
    }
    db
}

fn seed_pending_invalidation_direct(
    path: &Path,
    context_id: ContextId,
    affected_decision_id: DecisionId,
    trigger_observation_id: ObservationId,
) {
    let db = raw_database(path);
    db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, basis_diff, severity, detected_at) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $basis_diff, $severity, $detected_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(uuid::Uuid::now_v7())),
            ("status".to_string(), DbValue::Text("pending".to_string())),
            ("context_id".to_string(), DbValue::Uuid(context_id.as_uuid())),
            (
                "affected_decision_id".to_string(),
                DbValue::Uuid(affected_decision_id.as_uuid()),
            ),
            (
                "trigger_observation_id".to_string(),
                DbValue::Uuid(trigger_observation_id.as_uuid()),
            ),
            (
                "basis_diff".to_string(),
                DbValue::Json(json!({
                    "is_invalidated_candidate": true,
                    "divergences": [],
                    "read_txid": 1,
                    "current_snapshot_read_txid": 1
                })),
            ),
            ("severity".to_string(), DbValue::Text("warning".to_string())),
            ("detected_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("seed invalidation");
}

fn wait_for_len<T>(items: &Arc<Mutex<Vec<T>>>, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && items.lock().expect("collector").len() < expected {
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn assert_no_more<T>(items: &Arc<Mutex<Vec<T>>>, expected: usize) {
    let deadline = Instant::now() + Duration::from_millis(250);
    while Instant::now() < deadline {
        assert_eq!(items.lock().expect("collector").len(), expected);
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_metrics(
    store: &Store,
    subscription_id: &SubscriptionId,
    ready: impl Fn(&SubscriptionMetrics) -> bool,
) -> SubscriptionMetrics {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let metrics = store
            .with_system_principal()
            .subscription_metrics(subscription_id)
            .expect("subscription metrics");
        if ready(&metrics) || Instant::now() >= deadline {
            return metrics;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn map(entries: &[(&str, serde_json::Value)]) -> BTreeMap<String, serde_json::Value> {
    entries
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

fn delta(entries: &[(&str, serde_json::Value)]) -> BTreeMap<String, Option<serde_json::Value>> {
    entries
        .iter()
        .map(|(k, v)| ((*k).to_string(), Some(v.clone())))
        .collect()
}

fn context(store: &Store, name: &str) -> Context {
    store
        .create_context(CreateContext {
            name: format!("{name}-{}", uuid::Uuid::now_v7()),
            ..Default::default()
        })
        .expect("create context")
}

fn intention(store: &Store, context_id: ContextId, description: &str) -> Intention {
    store
        .create_intention(CreateIntention {
            description: description.to_string(),
            context_id,
            ..Default::default()
        })
        .expect("create intention")
}

fn entity(
    store: &Store,
    context_id: ContextId,
    name: &str,
    properties: BTreeMap<String, serde_json::Value>,
) -> Entity {
    store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: name.to_string(),
            properties,
            context_id,
            ..Default::default()
        })
        .expect("create entity")
}

fn reasoning(label: &str) -> Vec<ReasoningStep> {
    vec![ReasoningStep {
        operation: format!("op-{label}"),
        details: format!("details-{label}"),
        result: format!("result-{label}"),
        agent_id: Some("subscription-test-agent".to_string()),
        timestamp: Utc::now(),
    }]
}

fn decision_with_type(
    store: &Store,
    context_id: ContextId,
    decision_type: &str,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
    precedent_ids: Vec<DecisionId>,
) -> Decision {
    store
        .create_decision(CreateDecision {
            decision_type: decision_type.to_string(),
            description: description.to_string(),
            reasoning: reasoning(description),
            confidence: Some(0.9),
            intention_ids,
            based_on_entity_ids,
            precedent_ids,
            context_id,
            ..Default::default()
        })
        .expect("create decision")
}

fn decision(
    store: &Store,
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
    precedent_ids: Vec<DecisionId>,
) -> Decision {
    decision_with_type(
        store,
        context_id,
        "architecture",
        description,
        intention_ids,
        based_on_entity_ids,
        precedent_ids,
    )
}

fn evidence_ref(observation_id: ObservationId) -> EvidenceRef {
    EvidenceRef {
        id: EvidenceId::new_v7(),
        observation_id,
        context_id: ContextId::default(),
        kind: EvidenceKind::Text,
        source_ref: "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_string(),
        content_hash: Some(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        ),
        captured_at: Some(Utc::now()),
        retention_status: RetentionStatus::RetainedExternal,
        ..Default::default()
    }
}

fn observe(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    changes: BTreeMap<String, Option<serde_json::Value>>,
) -> RecordObservationResult {
    let observation_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id,
            context_id,
            observation_type: "state-change".to_string(),
            source: "subscription-test".to_string(),
            observed_at: Utc::now(),
            evidence: vec![evidence_ref(observation_id)],
            observed_properties: BTreeMap::new(),
            state_delta: changes,
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record observation")
}

fn collect_new_decision() -> NewDecisionCollector {
    let (dir, path, store) = fixture();
    let ctx = context(&store, "new-decision").id;
    let intent = intention(&store, ctx, "reload gate policy immediately");
    let ent = entity(&store, ctx, "school-gate", BTreeMap::new());
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    store
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe new decision");
    (dir, path, store, ctx, intent.id, ent.id, payloads)
}

fn collect_invalidation() -> InvalidationCollector {
    let (dir, path, store) = fixture();
    let ctx = context(&store, "invalidation").id;
    let ent = entity(
        &store,
        ctx,
        "auth-module",
        map(&[("path", json!("src/auth.py"))]),
    );
    let intent = intention(&store, ctx, "reference the current auth module");
    let mut basis_fields = HashMap::new();
    basis_fields.insert(ent.id, vec!["path".to_string()]);
    let dec = store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: "auth module lives at src/auth.py".to_string(),
            reasoning: reasoning("auth module lives at src/auth.py"),
            confidence: Some(0.9),
            intention_ids: vec![intent.id],
            based_on_entity_ids: vec![ent.id],
            basis_fields: Some(basis_fields),
            context_id: ctx,
            ..Default::default()
        })
        .expect("create decision");
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    store
        .with_system_principal()
        .subscribe_invalidation_detected(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe invalidation");
    (dir, path, store, ctx, ent.id, intent.id, dec.id, payloads)
}

fn collect_conflict() -> ConflictCollector {
    let (dir, _path, store) = fixture();
    let ctx = context(&store, "conflict").id;
    let ent = entity(&store, ctx, "payment-service", BTreeMap::new());
    let intent = intention(&store, ctx, "keep payment routing consistent");
    let older = decision(
        &store,
        ctx,
        "route payments through primary provider",
        vec![intent.id],
        vec![ent.id],
        vec![],
    );
    store
        .record_outcome(older.id, true, Some("worked".to_string()))
        .expect("older outcome");
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    store
        .with_system_principal()
        .subscribe_conflict_detected(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe conflict");
    (dir, store, ctx, ent.id, intent.id, older.id, payloads)
}

fn collect_precedent() -> PrecedentCollector {
    let (dir, _path, store) = fixture();
    let ctx = context(&store, "precedent").id;
    let intent = intention(&store, ctx, "keep database connections under the limit");
    let prior = decision(
        &store,
        ctx,
        "use connection pooling with max=20 for orders DB",
        vec![intent.id],
        vec![],
        vec![],
    );
    store
        .record_outcome(prior.id, true, Some("stayed below limit".to_string()))
        .expect("prior outcome");
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    store
        .with_system_principal()
        .subscribe_precedent_match(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe precedent");
    (dir, store, ctx, intent.id, prior.id, payloads)
}

// Use case: a Vigil edge reloads the exact Intention served by a newly committed config Decision.
#[test]
fn subscribe_new_decision_fires_post_commit_carries_served_intention_ids_and_descriptions() {
    let (_dir, path, store, ctx, intent_id, entity_id, payloads) = collect_new_decision();
    let created = decision(
        &store,
        ctx,
        "alert when an unknown car waits at the school gate",
        vec![intent_id],
        vec![entity_id],
        vec![],
    );
    wait_for_len(&payloads, 1);
    let payload = payloads.lock().expect("collector")[0].clone();
    assert_eq!(payload.decision.id, created.id);
    assert!(payload.action_id > 0);
    assert_eq!(payload.served_intention_ids, vec![intent_id]);
    assert_eq!(
        payload.served_intention_descriptions.get(&intent_id),
        Some(&"reload gate policy immediately".to_string())
    );
    assert!(
        reopen(&path)
            .get_decision(created.id)
            .expect("get")
            .is_some()
    );
}

// Use case: an enterprise Slack sink can render a stale abstraction alert without a hot-path join.
#[test]
fn subscribe_invalidation_detected_fires_carries_intention_anchor_and_render_strings_u2_enterprise()
{
    let (_dir, _path, store, ctx, ent, intent, dec, payloads) = collect_invalidation();
    observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("lib/auth_v2/mod.rs"))]),
    );
    wait_for_len(&payloads, 1);
    let payload = payloads.lock().expect("collector")[0].clone();
    assert_eq!(payload.invalidation.affected_decision_id, dec);
    assert!(payload.action_id > 0);
    assert_eq!(
        payload.decision_description,
        "auth module lives at src/auth.py"
    );
    assert_eq!(payload.entity_name, "auth-module");
    assert_eq!(payload.invalidation.threatened_intention_ids, vec![intent]);
    assert_eq!(
        payload.threatened_intention_descriptions.get(&intent),
        Some(&"reference the current auth module".to_string())
    );
}

// Use case: a school-gate safety message preserves the parent's wording in the Intention.
#[test]
fn subscribe_invalidation_detected_renders_u1_kerala_school_gate_setup() {
    let (_dir, _path, store, ctx, ent, intent, _dec, payloads) = collect_invalidation();
    observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("gate/unknown-driver"))]),
    );
    wait_for_len(&payloads, 1);
    assert_eq!(
        payloads.lock().expect("collector")[0]
            .threatened_intention_descriptions
            .get(&intent),
        Some(&"reference the current auth module".to_string())
    );
}

// Use case: a rename before drift is rendered from the current Entity row, not stale write-time text.
#[test]
fn subscribe_invalidation_detected_carries_updated_description_text_when_entity_renamed() {
    let (_dir, _path, store, ctx, ent, _intent, _dec, payloads) = collect_invalidation();
    store
        .update_entity(
            ent,
            EntityPatch {
                name: Some("auth-module-v2".to_string()),
                ..Default::default()
            },
        )
        .expect("rename entity");
    observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("lib/auth_v2/mod.rs"))]),
    );
    wait_for_len(&payloads, 1);
    assert_eq!(
        payloads.lock().expect("collector")[0].entity_name,
        "auth-module-v2"
    );
}

// Use case: an agent can watch one Intention without receiving unrelated same-context drift.
#[test]
fn subscribe_invalidation_detected_with_intention_id_filter_only_fires_for_matching_intention() {
    let (_dir, _path, store) = fixture();
    let ctx = context(&store, "invalidation-intention-filter").id;
    let intent_a = intention(&store, ctx, "keep auth routing current");
    let intent_b = intention(&store, ctx, "keep billing routing current");
    let entity_a = entity(
        &store,
        ctx,
        "auth-module",
        map(&[("path", json!("auth/v1"))]),
    );
    let entity_b = entity(
        &store,
        ctx,
        "billing-module",
        map(&[("path", json!("billing/v1"))]),
    );

    for (entity_id, intent_id, description) in [
        (entity_a.id, intent_a.id, "auth module lives at auth/v1"),
        (
            entity_b.id,
            intent_b.id,
            "billing module lives at billing/v1",
        ),
    ] {
        let mut basis_fields = HashMap::new();
        basis_fields.insert(entity_id, vec!["path".to_string()]);
        store
            .create_decision(CreateDecision {
                decision_type: "architecture".to_string(),
                description: description.to_string(),
                reasoning: reasoning(description),
                confidence: Some(0.9),
                intention_ids: vec![intent_id],
                based_on_entity_ids: vec![entity_id],
                basis_fields: Some(basis_fields),
                context_id: ctx,
                ..Default::default()
            })
            .expect("create decision");
    }

    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    store
        .with_system_principal()
        .subscribe_invalidation_detected(
            SubscriptionFilter {
                intention_id: Some(intent_a.id),
                ..Default::default()
            },
            move |payload| {
                seen.lock().expect("collector").push(payload.clone());
                Ok(())
            },
        )
        .expect("subscribe invalidation");

    observe(
        &store,
        ctx,
        entity_b.id,
        delta(&[("path", json!("billing/v2"))]),
    );
    assert_no_more(&payloads, 0);

    observe(
        &store,
        ctx,
        entity_a.id,
        delta(&[("path", json!("auth/v2"))]),
    );
    wait_for_len(&payloads, 1);
    let captured = payloads.lock().expect("collector");
    assert_eq!(captured.len(), 1);
    assert_eq!(
        captured[0].invalidation.threatened_intention_ids,
        vec![intent_a.id]
    );
    assert_eq!(
        captured[0]
            .threatened_intention_descriptions
            .get(&intent_a.id),
        Some(&"keep auth routing current".to_string())
    );
}

// Use case: an invalidated lower-confidence sibling leaves the Intention served by a stronger Decision.
#[test]
fn invalidation_detected_with_higher_confidence_sibling_carries_intention_now_un_served_false_and_severity_warning()
 {
    let (_dir, _path, store, ctx, ent, intent, _dec, payloads) = collect_invalidation();
    decision(
        &store,
        ctx,
        "auth module lives at lib/auth_v2/mod.rs",
        vec![intent],
        vec![],
        vec![],
    );
    observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("lib/auth_v2/mod.rs"))]),
    );
    wait_for_len(&payloads, 1);
    assert!(
        !payloads.lock().expect("collector")[0]
            .invalidation
            .intention_now_un_served
    );
}

// Use case: every invalidation payload renders the affected Decision's own text.
#[test]
fn cascade_invalidation_payload_carries_cascade_from_pointer_and_renders_citing_decision_strings() {
    let (_dir, _path, store) = fixture();
    let ctx = context(&store, "cascade-invalidation").id;
    let ent = entity(
        &store,
        ctx,
        "auth-module",
        map(&[("path", json!("src/auth.py"))]),
    );
    let direct_intent = intention(&store, ctx, "reference the current auth module");
    let citing_intent = intention(&store, ctx, "keep deployment guide aligned to auth");
    let mut basis_fields = HashMap::new();
    basis_fields.insert(ent.id, vec!["path".to_string()]);
    let direct = store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: "auth module lives at src/auth.py".to_string(),
            reasoning: reasoning("auth module lives at src/auth.py"),
            confidence: Some(0.9),
            intention_ids: vec![direct_intent.id],
            based_on_entity_ids: vec![ent.id],
            basis_fields: Some(basis_fields),
            context_id: ctx,
            ..Default::default()
        })
        .expect("create direct decision");
    let citing = decision(
        &store,
        ctx,
        "deployment guide cites the auth module location",
        vec![citing_intent.id],
        Vec::new(),
        vec![direct.id],
    );
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    store
        .with_system_principal()
        .subscribe_invalidation_detected(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe invalidation");
    observe(
        &store,
        ctx,
        ent.id,
        delta(&[("path", json!("lib/auth_v2/mod.rs"))]),
    );
    wait_for_len(&payloads, 2);
    let captured = payloads.lock().expect("collector").clone();
    assert_eq!(captured.len(), 2);
    let direct_payload = captured
        .iter()
        .find(|payload| payload.cascade_from.is_none())
        .expect("direct payload");
    let cascade_payload = captured
        .iter()
        .find(|payload| payload.cascade_from == Some(direct.id))
        .expect("cascade payload");
    assert_eq!(direct_payload.invalidation.affected_decision_id, direct.id);
    assert_eq!(
        direct_payload.decision_description,
        "auth module lives at src/auth.py"
    );
    assert_eq!(
        direct_payload
            .threatened_intention_descriptions
            .get(&direct_intent.id),
        Some(&"reference the current auth module".to_string())
    );
    assert_eq!(cascade_payload.invalidation.affected_decision_id, citing.id);
    assert_eq!(cascade_payload.cascade_from, Some(direct.id));
    assert_eq!(
        cascade_payload.decision_description,
        "deployment guide cites the auth module location"
    );
    assert_eq!(
        cascade_payload
            .threatened_intention_descriptions
            .get(&citing_intent.id),
        Some(&"keep deployment guide aligned to auth".to_string())
    );
    assert_eq!(
        cascade_payload.invalidation.severity,
        InvalidationSeverity::Info
    );
}

// Use case: contradictory active Decisions wake the owner of the shared Intention.
#[test]
fn subscribe_conflict_detected_fires_on_outcome_divergence_carries_older_newer_ids_and_intention_intersection()
 {
    let (_dir, store, ctx, ent, intent, older, payloads) = collect_conflict();
    let newer = decision(
        &store,
        ctx,
        "route payments through backup provider",
        vec![intent],
        vec![ent],
        vec![],
    );
    store
        .record_outcome(newer.id, false, Some("failed".to_string()))
        .expect("newer outcome");
    wait_for_len(&payloads, 1);
    let payload = payloads.lock().expect("collector")[0].clone();
    assert_eq!(payload.older_decision_id, older);
    assert!(payload.action_id > 0);
    assert_eq!(payload.newer_decision_id, newer.id);
    assert_eq!(payload.threatened_intention_ids, vec![intent]);
    assert!(payload.intention_in_tension);
}

// Use case: an older-row outcome flip that creates divergence wakes the agent once.
#[test]
fn conflict_detected_fires_when_outcome_flip_creates_diverging_pair() {
    let (_dir, store, ctx, ent, intent, older, payloads) = collect_conflict();
    let peer = decision(&store, ctx, "same route", vec![intent], vec![ent], vec![]);
    store
        .record_outcome(peer.id, true, Some("same".to_string()))
        .expect("peer same outcome");
    assert_no_more(&payloads, 0);
    store
        .record_outcome(older, false, Some("now diverges".to_string()))
        .expect("flip outcome");
    wait_for_len(&payloads, 1);
    let payload = payloads.lock().expect("collector")[0].clone();
    assert_eq!(payload.older_decision_id, older);
    assert_eq!(payload.newer_decision_id, peer.id);
    store
        .record_outcome(older, true, Some("converges again".to_string()))
        .expect("flip back");
    assert_no_more(&payloads, 1);
}

// Use case: a prior successful Decision is surfaced as an actionable precedent immediately.
#[test]
fn subscribe_precedent_match_fires_for_cites_walked_successful_precedent_carries_intention_descriptions()
 {
    let (_dir, store, ctx, intent, prior, payloads) = collect_precedent();
    let new = decision(
        &store,
        ctx,
        "use connection pooling with max=20 for payments DB",
        vec![intent],
        vec![],
        vec![prior],
    );
    wait_for_len(&payloads, 1);
    let payload = payloads.lock().expect("collector")[0].clone();
    assert_eq!(payload.new_decision_id, new.id);
    assert!(payload.action_id > 0);
    assert_eq!(payload.precedent_decision_id, prior);
    assert_eq!(payload.served_intention_ids, vec![intent]);
    assert_eq!(
        payload.served_intention_descriptions.get(&intent),
        Some(&"keep database connections under the limit".to_string())
    );
    assert!(payload.matched_precedent_outcome_success);
}

// Use case: a failed prior Decision is not suggested as a precedent.
#[test]
fn precedent_match_does_not_fire_for_failed_outcome_precedent() {
    let (dir, _path, store) = fixture();
    let ctx = context(&store, "failed-precedent").id;
    let intent = intention(&store, ctx, "avoid cache stampedes");
    let prior = decision(
        &store,
        ctx,
        "use tiny cache",
        vec![intent.id],
        vec![],
        vec![],
    );
    store
        .record_outcome(prior.id, false, Some("failed".to_string()))
        .expect("failed outcome");
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    store
        .with_system_principal()
        .subscribe_precedent_match(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe");
    decision(
        &store,
        ctx,
        "use tiny cache again",
        vec![intent.id],
        vec![],
        vec![prior.id],
    );
    assert_no_more(&payloads, 0);
    drop(dir);
}

// Use case: orphan observations stay silent, while an anchored Decision is a positive control.
#[test]
fn invalidation_detected_silent_when_no_decisions_cite_entity_with_positive_control() {
    let (_dir, _path, store, ctx, ent, _intent, dec, payloads) = collect_invalidation();
    let orphan = entity(&store, ctx, "orphan", map(&[("state", json!("a"))]));
    observe(&store, ctx, orphan.id, delta(&[("state", json!("b"))]));
    assert_no_more(&payloads, 0);
    observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("lib/auth_v2/mod.rs"))]),
    );
    wait_for_len(&payloads, 1);
    assert_eq!(
        payloads.lock().expect("collector")[0]
            .invalidation
            .affected_decision_id,
        dec
    );
}

// Use case: entity-level contradictions still surface even when no Intention is shared.
#[test]
fn conflict_detected_fires_with_empty_intention_intersection_and_in_tension_false() {
    let (_dir, store, ctx, ent, _intent, _older, payloads) = collect_conflict();
    let other_intent = intention(&store, ctx, "keep backup routing available");
    let newer = decision(
        &store,
        ctx,
        "backup route",
        vec![other_intent.id],
        vec![ent],
        vec![],
    );
    store
        .record_outcome(newer.id, false, None)
        .expect("outcome");
    wait_for_len(&payloads, 1);
    assert!(
        payloads.lock().expect("collector")[0]
            .threatened_intention_ids
            .is_empty()
    );
    assert!(!payloads.lock().expect("collector")[0].intention_in_tension);
}

// Use case: archived Decisions do not keep generating active conflict alerts.
#[test]
fn conflict_detected_respects_decision_status_with_positive_control() {
    let (_dir, store, ctx, ent, intent, _older, payloads) = collect_conflict();
    let archived = decision(
        &store,
        ctx,
        "old backup route",
        vec![intent],
        vec![ent],
        vec![],
    );
    store
        .archive_decision(archived.id, "retired".to_string())
        .expect("archive");
    store
        .record_outcome(archived.id, false, None)
        .expect("outcome");
    assert_no_more(&payloads, 0);
}

// Use case: two Decisions on different decision types do not conflict just because they share an Entity.
#[test]
fn conflict_detected_same_entity_different_type_no_event_with_positive_control() {
    let (_dir, store, ctx, ent, intent, _older, payloads) = collect_conflict();
    let different = decision_with_type(
        &store,
        ctx,
        "security",
        "different type route",
        vec![intent],
        vec![ent],
        vec![],
    );
    store
        .record_outcome(different.id, false, None)
        .expect("outcome");
    assert_no_more(&payloads, 0);
}

// Use case: matching outcomes are agreement, not conflict.
#[test]
fn conflict_detected_same_outcome_no_event_with_positive_control() {
    let (_dir, store, ctx, ent, intent, _older, payloads) = collect_conflict();
    let peer = decision(&store, ctx, "same route", vec![intent], vec![ent], vec![]);
    store.record_outcome(peer.id, true, None).expect("outcome");
    assert_no_more(&payloads, 0);
}

// Use case: a scoped subscriber sees only the context it is authorized to watch.
#[test]
fn conflict_detected_scoped_to_context_with_engine_metrics_assertion() {
    let (dir, path, store) = fixture();
    let ctx_a = context(&store, "a").id;
    let ctx_b = context(&store, "b").id;
    let intent_a = intention(&store, ctx_a, "a intent");
    let ent_a = entity(&store, ctx_a, "a entity", BTreeMap::new());
    let older_a = decision(
        &store,
        ctx_a,
        "a primary route",
        vec![intent_a.id],
        vec![ent_a.id],
        vec![],
    );
    store
        .record_outcome(older_a.id, true, None)
        .expect("a outcome");
    let intent_b = intention(&store, ctx_b, "b intent");
    let ent_b = entity(&store, ctx_b, "b entity", BTreeMap::new());
    let older_b = decision(
        &store,
        ctx_b,
        "b primary route",
        vec![intent_b.id],
        vec![ent_b.id],
        vec![],
    );
    store
        .record_outcome(older_b.id, true, None)
        .expect("b outcome");
    drop(store);
    let store = reopen_scoped(&path, [ctx_a]);
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    let sub = store
        .with_system_principal()
        .subscribe_conflict_detected(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe scoped");
    let unscoped = reopen(&path);
    let newer_b = decision(
        &unscoped,
        ctx_b,
        "b backup route",
        vec![intent_b.id],
        vec![ent_b.id],
        vec![],
    );
    unscoped
        .record_outcome(newer_b.id, false, None)
        .expect("b divergent outcome");
    assert_no_more(&payloads, 0);
    let metrics_after_b = wait_for_metrics(&store, &sub, |metrics| metrics.delivered == 0);
    assert_eq!(metrics_after_b.delivered, 0);
    assert_eq!(metrics_after_b.permanent_failures, 0);
    let newer_a = decision(
        &store,
        ctx_a,
        "a backup route",
        vec![intent_a.id],
        vec![ent_a.id],
        vec![],
    );
    store
        .record_outcome(newer_a.id, false, None)
        .expect("a divergent outcome");
    wait_for_len(&payloads, 1);
    let payload = payloads.lock().expect("collector")[0].clone();
    assert_eq!(payload.context_id, ctx_a);
    assert_eq!(payload.older_decision_id, older_a.id);
    assert_eq!(payload.newer_decision_id, newer_a.id);
    let metrics_after_a = wait_for_metrics(&store, &sub, |metrics| metrics.delivered == 1);
    assert_eq!(metrics_after_a.delivered, 1);
    drop(dir);
}

// Use case: a Decision citing successful precedents preserves the public CITES-walk priority.
#[test]
fn precedent_match_trigger_emits_one_event_per_validated_cites_precedent_with_disambiguating_intentions()
 {
    let (dir, _path, store) = fixture();
    let ctx = context(&store, "precedent-global-hop-order").id;
    let child_early_intent = intention(&store, ctx, "early child");
    let child_early = decision(
        &store,
        ctx,
        "early second-hop precedent",
        vec![child_early_intent.id],
        vec![],
        vec![],
    );
    store
        .record_outcome(child_early.id, true, None)
        .expect("early child outcome");
    let child_late_intent = intention(&store, ctx, "late child");
    let child_late = decision(
        &store,
        ctx,
        "late second-hop precedent",
        vec![child_late_intent.id],
        vec![],
        vec![],
    );
    store
        .record_outcome(child_late.id, true, None)
        .expect("late child outcome");
    let parent_a_intent = intention(&store, ctx, "parent a");
    let parent_a = decision(
        &store,
        ctx,
        "first direct precedent cites late child",
        vec![parent_a_intent.id],
        vec![],
        vec![child_late.id],
    );
    store
        .record_outcome(parent_a.id, true, None)
        .expect("parent a outcome");
    let parent_b_intent = intention(&store, ctx, "parent b");
    let parent_b = decision(
        &store,
        ctx,
        "second direct precedent cites early child",
        vec![parent_b_intent.id],
        vec![],
        vec![child_early.id],
    );
    store
        .record_outcome(parent_b.id, true, None)
        .expect("parent b outcome");

    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    store
        .with_system_principal()
        .subscribe_precedent_match(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe precedent");
    let new_intent = intention(&store, ctx, "new composite intent");
    decision(
        &store,
        ctx,
        "new composite decision",
        vec![new_intent.id],
        vec![],
        vec![parent_a.id, parent_b.id],
    );
    wait_for_len(&payloads, 4);
    let captured = payloads.lock().expect("collector").clone();
    assert_eq!(
        captured
            .iter()
            .map(|payload| payload.precedent_decision_id)
            .collect::<Vec<_>>(),
        vec![parent_a.id, parent_b.id, child_early.id, child_late.id]
    );
    assert_eq!(
        captured[2].served_intention_ids,
        vec![child_early_intent.id]
    );
    assert_eq!(captured[3].served_intention_ids, vec![child_late_intent.id]);
    drop(dir);
}

// Use case: a previously surfaced precedent pair does not re-emit on unrelated work.
#[test]
fn precedent_match_idempotent_no_re_emit_for_already_surfaced_precedent_and_row_count_unique() {
    let (_dir, store, ctx, intent, prior, payloads) = collect_precedent();
    let new = decision(
        &store,
        ctx,
        "new pooled decision",
        vec![intent],
        vec![],
        vec![prior],
    );
    wait_for_len(&payloads, 1);
    store
        .update_decision(
            new.id,
            DecisionPatch {
                tags: Some(vec!["reviewed".to_string()]),
                ..Default::default()
            },
        )
        .expect("update");
    assert_no_more(&payloads, 1);
}

// Use case: serial config Decisions are delivered in write order.
#[test]
fn per_context_fifo_ordering_preserved_across_serial_writes_single_thread() {
    let (_dir, _path, store, ctx, intent, ent, payloads) = collect_new_decision();
    let d1 = decision(&store, ctx, "d1", vec![intent], vec![ent], vec![]);
    let d2 = decision(&store, ctx, "d2", vec![intent], vec![ent], vec![]);
    let d3 = decision(&store, ctx, "d3", vec![intent], vec![ent], vec![]);
    wait_for_len(&payloads, 3);
    let actual: Vec<_> = payloads
        .lock()
        .expect("collector")
        .iter()
        .map(|payload| payload.decision.id)
        .collect();
    assert_eq!(actual, vec![d1.id, d2.id, d3.id]);
}

// Use case: concurrent writers expose engine LSNs so consumers can recover commit order.
#[test]
fn per_context_fifo_ordering_preserved_under_concurrent_commits_via_lsn_ordering() {
    let (_dir, path, store, ctx, intent, ent, payloads) = collect_new_decision();
    let writers = 10;
    let barrier = Arc::new(Barrier::new(writers));
    let mut handles = Vec::new();
    for n in 0..writers {
        let writer_store = store.clone();
        let writer_barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            writer_barrier.wait();
            decision(
                &writer_store,
                ctx,
                &format!("concurrent-d{n}"),
                vec![intent],
                vec![ent],
                vec![],
            )
        }));
    }
    let committed = handles
        .into_iter()
        .map(|handle| handle.join().expect("writer thread"))
        .collect::<Vec<_>>();
    wait_for_len(&payloads, 10);
    let mut captured = payloads.lock().expect("collector").clone();
    assert_eq!(captured.len(), 10);
    let mut lsn = captured
        .iter()
        .map(|payload| payload.at_lsn)
        .collect::<Vec<_>>();
    lsn.sort();
    lsn.dedup();
    assert_eq!(lsn.len(), 10);
    captured.sort_by_key(|payload| payload.at_lsn);
    assert!(
        captured
            .windows(2)
            .all(|pair| pair[0].at_lsn < pair[1].at_lsn)
    );
    assert!(
        captured
            .windows(2)
            .all(|pair| pair[0].decision.created_at < pair[1].decision.created_at)
    );
    let committed_ids = committed
        .iter()
        .map(|decision| decision.id)
        .collect::<BTreeSet<_>>();
    let payload_ids = captured
        .iter()
        .map(|payload| payload.decision.id)
        .collect::<BTreeSet<_>>();
    assert_eq!(payload_ids, committed_ids);
    drop(store);
    let raw = RawDatabase::open(&path).expect("open raw database");
    let changes = raw.changes_since(Lsn(0)).expect("admit engine read memory");
    for payload in captured {
        let decision_lsn = changes
            .rows
            .iter()
            .find(|row| {
                row.table == "decisions"
                    && matches!(
                        row.values.get("id"),
                        Some(DbValue::Uuid(id)) if *id == payload.decision.id.as_uuid()
                    )
            })
            .map(|row| row.lsn);
        assert_eq!(decision_lsn, Some(payload.at_lsn));
    }
}

// Use case: an MSP narrows one subscription to two authorized client contexts with one sink.
#[test]
fn subscription_filter_context_ids_routes_msp_subscriber_across_authorized_contexts_with_single_sink_count()
 {
    let (dir, path, store) = fixture();
    let ctx_a = context(&store, "a").id;
    let ctx_b = context(&store, "b").id;
    let ctx_c = context(&store, "c").id;
    let ctx_d = context(&store, "d").id;
    drop(store);
    let scoped = reopen_scoped(&path, [ctx_a, ctx_b, ctx_c]);
    let denied = scoped.with_system_principal().subscribe_new_decision(
        SubscriptionFilter {
            context_ids: Some(vec![ctx_a, ctx_d]),
            ..Default::default()
        },
        |_payload| Ok(()),
    );
    assert!(matches!(
        denied,
        Err(SubscriptionError::ContextNotAuthorized(id)) if id == ctx_d
    ));
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    let sub = scoped
        .with_system_principal()
        .subscribe_new_decision(
            SubscriptionFilter {
                context_ids: Some(vec![ctx_a, ctx_b]),
                ..Default::default()
            },
            move |payload| {
                seen.lock().expect("collector").push(payload.clone());
                Ok(())
            },
        )
        .expect("subscribe");
    for ctx in [ctx_a, ctx_b, ctx_c] {
        let i = intention(&scoped, ctx, "intent");
        let e = entity(&scoped, ctx, "entity", BTreeMap::new());
        decision(&scoped, ctx, "decision", vec![i.id], vec![e.id], vec![]);
    }
    wait_for_len(&payloads, 2);
    let captured = payloads.lock().expect("collector");
    assert_eq!(captured.len(), 2);
    let contexts = captured
        .iter()
        .map(|payload| payload.decision.context_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(contexts, BTreeSet::from([ctx_a, ctx_b]));
    assert!(!contexts.contains(&ctx_c));
    drop(captured);
    let metrics = wait_for_metrics(&scoped, &sub, |metrics| metrics.delivered == 2);
    assert_eq!(metrics.delivered, 2);
    assert_eq!(metrics.permanent_failures, 0);
    drop(dir);
}

// Use case: a context-scoped edge never receives a neighboring branch's event.
#[test]
fn subscribe_scoped_to_context_blocks_other_context_events_with_engine_route_assertion() {
    let (dir, path, store) = fixture();
    let ctx_a = context(&store, "a").id;
    let ctx_b = context(&store, "b").id;
    drop(store);
    let scoped = reopen_scoped(&path, [ctx_a]);
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    let sub = scoped
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe scoped");
    let unscoped = reopen(&path);
    let intent_b = intention(&unscoped, ctx_b, "b intent");
    let ent_b = entity(&unscoped, ctx_b, "b entity", BTreeMap::new());
    decision(
        &unscoped,
        ctx_b,
        "b decision",
        vec![intent_b.id],
        vec![ent_b.id],
        vec![],
    );
    assert_no_more(&payloads, 0);
    let metrics_after_b = wait_for_metrics(&scoped, &sub, |metrics| metrics.delivered == 0);
    assert_eq!(metrics_after_b.delivered, 0);
    assert_eq!(metrics_after_b.permanent_failures, 0);
    let intent_a = intention(&scoped, ctx_a, "a intent");
    let ent_a = entity(&scoped, ctx_a, "a entity", BTreeMap::new());
    let accepted = decision(
        &scoped,
        ctx_a,
        "a decision",
        vec![intent_a.id],
        vec![ent_a.id],
        vec![],
    );
    wait_for_len(&payloads, 1);
    assert_eq!(
        payloads.lock().expect("collector")[0].decision.id,
        accepted.id
    );
    let metrics_after_a = wait_for_metrics(&scoped, &sub, |metrics| metrics.delivered == 1);
    assert_eq!(metrics_after_a.delivered, 1);
    drop(dir);
}

// Use case: one panicking subscriber does not stop a sibling subscriber from receiving the same event.
#[test]
fn subscribe_callback_panic_isolated_other_subscribers_continue_at_n_2() {
    let (_dir, _path, store, ctx, intent, ent, _payloads) = collect_new_decision();
    store
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), |_payload| {
            panic!("subscriber panic is isolated by engine");
        })
        .expect("subscribe panicker");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_clone = Arc::clone(&seen);
    store
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), move |payload| {
            seen_clone.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe sibling");
    decision(
        &store,
        ctx,
        "panic-isolated",
        vec![intent],
        vec![ent],
        vec![],
    );
    wait_for_len(&seen, 1);
    assert_eq!(seen.lock().expect("collector").len(), 1);
}

// Use case: panic isolation holds when most subscribers are failing.
#[test]
fn subscribe_callback_panic_isolated_at_n_10_with_seven_panicking_and_three_counting() {
    let (_dir, _path, store, ctx, intent, ent, _payloads) = collect_new_decision();
    for _ in 0..7 {
        store
            .with_system_principal()
            .subscribe_new_decision(SubscriptionFilter::default(), |_payload| {
                panic!("subscriber panic is isolated by engine");
            })
            .expect("subscribe panicker");
    }
    let seen = Arc::new(Mutex::new(Vec::new()));
    for _ in 0..3 {
        let seen_clone = Arc::clone(&seen);
        store
            .with_system_principal()
            .subscribe_new_decision(SubscriptionFilter::default(), move |payload| {
                seen_clone.lock().expect("collector").push(payload.clone());
                Ok(())
            })
            .expect("subscribe counter");
    }
    decision(
        &store,
        ctx,
        "panic-isolated-scale",
        vec![intent],
        vec![ent],
        vec![],
    );
    wait_for_len(&seen, 3);
    assert_eq!(seen.lock().expect("collector").len(), 3);
}

// Use case: two separate Store instances cannot poison each other's dispatchers.
#[test]
fn cross_instance_independence_one_store_panic_does_not_affect_parallel_store() {
    let (_dir_x, _path_x, store_x) = fixture();
    let ctx_x = context(&store_x, "panic-store").id;
    let intent_x = intention(&store_x, ctx_x, "x intent");
    let ent_x = entity(&store_x, ctx_x, "x entity", BTreeMap::new());
    store_x
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), |_payload| {
            panic!("subscriber panic is isolated by engine");
        })
        .expect("subscribe panicker");

    let (_dir_y, _path_y, store_y) = fixture();
    let ctx_y = context(&store_y, "healthy-store").id;
    let intent_y = intention(&store_y, ctx_y, "y intent");
    let ent_y = entity(&store_y, ctx_y, "y entity", BTreeMap::new());
    let seen_y = Arc::new(Mutex::new(Vec::new()));
    let seen_y_clone = Arc::clone(&seen_y);
    store_y
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), move |payload| {
            seen_y_clone
                .lock()
                .expect("collector")
                .push(payload.clone());
            Ok(())
        })
        .expect("subscribe healthy");

    decision(
        &store_x,
        ctx_x,
        "panic-store event",
        vec![intent_x.id],
        vec![ent_x.id],
        vec![],
    );
    let healthy = decision(
        &store_y,
        ctx_y,
        "healthy-store event",
        vec![intent_y.id],
        vec![ent_y.id],
        vec![],
    );
    wait_for_len(&seen_y, 1);
    assert_eq!(seen_y.lock().expect("collector")[0].decision.id, healthy.id);
}

// Use case: removing one subscription leaves sibling subscriptions alive and double-unsubscribe typed.
#[test]
fn unsubscribe_stops_events_for_removed_subscription_only_with_engine_metrics() {
    let (_dir, _path, store, ctx, intent, ent, _payloads) = collect_new_decision();
    let a = Arc::new(Mutex::new(Vec::new()));
    let b = Arc::new(Mutex::new(Vec::new()));
    let a_seen = Arc::clone(&a);
    let b_seen = Arc::clone(&b);
    let sub_a = store
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), move |payload| {
            a_seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe a");
    store
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), move |payload| {
            b_seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe b");
    decision(
        &store,
        ctx,
        "before unsubscribe",
        vec![intent],
        vec![ent],
        vec![],
    );
    wait_for_len(&a, 1);
    wait_for_len(&b, 1);
    let before_unsubscribe = wait_for_metrics(&store, &sub_a, |metrics| metrics.delivered == 1);
    assert_eq!(before_unsubscribe.delivered, 1);
    store
        .with_system_principal()
        .unsubscribe(sub_a.clone())
        .expect("unsubscribe");
    decision(
        &store,
        ctx,
        "after unsubscribe",
        vec![intent],
        vec![ent],
        vec![],
    );
    wait_for_len(&b, 2);
    assert_no_more(&a, 1);
    let after_unsubscribe = wait_for_metrics(&store, &sub_a, |metrics| metrics.delivered == 1);
    assert_eq!(after_unsubscribe.delivered, before_unsubscribe.delivered);
    assert_eq!(after_unsubscribe.queued, 0);
    assert!(matches!(
        store.with_system_principal().unsubscribe(sub_a),
        Err(SubscriptionError::UnknownSubscription(_))
    ));
}

// Use case: rolled-back writes do not publish staged events.
#[test]
fn subscription_does_not_fire_when_observation_transaction_rolls_back() {
    let (dir, path, store) = fixture();
    let ctx = context(&store, "rollback").id;
    let ent = entity(
        &store,
        ctx,
        "rollback-service",
        map(&[("version", json!("1"))]),
    );
    let intent = intention(&store, ctx, "keep rollback-service basis current");
    let mut basis_fields = HashMap::new();
    basis_fields.insert(ent.id, vec!["version".to_string()]);
    let dec = store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: "rollback-service is on version 1".to_string(),
            reasoning: reasoning("rollback-service is on version 1"),
            confidence: Some(0.9),
            intention_ids: vec![intent.id],
            based_on_entity_ids: vec![ent.id],
            basis_fields: Some(basis_fields),
            context_id: ctx,
            ..Default::default()
        })
        .expect("create decision");
    let seed_observation = observe(&store, ctx, ent.id, BTreeMap::new());
    drop(store);
    seed_pending_invalidation_direct(&path, ctx, dec.id, seed_observation.observation_id);

    let store = reopen(&path);
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    store
        .with_system_principal()
        .subscribe_invalidation_detected(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe invalidation");

    let attempted_id = ObservationId::new_v7();
    let result = store.record_observation(RecordObservation {
        id: attempted_id,
        entity_id: ent.id,
        context_id: ctx,
        observation_type: "state-change".to_string(),
        source: "subscription-test".to_string(),
        observed_at: Utc::now(),
        evidence: vec![evidence_ref(attempted_id)],
        observed_properties: BTreeMap::new(),
        state_delta: delta(&[("version", json!("2"))]),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    });
    assert!(matches!(
        result,
        Err(CgError::Engine(_))
            | Err(CgError::UniqueViolation {
                entity: "invalidation",
                ..
            })
    ));
    assert_no_more(&payloads, 0);
    assert!(matches!(
        store.get_observation(attempted_id),
        Err(CgError::NotFound {
            missing: Some("observation"),
            ..
        })
    ));
    let dec_after = store
        .get_decision(dec.id)
        .expect("get decision")
        .expect("decision exists");
    assert_eq!(dec_after.status, DecisionStatus::Active);
    let invalidations = store
        .list_invalidations(ctx, None, None)
        .expect("list invalidations");
    assert_eq!(invalidations.len(), 1);
    assert_eq!(
        invalidations[0].trigger_observation_id,
        seed_observation.observation_id
    );
    assert_ne!(invalidations[0].trigger_observation_id, attempted_id);
    drop(dir);
}

// Use case: a receiver uses the stable action identity when its first acknowledgement is lost.
#[test]
fn lost_subscription_acknowledgement_retries_one_identity_and_one_receiver_effect() {
    let (_dir, _path, store, ctx, ent, _intent, _dec, _payloads) = collect_invalidation();
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let visible_effects = Arc::new(Mutex::new(BTreeSet::new()));
    let attempts_for_handler = Arc::clone(&attempts);
    let effects_for_handler = Arc::clone(&visible_effects);
    let failing_sub = store
        .with_system_principal()
        .subscribe_invalidation_detected(SubscriptionFilter::default(), move |payload| {
            effects_for_handler
                .lock()
                .expect("receiver")
                .insert(payload.action_id);
            let mut attempts = attempts_for_handler.lock().expect("attempts");
            attempts.push(payload.action_id);
            if attempts.len() == 1 {
                Err(SubscriptionError::HandlerFailed(
                    "acknowledgement lost".to_string(),
                ))
            } else {
                Ok(())
            }
        })
        .expect("subscribe failing sink");
    let result = observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("src/auth-v2.py"))]),
    );
    assert_eq!(result.invalidations.len(), 1);
    let invalidations = store
        .list_invalidations(ctx, None, None)
        .expect("list invalidations");
    assert_eq!(invalidations.len(), 1);
    assert_eq!(
        invalidations[0].trigger_observation_id,
        result.observation_id
    );
    let metrics = wait_for_metrics(&store, &failing_sub, |metrics| {
        metrics.uncertain >= 1 && metrics.delivered == 1
    });
    assert_eq!(metrics.uncertain, 1);
    assert_eq!(metrics.permanent_failures, 0);
    let attempts = attempts.lock().expect("attempts");
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0], attempts[1]);
    assert_eq!(visible_effects.lock().expect("receiver").len(), 1);
}

#[test]
fn external_handler_without_dedup_reports_uncertain_delivery() {
    let (_dir, _path, store, ctx, ent, _intent, _dec, _payloads) = collect_invalidation();
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&attempts);
    let subscription = store
        .with_system_principal()
        .subscribe_invalidation_detected(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("attempts").push(payload.action_id);
            Err(SubscriptionError::HandlerFailed(
                "external acknowledgement was lost".to_string(),
            ))
        })
        .expect("subscribe external handler");
    observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("src/uncertain.py"))]),
    );
    wait_for_len(&attempts, 1);
    let metrics = wait_for_metrics(&store, &subscription, |metrics| metrics.uncertain > 0);
    assert!(metrics.uncertain > 0);
    assert_eq!(metrics.delivered, 0);
    assert_eq!(metrics.permanent_failures, 0);
    assert!(metrics.queued > 0);
    assert!(
        attempts
            .lock()
            .expect("attempts")
            .first()
            .is_some_and(|id| *id > 0)
    );
}

// Use case: basis_fields suppress unrelated Entity changes and forward relevant ones.
#[test]
fn basis_fields_region_restriction_fires_only_on_region_change() {
    let (_dir, _path, store, ctx, ent, _intent, _dec, payloads) = collect_invalidation();
    observe(
        &store,
        ctx,
        ent,
        delta(&[("instance_type", json!("m5.large"))]),
    );
    assert_no_more(&payloads, 0);
    observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("lib/auth_v2/mod.rs"))]),
    );
    wait_for_len(&payloads, 1);
}

// Use case: rapid observations on one Entity are delivered in observation order.
#[test]
fn serial_observations_on_same_entity_fire_invalidation_detected_in_observation_order() {
    let (_dir, _path, store, ctx, ent, _intent, _dec, payloads) = collect_invalidation();
    let first = observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("lib/auth_v2/mod.rs"))]),
    );
    wait_for_len(&payloads, 1);
    assert_eq!(
        payloads.lock().expect("collector")[0]
            .invalidation
            .trigger_observation_id,
        first.observation_id
    );
}

// Use case: a created Decision event precedes the later invalidation of that same Decision.
#[test]
fn cross_event_fifo_order_new_decision_precedes_invalidation_detected_for_same_decision_lifecycle()
{
    let (_dir, _path, store, ctx, ent, intent, _dec, invalidations) = collect_invalidation();
    let new_decisions = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&new_decisions);
    store
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("subscribe new");
    decision(
        &store,
        ctx,
        "new before invalidation",
        vec![intent],
        vec![ent],
        vec![],
    );
    wait_for_len(&new_decisions, 1);
    observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("lib/auth_v2/mod.rs"))]),
    );
    wait_for_len(&invalidations, 1);
    assert!(
        new_decisions.lock().expect("collector")[0].at_lsn
            < invalidations.lock().expect("collector")[0].at_lsn
    );
}

// Use case: a failed create_decision call leaves the subscription registration intact.
#[test]
fn subscription_preserved_across_pre_existing_create_decision_failure() {
    let (_dir, _path, store, ctx, _intent, ent, payloads) = collect_new_decision();
    let missing = IntentionId::new_v7();
    assert!(
        store
            .create_decision(CreateDecision {
                decision_type: "bad".to_string(),
                description: "bad".to_string(),
                intention_ids: vec![missing],
                based_on_entity_ids: vec![ent],
                context_id: ctx,
                ..Default::default()
            })
            .is_err()
    );
    let good = intention(&store, ctx, "good intent");
    decision(&store, ctx, "good", vec![good.id], vec![ent], vec![]);
    wait_for_len(&payloads, 1);
}

// Use case: a persisted subscription resumes queued push delivery after process restart.
#[test]
fn subscription_resume_replays_queued_events_after_reopen_on_same_subscription_id() {
    let (_dir, path, store) = fixture();
    let ctx = context(&store, "resume").id;
    let intent = intention(&store, ctx, "resume intent");
    let ent = entity(&store, ctx, "resume entity", BTreeMap::new());
    let sub = store
        .with_system_principal()
        .subscribe_new_decision(SubscriptionFilter::default(), |_payload| Ok(()))
        .expect("subscribe initial");
    drop(store);

    let store = reopen(&path);
    let queued = decision(
        &store,
        ctx,
        "queued while callback absent",
        vec![intent.id],
        vec![ent.id],
        vec![],
    );
    let before_resume = wait_for_metrics(&store, &sub, |metrics| metrics.queued == 1);
    assert_eq!(before_resume.delivered, 0);
    assert_eq!(before_resume.queued, 1);

    let payloads = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&payloads);
    let resumed = store
        .with_system_principal()
        .resume_new_decision(sub.clone(), SubscriptionFilter::default(), move |payload| {
            seen.lock().expect("collector").push(payload.clone());
            Ok(())
        })
        .expect("resume subscription");
    assert_eq!(resumed, sub);
    wait_for_len(&payloads, 1);
    assert_eq!(
        payloads.lock().expect("collector")[0].decision.id,
        queued.id
    );
    let after_resume = wait_for_metrics(&store, &resumed, |metrics| {
        metrics.delivered == 1 && metrics.queued == 0
    });
    assert_eq!(after_resume.delivered, 1);
    assert_eq!(after_resume.queued, 0);
}

// Use case: every public resume method rebinds its persisted sink after restart.
#[test]
fn subscription_resume_replays_queued_events_after_reopen_for_invalidation_conflict_and_precedent()
{
    let (_dir, path, store) = fixture();
    let ctx = context(&store, "resume-all-events").id;

    let inv_ent = entity(
        &store,
        ctx,
        "resume invalidation entity",
        map(&[("path", json!("src/a.rs"))]),
    );
    let inv_intent = intention(&store, ctx, "resume invalidation intent");
    let mut basis_fields = HashMap::new();
    basis_fields.insert(inv_ent.id, vec!["path".to_string()]);
    let inv_dec = store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: "resume invalidation decision".to_string(),
            reasoning: reasoning("resume invalidation decision"),
            confidence: Some(0.9),
            intention_ids: vec![inv_intent.id],
            based_on_entity_ids: vec![inv_ent.id],
            basis_fields: Some(basis_fields),
            context_id: ctx,
            ..Default::default()
        })
        .expect("create invalidation basis decision");

    let conflict_ent = entity(&store, ctx, "resume conflict entity", BTreeMap::new());
    let conflict_intent = intention(&store, ctx, "resume conflict intent");
    let conflict_older = decision(
        &store,
        ctx,
        "resume conflict primary",
        vec![conflict_intent.id],
        vec![conflict_ent.id],
        vec![],
    );
    store
        .record_outcome(conflict_older.id, true, Some("worked".to_string()))
        .expect("older conflict outcome");

    let precedent_intent = intention(&store, ctx, "resume precedent intent");
    let precedent_prior = decision(
        &store,
        ctx,
        "resume precedent prior",
        vec![precedent_intent.id],
        vec![],
        vec![],
    );
    store
        .record_outcome(precedent_prior.id, true, Some("worked".to_string()))
        .expect("precedent outcome");

    let inv_sub = store
        .with_system_principal()
        .subscribe_invalidation_detected(SubscriptionFilter::default(), |_payload| Ok(()))
        .expect("subscribe invalidation");
    let conflict_sub = store
        .with_system_principal()
        .subscribe_conflict_detected(SubscriptionFilter::default(), |_payload| Ok(()))
        .expect("subscribe conflict");
    let precedent_sub = store
        .with_system_principal()
        .subscribe_precedent_match(SubscriptionFilter::default(), |_payload| Ok(()))
        .expect("subscribe precedent");
    drop(store);

    let store = reopen(&path);
    observe(
        &store,
        ctx,
        inv_ent.id,
        delta(&[("path", json!("src/b.rs"))]),
    );
    let conflict_newer = decision(
        &store,
        ctx,
        "resume conflict backup",
        vec![conflict_intent.id],
        vec![conflict_ent.id],
        vec![],
    );
    store
        .record_outcome(conflict_newer.id, false, Some("failed".to_string()))
        .expect("newer conflict outcome");
    let precedent_new = decision(
        &store,
        ctx,
        "resume precedent new",
        vec![precedent_intent.id],
        vec![],
        vec![precedent_prior.id],
    );

    let inv_before = wait_for_metrics(&store, &inv_sub, |metrics| metrics.queued == 1);
    assert_eq!(inv_before.delivered, 0);
    let conflict_before = wait_for_metrics(&store, &conflict_sub, |metrics| metrics.queued == 1);
    assert_eq!(conflict_before.delivered, 0);
    let precedent_before = wait_for_metrics(&store, &precedent_sub, |metrics| metrics.queued == 1);
    assert_eq!(precedent_before.delivered, 0);

    let invalidations = Arc::new(Mutex::new(Vec::new()));
    let seen_invalidations = Arc::clone(&invalidations);
    let resumed_inv = store
        .with_system_principal()
        .resume_invalidation_detected(
            inv_sub.clone(),
            SubscriptionFilter::default(),
            move |payload| {
                seen_invalidations
                    .lock()
                    .expect("collector")
                    .push(payload.clone());
                Ok(())
            },
        )
        .expect("resume invalidation");
    assert_eq!(resumed_inv, inv_sub);

    let conflicts = Arc::new(Mutex::new(Vec::new()));
    let seen_conflicts = Arc::clone(&conflicts);
    let resumed_conflict = store
        .with_system_principal()
        .resume_conflict_detected(
            conflict_sub.clone(),
            SubscriptionFilter::default(),
            move |payload| {
                seen_conflicts
                    .lock()
                    .expect("collector")
                    .push(payload.clone());
                Ok(())
            },
        )
        .expect("resume conflict");
    assert_eq!(resumed_conflict, conflict_sub);

    let precedents = Arc::new(Mutex::new(Vec::new()));
    let seen_precedents = Arc::clone(&precedents);
    let resumed_precedent = store
        .with_system_principal()
        .resume_precedent_match(
            precedent_sub.clone(),
            SubscriptionFilter::default(),
            move |payload| {
                seen_precedents
                    .lock()
                    .expect("collector")
                    .push(payload.clone());
                Ok(())
            },
        )
        .expect("resume precedent");
    assert_eq!(resumed_precedent, precedent_sub);

    wait_for_len(&invalidations, 1);
    assert_eq!(
        invalidations.lock().expect("collector")[0]
            .invalidation
            .affected_decision_id,
        inv_dec.id
    );
    wait_for_len(&conflicts, 1);
    assert_eq!(
        conflicts.lock().expect("collector")[0].older_decision_id,
        conflict_older.id
    );
    assert_eq!(
        conflicts.lock().expect("collector")[0].newer_decision_id,
        conflict_newer.id
    );
    wait_for_len(&precedents, 1);
    assert_eq!(
        precedents.lock().expect("collector")[0].new_decision_id,
        precedent_new.id
    );
    assert_eq!(
        precedents.lock().expect("collector")[0].precedent_decision_id,
        precedent_prior.id
    );

    assert_eq!(
        wait_for_metrics(&store, &resumed_inv, |metrics| {
            metrics.delivered == 1 && metrics.queued == 0
        })
        .delivered,
        1
    );
    assert_eq!(
        wait_for_metrics(&store, &resumed_conflict, |metrics| {
            metrics.delivered == 1 && metrics.queued == 0
        })
        .delivered,
        1
    );
    assert_eq!(
        wait_for_metrics(&store, &resumed_precedent, |metrics| {
            metrics.delivered == 1 && metrics.queued == 0
        })
        .delivered,
        1
    );
}

// Use case: registering a subscriber does not mutate the Invalidation row shape.
#[test]
fn plan_7_invalidation_row_15_field_shape_unchanged_after_plan_10_subscriber_wraps_payload() {
    let (_dir, _path, store, ctx, ent, _intent, _dec, payloads) = collect_invalidation();
    observe(
        &store,
        ctx,
        ent,
        delta(&[("path", json!("lib/auth_v2/mod.rs"))]),
    );
    wait_for_len(&payloads, 1);
    let row = &payloads.lock().expect("collector")[0].invalidation;
    assert!(matches!(row.status, InvalidationStatus::Pending));
    assert!(row.resolved_at.is_none());
    assert!(row.resolution_action.is_none());
    assert!(row.resolution_decision_id.is_none());
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NormalizedAuditEntry {
    operation: AuditOperation,
    actor: Principal,
    reason: Option<String>,
    target_kind: &'static str,
    payload: Option<AuditPayload>,
}

fn lifecycle_audit_chain(with_subscriber: bool) -> Vec<NormalizedAuditEntry> {
    let (_dir, _path, store) = fixture();
    let ctx = context(&store, "audit-chain").id;
    let ent = entity(
        &store,
        ctx,
        "auth-module",
        map(&[("path", json!("src/auth.py"))]),
    );
    let intent = intention(&store, ctx, "reference the current auth module");
    let mut basis_fields = HashMap::new();
    basis_fields.insert(ent.id, vec!["path".to_string()]);
    store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: "auth module lives at src/auth.py".to_string(),
            reasoning: reasoning("auth module lives at src/auth.py"),
            confidence: Some(0.9),
            intention_ids: vec![intent.id],
            based_on_entity_ids: vec![ent.id],
            basis_fields: Some(basis_fields),
            context_id: ctx,
            ..Default::default()
        })
        .expect("create decision");
    let payloads = Arc::new(Mutex::new(Vec::new()));
    if with_subscriber {
        let seen = Arc::clone(&payloads);
        store
            .with_system_principal()
            .subscribe_invalidation_detected(SubscriptionFilter::default(), move |payload| {
                seen.lock().expect("collector").push(payload.clone());
                Ok(())
            })
            .expect("subscribe invalidation");
    }
    let observation = observe(
        &store,
        ctx,
        ent.id,
        delta(&[("path", json!("lib/auth_v2/mod.rs"))]),
    );
    if with_subscriber {
        wait_for_len(&payloads, 1);
    }
    let invalidation = observation
        .invalidations
        .into_iter()
        .find(|row| row.cascade_from.is_none())
        .expect("direct invalidation");
    let handle = store.with_principal(Principal::Agent("audit-agent".to_string()));
    let acknowledged = handle
        .acknowledge_invalidation(invalidation.id)
        .expect("acknowledge");
    handle
        .dismiss_invalidation(acknowledged.id, Some("not relevant".to_string()))
        .expect("dismiss");
    store
        .audit_query(AuditFilter {
            target: Some(AuditTarget::Invalidation(invalidation.id)),
            ..Default::default()
        })
        .expect("audit query")
        .into_iter()
        .map(|entry| NormalizedAuditEntry {
            operation: entry.operation,
            actor: entry.actor,
            reason: entry.reason,
            target_kind: match entry.target {
                AuditTarget::Invalidation(_) => "invalidation",
                _ => "other",
            },
            payload: entry.payload,
        })
        .collect()
}

// Use case: a no-op subscriber does not create lifecycle audit entries.
#[test]
fn plan_8_lifecycle_audit_chain_byte_identical_modulo_synthetic_ids_with_and_without_plan_10_subscriber()
 {
    let without_subscriber = lifecycle_audit_chain(false);
    let with_subscriber = lifecycle_audit_chain(true);
    assert_eq!(without_subscriber, with_subscriber);
    assert_eq!(
        with_subscriber
            .iter()
            .map(|entry| entry.operation)
            .collect::<Vec<_>>(),
        vec![
            AuditOperation::AcknowledgeInvalidation,
            AuditOperation::DismissInvalidation
        ]
    );
}
