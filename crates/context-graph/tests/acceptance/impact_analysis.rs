use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::Utc;
use context_graph::*;
use contextdb_core::{Error as EngineError, TxId, Value};
use contextdb_engine::{Database, TriggerAuditStatus};
use serde_json::json;

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        ..Default::default()
    })
    .expect("open");
    (dir, store)
}

fn reopen_store(dir: &tempfile::TempDir) -> Store {
    Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
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
        kind: EvidenceKind::Text,
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

fn entity(
    store: &Store,
    context_id: ContextId,
    name: &str,
    properties: BTreeMap<String, serde_json::Value>,
) -> Entity {
    store
        .create_entity(CreateEntity {
            entity_type: EntityType::File,
            name: name.to_string(),
            properties,
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

#[allow(clippy::too_many_arguments)]
fn decision(
    store: &Store,
    context_id: ContextId,
    description: &str,
    confidence: Option<f32>,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
    basis_fields: Option<HashMap<EntityId, Vec<String>>>,
    precedent_ids: Vec<DecisionId>,
) -> Decision {
    store
        .create_decision(CreateDecision {
            decision_type: "acceptance".to_string(),
            description: description.to_string(),
            confidence,
            intention_ids,
            based_on_entity_ids,
            basis_fields,
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
    changes: BTreeMap<String, Option<serde_json::Value>>,
) -> RecordObservationResult {
    let observation_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id,
            context_id,
            observation_type: "change".to_string(),
            source: "acceptance".to_string(),
            observed_at: Utc::now(),
            evidence: vec![evidence_ref(observation_id)],
            observed_properties: BTreeMap::new(),
            state_delta: changes,
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record observation")
}

fn assert_path_diff(
    invalidation: &Invalidation,
    field: &str,
    before: serde_json::Value,
    after: serde_json::Value,
) {
    let divergence = invalidation
        .basis_diff
        .divergences
        .iter()
        .find(|divergence| {
            divergence
                .value_diffs
                .iter()
                .any(|diff| diff.field == field)
        })
        .expect("field divergence");
    let diff = divergence
        .value_diffs
        .iter()
        .find(|diff| diff.field == field)
        .expect("value diff");
    assert_eq!(diff.basis_value, Some(before));
    assert_eq!(diff.current_value, Some(after));
}

fn setup_single_decision(
    confidence: Option<f32>,
    field: &str,
    value: serde_json::Value,
) -> (
    tempfile::TempDir,
    ContextId,
    EntityId,
    IntentionId,
    DecisionId,
) {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "impact");
    let ent = entity(&store, ctx.id, "src/auth.py", props(&[(field, value)]));
    let intent = intention(&store, ctx.id, "Reference current code structure");
    let mut basis_fields = HashMap::new();
    basis_fields.insert(ent.id, vec![field.to_string()]);
    let dec = decision(
        &store,
        ctx.id,
        "auth module is at src/auth.py",
        confidence,
        vec![intent.id],
        vec![ent.id],
        Some(basis_fields),
        Vec::new(),
    );
    drop(store);
    (dir, ctx.id, ent.id, intent.id, dec.id)
}

// The agent should see stale reasoning before quoting an old file path.
#[test]
fn invalidation_created_by_engine_on_basis_change() {
    let (dir, ctx, entity_id, intention_id, decision_id) =
        setup_single_decision(None, "path", json!("src/auth.py"));
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx,
        entity_id,
        delta(&[("path", json!("src/authentication.py"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    let invalidation = &result.invalidations[0];
    assert_eq!(invalidation.affected_decision_id, decision_id);
    assert_eq!(invalidation.trigger_observation_id, result.observation_id);
    assert_eq!(invalidation.severity, InvalidationSeverity::Warning);
    assert_eq!(invalidation.cascade_from, None);
    assert_eq!(invalidation.threatened_intention_ids, vec![intention_id]);
    assert!(invalidation.intention_now_un_served);
    assert!(invalidation.basis_diff.is_invalidated_candidate);
    assert_path_diff(
        invalidation,
        "path",
        json!("src/auth.py"),
        json!("src/authentication.py"),
    );
}

// A config change should be traced to the engine trigger, not a caller-side scan.
#[test]
fn impact_analysis_direct_invalidation_by_engine() {
    let (dir, ctx, entity_id, _intention_id, decision_id) =
        setup_single_decision(Some(0.9), "region", json!("region-a"));
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx,
        entity_id,
        delta(&[("region", json!("region-b"))]),
    );
    let observation = store
        .get_observation(result.observation_id)
        .expect("observation");

    assert_eq!(result.invalidations.len(), 1);
    assert_eq!(result.invalidations[0].affected_decision_id, decision_id);
    let persisted = store
        .list_invalidations(ctx, Some(decision_id), Some(InvalidationStatus::Pending))
        .expect("list invalidations");
    assert_eq!(
        persisted.iter().map(|row| row.id).collect::<Vec<_>>(),
        result
            .invalidations
            .iter()
            .map(|row| row.id)
            .collect::<Vec<_>>()
    );

    drop(store);
    let db = open_raw_database(&dir);
    let fires = db
        .trigger_audit_log()
        .into_iter()
        .filter(|entry| {
            entry.trigger_name == "impact_analysis_trigger"
                && entry.firing_tx == observation.created_at
                && entry.status == TriggerAuditStatus::Fired
        })
        .count();
    assert_eq!(fires, 1);
}

// Irrelevant field changes should stay quiet, while a basis field change should still fire.
#[test]
fn impact_analysis_no_basis_change_no_invalidation() {
    let (dir, ctx, entity_id, _intention_id, _decision_id) =
        setup_single_decision(Some(0.7), "region", json!("us-east-1"));
    let store = reopen_store(&dir);

    let irrelevant = observe(
        &store,
        ctx,
        entity_id,
        delta(&[("instance_type", json!("m6.large"))]),
    );
    assert!(irrelevant.invalidations.is_empty());
    assert!(
        store
            .list_invalidations(ctx, None, Some(InvalidationStatus::Pending))
            .expect("list")
            .is_empty()
    );

    let relevant = observe(
        &store,
        ctx,
        entity_id,
        delta(&[("region", json!("eu-west-1"))]),
    );
    assert_eq!(relevant.invalidations.len(), 1);
}

// A precedent decision should be warned when the direct decision it cites becomes suspect.
#[test]
fn impact_analysis_cascade_via_cites() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "cascade");
    let ent = entity(
        &store,
        ctx.id,
        "database",
        props(&[("tier", json!("primary"))]),
    );
    let i1 = intention(&store, ctx.id, "Keep database routing reliable");
    let i2 = intention(&store, ctx.id, "Keep backup procedures aligned");
    let d1 = decision(
        &store,
        ctx.id,
        "route writes to primary",
        Some(0.9),
        vec![i1.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    let d2 = decision(
        &store,
        ctx.id,
        "backup playbook cites primary routing",
        Some(0.7),
        vec![i2.id],
        Vec::new(),
        None,
        vec![d1.id],
    );
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(&store, ctx.id, ent.id, delta(&[("tier", json!("replica"))]));

    assert_eq!(result.invalidations.len(), 2);
    let direct = result
        .invalidations
        .iter()
        .find(|row| row.cascade_from.is_none())
        .expect("direct row");
    let cascade = result
        .invalidations
        .iter()
        .find(|row| row.cascade_from == Some(d1.id))
        .expect("cascade row");
    assert_eq!(direct.affected_decision_id, d1.id);
    assert_eq!(direct.severity, InvalidationSeverity::Critical);
    assert_eq!(direct.threatened_intention_ids, vec![i1.id]);
    assert!(!direct.basis_diff.divergences.is_empty());
    assert_eq!(cascade.affected_decision_id, d2.id);
    assert_eq!(cascade.severity, InvalidationSeverity::Info);
    assert_eq!(cascade.threatened_intention_ids, vec![i2.id]);
    assert!(cascade.basis_diff.is_invalidated_candidate);
    assert!(cascade.basis_diff.divergences.is_empty());
}

// A moderate-confidence decision should produce a Warning with the changed field pinned.
#[test]
fn impact_analysis_severity_warning_for_mid_confidence() {
    let (dir, ctx, entity_id, intention_id, decision_id) =
        setup_single_decision(Some(0.65), "region", json!("us-east-1"));
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx,
        entity_id,
        delta(&[("region", json!("eu-west-1"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    let row = &result.invalidations[0];
    assert_eq!(row.severity, InvalidationSeverity::Warning);
    assert_eq!(row.affected_decision_id, decision_id);
    assert_eq!(row.cascade_from, None);
    assert_eq!(row.threatened_intention_ids, vec![intention_id]);
    assert_path_diff(row, "region", json!("us-east-1"), json!("eu-west-1"));
}

// A low-confidence decision should stay informational but still identify the exact drift.
#[test]
fn impact_analysis_severity_info_for_low_confidence() {
    let (dir, ctx, entity_id, intention_id, decision_id) =
        setup_single_decision(Some(0.3), "region", json!("us-east-1"));
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx,
        entity_id,
        delta(&[("region", json!("eu-west-1"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    let row = &result.invalidations[0];
    assert_eq!(row.severity, InvalidationSeverity::Info);
    assert_eq!(row.affected_decision_id, decision_id);
    assert_eq!(row.cascade_from, None);
    assert_eq!(row.threatened_intention_ids, vec![intention_id]);
    assert_path_diff(row, "region", json!("us-east-1"), json!("eu-west-1"));
}

// An unrated decision should be treated as important enough to revisit.
#[test]
fn impact_analysis_severity_warning_for_unrated_decision() {
    let (dir, ctx, entity_id, intention_id, decision_id) =
        setup_single_decision(None, "path", json!("src/auth.py"));
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx,
        entity_id,
        delta(&[("path", json!("src/authentication.py"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    let row = &result.invalidations[0];
    assert_eq!(row.severity, InvalidationSeverity::Warning);
    assert_eq!(row.affected_decision_id, decision_id);
    assert_eq!(row.threatened_intention_ids, vec![intention_id]);
    assert_path_diff(
        row,
        "path",
        json!("src/auth.py"),
        json!("src/authentication.py"),
    );
}

// A stronger active sibling should prevent a high-confidence decision from being Critical.
#[test]
fn impact_analysis_severity_downgraded_when_higher_confidence_sibling_exists() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "downgrade");
    let ent = entity(
        &store,
        ctx.id,
        "payment-service",
        props(&[("region", json!("us-east-1"))]),
    );
    let intent = intention(&store, ctx.id, "Payment service latency stays within SLA");
    let d1 = decision(
        &store,
        ctx.id,
        "alert at 200ms",
        Some(0.85),
        vec![intent.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    let _d2 = decision(
        &store,
        ctx.id,
        "alert at 250ms",
        Some(0.9),
        vec![intent.id],
        Vec::new(),
        None,
        Vec::new(),
    );
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx.id,
        ent.id,
        delta(&[("region", json!("eu-west-1"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    let row = &result.invalidations[0];
    assert_eq!(row.severity, InvalidationSeverity::Warning);
    assert_eq!(row.affected_decision_id, d1.id);
    assert_eq!(row.threatened_intention_ids, vec![intent.id]);
    assert!(!row.intention_now_un_served);
    assert_path_diff(row, "region", json!("us-east-1"), json!("eu-west-1"));
}

// A sole high-confidence decision should be Critical and mark its goal unserved.
#[test]
fn impact_analysis_severity_critical_when_sole_active_decision_serving_intention() {
    let (dir, ctx, entity_id, intention_id, decision_id) =
        setup_single_decision(Some(0.9), "region", json!("us-east-1"));
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx,
        entity_id,
        delta(&[("region", json!("eu-west-1"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    let row = &result.invalidations[0];
    assert_eq!(row.severity, InvalidationSeverity::Critical);
    assert!(row.intention_now_un_served);
    assert_eq!(row.threatened_intention_ids, vec![intention_id]);
    assert_eq!(row.affected_decision_id, decision_id);
    assert_eq!(row.cascade_from, None);
    assert_path_diff(row, "region", json!("us-east-1"), json!("eu-west-1"));
}

// Cross-context SERVES edges must not mask a high-confidence decision becoming unserved.
#[test]
fn impact_analysis_severity_siblings_are_context_scoped() {
    let (dir, store) = fresh_store();
    let ctx_a = context(&store, "severity-a");
    let ctx_b = context(&store, "severity-b");
    let ent_a = entity(
        &store,
        ctx_a.id,
        "service-a",
        props(&[("region", json!("us-east-1"))]),
    );
    let intent_a = intention(&store, ctx_a.id, "Keep A region current");
    let intent_b = intention(&store, ctx_b.id, "Keep B region current");
    let d_a = decision(
        &store,
        ctx_a.id,
        "A region",
        Some(0.9),
        vec![intent_a.id],
        vec![ent_a.id],
        None,
        Vec::new(),
    );
    let d_b = decision(
        &store,
        ctx_b.id,
        "B region",
        Some(0.95),
        vec![intent_b.id],
        Vec::new(),
        None,
        Vec::new(),
    );
    drop(store);
    let db = open_raw_database(&dir);
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $properties)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(uuid::Uuid::now_v7())),
            ("source_id".to_string(), Value::Uuid(d_b.id.as_uuid())),
            ("target_id".to_string(), Value::Uuid(intent_a.id.as_uuid())),
            ("edge_type".to_string(), Value::Text("SERVES".to_string())),
            ("context_id".to_string(), Value::Uuid(ctx_b.id.as_uuid())),
            ("valid_from".to_string(), Value::TxId(TxId(1))),
            ("properties".to_string(), Value::Json(json!({}))),
        ]),
    )
    .expect("cross-context serves edge");
    drop(db);
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx_a.id,
        ent_a.id,
        delta(&[("region", json!("eu-west-1"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    let row = &result.invalidations[0];
    assert_eq!(row.affected_decision_id, d_a.id);
    assert_eq!(row.severity, InvalidationSeverity::Critical);
    assert!(row.intention_now_un_served);
    assert_eq!(row.threatened_intention_ids, vec![intent_a.id]);
}

// Retired decisions should not be re-opened by a new observation.
#[test]
fn impact_analysis_respects_decision_status() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "status");
    let ent = entity(&store, ctx.id, "service", props(&[("version", json!("1"))]));
    let intent = intention(&store, ctx.id, "Keep service routing current");
    let d1 = decision(
        &store,
        ctx.id,
        "old route",
        Some(0.9),
        vec![intent.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    let d2 = store
        .supersede_decision(
            d1.id,
            CreateDecision {
                decision_type: "acceptance".to_string(),
                description: "new route".to_string(),
                confidence: Some(0.9),
                intention_ids: vec![intent.id],
                based_on_entity_ids: vec![ent.id],
                basis_fields: None,
                context_id: ctx.id,
                ..Default::default()
            },
        )
        .expect("supersede");
    let d3 = decision(
        &store,
        ctx.id,
        "archived route",
        Some(0.8),
        vec![intent.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    store
        .archive_decision(d3.id, "retired".to_string())
        .expect("archive");
    let d4 = decision(
        &store,
        ctx.id,
        "other active route",
        Some(0.85),
        vec![intent.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    let d5 = decision(
        &store,
        ctx.id,
        "manually invalidated route",
        Some(0.8),
        vec![intent.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    let marker = entity(&store, ctx.id, "manual-marker", BTreeMap::new());
    let prior_observation = observe(
        &store,
        ctx.id,
        marker.id,
        delta(&[("owner", json!("team-a"))]),
    );
    store
        .invalidate_decision(d5.id, prior_observation.observation_id, "basis changed")
        .expect("manual invalidation");
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(&store, ctx.id, ent.id, delta(&[("version", json!("2"))]));
    let affected = result
        .invalidations
        .iter()
        .map(|row| row.affected_decision_id)
        .collect::<HashSet<_>>();

    assert_eq!(result.invalidations.len(), 2);
    assert_eq!(affected, HashSet::from([d2.id, d4.id]));
    assert!(!affected.contains(&d1.id));
    assert!(!affected.contains(&d3.id));
    assert!(!affected.contains(&d5.id));
}

// Observation facts should reject mutation and preserve their stored bytes.
#[test]
fn impact_analysis_observation_immutable() {
    let (dir, ctx, entity_id, _intention_id, _decision_id) =
        setup_single_decision(Some(0.9), "region", json!("us-east-1"));
    let store = reopen_store(&dir);
    let result = observe(&store, ctx, entity_id, delta(&[("owner", json!("team-a"))]));
    drop(store);

    let db = open_raw_database(&dir);
    let before = db
        .scan("observations", db.snapshot())
        .expect("scan before")
        .into_iter()
        .find(|row| row.values.get("id") == Some(&Value::Uuid(result.observation_id.as_uuid())))
        .expect("observation before");
    let update = db.execute(
        "UPDATE observations SET state_delta = $state_delta WHERE id = $id",
        &HashMap::from([
            (
                "id".to_string(),
                Value::Uuid(result.observation_id.as_uuid()),
            ),
            (
                "state_delta".to_string(),
                Value::Json(json!({"region": "hacked"})),
            ),
        ]),
    );
    assert!(matches!(
        update,
        Err(EngineError::ImmutableTable(ref table)) if table == "observations"
    ));
    let after = db
        .scan("observations", db.snapshot())
        .expect("scan after")
        .into_iter()
        .find(|row| row.values.get("id") == Some(&Value::Uuid(result.observation_id.as_uuid())))
        .expect("observation after");
    assert_eq!(before.values, after.values);
}

// A newly observed field should still threaten decisions that relied on the whole entity.
#[test]
fn impact_analysis_new_field_triggers_invalidation() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "new-field");
    let ent = entity(
        &store,
        ctx.id,
        "service",
        props(&[("region", json!("us-east-1"))]),
    );
    let intent = intention(&store, ctx.id, "Keep service policy current");
    let dec = decision(
        &store,
        ctx.id,
        "policy based on whole service",
        Some(0.9),
        vec![intent.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx.id,
        ent.id,
        delta(&[("security_zone", json!("restricted"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    let row = &result.invalidations[0];
    assert_eq!(row.affected_decision_id, dec.id);
    let diff = &row.basis_diff.divergences[0].value_diffs[0];
    assert_eq!(diff.field, "security_zone");
    assert_eq!(diff.basis_value, None);
    assert_eq!(diff.current_value, Some(json!("restricted")));
}

// Informational relationships should not become hidden invalidation inputs.
#[test]
fn impact_analysis_about_edge_does_not_trigger() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "about");
    let ent = entity(&store, ctx.id, "service", props(&[("version", json!("1"))]));
    let intent = intention(&store, ctx.id, "Keep service understanding current");
    let d1 = decision(
        &store,
        ctx.id,
        "based on service version",
        Some(0.8),
        vec![intent.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    let d2 = decision(
        &store,
        ctx.id,
        "mentions service",
        Some(0.8),
        vec![intent.id],
        Vec::new(),
        None,
        Vec::new(),
    );
    drop(store);
    let db = open_raw_database(&dir);
    db.execute(
        "INSERT INTO edges (id, source_id, target_id, edge_type, context_id, valid_from, properties) VALUES ($id, $source_id, $target_id, $edge_type, $context_id, $valid_from, $properties)",
        &HashMap::from([
            ("id".to_string(), Value::Uuid(uuid::Uuid::now_v7())),
            ("source_id".to_string(), Value::Uuid(d2.id.as_uuid())),
            ("target_id".to_string(), Value::Uuid(ent.id.as_uuid())),
            ("edge_type".to_string(), Value::Text("ABOUT".to_string())),
            ("context_id".to_string(), Value::Uuid(ctx.id.as_uuid())),
            ("valid_from".to_string(), Value::TxId(TxId(1))),
            ("properties".to_string(), Value::Json(json!({}))),
        ]),
    ).expect("about edge");
    drop(db);
    let store = reopen_store(&dir);

    let result = observe(&store, ctx.id, ent.id, delta(&[("version", json!("2"))]));

    assert_eq!(result.invalidations.len(), 1);
    assert_eq!(result.invalidations[0].affected_decision_id, d1.id);
    assert!(
        !result
            .invalidations
            .iter()
            .any(|row| row.affected_decision_id == d2.id)
    );
}

// Cold-start observations should be silent, but the same entity should fire once a decision exists.
#[test]
fn impact_analysis_silent_when_no_active_decisions_cite_entity() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "cold-start");
    let ent = entity(
        &store,
        ctx.id,
        "config",
        props(&[("timeout_ms", json!(250))]),
    );
    drop(store);
    let store = reopen_store(&dir);

    let first = observe(&store, ctx.id, ent.id, delta(&[("timeout_ms", json!(500))]));
    assert!(first.invalidations.is_empty());

    let intent = intention(&store, ctx.id, "Keep timeout policy current");
    decision(
        &store,
        ctx.id,
        "timeout is 500ms",
        Some(0.8),
        vec![intent.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    let second = observe(&store, ctx.id, ent.id, delta(&[("timeout_ms", json!(750))]));
    assert_eq!(second.invalidations.len(), 1);
}

// A same-looking entity in another context must not receive this context's invalidation.
#[test]
fn impact_analysis_per_context_isolation() {
    let (dir, store) = fresh_store();
    let ctx_a = context(&store, "ctx-a");
    let ctx_b = context(&store, "ctx-b");
    let ent_a = entity(
        &store,
        ctx_a.id,
        "service",
        props(&[("region", json!("us-east-1"))]),
    );
    let ent_b = entity(
        &store,
        ctx_b.id,
        "service",
        props(&[("region", json!("us-east-1"))]),
    );
    let i_a = intention(&store, ctx_a.id, "Keep A current");
    let i_b = intention(&store, ctx_b.id, "Keep B current");
    let d_a = decision(
        &store,
        ctx_a.id,
        "A region",
        Some(0.8),
        vec![i_a.id],
        vec![ent_a.id],
        None,
        Vec::new(),
    );
    let _d_b = decision(
        &store,
        ctx_b.id,
        "B region",
        Some(0.8),
        vec![i_b.id],
        vec![ent_b.id],
        None,
        Vec::new(),
    );
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx_a.id,
        ent_a.id,
        delta(&[("region", json!("eu-west-1"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    assert_eq!(result.invalidations[0].affected_decision_id, d_a.id);
    assert!(
        store
            .list_invalidations(ctx_b.id, None, Some(InvalidationStatus::Pending))
            .expect("ctx b invalidations")
            .is_empty()
    );
}

// A listed basis field that did not actually change should stay quiet.
#[test]
fn impact_analysis_basis_field_listed_but_unchanged_no_invalidation() {
    let (dir, ctx, entity_id, _intention_id, _decision_id) =
        setup_single_decision(Some(0.8), "region", json!("us-east-1"));
    let store = reopen_store(&dir);

    let unchanged = observe(
        &store,
        ctx,
        entity_id,
        delta(&[("region", json!("us-east-1"))]),
    );
    assert!(unchanged.invalidations.is_empty());
    let changed = observe(
        &store,
        ctx,
        entity_id,
        delta(&[("region", json!("eu-west-1"))]),
    );
    assert_eq!(changed.invalidations.len(), 1);
}

// A field named in the basis list should fire on first appearance.
#[test]
fn impact_analysis_basis_fields_listed_field_absent_at_decision_time_triggers_on_first_appearance()
{
    let (dir, store) = fresh_store();
    let ctx = context(&store, "listed-new-field");
    let ent = entity(
        &store,
        ctx.id,
        "service",
        props(&[("region", json!("us-east-1"))]),
    );
    let intent = intention(&store, ctx.id, "Keep service safety current");
    let mut basis_fields = HashMap::new();
    basis_fields.insert(ent.id, vec!["security_zone".to_string()]);
    let dec = decision(
        &store,
        ctx.id,
        "security posture decision",
        Some(0.9),
        vec![intent.id],
        vec![ent.id],
        Some(basis_fields),
        Vec::new(),
    );
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx.id,
        ent.id,
        delta(&[("security_zone", json!("restricted"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    assert_eq!(result.invalidations[0].affected_decision_id, dec.id);
    assert_eq!(
        result.invalidations[0].severity,
        InvalidationSeverity::Critical
    );
}

// Critical severity must consider every threatened goal, not only the first one.
#[test]
fn impact_analysis_severity_critical_when_sole_server_on_at_least_one_threatened_intention() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "multi-intention");
    let ent = entity(
        &store,
        ctx.id,
        "service",
        props(&[("region", json!("us-east-1"))]),
    );
    let i1 = intention(&store, ctx.id, "Keep latency current");
    let i2 = intention(&store, ctx.id, "Keep failover current");
    let d = decision(
        &store,
        ctx.id,
        "shared decision",
        Some(0.9),
        vec![i1.id, i2.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    let _higher_i1 = decision(
        &store,
        ctx.id,
        "strong latency sibling",
        Some(0.95),
        vec![i1.id],
        Vec::new(),
        None,
        Vec::new(),
    );
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx.id,
        ent.id,
        delta(&[("region", json!("eu-west-1"))]),
    );

    assert_eq!(result.invalidations.len(), 1);
    let row = &result.invalidations[0];
    assert_eq!(row.affected_decision_id, d.id);
    assert_eq!(row.severity, InvalidationSeverity::Critical);
    assert!(row.intention_now_un_served);
    assert_eq!(
        row.threatened_intention_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>(),
        HashSet::from([i1.id, i2.id])
    );
}

// A retired decision in the middle of a citation chain should stop that branch.
#[test]
fn impact_analysis_cascade_skips_non_active_decisions_and_does_not_descend_through_them() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "cascade-status");
    let ent = entity(&store, ctx.id, "module", props(&[("exists", json!(true))]));
    let intent = intention(&store, ctx.id, "Keep module reasoning current");
    let d1 = decision(
        &store,
        ctx.id,
        "module exists",
        Some(0.9),
        vec![intent.id],
        vec![ent.id],
        None,
        Vec::new(),
    );
    let d2 = decision(
        &store,
        ctx.id,
        "cites module decision",
        Some(0.7),
        vec![intent.id],
        Vec::new(),
        None,
        vec![d1.id],
    );
    let d3 = decision(
        &store,
        ctx.id,
        "cites middle decision",
        Some(0.7),
        vec![intent.id],
        Vec::new(),
        None,
        vec![d2.id],
    );
    store
        .archive_decision(d2.id, "retired".to_string())
        .expect("archive middle");
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(&store, ctx.id, ent.id, delta(&[("exists", json!(false))]));
    let affected = result
        .invalidations
        .iter()
        .map(|row| row.affected_decision_id)
        .collect::<HashSet<_>>();

    assert_eq!(result.invalidations.len(), 1);
    assert!(affected.contains(&d1.id));
    assert!(!affected.contains(&d2.id));
    assert!(!affected.contains(&d3.id));
}
