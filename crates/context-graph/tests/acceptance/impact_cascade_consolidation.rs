use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::Utc;
use context_graph::*;
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
        source_ref: "cas://sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .to_string(),
        content_hash: Some(
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
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
            entity_type: EntityType::Service,
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

fn basis_fields(entity_id: EntityId, fields: &[&str]) -> HashMap<EntityId, Vec<String>> {
    HashMap::from([(
        entity_id,
        fields.iter().map(|field| (*field).to_string()).collect(),
    )])
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

fn update_entity_properties(
    store: &Store,
    entity_id: EntityId,
    properties: BTreeMap<String, serde_json::Value>,
) -> Entity {
    store
        .update_entity(
            entity_id,
            EntityPatch {
                properties: Some(properties),
                ..Default::default()
            },
        )
        .expect("update entity")
}

fn row_for(invalidations: &[Invalidation], decision_id: DecisionId) -> &Invalidation {
    invalidations
        .iter()
        .find(|row| row.affected_decision_id == decision_id)
        .expect("invalidation row")
}

#[allow(clippy::too_many_arguments)]
fn assert_emitted_row(
    row: &Invalidation,
    context_id: ContextId,
    decision_id: DecisionId,
    cascade_from: Option<DecisionId>,
    severity: InvalidationSeverity,
    threatened_intention_ids: Vec<IntentionId>,
    intention_now_un_served: bool,
) {
    assert_eq!(row.context_id, context_id);
    assert_eq!(row.affected_decision_id, decision_id);
    assert_eq!(row.status, InvalidationStatus::Pending);
    assert_eq!(row.cascade_from, cascade_from);
    assert_eq!(row.severity, severity);
    assert_eq!(row.threatened_intention_ids, threatened_intention_ids);
    assert_eq!(row.intention_now_un_served, intention_now_un_served);
}

fn assert_one_basis_diff(
    row: &Invalidation,
    entity_id: EntityId,
    basis_snapshot_id: EntitySnapshotId,
    field: &str,
    basis_value: serde_json::Value,
    current_value: serde_json::Value,
) {
    assert!(row.basis_diff.is_invalidated_candidate);
    assert_eq!(row.basis_diff.divergences.len(), 1);
    let divergence = &row.basis_diff.divergences[0];
    assert_eq!(divergence.entity_id, entity_id);
    assert_eq!(divergence.basis_snapshot_id, basis_snapshot_id);
    assert_eq!(divergence.value_diffs.len(), 1);
    let diff = &divergence.value_diffs[0];
    assert_eq!(diff.field, field);
    assert_eq!(diff.basis_value, Some(basis_value));
    assert_eq!(diff.current_value, Some(current_value));
}

#[test]
fn impact_analysis_two_root_rows_attribute_their_own_basis() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "two-root-basis");
    let observed = entity(
        &store,
        ctx.id,
        "deployment-target",
        props(&[("region", json!("us-east-1")), ("rollout", json!("stable"))]),
    );
    let region_intent = intention(&store, ctx.id, "Keep region routing current");
    let region_decision = decision(
        &store,
        ctx.id,
        "traffic is routed through us-east-1",
        Some(0.9),
        vec![region_intent.id],
        vec![observed.id],
        Some(basis_fields(observed.id, &["region"])),
        Vec::new(),
    );
    update_entity_properties(
        &store,
        observed.id,
        props(&[("region", json!("eu-west-1")), ("rollout", json!("canary"))]),
    );
    let rollout_intent = intention(&store, ctx.id, "Keep rollout policy current");
    let rollout_decision = decision(
        &store,
        ctx.id,
        "canary rollout is safe",
        Some(0.9),
        vec![rollout_intent.id],
        vec![observed.id],
        Some(basis_fields(observed.id, &["rollout"])),
        Vec::new(),
    );
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx.id,
        observed.id,
        delta(&[("region", json!("ap-south-1")), ("rollout", json!("blue"))]),
    );

    assert_eq!(result.invalidations.len(), 2);
    assert!(
        result
            .invalidations
            .iter()
            .all(|row| row.cascade_from.is_none())
    );

    let region_row = row_for(&result.invalidations, region_decision.id);
    assert_emitted_row(
        region_row,
        ctx.id,
        region_decision.id,
        None,
        InvalidationSeverity::Critical,
        vec![region_intent.id],
        true,
    );
    assert_one_basis_diff(
        region_row,
        observed.id,
        region_decision.based_on_snapshots[0],
        "region",
        json!("us-east-1"),
        json!("ap-south-1"),
    );

    let rollout_row = row_for(&result.invalidations, rollout_decision.id);
    assert_emitted_row(
        rollout_row,
        ctx.id,
        rollout_decision.id,
        None,
        InvalidationSeverity::Critical,
        vec![rollout_intent.id],
        true,
    );
    assert_one_basis_diff(
        rollout_row,
        observed.id,
        rollout_decision.based_on_snapshots[0],
        "rollout",
        json!("canary"),
        json!("blue"),
    );
}

#[test]
fn impact_analysis_innocent_observation_zero_rows_and_unchanged_state() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "innocent-heavy");
    let mut sampled_entity_id = None;
    for index in 0..50 {
        let unrelated = entity(
            &store,
            ctx.id,
            &format!("unrelated-service-{index}"),
            props(&[("region", json!("us-east-1")), ("version", json!(index))]),
        );
        if index == 17 {
            sampled_entity_id = Some(unrelated.id);
        }
        let unrelated_intent = intention(
            &store,
            ctx.id,
            &format!("Keep unrelated service {index} current"),
        );
        decision(
            &store,
            ctx.id,
            &format!("unrelated service {index} stays in us-east-1"),
            Some(0.8),
            vec![unrelated_intent.id],
            vec![unrelated.id],
            Some(basis_fields(unrelated.id, &["region"])),
            Vec::new(),
        );
    }
    let observed = entity(
        &store,
        ctx.id,
        "unused-config",
        props(&[("version", json!(1)), ("owner", json!("platform"))]),
    );
    let inactive_intent = intention(&store, ctx.id, "Keep unused config documented");
    let archived = decision(
        &store,
        ctx.id,
        "unused config version one",
        Some(0.7),
        vec![inactive_intent.id],
        vec![observed.id],
        Some(basis_fields(observed.id, &["version"])),
        Vec::new(),
    );
    store
        .archive_decision(archived.id, "unused config retired".to_string())
        .expect("archive");
    let superseded = decision(
        &store,
        ctx.id,
        "unused config owner platform",
        Some(0.7),
        vec![inactive_intent.id],
        vec![observed.id],
        Some(basis_fields(observed.id, &["owner"])),
        Vec::new(),
    );
    store
        .supersede_decision(
            superseded.id,
            CreateDecision {
                decision_type: "acceptance".to_string(),
                description: "unused config now tracked elsewhere".to_string(),
                confidence: Some(0.7),
                intention_ids: vec![inactive_intent.id],
                context_id: ctx.id,
                ..Default::default()
            },
        )
        .expect("supersede");
    let sampled_entity_id = sampled_entity_id.expect("sampled entity");
    drop(store);
    let store = reopen_store(&dir);
    let unrelated_before = store
        .current_entity_snapshot(sampled_entity_id)
        .expect("unrelated current before")
        .expect("unrelated snapshot before");

    let result = observe(
        &store,
        ctx.id,
        observed.id,
        delta(&[("version", json!(2)), ("owner", json!("sre"))]),
    );

    assert!(result.invalidations.is_empty());
    assert!(
        store
            .list_invalidations(ctx.id, None, Some(InvalidationStatus::Pending))
            .expect("pending invalidations")
            .is_empty()
    );
    let unrelated_after = store
        .current_entity_snapshot(sampled_entity_id)
        .expect("unrelated current after")
        .expect("unrelated snapshot after");
    assert_eq!(unrelated_after, unrelated_before);

    let observed_after = store
        .current_entity_snapshot(observed.id)
        .expect("observed current")
        .expect("observed snapshot");
    assert_eq!(result.snapshot_id, Some(observed_after.id));
    assert_eq!(observed_after.properties.get("version"), Some(&json!(2)));
    assert_eq!(observed_after.properties.get("owner"), Some(&json!("sre")));
}

#[test]
fn impact_analysis_direct_only_zero_cites_no_cascade() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "direct-only");
    let observed = entity(
        &store,
        ctx.id,
        "config-file",
        props(&[("version", json!(1))]),
    );
    let intent = intention(&store, ctx.id, "Keep config consumers current");
    let direct = decision(
        &store,
        ctx.id,
        "config version one is deployed",
        Some(0.9),
        vec![intent.id],
        vec![observed.id],
        Some(basis_fields(observed.id, &["version"])),
        Vec::new(),
    );
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(&store, ctx.id, observed.id, delta(&[("version", json!(2))]));

    assert_eq!(result.invalidations.len(), 1);
    let row = &result.invalidations[0];
    assert_emitted_row(
        row,
        ctx.id,
        direct.id,
        None,
        InvalidationSeverity::Critical,
        vec![intent.id],
        true,
    );
    assert_one_basis_diff(
        row,
        observed.id,
        direct.based_on_snapshots[0],
        "version",
        json!(1),
        json!(2),
    );
    assert!(
        result
            .invalidations
            .iter()
            .all(|row| row.cascade_from.is_none())
    );
}

#[test]
fn impact_analysis_deep_active_behind_non_active_not_invalidated() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "deep-non-active");
    let observed = entity(&store, ctx.id, "module", props(&[("enabled", json!(true))]));
    let root_intent = intention(&store, ctx.id, "Keep module availability current");
    let middle_intent = intention(&store, ctx.id, "Keep module runbook current");
    let deep_intent = intention(&store, ctx.id, "Keep escalation policy current");
    let root = decision(
        &store,
        ctx.id,
        "module is enabled",
        Some(0.9),
        vec![root_intent.id],
        vec![observed.id],
        Some(basis_fields(observed.id, &["enabled"])),
        Vec::new(),
    );
    let middle_one = decision(
        &store,
        ctx.id,
        "runbook cites enabled module",
        Some(0.7),
        vec![middle_intent.id],
        Vec::new(),
        None,
        vec![root.id],
    );
    let middle_two = decision(
        &store,
        ctx.id,
        "handoff cites old runbook",
        Some(0.7),
        vec![middle_intent.id],
        Vec::new(),
        None,
        vec![middle_one.id],
    );
    let deep = decision(
        &store,
        ctx.id,
        "active escalation cites handoff",
        Some(0.7),
        vec![deep_intent.id],
        Vec::new(),
        None,
        vec![middle_two.id],
    );
    store
        .archive_decision(middle_one.id, "runbook retired".to_string())
        .expect("archive middle one");
    store
        .archive_decision(middle_two.id, "handoff retired".to_string())
        .expect("archive middle two");
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx.id,
        observed.id,
        delta(&[("enabled", json!(false))]),
    );
    let affected = result
        .invalidations
        .iter()
        .map(|row| row.affected_decision_id)
        .collect::<HashSet<_>>();

    assert_eq!(result.invalidations.len(), 1);
    assert!(affected.contains(&root.id));
    assert!(!affected.contains(&middle_one.id));
    assert!(!affected.contains(&middle_two.id));
    assert!(!affected.contains(&deep.id));
    let root_row = row_for(&result.invalidations, root.id);
    assert_emitted_row(
        root_row,
        ctx.id,
        root.id,
        None,
        InvalidationSeverity::Critical,
        vec![root_intent.id],
        true,
    );
    assert_one_basis_diff(
        root_row,
        observed.id,
        root.based_on_snapshots[0],
        "enabled",
        json!(true),
        json!(false),
    );
}

#[test]
fn cascade_row_threatens_citing_decisions_own_intentions_under_fan_in() {
    let (dir, store) = fresh_store();
    let ctx = context(&store, "fan-in-intentions");
    let observed = entity(
        &store,
        ctx.id,
        "region-config",
        props(&[("region", json!("us-east-1"))]),
    );
    let root_one_intent = intention(&store, ctx.id, "Keep primary route current");
    let root_two_intent = intention(&store, ctx.id, "Keep backup route current");
    let cascade_intent = intention(&store, ctx.id, "Keep failover decision current");
    let root_one = decision(
        &store,
        ctx.id,
        "primary route uses us-east-1",
        Some(0.9),
        vec![root_one_intent.id],
        vec![observed.id],
        Some(basis_fields(observed.id, &["region"])),
        Vec::new(),
    );
    let root_two = decision(
        &store,
        ctx.id,
        "backup route uses us-east-1",
        Some(0.9),
        vec![root_two_intent.id],
        vec![observed.id],
        Some(basis_fields(observed.id, &["region"])),
        Vec::new(),
    );
    let citing = decision(
        &store,
        ctx.id,
        "failover cites both route decisions",
        Some(0.7),
        vec![cascade_intent.id],
        Vec::new(),
        None,
        vec![root_one.id, root_two.id],
    );
    drop(store);
    let store = reopen_store(&dir);

    let result = observe(
        &store,
        ctx.id,
        observed.id,
        delta(&[("region", json!("eu-west-1"))]),
    );

    assert_eq!(result.invalidations.len(), 3);
    let root_rows = result
        .invalidations
        .iter()
        .filter(|row| row.cascade_from.is_none())
        .map(|row| row.affected_decision_id)
        .collect::<HashSet<_>>();
    assert_eq!(root_rows, HashSet::from([root_one.id, root_two.id]));
    let citing_rows = result
        .invalidations
        .iter()
        .filter(|row| row.affected_decision_id == citing.id)
        .collect::<Vec<_>>();
    assert_eq!(citing_rows.len(), 1);
    let citing_row = citing_rows[0];
    assert_eq!(citing_row.context_id, ctx.id);
    assert_eq!(citing_row.status, InvalidationStatus::Pending);
    assert_eq!(citing_row.affected_decision_id, citing.id);
    assert!(citing_row.cascade_from.is_some());
    assert_eq!(citing_row.severity, InvalidationSeverity::Info);
    assert_eq!(citing_row.threatened_intention_ids, vec![cascade_intent.id]);
    assert!(
        !citing_row
            .threatened_intention_ids
            .contains(&root_one_intent.id)
    );
    assert!(
        !citing_row
            .threatened_intention_ids
            .contains(&root_two_intent.id)
    );
    assert!(citing_row.intention_now_un_served);
    assert!(citing_row.basis_diff.is_invalidated_candidate);
    assert!(citing_row.basis_diff.divergences.is_empty());
}
