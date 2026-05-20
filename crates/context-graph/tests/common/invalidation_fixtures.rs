#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use chrono::Utc;
use context_graph::*;
use contextdb_core::Value;
use contextdb_engine::Database;
use serde_json::json;

#[allow(dead_code)]
pub struct InvalidationFixture {
    pub _dir: tempfile::TempDir,
    pub config: StoreConfig,
    pub context_id: ContextId,
    pub entity_id: EntityId,
    pub intention_id: IntentionId,
    pub decision_id: DecisionId,
    pub observation_id: ObservationId,
    pub invalidation: Invalidation,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct SeededInvalidation {
    pub context_id: ContextId,
    pub entity_id: EntityId,
    pub intention_id: IntentionId,
    pub decision_id: DecisionId,
    pub observation_id: ObservationId,
    pub invalidation: Invalidation,
}

pub fn fresh_store() -> (tempfile::TempDir, StoreConfig, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        ..Default::default()
    };
    let store = Store::open(config.clone()).expect("open store");
    (dir, config, store)
}

pub fn reopen(config: &StoreConfig) -> Store {
    Store::open(config.clone()).expect("reopen store")
}

pub fn reopen_scoped(config: &StoreConfig, contexts: impl IntoIterator<Item = ContextId>) -> Store {
    Store::open_with_contexts(
        config.clone(),
        contexts.into_iter().collect::<BTreeSet<_>>(),
    )
    .expect("reopen scoped store")
}

pub fn seed_pending_fixture() -> InvalidationFixture {
    seed_fixture_with_options("pending", DirectSeedOptions::default())
}

pub fn seed_fixture_with_options(label: &str, options: DirectSeedOptions) -> InvalidationFixture {
    let (dir, config, store) = fresh_store();
    let base = create_seed_base(&store, label, options);
    drop(store);
    insert_invalidation_direct(&config.db_path, base.invalidation_id, base.insert.clone());
    let store = reopen(&config);
    let invalidation = find_invalidation(&store, base.context_id, base.invalidation_id);
    drop(store);
    InvalidationFixture {
        _dir: dir,
        config,
        context_id: base.context_id,
        entity_id: base.entity_id,
        intention_id: base.intention_id,
        decision_id: base.decision_id,
        observation_id: base.observation_id,
        invalidation,
    }
}

pub fn seed_acknowledged_fixture() -> InvalidationFixture {
    let mut fixture = seed_pending_fixture();
    set_invalidation_status(
        &fixture.config,
        fixture.invalidation.id,
        InvalidationStatus::Acknowledged,
        None,
        None,
    );
    fixture.invalidation = find_invalidation(
        &reopen(&fixture.config),
        fixture.context_id,
        fixture.invalidation.id,
    );
    fixture
}

pub fn seed_resolved_fixture() -> InvalidationFixture {
    let mut fixture = seed_pending_fixture();
    set_invalidation_status(
        &fixture.config,
        fixture.invalidation.id,
        InvalidationStatus::Resolved,
        Some("basis_still_valid"),
        None,
    );
    fixture.invalidation = find_invalidation(
        &reopen(&fixture.config),
        fixture.context_id,
        fixture.invalidation.id,
    );
    fixture
}

pub fn seed_dismissed_fixture() -> InvalidationFixture {
    let mut fixture = seed_pending_fixture();
    set_invalidation_status(
        &fixture.config,
        fixture.invalidation.id,
        InvalidationStatus::Dismissed,
        Some("dismissed_as_irrelevant"),
        None,
    );
    fixture.invalidation = find_invalidation(
        &reopen(&fixture.config),
        fixture.context_id,
        fixture.invalidation.id,
    );
    fixture
}

#[derive(Default, Clone)]
pub struct DirectSeedOptions {
    pub context_id: Option<ContextId>,
    pub status: Option<InvalidationStatus>,
    pub threatened_intention_ids: Option<Vec<IntentionId>>,
    pub intention_now_un_served: bool,
    pub cascade_from: Option<DecisionId>,
    pub reason: Option<String>,
    pub mode_two_decision: bool,
}

struct SeedBase {
    context_id: ContextId,
    entity_id: EntityId,
    intention_id: IntentionId,
    decision_id: DecisionId,
    observation_id: ObservationId,
    invalidation_id: InvalidationId,
    insert: DirectInvalidationInsert,
}

fn create_seed_base(store: &Store, label: &str, options: DirectSeedOptions) -> SeedBase {
    let context = match options.context_id {
        Some(context_id) => store
            .get_context(context_id)
            .expect("get context")
            .expect("context exists"),
        None => create_context(store, label),
    };
    let entity = create_entity(store, context.id, label);
    let observation = record_seed_observation(store, context.id, entity.id, label);
    let intention = create_intention(store, context.id, label);
    let decision = create_decision(
        store,
        context.id,
        entity.id,
        intention.id,
        label,
        options.mode_two_decision,
    );
    let threatened = options
        .threatened_intention_ids
        .unwrap_or_else(|| vec![intention.id]);
    let invalidation_id = InvalidationId::new_v7();
    let context_id = context.id;
    SeedBase {
        context_id,
        entity_id: entity.id,
        intention_id: intention.id,
        decision_id: decision.id,
        observation_id: observation.observation_id,
        invalidation_id,
        insert: DirectInvalidationInsert {
            status: options.status.unwrap_or(InvalidationStatus::Pending),
            context_id,
            affected_decision_id: decision.id,
            trigger_observation_id: observation.observation_id,
            entity_id: entity.id,
            threatened_intention_ids: threatened,
            intention_now_un_served: options.intention_now_un_served,
            cascade_from: options.cascade_from,
            reason: options.reason,
        },
    }
}

pub fn seed_pending_invalidation_in_context(
    config: &StoreConfig,
    context_id: ContextId,
    label: &str,
) -> SeededInvalidation {
    seed_invalidation_in_context_with_options(
        config,
        context_id,
        label,
        DirectSeedOptions {
            context_id: Some(context_id),
            ..Default::default()
        },
    )
}

pub fn seed_invalidation_in_context_with_options(
    config: &StoreConfig,
    context_id: ContextId,
    label: &str,
    mut options: DirectSeedOptions,
) -> SeededInvalidation {
    options.context_id = Some(context_id);
    let store = reopen(config);
    let base = create_seed_base(&store, label, options);
    drop(store);
    insert_invalidation_direct(&config.db_path, base.invalidation_id, base.insert.clone());
    let store = reopen(config);
    let invalidation = find_invalidation(&store, base.context_id, base.invalidation_id);
    drop(store);
    SeededInvalidation {
        context_id: base.context_id,
        entity_id: base.entity_id,
        intention_id: base.intention_id,
        decision_id: base.decision_id,
        observation_id: base.observation_id,
        invalidation,
    }
}

pub fn create_context(store: &Store, label: &str) -> Context {
    store
        .create_context(CreateContext {
            name: format!("{label}-{}", uuid::Uuid::now_v7()),
            ..Default::default()
        })
        .expect("create context")
}

pub fn create_intention(store: &Store, context_id: ContextId, label: &str) -> Intention {
    store
        .create_intention(CreateIntention {
            description: format!("{label} intention"),
            context_id,
            ..Default::default()
        })
        .expect("create intention")
}

pub fn create_entity(store: &Store, context_id: ContextId, label: &str) -> Entity {
    store
        .create_entity(CreateEntity {
            entity_type: EntityType::File,
            name: format!("{label}-file"),
            context_id,
            properties: BTreeMap::from([("path".to_string(), json!("src/old.rs"))]),
            ..Default::default()
        })
        .expect("create entity")
}

pub fn create_decision(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    intention_id: IntentionId,
    label: &str,
    mode_two_decision: bool,
) -> Decision {
    store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: format!("{label} decision"),
            confidence: Some(0.9),
            intention_ids: vec![intention_id],
            based_on_entity_ids: vec![entity_id],
            basis_fields: Some(HashMap::from([(
                entity_id,
                vec!["path".to_string(), "owner".to_string()],
            )])),
            context_id,
            agent_id: mode_two_decision.then(|| "system".to_string()),
            ..Default::default()
        })
        .expect("create decision")
}

pub fn create_serving_decision(
    store: &Store,
    context_id: ContextId,
    intention_id: IntentionId,
    label: &str,
) -> Decision {
    store
        .create_decision(CreateDecision {
            decision_type: "remediation".to_string(),
            description: format!("{label} remediation decision"),
            confidence: Some(0.8),
            intention_ids: vec![intention_id],
            context_id,
            ..Default::default()
        })
        .expect("create serving decision")
}

pub fn create_unrelated_decision(store: &Store, context_id: ContextId, label: &str) -> Decision {
    let intention = create_intention(store, context_id, &format!("{label}-other"));
    create_serving_decision(store, context_id, intention.id, label)
}

pub fn record_seed_observation(
    store: &Store,
    context_id: ContextId,
    entity_id: EntityId,
    label: &str,
) -> RecordObservationResult {
    let observation_id = ObservationId::new_v7();
    store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id,
            context_id,
            observation_type: "change".to_string(),
            source: label.to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: EvidenceId::new_v7(),
                observation_id,
                context_id: ContextId::default(),
                kind: EvidenceKind::StructuredSignal,
                source_ref:
                    "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .to_string(),
                content_hash: Some(
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .to_string(),
                ),
                captured_at: Some(Utc::now()),
                retention_status: RetentionStatus::RetainedExternal,
                text_projection: Some(format!("{label} evidence")),
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta: BTreeMap::from([
                ("path".to_string(), Some(json!("src/current.rs"))),
                ("owner".to_string(), Some(json!("platform"))),
            ]),
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record seed observation")
}

#[derive(Clone)]
struct DirectInvalidationInsert {
    status: InvalidationStatus,
    context_id: ContextId,
    affected_decision_id: DecisionId,
    trigger_observation_id: ObservationId,
    entity_id: EntityId,
    threatened_intention_ids: Vec<IntentionId>,
    intention_now_un_served: bool,
    cascade_from: Option<DecisionId>,
    reason: Option<String>,
}

fn insert_invalidation_direct(
    path: &std::path::Path,
    id: InvalidationId,
    input: DirectInvalidationInsert,
) {
    let db = open_raw_database(path);
    let basis_diff = BasisDiffResult {
        is_invalidated_candidate: true,
        divergences: vec![EntityDivergence {
            basis_snapshot_id: EntitySnapshotId::new_v7(),
            entity_id: input.entity_id,
            value_diffs: vec![ValueDiff {
                field: "path".to_string(),
                basis_value: Some(json!("src/old.rs")),
                current_value: Some(json!("src/current.rs")),
            }],
            read_txid: TxId(1),
        }],
        read_txid: TxId(1),
        current_snapshot_read_txid: TxId(1),
    };
    let params = HashMap::from([
        ("id".to_string(), Value::Uuid(id.as_uuid())),
        (
            "status".to_string(),
            Value::Text(invalidation_status_str(input.status).to_string()),
        ),
        (
            "context_id".to_string(),
            Value::Uuid(input.context_id.as_uuid()),
        ),
        (
            "affected_decision_id".to_string(),
            Value::Uuid(input.affected_decision_id.as_uuid()),
        ),
        (
            "trigger_observation_id".to_string(),
            Value::Uuid(input.trigger_observation_id.as_uuid()),
        ),
        (
            "reason".to_string(),
            input.reason.map(Value::Text).unwrap_or(Value::Null),
        ),
        (
            "basis_diff".to_string(),
            Value::Json(serde_json::to_value(basis_diff).expect("basis diff json")),
        ),
        (
            "cascade_from".to_string(),
            input
                .cascade_from
                .map(|id| Value::Uuid(id.as_uuid()))
                .unwrap_or(Value::Null),
        ),
        ("severity".to_string(), Value::Text("warning".to_string())),
        (
            "threatened_intention_ids".to_string(),
            Value::Json(
                serde_json::to_value(input.threatened_intention_ids)
                    .expect("threatened intention ids json"),
            ),
        ),
        (
            "intention_now_un_served".to_string(),
            Value::Bool(input.intention_now_un_served),
        ),
        ("detected_at".to_string(), Value::TxId(TxId(1))),
        ("resolved_at".to_string(), Value::Null),
        ("resolution_action".to_string(), Value::Null),
        ("resolution_decision_id".to_string(), Value::Null),
    ]);
    db.execute(
        "INSERT INTO invalidations (id, status, context_id, affected_decision_id, trigger_observation_id, reason, basis_diff, cascade_from, severity, threatened_intention_ids, intention_now_un_served, detected_at, resolved_at, resolution_action, resolution_decision_id) VALUES ($id, $status, $context_id, $affected_decision_id, $trigger_observation_id, $reason, $basis_diff, $cascade_from, $severity, $threatened_intention_ids, $intention_now_un_served, $detected_at, $resolved_at, $resolution_action, $resolution_decision_id)",
        &params,
    )
    .expect("insert invalidation");
}

pub fn set_invalidation_status(
    config: &StoreConfig,
    id: InvalidationId,
    status: InvalidationStatus,
    action: Option<&str>,
    resolution_decision_id: Option<DecisionId>,
) {
    let db = open_raw_database(&config.db_path);
    if matches!(
        status,
        InvalidationStatus::Acknowledged
            | InvalidationStatus::Resolved
            | InvalidationStatus::Dismissed
    ) {
        update_invalidation_lifecycle(&db, id, InvalidationStatus::Acknowledged, None, None, None);
    }
    if matches!(
        status,
        InvalidationStatus::Resolved | InvalidationStatus::Dismissed
    ) {
        update_invalidation_lifecycle(
            &db,
            id,
            status,
            Some(TxId(2)),
            action,
            resolution_decision_id,
        );
    }
}

fn update_invalidation_lifecycle(
    db: &Database,
    id: InvalidationId,
    status: InvalidationStatus,
    resolved_at: Option<TxId>,
    action: Option<&str>,
    resolution_decision_id: Option<DecisionId>,
) {
    let params = HashMap::from([
        ("id".to_string(), Value::Uuid(id.as_uuid())),
        (
            "status".to_string(),
            Value::Text(invalidation_status_str(status).to_string()),
        ),
        (
            "resolved_at".to_string(),
            resolved_at.map(Value::TxId).unwrap_or(Value::Null),
        ),
        (
            "resolution_action".to_string(),
            action
                .map(|value| Value::Text(value.to_string()))
                .unwrap_or(Value::Null),
        ),
        (
            "resolution_decision_id".to_string(),
            resolution_decision_id
                .map(|id| Value::Uuid(id.as_uuid()))
                .unwrap_or(Value::Null),
        ),
    ]);
    db.execute(
        "UPDATE invalidations SET status = $status, resolved_at = $resolved_at, resolution_action = $resolution_action, resolution_decision_id = $resolution_decision_id WHERE id = $id",
        &params,
    )
    .expect("update invalidation lifecycle");
}

pub fn set_decision_status(config: &StoreConfig, id: DecisionId, status: DecisionStatus) {
    let db = open_raw_database(&config.db_path);
    let params = HashMap::from([
        ("id".to_string(), Value::Uuid(id.as_uuid())),
        (
            "status".to_string(),
            Value::Text(decision_status_str(status).to_string()),
        ),
    ]);
    db.execute(
        "UPDATE decisions SET status = $status WHERE id = $id",
        &params,
    )
    .expect("update decision status");
}

fn open_raw_database(path: impl AsRef<std::path::Path>) -> Database {
    let db = Database::open(path).expect("raw database open");
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

pub fn find_invalidation(store: &Store, context_id: ContextId, id: InvalidationId) -> Invalidation {
    store
        .list_invalidations(context_id, None, None)
        .expect("list invalidations")
        .into_iter()
        .find(|row| row.id == id)
        .expect("find invalidation")
}

pub fn audit_rows(store: &Store, filter: AuditFilter) -> Vec<AuditEntry> {
    store.audit_query(filter).expect("audit query")
}

pub fn audit_rows_for_invalidation(store: &Store, id: InvalidationId) -> Vec<AuditEntry> {
    audit_rows(
        store,
        AuditFilter {
            target: Some(AuditTarget::Invalidation(id)),
            ..Default::default()
        },
    )
}

pub fn assert_no_audit_rows_for(store: &Store, id: InvalidationId) {
    assert_eq!(audit_rows_for_invalidation(store, id).len(), 0);
}

pub fn assert_not_found<T: std::fmt::Debug>(result: context_graph::Result<T>, id: InvalidationId) {
    assert!(
        matches!(result, Err(CgError::InvalidationNotFound { id: got }) if got == id),
        "expected InvalidationNotFound for {id}, got {result:?}"
    );
}

pub fn assert_decision_not_found<T: std::fmt::Debug>(
    result: context_graph::Result<T>,
    id: DecisionId,
) {
    assert!(
        matches!(result, Err(CgError::DecisionNotFound { id: got }) if got == id),
        "expected DecisionNotFound for {id}, got {result:?}"
    );
}

pub fn assert_invalid_transition<T: std::fmt::Debug>(
    result: context_graph::Result<T>,
    from: InvalidationStatus,
    to: InvalidationStatus,
) {
    assert!(
        matches!(
            result,
            Err(CgError::InvalidTransition { from: got_from, to: got_to })
                if got_from == from && got_to == to
        ),
        "expected InvalidTransition {from:?}->{to:?}, got {result:?}"
    );
}

pub fn assert_non_lifecycle_fields_preserved(before: &Invalidation, after: &Invalidation) {
    assert_eq!(after.id, before.id);
    assert_eq!(after.context_id, before.context_id);
    assert_eq!(after.affected_decision_id, before.affected_decision_id);
    assert_eq!(after.trigger_observation_id, before.trigger_observation_id);
    assert_eq!(after.reason, before.reason);
    assert_eq!(after.basis_diff, before.basis_diff);
    assert_eq!(after.cascade_from, before.cascade_from);
    assert_eq!(after.severity, before.severity);
    assert_eq!(
        after.threatened_intention_ids,
        before.threatened_intention_ids
    );
    assert_eq!(
        after.intention_now_un_served,
        before.intention_now_un_served
    );
    assert_eq!(after.detected_at, before.detected_at);
}

pub fn assert_refusal_preserved(before: &Invalidation, after: &Invalidation) {
    assert_eq!(after.status, before.status);
    assert_eq!(after.resolved_at, before.resolved_at);
    assert_eq!(after.resolution_action, before.resolution_action);
    assert_eq!(after.resolution_decision_id, before.resolution_decision_id);
    assert_non_lifecycle_fields_preserved(before, after);
}

pub fn assert_ack_shape(before: &Invalidation, after: &Invalidation) {
    assert_eq!(after.status, InvalidationStatus::Acknowledged);
    assert_eq!(after.resolved_at, None);
    assert_eq!(after.resolution_action, None);
    assert_eq!(after.resolution_decision_id, None);
    assert_non_lifecycle_fields_preserved(before, after);
}

pub fn assert_resolved_shape(
    before: &Invalidation,
    after: &Invalidation,
    action: &str,
    decision_id: Option<DecisionId>,
) {
    assert_eq!(after.status, InvalidationStatus::Resolved);
    assert!(after.resolved_at.is_some());
    assert_eq!(after.resolution_action.as_deref(), Some(action));
    assert_eq!(after.resolution_decision_id, decision_id);
    assert_non_lifecycle_fields_preserved(before, after);
}

pub fn assert_dismissed_shape(before: &Invalidation, after: &Invalidation) {
    assert_eq!(after.status, InvalidationStatus::Dismissed);
    assert!(after.resolved_at.is_some());
    assert_eq!(
        after.resolution_action.as_deref(),
        Some("dismissed_as_irrelevant")
    );
    assert_eq!(after.resolution_decision_id, None);
    assert_non_lifecycle_fields_preserved(before, after);
}

pub fn assert_context_set(rows: &[Invalidation], expected: &[ContextId], expected_len: usize) {
    assert_eq!(rows.len(), expected_len);
    let actual = rows
        .iter()
        .map(|row| row.context_id)
        .collect::<HashSet<_>>();
    let expected = expected.iter().copied().collect::<HashSet<_>>();
    assert_eq!(actual, expected);
}

#[allow(dead_code)]
pub fn assert_audit_identity(
    entry: &AuditEntry,
    expected_txid: TxId,
    principal: &Principal,
    operation: AuditOperation,
    target: InvalidationId,
    context_id: ContextId,
    payload: Option<AuditPayload>,
) {
    assert_eq!(entry.txid, expected_txid);
    assert_eq!(&entry.actor, principal);
    assert_eq!(entry.operation, operation);
    assert_eq!(entry.target, AuditTarget::Invalidation(target));
    assert_eq!(entry.context_id, context_id);
    assert_eq!(entry.payload, payload);
}

fn invalidation_status_str(status: InvalidationStatus) -> &'static str {
    match status {
        InvalidationStatus::Pending => "pending",
        InvalidationStatus::Acknowledged => "acknowledged",
        InvalidationStatus::Resolved => "resolved",
        InvalidationStatus::Dismissed => "dismissed",
    }
}

fn decision_status_str(status: DecisionStatus) -> &'static str {
    match status {
        DecisionStatus::Active => "active",
        DecisionStatus::Superseded => "superseded",
        DecisionStatus::Archived => "archived",
        DecisionStatus::Invalidated => "invalidated",
    }
}
