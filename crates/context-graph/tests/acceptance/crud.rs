use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use chrono::Utc;
use context_graph::{
    AuditFilter, AuditTarget, CgError, ContextId, CreateContext, CreateDecision, CreateDigest,
    CreateEntity, CreateIntention, DecisionId, DecisionPatch, DecisionStatus, EntityId,
    EntityMatchPolicy, EntityPatch, EntityStatus, EntityType, Intention, IntentionId,
    IntentionOrigin, IntentionPatch, IntentionStatus, ListDecisionFilter, ListEntityFilter,
    ReasoningStep, RecordObservation, RelationshipType, StorageKey, Store, TraverseFilter,
};
use serde_json::{Value, json};
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
    let store = Store::open(&db_path);
    (tempdir, db_path, store)
}

fn reopen(db_path: &Path) -> Store {
    Store::open(db_path)
}

fn map(entries: &[(&str, Value)]) -> BTreeMap<String, Value> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
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

fn reasoning(label: &str) -> Vec<ReasoningStep> {
    vec![ReasoningStep {
        operation: format!("op-{label}"),
        details: format!("details-{label}"),
        result: format!("result-{label}"),
        agent_id: Some("claude-code-instance-dev".to_string()),
        timestamp: Utc::now(),
    }]
}

fn create_decision_input(
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
) -> CreateDecision {
    CreateDecision {
        decision_type: "architecture".to_string(),
        description: description.to_string(),
        reasoning: reasoning(description),
        confidence: Some(0.9),
        intention_ids,
        based_on_entity_ids,
        tags: vec!["architecture".to_string()],
        properties: BTreeMap::new(),
        context_id,
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

fn create_digest_input(context_id: ContextId, summary: &str) -> CreateDigest {
    CreateDigest {
        source: "claude-web".to_string(),
        summary: summary.to_string(),
        embedding: vec![0.25, 0.5, 0.75],
        context_id,
        session_ref: Some("session-7".to_string()),
        properties: BTreeMap::new(),
    }
}

// Use case: an engineering manager records the rationale for a payments migration, reboots the laptop, and the next morning the agent must find the Intention with the same id, description, and creation txid.
#[test]
fn intention_create_persists_across_reopen() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ContextId::new_v7();
    let intention = ok!(
        store.create_intention(CreateIntention {
            id: None,
            description: "preserve rationale for the payments migration".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id: ctx,
            properties: BTreeMap::new(),
            tags: vec![],
        }),
        "tx1: create intention"
    );
    assert_eq!(intention.id.as_uuid().get_version_num(), 7);
    let invalid = store.create_intention(CreateIntention {
        id: Some(IntentionId::new_v7()),
        description: "caller supplied id".to_string(),
        status: IntentionStatus::Active,
        origin: IntentionOrigin::Human,
        context_id: ctx,
        properties: BTreeMap::new(),
        tags: vec![],
    });
    assert!(
        matches!(invalid, Err(CgError::InvalidIdentity { .. })),
        "tx2: expected InvalidIdentity, got {:?}",
        invalid
    );
    drop(store);
    let store = reopen(&db_path);
    let fetched = ok!(store.get_intention(intention.id), "tx3: get intention").expect("intention");
    assert_eq!(fetched.description, intention.description);
    assert_eq!(fetched.status, IntentionStatus::Active);
    assert_eq!(fetched.origin, IntentionOrigin::Human);
    assert_eq!(fetched.context_id, ctx);
    assert_eq!(fetched.created_at, intention.created_at);
}

// Use case: a normal write and a retrying recovery path both try to land an Intention under the same id; the engine must refuse the second so no later get can return shadowed data.
#[test]
fn intention_identity_collision_rejected() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ContextId::new_v7();
    let original = ok!(
        store.create_intention(create_intention_input(ctx, "original")),
        "tx1: create original intention"
    );
    let forged = Intention {
        id: original.id,
        description: "forged".to_string(),
        status: IntentionStatus::Active,
        origin: IntentionOrigin::Human,
        context_id: ctx,
        properties: BTreeMap::new(),
        tags: vec![],
        created_at: original.created_at,
    };
    let result = store.admin_raw_insert_intention(forged);
    assert!(
        matches!(
            result,
            Err(CgError::UniqueViolation {
                entity: "intention",
                id: Some(ref id),
            }) if id == &original.id.as_uuid().to_string()
        ),
        "tx2: expected UniqueViolation for original id, got {:?}",
        result
    );
    drop(store);
    let store = reopen(&db_path);
    let fetched = ok!(store.get_intention(original.id), "tx3: get original").expect("original");
    assert_eq!(fetched.description, "original");
    let rows = ok!(
        store.admin_raw_scan_intention_rows(original.id),
        "tx4: raw scan"
    );
    assert_eq!(rows.len(), 1);
}

// Use case: an agent folds two Intentions into one via a batch update, but one id in the batch is stale; the engine rolls back atomically so the graph never enters a half-folded state.
#[test]
fn intention_update_transaction_rollback_on_error() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ContextId::new_v7();
    let http = ok!(
        store.create_intention(create_intention_input(
            ctx,
            "HTTP server for request/response"
        )),
        "tx1"
    );
    let mqtt = ok!(
        store.create_intention(create_intention_input(ctx, "MQTT for pub/sub transport")),
        "tx2"
    );
    let stale = IntentionId::new_v7();
    let result = store.update_intentions_batch(vec![
        (
            http.id,
            IntentionPatch {
                description: Some("NAT-friendly transport layer".to_string()),
                status: None,
                properties: None,
                tags: None,
            },
        ),
        (
            mqtt.id,
            IntentionPatch {
                description: None,
                status: Some(IntentionStatus::Superseded),
                properties: None,
                tags: None,
            },
        ),
        (
            stale,
            IntentionPatch {
                description: Some("stale".to_string()),
                status: None,
                properties: None,
                tags: None,
            },
        ),
    ]);
    assert!(
        matches!(result, Err(CgError::NotFound { id: Some(ref id), .. }) if id == &stale.as_uuid().to_string()),
        "tx3: expected NotFound for stale id, got {:?}",
        result
    );
    drop(store);
    let store = reopen(&db_path);
    let http_after = ok!(store.get_intention(http.id), "get http").expect("http");
    let mqtt_after = ok!(store.get_intention(mqtt.id), "get mqtt").expect("mqtt");
    assert_eq!(http_after.description, "HTTP server for request/response");
    assert_eq!(mqtt_after.description, "MQTT for pub/sub transport");
    assert_eq!(mqtt_after.status, IntentionStatus::Active);
    assert_eq!(ok!(store.history(http.id), "history http").len(), 1);
    assert_eq!(ok!(store.history(mqtt.id), "history mqtt").len(), 1);
}

// Use case: end-of-engagement cleanup asks the engine to archive an Intention whose active Decisions still SERVE it; the engine refuses at commit time and names the blocking Decisions (Invariant 1).
#[test]
fn intention_archive_blocked_by_active_decision() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "create context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "pick a web framework")),
        "create intention"
    );
    let decision = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "use axum",
            vec![intention.id],
            vec![]
        )),
        "create decision"
    );
    drop(store);
    let store = reopen(&db_path);
    store.disable_app_decision_scan_for_tests();
    let archived = store.archive_intention(intention.id, "done".to_string());
    assert!(
        matches!(
            archived,
            Err(CgError::ActiveDecisionsReference {
                intention_id,
                ref decision_ids,
            }) if intention_id == intention.id && decision_ids == &vec![decision.id.as_uuid().to_string()]
        ),
        "expected ActiveDecisionsReference for intention and blocking decision, got {:?}",
        archived
    );
    let fetched = ok!(store.get_intention(intention.id), "get intention").expect("intention");
    assert_eq!(fetched.status, IntentionStatus::Active);
    assert_ne!(decision.id, DecisionId::new_v7());
}

// Use case: once every SERVES-linked Decision is superseded or archived, the Intention may finally be archived; the engine's liveness check flips branch and commits (Invariant 1 permissive path).
#[test]
fn intention_archive_succeeds_when_decisions_superseded() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "create context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "pick a web framework")),
        "create intention"
    );
    let old = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "use rocket",
            vec![intention.id],
            vec![]
        )),
        "create old decision"
    );
    let newer = ok!(
        store.supersede_decision(
            old.id,
            create_decision_input(ctx.id, "use axum", vec![intention.id], vec![]),
        ),
        "supersede old decision"
    );
    let archived_new = ok!(
        store.archive_decision(newer.id, "project complete".to_string()),
        "archive new decision"
    );
    assert_eq!(archived_new.status, DecisionStatus::Invalidated);
    let archived_intention = ok!(
        store.archive_intention(intention.id, "done".to_string()),
        "archive intention"
    );
    assert_eq!(archived_intention.status, IntentionStatus::Archived);
    drop(store);
    let store = reopen(&db_path);
    let intention_after =
        ok!(store.get_intention(intention.id), "get intention").expect("intention");
    let old_after = ok!(store.get_decision(old.id), "get old").expect("old");
    let new_after = ok!(store.get_decision(newer.id), "get new").expect("new");
    assert_eq!(intention_after.status, IntentionStatus::Archived);
    assert_eq!(old_after.status, DecisionStatus::Superseded);
    assert_eq!(new_after.status, DecisionStatus::Invalidated);
    let serves = ok!(
        store.decision_targets_intention(newer.id, RelationshipType::Serves, intention.id),
        "SERVES edge persisted"
    );
    assert!(serves);
    let supersedes = ok!(
        store.edge_exists(newer.id, RelationshipType::Supersedes, old.id),
        "SUPERSEDES edge persisted"
    );
    assert!(supersedes);
}

// Use case: a misconfigured agent omits intention_ids (or passes a non-existent one) when recording a Decision; the engine refuses at commit so no orphan Decision ever persists.
#[test]
fn decision_create_requires_intention() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "setup: create context"
    );
    let missing = store.create_decision(CreateDecision {
        decision_type: "architecture".to_string(),
        description: "floating decision".to_string(),
        reasoning: reasoning("floating"),
        confidence: Some(0.9),
        intention_ids: vec![],
        based_on_entity_ids: vec![],
        tags: vec![],
        properties: BTreeMap::new(),
        context_id: ctx.id,
    });
    assert!(
        matches!(missing, Err(CgError::MissingIntention)),
        "tx1: expected MissingIntention, got {:?}",
        missing
    );
    drop(store);
    let store = reopen(&db_path);
    assert!(
        ok!(
            store.list_decisions(ListDecisionFilter::default()),
            "list after first reject"
        )
        .is_empty()
    );
    let non_existent_id = IntentionId::new_v7();
    let bad_id = store.create_decision(CreateDecision {
        decision_type: "architecture".to_string(),
        description: "floating decision".to_string(),
        reasoning: reasoning("bad-id"),
        confidence: Some(0.9),
        intention_ids: vec![non_existent_id],
        based_on_entity_ids: vec![],
        tags: vec![],
        properties: BTreeMap::new(),
        context_id: ctx.id,
    });
    assert!(
        matches!(bad_id, Err(CgError::IntentionNotFound { id }) if id == non_existent_id),
        "tx2: expected IntentionNotFound for the bad id, got {:?}",
        bad_id
    );
    drop(store);
    let store = reopen(&db_path);
    assert!(
        ok!(
            store.list_decisions(ListDecisionFilter::default()),
            "list after second reject"
        )
        .is_empty()
    );
}

// Use case: a platform engineer records "use axum" with three reasoning steps and two BASED_ON entities; six months later the graph must return reasoning, SERVES, and BASED_ON snapshots as they were at decision time (Invariant 2).
#[test]
fn decision_create_with_serves_and_based_on_edges_persisted() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(
            ctx.id,
            "pick a web framework for the new internal tool"
        )),
        "intention"
    );
    let e1 = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "internal-tool-api".to_string(),
            properties: map(&[
                ("language", json!("rust")),
                ("deployment", json!("kubernetes"))
            ]),
            tags: vec![],
            context_id: ctx.id,
        }),
        "E1"
    );
    let e2 = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Team,
            name: "platform-team".to_string(),
            properties: map(&[("size", json!("4")), ("skill", json!("rust-heavy"))]),
            tags: vec![],
            context_id: ctx.id,
        }),
        "E2"
    );
    let decision = ok!(
        store.create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: "use axum as the web framework".to_string(),
            reasoning: vec![
                reasoning("rs1")[0].clone(),
                reasoning("rs2")[0].clone(),
                reasoning("rs3")[0].clone(),
            ],
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![e1.id, e2.id],
            tags: vec!["architecture".to_string()],
            properties: BTreeMap::new(),
            context_id: ctx.id,
        }),
        "decision"
    );
    ok!(
        store.update_entity(
            e1.id,
            EntityPatch {
                name: None,
                properties: Some(map(&[
                    ("language", json!("go")),
                    ("deployment", json!("ecs"))
                ])),
                tags: None,
            },
        ),
        "later update E1"
    );
    ok!(
        store.update_entity(
            e2.id,
            EntityPatch {
                name: None,
                properties: Some(map(&[("size", json!("7")), ("skill", json!("polyglot"))])),
                tags: None,
            },
        ),
        "later update E2"
    );
    drop(store);
    let store = reopen(&db_path);
    let fetched = ok!(store.get_decision(decision.id), "get decision").expect("decision");
    assert_eq!(fetched.id, decision.id);
    assert_eq!(fetched.id.as_uuid().get_version_num(), 7);
    assert_eq!(fetched.reasoning, decision.reasoning);
    assert_eq!(
        fetched.reasoning[0].timestamp,
        decision.reasoning[0].timestamp
    );
    assert_eq!(fetched.status, DecisionStatus::Active);
    assert_eq!(fetched.intention_ids, vec![intention.id]);
    let serves = ok!(
        store.decision_targets_intention(decision.id, RelationshipType::Serves, intention.id),
        "SERVES edge persisted"
    );
    let based_on_e1 = ok!(
        store.decision_targets_entity(decision.id, RelationshipType::BasedOn, e1.id),
        "BASED_ON E1 persisted"
    );
    let based_on_e2 = ok!(
        store.decision_targets_entity(decision.id, RelationshipType::BasedOn, e2.id),
        "BASED_ON E2 persisted"
    );
    assert!(serves);
    assert!(based_on_e1);
    assert!(based_on_e2);
    assert_eq!(
        fetched.based_on_snapshots[&e1.id].properties,
        map(&[
            ("language", json!("rust")),
            ("deployment", json!("kubernetes"))
        ])
    );
    assert_eq!(
        fetched.based_on_snapshots[&e2.id].properties,
        map(&[("size", json!("4")), ("skill", json!("rust-heavy"))])
    );
    assert_ne!(
        fetched.based_on_snapshots[&e1.id].properties,
        map(&[("language", json!("go")), ("deployment", json!("ecs"))])
    );
    assert_ne!(
        fetched.based_on_snapshots[&e2.id].properties,
        map(&[("size", json!("7")), ("skill", json!("polyglot"))])
    );
}

// Use case: a service moves from region-a to region-b after a Decision was recorded on region-a; the BASED_ON snapshot must still read region-a so the invalidation signal fires instead of silently agreeing with current state.
#[test]
fn decision_based_on_snapshot_frozen_after_entity_update() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let entity = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "alerts-api".to_string(),
            properties: map(&[
                ("region", json!("region-a")),
                ("threshold_ms", json!("200"))
            ]),
            tags: vec![],
            context_id: ctx.id,
        }),
        "entity"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "define alerting thresholds")),
        "intention"
    );
    let decision = ok!(
        store.create_decision(CreateDecision {
            decision_type: "alerting_policy".to_string(),
            description: "alert threshold 200ms because service runs in region-a".to_string(),
            reasoning: reasoning("alerting"),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx.id,
        }),
        "decision"
    );
    drop(store);
    let store = reopen(&db_path);
    ok!(
        store.update_entity(
            entity.id,
            EntityPatch {
                name: None,
                properties: Some(map(&[
                    ("region", json!("region-b")),
                    ("threshold_ms", json!("200"))
                ])),
                tags: None,
            },
        ),
        "update entity"
    );
    let fetched = ok!(store.get_decision(decision.id), "get decision").expect("decision");
    let historical =
        ok!(store.state_at(entity.id, decision.created_at), "state_at").expect("history");
    let current = ok!(store.get_entity(entity.id), "current entity").expect("entity");
    assert_eq!(
        fetched.based_on_snapshots[&entity.id].properties["region"],
        json!("region-a")
    );
    assert_eq!(historical.properties["region"], json!("region-a"));
    assert_eq!(current.properties["region"], json!("region-b"));
}

// Use case: an agent resuming mid-conversation asks "what Decisions already SERVE this Intention?" and must get the indexed answer fast enough to inline at prompt time, even with 10k unrelated Decisions in the store.
#[test]
fn decision_list_by_intention() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let i1 = ok!(
        store.create_intention(create_intention_input(ctx.id, "pick a web framework")),
        "I1"
    );
    let i2 = ok!(
        store.create_intention(create_intention_input(ctx.id, "pick a message broker")),
        "I2"
    );
    let d1 = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "use axum",
            vec![i1.id],
            vec![]
        )),
        "D1"
    );
    let d2 = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "use actix",
            vec![i1.id],
            vec![]
        )),
        "D2"
    );
    let d3 = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "use nats",
            vec![i2.id],
            vec![]
        )),
        "D3"
    );
    for i in 0..10_000 {
        ok!(
            store.create_decision(create_decision_input(
                ctx.id,
                &format!("unrelated-{i}"),
                vec![i2.id],
                vec![]
            )),
            "seed unrelated decisions"
        );
    }
    drop(store);
    let store = reopen(&db_path);
    let started = Instant::now();
    let results = ok!(
        store.list_decisions(ListDecisionFilter {
            intention_id: Some(i1.id),
            ..ListDecisionFilter::default()
        }),
        "list decisions"
    );
    let elapsed = started.elapsed();
    let trace = ok!(store.last_query_trace(), "query trace");
    assert_eq!(results, vec![d2.clone(), d1.clone()]);
    assert!(!results.iter().any(|decision| decision.id == d3.id));
    assert!(elapsed <= Duration::from_millis(100));
    assert!(trace.contains("SERVES"));
}

// Use case: a network-posture review asks for Decisions tagged exactly "networking" and must not silently fold in "networking-legacy" via substring matching; tag exactness is a storage-index invariant.
#[test]
fn decision_list_by_tags() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "manage network posture")),
        "intention"
    );
    for i in 0..100 {
        let bucket = match i % 10 {
            0 => "networking",
            1 => "storage",
            2 => "compute",
            3 => "security",
            4 => "platform",
            5 => "reliability",
            6 => "messaging",
            7 => "compliance",
            8 => "identity",
            _ => "delivery",
        };
        ok!(
            store.create_decision(CreateDecision {
                decision_type: "policy".to_string(),
                description: format!("net-policy-{i}"),
                reasoning: reasoning("tagged"),
                confidence: Some(0.9),
                intention_ids: vec![intention.id],
                based_on_entity_ids: vec![],
                tags: vec![bucket.to_string()],
                properties: BTreeMap::new(),
                context_id: ctx.id,
            }),
            "seed decision"
        );
    }
    let trap = ok!(
        store.create_decision(CreateDecision {
            decision_type: "policy".to_string(),
            description: "legacy migration plan".to_string(),
            reasoning: reasoning("trap"),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![],
            tags: vec!["networking-legacy".to_string()],
            properties: BTreeMap::new(),
            context_id: ctx.id,
        }),
        "trap decision"
    );
    drop(store);
    let store = reopen(&db_path);
    let results = ok!(
        store.list_decisions(ListDecisionFilter {
            tags: vec!["networking".to_string()],
            ..ListDecisionFilter::default()
        }),
        "list by tags"
    );
    assert_eq!(results.len(), 10);
    assert!(
        results
            .iter()
            .all(|decision| decision.tags.iter().any(|tag| tag == "networking"))
    );
    assert!(
        results
            .iter()
            .all(|decision| decision.tags.iter().all(|tag| tag != "networking-legacy"))
    );
    assert!(!results.iter().any(|decision| decision.id == trap.id));
    assert!(
        results
            .windows(2)
            .all(|pair| pair[0].created_at >= pair[1].created_at)
    );
}

// Use case: an engineer planning a refactor asks for Decisions BASED_ON either the payments-api or the payments-db; each matching Decision must appear exactly once so blast-radius counts are not inflated.
#[test]
fn decision_list_by_entities_any_semantics() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "refactor payments pipeline")),
        "intention"
    );
    let e1 = ok!(
        store.create_entity(create_entity_input(
            ctx.id,
            EntityType::Service,
            "payments-api"
        )),
        "E1"
    );
    let e2 = ok!(
        store.create_entity(create_entity_input(
            ctx.id,
            EntityType::Database,
            "payments-db"
        )),
        "E2"
    );
    let e3 = ok!(
        store.create_entity(create_entity_input(
            ctx.id,
            EntityType::Service,
            "search-api"
        )),
        "E3"
    );
    let d_a = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "cache invalidation policy",
            vec![intention.id],
            vec![e1.id]
        )),
        "D_a"
    );
    let d_b = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "schema migration plan",
            vec![intention.id],
            vec![e2.id]
        )),
        "D_b"
    );
    let d_both = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "transactional boundary for checkout",
            vec![intention.id],
            vec![e1.id, e2.id]
        )),
        "D_both"
    );
    let d_c = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "search relevance tuning",
            vec![intention.id],
            vec![e3.id]
        )),
        "D_c"
    );
    drop(store);
    let store = reopen(&db_path);
    let results = ok!(
        store.list_decisions(ListDecisionFilter {
            based_on_entity_ids: vec![e1.id, e2.id],
            entity_match_policy: EntityMatchPolicy::Any,
            ..ListDecisionFilter::default()
        }),
        "list any"
    );
    assert_eq!(results, vec![d_both.clone(), d_b.clone(), d_a.clone()]);
    assert_eq!(results.len(), 3);
    assert!(!results.iter().any(|decision| decision.id == d_c.id));
}

// Use case: a cross-service regression investigation asks for Decisions that depend on BOTH auth and rate-limiter; the engine must intersect BASED_ON reverse indexes, not union them.
#[test]
fn decision_list_by_entities_all_semantics() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "refactor payments pipeline")),
        "intention"
    );
    let e1 = ok!(
        store.create_entity(create_entity_input(
            ctx.id,
            EntityType::Service,
            "payments-api"
        )),
        "E1"
    );
    let e2 = ok!(
        store.create_entity(create_entity_input(
            ctx.id,
            EntityType::Database,
            "payments-db"
        )),
        "E2"
    );
    let e3 = ok!(
        store.create_entity(create_entity_input(
            ctx.id,
            EntityType::Service,
            "search-api"
        )),
        "E3"
    );
    let d_a = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "cache invalidation policy",
            vec![intention.id],
            vec![e1.id]
        )),
        "D_a"
    );
    let d_b = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "schema migration plan",
            vec![intention.id],
            vec![e2.id]
        )),
        "D_b"
    );
    let d_both = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "transactional boundary for checkout",
            vec![intention.id],
            vec![e1.id, e2.id]
        )),
        "D_both"
    );
    let d_c = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "search relevance tuning",
            vec![intention.id],
            vec![e3.id]
        )),
        "D_c"
    );
    drop(store);
    let store = reopen(&db_path);
    let results = ok!(
        store.list_decisions(ListDecisionFilter {
            based_on_entity_ids: vec![e1.id, e2.id],
            entity_match_policy: EntityMatchPolicy::All,
            ..ListDecisionFilter::default()
        }),
        "list all"
    );
    assert_eq!(results, vec![d_both.clone()]);
    assert!(!results.iter().any(|decision| decision.id == d_a.id));
    assert!(!results.iter().any(|decision| decision.id == d_b.id));
    assert!(!results.iter().any(|decision| decision.id == d_c.id));
}

// Use case: an agent tries to quietly rewrite a Decision's description for a nicer report; the engine refuses at the schema layer so the audit trail cannot be polished after the fact.
#[test]
fn decision_update_immutable_fields_rejected() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "choose a web framework")),
        "intention"
    );
    let decision = ok!(
        store.create_decision(CreateDecision {
            decision_type: "framework_choice".to_string(),
            description: "use axum for the HTTP layer".to_string(),
            reasoning: reasoning("framework"),
            confidence: Some(0.8),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![],
            tags: vec!["framework".to_string()],
            properties: BTreeMap::new(),
            context_id: ctx.id,
        }),
        "decision"
    );
    let updated = ok!(
        store.update_decision(
            decision.id,
            DecisionPatch {
                properties: Some(map(&[("revisit_on", json!("2026-Q4"))])),
                tags: Some(vec!["framework".to_string(), "rust".to_string()]),
                confidence: Some(0.9),
            },
        ),
        "update mutable fields"
    );
    assert_eq!(updated.properties["revisit_on"], json!("2026-Q4"));
    assert_eq!(
        updated.tags,
        vec!["framework".to_string(), "rust".to_string()]
    );
    assert_eq!(updated.confidence, 0.9);
    assert_eq!(updated.description, "use axum for the HTTP layer");
    assert_eq!(updated.decision_type, "framework_choice");
    let bad_type = store.admin_raw_update_decision_immutable_fields(
        decision.id,
        Some("architecture_choice".to_string()),
        None,
    );
    assert!(matches!(
        bad_type,
        Err(CgError::ImmutableField { ref field }) if field == "decision_type"
    ));
    let bad_description = store.admin_raw_update_decision_immutable_fields(
        decision.id,
        None,
        Some("use axum 0.7 for the HTTP layer".to_string()),
    );
    assert!(matches!(
        bad_description,
        Err(CgError::ImmutableField { ref field }) if field == "description"
    ));
    drop(store);
    let store = reopen(&db_path);
    let after = ok!(store.get_decision(decision.id), "get decision").expect("decision");
    assert_eq!(after.decision_type, "framework_choice");
    assert_eq!(after.description, "use axum for the HTTP layer");
    assert_eq!(
        after.tags,
        vec!["framework".to_string(), "rust".to_string()]
    );
    assert_eq!(after.confidence, 0.9);
}

// Use case: an architect replaces last quarter's framework choice because upstream is being deprecated; the engine flips the old Decision to Superseded, writes a SUPERSEDES edge atomically, and refuses any second supersede attempt (Invariant 5 DAG).
#[test]
fn decision_supersede_creates_edge_and_transitions_status() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "pick a web framework")),
        "intention"
    );
    let d1 = ok!(
        store.create_decision(CreateDecision {
            decision_type: "framework_choice".to_string(),
            description: "use axum".to_string(),
            reasoning: reasoning("d1"),
            confidence: Some(0.8),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx.id,
        }),
        "D1"
    );
    assert_eq!(d1.status, DecisionStatus::Active);
    let d2 = ok!(
        store.supersede_decision(
            d1.id,
            CreateDecision {
                decision_type: "framework_choice".to_string(),
                description: "use actix — axum upstream deprecated".to_string(),
                reasoning: reasoning("d2"),
                confidence: Some(0.8),
                intention_ids: vec![intention.id],
                based_on_entity_ids: vec![],
                tags: vec![],
                properties: BTreeMap::new(),
                context_id: ctx.id,
            },
        ),
        "D2"
    );
    drop(store);
    let store = reopen(&db_path);
    let d1_after = ok!(store.get_decision(d1.id), "get d1").expect("d1");
    let d2_after = ok!(store.get_decision(d2.id), "get d2").expect("d2");
    assert_eq!(d1_after.status, DecisionStatus::Superseded);
    assert_eq!(d2_after.status, DecisionStatus::Active);
    let edge = ok!(
        store.edge_exists(d2.id, RelationshipType::Supersedes, d1.id),
        "edge exists"
    );
    assert!(edge);
    let chain = ok!(store.precedent_chain(d2.id, 10), "precedent chain");
    assert_eq!(chain, vec![d2_after.clone(), d1_after.clone()]);
    let second = store.supersede_decision(
        d1.id,
        create_decision_input(ctx.id, "retry supersede", vec![intention.id], vec![]),
    );
    assert!(matches!(
        second,
        Err(CgError::IllegalTransition {
            from: DecisionStatus::Superseded,
            to: DecisionStatus::Superseded
        })
    ));
}

// Use case: an agent asked to "revise" an already-invalidated Decision tries to supersede it; the engine refuses the illegal transition so retired work cannot come back to life through a side channel.
#[test]
fn decision_invalid_state_transition_rejected() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "pick a caching layer")),
        "intention"
    );
    let decision = ok!(
        store.create_decision(CreateDecision {
            decision_type: "cache_choice".to_string(),
            description: "use redis".to_string(),
            reasoning: reasoning("cache"),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx.id,
        }),
        "decision"
    );
    let archived = ok!(
        store.archive_decision(
            decision.id,
            "redis dependency dropped — caching disabled project-wide".to_string()
        ),
        "archive"
    );
    assert_eq!(archived.status, DecisionStatus::Invalidated);
    drop(store);
    let store = reopen(&db_path);
    let persisted = ok!(store.get_decision(decision.id), "get decision").expect("decision");
    assert_eq!(persisted.status, DecisionStatus::Invalidated);
    let illegal = store.supersede_decision(
        decision.id,
        CreateDecision {
            decision_type: "cache_choice".to_string(),
            description: "use memcached".to_string(),
            reasoning: reasoning("memcached"),
            confidence: Some(0.7),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx.id,
        },
    );
    assert!(matches!(
        illegal,
        Err(CgError::IllegalTransition {
            from: DecisionStatus::Invalidated,
            to: DecisionStatus::Superseded
        })
    ));
    let after = ok!(store.get_decision(decision.id), "get decision again").expect("decision");
    assert_eq!(after.status, DecisionStatus::Invalidated);
    assert!(
        ok!(
            store.incoming_supersedes(decision.id),
            "incoming supersedes"
        )
        .is_empty()
    );
}

// Use case: an agent looks up a hallucinated id and, separately, a garbled string; Ok(None) for well-formed-but-unknown and CgError::InvalidIdentity for unparseable must stay categorically distinct at the read boundary.
#[test]
fn decision_get_not_found() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "demo")),
        "intention"
    );
    let _real = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "real",
            vec![intention.id],
            vec![]
        )),
        "real"
    );
    let unknown_id = DecisionId::new_v7();
    assert!(matches!(store.get_decision(unknown_id), Ok(None)));
    let malformed = DecisionId::from_raw_bytes("not-a-uuid");
    assert!(matches!(
        store.get_decision(malformed),
        Err(CgError::InvalidIdentity {
            context: "DecisionId"
        })
    ));
    drop(store);
    let store = reopen(&db_path);
    assert!(matches!(store.get_decision(unknown_id), Ok(None)));
    assert!(matches!(
        store.get_decision(malformed),
        Err(CgError::InvalidIdentity {
            context: "DecisionId"
        })
    ));
}

// Use case: a batch cleanup tries to archive a Decision id that was never recorded; the engine reports NotFound loudly rather than silently succeeding on a phantom target.
#[test]
fn decision_archive_not_found() {
    let (_tempdir, db_path, store) = fixture();
    let unknown_id = DecisionId::new_v7();
    let result = store.archive_decision(unknown_id, "cleanup".to_string());
    assert!(matches!(result, Err(CgError::NotFound { .. })));
    drop(store);
    let store = reopen(&db_path);
    assert!(matches!(store.get_decision(unknown_id), Ok(None)));
    let archived = ok!(
        store.list_decisions(ListDecisionFilter {
            status: Some(DecisionStatus::Invalidated),
            ..ListDecisionFilter::default()
        }),
        "list archived"
    );
    assert!(archived.is_empty());
}

// Use case: a compliance auditor a year later asks "what Decisions were active at that review?"; archived Decisions must remain queryable with the row physically present under admin raw scan (Invariant 3 — no hard deletes).
#[test]
fn decision_archive_preserves_history() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "evaluate queue providers")),
        "intention"
    );
    let decision = ok!(
        store.create_decision(CreateDecision {
            decision_type: "queue_choice".to_string(),
            description: "use rabbitmq for now".to_string(),
            reasoning: reasoning("rabbitmq"),
            confidence: Some(0.7),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![],
            tags: vec![],
            properties: BTreeMap::new(),
            context_id: ctx.id,
        }),
        "decision"
    );
    let archived = ok!(
        store.archive_decision(
            decision.id,
            "rabbitmq dropped for redis streams — see D' under new intention".to_string()
        ),
        "archive"
    );
    assert_eq!(archived.status, DecisionStatus::Invalidated);
    assert_eq!(archived.description, "use rabbitmq for now");
    drop(store);
    let store = reopen(&db_path);
    let fetched = ok!(store.get_decision(decision.id), "get decision").expect("decision");
    assert_eq!(fetched.status, DecisionStatus::Invalidated);
    assert_eq!(fetched.description, "use rabbitmq for now");
    let audit = ok!(
        store.audit_query(AuditFilter {
            decision_id: Some(decision.id),
            ..AuditFilter::default()
        }),
        "audit query"
    );
    assert!(audit.len() >= 2);
    assert!(audit.windows(2).all(|pair| pair[0].txid <= pair[1].txid));
    assert!(
        audit
            .iter()
            .any(|entry| matches!(entry.operation, context_graph::AuditOperation::Create))
    );
    assert!(audit.iter().any(|entry| matches!(
        entry.operation,
        context_graph::AuditOperation::Archive
    ) && entry.reason.as_deref()
        == Some("rabbitmq dropped for redis streams — see D' under new intention")));
    let raw = ok!(
        store.admin_raw_scan(StorageKey::Decision(decision.id)),
        "raw scan"
    )
    .expect("raw row should exist");
    assert!(!raw.tombstone);
}

// Use case: an operator records a payments-api Entity with region, owner, and version; after a process restart the id resolves, the initial snapshot is queryable, and list_entities surfaces it under its SERVICE type.
#[test]
fn entity_create_and_snapshot_persist_across_reopen() {
    let (_tempdir, db_path, store) = fixture();
    let entity = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "payments-api".to_string(),
            properties: map(&[
                ("region", json!("us-east-1")),
                ("owner", json!("team-payments")),
                ("version", json!("1.4.2")),
            ]),
            tags: vec!["tier-1".to_string(), "customer-facing".to_string()],
            context_id: ContextId::new_v7(),
        }),
        "create entity"
    );
    let decoy = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Database,
            name: "payments-db".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: entity.context_id,
        }),
        "create decoy entity"
    );
    assert_eq!(entity.id.as_uuid().get_version_num(), 7);
    assert_eq!(entity.properties["region"], json!("us-east-1"));
    drop(store);
    let store = reopen(&db_path);
    let fetched = ok!(store.get_entity(entity.id), "get entity").expect("entity");
    assert_eq!(fetched.id, entity.id);
    assert_eq!(fetched.entity_type, EntityType::Service);
    assert_eq!(fetched.name, "payments-api");
    assert_eq!(fetched.properties, entity.properties);
    assert_eq!(
        fetched.tags,
        vec!["tier-1".to_string(), "customer-facing".to_string()]
    );
    assert_eq!(fetched.created_at, entity.created_at);
    let snapshot = ok!(store.snapshot_at(entity.id, Utc::now()), "snapshot").expect("snapshot");
    assert_eq!(snapshot.entity_id, entity.id);
    assert_eq!(snapshot.properties, entity.properties);
    assert_eq!(snapshot.captured_at, entity.created_at);
    let listed = ok!(
        store.list_entities(ListEntityFilter {
            entity_type: Some(EntityType::Service),
            ..ListEntityFilter::default()
        }),
        "list entities"
    );
    assert!(listed.iter().any(|candidate| candidate.id == entity.id));
    assert!(!listed.iter().any(|candidate| candidate.id == decoy.id));
}

// Use case: an operator investigating last month's regression asks what the service looked like two weeks ago versus three; each snapshot's valid_to must equal the next's valid_from as exact txid equality (Invariant 11).
#[test]
fn entity_update_creates_new_snapshot_contiguous_after_reopen() {
    let (_tempdir, db_path, store) = fixture();
    let entity = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "payments-api".to_string(),
            properties: map(&[("region", json!("us-east-1")), ("version", json!("1.4.2"))]),
            tags: vec![],
            context_id: ContextId::new_v7(),
        }),
        "tx1"
    );
    ok!(
        store.update_entity(
            entity.id,
            EntityPatch {
                name: None,
                properties: Some(map(&[
                    ("region", json!("us-east-1")),
                    ("version", json!("1.5.0"))
                ])),
                tags: None,
            },
        ),
        "tx2"
    );
    ok!(
        store.update_entity(
            entity.id,
            EntityPatch {
                name: None,
                properties: Some(map(&[
                    ("region", json!("us-west-2")),
                    ("version", json!("1.5.0"))
                ])),
                tags: None,
            },
        ),
        "tx3"
    );
    drop(store);
    let store = reopen(&db_path);
    let history = ok!(store.entity_history(entity.id), "history");
    assert_eq!(history.len(), 3);
    let tx0 = history[0].valid_from;
    let tx1 = history[1].valid_from;
    let tx2 = history[2].valid_from;
    assert_eq!(history[0].valid_to, Some(history[1].valid_from));
    assert_eq!(history[1].valid_to, Some(history[2].valid_from));
    assert!(history[0].valid_from < history[1].valid_from);
    assert!(history[1].valid_from < history[2].valid_from);
    assert_eq!(history[2].valid_to, None);
    assert_eq!(history[0].properties["version"], json!("1.4.2"));
    assert_eq!(history[1].properties["version"], json!("1.5.0"));
    assert_eq!(history[2].properties["region"], json!("us-west-2"));
    assert_eq!(history[0].valid_from, tx0);
    assert_eq!(history[1].valid_from, tx1);
    assert_eq!(history[2].valid_from, tx2);
}

// Use case: a retired service is archived so default listings no longer surface it, but every prior snapshot stays queryable so historical Decisions that BASED_ON it remain explainable.
#[test]
fn entity_archive_retains_history() {
    let (_tempdir, db_path, store) = fixture();
    let entity = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "payments-v1".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ContextId::new_v7(),
        }),
        "tx1"
    );
    ok!(
        store.update_entity(
            entity.id,
            EntityPatch {
                name: None,
                properties: Some(map(&[("region", json!("us-east"))])),
                tags: None,
            },
        ),
        "tx2"
    );
    let mid_point = Utc::now();
    ok!(
        store.update_entity(
            entity.id,
            EntityPatch {
                name: Some("payments-legacy".to_string()),
                properties: None,
                tags: None,
            },
        ),
        "tx3"
    );
    let archived = ok!(store.archive_entity(entity.id), "tx4");
    assert_eq!(archived.status, EntityStatus::Archived);
    drop(store);
    let store = reopen(&db_path);
    let history = ok!(store.entity_history(entity.id), "history");
    assert_eq!(history.len(), 3);
    let snapshot = ok!(store.snapshot_at(entity.id, mid_point), "snapshot").expect("snapshot");
    assert_eq!(snapshot.entity_id, entity.id);
    assert_eq!(snapshot.name, "payments-v1");
    assert_eq!(snapshot.properties["region"], json!("us-east"));
    assert!(history[2].valid_to.is_some());
    let listed = ok!(
        store.list_entities(ListEntityFilter::default()),
        "default list"
    );
    assert!(!listed.iter().any(|candidate| candidate.id == entity.id));
    let archived_list = ok!(
        store.list_entities(ListEntityFilter {
            include_archived: true,
            ..ListEntityFilter::default()
        }),
        "archived list"
    );
    assert!(
        archived_list
            .iter()
            .any(|candidate| candidate.id == entity.id)
    );
}

// Use case: an agent drifts onto a stale id and calls update_entity; the engine refuses rather than upserting a phantom Entity into the graph.
#[test]
fn entity_update_not_found() {
    let (_tempdir, db_path, store) = fixture();
    let id = EntityId::new_v7();
    let result = store.update_entity(id, EntityPatch::default());
    assert!(matches!(result, Err(CgError::NotFound { .. })));
    drop(store);
    let store = reopen(&db_path);
    assert!(
        ok!(
            store.list_entities(ListEntityFilter {
                include_archived: true,
                ..ListEntityFilter::default()
            }),
            "list all"
        )
        .is_empty()
    );
}

// Use case: an agent searches a 5k-Entity store for payments-related SERVICE rows; case-insensitive name filter combined with entity_type must return deterministic, pageable results under the p95 budget.
#[test]
fn entity_list_filters_use_engine_indexes() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ContextId::new_v7();
    for i in 0..5_000 {
        let entity_type = if i % 2 == 0 {
            EntityType::Service
        } else {
            EntityType::Database
        };
        let name = if i % 5 == 0 {
            if i % 10 == 0 {
                format!("PaY-entity-{i}")
            } else {
                format!("pay-entity-{i}")
            }
        } else {
            format!("misc-entity-{i}")
        };
        ok!(
            store.create_entity(CreateEntity {
                entity_type,
                name,
                properties: BTreeMap::new(),
                tags: vec![],
                context_id: ctx,
            }),
            "seed entity"
        );
    }
    drop(store);
    let store = reopen(&db_path);
    let first = ok!(
        store.list_entities(ListEntityFilter {
            name_contains: Some("pay".to_string()),
            entity_type: Some(EntityType::Service),
            include_archived: false,
            ..ListEntityFilter::default()
        }),
        "list entities first"
    );
    let second = ok!(
        store.list_entities(ListEntityFilter {
            name_contains: Some("pay".to_string()),
            entity_type: Some(EntityType::Service),
            include_archived: false,
            ..ListEntityFilter::default()
        }),
        "list entities second"
    );
    let third = ok!(
        store.list_entities(ListEntityFilter {
            name_contains: Some("PAY".to_string()),
            entity_type: Some(EntityType::Service),
            include_archived: false,
            ..ListEntityFilter::default()
        }),
        "list entities uppercase"
    );
    assert_eq!(first, second);
    assert_eq!(first, third);
    assert!(
        first
            .iter()
            .all(|entity| entity.entity_type == EntityType::Service
                && entity.name.to_lowercase().contains("pay"))
    );
    assert!(
        first
            .windows(2)
            .all(|pair| pair[0].created_at > pair[1].created_at
                || (pair[0].created_at == pair[1].created_at && pair[0].id >= pair[1].id))
    );
    let mut samples = Vec::new();
    for _ in 0..20 {
        let started = Instant::now();
        let _ = ok!(
            store.list_entities(ListEntityFilter {
                name_contains: Some("PaY".to_string()),
                entity_type: Some(EntityType::Service),
                include_archived: false,
                ..ListEntityFilter::default()
            }),
            "timed list entities"
        );
        samples.push(started.elapsed());
    }
    samples.sort();
    let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    assert!(p95 <= Duration::from_millis(100));
    let trace = ok!(store.last_query_trace(), "entity query trace");
    assert!(trace.contains("entity_type"));
    assert!(trace.contains("name_contains"));
}

// Use case: a cleanup script tries to "fix" a camera sighting recorded at 08:05 by rewriting its timestamp; the engine refuses even admin overwrites so recorded facts stay the canonical source of truth (Invariant 4).
#[test]
fn observation_immutable_mutation_rejected() {
    let (_tempdir, db_path, store) = fixture();
    let entity = ok!(
        store.create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "camera-target".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id: ContextId::new_v7(),
        }),
        "tx1"
    );
    let observation = ok!(
        store.record_observation(RecordObservation {
            entity_id: entity.id,
            context_id: entity.context_id,
            observation_type: "camera.sighting".to_string(),
            source: "cam-3".to_string(),
            data: map(&[("at", json!("08:05"))]),
            observed_at: Some(Utc::now()),
        }),
        "tx2"
    );
    let overwrite = store
        .admin_overwrite_observation(observation.observation_id, map(&[("at", json!("08:06"))]));
    assert!(matches!(
        overwrite,
        Err(CgError::ImmutableEntity {
            kind: "observation"
        })
    ));
    drop(store);
    let store = reopen(&db_path);
    let fetched = ok!(
        store.get_observation(observation.observation_id),
        "get observation"
    )
    .expect("observation");
    assert_eq!(fetched.data, map(&[("at", json!("08:05"))]));
}

// Use case: an ingest-pipeline refactor tries to rewrite a Digest's summary; the engine refuses any mutation because every Intention and Decision extracted from that Digest carries a provenance edge back to its bytes (Invariant 16).
#[test]
fn digest_immutable_no_update_symbol() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ContextId::new_v7();
    let digest = ok!(
        store.create_digest(create_digest_input(
            ctx,
            "chose nats for transport after NAT-survival discussion"
        )),
        "tx1"
    );
    let overwrite = store.admin_overwrite_digest(digest.id, "rewritten summary".to_string());
    assert!(matches!(
        overwrite,
        Err(CgError::ImmutableEntity { kind: "digest" })
    ));
    drop(store);
    let store = reopen(&db_path);
    let fetched = ok!(store.get_digest(digest.id), "tx4").expect("digest");
    assert_eq!(
        fetched.summary,
        "chose nats for transport after NAT-survival discussion"
    );
    assert_eq!(fetched.embedding, vec![0.25, 0.5, 0.75]);
}

// Use case: a rollout is first recorded as successful, then a week later re-recorded as reverted; the engine replaces the outcome in place and keeps exactly one HAS_OUTCOME edge (Invariant 9).
#[test]
fn outcome_record_idempotent() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "demo")),
        "intention"
    );
    let decision = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "rollout",
            vec![intention.id],
            vec![]
        )),
        "decision"
    );
    ok!(
        store.record_outcome(decision.id, true, None),
        "first outcome"
    );
    ok!(
        store.record_outcome(decision.id, false, Some("reverted".to_string())),
        "second outcome"
    );
    drop(store);
    let store = reopen(&db_path);
    let fetched = ok!(store.get_decision(decision.id), "get decision").expect("decision");
    let outcome = fetched.outcome.expect("outcome");
    assert!(!outcome.success);
    assert_eq!(outcome.notes, Some("reverted".to_string()));
    assert_eq!(
        ok!(
            store.admin_raw_has_outcome_edge_count(decision.id),
            "edge count"
        ),
        1
    );
}

// Use case: two agents reach opposite verdicts on the same Decision simultaneously; the engine serializes them, picks one winner with no blended row, and preserves the loser in the audit trail.
#[test]
fn outcome_record_concurrent_serializable() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "context"
    );
    let intention = ok!(
        store.create_intention(create_intention_input(ctx.id, "demo")),
        "intention"
    );
    let decision = ok!(
        store.create_decision(create_decision_input(
            ctx.id,
            "parallel outcome",
            vec![intention.id],
            vec![]
        )),
        "decision"
    );
    let store = Arc::new(store);
    let decision_id = decision.id;
    let store_a = Arc::clone(&store);
    let store_b = Arc::clone(&store);
    let t_a =
        thread::spawn(move || store_a.record_outcome(decision_id, true, Some("a".to_string())));
    let t_b =
        thread::spawn(move || store_b.record_outcome(decision_id, false, Some("b".to_string())));
    let outcome_a = t_a.join().expect("thread a");
    let outcome_b = t_b.join().expect("thread b");
    assert!(outcome_a.is_ok());
    assert!(outcome_b.is_ok());
    let outcome_a = outcome_a.expect("outcome a");
    let outcome_b = outcome_b.expect("outcome b");
    drop(store);
    let store = reopen(&db_path);
    let fetched = ok!(store.get_decision(decision.id), "get decision").expect("decision");
    let persisted = fetched.outcome.expect("persisted outcome");
    assert!(persisted == outcome_a || persisted == outcome_b);
    assert!(persisted.notes == Some("a".to_string()) || persisted.notes == Some("b".to_string()));
    assert!(persisted.success || persisted.notes != Some("a".to_string()));
    assert!(!persisted.success || persisted.notes != Some("b".to_string()));
    assert_eq!(
        ok!(
            store.admin_raw_has_outcome_edge_count(decision.id),
            "edge count"
        ),
        1
    );
    let audit = ok!(
        store.audit_query(AuditFilter {
            target: Some(AuditTarget::Outcome(decision.id)),
            ..AuditFilter::default()
        }),
        "audit query"
    );
    let loser = if persisted == outcome_a {
        &outcome_b
    } else {
        &outcome_a
    };
    assert!(audit.iter().any(|entry| {
        matches!(entry.target, AuditTarget::Outcome(id) if id == decision.id)
            && entry.outcome.as_ref() == Some(loser)
    }));
}

// Use case: an agent with a stale Decision id tries to record an outcome; the engine refuses so no dangling Outcome row or HAS_OUTCOME edge ever appears pointing at a non-existent Decision.
#[test]
fn outcome_record_not_found() {
    let (_tempdir, db_path, store) = fixture();
    let id = DecisionId::new_v7();
    let result = store.record_outcome(id, true, None);
    assert!(matches!(result, Err(CgError::NotFound { .. })));
    drop(store);
    let store = reopen(&db_path);
    assert_eq!(
        ok!(store.admin_raw_has_outcome_edge_count(id), "edge count"),
        0
    );
    assert!(ok!(store.admin_raw_outcomes_for_decision(id), "raw outcomes").is_empty());
    assert!(
        ok!(
            store.audit_query(AuditFilter {
                target: Some(AuditTarget::Outcome(id)),
                ..AuditFilter::default()
            }),
            "audit query"
        )
        .is_empty()
    );
}

// Use case: a fractional consultant opens a new client-acme engagement with tier and owner labels; labels and properties survive reopen and a second create under the same name is refused by engine uniqueness.
#[test]
fn context_create_persists_with_labels() {
    let (_tempdir, db_path, store) = fixture();
    let created = ok!(
        store.create_context(CreateContext {
            name: "client-acme".to_string(),
            labels: vec!["tier:platinum".to_string(), "owner:j.doe".to_string()],
            properties: map(&[("billing_code", json!("ACME-2026"))]),
        }),
        "create context"
    );
    let duplicate = store.create_context(CreateContext {
        name: "client-acme".to_string(),
        labels: vec![],
        properties: BTreeMap::new(),
    });
    assert!(matches!(
        duplicate,
        Err(CgError::UniqueViolation {
            entity: "context",
            ..
        })
    ));
    drop(store);
    let store = reopen(&db_path);
    let contexts = ok!(store.list_contexts(), "list contexts");
    let persisted = contexts
        .iter()
        .find(|context| context.id == created.id)
        .expect("context");
    assert_eq!(
        persisted.labels,
        vec!["tier:platinum".to_string(), "owner:j.doe".to_string()]
    );
    assert_eq!(persisted.properties["billing_code"], json!("ACME-2026"));
}

// Use case: an ingest pipeline repeatedly reports that site-alpha DEPENDS_ON db-alpha; the engine collapses duplicates to a single edge row so impact-analysis counts are not inflated (Invariant 10).
#[test]
fn relationship_add_idempotent_single_edge() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ContextId::new_v7();
    let e1 = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "site-alpha")),
        "E1"
    );
    let e2 = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Database, "db-alpha")),
        "E2"
    );
    ok!(
        store.add_relationship(e1.id, e2.id, "DEPENDS_ON".to_string()),
        "add first"
    );
    ok!(
        store.add_relationship(e1.id, e2.id, "DEPENDS_ON".to_string()),
        "add second"
    );
    drop(store);
    let store = reopen(&db_path);
    let neighbours = ok!(
        store.traverse(
            e1.id,
            TraverseFilter {
                depth: 1,
                rel: Some("DEPENDS_ON".to_string()),
            },
        ),
        "traverse"
    );
    assert_eq!(neighbours, vec![e2.id]);
    assert_eq!(
        ok!(
            store.admin_raw_edge_count(e1.id, "DEPENDS_ON".to_string(), e2.id),
            "raw count"
        ),
        1
    );
}

// Use case: a dependency-mapping agent passes a typoed Entity id as the edge source; the engine refuses at write time so traversals never hit a dangling half-edge pointing at nothing.
#[test]
fn relationship_add_missing_entity_rejected() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ContextId::new_v7();
    let e2 = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Database, "db-alpha")),
        "E2"
    );
    let bad_from = EntityId::new_v7();
    let result = store.add_relationship(bad_from, e2.id, "DEPENDS_ON".to_string());
    assert!(matches!(
        result,
        Err(CgError::NotFound {
            missing: Some("from"),
            ..
        })
    ));
    drop(store);
    let store = reopen(&db_path);
    let raw = ok!(
        store.admin_raw_edge_scan("DEPENDS_ON".to_string()),
        "edge scan"
    );
    assert!(raw.iter().all(|edge| edge.from != bad_from));
    assert!(raw.iter().all(|edge| edge.to != e2.id));
}
