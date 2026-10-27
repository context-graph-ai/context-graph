use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::{
    CgError, ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId,
    EntityId, EntityType, IntentionId, IntentionOrigin, IntentionStatus, ReasoningStep, Store,
    StoreConfig,
};
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
        blueprint_catalog_id: None,
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

fn create_decision_input(
    context_id: ContextId,
    description: &str,
    intention_ids: Vec<IntentionId>,
    based_on_entity_ids: Vec<EntityId>,
    precedent_ids: Vec<DecisionId>,
) -> CreateDecision {
    CreateDecision {
        decision_type: "architecture".to_string(),
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

// Use case: An engineer at client-acme is about to change an entity and asks "what sits within two hops across any edge type?" The engine returns set-equal collections of reachable Entities, Decisions, and Intentions — not a subset, not a superset.
#[test]
fn entity_neighborhood_returns_everything_connected() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let i = ok!(
        store.create_intention(create_intention_input(ctx, "keep-center-healthy")),
        "tx1: I"
    );
    let e_center = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "E_center")),
        "tx2: E_center"
    );
    let e_peer1 = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "E_peer1")),
        "tx3: E_peer1"
    );
    let e_peer2 = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "E_peer2")),
        "tx4: E_peer2"
    );
    let e_far = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "E_far")),
        "tx5: E_far"
    );
    let e_disconnected = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Service,
            "E_disconnected"
        )),
        "tx6: E_disconnected"
    );
    let e_too_far = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "E_too_far")),
        "tx7: E_too_far"
    );
    let e_way_too_far = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Service,
            "E_way_too_far"
        )),
        "tx8: E_way_too_far"
    );
    ok!(
        store.add_relationship(e_peer1.id, e_center.id, "DEPENDS_ON".to_string()),
        "tx9: E_peer1 -> E_center"
    );
    ok!(
        store.add_relationship(e_peer2.id, e_center.id, "DEPENDS_ON".to_string()),
        "tx10: E_peer2 -> E_center"
    );
    ok!(
        store.add_relationship(e_far.id, e_peer1.id, "RELATES_TO".to_string()),
        "tx11: E_far -> E_peer1"
    );
    ok!(
        store.add_relationship(e_too_far.id, e_far.id, "RELATES_TO".to_string()),
        "tx12: E_too_far -> E_far (depth 3)"
    );
    ok!(
        store.add_relationship(e_way_too_far.id, e_too_far.id, "RELATES_TO".to_string()),
        "tx13: E_way_too_far -> E_too_far (depth 4)"
    );
    let d_center = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "baseline D_center",
            vec![i.id],
            vec![e_center.id],
            vec![],
        )),
        "tx14: D_center"
    );
    let i_other = ok!(
        store.create_intention(create_intention_input(ctx, "unrelated intention")),
        "tx15: I_other"
    );
    let _d_other = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "D_other on disconnected",
            vec![i_other.id],
            vec![e_disconnected.id],
            vec![],
        )),
        "tx16: D_other"
    );
    drop(store);
    let store = reopen(&db_path);
    let nb = ok!(
        store.entity_neighborhood(e_center.id, 2_u8),
        "tx17: neighborhood"
    );
    let entity_ids: HashSet<EntityId> = nb.entities.iter().map(|e| e.id).collect();
    let expected_entities: HashSet<EntityId> = [e_center.id, e_peer1.id, e_peer2.id, e_far.id]
        .into_iter()
        .collect();
    assert_eq!(entity_ids, expected_entities);

    let decision_ids: HashSet<DecisionId> = nb.decisions.iter().map(|d| d.id).collect();
    let expected_decisions: HashSet<DecisionId> = [d_center.id].into_iter().collect();
    assert_eq!(decision_ids, expected_decisions);

    let intention_ids: HashSet<IntentionId> = nb.intentions.iter().map(|x| x.id).collect();
    let expected_intentions: HashSet<IntentionId> = [i.id].into_iter().collect();
    assert_eq!(intention_ids, expected_intentions);
}

// Use case: An audit reader asks for the Decisions attached to a hub within one hop. The engine includes Decisions on the hub and its direct neighbours but excludes Decisions on entities at depth > 1.
#[test]
fn subgraph_decisions_by_depth() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let i = ok!(
        store.create_intention(create_intention_input(ctx, "subgraph-depth")),
        "tx1: I"
    );
    let e_hub = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "E_hub")),
        "tx2: E_hub"
    );
    let e_near = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "E_near")),
        "tx3: E_near"
    );
    let e_out = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "E_out")),
        "tx4: E_out"
    );
    ok!(
        store.add_relationship(e_hub.id, e_near.id, "DEPENDS_ON".to_string()),
        "tx5: E_hub -> E_near"
    );
    let d_near = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "D_near",
            vec![i.id],
            vec![e_near.id],
            vec![],
        )),
        "tx6: D_near"
    );
    let _d_out = ok!(
        store.create_decision(create_decision_input(
            ctx,
            "D_out",
            vec![i.id],
            vec![e_out.id],
            vec![],
        )),
        "tx7: D_out"
    );
    drop(store);
    let store = reopen(&db_path);
    let nb = ok!(
        store.entity_neighborhood(e_hub.id, 1_u8),
        "tx8: neighborhood"
    );
    let entity_ids: HashSet<EntityId> = nb.entities.iter().map(|e| e.id).collect();
    let expected_entities: HashSet<EntityId> = [e_hub.id, e_near.id].into_iter().collect();
    assert_eq!(entity_ids, expected_entities);

    let decision_ids: HashSet<DecisionId> = nb.decisions.iter().map(|d| d.id).collect();
    let expected_decisions: HashSet<DecisionId> = [d_near.id].into_iter().collect();
    assert_eq!(decision_ids, expected_decisions);

    let intention_ids: HashSet<IntentionId> = nb.intentions.iter().map(|x| x.id).collect();
    let expected_intentions: HashSet<IntentionId> = [i.id].into_iter().collect();
    assert_eq!(intention_ids, expected_intentions);
}

// Use case: An agent at client-acme asks for a neighbourhood at depth 20. The engine refuses with the same `CgError::InvalidDepth` it applies to every graph walk — the cap-10 contract is uniform across `traverse`, `precedent_chain`, and `entity_neighborhood`.
#[test]
fn subgraph_hard_cap_ten() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let e_hub = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "E_hub")),
        "tx1: E_hub"
    );
    drop(store);
    let store = reopen(&db_path);
    let err = store.entity_neighborhood(e_hub.id, 20_u8).unwrap_err();
    assert!(
        matches!(
            err,
            CgError::InvalidDepth {
                requested: 20,
                cap: 10
            }
        ),
        "expected InvalidDepth {{ requested: 20, cap: 10 }}, got {:?}",
        err
    );
    let ok_nb = ok!(
        store.entity_neighborhood(e_hub.id, 10_u8),
        "tx2: cap accepted"
    );
    assert!(!ok_nb.entities.is_empty());
}
