use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use chrono::Utc;
use context_graph::{
    ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId, EntityId,
    EntityType, IntentionId, IntentionOrigin, IntentionStatus, ReasoningStep, Store, StoreConfig,
    TraverseFilter,
};
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

// Use case: A fractional consultant juggling client-a and client-b opens their agent's graph view for client-a. They need trust that exploring client-a's neighborhood, walking client-a's precedent chain, or expanding client-a's intention tree never surfaces a single client-b row. One leak destroys the client relationship.
#[test]
fn graph_operations_respect_context_isolation() {
    let (_tempdir, db_path, store) = fixture();
    let ctx_a = ok!(
        store.create_context(context_input("client-a")),
        "tx0: ctx_a"
    )
    .id;
    let ctx_b = ok!(
        store.create_context(context_input("client-b")),
        "tx1: ctx_b"
    )
    .id;

    // Client A parallel graph.
    let i_a = ok!(
        store.create_intention(create_intention_input(ctx_a, "client-a auth")),
        "tx2: I_a"
    );
    let e_a1 = ok!(
        store.create_entity(create_entity_input(
            ctx_a,
            EntityType::Service,
            "a-auth-service"
        )),
        "tx3: E_a1"
    );
    let e_a2 = ok!(
        store.create_entity(create_entity_input(
            ctx_a,
            EntityType::Database,
            "a-token-store"
        )),
        "tx4: E_a2"
    );
    ok!(
        store.add_relationship(e_a1.id, e_a2.id, "DEPENDS_ON".to_string()),
        "tx5: a edges"
    );
    let d_a_1 = ok!(
        store.create_decision(create_decision_input(
            ctx_a,
            "a baseline",
            vec![i_a.id],
            vec![e_a1.id],
            vec![],
        )),
        "tx6: D_a_1"
    );
    let d_a_2 = ok!(
        store.create_decision(create_decision_input(
            ctx_a,
            "a cites baseline",
            vec![i_a.id],
            vec![e_a1.id],
            vec![d_a_1.id],
        )),
        "tx7: D_a_2"
    );

    // Client B parallel graph.
    let i_b = ok!(
        store.create_intention(create_intention_input(ctx_b, "client-b auth")),
        "tx8: I_b"
    );
    let e_b1 = ok!(
        store.create_entity(create_entity_input(
            ctx_b,
            EntityType::Service,
            "b-auth-service"
        )),
        "tx9: E_b1"
    );
    let e_b2 = ok!(
        store.create_entity(create_entity_input(
            ctx_b,
            EntityType::Database,
            "b-token-store"
        )),
        "tx10: E_b2"
    );
    ok!(
        store.add_relationship(e_b1.id, e_b2.id, "DEPENDS_ON".to_string()),
        "tx11: b edges"
    );
    let d_b_1 = ok!(
        store.create_decision(create_decision_input(
            ctx_b,
            "b baseline",
            vec![i_b.id],
            vec![e_b1.id],
            vec![],
        )),
        "tx12: D_b_1"
    );
    let d_b_2 = ok!(
        store.create_decision(create_decision_input(
            ctx_b,
            "b cites baseline",
            vec![i_b.id],
            vec![e_b1.id],
            vec![d_b_1.id],
        )),
        "tx13: D_b_2"
    );

    drop(store);
    let store = reopen(&db_path);

    let mut b_ids: HashSet<Uuid> = HashSet::new();
    b_ids.insert(i_b.id.as_uuid());
    b_ids.insert(e_b1.id.as_uuid());
    b_ids.insert(e_b2.id.as_uuid());
    b_ids.insert(d_b_1.id.as_uuid());
    b_ids.insert(d_b_2.id.as_uuid());

    let trav = ok!(
        store.traverse(
            e_a1.id,
            TraverseFilter {
                depth: 5,
                rel: None,
            },
        ),
        "tx14: traverse"
    );
    for node in &trav.nodes {
        assert!(
            !b_ids.contains(&node.entity_id.as_uuid()),
            "traverse leaked client-b id {:?}",
            node.entity_id
        );
    }

    let chain = ok!(
        store.precedent_chain(d_a_2.id, 10_u8),
        "tx15: precedent_chain"
    );
    for d in &chain {
        assert!(
            !b_ids.contains(&d.id.as_uuid()),
            "precedent_chain leaked client-b decision {:?}",
            d.id
        );
    }

    let tree = ok!(store.intention_tree(i_a.id), "tx16: intention_tree");
    assert!(!b_ids.contains(&tree.intention.id.as_uuid()));
    for d in &tree.decisions {
        assert!(
            !b_ids.contains(&d.id.as_uuid()),
            "intention_tree leaked client-b decision {:?}",
            d.id
        );
    }
    for e in &tree.entities {
        assert!(
            !b_ids.contains(&e.id.as_uuid()),
            "intention_tree leaked client-b entity {:?}",
            e.id
        );
    }
    for o in &tree.outcomes {
        assert!(
            !b_ids.contains(&o.decision_id.as_uuid()),
            "intention_tree outcome references client-b decision {:?}",
            o.decision_id
        );
    }
    for inv in &tree.invalidations {
        assert!(
            !b_ids.contains(&inv.affected_decision_id.as_uuid()),
            "intention_tree invalidation references client-b decision {:?}",
            inv.affected_decision_id
        );
    }

    let nb = ok!(
        store.entity_neighborhood(e_a1.id, 2_u8),
        "tx17: entity_neighborhood"
    );
    for e in &nb.entities {
        assert!(
            !b_ids.contains(&e.id.as_uuid()),
            "neighborhood leaked client-b entity {:?}",
            e.id
        );
    }
    for d in &nb.decisions {
        assert!(
            !b_ids.contains(&d.id.as_uuid()),
            "neighborhood leaked client-b decision {:?}",
            d.id
        );
    }
    for n in &nb.intentions {
        assert!(
            !b_ids.contains(&n.id.as_uuid()),
            "neighborhood leaked client-b intention {:?}",
            n.id
        );
    }

    // Strength gate: the stub returns empty collections. To guarantee a real
    // implementation surfaces *something* for the A side (not just any empty
    // vec that trivially passes the disjoint check), bind known-A content
    // to a lower-bound assertion here.
    let a_surfaced = !trav.nodes.is_empty() && !chain.is_empty() && !nb.entities.is_empty();
    assert!(
        a_surfaced,
        "graph operations produced no client-a rows — the disjoint check would pass trivially"
    );
}
