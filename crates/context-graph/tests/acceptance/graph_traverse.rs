use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use context_graph::{
    CgError, ContextId, CreateContext, CreateEntity, EntityId, EntityType, RelationshipType, Store,
    TraverseFilter,
};
use tempfile::TempDir;

#[path = "../../../../benches/common/fanout_fixture.rs"]
mod fanout_fixture;

use fanout_fixture::seed_fanout_fixture;

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

fn context_input(name: &str) -> CreateContext {
    CreateContext {
        name: name.to_string(),
        labels: vec![],
        properties: BTreeMap::new(),
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

// Use case: An infrastructure operator at site-alpha asks the graph for every Entity its gate sensor ultimately depends on — four hops out. After a process restart the same walk returns the same answer because the edges live on disk, not in memory.
#[test]
fn traverse_bfs_four_hop_after_reopen() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("site-alpha")),
        "tx0: create context"
    )
    .id;
    let e1 = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "gate-sensor")),
        "tx1: create E1"
    );
    let e2 = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Service,
            "site-controller"
        )),
        "tx2: create E2"
    );
    let e3 = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Infrastructure,
            "net-switch"
        )),
        "tx3: create E3"
    );
    let e4 = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Infrastructure,
            "regional-hub"
        )),
        "tx4: create E4"
    );
    ok!(
        store.add_relationship(e1.id, e2.id, "RELATES_TO".to_string()),
        "tx5: E1->E2"
    );
    ok!(
        store.add_relationship(e2.id, e3.id, "RELATES_TO".to_string()),
        "tx6: E2->E3"
    );
    ok!(
        store.add_relationship(e3.id, e4.id, "RELATES_TO".to_string()),
        "tx7: E3->E4"
    );
    drop(store);
    let store = reopen(&db_path);
    let result = ok!(
        store.traverse(
            e1.id,
            TraverseFilter {
                depth: 5,
                rel: None,
            },
        ),
        "tx8: traverse"
    );
    assert_eq!(result.nodes.len(), 4);
    let expected_ids = [e1.id, e2.id, e3.id, e4.id];
    for (i, expected) in expected_ids.iter().enumerate() {
        assert_eq!(result.nodes[i].entity_id, *expected);
        assert_eq!(result.nodes[i].hop, i as u8);
    }
    assert_eq!(result.nodes[0].relationship, None);
    for i in 1..4 {
        assert_eq!(
            result.nodes[i].relationship,
            Some(RelationshipType::RelatesTo)
        );
    }
    let actual_set: HashSet<EntityId> = result.nodes.iter().map(|n| n.entity_id).collect();
    let expected_set: HashSet<EntityId> = expected_ids.iter().copied().collect();
    assert_eq!(actual_set, expected_set);
}

// Use case: An incident reviewer at client-acme asks the engine for just the hard DEPENDS_ON chain out of a flapping service — not the broader RELATES_TO neighbours. Engine-side edge-type filter means no out-of-filter rows leave the store.
#[test]
fn traverse_with_relationship_filter() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let e1 = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Service,
            "checkout-api"
        )),
        "tx1: E1"
    );
    let e2 = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Database,
            "checkout-db"
        )),
        "tx2: E2"
    );
    let e3 = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Service,
            "inventory-api"
        )),
        "tx3: E3"
    );
    ok!(
        store.add_relationship(e1.id, e2.id, "DEPENDS_ON".to_string()),
        "tx4: DEPENDS_ON"
    );
    ok!(
        store.add_relationship(e1.id, e3.id, "RELATES_TO".to_string()),
        "tx5: RELATES_TO"
    );
    drop(store);
    let store = reopen(&db_path);
    let result = ok!(
        store.traverse(
            e1.id,
            TraverseFilter {
                depth: 2,
                rel: Some(RelationshipType::DependsOn),
            },
        ),
        "tx6: traverse"
    );
    assert!(result.nodes.iter().any(|n| n.entity_id == e1.id));
    assert!(result.nodes.iter().any(|n| n.entity_id == e2.id));
    assert!(!result.nodes.iter().any(|n| n.entity_id == e3.id));
}

// Use case: A dashboard at client-acme queries the neighbourhood of an Entity at depth 0 as a clean identity lookup. The engine returns the start entity only — edges are not loaded.
#[test]
fn traverse_depth_zero() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let e1 = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Service,
            "auth-service"
        )),
        "tx1: E1"
    );
    let e2 = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Service,
            "peer-service"
        )),
        "tx2: E2"
    );
    ok!(
        store.add_relationship(e1.id, e2.id, "RELATES_TO".to_string()),
        "tx3: add edge"
    );
    drop(store);
    let store = reopen(&db_path);
    let result = ok!(
        store.traverse(
            e1.id,
            TraverseFilter {
                depth: 0,
                rel: None,
            },
        ),
        "tx4: traverse"
    );
    assert_eq!(result.nodes.len(), 1);
    assert_eq!(result.nodes[0].entity_id, e1.id);
    assert_eq!(result.nodes[0].hop, 0);
    assert!(result.nodes[0].relationship.is_none());
    assert!(result.nodes.iter().all(|n| n.entity_id != e2.id));
}

// Use case: An agent at client-acme walks a microservices cluster with a legitimate circular dependency — A depends on B, B on C, C back on A for service discovery. The engine returns each node exactly once; the visited-set is the cycle-safety mechanism.
#[test]
fn traverse_cycle_terminates_without_duplicates() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let a = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "A")),
        "tx1: A"
    );
    let b = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "B")),
        "tx2: B"
    );
    let c = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "C")),
        "tx3: C"
    );
    ok!(
        store.add_relationship(a.id, b.id, "DEPENDS_ON".to_string()),
        "tx4: A->B"
    );
    ok!(
        store.add_relationship(b.id, c.id, "DEPENDS_ON".to_string()),
        "tx5: B->C"
    );
    ok!(
        store.add_relationship(c.id, a.id, "DEPENDS_ON".to_string()),
        "tx6: C->A"
    );
    drop(store);
    let store = reopen(&db_path);
    let result = ok!(
        store.traverse(
            a.id,
            TraverseFilter {
                depth: 10,
                rel: None,
            },
        ),
        "tx7: traverse"
    );
    assert_eq!(result.nodes.len(), 3);
    let actual: HashSet<EntityId> = result.nodes.iter().map(|n| n.entity_id).collect();
    let expected: HashSet<EntityId> = [a.id, b.id, c.id].into_iter().collect();
    assert_eq!(actual, expected);
}

// Use case: An agent at client-acme fires a traversal with depth 20 thinking bigger is better. The engine refuses loudly with `CgError::InvalidDepth` rather than clamping silently — the cap-10 contract is enforced as input validation before any BFS work.
#[test]
fn traverse_depth_cap_rejects_over_ten() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let e0 = ok!(
        store.create_entity(create_entity_input(ctx, EntityType::Service, "E0")),
        "tx1: E0"
    );
    drop(store);
    let store = reopen(&db_path);
    let err = store
        .traverse(
            e0.id,
            TraverseFilter {
                depth: 20,
                rel: None,
            },
        )
        .unwrap_err();
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
    let ok_r = ok!(
        store.traverse(
            e0.id,
            TraverseFilter {
                depth: 10,
                rel: None,
            },
        ),
        "tx2: cap accepted"
    );
    assert!(!ok_r.nodes.is_empty());
}

// Use case: A dashboard at client-acme queries the neighbourhood of a stale Entity id that was archived or never existed. The engine returns `Ok` with an empty node list — dashboards render a blank card, not an exception.
#[test]
fn traverse_missing_start_returns_empty() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("client-acme")),
        "tx0: context"
    )
    .id;
    let real = ok!(
        store.create_entity(create_entity_input(
            ctx,
            EntityType::Service,
            "real-service"
        )),
        "tx1: real entity"
    );
    let phantom = EntityId::new_v7();
    let result = ok!(
        store.traverse(
            phantom,
            TraverseFilter {
                depth: 1,
                rel: None,
            },
        ),
        "tx2: traverse phantom"
    );
    assert!(result.nodes.is_empty());
    // Contrast with real start: depth-1 traversal of a seeded entity must
    // return at least the start node. An always-empty stub fails here.
    let real_result = ok!(
        store.traverse(
            real.id,
            TraverseFilter {
                depth: 1,
                rel: None,
            },
        ),
        "tx3: traverse real"
    );
    assert!(!real_result.nodes.is_empty());
    assert!(real_result.nodes.iter().any(|n| n.entity_id == real.id));
    drop(store);
    let store = reopen(&db_path);
    let result = ok!(
        store.traverse(
            phantom,
            TraverseFilter {
                depth: 1,
                rel: None,
            },
        ),
        "tx2: traverse phantom after reopen"
    );
    assert!(result.nodes.is_empty());
}

// Use case: An operator at a large tenant asks for the two-hop neighbourhood of a hub service with multiple direct dependents, each with further neighbours across two edge types. The engine returns every reachable Entity exactly once after a restart; de-duplication and durable indexing are engine properties, not wrapper logic.
#[test]
fn traverse_fanout_heavy_path() {
    let (_tempdir, db_path, store) = fixture();
    let ctx = ok!(
        store.create_context(context_input("large-tenant")),
        "tx0: context"
    )
    .id;
    let ids = seed_fanout_fixture(&store, ctx, 3, 2);
    drop(store);
    let store = reopen(&db_path);
    let result = ok!(
        store.traverse(
            ids.hub,
            TraverseFilter {
                depth: 2,
                rel: None,
            },
        ),
        "tx1: traverse"
    );
    assert_eq!(result.nodes.len(), 10);
    let mut expected: HashSet<EntityId> = HashSet::new();
    expected.insert(ids.hub);
    for n in &ids.neighbors {
        expected.insert(*n);
    }
    for row in &ids.leaves {
        for l in row {
            expected.insert(*l);
        }
    }
    let actual: HashSet<EntityId> = result.nodes.iter().map(|n| n.entity_id).collect();
    assert_eq!(actual, expected);

    let start_nodes: Vec<_> = result
        .nodes
        .iter()
        .filter(|n| n.entity_id == ids.hub)
        .collect();
    assert_eq!(start_nodes.len(), 1);
    assert_eq!(start_nodes[0].hop, 0);
    assert_eq!(start_nodes[0].relationship, None);

    let neighbor_set: HashSet<EntityId> = ids.neighbors.iter().copied().collect();
    for node in result
        .nodes
        .iter()
        .filter(|n| neighbor_set.contains(&n.entity_id))
    {
        assert_eq!(node.hop, 1);
        assert_eq!(node.relationship, Some(RelationshipType::DependsOn));
    }

    let leaf_set: HashSet<EntityId> = ids.leaves.iter().flatten().copied().collect();
    for node in result
        .nodes
        .iter()
        .filter(|n| leaf_set.contains(&n.entity_id))
    {
        assert_eq!(node.hop, 2);
        assert_eq!(node.relationship, Some(RelationshipType::RelatesTo));
    }
}
