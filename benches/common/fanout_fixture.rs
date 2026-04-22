use std::collections::BTreeMap;

use context_graph::{ContextId, CreateEntity, EntityId, EntityType, Store};

pub struct FanoutFixtureIds {
    pub hub: EntityId,
    pub neighbors: Vec<EntityId>,
    pub leaves: Vec<Vec<EntityId>>,
}

pub fn seed_fanout_fixture(
    store: &Store,
    context_id: ContextId,
    neighbor_count: usize,
    leaves_per_neighbor: usize,
) -> FanoutFixtureIds {
    let hub = store
        .create_entity(CreateEntity {
            entity_type: EntityType::Service,
            name: "hub".to_string(),
            properties: BTreeMap::new(),
            tags: vec![],
            context_id,
        })
        .expect("seed");

    let mut neighbors: Vec<EntityId> = Vec::with_capacity(neighbor_count);
    for i in 0..neighbor_count {
        let n = store
            .create_entity(CreateEntity {
                entity_type: EntityType::Service,
                name: format!("neighbor-{i}"),
                properties: BTreeMap::new(),
                tags: vec![],
                context_id,
            })
            .expect("seed");
        store
            .add_relationship(hub.id, n.id, "DEPENDS_ON".to_string())
            .expect("seed");
        neighbors.push(n.id);
    }

    let mut leaves: Vec<Vec<EntityId>> = Vec::with_capacity(neighbor_count);
    for (i, n_id) in neighbors.iter().enumerate().take(neighbor_count) {
        let mut row: Vec<EntityId> = Vec::with_capacity(leaves_per_neighbor);
        for j in 0..leaves_per_neighbor {
            let l = store
                .create_entity(CreateEntity {
                    entity_type: EntityType::Service,
                    name: format!("leaf-{i}-{j}"),
                    properties: BTreeMap::new(),
                    tags: vec![],
                    context_id,
                })
                .expect("seed");
            store
                .add_relationship(*n_id, l.id, "RELATES_TO".to_string())
                .expect("seed");
            row.push(l.id);
        }
        leaves.push(row);
    }

    // Fan-in cross edges: for each N_i, add RELATES_TO to leaves[(i+1) % n][0].
    for (i, n_id) in neighbors.iter().enumerate().take(neighbor_count) {
        let target = leaves[(i + 1) % neighbor_count][0];
        store
            .add_relationship(*n_id, target, "RELATES_TO".to_string())
            .expect("seed");
    }

    FanoutFixtureIds {
        hub: hub.id,
        neighbors,
        leaves,
    }
}
