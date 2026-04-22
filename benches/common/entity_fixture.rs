use std::collections::BTreeMap;

use context_graph::{ContextId, CreateEntity, Entity, EntityType, Store};

pub fn seed_entity_fixture(store: &Store, context_id: ContextId, count: usize) -> Vec<Entity> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
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
        let created = store
            .create_entity(CreateEntity {
                entity_type,
                name,
                properties: BTreeMap::new(),
                tags: vec![],
                context_id,
            })
            .expect("seed entity");
        out.push(created);
    }
    out
}
