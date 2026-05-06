use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::{ContextId, EntityId, Observation, ObservationId, TxId};

fn main() {
    let _ = Observation {
        id: ObservationId::new_v7(),
        observation_type: "config_change".to_string(),
        evidence: Vec::new(),
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        source: "git-hook".to_string(),
        entity_id: EntityId::new_v7(),
        context_id: ContextId::new_v7(),
        observed_at: Utc::now(),
        created_at: TxId(1),
        data: BTreeMap::new(),
    };
}
