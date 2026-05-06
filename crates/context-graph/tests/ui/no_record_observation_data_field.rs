use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::{ContextId, EntityId, ObservationId, RecordObservation};

fn main() {
    let _ = RecordObservation {
        id: ObservationId::new_v7(),
        entity_id: EntityId::new_v7(),
        context_id: ContextId::new_v7(),
        observation_type: "file_rename".to_string(),
        source: "agent-session".to_string(),
        observed_at: Utc::now(),
        evidence: Vec::new(),
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
        data: BTreeMap::new(),
    };
}
