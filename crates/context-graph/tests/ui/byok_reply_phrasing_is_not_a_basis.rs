use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::{ContextId, EntityId, ModelReply, ObservationId, RecordObservation};

fn main() {
    let reply = ModelReply {
        marker: "assistant-advisory".to_string(),
        phrasing: "model text must not become basis".to_string(),
        advisory_flag: true,
        malformed: false,
    };

    let _observation = RecordObservation {
        id: ObservationId::new_v7(),
        entity_id: EntityId::new_v7(),
        context_id: ContextId::new_v7(),
        observation_type: "change".to_string(),
        source: "agent-session".to_string(),
        observed_at: Utc::now(),
        evidence: vec![reply.phrasing.into()],
        observed_properties: BTreeMap::new(),
        state_delta: BTreeMap::new(),
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    };
}
