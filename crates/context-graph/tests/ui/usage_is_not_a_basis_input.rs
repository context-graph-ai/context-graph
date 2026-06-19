use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::{
    CapturedUsage, ContextId, EntityId, ObservationId, RecordObservation,
};

fn main() {
    let usage = CapturedUsage { total_tokens: 2580 };
    let mut state_delta = BTreeMap::new();
    state_delta.insert("usage.total_tokens".to_string(), Some(usage.into()));

    let _observation = RecordObservation {
        id: ObservationId::new_v7(),
        entity_id: EntityId::new_v7(),
        context_id: ContextId::new_v7(),
        observation_type: "provider_usage".to_string(),
        source: "model-provider".to_string(),
        observed_at: Utc::now(),
        evidence: Vec::new(),
        observed_properties: BTreeMap::new(),
        state_delta,
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    };
}
