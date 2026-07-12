use std::collections::BTreeMap;

use chrono::Utc;
use context_graph::fabric::FabricExecutionReceipt;
use context_graph::{ContextId, EntityId, ObservationId, RecordObservation};

fn main() {
    let receipt = FabricExecutionReceipt {
        executor_node_id: "worker-pod".to_string(),
        total_tokens: 2580,
        wall_clock_ms: 60_000,
    };
    let mut state_delta = BTreeMap::new();
    // A remote-execution receipt must never flow into a decision-basis lane. Once the operational
    // barrier holds, `serde_json::Value: From<FabricExecutionReceipt>` is not implemented, so this
    // `.into()` fails to compile — mirroring the shipped `usage_is_not_a_basis_input` barrier.
    state_delta.insert("fabric.total_tokens".to_string(), Some(receipt.into()));

    let _observation = RecordObservation {
        id: ObservationId::new_v7(),
        entity_id: EntityId::new_v7(),
        context_id: ContextId::new_v7(),
        observation_type: "fabric_execution".to_string(),
        source: "fabric-worker".to_string(),
        observed_at: Utc::now(),
        evidence: Vec::new(),
        observed_properties: BTreeMap::new(),
        state_delta,
        properties: BTreeMap::new(),
        embeddings: Vec::new(),
    };
}
