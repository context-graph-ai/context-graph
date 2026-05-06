use std::collections::BTreeMap;

use context_graph::{ContextId, CreateDigest, DigestId, EvidenceKind};

fn main() {
    let _ = CreateDigest {
        id: DigestId::new_v7(),
        source: "claude-cli".to_string(),
        session_ref: Some("session-1".to_string()),
        summary: "summary".to_string(),
        kind: EvidenceKind::Text,
        context_id: ContextId::new_v7(),
        properties: BTreeMap::new(),
        embedding: vec![0.0; 384],
    };
}
