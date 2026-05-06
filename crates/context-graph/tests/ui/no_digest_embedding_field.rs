use std::collections::BTreeMap;

use context_graph::{ContextId, Digest, DigestId, TxId};

fn main() {
    let _ = Digest {
        id: DigestId::new_v7(),
        source: "claude-cli".to_string(),
        summary: "summary".to_string(),
        vector_text: Some(vec![0.0; 768]),
        dimension: 768,
        context_id: ContextId::new_v7(),
        session_ref: Some("session-1".to_string()),
        properties: BTreeMap::new(),
        created_at: TxId(1),
        embedding: vec![0.0; 384],
    };
}
