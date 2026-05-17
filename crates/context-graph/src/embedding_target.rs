pub use crate::types::EmbeddingTarget;

use crate::EmbeddingTarget as Target;

pub fn target_table(target: &Target) -> &'static str {
    match target {
        Target::Evidence(_) => "evidence_embedding_handles",
        Target::EntityReference(_, _) => "entity_reference_embedding_handles",
        Target::Digest(_) => "digest_embedding_handles",
    }
}
