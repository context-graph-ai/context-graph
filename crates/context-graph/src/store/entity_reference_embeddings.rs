use serde::{Deserialize, Serialize};

use crate::{CgError, EmbeddingId, EntityId, EvidenceProducer, Result};

use super::Store;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityReferenceEmbeddingRow {
    pub id: EmbeddingId,
    pub entity_id: EntityId,
    pub embedding_space_id: String,
    pub label: Option<String>,
    pub vector: Vec<f32>,
    pub produced_by: EvidenceProducer,
    pub created_at: crate::TxId,
}

impl Store {
    pub fn enroll_entity_reference(
        &self,
        entity_id: EntityId,
        embedding_space_id: &str,
        label: Option<String>,
        vector: Vec<f32>,
        produced_by: EvidenceProducer,
    ) -> Result<EmbeddingId> {
        let _ = (entity_id, embedding_space_id, label, vector, produced_by);
        Ok(EmbeddingId::default())
    }

    pub fn list_entity_reference_embeddings_for_entity(
        &self,
        entity_id: EntityId,
    ) -> Result<Vec<EntityReferenceEmbeddingRow>> {
        Err(CgError::NotFound {
            id: Some(entity_id.to_string()),
            missing: Some("entity_reference_embedding"),
        })
    }
}

#[cfg(context_graph_internal_test_support)]
pub fn entity_reference_cosine_match(
    _store: &Store,
    _live_space: &str,
    _reference_embedding_id: EmbeddingId,
) -> Result<()> {
    Ok(())
}
