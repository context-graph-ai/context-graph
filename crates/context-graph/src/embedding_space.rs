use crate::{CgError, DistanceMetric, EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind, Result};

pub type EmbeddingSpaceDecl = EmbeddingSpace;

pub struct BuiltinSpaceRegistry;

impl BuiltinSpaceRegistry {
    pub const TEXT_DEFAULT_ID: &'static str = "text_default_embeddinggemma_300m_768";
    pub const TEXT_DEFAULT_MODEL_NAME: &'static str = "embeddinggemma-300m";
    pub const TEXT_DEFAULT_MODEL_VERSION: &'static str = "default";
    pub const TEXT_DEFAULT_DIMENSION: usize = 768;
}

pub fn default_text_embedding_space() -> EmbeddingSpace {
    EmbeddingSpace {
        embedding_space_id: BuiltinSpaceRegistry::TEXT_DEFAULT_ID.to_string(),
        dimension: BuiltinSpaceRegistry::TEXT_DEFAULT_DIMENSION,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_NAME.to_string(),
        model_version: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_VERSION.to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text, EvidenceKind::StructuredSignal],
        default_for_evidence_kind: Some(EvidenceKind::Text),
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: BuiltinSpaceRegistry::TEXT_DEFAULT_ID.to_string(),
            table_name: "evidence".to_string(),
            vector_column: "vector_text".to_string(),
        }],
    }
}

pub fn validate_extra_spaces(extra: &[EmbeddingSpaceDecl]) -> Result<()> {
    if extra
        .iter()
        .any(|space| space.embedding_space_id.is_empty())
    {
        return Err(CgError::InvalidEmbeddingSpaceId { reason: "empty" });
    }
    Ok(())
}
