use crate::{DistanceMetric, EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind};

pub type EmbeddingSpaceDecl = EmbeddingSpace;

pub struct BuiltinSpaceRegistry;

impl BuiltinSpaceRegistry {
    pub const TEXT_DEFAULT_ID: &'static str = "text_default_nomic_v1_5_768";
    pub const TEXT_DEFAULT_MODEL_FAMILY: &'static str = "text";
    pub const TEXT_DEFAULT_MODEL_NAME: &'static str = "nomic-embed-text-v1.5";
    pub const TEXT_DEFAULT_MODEL_VERSION: &'static str = "1.5";
    pub const TEXT_DEFAULT_DIMENSION: usize = 768;

    pub(crate) fn text_space_id_for_dimension(dimension: usize) -> String {
        if dimension == Self::TEXT_DEFAULT_DIMENSION {
            Self::TEXT_DEFAULT_ID.to_string()
        } else {
            format!("text_default_nomic_v1_5_{dimension}")
        }
    }
}

pub fn default_text_embedding_space() -> EmbeddingSpace {
    text_nomic_space(BuiltinSpaceRegistry::TEXT_DEFAULT_DIMENSION, true)
}

pub(crate) fn text_nomic_space(dimension: usize, default_for_text: bool) -> EmbeddingSpace {
    let id = BuiltinSpaceRegistry::text_space_id_for_dimension(dimension);
    let mut bindings = vec![EmbeddingSpaceBinding {
        embedding_space_id: id.clone(),
        table_name: "evidence".to_string(),
        vector_column: "vector_text".to_string(),
    }];
    if dimension == BuiltinSpaceRegistry::TEXT_DEFAULT_DIMENSION {
        bindings.push(EmbeddingSpaceBinding {
            embedding_space_id: id.clone(),
            table_name: "digests".to_string(),
            vector_column: "vector_text".to_string(),
        });
    }
    EmbeddingSpace {
        embedding_space_id: id,
        dimension,
        metric: DistanceMetric::Cosine,
        model_family: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_FAMILY.to_string(),
        model_name: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_NAME.to_string(),
        model_version: BuiltinSpaceRegistry::TEXT_DEFAULT_MODEL_VERSION.to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text, EvidenceKind::StructuredSignal],
        default_for_evidence_kind: default_for_text.then_some(EvidenceKind::Text),
        bindings,
    }
}
