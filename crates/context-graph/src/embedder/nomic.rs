use crate::embedder::{Embedder, EmbedderConfig, EmbeddingInput, EmbeddingOutput};
use crate::{
    CgError, DistanceMetric, EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind, EvidenceProducer,
    Result,
};

pub const NOMIC_TEXT_DEFAULT_ID: &str = "text_default_nomic_v1_5_768";
pub const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
pub const NOMIC_DEFAULT_DIMENSION: usize = 768;

#[derive(Debug, Clone)]
pub struct NomicTextEmbedder {
    pub config: EmbedderConfig,
    pub dimension: usize,
}

impl NomicTextEmbedder {
    pub fn new(config: EmbedderConfig) -> Self {
        Self::new_for_dimension(config, NOMIC_DEFAULT_DIMENSION)
    }

    pub fn new_for_dimension(config: EmbedderConfig, dimension: usize) -> Self {
        Self { config, dimension }
    }
}

impl Embedder for NomicTextEmbedder {
    fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
        if !matches!(input, EmbeddingInput::Text(_)) {
            return Err(CgError::UnsupportedEvidenceKind {
                kind: match input {
                    EmbeddingInput::Text(_) => EvidenceKind::Text,
                    EmbeddingInput::ImageFrame { .. } => EvidenceKind::ImageFrame,
                    EmbeddingInput::AudioSegment { .. } => EvidenceKind::AudioSegment,
                    EmbeddingInput::VideoSegment { .. } => EvidenceKind::VideoSegment,
                    EmbeddingInput::StructuredSignal(_) => EvidenceKind::StructuredSignal,
                },
            });
        }
        Ok(EmbeddingOutput {
            embedding_space_id: self.embedding_space().embedding_space_id,
            dimension: self.dimension,
            producer: EvidenceProducer {
                system: crate::CG_EDGE_SYSTEM_TAG.to_string(),
                model_name: NOMIC_MODEL_ID.to_string(),
                model_version: "1.5".to_string(),
                pipeline_version: "nomic-stub".to_string(),
            },
            vector: vec![0.0; self.dimension],
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: if self.dimension == NOMIC_DEFAULT_DIMENSION {
                NOMIC_TEXT_DEFAULT_ID.to_string()
            } else {
                format!("text_default_nomic_v1_5_{}", self.dimension)
            },
            dimension: self.dimension,
            metric: DistanceMetric::Cosine,
            model_family: "text".to_string(),
            model_name: NOMIC_MODEL_ID.to_string(),
            model_version: "1.5".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: (self.dimension == NOMIC_DEFAULT_DIMENSION)
                .then_some(EvidenceKind::Text),
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: if self.dimension == NOMIC_DEFAULT_DIMENSION {
                    NOMIC_TEXT_DEFAULT_ID.to_string()
                } else {
                    format!("text_default_nomic_v1_5_{}", self.dimension)
                },
                table_name: "evidence".to_string(),
                vector_column: "vector_text".to_string(),
            }],
        }
    }
}
