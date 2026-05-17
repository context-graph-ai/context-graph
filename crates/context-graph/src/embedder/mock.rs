use crate::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use crate::{
    DistanceMetric, EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind, EvidenceProducer, Result,
};

#[derive(Debug, Clone)]
pub struct MockEmbedder {
    pub embedding_space_id: String,
    pub dimension: usize,
    pub producer: EvidenceProducer,
}

impl MockEmbedder {
    pub fn new(embedding_space_id: impl Into<String>, dimension: usize) -> Self {
        Self {
            embedding_space_id: embedding_space_id.into(),
            dimension,
            producer: EvidenceProducer {
                system: "cg-edge".to_string(),
                model_name: "mock".to_string(),
                model_version: "0".to_string(),
                pipeline_version: "0".to_string(),
            },
        }
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        Ok(EmbeddingOutput {
            embedding_space_id: self.producer.model_name.clone(),
            dimension: self.dimension,
            producer: self.producer.clone(),
            vector: vec![0.0; self.dimension],
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        EmbeddingSpace {
            embedding_space_id: self.embedding_space_id.clone(),
            dimension: self.dimension,
            metric: DistanceMetric::Cosine,
            model_family: "mock".to_string(),
            model_name: self.producer.model_name.clone(),
            model_version: self.producer.model_version.clone(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: None,
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: self.embedding_space_id.clone(),
                table_name: "evidence".to_string(),
                vector_column: "vector_text".to_string(),
            }],
        }
    }
}
