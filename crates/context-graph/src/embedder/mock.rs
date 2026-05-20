use crate::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use crate::{
    CgError, DistanceMetric, EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind, EvidenceProducer,
    Result,
};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct MockEmbedder {
    pub embedding_space_id: String,
    pub dimension: usize,
    pub producer: EvidenceProducer,
}

impl MockEmbedder {
    pub fn new(embedding_space_id: impl Into<String>, dimension: usize) -> Self {
        let embedding_space_id = embedding_space_id.into();
        Self {
            embedding_space_id: embedding_space_id.clone(),
            dimension,
            producer: EvidenceProducer {
                system: "cg-edge".to_string(),
                model_name: "mock".to_string(),
                model_version: "0".to_string(),
                pipeline_version: embedding_space_id,
            },
        }
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
        let value = match input {
            EmbeddingInput::Text(text) => text,
            EmbeddingInput::ImageFrame { .. } => {
                return Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::ImageFrame,
                });
            }
            EmbeddingInput::AudioSegment { .. } => {
                return Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::AudioSegment,
                });
            }
            EmbeddingInput::VideoSegment { .. } => {
                return Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::VideoSegment,
                });
            }
            EmbeddingInput::StructuredSignal(_) => {
                return Err(CgError::UnsupportedEvidenceKind {
                    kind: EvidenceKind::StructuredSignal,
                });
            }
        };
        let mut vector = mock_vector("text", &value, self.dimension);
        l2_normalize(&mut vector);
        Ok(EmbeddingOutput {
            embedding_space_id: self.embedding_space_id.clone(),
            dimension: self.dimension,
            producer: self.producer.clone(),
            vector,
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

fn mock_vector(kind: &str, value: &str, dimension: usize) -> Vec<f32> {
    let mut vector = Vec::with_capacity(dimension);
    let mut block = 0_u64;
    while vector.len() < dimension {
        let mut hasher = Sha256::new();
        hasher.update(kind.as_bytes());
        hasher.update([0]);
        hasher.update(value.as_bytes());
        hasher.update(block.to_le_bytes());
        for chunk in hasher.finalize().chunks_exact(4) {
            if vector.len() == dimension {
                break;
            }
            let mut bytes = [0_u8; 4];
            bytes.copy_from_slice(chunk);
            vector.push(u32::from_le_bytes(bytes) as f32 / u32::MAX as f32);
        }
        block += 1;
    }
    vector
}

fn l2_normalize(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|component| component.powi(2))
        .sum::<f32>()
        .sqrt();
    if norm == 0.0 {
        return;
    }
    for component in vector {
        *component /= norm;
    }
}
