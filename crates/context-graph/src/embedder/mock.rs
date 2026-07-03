use crate::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use crate::embedding_space::default_text_embedding_space;
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
            EmbeddingInput::ImageFrame { .. } | EmbeddingInput::ImageBytes(..) => {
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

#[derive(Debug, Clone)]
pub struct LexicalMockEmbedder {
    pub producer: EvidenceProducer,
}

impl Default for LexicalMockEmbedder {
    fn default() -> Self {
        Self {
            producer: EvidenceProducer {
                system: "cg-edge".to_string(),
                model_name: "lexical-mock".to_string(),
                model_version: "0".to_string(),
                pipeline_version: default_text_embedding_space().embedding_space_id,
            },
        }
    }
}

impl Embedder for LexicalMockEmbedder {
    fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
        let value = match input {
            EmbeddingInput::Text(text) => text,
            EmbeddingInput::ImageFrame { .. } | EmbeddingInput::ImageBytes(..) => {
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
        let space = default_text_embedding_space();
        let mut vector = lexical_vector(&value, space.dimension);
        l2_normalize(&mut vector);
        Ok(EmbeddingOutput {
            embedding_space_id: space.embedding_space_id,
            dimension: space.dimension,
            producer: self.producer.clone(),
            vector,
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        default_text_embedding_space()
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

fn lexical_vector(value: &str, dimension: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];
    let normalized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>();
    let tokens = normalized
        .split_whitespace()
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    for token in &tokens {
        add_feature(&mut vector, "word", token, 1.0);
        for gram in char_ngrams(token, 3) {
            add_feature(&mut vector, "gram", &gram, 0.5);
        }
    }
    for pair in tokens.windows(2) {
        add_feature(
            &mut vector,
            "pair",
            &format!("{} {}", pair[0], pair[1]),
            0.75,
        );
    }
    if vector.iter().all(|value| *value == 0.0) && !vector.is_empty() {
        vector[0] = 1.0;
    }
    vector
}

fn char_ngrams(token: &str, width: usize) -> Vec<String> {
    let chars = token.chars().collect::<Vec<_>>();
    if chars.len() <= width {
        return vec![token.to_string()];
    }
    chars
        .windows(width)
        .map(|window| window.iter().collect::<String>())
        .collect()
}

fn add_feature(vector: &mut [f32], namespace: &str, feature: &str, weight: f32) {
    if vector.is_empty() {
        return;
    }
    let mut hasher = Sha256::new();
    hasher.update(namespace.as_bytes());
    hasher.update([0]);
    hasher.update(feature.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    let index = u64::from_le_bytes(bytes) as usize % vector.len();
    vector[index] += weight;
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
