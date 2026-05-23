pub mod embeddinggemma;
pub mod load_observer;
pub mod mock;
pub mod nomic;

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{EmbeddingSpace, EmbeddingSpaceDecl, EvidenceKind, EvidenceProducer, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingInput {
    Text(String),
    ImageFrame { uri: String },
    AudioSegment { uri: String },
    VideoSegment { uri: String },
    StructuredSignal(serde_json::Value),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingOutput {
    pub embedding_space_id: String,
    pub dimension: usize,
    pub producer: EvidenceProducer,
    pub vector: Vec<f32>,
}

pub trait Embedder: Send + Sync {
    fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput>;
    fn supported_inputs(&self) -> Vec<EvidenceKind>;
    fn embedding_space(&self) -> EmbeddingSpace;
}

#[derive(Clone)]
pub struct EmbedderRegistration {
    pub space: EmbeddingSpaceDecl,
    pub embedder: Arc<dyn Embedder>,
}

impl std::fmt::Debug for EmbedderRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedderRegistration")
            .field("space", &self.space)
            .field(
                "embedder_declared_space",
                &self.embedder.embedding_space().embedding_space_id,
            )
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSource {
    HuggingFace { repo: String },
    LocalPath(PathBuf),
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderConfig {
    pub source: ModelSource,
    pub offline: bool,
    pub cache_dir: Option<PathBuf>,
    pub pinned_revision: Option<String>,
    pub network_policy: String,
}

impl EmbedderConfig {
    pub fn default_embeddinggemma_offline() -> Self {
        Self {
            source: ModelSource::HuggingFace {
                repo: "google/embeddinggemma-300m".to_string(),
            },
            offline: true,
            cache_dir: None,
            pinned_revision: None,
            network_policy: "offline-cache-only".to_string(),
        }
    }

    pub fn disabled() -> Self {
        Self {
            source: ModelSource::Disabled,
            offline: true,
            cache_dir: None,
            pinned_revision: None,
            network_policy: "disabled".to_string(),
        }
    }
}
