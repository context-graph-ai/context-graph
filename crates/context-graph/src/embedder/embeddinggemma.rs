use std::sync::Arc;

use candle_core::Tensor;

use crate::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use crate::{
    DistanceMetric, EmbedderConfig, EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind,
    EvidenceProducer, Result,
};

#[derive(Debug, Clone)]
pub struct LazyHandle {
    loaded: bool,
}

impl LazyHandle {
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingGemmaForwardStage {
    TokenEmbedding,
    AttentionQkv,
    AttentionOutput,
    Mlp,
    FinalNorm,
    MeanPool,
    ProjectionMlp,
    NormalizedOutput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingGemmaLoadEvent {
    ArtifactOpened {
        path_suffix: String,
    },
    TokenizerLoaded {
        path_suffix: String,
        vocab_size: usize,
    },
    TensorLoaded {
        name: String,
        shape: Vec<usize>,
        dtype: String,
        device: String,
    },
    TokenIds {
        ids: Vec<u32>,
    },
    ForwardStage {
        stage: EmbeddingGemmaForwardStage,
        shape: Vec<usize>,
        dtype: String,
        device: String,
    },
    CandleTensorDigest {
        stage: EmbeddingGemmaForwardStage,
        shape: Vec<usize>,
        dtype: String,
        device: String,
        value_sha256: String,
    },
    ArtifactFingerprinted {
        path_suffix: String,
        sha256: String,
    },
}

pub trait EmbeddingGemmaLoadObserver: Send + Sync {
    fn observe(&self, event: EmbeddingGemmaLoadEvent);

    fn observe_candle_tensor(&self, _stage: EmbeddingGemmaForwardStage, _tensor: &Tensor) {}
}

#[derive(Clone)]
pub struct EmbeddingGemmaTextEmbedder {
    pub config: EmbedderConfig,
    pub dimension: usize,
    load_observer: Option<Arc<dyn EmbeddingGemmaLoadObserver>>,
}

impl std::fmt::Debug for EmbeddingGemmaTextEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGemmaTextEmbedder")
            .field("config", &self.config)
            .field("dimension", &self.dimension)
            .field("has_load_observer", &self.load_observer.is_some())
            .finish()
    }
}

impl EmbeddingGemmaTextEmbedder {
    pub fn new(config: EmbedderConfig) -> Self {
        Self::new_for_dimension(config, 768)
    }

    pub fn new_for_dimension(config: EmbedderConfig, dimension: usize) -> Self {
        Self {
            config,
            dimension,
            load_observer: None,
        }
    }

    pub fn new_with_load_observer(
        config: EmbedderConfig,
        load_observer: Arc<dyn EmbeddingGemmaLoadObserver>,
    ) -> Self {
        Self {
            config,
            dimension: 768,
            load_observer: Some(load_observer),
        }
    }

    pub fn lazy_handle(&self) -> LazyHandle {
        LazyHandle { loaded: false }
    }

    pub fn loader_entry_count() -> usize {
        0
    }

    pub fn reset_loader_entry_count() {}

    pub fn runtime_backend(&self) -> &'static str {
        "stub"
    }

    pub fn model_artifact_fingerprint(&self) -> Option<String> {
        None
    }

    pub fn loader_events(&self) -> Vec<String> {
        Vec::new()
    }

    pub fn structured_loader_events(&self) -> Vec<EmbeddingGemmaLoadEvent> {
        Vec::new()
    }

    pub fn network_attempt_count() -> usize {
        0
    }

    pub fn emit_cold_start_progress(&self) -> String {
        "loading".to_string()
    }

    fn load_model(&self) {}
}

impl Embedder for EmbeddingGemmaTextEmbedder {
    fn embed(&self, _input: EmbeddingInput) -> Result<EmbeddingOutput> {
        self.load_model();
        Ok(EmbeddingOutput {
            embedding_space_id: self.embedding_space().embedding_space_id,
            dimension: self.dimension,
            producer: EvidenceProducer {
                system: "cg-edge".to_string(),
                model_name: "embeddinggemma-300m".to_string(),
                model_version: "0".to_string(),
                pipeline_version: "embeddinggemma".to_string(),
            },
            vector: vec![0.0; self.dimension],
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        let id = format!("text_default_embeddinggemma_300m_{}", self.dimension);
        EmbeddingSpace {
            embedding_space_id: id.clone(),
            dimension: self.dimension,
            metric: DistanceMetric::Cosine,
            model_family: "text".to_string(),
            model_name: "embeddinggemma-300m".to_string(),
            model_version: "0".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: Some(EvidenceKind::Text),
            bindings: vec![EmbeddingSpaceBinding {
                embedding_space_id: id,
                table_name: "evidence".to_string(),
                vector_column: "vector_text".to_string(),
            }],
        }
    }
}
