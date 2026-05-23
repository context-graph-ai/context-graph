use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use candle_core::{D, DType, Device, Tensor, shape::ShapeWithOneHole};
use candle_nn::{Activation, Module};
use safetensors::SafeTensors;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use context_graph::embedder::{Embedder, EmbeddingInput, EmbeddingOutput};
use context_graph::{
    CgError, DistanceMetric, EmbedderConfig, EmbedderLoadEvent, EmbedderLoadObserver,
    EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind, EvidenceProducer, Result,
};

static NETWORK_ATTEMPT_COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static LOADER_ENTRY_COUNT: Cell<usize> = const { Cell::new(0) };
}

const TOKEN_EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";
const FINAL_NORM_TENSOR: &str = "model.norm.weight";
const TEXT_PROJECTION_UP_TENSOR: &str = "model.text_projection.up_proj.weight";
const TEXT_PROJECTION_DOWN_TENSOR: &str = "model.text_projection.down_proj.weight";
const DEFAULT_RMS_NORM_EPS: f64 = 1.0e-6;
const EMBEDDINGGEMMA_MODEL_FAMILY: &str = "text";
const EMBEDDINGGEMMA_MODEL_NAME: &str = "embeddinggemma-300m";
const EMBEDDINGGEMMA_MODEL_VERSION: &str = "default";
const EMBEDDINGGEMMA_DEFAULT_DIMENSION: usize = 768;

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
    ModelLookupStarted {
        model_id: String,
        cache_path: String,
        pinned_revision: Option<String>,
        source: String,
        network_policy: String,
    },
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
    ModelLoaded {
        model_id: String,
        cache_path: String,
        pinned_revision: Option<String>,
        artifact_sha256: String,
    },
}

#[derive(Clone)]
pub struct EmbeddingGemmaTextEmbedder {
    pub config: EmbedderConfig,
    pub dimension: usize,
    load_observer: Option<Arc<dyn EmbedderLoadObserver>>,
    loaded_model: Arc<OnceLock<LoadedEmbeddingGemmaModel>>,
    model_loaded_event_emitted: Arc<AtomicBool>,
    load_lock: Arc<Mutex<()>>,
    loader_events: Arc<Mutex<Vec<EmbeddingGemmaLoadEvent>>>,
}

impl std::fmt::Debug for EmbeddingGemmaTextEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGemmaTextEmbedder")
            .field("config", &self.config)
            .field("dimension", &self.dimension)
            .field("has_load_observer", &self.load_observer.is_some())
            .field("is_loaded", &self.loaded_model.get().is_some())
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
            loaded_model: Arc::new(OnceLock::new()),
            model_loaded_event_emitted: Arc::new(AtomicBool::new(false)),
            load_lock: Arc::new(Mutex::new(())),
            loader_events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn new_with_load_observer(
        config: EmbedderConfig,
        load_observer: Arc<dyn EmbedderLoadObserver>,
    ) -> Self {
        Self {
            config,
            dimension: 768,
            load_observer: Some(load_observer),
            loaded_model: Arc::new(OnceLock::new()),
            model_loaded_event_emitted: Arc::new(AtomicBool::new(false)),
            load_lock: Arc::new(Mutex::new(())),
            loader_events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn lazy_handle(&self) -> LazyHandle {
        LazyHandle {
            loaded: self.loaded_model.get().is_some(),
        }
    }

    pub fn loader_entry_count() -> usize {
        LOADER_ENTRY_COUNT.with(Cell::get)
    }

    pub fn reset_loader_entry_count() {
        LOADER_ENTRY_COUNT.with(|count| count.set(0));
        NETWORK_ATTEMPT_COUNT.store(0, Ordering::SeqCst);
    }

    pub fn runtime_backend(&self) -> &'static str {
        "candle"
    }

    pub fn model_artifact_fingerprint(&self) -> Option<String> {
        self.loaded_model
            .get()
            .map(|model| model.model_sha256.clone())
    }

    pub fn loader_events(&self) -> Vec<String> {
        self.structured_loader_events()
            .into_iter()
            .map(|event| format!("{event:?}"))
            .collect()
    }

    pub fn structured_loader_events(&self) -> Vec<EmbeddingGemmaLoadEvent> {
        self.loader_events
            .lock()
            .map(|events| events.clone())
            .unwrap_or_default()
    }

    pub fn network_attempt_count() -> usize {
        NETWORK_ATTEMPT_COUNT.load(Ordering::SeqCst)
    }

    pub fn emit_cold_start_progress(&self) -> String {
        if matches!(self.config.source, context_graph::ModelSource::Disabled) {
            return "embedding_loader=disabled".to_string();
        }
        format!(
            "embedding_loader=lookup model=embeddinggemma-300m path={} artifacts=model.safetensors,tokenizer.json revision={} network_policy={} no preload at Store::open",
            model_dir(&self.config).display(),
            self.config.pinned_revision.as_deref().unwrap_or("unpinned"),
            self.config.network_policy
        )
    }

    fn loaded_model(&self) -> Result<&LoadedEmbeddingGemmaModel> {
        if matches!(self.config.source, context_graph::ModelSource::Disabled) {
            return Err(CgError::TextEmbedderDisabled);
        }
        if self.loaded_model.get().is_none() {
            let _guard = self
                .load_lock
                .lock()
                .map_err(|_| CgError::Engine("embeddinggemma load lock poisoned".to_string()))?;
            if self.loaded_model.get().is_none() {
                let model = self.load_model_artifacts()?;
                let _ = self.loaded_model.set(model);
            }
        }
        self.loaded_model
            .get()
            .ok_or_else(|| CgError::Engine("embeddinggemma load did not publish model".to_string()))
    }

    fn load_model_artifacts(&self) -> Result<LoadedEmbeddingGemmaModel> {
        if matches!(self.config.source, context_graph::ModelSource::Disabled) {
            return Err(CgError::TextEmbedderDisabled);
        }
        let started = Instant::now();
        LOADER_ENTRY_COUNT.with(|count| count.set(count.get() + 1));
        let model_dir = resolve_model_dir(&self.config)?;
        let model_id = EMBEDDINGGEMMA_MODEL_NAME.to_string();
        let cache_path = model_dir.display().to_string();
        let pinned_revision = self.config.pinned_revision.clone();
        self.observe_load(EmbedderLoadEvent::LoadStarted {
            model_id: model_id.clone(),
            byte_size: None,
            destination: model_dir.clone(),
        });
        self.observe(EmbeddingGemmaLoadEvent::ModelLookupStarted {
            model_id: model_id.clone(),
            cache_path: cache_path.clone(),
            pinned_revision: pinned_revision.clone(),
            source: source_name(&self.config),
            network_policy: self.config.network_policy.clone(),
        });
        let model_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config.json");
        let revision_path = model_dir.join("REVISION");
        if !model_path.exists() || !tokenizer_path.exists() || !config_path.exists() {
            if model_dir.exists()
                && matches!(
                    self.config.source,
                    context_graph::ModelSource::HuggingFace { .. }
                )
            {
                return Err(CgError::EmbedderModelCorrupt {
                    path: model_dir.display().to_string(),
                });
            }
            return Err(CgError::EmbedderModelNotAvailable {
                source: source_name(&self.config),
                path: model_dir.display().to_string(),
            });
        }
        let revision = std::fs::read_to_string(&revision_path).unwrap_or_default();
        if let Some(expected) = self.config.pinned_revision.as_deref()
            && revision.trim() != expected
        {
            return Err(CgError::EmbedderModelRevisionMismatch {
                expected: expected.to_string(),
                actual: revision.trim().to_string(),
            });
        }
        let model_bytes =
            std::fs::read(&model_path).map_err(|_| CgError::EmbedderModelCorrupt {
                path: model_path.display().to_string(),
            })?;
        let model_sha256 = sha256_hex(&model_bytes);
        let tensors =
            SafeTensors::deserialize(&model_bytes).map_err(|_| CgError::EmbedderModelCorrupt {
                path: model_path.display().to_string(),
            })?;
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|_| CgError::EmbedderModelCorrupt {
                path: tokenizer_path.display().to_string(),
            })?;
        let config_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).map_err(|_| {
                CgError::EmbedderModelCorrupt {
                    path: config_path.display().to_string(),
                }
            })?)
            .map_err(|_| CgError::EmbedderModelCorrupt {
                path: config_path.display().to_string(),
            })?;
        let required = required_graph_tensors(&config_json);
        for name in &required {
            if tensors.tensor(name).is_err() {
                return Err(CgError::EmbedderModelCorrupt {
                    path: model_path.display().to_string(),
                });
            }
        }
        if !config_json
            .get("architectures")
            .and_then(|value| value.as_array())
            .is_some_and(|items| items.iter().any(|item| item == "EmbeddingGemmaTextModel"))
        {
            return Err(CgError::EmbedderModelCorrupt {
                path: config_path.display().to_string(),
            });
        }
        self.observe(EmbeddingGemmaLoadEvent::ArtifactOpened {
            path_suffix: "model.safetensors".to_string(),
        });
        self.observe(EmbeddingGemmaLoadEvent::TokenizerLoaded {
            path_suffix: "tokenizer.json".to_string(),
            vocab_size: tokenizer.get_vocab_size(false),
        });
        self.observe(EmbeddingGemmaLoadEvent::ArtifactFingerprinted {
            path_suffix: "model.safetensors".to_string(),
            sha256: model_sha256.clone(),
        });
        let weights = EmbeddingGemmaWeights::from_safetensors(&tensors, &config_json, &model_path)?;
        for (name, shape) in weights.tensor_shapes() {
            self.observe(EmbeddingGemmaLoadEvent::TensorLoaded {
                name: name.to_string(),
                shape,
                dtype: "F32".to_string(),
                device: "Cpu".to_string(),
            });
        }
        self.observe_load(EmbedderLoadEvent::LoadCompleted {
            model_id: model_id.clone(),
            byte_size: None,
            destination: model_dir,
            elapsed: started.elapsed(),
        });
        Ok(LoadedEmbeddingGemmaModel {
            weights,
            tokenizer,
            tokenizer_path: tokenizer_path.display().to_string(),
            model_id,
            cache_path,
            pinned_revision,
            model_sha256,
        })
    }

    fn observe(&self, event: EmbeddingGemmaLoadEvent) {
        if let Ok(mut events) = self.loader_events.lock() {
            events.push(event);
        }
    }

    fn observe_load(&self, event: EmbedderLoadEvent) {
        if let Some(observer) = &self.load_observer {
            observer.observe(event);
        }
    }

    fn observe_tensor(&self, stage: EmbeddingGemmaForwardStage, tensor: &Tensor) {
        self.observe(EmbeddingGemmaLoadEvent::ForwardStage {
            stage: stage.clone(),
            shape: tensor.dims().to_vec(),
            dtype: format!("{:?}", tensor.dtype()),
            device: format!("{:?}", tensor.device()),
        });
        if let Ok(value_sha256) = tensor_digest(tensor) {
            self.observe(EmbeddingGemmaLoadEvent::CandleTensorDigest {
                stage,
                shape: tensor.dims().to_vec(),
                dtype: format!("{:?}", tensor.dtype()),
                device: format!("{:?}", tensor.device()),
                value_sha256,
            });
        }
    }

    fn forward(&self, weights: &EmbeddingGemmaWeights, token_ids: &[u32]) -> Result<Vec<f32>> {
        let ids = if token_ids.is_empty() {
            vec![0]
        } else {
            token_ids.iter().map(|id| *id as usize).collect::<Vec<_>>()
        };
        let parts = weights.forward_sequence(&ids)?;
        self.observe_tensor(
            EmbeddingGemmaForwardStage::TokenEmbedding,
            &parts.token_embedding,
        );
        self.observe_tensor(
            EmbeddingGemmaForwardStage::AttentionQkv,
            &parts.attention_scores,
        );
        self.observe_tensor(
            EmbeddingGemmaForwardStage::AttentionOutput,
            &parts.attention_output,
        );
        self.observe_tensor(EmbeddingGemmaForwardStage::Mlp, &parts.mlp_output);
        self.observe_tensor(EmbeddingGemmaForwardStage::FinalNorm, &parts.final_sequence);
        let pooled = parts.pooled;
        self.observe_tensor(EmbeddingGemmaForwardStage::MeanPool, &pooled);
        let projection_hidden = pooled
            .matmul(&weights.text_projection_up_tensor()?)
            .map_err(|err| weights.candle_error(err))?
            .silu()
            .map_err(|err| weights.candle_error(err))?;
        let projection = projection_hidden
            .matmul(&weights.text_projection_down_tensor()?)
            .map_err(|err| weights.candle_error(err))?;
        self.observe_tensor(EmbeddingGemmaForwardStage::ProjectionMlp, &projection);
        let projection_values = projection
            .to_vec2::<f32>()
            .map_err(|err| weights.candle_error(err))?[0]
            .clone();
        let normalized_values = matryoshka_head(&projection_values, 768);
        let normalized = Tensor::from_slice(&normalized_values, (1, 768), &Device::Cpu)
            .map_err(|err| weights.candle_error(err))?;
        self.observe_tensor(EmbeddingGemmaForwardStage::NormalizedOutput, &normalized);
        Ok(normalized_values)
    }
}

impl Embedder for EmbeddingGemmaTextEmbedder {
    fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
        let text = match input {
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
        let loaded = self.loaded_model()?;
        let token_ids = loaded
            .tokenizer
            .encode(text, false)
            .map_err(|_| CgError::EmbedderModelCorrupt {
                path: loaded.tokenizer_path.clone(),
            })?
            .get_ids()
            .to_vec();
        self.observe(EmbeddingGemmaLoadEvent::TokenIds {
            ids: token_ids.clone(),
        });
        let full = self.forward(&loaded.weights, &token_ids)?;
        let vector = matryoshka_head(&full, self.dimension);
        if self
            .model_loaded_event_emitted
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            self.observe(EmbeddingGemmaLoadEvent::ModelLoaded {
                model_id: loaded.model_id.clone(),
                cache_path: loaded.cache_path.clone(),
                pinned_revision: loaded.pinned_revision.clone(),
                artifact_sha256: loaded.model_sha256.clone(),
            });
        }
        Ok(EmbeddingOutput {
            embedding_space_id: self.embedding_space().embedding_space_id,
            dimension: self.dimension,
            producer: EvidenceProducer {
                system: context_graph::CG_EDGE_SYSTEM_TAG.to_string(),
                model_name: EMBEDDINGGEMMA_MODEL_NAME.to_string(),
                model_version: EMBEDDINGGEMMA_MODEL_VERSION.to_string(),
                pipeline_version: "embeddinggemma".to_string(),
            },
            vector,
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        embeddinggemma_space(
            self.dimension,
            self.dimension == EMBEDDINGGEMMA_DEFAULT_DIMENSION,
        )
    }
}

fn embeddinggemma_space(dimension: usize, default_for_text: bool) -> EmbeddingSpace {
    let id = if dimension == EMBEDDINGGEMMA_DEFAULT_DIMENSION {
        "text_default_embeddinggemma_300m_768".to_string()
    } else {
        format!("text_default_embeddinggemma_300m_{dimension}")
    };
    let mut bindings = vec![EmbeddingSpaceBinding {
        embedding_space_id: id.clone(),
        table_name: "evidence".to_string(),
        vector_column: "vector_text".to_string(),
    }];
    if dimension == EMBEDDINGGEMMA_DEFAULT_DIMENSION {
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
        model_family: EMBEDDINGGEMMA_MODEL_FAMILY.to_string(),
        model_name: EMBEDDINGGEMMA_MODEL_NAME.to_string(),
        model_version: EMBEDDINGGEMMA_MODEL_VERSION.to_string(),
        supported_evidence_kinds: vec![EvidenceKind::Text, EvidenceKind::StructuredSignal],
        default_for_evidence_kind: default_for_text.then_some(EvidenceKind::Text),
        bindings,
    }
}

fn required_graph_tensors(config: &serde_json::Value) -> Vec<String> {
    let layer_count = config
        .get("num_hidden_layers")
        .and_then(|value| value.as_u64())
        .unwrap_or(1)
        .max(1) as usize;
    let mut tensors = Vec::with_capacity(4 + layer_count * 13);
    tensors.push(TOKEN_EMBEDDING_TENSOR.to_string());
    for layer in 0..layer_count {
        tensors.push(layer_tensor(layer, "self_attn.q_proj.weight"));
        tensors.push(layer_tensor(layer, "self_attn.k_proj.weight"));
        tensors.push(layer_tensor(layer, "self_attn.v_proj.weight"));
        tensors.push(layer_tensor(layer, "self_attn.o_proj.weight"));
        tensors.push(layer_tensor(layer, "self_attn.q_norm.weight"));
        tensors.push(layer_tensor(layer, "self_attn.k_norm.weight"));
        tensors.push(layer_tensor(layer, "input_layernorm.weight"));
        tensors.push(layer_tensor(layer, "pre_feedforward_layernorm.weight"));
        tensors.push(layer_tensor(layer, "post_feedforward_layernorm.weight"));
        tensors.push(layer_tensor(layer, "post_attention_layernorm.weight"));
        tensors.push(layer_tensor(layer, "mlp.gate_proj.weight"));
        tensors.push(layer_tensor(layer, "mlp.up_proj.weight"));
        tensors.push(layer_tensor(layer, "mlp.down_proj.weight"));
    }
    tensors.push(FINAL_NORM_TENSOR.to_string());
    tensors.push(TEXT_PROJECTION_UP_TENSOR.to_string());
    tensors.push(TEXT_PROJECTION_DOWN_TENSOR.to_string());
    tensors
}

fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("model.layers.{layer}.{suffix}")
}

fn model_dir(config: &EmbedderConfig) -> PathBuf {
    match &config.source {
        context_graph::ModelSource::LocalPath(path) => path.clone(),
        context_graph::ModelSource::HuggingFace { .. } => config
            .cache_dir
            .clone()
            .unwrap_or_else(std::env::temp_dir)
            .join("embeddinggemma-300m"),
        context_graph::ModelSource::Disabled => PathBuf::new(),
    }
}

fn resolve_model_dir(config: &EmbedderConfig) -> Result<PathBuf> {
    let dir = model_dir(config);
    Ok(dir)
}

fn source_name(config: &EmbedderConfig) -> String {
    match &config.source {
        context_graph::ModelSource::LocalPath(path) => format!("LocalPath({})", path.display()),
        context_graph::ModelSource::HuggingFace { repo } => format!("HuggingFace({repo})"),
        context_graph::ModelSource::Disabled => "Disabled".to_string(),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn tensor_digest(tensor: &Tensor) -> Result<String> {
    let values = tensor
        .flatten_all()
        .map_err(|err| CgError::Engine(format!("embeddinggemma tensor digest flatten: {err}")))?
        .to_vec1::<f32>()
        .map_err(|err| CgError::Engine(format!("embeddinggemma tensor digest values: {err}")))?;
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

struct LoadedEmbeddingGemmaModel {
    weights: EmbeddingGemmaWeights,
    tokenizer: Tokenizer,
    tokenizer_path: String,
    model_id: String,
    cache_path: String,
    pinned_revision: Option<String>,
    model_sha256: String,
}

struct EmbeddingGemmaWeights {
    artifact_path: String,
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    rope_local_base_freq: f64,
    rms_norm_eps: f64,
    attention_activation: Activation,
    projection_intermediate_size: usize,
    token_embedding: Vec<f32>,
    layers: Vec<EmbeddingGemmaLayerWeights>,
    final_norm: Vec<f32>,
    text_projection_up: Vec<f32>,
    text_projection_down: Vec<f32>,
}

struct EmbeddingGemmaLayerWeights {
    index: usize,
    intermediate_size: usize,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    input_layernorm: Vec<f32>,
    pre_feedforward_layernorm: Vec<f32>,
    post_feedforward_layernorm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

impl EmbeddingGemmaWeights {
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        config: &serde_json::Value,
        model_path: &Path,
    ) -> Result<Self> {
        let artifact_path = model_path.display().to_string();
        let (token_embedding, token_embedding_shape) =
            tensor_f32(tensors, TOKEN_EMBEDDING_TENSOR, model_path)?;
        let (final_norm, final_norm_shape) = tensor_f32(tensors, FINAL_NORM_TENSOR, model_path)?;
        let (text_projection_up, text_projection_up_shape) =
            tensor_f32(tensors, TEXT_PROJECTION_UP_TENSOR, model_path)?;
        let (text_projection_down, text_projection_down_shape) =
            tensor_f32(tensors, TEXT_PROJECTION_DOWN_TENSOR, model_path)?;

        let hidden_size = match final_norm_shape.as_slice() {
            [hidden] if *hidden > 0 => *hidden,
            _ => return Err(model_corrupt(model_path)),
        };
        require_shape_2(&token_embedding_shape, None, Some(hidden_size), model_path)?;
        let num_attention_heads = config_usize(config, "num_attention_heads")
            .unwrap_or(1)
            .max(1);
        let num_key_value_heads = config_usize(config, "num_key_value_heads")
            .unwrap_or(num_attention_heads)
            .max(1);
        let head_dim = config_usize(config, "head_dim")
            .unwrap_or_else(|| hidden_size / num_attention_heads)
            .max(1);
        if num_attention_heads * head_dim != hidden_size {
            return Err(model_corrupt(model_path));
        }
        if !head_dim.is_multiple_of(2) {
            return Err(model_corrupt(model_path));
        }
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            return Err(model_corrupt(model_path));
        }
        let rope_theta = config_f64(config, "rope_theta").unwrap_or(1_000_000.0);
        let rope_local_base_freq = config_f64(config, "rope_local_base_freq").unwrap_or(10_000.0);
        let rms_norm_eps = config_f64(config, "rms_norm_eps").unwrap_or(DEFAULT_RMS_NORM_EPS);
        let attention_activation = config
            .get("hidden_activation")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|_| model_corrupt(model_path))?
            .unwrap_or(Activation::Silu);
        let layer_count = config
            .get("num_hidden_layers")
            .and_then(|value| value.as_u64())
            .unwrap_or(1)
            .max(1) as usize;
        let mut layers = Vec::with_capacity(layer_count);
        for index in 0..layer_count {
            layers.push(EmbeddingGemmaLayerWeights::from_safetensors(
                tensors,
                index,
                hidden_size,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                model_path,
            )?);
        }
        require_shape_2(
            &text_projection_up_shape,
            Some(hidden_size),
            None,
            model_path,
        )?;
        let projection_intermediate_size = text_projection_up_shape[1];
        require_shape(
            &text_projection_down_shape,
            &[projection_intermediate_size, 768],
            model_path,
        )?;

        Ok(Self {
            artifact_path,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rope_theta,
            rope_local_base_freq,
            rms_norm_eps,
            attention_activation,
            projection_intermediate_size,
            token_embedding,
            layers,
            final_norm,
            text_projection_up,
            text_projection_down,
        })
    }

    fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let mut shapes = vec![(
            TOKEN_EMBEDDING_TENSOR.to_string(),
            vec![
                self.token_embedding.len() / self.hidden_size,
                self.hidden_size,
            ],
        )];
        for layer in &self.layers {
            shapes.extend(layer.tensor_shapes(
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            ));
        }
        shapes.extend([
            (FINAL_NORM_TENSOR.to_string(), vec![self.hidden_size]),
            (
                TEXT_PROJECTION_UP_TENSOR.to_string(),
                vec![self.hidden_size, self.projection_intermediate_size],
            ),
            (
                TEXT_PROJECTION_DOWN_TENSOR.to_string(),
                vec![self.projection_intermediate_size, 768],
            ),
        ]);
        shapes
    }

    fn candle_error(&self, err: candle_core::Error) -> CgError {
        CgError::Engine(format!("candle tensor: {err}"))
    }

    fn corrupt(&self) -> CgError {
        CgError::EmbedderModelCorrupt {
            path: self.artifact_path.clone(),
        }
    }

    fn tensor_from_slice<S: ShapeWithOneHole>(&self, values: &[f32], shape: S) -> Result<Tensor> {
        Tensor::from_slice(values, shape, &Device::Cpu).map_err(|err| self.candle_error(err))
    }

    fn token_sequence_tensor(&self, token_ids: &[usize]) -> Result<Tensor> {
        let mut values = Vec::with_capacity(token_ids.len() * self.hidden_size);
        for token_id in token_ids {
            let start = token_id
                .checked_mul(self.hidden_size)
                .ok_or_else(|| self.corrupt())?;
            let end = start
                .checked_add(self.hidden_size)
                .ok_or_else(|| self.corrupt())?;
            values.extend_from_slice(
                self.token_embedding
                    .get(start..end)
                    .ok_or_else(|| self.corrupt())?,
            );
        }
        self.tensor_from_slice(&values, (1, token_ids.len(), self.hidden_size))
    }

    fn matrix_tensor(&self, values: &[f32], rows: usize, cols: usize) -> Result<Tensor> {
        self.tensor_from_slice(values, (rows, cols))
    }

    fn final_norm_tensor(&self) -> Result<Tensor> {
        Tensor::from_slice(&self.final_norm, self.hidden_size, &Device::Cpu)
            .map_err(|err| self.candle_error(err))
    }

    fn text_projection_up_tensor(&self) -> Result<Tensor> {
        self.matrix_tensor(
            &self.text_projection_up,
            self.hidden_size,
            self.projection_intermediate_size,
        )
    }

    fn text_projection_down_tensor(&self) -> Result<Tensor> {
        self.matrix_tensor(
            &self.text_projection_down,
            self.projection_intermediate_size,
            768,
        )
    }

    fn forward_sequence(&self, token_ids: &[usize]) -> Result<EmbeddingGemmaForwardParts> {
        let token_embedding = self.token_sequence_tensor(token_ids)?;
        let mut state = token_embedding.clone();
        let mut first_attention = None;
        let mut first_attention_out = None;
        let mut first_mlp = None;
        for layer in &self.layers {
            let parts = layer.forward(self, &state)?;
            if first_attention.is_none() {
                first_attention = Some(parts.attention_scores.clone());
                first_attention_out = Some(parts.attention_output.clone());
                first_mlp = Some(parts.mlp_output.clone());
            }
            state = parts.layer_output;
        }
        let final_sequence = rms_norm(&state, &self.final_norm_tensor()?, self.rms_norm_eps)
            .map_err(|err| self.candle_error(err))?;
        let pooled = final_sequence
            .mean_keepdim(1)
            .map_err(|err| self.candle_error(err))?
            .reshape((1, self.hidden_size))
            .map_err(|err| self.candle_error(err))?;
        Ok(EmbeddingGemmaForwardParts {
            token_embedding,
            attention_scores: first_attention.ok_or_else(|| self.corrupt())?,
            attention_output: first_attention_out.ok_or_else(|| self.corrupt())?,
            mlp_output: first_mlp.ok_or_else(|| self.corrupt())?,
            final_sequence,
            pooled,
        })
    }
}

struct EmbeddingGemmaForwardParts {
    token_embedding: Tensor,
    attention_scores: Tensor,
    attention_output: Tensor,
    mlp_output: Tensor,
    final_sequence: Tensor,
    pooled: Tensor,
}

impl EmbeddingGemmaLayerWeights {
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        index: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        model_path: &Path,
    ) -> Result<Self> {
        let (q_proj, q_proj_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "self_attn.q_proj.weight"),
            model_path,
        )?;
        let (k_proj, k_proj_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "self_attn.k_proj.weight"),
            model_path,
        )?;
        let (v_proj, v_proj_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "self_attn.v_proj.weight"),
            model_path,
        )?;
        let (o_proj, o_proj_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "self_attn.o_proj.weight"),
            model_path,
        )?;
        let (q_norm, q_norm_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "self_attn.q_norm.weight"),
            model_path,
        )?;
        let (k_norm, k_norm_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "self_attn.k_norm.weight"),
            model_path,
        )?;
        let (input_layernorm, input_layernorm_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "input_layernorm.weight"),
            model_path,
        )?;
        let (pre_feedforward_layernorm, pre_feedforward_layernorm_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "pre_feedforward_layernorm.weight"),
            model_path,
        )?;
        let (post_feedforward_layernorm, post_feedforward_layernorm_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "post_feedforward_layernorm.weight"),
            model_path,
        )?;
        let (post_attention_layernorm, post_attention_layernorm_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "post_attention_layernorm.weight"),
            model_path,
        )?;
        let (gate_proj, gate_proj_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "mlp.gate_proj.weight"),
            model_path,
        )?;
        let (up_proj, up_proj_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "mlp.up_proj.weight"),
            model_path,
        )?;
        let (down_proj, down_proj_shape) = tensor_f32(
            tensors,
            &layer_tensor(index, "mlp.down_proj.weight"),
            model_path,
        )?;
        require_shape(
            &q_proj_shape,
            &[hidden_size, num_attention_heads * head_dim],
            model_path,
        )?;
        require_shape(
            &k_proj_shape,
            &[hidden_size, num_key_value_heads * head_dim],
            model_path,
        )?;
        require_shape(
            &v_proj_shape,
            &[hidden_size, num_key_value_heads * head_dim],
            model_path,
        )?;
        require_shape(
            &o_proj_shape,
            &[num_attention_heads * head_dim, hidden_size],
            model_path,
        )?;
        require_shape(&q_norm_shape, &[head_dim], model_path)?;
        require_shape(&k_norm_shape, &[head_dim], model_path)?;
        require_shape(&input_layernorm_shape, &[hidden_size], model_path)?;
        require_shape(&pre_feedforward_layernorm_shape, &[hidden_size], model_path)?;
        require_shape(
            &post_feedforward_layernorm_shape,
            &[hidden_size],
            model_path,
        )?;
        require_shape(&post_attention_layernorm_shape, &[hidden_size], model_path)?;
        require_shape_2(&gate_proj_shape, Some(hidden_size), None, model_path)?;
        let intermediate_size = gate_proj_shape[1];
        require_shape(
            &up_proj_shape,
            &[hidden_size, intermediate_size],
            model_path,
        )?;
        require_shape(
            &down_proj_shape,
            &[intermediate_size, hidden_size],
            model_path,
        )?;
        Ok(Self {
            index,
            intermediate_size,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn tensor_shapes(
        &self,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) -> Vec<(String, Vec<usize>)> {
        vec![
            (
                layer_tensor(self.index, "self_attn.q_proj.weight"),
                vec![hidden_size, num_attention_heads * head_dim],
            ),
            (
                layer_tensor(self.index, "self_attn.k_proj.weight"),
                vec![hidden_size, num_key_value_heads * head_dim],
            ),
            (
                layer_tensor(self.index, "self_attn.v_proj.weight"),
                vec![hidden_size, num_key_value_heads * head_dim],
            ),
            (
                layer_tensor(self.index, "self_attn.o_proj.weight"),
                vec![num_attention_heads * head_dim, hidden_size],
            ),
            (
                layer_tensor(self.index, "self_attn.q_norm.weight"),
                vec![head_dim],
            ),
            (
                layer_tensor(self.index, "self_attn.k_norm.weight"),
                vec![head_dim],
            ),
            (
                layer_tensor(self.index, "input_layernorm.weight"),
                vec![hidden_size],
            ),
            (
                layer_tensor(self.index, "pre_feedforward_layernorm.weight"),
                vec![hidden_size],
            ),
            (
                layer_tensor(self.index, "post_feedforward_layernorm.weight"),
                vec![hidden_size],
            ),
            (
                layer_tensor(self.index, "post_attention_layernorm.weight"),
                vec![hidden_size],
            ),
            (
                layer_tensor(self.index, "mlp.gate_proj.weight"),
                vec![hidden_size, self.intermediate_size],
            ),
            (
                layer_tensor(self.index, "mlp.up_proj.weight"),
                vec![hidden_size, self.intermediate_size],
            ),
            (
                layer_tensor(self.index, "mlp.down_proj.weight"),
                vec![self.intermediate_size, hidden_size],
            ),
        ]
    }

    fn matrix_tensor(
        weights: &EmbeddingGemmaWeights,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Tensor> {
        weights.matrix_tensor(values, rows, cols)
    }

    fn forward(
        &self,
        weights: &EmbeddingGemmaWeights,
        state: &Tensor,
    ) -> Result<LayerForwardParts> {
        let normalized = rms_norm(
            state,
            &weights.tensor_from_slice(&self.input_layernorm, weights.hidden_size)?,
            weights.rms_norm_eps,
        )
        .map_err(|err| weights.candle_error(err))?;
        let (_, seq_len, _) = normalized
            .dims3()
            .map_err(|err| weights.candle_error(err))?;
        let normalized_2d = normalized
            .reshape((seq_len, weights.hidden_size))
            .map_err(|err| weights.candle_error(err))?;
        let q = normalized_2d
            .matmul(&Self::matrix_tensor(
                weights,
                &self.q_proj,
                weights.hidden_size,
                weights.num_attention_heads * weights.head_dim,
            )?)
            .map_err(|err| weights.candle_error(err))?
            .reshape((1, seq_len, weights.num_attention_heads, weights.head_dim))
            .map_err(|err| weights.candle_error(err))?
            .transpose(1, 2)
            .map_err(|err| weights.candle_error(err))?;
        let k = normalized_2d
            .matmul(&Self::matrix_tensor(
                weights,
                &self.k_proj,
                weights.hidden_size,
                weights.num_key_value_heads * weights.head_dim,
            )?)
            .map_err(|err| weights.candle_error(err))?
            .reshape((1, seq_len, weights.num_key_value_heads, weights.head_dim))
            .map_err(|err| weights.candle_error(err))?
            .transpose(1, 2)
            .map_err(|err| weights.candle_error(err))?;
        let v = normalized_2d
            .matmul(&Self::matrix_tensor(
                weights,
                &self.v_proj,
                weights.hidden_size,
                weights.num_key_value_heads * weights.head_dim,
            )?)
            .map_err(|err| weights.candle_error(err))?
            .reshape((1, seq_len, weights.num_key_value_heads, weights.head_dim))
            .map_err(|err| weights.candle_error(err))?
            .transpose(1, 2)
            .map_err(|err| weights.candle_error(err))?;
        let q = rms_norm(
            &q,
            &weights.tensor_from_slice(&self.q_norm, weights.head_dim)?,
            weights.rms_norm_eps,
        )
        .map_err(|err| weights.candle_error(err))?;
        let k = rms_norm(
            &k,
            &weights.tensor_from_slice(&self.k_norm, weights.head_dim)?,
            weights.rms_norm_eps,
        )
        .map_err(|err| weights.candle_error(err))?;
        let (q, k) = apply_rotary_emb_qkv(
            &q,
            &k,
            weights.head_dim,
            if seq_len > 1 {
                weights.rope_local_base_freq
            } else {
                weights.rope_theta
            },
            seq_len,
        )
        .map_err(|err| weights.candle_error(err))?;
        let kv_groups = weights.num_attention_heads / weights.num_key_value_heads;
        let k = candle_transformers::utils::repeat_kv(k, kv_groups)
            .map_err(|err| weights.candle_error(err))?
            .contiguous()
            .map_err(|err| weights.candle_error(err))?;
        let v = candle_transformers::utils::repeat_kv(v, kv_groups)
            .map_err(|err| weights.candle_error(err))?
            .contiguous()
            .map_err(|err| weights.candle_error(err))?;
        let attention_scores = (q
            .matmul(&k.transpose(2, 3).map_err(|err| weights.candle_error(err))?)
            .map_err(|err| weights.candle_error(err))?
            * (1.0 / (weights.head_dim as f64).sqrt()))
        .map_err(|err| weights.candle_error(err))?;
        let attention_weights = candle_nn::ops::softmax_last_dim(&attention_scores)
            .map_err(|err| weights.candle_error(err))?;
        let attention_output = attention_weights
            .matmul(&v)
            .map_err(|err| weights.candle_error(err))?
            .transpose(1, 2)
            .map_err(|err| weights.candle_error(err))?
            .reshape((seq_len, weights.num_attention_heads * weights.head_dim))
            .map_err(|err| weights.candle_error(err))?
            .matmul(&Self::matrix_tensor(
                weights,
                &self.o_proj,
                weights.num_attention_heads * weights.head_dim,
                weights.hidden_size,
            )?)
            .map_err(|err| weights.candle_error(err))?
            .reshape((1, seq_len, weights.hidden_size))
            .map_err(|err| weights.candle_error(err))?;
        let attention_output = rms_norm(
            &attention_output,
            &weights.tensor_from_slice(&self.post_attention_layernorm, weights.hidden_size)?,
            weights.rms_norm_eps,
        )
        .map_err(|err| weights.candle_error(err))?;
        let post_attention =
            (&attention_output + state).map_err(|err| weights.candle_error(err))?;
        let mlp_input = rms_norm(
            &post_attention,
            &weights.tensor_from_slice(&self.pre_feedforward_layernorm, weights.hidden_size)?,
            weights.rms_norm_eps,
        )
        .map_err(|err| weights.candle_error(err))?;
        let mlp_input_2d = mlp_input
            .reshape((seq_len, weights.hidden_size))
            .map_err(|err| weights.candle_error(err))?;
        let gate = mlp_input_2d
            .matmul(&Self::matrix_tensor(
                weights,
                &self.gate_proj,
                weights.hidden_size,
                self.intermediate_size,
            )?)
            .map_err(|err| weights.candle_error(err))?;
        let gate = weights
            .attention_activation
            .forward(&gate)
            .map_err(|err| weights.candle_error(err))?;
        let up = mlp_input_2d
            .matmul(&Self::matrix_tensor(
                weights,
                &self.up_proj,
                weights.hidden_size,
                self.intermediate_size,
            )?)
            .map_err(|err| weights.candle_error(err))?;
        let gated = (&gate * &up).map_err(|err| weights.candle_error(err))?;
        let mlp_output = gated
            .matmul(&Self::matrix_tensor(
                weights,
                &self.down_proj,
                self.intermediate_size,
                weights.hidden_size,
            )?)
            .map_err(|err| weights.candle_error(err))?
            .reshape((1, seq_len, weights.hidden_size))
            .map_err(|err| weights.candle_error(err))?;
        let mlp_output = rms_norm(
            &mlp_output,
            &weights.tensor_from_slice(&self.post_feedforward_layernorm, weights.hidden_size)?,
            weights.rms_norm_eps,
        )
        .map_err(|err| weights.candle_error(err))?;
        let layer_output =
            (&post_attention + &mlp_output).map_err(|err| weights.candle_error(err))?;
        Ok(LayerForwardParts {
            attention_scores,
            attention_output,
            mlp_output,
            layer_output,
        })
    }
}

struct LayerForwardParts {
    attention_scores: Tensor,
    attention_output: Tensor,
    mlp_output: Tensor,
    layer_output: Tensor,
}

fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let x_dtype = xs.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    let hidden_size = xs.dim(D::Minus1)?;
    let xs = xs.to_dtype(internal_dtype)?;
    let norm = (xs.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let normalized = xs.broadcast_div(&(norm + eps)?.sqrt()?)?;
    normalized
        .to_dtype(x_dtype)?
        .broadcast_mul(&(weight + 1.0)?)
}

fn apply_rotary_emb_qkv(
    q: &Tensor,
    k: &Tensor,
    head_dim: usize,
    rope_base: f64,
    seq_len: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    let inv_freq = (0..head_dim)
        .step_by(2)
        .map(|idx| 1f32 / rope_base.powf(idx as f64 / head_dim as f64) as f32)
        .collect::<Vec<_>>();
    let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), q.device())?;
    let positions = Tensor::arange(0u32, seq_len as u32, q.device())?
        .to_dtype(DType::F32)?
        .reshape((seq_len, 1))?;
    let freqs = positions.matmul(&inv_freq)?;
    let sin = freqs.sin()?;
    let cos = freqs.cos()?;
    let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
    let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
    Ok((q, k))
}

fn config_usize(config: &serde_json::Value, field: &str) -> Option<usize> {
    config
        .get(field)
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
}

fn config_f64(config: &serde_json::Value, field: &str) -> Option<f64> {
    config.get(field).and_then(|value| value.as_f64())
}

fn tensor_f32(
    tensors: &SafeTensors<'_>,
    name: &str,
    model_path: &Path,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let tensor = tensors
        .tensor(name)
        .map_err(|_| CgError::EmbedderModelCorrupt {
            path: model_path.display().to_string(),
        })?;
    let shape = tensor.shape().to_vec();
    let values = match tensor.dtype() {
        safetensors::Dtype::F32 => {
            if tensor.data().len() % 4 != 0 {
                return Err(model_corrupt(model_path));
            }
            tensor
                .data()
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes =
                        <[u8; 4]>::try_from(chunk).map_err(|_| model_corrupt(model_path))?;
                    Ok(f32::from_le_bytes(bytes))
                })
                .collect::<Result<Vec<_>>>()?
        }
        safetensors::Dtype::BF16 => {
            if tensor.data().len() % 2 != 0 {
                return Err(model_corrupt(model_path));
            }
            tensor
                .data()
                .chunks_exact(2)
                .map(|chunk| {
                    let bytes =
                        <[u8; 2]>::try_from(chunk).map_err(|_| model_corrupt(model_path))?;
                    let bits = (u16::from_le_bytes(bytes) as u32) << 16;
                    Ok(f32::from_bits(bits))
                })
                .collect::<Result<Vec<_>>>()?
        }
        _ => {
            return Err(CgError::EmbedderModelCorrupt {
                path: model_path.display().to_string(),
            });
        }
    };
    if values.len() != shape.iter().product::<usize>() {
        return Err(model_corrupt(model_path));
    }
    Ok((values, shape))
}

fn require_shape(actual: &[usize], expected: &[usize], model_path: &Path) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(model_corrupt(model_path))
    }
}

fn require_shape_2(
    actual: &[usize],
    expected_rows: Option<usize>,
    expected_cols: Option<usize>,
    model_path: &Path,
) -> Result<()> {
    match actual {
        [rows, cols]
            if expected_rows.is_none_or(|expected| *rows == expected)
                && expected_cols.is_none_or(|expected| *cols == expected)
                && *rows > 0
                && *cols > 0 =>
        {
            Ok(())
        }
        _ => Err(model_corrupt(model_path)),
    }
}

fn model_corrupt(model_path: &Path) -> CgError {
    CgError::EmbedderModelCorrupt {
        path: model_path.display().to_string(),
    }
}

fn matryoshka_head(values: &[f32], dimension: usize) -> Vec<f32> {
    let head_dimension = dimension.min(values.len());
    let mut head = values[..head_dimension].to_vec();
    let norm = head.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in &mut head {
        *value /= norm;
    }
    head
}
