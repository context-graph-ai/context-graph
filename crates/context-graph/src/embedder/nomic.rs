use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use std::{
    fs as file_system, io,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{
    Config as NomicBertConfig, NomicBertModel, l2_normalize as candle_l2_normalize, mean_pooling,
};
use safetensors::{Dtype as SafeTensorDType, SafeTensors};
use tokenizers::Tokenizer;

use crate::embedder::load_observer::{EmbedderLoadEvent, EmbedderLoadObserver};
use crate::embedder::{Embedder, EmbedderConfig, EmbeddingInput, EmbeddingOutput};
use crate::{
    CgError, DistanceMetric, EmbeddingSpace, EmbeddingSpaceBinding, EvidenceKind, EvidenceProducer,
    ModelSource, Result,
};

const NOMIC_TEXT_DEFAULT_ID: &str = "text_default_nomic_v1_5_768";
const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
const NOMIC_DEFAULT_DIMENSION: usize = 768;
const NOMIC_MODEL_BYTES: u64 = 547_112_608;

const NOMIC_REQUIRED_TENSORS: &[(&str, &[usize])] = &[
    ("embeddings.word_embeddings.weight", &[30_528, 768]),
    ("embeddings.token_type_embeddings.weight", &[2, 768]),
    ("emb_ln.weight", &[768]),
    ("emb_ln.bias", &[768]),
    ("encoder.layers.0.attn.Wqkv.weight", &[2_304, 768]),
    ("encoder.layers.0.attn.out_proj.weight", &[768, 768]),
    ("encoder.layers.0.mlp.fc11.weight", &[3_072, 768]),
    ("encoder.layers.0.mlp.fc12.weight", &[3_072, 768]),
    ("encoder.layers.0.mlp.fc2.weight", &[768, 3_072]),
    ("encoder.layers.0.norm1.weight", &[768]),
    ("encoder.layers.0.norm2.weight", &[768]),
    ("encoder.layers.11.attn.Wqkv.weight", &[2_304, 768]),
    ("encoder.layers.11.norm2.weight", &[768]),
];

static NOMIC_MODEL_CACHE: OnceLock<Mutex<HashMap<NomicModelCacheKey, Arc<LoadedNomicModel>>>> =
    OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NomicModelCacheKey {
    model_dir: PathBuf,
    pinned_revision: Option<String>,
}

#[derive(Clone)]
pub struct NomicTextEmbedder {
    config: EmbedderConfig,
    dimension: usize,
    load_observer: Option<Arc<dyn EmbedderLoadObserver>>,
    loader_events: Arc<Mutex<Vec<EmbedderLoadEvent>>>,
}

impl std::fmt::Debug for NomicTextEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NomicTextEmbedder")
            .field("config", &self.config)
            .field("dimension", &self.dimension)
            .field("has_load_observer", &self.load_observer.is_some())
            .finish()
    }
}

impl NomicTextEmbedder {
    pub fn new(config: EmbedderConfig) -> Self {
        Self::new_for_dimension(config, NOMIC_DEFAULT_DIMENSION)
    }

    pub fn new_for_dimension(config: EmbedderConfig, dimension: usize) -> Self {
        Self {
            config,
            dimension,
            load_observer: None,
            loader_events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn new_with_load_observer(
        config: EmbedderConfig,
        load_observer: Arc<dyn EmbedderLoadObserver>,
    ) -> Self {
        Self {
            config,
            dimension: NOMIC_DEFAULT_DIMENSION,
            load_observer: Some(load_observer),
            loader_events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn structured_loader_events(&self) -> Vec<EmbedderLoadEvent> {
        self.loader_events
            .lock()
            .map(|events| events.clone())
            .unwrap_or_default()
    }

    fn observe(&self, event: EmbedderLoadEvent) {
        if let Ok(mut events) = self.loader_events.lock() {
            events.push(event.clone());
        }
        if let Some(observer) = &self.load_observer {
            observer.observe(event);
        }
    }

    fn load_model(&self) -> Result<Arc<LoadedNomicModel>> {
        if matches!(self.config.source, ModelSource::Disabled) {
            return Err(CgError::TextEmbedderDisabled);
        }
        let model_dir = resolve_model_dir(&self.config)?;
        let cache_key = NomicModelCacheKey {
            model_dir: model_dir.clone(),
            pinned_revision: self.config.pinned_revision.clone(),
        };
        let cache = NOMIC_MODEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache
            .lock()
            .map_err(|_| CgError::Engine("nomic model cache lock poisoned".to_string()))?;
        if let Some(model) = guard.get(&cache_key) {
            let loaded = Arc::clone(model);
            let revision_path = model_dir.join("REVISION");
            check_revision_marker(&revision_path, self.config.pinned_revision.as_deref())?;
            self.observe(EmbedderLoadEvent::CacheHit {
                model_id: NOMIC_MODEL_ID.to_string(),
                destination: model_dir,
            });
            return Ok(loaded);
        }

        self.ensure_cache_ready(&model_dir)?;
        let loaded = Arc::new(load_nomic_artifacts(&model_dir)?);
        guard.insert(cache_key, Arc::clone(&loaded));
        Ok(loaded)
    }

    fn ensure_cache_ready(&self, model_dir: &Path) -> Result<()> {
        if matches!(self.config.source, ModelSource::Disabled) {
            return Ok(());
        }

        let model_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config.json");
        let revision_path = model_dir.join("REVISION");
        let all_present = cache_file_present(&model_path)?
            && cache_file_present(&tokenizer_path)?
            && cache_file_present(&config_path)?;
        if all_present {
            cache_files_parse(
                &model_path,
                &tokenizer_path,
                &config_path,
                &revision_path,
                self.config.pinned_revision.as_deref(),
            )?;
            self.observe(EmbedderLoadEvent::CacheHit {
                model_id: NOMIC_MODEL_ID.to_string(),
                destination: model_dir.to_path_buf(),
            });
            return Ok(());
        }
        if !matches!(self.config.source, ModelSource::HuggingFace { .. }) {
            return Err(CgError::EmbedderModelNotAvailable {
                source: source_name(&self.config),
                path: model_dir.display().to_string(),
            });
        }
        if self.config.offline || self.config.network_policy.contains("offline") {
            return Err(CgError::EmbedderModelNotAvailable {
                source: source_name(&self.config),
                path: model_dir.display().to_string(),
            });
        }

        let started = Instant::now();
        self.observe(EmbedderLoadEvent::FetchStarted {
            model_id: NOMIC_MODEL_ID.to_string(),
            byte_size: Some(NOMIC_MODEL_BYTES),
            destination: model_dir.to_path_buf(),
        });
        eprintln!(
            "context-graph embedding fetch started model={} path={}",
            NOMIC_MODEL_ID,
            model_dir.display()
        );
        file_system::create_dir_all(model_dir).map_err(|err| {
            CgError::Engine(format!(
                "nomic cache directory create failed at {}: {err}",
                model_dir.display()
            ))
        })?;
        let repo = match &self.config.source {
            ModelSource::HuggingFace { repo } => repo.as_str(),
            _ => unreachable!("source checked above"),
        };
        let revision = self.config.pinned_revision.as_deref().unwrap_or("main");
        download_hf_file(
            repo,
            revision,
            "model.safetensors",
            &model_path,
            NOMIC_MODEL_BYTES + 16_777_216,
        )?;
        download_hf_file(
            repo,
            revision,
            "tokenizer.json",
            &tokenizer_path,
            64 * 1024 * 1024,
        )?;
        download_hf_file(repo, revision, "config.json", &config_path, 1024 * 1024)?;
        file_system::write(&revision_path, revision).map_err(|err| {
            CgError::Engine(format!(
                "nomic cache revision write failed at {}: {err}",
                revision_path.display()
            ))
        })?;
        cache_files_parse(
            &model_path,
            &tokenizer_path,
            &config_path,
            &revision_path,
            self.config.pinned_revision.as_deref(),
        )?;
        let elapsed = started.elapsed();
        self.observe(EmbedderLoadEvent::FetchCompleted {
            model_id: NOMIC_MODEL_ID.to_string(),
            byte_size: Some(NOMIC_MODEL_BYTES),
            destination: model_dir.to_path_buf(),
            elapsed,
        });
        eprintln!(
            "context-graph embedding fetch completed model={} path={}",
            NOMIC_MODEL_ID,
            model_dir.display()
        );
        Ok(())
    }

    fn forward(&self, loaded: &LoadedNomicModel, text: &str) -> Result<Vec<f32>> {
        let encoding =
            loaded
                .tokenizer
                .encode(text, true)
                .map_err(|_| CgError::EmbedderModelCorrupt {
                    path: loaded.tokenizer_path.clone(),
                })?;
        let input_ids = Tensor::new(encoding.get_ids(), &loaded.device)
            .map_err(candle_err)?
            .unsqueeze(0)
            .map_err(candle_err)?;
        let attention_mask = Tensor::new(encoding.get_attention_mask(), &loaded.device)
            .map_err(candle_err)?
            .unsqueeze(0)
            .map_err(candle_err)?;
        let hidden = loaded
            .model
            .forward(&input_ids, None, Some(&attention_mask))
            .map_err(candle_err)?;
        let pooled = mean_pooling(&hidden, &attention_mask).map_err(candle_err)?;
        let normalized = candle_l2_normalize(&pooled).map_err(candle_err)?;
        let values = normalized.to_vec2::<f32>().map_err(candle_err)?;
        let full = values
            .first()
            .ok_or_else(|| CgError::Engine("nomic model returned empty batch".to_string()))?;
        if self.dimension > full.len() {
            return Err(CgError::EmbeddingDimensionMismatch {
                space_id: text_space_id_for_dimension(self.dimension),
                expected: full.len() as u32,
                found: self.dimension as u32,
            });
        }
        let mut vector = full[..self.dimension].to_vec();
        l2_normalize(&mut vector);
        Ok(vector)
    }
}

impl Embedder for NomicTextEmbedder {
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
        if matches!(self.config.source, ModelSource::Disabled) {
            return Err(CgError::TextEmbedderDisabled);
        }
        let loaded = self.load_model()?;
        let vector = self.forward(&loaded, &text)?;
        Ok(EmbeddingOutput {
            embedding_space_id: self.embedding_space().embedding_space_id,
            dimension: self.dimension,
            producer: EvidenceProducer {
                system: crate::CG_EDGE_SYSTEM_TAG.to_string(),
                model_name: NOMIC_MODEL_ID.to_string(),
                model_version: "1.5".to_string(),
                pipeline_version: "nomic-bert-candle".to_string(),
            },
            vector,
        })
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        let embedding_space_id = text_space_id_for_dimension(self.dimension);
        let mut bindings = vec![EmbeddingSpaceBinding {
            embedding_space_id: embedding_space_id.clone(),
            table_name: "evidence".to_string(),
            vector_column: "vector_text".to_string(),
        }];
        if self.dimension == NOMIC_DEFAULT_DIMENSION {
            bindings.push(EmbeddingSpaceBinding {
                embedding_space_id: embedding_space_id.clone(),
                table_name: "digests".to_string(),
                vector_column: "vector_text".to_string(),
            });
        }
        EmbeddingSpace {
            embedding_space_id,
            dimension: self.dimension,
            metric: DistanceMetric::Cosine,
            model_family: "text".to_string(),
            model_name: NOMIC_MODEL_ID.to_string(),
            model_version: "1.5".to_string(),
            supported_evidence_kinds: vec![EvidenceKind::Text],
            default_for_evidence_kind: (self.dimension == NOMIC_DEFAULT_DIMENSION)
                .then_some(EvidenceKind::Text),
            bindings,
        }
    }
}

struct LoadedNomicModel {
    model: NomicBertModel,
    tokenizer: Tokenizer,
    tokenizer_path: String,
    device: Device,
}

fn cache_files_parse(
    model_path: &Path,
    tokenizer_path: &Path,
    config_path: &Path,
    revision_path: &Path,
    expected_revision: Option<&str>,
) -> Result<()> {
    check_revision_marker(revision_path, expected_revision)?;
    let model_metadata =
        file_system::metadata(model_path).map_err(|_| CgError::EmbedderModelCorrupt {
            path: model_path.display().to_string(),
        })?;
    if model_metadata.len() < 1_000_000 {
        return Err(CgError::EmbedderModelCorrupt {
            path: model_path.display().to_string(),
        });
    }
    let config_bytes =
        file_system::read(config_path).map_err(|_| CgError::EmbedderModelCorrupt {
            path: config_path.display().to_string(),
        })?;
    let config: NomicBertConfig =
        serde_json::from_slice(&config_bytes).map_err(|_| CgError::EmbedderModelCorrupt {
            path: config_path.display().to_string(),
        })?;
    if config.n_embd != NOMIC_DEFAULT_DIMENSION || config.n_head != 12 || config.n_layer != 12 {
        return Err(CgError::EmbedderModelCorrupt {
            path: config_path.display().to_string(),
        });
    }
    let model_bytes = file_system::read(model_path).map_err(|_| CgError::EmbedderModelCorrupt {
        path: model_path.display().to_string(),
    })?;
    let tensors =
        SafeTensors::deserialize(&model_bytes).map_err(|_| CgError::EmbedderModelCorrupt {
            path: model_path.display().to_string(),
        })?;
    for (name, shape) in NOMIC_REQUIRED_TENSORS {
        let tensor = tensors
            .tensor(name)
            .map_err(|_| CgError::EmbedderModelCorrupt {
                path: model_path.display().to_string(),
            })?;
        if tensor.dtype() != SafeTensorDType::F32 || tensor.shape() != *shape {
            return Err(CgError::EmbedderModelCorrupt {
                path: model_path.display().to_string(),
            });
        }
    }
    Tokenizer::from_file(tokenizer_path).map_err(|_| CgError::EmbedderModelCorrupt {
        path: tokenizer_path.display().to_string(),
    })?;
    Ok(())
}

fn check_revision_marker(revision_path: &Path, expected_revision: Option<&str>) -> Result<()> {
    if let Some(expected) = expected_revision {
        let actual = file_system::read_to_string(revision_path).unwrap_or_default();
        if actual.trim() != expected {
            return Err(CgError::EmbedderModelRevisionMismatch {
                expected: expected.to_string(),
                actual: actual.trim().to_string(),
            });
        }
    }
    Ok(())
}

fn load_nomic_artifacts(model_dir: &Path) -> Result<LoadedNomicModel> {
    let model_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let config_path = model_dir.join("config.json");
    let config_bytes =
        file_system::read(&config_path).map_err(|_| CgError::EmbedderModelCorrupt {
            path: config_path.display().to_string(),
        })?;
    let config: NomicBertConfig =
        serde_json::from_slice(&config_bytes).map_err(|_| CgError::EmbedderModelCorrupt {
            path: config_path.display().to_string(),
        })?;
    let model_bytes =
        file_system::read(&model_path).map_err(|_| CgError::EmbedderModelCorrupt {
            path: model_path.display().to_string(),
        })?;
    let device = Device::Cpu;
    let vb =
        VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, &device).map_err(|_| {
            CgError::EmbedderModelCorrupt {
                path: model_path.display().to_string(),
            }
        })?;
    let model = NomicBertModel::load(vb, &config).map_err(|_| CgError::EmbedderModelCorrupt {
        path: model_path.display().to_string(),
    })?;
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|_| CgError::EmbedderModelCorrupt {
            path: tokenizer_path.display().to_string(),
        })?;
    Ok(LoadedNomicModel {
        model,
        tokenizer,
        tokenizer_path: tokenizer_path.display().to_string(),
        device,
    })
}

fn download_hf_file(
    repo: &str,
    revision: &str,
    file_name: &str,
    destination: &Path,
    byte_limit: u64,
) -> Result<()> {
    let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{file_name}");
    let tmp = destination.with_extension("download");
    let mut response = ureq::get(&url)
        .call()
        .map_err(|_| CgError::EmbedderModelNotAvailable {
            source: format!("HuggingFace({repo})"),
            path: destination.display().to_string(),
        })?;
    let mut reader = response.body_mut().with_config().limit(byte_limit).reader();
    let mut out = file_system::File::create(&tmp).map_err(|err| {
        CgError::Engine(format!(
            "nomic artifact temp create failed at {}: {err}",
            tmp.display()
        ))
    })?;
    io::copy(&mut reader, &mut out).map_err(|_| CgError::EmbedderModelNotAvailable {
        source: format!("HuggingFace({repo})"),
        path: destination.display().to_string(),
    })?;
    file_system::rename(&tmp, destination).map_err(|err| {
        CgError::Engine(format!(
            "nomic artifact move failed {} -> {}: {err}",
            tmp.display(),
            destination.display()
        ))
    })
}

fn cache_file_present(path: &Path) -> Result<bool> {
    path.try_exists().map_err(|err| {
        CgError::Engine(format!(
            "nomic cache path status failed at {}: {err}",
            path.display()
        ))
    })
}

fn text_space_id_for_dimension(dimension: usize) -> String {
    if dimension == NOMIC_DEFAULT_DIMENSION {
        NOMIC_TEXT_DEFAULT_ID.to_string()
    } else {
        format!("text_default_nomic_v1_5_{dimension}")
    }
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

fn resolve_model_dir(config: &EmbedderConfig) -> Result<PathBuf> {
    match &config.source {
        ModelSource::LocalPath(path) => Ok(path.clone()),
        ModelSource::HuggingFace { .. } => Ok(config
            .cache_dir
            .clone()
            .unwrap_or_else(default_cache_root)
            .join(NOMIC_MODEL_ID)),
        ModelSource::Disabled => Ok(PathBuf::new()),
    }
}

fn source_name(config: &EmbedderConfig) -> String {
    match &config.source {
        ModelSource::LocalPath(path) => format!("LocalPath({})", path.display()),
        ModelSource::HuggingFace { repo } => format!("HuggingFace({repo})"),
        ModelSource::Disabled => "Disabled".to_string(),
    }
}

fn default_cache_root() -> PathBuf {
    if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("cg")
            .join("embedders")
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|home| home.join(".cache"))
            })
            .unwrap_or_else(std::env::temp_dir)
            .join("cg")
            .join("embedders")
    }
}

fn candle_err(err: candle_core::Error) -> CgError {
    CgError::Engine(format!("nomic candle tensor: {err}"))
}
