//! Bundled joint image+text vision embedder (SigLIP-family on candle).
//!
//! The vision sibling of the bundled text embedders: loads permissively-licensed
//! SigLIP/SigLIP2 weights from a LOCAL directory (mmap, no network at inference)
//! and implements [`context_graph::Embedder`], so an embedded consumer registers
//! it as an embedding space and hands context-graph precomputed vectors — or asks
//! it to embed in-memory image bytes directly. CPU by default (pure-Rust gemm);
//! GPU is the `cuda` feature flag on the same call site.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip;
use context_graph::{
    CgError, EmbeddingInput, EmbeddingOutput, EvidenceKind, EvidenceProducer, Result,
};
use tokenizers::Tokenizer;

/// Where the weights live and how the embedding space is named.
///
/// `weights_dir` must contain `model.safetensors` and the checkpoint's own
/// `config.json`; `tokenizer.json` is optional and enables the text tower
/// (zero-shot class checks). No network is touched at load or inference.
#[derive(Debug, Clone)]
pub struct SiglipEmbedderConfig {
    pub weights_dir: PathBuf,
    pub embedding_space_id: String,
    pub model_name: String,
    pub model_version: String,
}

/// Either the full joint model (vision + text) or the vision tower alone, for
/// checkpoints whose text encoder does not match candle's siglip text
/// architecture. A crop-only consumer works either way.
enum Backbone {
    Full(Box<siglip::Model>),
    VisionOnly(Box<siglip::VisionModel>),
}

/// A loaded SigLIP-family embedder producing L2-normalized joint-space vectors.
pub struct SiglipVisionEmbedder {
    backbone: Backbone,
    tokenizer: Option<Tokenizer>,
    model_config: siglip::Config,
    device: Device,
    embedding_space_id: String,
    model_name: String,
    model_version: String,
}

fn engine_err(context: &str, err: impl std::fmt::Display) -> CgError {
    CgError::Engine(format!("vision embedder {context}: {err}"))
}

fn select_device() -> Device {
    // GPU-as-a-flag: with the `cuda` feature and an NVIDIA device present this
    // selects the GPU at the same call site; otherwise CPU. Same code.
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::cuda_if_available(0) {
            return device;
        }
    }
    Device::Cpu
}

impl SiglipVisionEmbedder {
    /// Load weights from the local directory. Fails with a typed error when the
    /// directory or required files are missing — never panics, never fetches.
    pub fn load(config: SiglipEmbedderConfig) -> Result<Self> {
        let weights = config.weights_dir.join("model.safetensors");
        if !weights.exists() {
            return Err(engine_err(
                "load",
                format!("missing weights file {}", weights.display()),
            ));
        }
        let model_config_path = config.weights_dir.join("config.json");
        let model_config: siglip::Config = if model_config_path.exists() {
            serde_json::from_slice(
                &std::fs::read(&model_config_path).map_err(|e| engine_err("read config", e))?,
            )
            .map_err(|e| engine_err("parse config.json", e))?
        } else {
            siglip::Config::base_patch16_224()
        };
        let device = select_device();
        let weights_file = weights.to_string_lossy().to_string();
        let var_builder = || unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights_file),
                DType::F32,
                &device,
            )
        };
        // Prefer the full joint model; fall back to the vision tower alone when
        // the checkpoint's text encoder does not match (crop path unaffected).
        let (backbone, tokenizer) = match siglip::Model::new(
            &model_config,
            var_builder().map_err(|e| engine_err("mmap", e))?,
        ) {
            Ok(model) => {
                let tokenizer =
                    Tokenizer::from_file(config.weights_dir.join("tokenizer.json")).ok();
                (Backbone::Full(Box::new(model)), tokenizer)
            }
            Err(_) => {
                let vb = var_builder().map_err(|e| engine_err("mmap", e))?;
                let vision = siglip::VisionModel::new(
                    &model_config.vision_config,
                    true,
                    vb.pp("vision_model"),
                )
                .map_err(|e| engine_err("load vision tower", e))?;
                (Backbone::VisionOnly(Box::new(vision)), None)
            }
        };
        Ok(Self {
            backbone,
            tokenizer,
            model_config,
            device,
            embedding_space_id: config.embedding_space_id,
            model_name: config.model_name,
            model_version: config.model_version,
        })
    }

    /// The pooled embedding dimension of the loaded checkpoint (e.g. 768).
    pub fn dimension(&self) -> usize {
        self.model_config.vision_config.hidden_size
    }

    /// Whether the checkpoint's text tower loaded (enables text embedding).
    pub fn has_text_tower(&self) -> bool {
        matches!(self.backbone, Backbone::Full(_)) && self.tokenizer.is_some()
    }

    /// Embed an in-memory encoded image (PNG/JPEG bytes) to an L2-normalized vector.
    pub fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let decoded = image::load_from_memory(bytes).map_err(|e| engine_err("decode image", e))?;
        self.embed_dynamic_image(decoded)
    }

    /// Embed an image file on the local filesystem.
    pub fn embed_image_path(&self, path: &Path) -> Result<Vec<f32>> {
        let decoded = image::ImageReader::open(path)
            .map_err(|e| engine_err("open image", e))?
            .decode()
            .map_err(|e| engine_err("decode image", e))?;
        self.embed_dynamic_image(decoded)
    }

    fn embed_dynamic_image(&self, decoded: image::DynamicImage) -> Result<Vec<f32>> {
        // Preprocess exactly like SiglipImageProcessor: resize-to-fill to the
        // checkpoint's input size, rescale 1/255, normalize mean=.5 std=.5.
        let image_size = self.model_config.vision_config.image_size;
        let resized = decoded.resize_to_fill(
            image_size as u32,
            image_size as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let raw = resized.to_rgb8().into_raw();
        let tensor = Tensor::from_vec(raw, (image_size, image_size, 3), &self.device)
            .and_then(|t| t.permute((2, 0, 1)))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.affine(2. / 255., -1.))
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| engine_err("preprocess", e))?;
        let features = match &self.backbone {
            Backbone::Full(model) => model
                .get_image_features(&tensor)
                .map_err(|e| engine_err("vision forward", e))?,
            Backbone::VisionOnly(vision) => vision
                .forward(&tensor)
                .map_err(|e| engine_err("vision forward", e))?,
        };
        let vector = features
            .squeeze(0)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| engine_err("read features", e))?;
        Ok(l2_normalize(vector))
    }

    /// Embed text through the joint space (requires the text tower).
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let (Backbone::Full(model), Some(tokenizer)) = (&self.backbone, &self.tokenizer) else {
            return Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::Text,
            });
        };
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| engine_err("tokenize", e))?;
        let mut ids = encoding.get_ids().to_vec();
        let max_len = self.model_config.text_config.max_position_embeddings;
        let pad = self.model_config.text_config.pad_token_id;
        ids.truncate(max_len);
        while ids.len() < max_len {
            ids.push(pad);
        }
        let input = Tensor::new(vec![ids], &self.device).map_err(|e| engine_err("tokens", e))?;
        let features = model
            .get_text_features(&input)
            .map_err(|e| engine_err("text forward", e))?;
        let vector = features
            .squeeze(0)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| engine_err("read features", e))?;
        Ok(l2_normalize(vector))
    }

    fn producer(&self) -> EvidenceProducer {
        EvidenceProducer {
            system: "cg-vision-embedder".to_string(),
            model_name: self.model_name.clone(),
            model_version: self.model_version.clone(),
            pipeline_version: "1".to_string(),
        }
    }

    fn output(&self, vector: Vec<f32>) -> EmbeddingOutput {
        EmbeddingOutput {
            embedding_space_id: self.embedding_space_id.clone(),
            dimension: vector.len(),
            producer: self.producer(),
            vector,
        }
    }
}

fn l2_normalize(vector: Vec<f32>) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        vector
    } else {
        vector.into_iter().map(|x| x / norm).collect()
    }
}

impl context_graph::Embedder for SiglipVisionEmbedder {
    fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
        match input {
            EmbeddingInput::ImageBytes(bytes) => Ok(self.output(self.embed_image_bytes(&bytes)?)),
            EmbeddingInput::ImageFrame { uri } => {
                let path = uri.strip_prefix("file://").unwrap_or(&uri);
                Ok(self.output(self.embed_image_path(Path::new(path))?))
            }
            EmbeddingInput::Text(text) => Ok(self.output(self.embed_text(&text)?)),
            EmbeddingInput::AudioSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::AudioSegment,
            }),
            EmbeddingInput::VideoSegment { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::VideoSegment,
            }),
            EmbeddingInput::StructuredSignal(_) => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::StructuredSignal,
            }),
        }
    }

    fn supported_inputs(&self) -> Vec<EvidenceKind> {
        if self.has_text_tower() {
            vec![EvidenceKind::ImageFrame, EvidenceKind::Text]
        } else {
            vec![EvidenceKind::ImageFrame]
        }
    }

    fn embedding_space(&self) -> context_graph::EmbeddingSpace {
        context_graph::EmbeddingSpace {
            embedding_space_id: self.embedding_space_id.clone(),
            dimension: self.dimension(),
            metric: context_graph::DistanceMetric::Cosine,
            model_family: "vision".to_string(),
            model_name: self.model_name.clone(),
            model_version: self.model_version.clone(),
            supported_evidence_kinds: self.supported_inputs(),
            default_for_evidence_kind: None,
            bindings: Vec::new(),
        }
    }
}
