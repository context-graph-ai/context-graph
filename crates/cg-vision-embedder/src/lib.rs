//! Bundled joint image+text vision embedder (SigLIP-family on candle).
//!
//! The vision sibling of the bundled text embedders: loads permissively-licensed
//! SigLIP/SigLIP2 weights from a LOCAL directory (mmap, no network at inference)
//! and implements [`context_graph::Embedder`], so an embedded consumer registers
//! it as an embedding space and hands context-graph precomputed vectors — or asks
//! it to embed in-memory image bytes directly. CPU by default (pure-Rust gemm);
//! GPU is the `cuda` feature flag on the same call site.

use std::path::{Path, PathBuf};

use context_graph::{CgError, EmbeddingInput, EmbeddingOutput, Result};

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

/// A loaded SigLIP-family embedder producing L2-normalized joint-space vectors.
pub struct SiglipVisionEmbedder {
    _private: (),
}

impl SiglipVisionEmbedder {
    /// Load weights from the local directory. Fails with a typed error when the
    /// directory or required files are missing — never panics, never fetches.
    pub fn load(config: SiglipEmbedderConfig) -> Result<Self> {
        let _ = config;
        todo!("stub for RED verification; implemented in the next commit")
    }

    /// The pooled embedding dimension of the loaded checkpoint (e.g. 768).
    pub fn dimension(&self) -> usize {
        todo!("stub")
    }

    /// Whether the checkpoint's text tower loaded (enables text embedding).
    pub fn has_text_tower(&self) -> bool {
        todo!("stub")
    }

    /// Embed an in-memory encoded image (PNG/JPEG bytes) to an L2-normalized vector.
    pub fn embed_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let _ = bytes;
        todo!("stub")
    }

    /// Embed an image file on the local filesystem.
    pub fn embed_image_path(&self, path: &Path) -> Result<Vec<f32>> {
        let _ = path;
        todo!("stub")
    }

    /// Embed text through the joint space (requires the text tower).
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let _ = text;
        todo!("stub")
    }
}

impl context_graph::Embedder for SiglipVisionEmbedder {
    fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
        let _ = input;
        todo!("stub")
    }

    fn supported_inputs(&self) -> Vec<context_graph::EvidenceKind> {
        todo!("stub")
    }

    fn embedding_space(&self) -> context_graph::EmbeddingSpace {
        todo!("stub")
    }
}

/// Keep the unused-dep lints quiet in the stub commit; removed by the implementation.
#[allow(unused)]
fn _stub_uses() {
    let _ = CgError::Engine(String::new());
}
