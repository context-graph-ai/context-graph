mod embeddinggemma;

pub use embeddinggemma::{
    EmbeddingGemmaForwardStage, EmbeddingGemmaLoadEvent, EmbeddingGemmaTextEmbedder, LazyHandle,
};

pub fn default_embeddinggemma_offline_config() -> context_graph::EmbedderConfig {
    context_graph::EmbedderConfig {
        source: context_graph::ModelSource::HuggingFace {
            repo: "google/embeddinggemma-300m".to_string(),
        },
        offline: true,
        cache_dir: None,
        pinned_revision: None,
        network_policy: "offline-cache-only".to_string(),
    }
}
