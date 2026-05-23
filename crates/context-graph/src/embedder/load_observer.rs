use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedderLoadEvent {
    FetchStarted {
        model_id: String,
        byte_size: Option<u64>,
        destination: PathBuf,
    },
    FetchCompleted {
        model_id: String,
        byte_size: Option<u64>,
        destination: PathBuf,
        elapsed: Duration,
    },
    CacheHit {
        model_id: String,
        destination: PathBuf,
    },
}

pub trait EmbedderLoadObserver: Send + Sync {
    fn observe(&self, event: EmbedderLoadEvent);
}
