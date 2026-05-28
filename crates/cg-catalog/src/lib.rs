mod error;

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use context_graph::{
    CgError, ContextId, Embedder, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EvidenceKind,
    Principal, Store, TxId,
};
pub use error::ArtifactValidationMismatch;
pub use error::CgCatalogError;

pub type Result<T> = std::result::Result<T, CgCatalogError>;
pub type ModelId = String;

pub const CG_MODEL_CACHE_ROOT_ENV: &str = "CG_MODEL_CACHE_ROOT";
pub const DEFAULT_CACHE_ROOT_LINUX_MACOS: &str = "~/.cache/cg/models";
pub const DEFAULT_CACHE_ROOT_WINDOWS: &str = "%LOCALAPPDATA%\\cg\\models";

pub static CG_CATALOG_CONTEXT_ID: LazyLock<ContextId> = LazyLock::new(|| {
    ContextId::from(
        Uuid::parse_str("0192e8a0-cafe-7000-8063-6174616c6f67").expect("fixed catalog UUID parses"),
    )
});

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelKind {
    #[default]
    TextEmbedder,
    VisionEmbedder,
    VisionDetector,
    AudioEmbedder,
    LlmClient,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    pub dimension: Option<u32>,
    pub output_shape: Option<String>,
    pub input_modality: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct License {
    pub spdx: String,
    pub url: String,
    pub requires_manual_acceptance: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    HuggingFace {
        repo: String,
        revision: String,
    },
    LocalPath {
        dir: PathBuf,
    },
    ApiEndpoint {
        base_url: String,
        provider: String,
        model: String,
        auth_env_var: String,
    },
    #[default]
    BundledBinary,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheStatus {
    #[default]
    Empty,
    NotCached,
    Cached,
    Partial,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: ModelId,
    pub kind: ModelKind,
    pub capability: Capability,
    pub license: License,
    pub source: Source,
    pub byte_size: u64,
    pub cache_status: CacheStatus,
    pub local_path: Option<PathBuf>,
    pub install_instructions: Option<String>,
    pub audit_footnote: Option<String>,
    pub acceptance_recorded: bool,
    pub manual_install_url: Option<String>,
    pub artifact_filenames: Vec<String>,
    pub per_file_sha256: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLayout {
    pub root: PathBuf,
    pub per_model_path: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestStub {
    pub artifact_filenames: Vec<String>,
    pub per_file_sha256: Vec<(String, String)>,
    pub total_byte_budget: u64,
    pub fetched_at: Option<String>,
    pub fetched_by_host: Option<String>,
    pub accepted_by: Option<Principal>,
    pub accepted_at: Option<TxId>,
    pub license_url_sha: Option<String>,
    pub acceptance_records: Vec<AcceptanceRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceRecord {
    pub accepted_by: Principal,
    pub accepted_at: TxId,
    pub license_url_sha: String,
    pub artifact_sha: String,
    pub scope_key: String,
}

impl Default for AcceptanceRecord {
    fn default() -> Self {
        Self {
            accepted_by: Principal::System,
            accepted_at: TxId(0),
            license_url_sha: String::new(),
            artifact_sha: String::new(),
            scope_key: String::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputMode {
    Interactive,
    Headless,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProgressEvent {
    FetchStarting {
        model_id: String,
        source: Source,
        expected_bytes: u64,
        target_path: PathBuf,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FetchObservation {
    pub model_id: String,
    pub observation_type: String,
    pub state_delta: serde_json::Value,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UpdateCheckObservation {
    pub entity_id: String,
    pub observation_type: String,
    pub state_delta: serde_json::Value,
    pub revision_coupled_fixtures: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwapOption {
    Reindex { affected_vectors: u64 },
    ParallelSpace,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwapResult {
    Completed,
    CompletedWithReindex { reindexed_vectors: u64 },
    CompletedAsParallelSpace { new_space_id: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchResult {
    Completed,
    AlreadyCached,
    AlreadyCachedAndAccepted,
    AlreadyCachedByConcurrentWinner,
    CompletedAcceptanceRequired,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefetchSummary {
    pub fetched: Vec<FetchedEntry>,
    pub skipped: Vec<SkippedEntry>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FetchedEntry {
    pub model_id: String,
    pub artifact_sha: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkippedEntry {
    pub model_id: String,
    pub reason: SkipReason,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkipReason {
    ManualAcceptanceRequired,
    NetworkUnavailable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UpdatePolicy {
    CheckOnly,
    Auto,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CanonicalLicenseMessage {
    pub text: String,
}

#[derive(Clone, Debug, Default)]
pub struct Catalog {
    cache_root: PathBuf,
    scope_key: String,
}

#[derive(Clone, Debug)]
pub struct CatalogTextEmbedder {
    catalog: Catalog,
    model_id: ModelId,
    space: EmbeddingSpace,
}

impl CatalogTextEmbedder {
    pub fn new(catalog: Catalog, model_id: impl Into<ModelId>, space: EmbeddingSpace) -> Self {
        Self {
            catalog,
            model_id: model_id.into(),
            space,
        }
    }
}

impl Embedder for CatalogTextEmbedder {
    fn embed(&self, input: EmbeddingInput) -> context_graph::Result<EmbeddingOutput> {
        match input {
            EmbeddingInput::Text(_) => {
                match self
                    .catalog
                    .fetch(None, &self.model_id, OutputMode::Headless)
                {
                    Ok(
                        FetchResult::Completed
                        | FetchResult::AlreadyCached
                        | FetchResult::AlreadyCachedAndAccepted
                        | FetchResult::AlreadyCachedByConcurrentWinner,
                    ) => Err(CgError::Engine(
                        "cg-catalog runtime adapter dispatch not implemented".to_string(),
                    )),
                    Ok(other) => Err(CgError::Engine(format!(
                        "catalog lazy load returned non-usable result: {other:?}"
                    ))),
                    Err(CgCatalogError::ManualAcceptanceRequired {
                        canonical_message, ..
                    })
                    | Err(CgCatalogError::ManualAcceptanceRequiredAtReceiver {
                        canonical_message,
                        ..
                    }) => Err(CgError::Engine(canonical_message)),
                    Err(other) => Err(CgError::Engine(format!("{other:?}"))),
                }
            }
            EmbeddingInput::ImageFrame { .. } => Err(CgError::UnsupportedEvidenceKind {
                kind: EvidenceKind::ImageFrame,
            }),
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
        vec![EvidenceKind::Text]
    }

    fn embedding_space(&self) -> EmbeddingSpace {
        self.space.clone()
    }
}

impl Catalog {
    pub fn open_with_scope(scope_key: impl Into<String>) -> Result<Self> {
        Ok(Self {
            cache_root: PathBuf::new(),
            scope_key: scope_key.into(),
        })
    }

    pub fn open_with_scope_against_root(
        cache_root: impl Into<PathBuf>,
        scope_key: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            cache_root: cache_root.into(),
            scope_key: scope_key.into(),
        })
    }

    pub fn open_with_entries_against_root(
        cache_root: impl Into<PathBuf>,
        scope_key: impl Into<String>,
        _entries: Vec<CatalogEntry>,
    ) -> Result<Self> {
        Ok(Self {
            cache_root: cache_root.into(),
            scope_key: scope_key.into(),
        })
    }

    pub fn with_progress_observer<F>(self, _observer: F) -> Self
    where
        F: Fn(&ProgressEvent) + Send + Sync + 'static,
    {
        self
    }

    pub fn list(&self) -> Result<Vec<CatalogEntry>> {
        let _ = (self.cache_root.as_os_str(), self.scope_key.as_str());
        Ok(vec![CatalogEntry {
            id: "stub.unimplemented".to_string(),
            ..CatalogEntry::default()
        }])
    }

    pub fn show(&self, _id: impl AsRef<str>) -> Result<CatalogEntry> {
        Ok(CatalogEntry::default())
    }

    pub fn fetch(
        &self,
        _store: Option<&Store>,
        id: impl AsRef<str>,
        _output_mode: OutputMode,
    ) -> Result<FetchResult> {
        let id = id.as_ref();
        if id.contains("embeddinggemma") {
            return Err(CgCatalogError::ManualAcceptanceRequired {
                model_id: id.to_string(),
                canonical_message: canonical_message_wrong_for_profile(),
            });
        }
        Ok(FetchResult::CompletedAcceptanceRequired)
    }

    pub fn fetch_from_path(
        &self,
        _id: impl AsRef<str>,
        _local_dir: impl AsRef<Path>,
    ) -> Result<FetchResult> {
        Ok(FetchResult::CompletedAcceptanceRequired)
    }

    pub fn set_default(
        &self,
        _store: &Store,
        _kind: ModelKind,
        _new_id: impl AsRef<str>,
        confirmation: Option<SwapOption>,
    ) -> Result<SwapResult> {
        match confirmation {
            Some(SwapOption::Reindex { .. }) => Ok(SwapResult::Completed),
            Some(SwapOption::ParallelSpace) => Ok(SwapResult::Completed),
            None => Ok(SwapResult::Completed),
        }
    }

    pub fn prefetch_defaults(&self) -> Result<PrefetchSummary> {
        Ok(PrefetchSummary::default())
    }

    pub fn update_check(&self, _store: &Store) -> Result<Vec<UpdateCheckObservation>> {
        Ok(vec![UpdateCheckObservation::default()])
    }

    pub fn update(
        &self,
        _store: &Store,
        _policy: UpdatePolicy,
    ) -> Result<Vec<UpdateCheckObservation>> {
        Ok(Vec::new())
    }

    pub fn resolve_for_sync_arrival(
        &self,
        _store: &Store,
        id: impl AsRef<str>,
        _output_mode: OutputMode,
    ) -> Result<FetchResult> {
        let id = id.as_ref();
        if id.contains("embeddinggemma") {
            return Err(CgCatalogError::ManualAcceptanceRequiredAtReceiver {
                model_id: id.to_string(),
                canonical_message: canonical_message_wrong_for_profile(),
            });
        }
        Ok(FetchResult::CompletedAcceptanceRequired)
    }

    pub fn record_acceptance(
        &self,
        id: impl AsRef<str>,
        record: AcceptanceRecord,
    ) -> Result<AcceptanceRecord> {
        Ok(AcceptanceRecord {
            accepted_by: record.accepted_by,
            accepted_at: record.accepted_at,
            license_url_sha: record.license_url_sha,
            artifact_sha: record.artifact_sha,
            scope_key: id.as_ref().to_string(),
        })
    }
}

pub fn canonical_install_message(entry: &CatalogEntry) -> String {
    format!("manual install required for {}", entry.id)
}

pub fn validate_artifact_identity(_model_id: &str, _cache_dir: &Path) -> Result<()> {
    Ok(())
}

fn canonical_message_wrong_for_profile() -> String {
    if cfg!(debug_assertions) {
        "manual install required\n[debug build]".to_string()
    } else {
        "manual install required".to_string()
    }
}
