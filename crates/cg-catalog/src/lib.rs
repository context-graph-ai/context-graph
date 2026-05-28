mod error;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use candle_embeddinggemma::EmbeddingGemmaTextEmbedder;
#[cfg(any(test, feature = "embeddinggemma-test-support"))]
#[doc(hidden)]
pub mod embeddinggemma_test_support {
    pub use candle_embeddinggemma::{
        EmbeddingGemmaForwardStage, EmbeddingGemmaLoadEvent, EmbeddingGemmaTextEmbedder,
        default_embeddinggemma_offline_config,
    };
}
use chrono::Utc;
use fs4::fs_std::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use context_graph::DistanceMetric;
use context_graph::embedder::nomic::NomicTextEmbedder;
use context_graph::{
    CgError, ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, Embedder,
    EmbedderConfig, EmbeddingInput, EmbeddingOutput, EmbeddingSpace, EmbeddingSpaceBinding,
    EmbeddingSpaceDecl, Entity, EntityId, EntityPatch, EntityType, EvidenceId, EvidenceKind,
    EvidenceProducer, EvidenceRef, IntentionId, ListEntityFilter, ModelSource, ObservationId,
    Principal, ReasoningStep, RecordObservation, ReindexOptions, RetentionStatus, Store, TxId,
};
pub use error::{ArtifactValidationMismatch, CgCatalogError};

pub type Result<T> = std::result::Result<T, CgCatalogError>;
pub type ModelId = String;

pub const CG_MODEL_CACHE_ROOT_ENV: &str = "CG_MODEL_CACHE_ROOT";
pub const DEFAULT_CACHE_ROOT_LINUX_MACOS: &str = "~/.cache/cg/models";
pub const DEFAULT_CACHE_ROOT_WINDOWS: &str = "%LOCALAPPDATA%\\cg\\models";

const NOMIC_ID: &str = "nomic-embed-text-v1.5";
const GEMMA_ID: &str = "embeddinggemma-300m";

const NOMIC_REPO: &str = "nomic-ai/nomic-embed-text-v1.5";
const GEMMA_REPO: &str = "google/embeddinggemma-300m";
const NOMIC_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";
const GEMMA_REVISION: &str = "57c266a740f537b4dc058e1b0cda161fd15afa75";
const NOMIC_BYTES: u64 = 546_938_168;
const GEMMA_BYTES: u64 = 1_241_923_584;
const CG_BINARY_ID: &str = "cg-binary";
const FOOTNOTE_EMBEDDINGGEMMA: &str = "OSS purity is enforced by .github/workflows/ci.yml; a downstream consumer's own dependency is outside this audit.";

const PRODUCTION_NOMIC_MODEL_SAFETENSORS_SHA: &str =
    "9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c";
const PRODUCTION_NOMIC_TOKENIZER_JSON_SHA: &str =
    "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66";
const PRODUCTION_NOMIC_CONFIG_JSON_SHA: &str =
    "9ab00bd92cee80a569f708140b7b6c1661a65891ff3765b1519e181ba2f2c92b";
const PRODUCTION_NOMIC_REVISION_SHA: &str =
    "b6e8c34dc906161922aedff26e55cc6ec85162b83826c43ef604452489c407f1";
const PRODUCTION_GEMMA_MODEL_SAFETENSORS_SHA: &str =
    "cbf5a78393b6a033e0b8a63a57549964f7ed5c6fbeb4ba0694214f36123f2fd2";
const PRODUCTION_GEMMA_TOKENIZER_JSON_SHA: &str =
    "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e";
const PRODUCTION_GEMMA_CONFIG_JSON_SHA: &str =
    "8f863f76e2d9c710cc833dc92efa898c9adfd41031c786507cc6b0e49c2e3e68";
const PRODUCTION_GEMMA_REVISION_SHA: &str =
    "240bf6458e37b64ab66077a6e6900fdb4089c848f8372fa64b8dc490d4ef81e3";

const ARTIFACT_FILES: [&str; 4] = [
    "model.safetensors",
    "tokenizer.json",
    "config.json",
    "REVISION",
];
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
struct CatalogRegistrations {
    entries: Vec<CatalogEntry>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

type ProgressObserver = dyn Fn(&ProgressEvent) + Send + Sync + 'static;

struct StoreObservationRequest<'a> {
    observation_id: ObservationId,
    evidence_id: EvidenceId,
    entity_id: EntityId,
    observation_type: &'a str,
    state_delta: BTreeMap<String, Option<serde_json::Value>>,
    properties: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Default)]
pub struct Catalog {
    cache_root: PathBuf,
    scope_key: String,
    progress_observer: Option<Arc<ProgressObserver>>,
}

impl fmt::Debug for Catalog {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Catalog")
            .field("cache_root", &self.cache_root)
            .field("scope_key", &self.scope_key)
            .field("progress_observer", &self.progress_observer.is_some())
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct CatalogTextEmbedder {
    catalog: Catalog,
    model_id: ModelId,
    space: EmbeddingSpace,
}

#[derive(Clone)]
struct ModelSpec {
    id: &'static str,
    kind: ModelKind,
    dimension: Option<u32>,
    input_modality: Option<&'static str>,
    license_spdx: &'static str,
    license_url: &'static str,
    requires_manual_acceptance: bool,
    source_repo: Option<&'static str>,
    revision: &'static str,
    byte_size: u64,
    manual_install_url: Option<&'static str>,
    audit_footnote: Option<&'static str>,
    visible: bool,
}

#[derive(Clone, Debug)]
struct SourceOverride {
    base_url: String,
    revision: String,
    sha256: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArtifactValidationFlavor {
    Production,
    OverrideOrLocal,
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
            EmbeddingInput::Text(text) => {
                match self
                    .catalog
                    .fetch(None, &self.model_id, OutputMode::Headless)
                {
                    Ok(
                        FetchResult::Completed
                        | FetchResult::AlreadyCached
                        | FetchResult::AlreadyCachedAndAccepted
                        | FetchResult::AlreadyCachedByConcurrentWinner,
                    ) => {
                        let config = EmbedderConfig {
                            source: ModelSource::LocalPath(self.catalog.model_dir(&self.model_id)),
                            offline: true,
                            cache_dir: Some(self.catalog.root()),
                            pinned_revision: Some(
                                self.catalog
                                    .model_revision_for_runtime(&self.model_id)
                                    .map_err(|err| CgError::Engine(err.to_string()))?,
                            ),
                            network_policy: "catalog-managed-local".to_string(),
                        };
                        if self.model_id == NOMIC_ID {
                            let embedder =
                                NomicTextEmbedder::new_for_dimension(config, self.space.dimension);
                            let mut output = embedder.embed(EmbeddingInput::Text(text))?;
                            output.embedding_space_id = self.space.embedding_space_id.clone();
                            output.dimension = self.space.dimension;
                            output.producer = EvidenceProducer {
                                system: context_graph::CG_EDGE_SYSTEM_TAG.to_string(),
                                model_name: self.model_id.clone(),
                                model_version: NOMIC_REVISION.to_string(),
                                pipeline_version: "nomic-bert-candle-catalog-managed".to_string(),
                            };
                            Ok(output)
                        } else if self.model_id == GEMMA_ID {
                            let embedder = EmbeddingGemmaTextEmbedder::new_for_dimension(
                                config,
                                self.space.dimension,
                            );
                            let mut output = embedder.embed(EmbeddingInput::Text(text))?;
                            output.embedding_space_id = self.space.embedding_space_id.clone();
                            output.dimension = self.space.dimension;
                            Ok(output)
                        } else {
                            Err(CgError::Engine(format!(
                                "catalog-managed embedder {model_id} has no linked runtime adapter",
                                model_id = self.model_id
                            )))
                        }
                    }
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
            cache_root: default_cache_root(),
            scope_key: scope_key.into(),
            progress_observer: None,
        })
    }

    pub fn open_with_scope_against_root(
        cache_root: impl Into<PathBuf>,
        scope_key: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            cache_root: cache_root.into(),
            scope_key: scope_key.into(),
            progress_observer: None,
        })
    }

    pub fn open_with_entries_against_root(
        cache_root: impl Into<PathBuf>,
        scope_key: impl Into<String>,
        entries: Vec<CatalogEntry>,
    ) -> Result<Self> {
        let catalog = Self {
            cache_root: cache_root.into(),
            scope_key: scope_key.into(),
            progress_observer: None,
        };
        catalog.write_registered_entries(&entries)?;
        Ok(catalog)
    }

    pub fn with_progress_observer<F>(mut self, observer: F) -> Self
    where
        F: Fn(&ProgressEvent) + Send + Sync + 'static,
    {
        self.progress_observer = Some(Arc::new(observer));
        self
    }

    pub fn list(&self) -> Result<Vec<CatalogEntry>> {
        let registered_entries = self.registered_entries()?;
        let mut entries = Vec::new();
        if let Some(registered_entries) = &registered_entries {
            for entry in registered_entries {
                entries.push(self.entry_for_registration(entry)?);
            }
        } else {
            for spec in default_specs().into_iter().filter(|spec| spec.visible) {
                entries.push(self.entry_for_spec(spec)?);
            }
        }

        let registered_ids = registered_entries.as_ref().map(|entries| {
            entries
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<BTreeSet<_>>()
        });
        for dir in self.cache_dirs()? {
            let Some(id) = dir.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let is_registered = registered_ids
                .as_ref()
                .is_some_and(|registered_ids| registered_ids.contains(id));
            let is_default =
                registered_ids.is_none() && default_specs().iter().any(|spec| spec.id == id);
            if id == "cg" || is_registered || is_default {
                continue;
            }
            let manifest = dir.join("catalog.toml");
            if manifest.exists() {
                entries.push(self.entry_from_manifest(id, None)?);
            }
        }

        entries.sort_by(compare_catalog_entries);
        Ok(entries)
    }

    pub fn show(&self, id: impl AsRef<str>) -> Result<CatalogEntry> {
        let id = id.as_ref();
        if let Some(entry) = self
            .registered_entries()?
            .and_then(|entries| entries.into_iter().find(|entry| entry.id == id))
        {
            return self.entry_for_registration(&entry);
        }
        if let Some(spec) = spec_by_id(id) {
            return self.entry_for_spec(spec);
        }
        let manifest = self.model_dir(id).join("catalog.toml");
        if manifest.exists() {
            return self.entry_from_manifest(id, None);
        }
        Err(CgCatalogError::NotFound)
    }

    pub fn fetch(
        &self,
        store: Option<&Store>,
        id: impl AsRef<str>,
        output_mode: OutputMode,
    ) -> Result<FetchResult> {
        self.fetch_inner(store, id.as_ref(), output_mode, false)
    }

    pub fn fetch_from_path(
        &self,
        id: impl AsRef<str>,
        local_dir: impl AsRef<Path>,
    ) -> Result<FetchResult> {
        let id = id.as_ref();
        let spec = spec_by_id(id).ok_or(CgCatalogError::NotFound)?;
        let lock = self.lock_model(id)?;
        let source_dir = local_dir.as_ref();
        let temp_dir = self.temp_publish_dir(id);
        clean_dir(&temp_dir)?;
        fs::create_dir_all(&temp_dir).map_err(|err| io_error(&temp_dir, err))?;
        for file in ARTIFACT_FILES {
            fs::copy(source_dir.join(file), temp_dir.join(file))
                .map_err(|err| io_error(&source_dir.join(file), err))?;
        }
        let revision = self
            .source_override(id)?
            .map(|source| source.revision)
            .unwrap_or_else(|| spec.revision.to_string());
        self.validate_temp_artifacts(
            &spec,
            &temp_dir,
            ArtifactValidationFlavor::OverrideOrLocal,
            &revision,
            false,
        )?;
        let source = Source::LocalPath {
            dir: source_dir
                .canonicalize()
                .unwrap_or_else(|_| source_dir.to_path_buf()),
        };
        self.publish_temp_artifacts_with_revision_and_source(
            &spec,
            &temp_dir,
            &revision,
            Some(&source),
        )?;
        drop(lock);
        Ok(if spec.requires_manual_acceptance {
            FetchResult::CompletedAcceptanceRequired
        } else {
            FetchResult::Completed
        })
    }

    pub fn set_default(
        &self,
        store: &Store,
        kind: ModelKind,
        new_id: impl AsRef<str>,
        confirmation: Option<SwapOption>,
    ) -> Result<SwapResult> {
        if kind != ModelKind::TextEmbedder {
            return Err(CgCatalogError::NotFound);
        }
        let new_id = new_id.as_ref();
        let proposed_dim = self.model_dimension_for_id(new_id)?;
        let config_path = self.runtime_config_path();
        let mut config = read_toml(&config_path)?.unwrap_or_else(empty_toml_table);
        let current_id = toml_str(&config, "default_text_embedder")
            .unwrap_or(NOMIC_ID)
            .to_string();
        let current_dim = self.model_dimension_for_id(&current_id).unwrap_or(768);
        let proposed_space_id = text_space_id_for_model(new_id, proposed_dim);
        let has_existing_space =
            self.store_has_text_space(store, &proposed_space_id, proposed_dim)?;
        if proposed_dim != current_dim && confirmation.is_none() && !has_existing_space {
            let affected_vectors =
                self.affected_vectors_for_swap(store, &current_id, current_dim)?;
            return Err(CgCatalogError::DimensionChangeRequiresConfirmation {
                current_dim,
                proposed_dim,
                options: vec![
                    SwapOption::Reindex { affected_vectors },
                    SwapOption::ParallelSpace,
                ],
            });
        }

        let mut decision_option = confirmation.clone();
        let result = match confirmation.clone() {
            Some(SwapOption::Reindex { .. }) => {
                let reindexed_vectors =
                    self.run_reindex_for_swap(store, &current_id, current_dim)?;
                decision_option = Some(SwapOption::Reindex {
                    affected_vectors: reindexed_vectors,
                });
                SwapResult::CompletedWithReindex { reindexed_vectors }
            }
            Some(SwapOption::ParallelSpace) => {
                let decl = parallel_text_embedding_space_decl(new_id, proposed_dim);
                store
                    .register_embedding_space(decl)
                    .map_err(|err| store_error(store.db_path(), err))?;
                SwapResult::CompletedAsParallelSpace {
                    new_space_id: proposed_space_id,
                }
            }
            None => SwapResult::Completed,
        };
        self.record_swap_decision(
            store,
            &current_id,
            new_id,
            decision_option.as_ref(),
            current_dim,
            proposed_dim,
        )?;
        set_toml_string(&mut config, "default_text_embedder", new_id);
        write_toml(&config_path, &config)?;
        Ok(result)
    }

    pub fn prefetch_defaults(&self) -> Result<PrefetchSummary> {
        let mut summary = PrefetchSummary::default();
        match self.fetch(None, NOMIC_ID, OutputMode::Headless) {
            Ok(
                FetchResult::Completed
                | FetchResult::AlreadyCached
                | FetchResult::AlreadyCachedByConcurrentWinner,
            ) => {
                let artifact_sha =
                    sha256_file(&self.model_dir(NOMIC_ID).join("model.safetensors"))?;
                summary.fetched.push(FetchedEntry {
                    model_id: NOMIC_ID.to_string(),
                    artifact_sha,
                });
            }
            Ok(
                FetchResult::AlreadyCachedAndAccepted | FetchResult::CompletedAcceptanceRequired,
            ) => {}
            Err(err) => return Err(err),
        }
        summary.skipped.push(SkippedEntry {
            model_id: GEMMA_ID.to_string(),
            reason: SkipReason::ManualAcceptanceRequired,
        });
        Ok(summary)
    }

    pub fn update_check(&self, store: &Store) -> Result<Vec<UpdateCheckObservation>> {
        self.update_inner(store, UpdatePolicy::CheckOnly)
    }

    pub fn update(
        &self,
        store: &Store,
        policy: UpdatePolicy,
    ) -> Result<Vec<UpdateCheckObservation>> {
        self.update_inner(store, policy)
    }

    pub fn resolve_for_sync_arrival(
        &self,
        store: &Store,
        id: impl AsRef<str>,
        output_mode: OutputMode,
    ) -> Result<FetchResult> {
        self.fetch_inner(Some(store), id.as_ref(), output_mode, true)
    }

    pub fn record_acceptance(
        &self,
        id: impl AsRef<str>,
        mut record: AcceptanceRecord,
    ) -> Result<AcceptanceRecord> {
        let id = id.as_ref();
        let spec = spec_by_id(id).ok_or(CgCatalogError::NotFound)?;
        let lock = self.lock_model(id)?;
        let missing = self.missing_files(id);
        if !missing.is_empty() {
            return Err(CgCatalogError::CacheIncomplete {
                missing_files: missing,
            });
        }
        validate_artifact_identity(id, &self.model_dir(id))?;
        let actual = sha256_file(&self.model_dir(id).join("model.safetensors"))?;
        let source = self.source_override(id)?;
        let expected = expected_model_shas(
            &spec,
            ArtifactValidationFlavor::OverrideOrLocal,
            source.as_ref(),
        );
        if !expected.contains(&actual) {
            return Err(CgCatalogError::ArtifactValidationFailed {
                file: "model.safetensors".to_string(),
                expected_sha: expected.join(" or "),
                actual_sha: actual,
                mismatches: Vec::new(),
            });
        }
        if record.artifact_sha != actual {
            return Err(CgCatalogError::ArtifactShaMismatch {
                expected: actual,
                actual: record.artifact_sha,
            });
        }
        record.scope_key = self.scope_key.clone();
        let manifest_path = self.manifest_path(id);
        let mut manifest = read_toml(&manifest_path)?.unwrap_or_else(empty_toml_table);
        let records = acceptance_records_from_manifest(&manifest);
        let mut accepted_ats = records
            .iter()
            .map(|record| record.accepted_at)
            .chain(std::iter::once(record.accepted_at))
            .collect::<Vec<_>>();
        accepted_ats.sort_unstable_by_key(|accepted_at| accepted_at.0);
        if let Some(window) = accepted_ats
            .windows(2)
            .find(|window| window[0].0 >= window[1].0)
        {
            return Err(CgCatalogError::NonMonotonicAcceptanceTimestamp {
                previous: window[0],
                proposed: window[1],
            });
        }
        append_acceptance_record(&mut manifest, &record);
        write_toml(&manifest_path, &manifest)?;
        drop(lock);
        Ok(record)
    }

    fn fetch_inner(
        &self,
        store: Option<&Store>,
        id: &str,
        output_mode: OutputMode,
        sync_arrival: bool,
    ) -> Result<FetchResult> {
        let Some(spec) = spec_by_id(id) else {
            if let Some(entry) = self
                .registered_entries()?
                .and_then(|entries| entries.into_iter().find(|entry| entry.id == id))
            {
                if entry.license.requires_manual_acceptance {
                    return Err(CgCatalogError::ManualAcceptanceRequired {
                        model_id: id.to_string(),
                        canonical_message: canonical_install_message(&entry),
                    });
                }
                if matches!(entry.source, Source::ApiEndpoint { .. }) {
                    return Ok(FetchResult::AlreadyCached);
                }
            }
            if sync_arrival {
                if let Some(store) = store {
                    self.record_unknown_observation(store, id)?;
                }
                return Err(CgCatalogError::UnknownModelId { id: id.to_string() });
            }
            return Err(CgCatalogError::NotFound);
        };

        let complete_before = self.cache_complete(id)?;
        if spec.requires_manual_acceptance && !self.acceptance_recorded_for_scope(id)? {
            if sync_arrival {
                let canonical_message = canonical_install_message(&self.entry_for_spec(spec)?);
                if let Some(store) = store {
                    self.record_license_deferred(store, id, &canonical_message)?;
                }
                return Err(CgCatalogError::ManualAcceptanceRequiredAtReceiver {
                    model_id: id.to_string(),
                    canonical_message,
                });
            }
            return Err(CgCatalogError::ManualAcceptanceRequired {
                model_id: id.to_string(),
                canonical_message: canonical_install_message(&self.entry_for_spec(spec)?),
            });
        }
        if complete_before {
            if sync_arrival
                && !spec.requires_manual_acceptance
                && self.model_dir(id).join(".lock").exists()
            {
                return Ok(FetchResult::AlreadyCachedByConcurrentWinner);
            }
            return Ok(if spec.requires_manual_acceptance {
                FetchResult::AlreadyCachedAndAccepted
            } else {
                FetchResult::AlreadyCached
            });
        }

        self.validate_existing_cache_before_fetch(id)?;

        self.emit_progress(&spec, &output_mode)?;
        let lock = self.lock_model(id)?;
        if self.cache_complete(id)? {
            drop(lock);
            return Ok(FetchResult::AlreadyCachedByConcurrentWinner);
        }
        if let Err(err) = self.validate_existing_cache_before_fetch(id) {
            drop(lock);
            return Err(err);
        }

        self.fetch_to_temp_and_publish(spec.clone())?;
        drop(lock);
        if let Some(store) = store
            && (sync_arrival || matches!(output_mode, OutputMode::Interactive))
        {
            self.record_fetch_observation(store, spec, output_mode)?;
        }
        Ok(FetchResult::Completed)
    }

    fn update_inner(
        &self,
        store: &Store,
        policy: UpdatePolicy,
    ) -> Result<Vec<UpdateCheckObservation>> {
        let mut observations = Vec::new();
        for spec in default_specs().into_iter().filter(|spec| spec.visible) {
            if spec.requires_manual_acceptance && self.source_override(spec.id)?.is_none() {
                continue;
            }
            let current_revision = self
                .revision_on_disk(spec.id)
                .or_else(|| self.existing_catalog_entity_revision(store, spec.id))
                .unwrap_or_else(|| spec.revision.to_string());
            let upstream_revision = self.upstream_revision(spec.clone())?;
            if current_revision == upstream_revision {
                continue;
            }
            if spec.requires_manual_acceptance {
                if matches!(policy, UpdatePolicy::Auto) {
                    observations.push(self.record_update_observation(
                        store,
                        spec.id,
                        "manual_acceptance_required",
                        json!({
                            "model_id": spec.id,
                            "previous_revision": current_revision,
                            "new_revision": upstream_revision,
                            "applied": false,
                        }),
                        BTreeMap::new(),
                    )?);
                }
                continue;
            }
            let applied = matches!(policy, UpdatePolicy::Auto);
            if applied {
                let lock = self.lock_model(spec.id)?;
                self.fetch_to_temp_and_publish_with_revision(spec.clone(), &upstream_revision)?;
                drop(lock);
            }
            let revision_coupled_fixtures = revision_coupled_fixtures(spec.id);
            let release_notes_url = spec
                .source_repo
                .map(|repo| format!("https://huggingface.co/{repo}/releases"))
                .unwrap_or_default();
            observations.push(self.record_update_observation(
                store,
                spec.id,
                "model_update_available",
                json!({
                    "model_id": spec.id,
                    "catalog_revision": upstream_revision.clone(),
                    "previous_revision": current_revision,
                    "current_revision": upstream_revision.clone(),
                    "new_revision": upstream_revision,
                    "release_notes_url": release_notes_url,
                    "applied": applied,
                    "revision_coupled_fixtures": revision_coupled_fixtures,
                }),
                BTreeMap::new(),
            )?);
        }

        if let Some(source) = self.source_override(CG_BINARY_ID)? {
            let current_release = self
                .existing_catalog_entity_revision(store, CG_BINARY_ID)
                .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
            let upstream_release = fetch_revision_from_source_override(CG_BINARY_ID, &source)?;
            if current_release != upstream_release {
                observations.push(self.record_update_observation(
                    store,
                    CG_BINARY_ID,
                    "binary_update_available",
                    json!({
                        "binary": "cg",
                        "previous_release": current_release,
                        "current_revision": upstream_release,
                        "new_release": upstream_release,
                        "release_notes_url": "https://github.com/context-graph-ai/context-graph/releases",
                        "applied": false,
                    }),
                    BTreeMap::from([("binary".to_string(), json!("cg"))]),
                )?);
            }
        }

        Ok(observations)
    }

    fn upstream_revision(&self, spec: ModelSpec) -> Result<String> {
        if let Some(source) = self.source_override(spec.id)? {
            return fetch_revision_from_source_override(spec.id, &source);
        }
        fetch_latest_huggingface_revision(&spec)
    }

    fn entry_for_spec(&self, spec: ModelSpec) -> Result<CatalogEntry> {
        self.entry_from_manifest(spec.id, Some(spec))
    }

    fn entry_for_registration(&self, registration: &CatalogEntry) -> Result<CatalogEntry> {
        self.entry_from_manifest_or_registration(&registration.id, None, Some(registration))
    }

    fn entry_from_manifest(&self, id: &str, spec: Option<ModelSpec>) -> Result<CatalogEntry> {
        self.entry_from_manifest_or_registration(id, spec, None)
    }

    fn entry_from_manifest_or_registration(
        &self,
        id: &str,
        spec: Option<ModelSpec>,
        registration: Option<&CatalogEntry>,
    ) -> Result<CatalogEntry> {
        let model_dir = self.model_dir(id);
        let spec = spec.or_else(|| spec_by_id(id));
        let _snapshot_lock = if spec.is_some() {
            Some(self.lock_model(id)?)
        } else {
            None
        };
        let manifest = read_toml(&self.manifest_path(id))?;
        let source_override = self.source_override(id)?;
        if spec.is_none() && manifest.is_none() && registration.is_none() {
            return Err(CgCatalogError::NotFound);
        }
        let kind = manifest
            .as_ref()
            .and_then(|manifest| toml_str(manifest, "kind"))
            .and_then(parse_kind)
            .or_else(|| spec.as_ref().map(|spec| spec.kind.clone()))
            .or_else(|| registration.map(|entry| entry.kind.clone()))
            .unwrap_or(ModelKind::TextEmbedder);
        let dimension = manifest
            .as_ref()
            .and_then(|manifest| toml_u64(manifest, "dimension"))
            .map(|value| value as u32)
            .or_else(|| spec.as_ref().and_then(|spec| spec.dimension))
            .or_else(|| registration.and_then(|entry| entry.capability.dimension));
        let input_modality = manifest
            .as_ref()
            .and_then(|manifest| toml_str(manifest, "input_modality").map(str::to_string))
            .or_else(|| {
                spec.as_ref()
                    .and_then(|spec| spec.input_modality.map(str::to_string))
            })
            .or_else(|| registration.and_then(|entry| entry.capability.input_modality.clone()));
        let license_url = manifest
            .as_ref()
            .and_then(|manifest| toml_str(manifest, "license_url"))
            .or_else(|| spec.as_ref().map(|spec| spec.license_url))
            .or_else(|| registration.map(|entry| entry.license.url.as_str()))
            .unwrap_or("");
        let license_spdx = manifest
            .as_ref()
            .and_then(|manifest| toml_str(manifest, "license_spdx"))
            .or_else(|| spec.as_ref().map(|spec| spec.license_spdx))
            .or_else(|| registration.map(|entry| entry.license.spdx.as_str()))
            .unwrap_or("Apache-2.0");
        let requires_manual_acceptance = spec
            .as_ref()
            .is_some_and(|spec| spec.requires_manual_acceptance)
            || registration.is_some_and(|entry| entry.license.requires_manual_acceptance)
            || license_spdx == "Gemma";
        let repo = manifest
            .as_ref()
            .and_then(|manifest| toml_str(manifest, "source_repo"))
            .or_else(|| spec.as_ref().and_then(|spec| spec.source_repo))
            .or_else(|| match registration.map(|entry| &entry.source) {
                Some(Source::HuggingFace { repo, .. }) => Some(repo.as_str()),
                _ => None,
            })
            .unwrap_or("");
        let revision = source_override
            .as_ref()
            .map(|source| source.revision.as_str())
            .or_else(|| {
                manifest
                    .as_ref()
                    .and_then(|manifest| toml_str(manifest, "source_revision"))
            })
            .or_else(|| spec.as_ref().map(|spec| spec.revision))
            .or_else(|| match registration.map(|entry| &entry.source) {
                Some(Source::HuggingFace { revision, .. }) => Some(revision.as_str()),
                _ => None,
            })
            .unwrap_or("");
        let source = if manifest
            .as_ref()
            .and_then(|manifest| toml_str(manifest, "source_kind"))
            == Some("LocalPath")
        {
            Source::LocalPath {
                dir: manifest
                    .as_ref()
                    .and_then(|manifest| toml_str(manifest, "source_dir"))
                    .map(PathBuf::from)
                    .unwrap_or_else(|| model_dir.clone()),
            }
        } else if let Some(Source::ApiEndpoint {
            base_url,
            provider,
            model,
            auth_env_var,
        }) = registration.map(|entry| &entry.source)
        {
            Source::ApiEndpoint {
                base_url: base_url.clone(),
                provider: provider.clone(),
                model: model.clone(),
                auth_env_var: auth_env_var.clone(),
            }
        } else if repo.is_empty() {
            match registration.map(|entry| &entry.source) {
                Some(Source::LocalPath { dir }) => Source::LocalPath { dir: dir.clone() },
                _ => Source::BundledBinary,
            }
        } else {
            Source::HuggingFace {
                repo: repo.to_string(),
                revision: revision.to_string(),
            }
        };
        let is_api_endpoint = matches!(&source, Source::ApiEndpoint { .. });
        let cache_status = if is_api_endpoint {
            CacheStatus::Cached
        } else {
            self.cache_status(id, manifest.as_ref(), spec.as_ref())?
        };
        let artifact_filenames = if is_api_endpoint {
            Vec::new()
        } else if manifest.is_none() {
            registration
                .map(|entry| entry.artifact_filenames.clone())
                .unwrap_or_else(|| artifact_filenames(&model_dir, manifest.as_ref(), &cache_status))
        } else {
            artifact_filenames(&model_dir, manifest.as_ref(), &cache_status)
        };
        let per_file_sha256 = if is_api_endpoint {
            Vec::new()
        } else if manifest.is_none() {
            if let Some(registration) = registration {
                registration.per_file_sha256.clone()
            } else {
                per_file_sha256(&model_dir, manifest.as_ref(), spec.as_ref())?
            }
        } else {
            per_file_sha256(&model_dir, manifest.as_ref(), spec.as_ref())?
        };
        let byte_size = if is_api_endpoint {
            0
        } else {
            manifest
                .as_ref()
                .and_then(|manifest| toml_u64(manifest, "byte_size"))
                .or_else(|| {
                    if spec.as_ref().is_some_and(|spec| spec.id == NOMIC_ID) {
                        Some(NOMIC_BYTES)
                    } else if cache_status == CacheStatus::Cached {
                        Some(sum_regular_file_sizes(&model_dir)?)
                    } else {
                        spec.as_ref().map(|spec| spec.byte_size)
                    }
                })
                .or_else(|| registration.map(|entry| entry.byte_size))
                .unwrap_or_default()
        };
        let manual_install_url = if requires_manual_acceptance {
            manifest
                .as_ref()
                .and_then(|manifest| toml_str(manifest, "manual_install_url").map(str::to_string))
                .or_else(|| {
                    spec.as_ref()
                        .and_then(|spec| spec.manual_install_url.map(str::to_string))
                })
                .or_else(|| registration.and_then(|entry| entry.manual_install_url.clone()))
        } else {
            None
        };
        let audit_footnote = manifest
            .as_ref()
            .and_then(|manifest| toml_str(manifest, "audit_footnote").map(str::to_string))
            .or_else(|| {
                spec.as_ref()
                    .and_then(|spec| spec.audit_footnote.map(str::to_string))
            })
            .or_else(|| registration.and_then(|entry| entry.audit_footnote.clone()));
        let acceptance_records = manifest
            .as_ref()
            .map(acceptance_records_from_manifest)
            .unwrap_or_default();
        let latest_acceptance_for_scope = acceptance_records
            .iter()
            .filter(|record| record.scope_key == self.scope_key)
            .max_by_key(|record| record.accepted_at)
            .cloned();
        let manifest_stub = ManifestStub {
            artifact_filenames,
            per_file_sha256,
            total_byte_budget: byte_size,
            fetched_at: manifest
                .as_ref()
                .and_then(|manifest| toml_str(manifest, "fetched_at").map(str::to_string)),
            fetched_by_host: manifest
                .as_ref()
                .and_then(|manifest| toml_str(manifest, "fetched_by_host").map(str::to_string)),
            accepted_by: latest_acceptance_for_scope
                .as_ref()
                .map(|record| record.accepted_by.clone()),
            accepted_at: latest_acceptance_for_scope
                .as_ref()
                .map(|record| record.accepted_at),
            license_url_sha: latest_acceptance_for_scope
                .as_ref()
                .map(|record| record.license_url_sha.clone()),
            acceptance_records,
        };
        let local_path = if is_api_endpoint {
            None
        } else {
            (cache_status == CacheStatus::Cached).then_some(model_dir)
        };
        let acceptance_recorded = manifest_stub.acceptance_records.iter().any(|record| {
            record.scope_key == self.scope_key
                && manifest_stub.accepted_at == Some(record.accepted_at)
                && manifest_stub.accepted_by.as_ref() == Some(&record.accepted_by)
                && manifest_stub.license_url_sha.as_ref() == Some(&record.license_url_sha)
        });
        let byte_size = manifest_stub.total_byte_budget;
        let artifact_filenames = manifest_stub.artifact_filenames;
        let per_file_sha256 = manifest_stub.per_file_sha256;
        let capability = Capability {
            dimension,
            output_shape: None,
            input_modality,
        };
        let license = License {
            spdx: license_spdx.to_string(),
            url: license_url.to_string(),
            requires_manual_acceptance,
        };
        let install_instructions = if requires_manual_acceptance {
            let install_entry = CatalogEntry {
                id: id.to_string(),
                kind: kind.clone(),
                capability: capability.clone(),
                license: license.clone(),
                source: source.clone(),
                byte_size,
                cache_status: cache_status.clone(),
                local_path: local_path.clone(),
                install_instructions: None,
                audit_footnote: audit_footnote.clone(),
                acceptance_recorded,
                manual_install_url: manual_install_url.clone(),
                artifact_filenames: artifact_filenames.clone(),
                per_file_sha256: per_file_sha256.clone(),
            };
            Some(canonical_install_message(&install_entry))
        } else {
            None
        };

        Ok(CatalogEntry {
            id: id.to_string(),
            kind,
            capability,
            license,
            source,
            byte_size,
            cache_status: cache_status.clone(),
            local_path,
            install_instructions,
            audit_footnote,
            acceptance_recorded,
            manual_install_url,
            artifact_filenames,
            per_file_sha256,
        })
    }

    fn cache_status(
        &self,
        id: &str,
        manifest: Option<&toml::Value>,
        spec: Option<&ModelSpec>,
    ) -> Result<CacheStatus> {
        if manifest
            .and_then(|manifest| toml_str(manifest, "cache_status"))
            .is_some_and(|status| status.eq_ignore_ascii_case("partial"))
        {
            return Ok(CacheStatus::Partial);
        }
        if spec.is_some()
            && let Some(manifest) = manifest
            && self.cache_complete_for_manifest(id, manifest)?
        {
            return Ok(CacheStatus::Cached);
        }
        if spec.is_none()
            && manifest
                .and_then(|manifest| toml_str(manifest, "cache_status"))
                .is_some_and(|status| status.eq_ignore_ascii_case("cached"))
            && manifest_artifacts_exist(&self.model_dir(id), manifest)
        {
            return Ok(CacheStatus::Cached);
        }
        Ok(CacheStatus::NotCached)
    }

    fn cache_complete(&self, id: &str) -> Result<bool> {
        let model_dir = self.model_dir(id);
        let manifest_path = model_dir.join("catalog.toml");
        if !manifest_path.is_file() {
            return Ok(false);
        }
        let Some(manifest) = read_toml(&manifest_path)? else {
            return Ok(false);
        };
        self.cache_complete_for_manifest(id, &manifest)
    }

    fn cache_complete_for_manifest(&self, id: &str, manifest: &toml::Value) -> Result<bool> {
        let model_dir = self.model_dir(id);
        if toml_str(manifest, "cache_status")
            .is_some_and(|status| status.eq_ignore_ascii_case("partial"))
        {
            return Ok(false);
        }
        Ok(self.artifact_files_present(id)
            && manifest_artifact_shas_match(&model_dir, manifest)?
            && self.known_model_artifact_valid(id)?)
    }

    fn artifact_files_present(&self, id: &str) -> bool {
        let model_dir = self.model_dir(id);
        ARTIFACT_FILES
            .iter()
            .all(|file| model_dir.join(file).is_file())
    }

    fn known_model_artifact_valid(&self, id: &str) -> Result<bool> {
        let Some(spec) = spec_by_id(id) else {
            return Ok(true);
        };
        let revision = self
            .revision_on_disk(id)
            .unwrap_or_else(|| spec.revision.to_string());
        let source = self.source_override(id)?;
        let validation_source = validation_source_override_for_revision(source.as_ref(), &revision);
        Ok(validate_artifact_dir(
            &spec,
            &self.model_dir(id),
            ArtifactValidationFlavor::OverrideOrLocal,
            &revision,
            validation_source.as_ref(),
            false,
        )
        .is_ok())
    }

    fn missing_files(&self, id: &str) -> Vec<String> {
        let model_dir = self.model_dir(id);
        let mut missing: Vec<String> = ARTIFACT_FILES
            .iter()
            .filter(|file| !model_dir.join(file).is_file())
            .map(|file| (*file).to_string())
            .collect();
        if !model_dir.join("catalog.toml").is_file() {
            missing.push("catalog.toml".to_string());
        }
        missing
    }

    fn validate_existing_cache_before_fetch(&self, id: &str) -> Result<()> {
        let model_dir = self.model_dir(id);
        if !model_dir.exists() {
            return Ok(());
        }
        let has_manifest = model_dir.join("catalog.toml").is_file();
        let has_artifact_payload = ARTIFACT_FILES
            .iter()
            .any(|file| model_dir.join(file).exists());
        let has_cache_payload = has_manifest || has_artifact_payload;
        if !has_cache_payload {
            return Ok(());
        }
        if self.partial_manifest(id)? {
            return Err(CgCatalogError::CacheIncomplete {
                missing_files: self.missing_files(id),
            });
        }
        if !has_artifact_payload {
            return Ok(());
        }
        let missing_files = self.missing_files(id);
        if !missing_files.is_empty() {
            return Err(CgCatalogError::CacheIncomplete { missing_files });
        }
        validate_artifact_identity(id, &model_dir)
    }

    fn partial_manifest(&self, id: &str) -> Result<bool> {
        Ok(read_toml(&self.manifest_path(id))?
            .and_then(|manifest| toml_str(&manifest, "cache_status").map(str::to_string))
            .is_some_and(|status| status.eq_ignore_ascii_case("partial")))
    }

    fn fetch_to_temp_and_publish(&self, spec: ModelSpec) -> Result<()> {
        let revision = self
            .source_override(spec.id)?
            .map(|source| source.revision)
            .unwrap_or_else(|| spec.revision.to_string());
        self.fetch_to_temp_and_publish_with_revision(spec, &revision)
    }

    fn fetch_to_temp_and_publish_with_revision(
        &self,
        spec: ModelSpec,
        revision: &str,
    ) -> Result<()> {
        let temp_dir = self.temp_publish_dir(spec.id);
        clean_dir(&temp_dir)?;
        fs::create_dir_all(&temp_dir).map_err(|err| io_error(&temp_dir, err))?;
        let source = self.source_override(spec.id)?;
        let validation_flavor = if source.is_some() {
            ArtifactValidationFlavor::OverrideOrLocal
        } else {
            ArtifactValidationFlavor::Production
        };
        for file in ARTIFACT_FILES {
            if file == "REVISION" && source.is_none() {
                fs::write(temp_dir.join(file), revision).map_err(|err| io_error(&temp_dir, err))?;
            } else {
                let url = fetch_url(&spec, revision, source.as_ref(), file);
                fetch_url_to_file(spec.id, &url, &temp_dir.join(file), spec.byte_size)?;
            }
        }
        self.validate_temp_artifacts(&spec, &temp_dir, validation_flavor, revision, false)?;
        self.publish_temp_artifacts_with_revision(&spec, &temp_dir, revision)
    }

    fn publish_temp_artifacts_with_revision(
        &self,
        spec: &ModelSpec,
        temp_dir: &Path,
        revision: &str,
    ) -> Result<()> {
        self.publish_temp_artifacts_with_revision_and_source(spec, temp_dir, revision, None)
    }

    fn publish_temp_artifacts_with_revision_and_source(
        &self,
        spec: &ModelSpec,
        temp_dir: &Path,
        revision: &str,
        source: Option<&Source>,
    ) -> Result<()> {
        let model_dir = self.model_dir(spec.id);
        let shas = shas_from_artifact_dir(temp_dir)?;
        fs::create_dir_all(&model_dir).map_err(|err| io_error(&model_dir, err))?;
        for file in [
            "model.safetensors",
            "tokenizer.json",
            "config.json",
            "REVISION",
        ] {
            replace_file(&temp_dir.join(file), &model_dir.join(file))?;
        }
        self.write_manifest(spec, revision, source, &shas)?;
        clean_dir(temp_dir)?;
        Ok(())
    }

    fn write_manifest(
        &self,
        spec: &ModelSpec,
        revision: &str,
        source: Option<&Source>,
        shas: &[(String, String)],
    ) -> Result<()> {
        let manifest_path = self.manifest_path(spec.id);
        let old_acceptance = read_toml(&manifest_path)?
            .map(|manifest| {
                manifest
                    .get("acceptance_records")
                    .and_then(toml::Value::as_array)
                    .cloned()
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        let mut table = toml::Table::new();
        table.insert(
            "model_id".to_string(),
            toml::Value::String(spec.id.to_string()),
        );
        table.insert(
            "kind".to_string(),
            toml::Value::String(kind_name(&spec.kind).to_string()),
        );
        if let Some(dimension) = spec.dimension {
            table.insert(
                "dimension".to_string(),
                toml::Value::Integer(dimension as i64),
            );
        }
        if let Some(modality) = spec.input_modality {
            table.insert(
                "input_modality".to_string(),
                toml::Value::String(modality.to_string()),
            );
        }
        table.insert(
            "license_spdx".to_string(),
            toml::Value::String(spec.license_spdx.to_string()),
        );
        table.insert(
            "license_url".to_string(),
            toml::Value::String(spec.license_url.to_string()),
        );
        if let Some(repo) = spec.source_repo {
            table.insert(
                "source_repo".to_string(),
                toml::Value::String(repo.to_string()),
            );
            table.insert(
                "source_revision".to_string(),
                toml::Value::String(revision.to_string()),
            );
        }
        if let Some(Source::LocalPath { dir }) = source {
            table.insert(
                "source_kind".to_string(),
                toml::Value::String("LocalPath".to_string()),
            );
            table.insert(
                "source_dir".to_string(),
                toml::Value::String(dir.display().to_string()),
            );
        }
        table.insert(
            "byte_size".to_string(),
            toml::Value::Integer(spec.byte_size as i64),
        );
        table.insert(
            "cache_status".to_string(),
            toml::Value::String("cached".to_string()),
        );
        if let Some(url) = spec.manual_install_url {
            table.insert(
                "manual_install_url".to_string(),
                toml::Value::String(url.to_string()),
            );
        }
        if let Some(footnote) = spec.audit_footnote {
            table.insert(
                "audit_footnote".to_string(),
                toml::Value::String(footnote.to_string()),
            );
        }
        table.insert(
            "artifact_filenames".to_string(),
            toml::Value::Array(
                [
                    "model.safetensors",
                    "tokenizer.json",
                    "config.json",
                    "REVISION",
                    "catalog.toml",
                ]
                .into_iter()
                .map(|file| toml::Value::String(file.to_string()))
                .collect(),
            ),
        );
        let mut sha_table = toml::Table::new();
        for (file, sha) in shas {
            sha_table.insert(file.to_string(), toml::Value::String(sha.to_string()));
        }
        table.insert("per_file_sha256".to_string(), toml::Value::Table(sha_table));
        if !old_acceptance.is_empty() {
            table.insert(
                "acceptance_records".to_string(),
                toml::Value::Array(old_acceptance),
            );
        }
        write_toml_value(&manifest_path, &toml::Value::Table(table))
    }

    fn emit_progress(&self, spec: &ModelSpec, output_mode: &OutputMode) -> Result<()> {
        let revision = self
            .source_override(spec.id)?
            .map(|source| source.revision)
            .unwrap_or_else(|| spec.revision.to_string());
        let repo = spec.source_repo.unwrap_or("");
        let target = self.model_dir(spec.id);
        let event = ProgressEvent::FetchStarting {
            model_id: spec.id.to_string(),
            source: Source::HuggingFace {
                repo: repo.to_string(),
                revision,
            },
            expected_bytes: spec.byte_size,
            target_path: target,
        };
        if let Some(observer) = &self.progress_observer {
            observer(&event);
        }
        let ProgressEvent::FetchStarting {
            model_id,
            source,
            expected_bytes,
            target_path,
        } = &event;
        let (source_kind, source_repo, source_revision) = match source {
            Source::HuggingFace { repo, revision } => {
                ("HuggingFace", repo.as_str(), revision.as_str())
            }
            _ => ("unknown", "", ""),
        };
        let payload = json!({
            "model_id": model_id,
            "source": {"kind": source_kind, "repo": source_repo, "revision": source_revision},
            "source.kind": source_kind,
            "source.repo": source_repo,
            "source.revision": source_revision,
            "revision": source_revision,
            "expected_bytes": expected_bytes,
            "target_path": target_path,
        });
        match output_mode {
            OutputMode::Interactive => {
                let line = format!(
                    "cg-catalog fetch starting model_id={} source=https://huggingface.co/{}/resolve/{} expected_bytes={} target_path={}",
                    model_id,
                    source_repo,
                    payload["revision"].as_str().unwrap_or(spec.revision),
                    expected_bytes,
                    target_path.display()
                );
                let progress = indicatif::ProgressBar::new_spinner();
                progress.set_message(line.clone());
                progress.finish_and_clear();
                let mut stderr = io::stderr().lock();
                writeln!(stderr, "{line}").map_err(|err| io_error(Path::new("stderr"), err))?;
                stderr
                    .flush()
                    .map_err(|err| io_error(Path::new("stderr"), err))?;
            }
            OutputMode::Headless => {
                tracing::info!(
                    model_id = %model_id,
                    source.kind = source_kind,
                    source.repo = source_repo,
                    source.revision = source_revision,
                    source_kind = source_kind,
                    source_repo = source_repo,
                    revision = source_revision,
                    expected_bytes = *expected_bytes,
                    target_path = %target_path.display(),
                    "cg_catalog_fetch_starting"
                );
            }
        }
        Ok(())
    }

    fn record_fetch_observation(
        &self,
        store: &Store,
        spec: ModelSpec,
        output_mode: OutputMode,
    ) -> Result<()> {
        let repo = spec.source_repo.unwrap_or("");
        let output_mode = match output_mode {
            OutputMode::Interactive => "interactive",
            OutputMode::Headless => "headless",
        };
        let log_payload = json!({
            "model_id": spec.id,
            "source": {"kind": "HuggingFace", "repo": repo},
            "expected_bytes": spec.byte_size,
        });
        let observation = FetchObservation {
            model_id: spec.id.to_string(),
            observation_type: "model_fetch_completed".to_string(),
            state_delta: json!({
                "model_id": spec.id,
                "source_kind": "HuggingFace",
                "source_repo": repo,
                "expected_bytes": spec.byte_size,
            }),
        };
        let state_delta = object_state_delta(&observation.state_delta);
        let properties = BTreeMap::from([
            ("log_payload".to_string(), log_payload),
            ("output_mode".to_string(), json!(output_mode)),
        ]);
        self.record_store_observation(
            store,
            StoreObservationRequest {
                observation_id: ObservationId::new_v7(),
                evidence_id: EvidenceId::new_v7(),
                entity_id: fixed_entity_id(spec.id),
                observation_type: &observation.observation_type,
                state_delta,
                properties,
            },
        )
        .map_err(observation_write_error)
    }

    fn record_license_deferred(
        &self,
        store: &Store,
        id: &str,
        canonical_message: &str,
    ) -> Result<()> {
        let state_delta = BTreeMap::from([
            ("model_id".to_string(), Some(json!(id))),
            (
                "canonical_message".to_string(),
                Some(json!(canonical_message)),
            ),
        ]);
        self.record_store_observation(
            store,
            StoreObservationRequest {
                observation_id: ObservationId::new_v7(),
                evidence_id: EvidenceId::new_v7(),
                entity_id: fixed_entity_id(id),
                observation_type: "license_gate_deferred",
                state_delta,
                properties: BTreeMap::new(),
            },
        )
        .map_err(observation_write_error)
    }

    fn record_unknown_observation(&self, store: &Store, id: &str) -> Result<()> {
        let state_delta = BTreeMap::from([("unknown_model_id".to_string(), Some(json!(id)))]);
        self.record_store_observation(
            store,
            StoreObservationRequest {
                observation_id: ObservationId::new_v7(),
                evidence_id: EvidenceId::new_v7(),
                entity_id: EntityId::new_v7(),
                observation_type: "model_id_unknown_at_receiver",
                state_delta,
                properties: BTreeMap::new(),
            },
        )
        .map_err(observation_write_error)
    }

    fn validate_temp_artifacts(
        &self,
        spec: &ModelSpec,
        temp_dir: &Path,
        flavor: ArtifactValidationFlavor,
        revision: &str,
        runtime_validation: bool,
    ) -> Result<()> {
        let source = self.source_override(spec.id)?;
        let validation_source = validation_source_override_for_revision(source.as_ref(), revision);
        if let Err(err) = validate_artifact_dir(
            spec,
            temp_dir,
            flavor,
            revision,
            validation_source.as_ref(),
            runtime_validation,
        ) {
            if let CgCatalogError::ArtifactValidationFailed {
                file,
                expected_sha,
                actual_sha,
                ..
            } = &err
            {
                self.record_artifact_validation_failed(spec.id, file, expected_sha, actual_sha);
            }
            clean_dir(temp_dir)?;
            return Err(err);
        }
        Ok(())
    }

    fn record_artifact_validation_failed(
        &self,
        model_id: &str,
        file: &str,
        expected_sha: &str,
        actual_sha: &str,
    ) {
        use tracing::__macro_support::Callsite as _;
        static __CALLSITE: tracing::__macro_support::MacroCallsite = tracing::callsite2! {
            name: "artifact_validation_failed",
            kind: tracing::metadata::Kind::EVENT,
            target: module_path!(),
            level: tracing::Level::ERROR,
            fields: event, model_id, file, expected_sha, actual_sha
        };
        let fields = __CALLSITE.metadata().fields();
        let event = "artifact_validation_failed";
        tracing::Event::dispatch(
            __CALLSITE.metadata(),
            &tracing::valueset_all!(
                fields,
                event = event,
                model_id = model_id,
                file = file,
                expected_sha = expected_sha,
                actual_sha = actual_sha,
            ),
        );
    }

    fn record_update_observation(
        &self,
        store: &Store,
        entity_key: &str,
        observation_type: &str,
        state_delta_json: serde_json::Value,
        properties: BTreeMap<String, serde_json::Value>,
    ) -> Result<UpdateCheckObservation> {
        let existing_entity_id = self.existing_catalog_entity_id(store, entity_key);
        let entity_id = existing_entity_id.unwrap_or_else(|| fixed_entity_id(entity_key));
        let state_delta = state_delta_json
            .as_object()
            .map(|object| {
                object
                    .iter()
                    .map(|(key, value)| (key.clone(), Some(value.clone())))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        self.record_store_observation(
            store,
            StoreObservationRequest {
                observation_id: ObservationId::new_v7(),
                evidence_id: EvidenceId::new_v7(),
                entity_id,
                observation_type,
                state_delta,
                properties,
            },
        )
        .map_err(observation_write_error)?;
        if existing_entity_id.is_none() {
            self.update_auto_created_catalog_entity(store, entity_id, entity_key)?;
        }
        let revision_coupled_fixtures = state_delta_json
            .get("revision_coupled_fixtures")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect();
        Ok(UpdateCheckObservation {
            entity_id: entity_id.to_string(),
            observation_type: observation_type.to_string(),
            state_delta: state_delta_json,
            revision_coupled_fixtures,
        })
    }

    fn update_auto_created_catalog_entity(
        &self,
        store: &Store,
        entity_id: EntityId,
        entity_key: &str,
    ) -> Result<()> {
        let (name, properties, tags) = if let Some(spec) = spec_by_id(entity_key) {
            let properties = BTreeMap::from([
                ("model_id".to_string(), json!(spec.id)),
                ("kind".to_string(), json!(format!("{:?}", spec.kind))),
                (
                    "current_revision".to_string(),
                    json!(
                        self.revision_on_disk(spec.id)
                            .unwrap_or_else(|| spec.revision.to_string())
                    ),
                ),
                (
                    "current_byte_size".to_string(),
                    json!(self.current_model_byte_size(&spec)),
                ),
            ]);
            (
                entity_key.to_string(),
                properties,
                vec!["cg-catalog".to_string(), "model".to_string()],
            )
        } else if entity_key == CG_BINARY_ID {
            (
                entity_key.to_string(),
                BTreeMap::from([
                    ("binary".to_string(), json!("cg")),
                    ("component".to_string(), json!("context-graph")),
                ]),
                vec!["cg-catalog".to_string(), "binary".to_string()],
            )
        } else {
            (
                entity_key.to_string(),
                BTreeMap::new(),
                vec!["cg-catalog".to_string()],
            )
        };
        store
            .update_entity(
                entity_id,
                EntityPatch {
                    name: Some(name),
                    properties: Some(properties),
                    tags: Some(tags),
                },
            )
            .map_err(|err| store_error(store.db_path(), err))?;
        Ok(())
    }

    fn current_model_byte_size(&self, spec: &ModelSpec) -> u64 {
        let model_dir = self.model_dir(spec.id);
        let mut total = 0_u64;
        for file in ARTIFACT_FILES {
            let Ok(metadata) = fs::metadata(model_dir.join(file)) else {
                return spec.byte_size;
            };
            total = total.saturating_add(metadata.len());
        }
        total
    }

    fn record_store_observation(
        &self,
        store: &Store,
        request: StoreObservationRequest<'_>,
    ) -> std::result::Result<(), CgError> {
        let StoreObservationRequest {
            observation_id,
            evidence_id,
            entity_id,
            observation_type,
            state_delta,
            properties,
        } = request;
        store
            .record_observation(RecordObservation {
                id: observation_id,
                entity_id,
                context_id: *CG_CATALOG_CONTEXT_ID,
                observation_type: observation_type.to_string(),
                source: "cg-catalog".to_string(),
                observed_at: Utc::now(),
                evidence: vec![EvidenceRef {
                    id: evidence_id,
                    observation_id,
                    context_id: *CG_CATALOG_CONTEXT_ID,
                    kind: EvidenceKind::Text,
                    source_ref: format!("cas:cg-catalog:{observation_type}"),
                    mime_type: Some("text/plain".to_string()),
                    producer: EvidenceProducer {
                        system: "cg-catalog".to_string(),
                        model_name: "catalog".to_string(),
                        model_version: "catalog-managed".to_string(),
                        pipeline_version: "catalog-observation".to_string(),
                    },
                    text_projection: Some(format!("cg catalog {observation_type}")),
                    retention_status: RetentionStatus::Ephemeral,
                    ..EvidenceRef::default()
                }],
                observed_properties: BTreeMap::new(),
                state_delta,
                properties,
                embeddings: Vec::new(),
            })
            .map(|_| ())
    }

    fn existing_catalog_entity_id(&self, store: &Store, name: &str) -> Option<EntityId> {
        self.existing_catalog_entity(store, name)
            .map(|entity| entity.id)
    }

    fn existing_catalog_entity_revision(&self, store: &Store, name: &str) -> Option<String> {
        self.existing_catalog_entity(store, name)
            .and_then(|entity| {
                entity
                    .properties
                    .get("current_revision")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
    }

    fn existing_catalog_entity(&self, store: &Store, name: &str) -> Option<Entity> {
        store
            .list_entities(ListEntityFilter {
                name_contains: Some(name.to_string()),
                include_archived: true,
                context_id: Some(*CG_CATALOG_CONTEXT_ID),
                ..Default::default()
            })
            .ok()?
            .into_iter()
            .find(|entity| entity.name == name)
    }

    fn record_swap_decision(
        &self,
        store: &Store,
        from: &str,
        to: &str,
        option: Option<&SwapOption>,
        current_dim: u32,
        proposed_dim: u32,
    ) -> Result<()> {
        let context_id = self.ensure_catalog_context(store)?;
        let intention_id = self.create_swap_intention(store, context_id)?;
        let from_entity = self.ensure_model_entity(store, context_id, from)?;
        let to_entity = self.ensure_model_entity(store, context_id, to)?;
        let option_text = option
            .map(swap_option_name)
            .unwrap_or_else(|| "same-dimension".to_string());
        let mut basis_fields = HashMap::new();
        basis_fields.insert(
            from_entity,
            vec![
                "default_text_embedder".to_string(),
                format!("from_model_id={from}"),
                format!("from_dimension={current_dim}"),
            ],
        );
        basis_fields.insert(
            to_entity,
            vec![
                format!("to_model_id={to}"),
                format!("to_dimension={proposed_dim}"),
                format!("option={option_text}"),
            ],
        );
        store
            .create_decision(CreateDecision {
                decision_type: "model-swap".to_string(),
                description: format!(
                    "Set default text embedder from {from} to {to} using {option_text}"
                ),
                reasoning: vec![ReasoningStep {
                    operation: "catalog.set_default".to_string(),
                    details: format!(
                        "from={from}; to={to}; current_dim={current_dim}; proposed_dim={proposed_dim}"
                    ),
                    result: format!("selected option {option_text}"),
                    agent_id: Some("cg-catalog".to_string()),
                    timestamp: Utc::now(),
                }],
                confidence: Some(1.0),
                intention_ids: vec![intention_id],
                based_on_entity_ids: vec![from_entity, to_entity],
                basis_fields: Some(basis_fields),
                based_on_snapshots: Vec::new(),
                tags: vec![
                    "cg-catalog".to_string(),
                    "model-lifecycle".to_string(),
                    format!("from:{from}"),
                    format!("to:{to}"),
                    format!("option:{option_text}"),
                ],
                properties: BTreeMap::new(),
                context_id,
                precedent_ids: Vec::new(),
                agent_id: Some("cg-catalog".to_string()),
            })
            .map_err(|err| store_error(store.db_path(), err))?;
        Ok(())
    }

    fn ensure_model_entity(
        &self,
        store: &Store,
        context_id: ContextId,
        model_id: &str,
    ) -> Result<EntityId> {
        if let Some(entity) = store
            .list_entities(ListEntityFilter {
                name_contains: Some(model_id.to_string()),
                include_archived: true,
                context_id: Some(context_id),
                ..Default::default()
            })
            .map_err(|err| store_error(store.db_path(), err))?
            .into_iter()
            .find(|entity| entity.name == model_id)
        {
            return Ok(entity.id);
        }
        Ok(store
            .create_entity(CreateEntity {
                entity_type: EntityType::Configuration,
                name: model_id.to_string(),
                properties: BTreeMap::new(),
                tags: vec!["cg-catalog".to_string(), "model".to_string()],
                context_id,
            })
            .map_err(|err| store_error(store.db_path(), err))?
            .id)
    }

    fn affected_vectors_for_swap(
        &self,
        store: &Store,
        current_id: &str,
        current_dim: u32,
    ) -> Result<u64> {
        let space_id = text_space_id_for_model(current_id, current_dim);
        store
            .count_vectors_in_space(&space_id)
            .map_err(|err| store_error(store.db_path(), err))
    }

    fn store_has_text_space(&self, store: &Store, space_id: &str, dimension: u32) -> Result<bool> {
        Ok(store
            .list_embedding_spaces()
            .map_err(|err| store_error(store.db_path(), err))?
            .into_iter()
            .any(|space| {
                space.embedding_space_id == space_id
                    && space.dimension == dimension as usize
                    && space.model_family == "text"
                    && space.supported_evidence_kinds.contains(&EvidenceKind::Text)
            }))
    }

    fn run_reindex_for_swap(
        &self,
        store: &Store,
        current_id: &str,
        current_dim: u32,
    ) -> Result<u64> {
        let space_id = text_space_id_for_model(current_id, current_dim);
        let affected_vectors = store
            .count_vectors_in_space(&space_id)
            .map_err(|err| store_error(store.db_path(), err))?;
        store
            .reindex_space(&space_id, ReindexOptions::default())
            .map_err(|err| store_error(store.db_path(), err))?;
        Ok(affected_vectors)
    }

    fn ensure_catalog_context(&self, store: &Store) -> Result<ContextId> {
        if let Some(context) = store
            .list_contexts()
            .map_err(|err| store_error(store.db_path(), err))?
            .into_iter()
            .find(|context| context.name == "cg model lifecycle")
        {
            return Ok(context.id);
        }
        Ok(store
            .create_context(CreateContext {
                name: "cg model lifecycle".to_string(),
                labels: vec!["cg-catalog".to_string()],
                properties: BTreeMap::new(),
            })
            .map_err(|err| store_error(store.db_path(), err))?
            .id)
    }

    fn create_swap_intention(&self, store: &Store, context_id: ContextId) -> Result<IntentionId> {
        Ok(store
            .create_intention(CreateIntention {
                description: "manage model lifecycle".to_string(),
                context_id,
                tags: vec!["cg-catalog".to_string(), "model-swap".to_string()],
                ..CreateIntention::default()
            })
            .map_err(|err| store_error(store.db_path(), err))?
            .id)
    }

    fn acceptance_recorded_for_scope(&self, id: &str) -> Result<bool> {
        Ok(read_toml(&self.manifest_path(id))?
            .map(|manifest| {
                acceptance_records_from_manifest(&manifest)
                    .into_iter()
                    .any(|record| record.scope_key == self.scope_key)
            })
            .unwrap_or(false))
    }

    fn lock_model(&self, id: &str) -> Result<File> {
        let lock_path = self.model_dir(id).join(".lock");
        fs::create_dir_all(lock_path.parent().expect("lock path has parent"))
            .map_err(|err| io_error(&lock_path, err))?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|err| io_error(&lock_path, err))?;
        file.lock_exclusive()
            .map_err(|err| io_error(&lock_path, err))?;
        Ok(file)
    }

    fn cache_dirs(&self) -> Result<Vec<PathBuf>> {
        let root = self.root();
        if !root.exists() {
            return Ok(Vec::new());
        }
        let mut dirs = Vec::new();
        for entry in fs::read_dir(&root).map_err(|err| io_error(&root, err))? {
            let entry = entry.map_err(|err| io_error(&root, err))?;
            if entry
                .file_type()
                .map_err(|err| io_error(&root, err))?
                .is_dir()
            {
                dirs.push(entry.path());
            }
        }
        Ok(dirs)
    }

    fn revision_on_disk(&self, id: &str) -> Option<String> {
        fs::read_to_string(self.model_dir(id).join("REVISION"))
            .ok()
            .map(|revision| revision.trim().to_string())
    }

    fn source_override(&self, id: &str) -> Result<Option<SourceOverride>> {
        source_override_from_config(id, &self.source_override_path())
    }

    fn model_revision_for_runtime(&self, id: &str) -> Result<String> {
        Ok(self
            .revision_on_disk(id)
            .or(self.source_override(id)?.map(|source| source.revision))
            .unwrap_or_else(|| model_revision(id)))
    }

    fn model_dimension_for_id(&self, id: &str) -> Result<u32> {
        if let Some(entry) = self
            .registered_entries()?
            .and_then(|entries| entries.into_iter().find(|entry| entry.id == id))
            && let Some(dimension) = entry.capability.dimension
        {
            return Ok(dimension);
        }
        if let Some(spec) = spec_by_id(id)
            && let Some(dimension) = spec.dimension
        {
            return Ok(dimension);
        }
        read_toml(&self.manifest_path(id))?
            .and_then(|manifest| toml_u64(&manifest, "dimension"))
            .map(|dimension| dimension as u32)
            .ok_or(CgCatalogError::NotFound)
    }

    fn source_override_path(&self) -> PathBuf {
        self.root().join("cg/catalog-sources.toml")
    }

    fn write_registered_entries(&self, entries: &[CatalogEntry]) -> Result<()> {
        let registrations = CatalogRegistrations {
            entries: entries.to_vec(),
        };
        let value = toml::Value::try_from(registrations)
            .map_err(|err| path_error(&self.registration_path(), err.to_string()))?;
        write_toml_value(&self.registration_path(), &value)
    }

    fn registered_entries(&self) -> Result<Option<Vec<CatalogEntry>>> {
        let path = self.registration_path();
        if !path.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path).map_err(|err| io_error(&path, err))?;
        let registrations = toml::from_str::<CatalogRegistrations>(&text)
            .map_err(|err| path_error(&path, err.to_string()))?;
        Ok(Some(registrations.entries))
    }

    fn registration_path(&self) -> PathBuf {
        self.root().join("cg/catalog-registrations.toml")
    }

    fn runtime_config_path(&self) -> PathBuf {
        self.root().join("cg/catalog.toml")
    }

    fn manifest_path(&self, id: &str) -> PathBuf {
        self.model_dir(id).join("catalog.toml")
    }

    fn temp_publish_dir(&self, id: &str) -> PathBuf {
        self.root()
            .join(format!(".{id}.tmp-{}", std::process::id()))
    }

    fn model_dir(&self, id: &str) -> PathBuf {
        self.cache_layout(id).per_model_path
    }

    fn cache_layout(&self, id: &str) -> CacheLayout {
        let root = self.root();
        CacheLayout {
            per_model_path: root.join(id),
            root,
        }
    }

    fn root(&self) -> PathBuf {
        if self.cache_root.as_os_str().is_empty() {
            default_cache_root()
        } else {
            self.cache_root.clone()
        }
    }
}

pub fn canonical_install_message(entry: &CatalogEntry) -> String {
    CanonicalLicenseMessage {
        text: canonical_license_gate_message(entry),
    }
    .text
}

fn canonical_license_gate_message(entry: &CatalogEntry) -> String {
    let model_id = if entry.id.is_empty() {
        GEMMA_ID
    } else {
        entry.id.as_str()
    };
    let spec = spec_by_id(model_id);
    let model_card = match &entry.source {
        Source::HuggingFace { repo, .. } if !repo.is_empty() => {
            format!("https://huggingface.co/{repo}")
        }
        _ => entry
            .manual_install_url
            .clone()
            .or_else(|| {
                spec.as_ref()
                    .and_then(|spec| spec.manual_install_url.map(str::to_string))
            })
            .or_else(|| {
                spec.as_ref().and_then(|spec| {
                    spec.source_repo
                        .map(|repo| format!("https://huggingface.co/{repo}"))
                })
            })
            .unwrap_or_else(|| "unavailable".to_string()),
    };
    let license_url = if entry.license.url.is_empty() {
        spec.as_ref()
            .map(|spec| spec.license_url.to_string())
            .unwrap_or_else(|| "unavailable".to_string())
    } else {
        entry.license.url.clone()
    };
    let mut shas = if let Some(spec) = spec.as_ref() {
        bundled_manifest_shas(spec)
            .into_iter()
            .map(|(file, sha)| (file.to_string(), sha.to_string()))
            .collect::<Vec<_>>()
    } else if entry.per_file_sha256.is_empty() {
        Vec::new()
    } else {
        entry.per_file_sha256.clone()
    };
    shas.sort_by_key(|(file, _)| artifact_order(file));
    let artifact_lines = if shas.is_empty() {
        "- see the model card for required files\n".to_string()
    } else {
        shas.into_iter()
            .map(|(file, sha)| format!("- {file} sha256={sha}\n"))
            .collect::<String>()
    };
    format!(
        "Model id: {model_id}\n\
         Model card: {model_card}\n\
         License: {license_url}\n\
         Target cache path: {{cache_root}}/{model_id}\n\
         Artifacts:\n\
         {artifact_lines}"
    )
}

pub fn validate_artifact_identity(model_id: &str, cache_dir: &Path) -> Result<()> {
    let spec = spec_by_id(model_id).ok_or(CgCatalogError::NotFound)?;
    let revision = fs::read_to_string(cache_dir.join("REVISION"))
        .ok()
        .map(|revision| revision.trim().to_string())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| spec.revision.to_string());
    let source = source_override_from_model_dir(model_id, cache_dir)?;
    let validation_source = validation_source_override_for_revision(source.as_ref(), &revision);
    let manifest = read_toml(&cache_dir.join("catalog.toml"))?;
    let has_published_sha_manifest = manifest
        .as_ref()
        .and_then(|manifest| manifest.get("per_file_sha256"))
        .is_some();
    validate_artifact_dir(
        &spec,
        cache_dir,
        ArtifactValidationFlavor::OverrideOrLocal,
        &revision,
        validation_source.as_ref(),
        spec.id == GEMMA_ID && source.is_some() && !has_published_sha_manifest,
    )
}

fn default_specs() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            id: GEMMA_ID,
            kind: ModelKind::TextEmbedder,
            dimension: Some(768),
            input_modality: Some("text"),
            license_spdx: "Gemma",
            license_url: "https://ai.google.dev/gemma/terms",
            requires_manual_acceptance: true,
            source_repo: Some(GEMMA_REPO),
            revision: GEMMA_REVISION,
            byte_size: GEMMA_BYTES,
            manual_install_url: Some("https://huggingface.co/google/embeddinggemma-300m"),
            audit_footnote: Some(FOOTNOTE_EMBEDDINGGEMMA),
            visible: true,
        },
        ModelSpec {
            id: NOMIC_ID,
            kind: ModelKind::TextEmbedder,
            dimension: Some(768),
            input_modality: Some("text"),
            license_spdx: "Apache-2.0",
            license_url: "https://www.apache.org/licenses/LICENSE-2.0",
            requires_manual_acceptance: false,
            source_repo: Some(NOMIC_REPO),
            revision: NOMIC_REVISION,
            byte_size: NOMIC_BYTES,
            manual_install_url: None,
            audit_footnote: None,
            visible: true,
        },
    ]
}

fn spec_by_id(id: &str) -> Option<ModelSpec> {
    default_specs().into_iter().find(|spec| spec.id == id)
}

fn model_revision(id: &str) -> String {
    spec_by_id(id)
        .map(|spec| spec.revision.to_string())
        .unwrap_or_default()
}

fn revision_coupled_fixtures(id: &str) -> Vec<String> {
    match id {
        NOMIC_ID => vec![format!("NOMIC_BYTE_SIZE={NOMIC_BYTES}")],
        GEMMA_ID => vec![format!("GEMMA_BYTE_SIZE={GEMMA_BYTES}")],
        _ => Vec::new(),
    }
}

fn compare_catalog_entries(a: &CatalogEntry, b: &CatalogEntry) -> std::cmp::Ordering {
    (catalog_sort_bucket(&a.id), a.id.as_str()).cmp(&(catalog_sort_bucket(&b.id), b.id.as_str()))
}

fn catalog_sort_bucket(id: &str) -> u8 {
    if id == NOMIC_ID { 0 } else { 1 }
}

fn expected_model_shas(
    spec: &ModelSpec,
    flavor: ArtifactValidationFlavor,
    source: Option<&SourceOverride>,
) -> Vec<String> {
    expected_artifact_shas(spec, "model.safetensors", flavor, spec.revision, source)
}

fn validate_artifact_dir(
    spec: &ModelSpec,
    dir: &Path,
    flavor: ArtifactValidationFlavor,
    revision: &str,
    source: Option<&SourceOverride>,
    runtime_validation: bool,
) -> Result<()> {
    let mut mismatches = Vec::new();
    for file in ARTIFACT_FILES {
        let path = dir.join(file);
        if !path.is_file() {
            return Err(CgCatalogError::CacheIncomplete {
                missing_files: vec![file.to_string()],
            });
        }
        let actual = sha256_file(&path)?;
        let expected = expected_artifact_shas(spec, file, flavor, revision, source);
        if !expected.iter().any(|sha| sha == &actual) {
            let expected_text = expected.first().cloned().unwrap_or_default();
            mismatches.push(ArtifactValidationMismatch {
                file: file.to_string(),
                expected_sha: expected_text,
                actual_sha: actual,
            });
        }
    }
    if let Some(first) = mismatches.first().cloned() {
        return Err(CgCatalogError::ArtifactValidationFailed {
            file: first.file,
            expected_sha: first.expected_sha,
            actual_sha: first.actual_sha,
            mismatches,
        });
    }
    if runtime_validation && spec.id == GEMMA_ID && source.is_some() {
        validate_embeddinggemma_runtime_artifacts(spec, dir, revision, source)?;
    }
    Ok(())
}

fn validate_embeddinggemma_runtime_artifacts(
    spec: &ModelSpec,
    dir: &Path,
    revision: &str,
    source: Option<&SourceOverride>,
) -> Result<()> {
    let config = EmbedderConfig {
        source: ModelSource::LocalPath(dir.to_path_buf()),
        offline: true,
        cache_dir: dir.parent().map(Path::to_path_buf),
        pinned_revision: Some(revision.to_string()),
        network_policy: "catalog-validation-local".to_string(),
    };
    let embedder = EmbeddingGemmaTextEmbedder::new(config);
    if embedder
        .embed(EmbeddingInput::Text("catalog validation probe".to_string()))
        .is_ok()
    {
        return Ok(());
    }
    let expected_sha = expected_artifact_shas(
        spec,
        "model.safetensors",
        ArtifactValidationFlavor::OverrideOrLocal,
        revision,
        source,
    )
    .into_iter()
    .next()
    .unwrap_or_default();
    let actual_sha = sha256_file(&dir.join("model.safetensors"))?;
    let mismatch = ArtifactValidationMismatch {
        file: "model.safetensors".to_string(),
        expected_sha: expected_sha.clone(),
        actual_sha: actual_sha.clone(),
    };
    Err(CgCatalogError::ArtifactValidationFailed {
        file: mismatch.file.clone(),
        expected_sha,
        actual_sha,
        mismatches: vec![mismatch],
    })
}

fn expected_artifact_shas(
    spec: &ModelSpec,
    file: &str,
    _flavor: ArtifactValidationFlavor,
    revision: &str,
    source: Option<&SourceOverride>,
) -> Vec<String> {
    if let Some(source) = source
        && let Some(sha) = source.sha256.get(file)
    {
        return vec![sha.clone()];
    }
    if file == "REVISION" {
        if spec.id == GEMMA_ID {
            return vec![PRODUCTION_GEMMA_REVISION_SHA.to_string()];
        }
        return vec![
            sha256_bytes(revision.as_bytes()),
            sha256_bytes(format!("{revision}\n").as_bytes()),
        ];
    }
    if spec.id == GEMMA_ID {
        return match file {
            "model.safetensors" => vec![PRODUCTION_GEMMA_MODEL_SAFETENSORS_SHA.to_string()],
            "tokenizer.json" => vec![PRODUCTION_GEMMA_TOKENIZER_JSON_SHA.to_string()],
            "config.json" => vec![PRODUCTION_GEMMA_CONFIG_JSON_SHA.to_string()],
            _ => Vec::new(),
        };
    }
    if spec.id == NOMIC_ID {
        return match file {
            "model.safetensors" => {
                vec![PRODUCTION_NOMIC_MODEL_SAFETENSORS_SHA.to_string()]
            }
            "tokenizer.json" => {
                vec![PRODUCTION_NOMIC_TOKENIZER_JSON_SHA.to_string()]
            }
            "config.json" => vec![PRODUCTION_NOMIC_CONFIG_JSON_SHA.to_string()],
            _ => Vec::new(),
        };
    }
    Vec::new()
}

fn fetch_url(
    spec: &ModelSpec,
    revision: &str,
    source: Option<&SourceOverride>,
    file: &str,
) -> String {
    if let Some(source) = source {
        format!("{}/{}", source.base_url.trim_end_matches('/'), file)
    } else if let Some(repo) = spec.source_repo {
        format!("https://huggingface.co/{repo}/resolve/{revision}/{file}")
    } else {
        String::new()
    }
}

fn fetch_revision_from_source_override(model_id: &str, source: &SourceOverride) -> Result<String> {
    let url = format!("{}/REVISION", source.base_url.trim_end_matches('/'));
    let mut response = ureq::get(&url)
        .call()
        .map_err(|err| CgCatalogError::SourceUnavailable {
            model_id: model_id.to_string(),
            message: err.to_string(),
        })?;
    let mut reader = response
        .body_mut()
        .with_config()
        .limit(NOMIC_BYTES + 1024)
        .reader();
    let mut body = String::new();
    use std::io::Read;
    reader
        .read_to_string(&mut body)
        .map_err(|err| CgCatalogError::SourceUnavailable {
            model_id: model_id.to_string(),
            message: err.to_string(),
        })?;
    let revision = body.trim();
    if revision.is_empty() {
        Err(CgCatalogError::SourceUnavailable {
            model_id: model_id.to_string(),
            message: "empty REVISION response from source".to_string(),
        })
    } else {
        Ok(revision.to_string())
    }
}

fn fetch_latest_huggingface_revision(spec: &ModelSpec) -> Result<String> {
    let Some(repo) = spec.source_repo else {
        return Ok(spec.revision.to_string());
    };
    let url = format!("https://huggingface.co/api/models/{repo}");
    let mut response = ureq::get(&url)
        .call()
        .map_err(|err| CgCatalogError::SourceUnavailable {
            model_id: spec.id.to_string(),
            message: err.to_string(),
        })?;
    let reader = response
        .body_mut()
        .with_config()
        .limit(1024 * 1024)
        .reader();
    let payload: serde_json::Value =
        serde_json::from_reader(reader).map_err(|err| CgCatalogError::SourceUnavailable {
            model_id: spec.id.to_string(),
            message: err.to_string(),
        })?;
    payload
        .get("sha")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| CgCatalogError::SourceUnavailable {
            model_id: spec.id.to_string(),
            message: "missing upstream revision sha".to_string(),
        })
}

fn fetch_url_to_file(
    model_id: &str,
    url: &str,
    destination: &Path,
    expected_byte_budget: u64,
) -> Result<()> {
    let mut response = ureq::get(url)
        .call()
        .map_err(|err| CgCatalogError::SourceUnavailable {
            model_id: model_id.to_string(),
            message: err.to_string(),
        })?;
    let mut reader = response
        .body_mut()
        .with_config()
        .limit(expected_byte_budget.saturating_add(1024))
        .reader();
    let mut out = File::create(destination).map_err(|err| io_error(destination, err))?;
    io::copy(&mut reader, &mut out).map_err(|err| io_error(destination, err))?;
    Ok(())
}

fn artifact_filenames(
    model_dir: &Path,
    manifest: Option<&toml::Value>,
    cache_status: &CacheStatus,
) -> Vec<String> {
    if let Some(files) = manifest
        .and_then(|manifest| manifest.get("artifact_filenames"))
        .and_then(toml::Value::as_array)
    {
        return files
            .iter()
            .filter_map(|file| file.as_str().map(str::to_string))
            .collect();
    }
    if *cache_status != CacheStatus::Cached {
        return Vec::new();
    }
    let mut files = Vec::new();
    for file in [
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "REVISION",
        "catalog.toml",
    ] {
        if model_dir.join(file).exists() {
            files.push(file.to_string());
        }
    }
    files
}

fn manifest_artifacts_exist(model_dir: &Path, manifest: Option<&toml::Value>) -> bool {
    let Some(files) = manifest
        .and_then(|manifest| manifest.get("artifact_filenames"))
        .and_then(toml::Value::as_array)
    else {
        return false;
    };
    files
        .iter()
        .filter_map(toml::Value::as_str)
        .filter(|file| *file != "catalog.toml")
        .all(|file| model_dir.join(file).is_file())
}

fn manifest_artifact_shas_match(model_dir: &Path, manifest: &toml::Value) -> Result<bool> {
    let Some(table) = manifest
        .get("per_file_sha256")
        .and_then(toml::Value::as_table)
    else {
        return Ok(true);
    };
    let mut expected = Vec::new();
    flatten_sha_table("", table, &mut expected);
    let expected = expected.into_iter().collect::<BTreeMap<_, _>>();
    for file in ARTIFACT_FILES {
        let Some(expected_sha) = expected.get(file) else {
            return Ok(false);
        };
        let path = model_dir.join(file);
        if !path.is_file() || sha256_file(&path)? != *expected_sha {
            return Ok(false);
        }
    }
    Ok(true)
}

fn per_file_sha256(
    model_dir: &Path,
    manifest: Option<&toml::Value>,
    spec: Option<&ModelSpec>,
) -> Result<Vec<(String, String)>> {
    if let Some(table) = manifest
        .and_then(|manifest| manifest.get("per_file_sha256"))
        .and_then(toml::Value::as_table)
    {
        let mut out = Vec::new();
        flatten_sha_table("", table, &mut out);
        out.sort_by_key(|(file, _)| artifact_order(file));
        return Ok(out);
    }
    let mut files = if spec.is_some() {
        ARTIFACT_FILES
            .iter()
            .map(|file| (*file).to_string())
            .collect()
    } else {
        fs::read_dir(model_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_file())
            .filter_map(|entry| entry.file_name().to_str().map(str::to_string))
            .filter(|file| file != "catalog.toml" && file != ".lock")
            .collect::<Vec<_>>()
    };
    files.sort_by_key(|file| artifact_order(file));
    let mut out = Vec::new();
    for file in files {
        let path = model_dir.join(&file);
        if path.is_file() {
            out.push((file, sha256_file(&path)?));
        }
    }
    if out.is_empty()
        && let Some(spec) = spec
    {
        out = bundled_manifest_shas(spec)
            .into_iter()
            .map(|(file, sha)| (file.to_string(), sha.to_string()))
            .collect();
    }
    Ok(out)
}

fn bundled_manifest_shas(spec: &ModelSpec) -> Vec<(&'static str, &'static str)> {
    if spec.id == GEMMA_ID {
        return vec![
            ("model.safetensors", PRODUCTION_GEMMA_MODEL_SAFETENSORS_SHA),
            ("tokenizer.json", PRODUCTION_GEMMA_TOKENIZER_JSON_SHA),
            ("config.json", PRODUCTION_GEMMA_CONFIG_JSON_SHA),
            ("REVISION", PRODUCTION_GEMMA_REVISION_SHA),
        ];
    }
    if spec.id == NOMIC_ID {
        return vec![
            ("model.safetensors", PRODUCTION_NOMIC_MODEL_SAFETENSORS_SHA),
            ("tokenizer.json", PRODUCTION_NOMIC_TOKENIZER_JSON_SHA),
            ("config.json", PRODUCTION_NOMIC_CONFIG_JSON_SHA),
            ("REVISION", PRODUCTION_NOMIC_REVISION_SHA),
        ];
    }
    Vec::new()
}

fn source_override_from_model_dir(
    model_id: &str,
    model_dir: &Path,
) -> Result<Option<SourceOverride>> {
    let Some(cache_root) = model_dir.parent() else {
        return Ok(None);
    };
    source_override_from_config(model_id, &cache_root.join("cg/catalog-sources.toml"))
}

fn source_override_from_config(model_id: &str, path: &Path) -> Result<Option<SourceOverride>> {
    let Some(config) = read_toml(path)? else {
        return Ok(None);
    };
    let Some(source) = config
        .get("sources")
        .and_then(|sources| sources.get(model_id))
    else {
        return Ok(None);
    };
    let mut sha256 = Vec::new();
    if let Some(table) = source.get("sha256").and_then(toml::Value::as_table) {
        flatten_sha_table("", table, &mut sha256);
    }
    let base_url = required_source_override_str(model_id, source, "base_url")?;
    let revision = required_source_override_str(model_id, source, "revision")?;
    Ok(Some(SourceOverride {
        base_url: base_url.to_string(),
        revision: revision.to_string(),
        sha256: sha256.into_iter().collect(),
    }))
}

fn required_source_override_str<'a>(
    model_id: &str,
    source: &'a toml::Value,
    key: &str,
) -> Result<&'a str> {
    source
        .get(key)
        .and_then(toml::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| CgCatalogError::SourceUnavailable {
            model_id: model_id.to_string(),
            message: format!("catalog-sources.toml entry missing {key}"),
        })
}

fn validation_source_override_for_revision(
    source: Option<&SourceOverride>,
    revision: &str,
) -> Option<SourceOverride> {
    let mut source = source.cloned()?;
    // A sha256.REVISION pin covers the revision declared with the source override.
    // Update checks validate newly discovered REVISION bytes against that discovered revision.
    if source.revision != revision {
        source.sha256.remove("REVISION");
    }
    Some(source)
}

fn flatten_sha_table(prefix: &str, table: &toml::Table, out: &mut Vec<(String, String)>) {
    for (key, value) in table {
        let name = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        if let Some(sha) = value.as_str() {
            out.push((name, sha.to_string()));
        } else if let Some(child) = value.as_table() {
            flatten_sha_table(&name, child, out);
        }
    }
}

fn object_state_delta(value: &serde_json::Value) -> BTreeMap<String, Option<serde_json::Value>> {
    value
        .as_object()
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), Some(value.clone())))
                .collect()
        })
        .unwrap_or_default()
}

fn artifact_order(file: &str) -> usize {
    match file {
        "model.safetensors" => 0,
        "tokenizer.json" => 1,
        "config.json" => 2,
        "REVISION" => 3,
        "catalog.toml" => 4,
        _ => 10,
    }
}

fn append_acceptance_record(manifest: &mut toml::Value, record: &AcceptanceRecord) {
    let Some(table) = manifest.as_table_mut() else {
        *manifest = empty_toml_table();
        return append_acceptance_record(manifest, record);
    };
    let mut accepted_by = toml::Table::new();
    match &record.accepted_by {
        Principal::Human(value) => {
            accepted_by.insert("Human".to_string(), toml::Value::String(value.clone()));
        }
        Principal::Agent(value) => {
            accepted_by.insert("Agent".to_string(), toml::Value::String(value.clone()));
        }
        Principal::System => {
            accepted_by.insert("System".to_string(), toml::Value::Boolean(true));
        }
    }
    let mut durable = toml::Table::new();
    durable.insert("accepted_by".to_string(), toml::Value::Table(accepted_by));
    durable.insert(
        "accepted_at".to_string(),
        toml::Value::Integer(record.accepted_at.0 as i64),
    );
    durable.insert(
        "license_url_sha".to_string(),
        toml::Value::String(record.license_url_sha.clone()),
    );
    durable.insert(
        "artifact_sha".to_string(),
        toml::Value::String(record.artifact_sha.clone()),
    );
    durable.insert(
        "scope_key".to_string(),
        toml::Value::String(record.scope_key.clone()),
    );
    let value = table
        .entry("acceptance_records".to_string())
        .or_insert_with(|| toml::Value::Array(Vec::new()));
    if let Some(records) = value.as_array_mut() {
        records.push(toml::Value::Table(durable));
        records.sort_by_key(|record| {
            record
                .get("accepted_at")
                .and_then(toml::Value::as_integer)
                .unwrap_or(i64::MAX)
        });
    }
}

fn acceptance_records_from_manifest(manifest: &toml::Value) -> Vec<AcceptanceRecord> {
    manifest
        .get("acceptance_records")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|record| {
            let table = record.as_table()?;
            let accepted_by = table
                .get("accepted_by")
                .map(principal_from_toml)
                .unwrap_or(Principal::System);
            Some(AcceptanceRecord {
                accepted_by,
                accepted_at: TxId(table.get("accepted_at")?.as_integer()? as u64),
                license_url_sha: table.get("license_url_sha")?.as_str()?.to_string(),
                artifact_sha: table.get("artifact_sha")?.as_str()?.to_string(),
                scope_key: table.get("scope_key")?.as_str()?.to_string(),
            })
        })
        .collect()
}

fn principal_from_toml(value: &toml::Value) -> Principal {
    if let Some(table) = value.as_table() {
        if let Some(human) = table.get("Human").and_then(toml::Value::as_str) {
            return Principal::Human(human.to_string());
        }
        if let Some(agent) = table.get("Agent").and_then(toml::Value::as_str) {
            return Principal::Agent(agent.to_string());
        }
        if table.contains_key("System") {
            return Principal::System;
        }
    }
    value
        .as_str()
        .map(Principal::decode)
        .unwrap_or(Principal::System)
}

fn swap_option_name(option: &SwapOption) -> String {
    match option {
        SwapOption::Reindex { .. } => "Reindex",
        SwapOption::ParallelSpace => "ParallelSpace",
    }
    .to_string()
}

fn text_space_id_for_model(model_id: &str, dimension: u32) -> String {
    if model_id == NOMIC_ID && dimension == 768 {
        return "text_default_nomic_v1_5_768".to_string();
    }
    format!(
        "text_default_{}_{}",
        model_id.replace(['-', '.'], "_"),
        dimension
    )
}

fn parallel_text_embedding_space_decl(model_id: &str, dimension: u32) -> EmbeddingSpaceDecl {
    let space_id = text_space_id_for_model(model_id, dimension);
    EmbeddingSpaceDecl {
        embedding_space_id: space_id.clone(),
        dimension: dimension as usize,
        metric: DistanceMetric::Cosine,
        model_family: "text".to_string(),
        model_name: model_id.to_string(),
        model_version: model_revision(model_id),
        supported_evidence_kinds: vec![EvidenceKind::Text, EvidenceKind::StructuredSignal],
        default_for_evidence_kind: None,
        bindings: vec![EmbeddingSpaceBinding {
            embedding_space_id: space_id,
            table_name: "evidence".to_string(),
            vector_column: runtime_text_vector_column(model_id, dimension),
        }],
    }
}

fn runtime_text_vector_column(model_id: &str, dimension: u32) -> String {
    let mut suffix = model_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    while suffix.contains("__") {
        suffix = suffix.replace("__", "_");
    }
    suffix = suffix.trim_matches('_').to_string();
    if suffix.is_empty() {
        suffix = "model".to_string();
    }
    format!("vector_text_{suffix}_{dimension}")
}

fn parse_kind(raw: &str) -> Option<ModelKind> {
    match raw {
        "TextEmbedder" => Some(ModelKind::TextEmbedder),
        "VisionEmbedder" => Some(ModelKind::VisionEmbedder),
        "VisionDetector" => Some(ModelKind::VisionDetector),
        "AudioEmbedder" => Some(ModelKind::AudioEmbedder),
        "LlmClient" => Some(ModelKind::LlmClient),
        _ => None,
    }
}

fn kind_name(kind: &ModelKind) -> &'static str {
    match kind {
        ModelKind::TextEmbedder => "TextEmbedder",
        ModelKind::VisionEmbedder => "VisionEmbedder",
        ModelKind::VisionDetector => "VisionDetector",
        ModelKind::AudioEmbedder => "AudioEmbedder",
        ModelKind::LlmClient => "LlmClient",
    }
}

fn fixed_entity_id(key: &str) -> EntityId {
    let uuid = match key {
        NOMIC_ID => "0192e8a0-cafe-7000-8063-6e6f6d696301",
        GEMMA_ID => "0192e8a0-cafe-7000-8063-67656d6d6101",
        CG_BINARY_ID => "0192e8a0-cafe-7000-8063-62696e617279",
        _ => "0192e8a0-cafe-7000-8063-636174616c01",
    };
    EntityId::from(Uuid::parse_str(uuid).expect("fixed entity UUID parses"))
}

fn read_toml(path: &Path) -> Result<Option<toml::Value>> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(io_error(path, err)),
    };
    toml::from_str(&text)
        .map(Some)
        .map_err(|err| path_error(path, format!("invalid TOML: {err}")))
}

fn write_toml(path: &Path, value: &toml::Value) -> Result<()> {
    write_toml_value(path, value)
}

fn write_toml_value(path: &Path, value: &toml::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| io_error(parent, err))?;
    }
    let text = toml::to_string(value).map_err(|err| path_error(path, err.to_string()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("catalog.toml");
    let temp_path = path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()));
    let mut file = File::create(&temp_path).map_err(|err| io_error(&temp_path, err))?;
    file.write_all(text.as_bytes())
        .map_err(|err| io_error(&temp_path, err))?;
    file.sync_all().map_err(|err| io_error(&temp_path, err))?;
    drop(file);
    fs::rename(&temp_path, path).map_err(|err| {
        let _ = fs::remove_file(&temp_path);
        io_error(path, err)
    })
}

fn empty_toml_table() -> toml::Value {
    toml::Value::Table(toml::Table::new())
}

fn toml_str<'a>(value: &'a toml::Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(toml::Value::as_str)
}

fn toml_u64(value: &toml::Value, key: &str) -> Option<u64> {
    value
        .get(key)
        .and_then(toml::Value::as_integer)
        .and_then(|value| u64::try_from(value).ok())
}

fn set_toml_string(value: &mut toml::Value, key: &str, text: &str) {
    if !value.is_table() {
        *value = empty_toml_table();
    }
    if let Some(table) = value.as_table_mut() {
        table.insert(key.to_string(), toml::Value::String(text.to_string()));
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|err| io_error(path, err))?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn shas_from_artifact_dir(dir: &Path) -> Result<Vec<(String, String)>> {
    let mut shas = Vec::new();
    for file in ARTIFACT_FILES {
        let path = dir.join(file);
        if path.is_file() {
            shas.push((file.to_string(), sha256_file(&path)?));
        }
    }
    Ok(shas)
}

fn sum_regular_file_sizes(root: &Path) -> Option<u64> {
    let mut total = 0;
    for entry in fs::read_dir(root).ok()? {
        let entry = entry.ok()?;
        let metadata = entry.metadata().ok()?;
        if metadata.is_file() {
            total += metadata.len();
        }
    }
    Some(total)
}

fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    match fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(rename_err) => {
            if !destination.exists() {
                return Err(io_error(destination, rename_err));
            }
            let file_name = destination
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("artifact");
            let backup = destination.with_file_name(format!(
                ".{file_name}.replace-backup-{}",
                std::process::id()
            ));
            if backup.exists() {
                fs::remove_file(&backup).map_err(|err| io_error(&backup, err))?;
            }
            fs::rename(destination, &backup).map_err(|err| io_error(&backup, err))?;
            match fs::rename(source, destination) {
                Ok(()) => {
                    let _ = fs::remove_file(&backup);
                    Ok(())
                }
                Err(err) => {
                    let _ = fs::rename(&backup, destination);
                    Err(io_error(destination, err))
                }
            }
        }
    }
}

fn clean_dir(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_dir_all(path).map_err(|err| io_error(path, err))?;
    }
    Ok(())
}

fn io_error(path: &Path, err: io::Error) -> CgCatalogError {
    path_error(path, err.to_string())
}

fn path_error(path: &Path, message: String) -> CgCatalogError {
    CgCatalogError::SourceUnavailable {
        model_id: path.display().to_string(),
        message,
    }
}

fn observation_write_error(err: CgError) -> CgCatalogError {
    CgCatalogError::SourceUnavailable {
        model_id: "cg-catalog-observation".to_string(),
        message: err.to_string(),
    }
}

fn store_error(path: &Path, err: CgError) -> CgCatalogError {
    path_error(path, err.to_string())
}

fn default_cache_root() -> PathBuf {
    if let Some(value) = std::env::var_os(CG_MODEL_CACHE_ROOT_ENV) {
        return PathBuf::from(value);
    }
    if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .map(|root| root.join("cg").join("models"))
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_ROOT_WINDOWS))
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache/cg/models")
    } else {
        PathBuf::from(DEFAULT_CACHE_ROOT_LINUX_MACOS)
    }
}
