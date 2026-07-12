//! Local-understanding acceleration support that lives BELOW the CLI: the combined runtime report
//! that `cg doctor` and `cg model status` render, the path-scoped fake local-engine seam the
//! catalog GGUF route consults, the shared release-artifact manifest parser/selector shared by the
//! installer and CI, and the acceleration validation-receipt checker.
//!
//! Nothing here computes a CLI answer directly — `cg-cli` renders these types. The wrong scaffold
//! bodies (Step 3) return deliberately wrong typed values so the frozen tests fail on assertions;
//! Step 4 replaces those bodies with real behavior.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub use cg_local_llm::{
    ActiveBackend, BinaryCapability, CUDA_ARTIFACT_NAME, ConfiguredBackend, FallbackReason,
    GpuProbe, GpuProbeResult, LoadedEngineObservation, LocalBackendConfig, LocalBackendReport,
    LocalUnderstandingEngine, NoGpuProbe, resolve_backend,
};

/// The product artifact names. These are the shared vocabulary between the installer, CI, and the
/// doctor "install the CUDA artifact" fix action. No RunPod / pod-specific identifier appears here.
/// [`CUDA_ARTIFACT_NAME`] is single-sourced in `cg-local-llm` (where the backend report's own
/// `recommended_action` names it) and re-exported here.
pub const CPU_ARTIFACT_NAME: &str = "cg-linux-x86_64-cpu";

/// The exact action a CPU binary on an NVIDIA host prints so the user can get GPU acceleration:
/// re-run the installer selecting the CUDA artifact.
#[must_use]
pub fn cuda_install_action() -> String {
    format!(
        "This binary is the CPU artifact and cannot use the detected NVIDIA GPU. Install the CUDA \
         artifact `{CUDA_ARTIFACT_NAME}`: `curl -fsSL https://github.com/context-graph-ai/context-graph/releases/latest/download/install.sh | bash -s -- --capability cuda`"
    )
}

/// Local token/time usage for one local-understanding run, mirrored from the `LocalReceipt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LocalRunUsage {
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub wall_clock_ms: u64,
}

/// The single local-understanding runtime report: what the binary can do, what backend ran, the
/// GPU-layer request/offload, the fallback reason, the model id, and the run's local usage. `cg
/// doctor` and `cg model status` render THIS; they do not compute any of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LocalUnderstandingRuntimeReport {
    /// A stable id so the doctor and status renderers can be proven to render the SAME report.
    pub report_id: String,
    pub binary_capability: BinaryCapability,
    pub configured_backend: ConfiguredBackend,
    pub active_backend: ActiveBackend,
    pub requested_gpu_layers: Option<i32>,
    pub observed_gpu_layers: Option<i32>,
    /// Convenience: the effective GPU layer count (observed when a load happened, else requested).
    pub gpu_layers: Option<i32>,
    pub fallback_reason: Option<FallbackReason>,
    pub gpu_available_but_unusable_by_this_binary: bool,
    /// Present only when the CPU binary sits on an NVIDIA host: the exact CUDA-artifact install fix.
    pub cuda_install_action: Option<String>,
    pub model_id: Option<String>,
    pub usage: Option<LocalRunUsage>,
}

impl LocalUnderstandingRuntimeReport {
    /// Build a report from a resolved backend report, a model id, and an optional run usage.
    #[must_use]
    pub fn from_backend_report(
        report_id: impl Into<String>,
        backend: &LocalBackendReport,
        model_id: Option<String>,
        usage: Option<LocalRunUsage>,
    ) -> Self {
        Self {
            report_id: report_id.into(),
            binary_capability: backend.binary_capability,
            configured_backend: backend.configured_backend,
            active_backend: backend.active_backend,
            requested_gpu_layers: backend.requested_gpu_layers,
            observed_gpu_layers: backend.observed_gpu_layers,
            gpu_layers: backend.observed_gpu_layers.or(backend.requested_gpu_layers),
            fallback_reason: backend.fallback_reason,
            gpu_available_but_unusable_by_this_binary: backend
                .gpu_available_but_unusable_by_this_binary,
            cuda_install_action: backend
                .gpu_available_but_unusable_by_this_binary
                .then(cuda_install_action),
            model_id,
            usage,
        }
    }

    /// The JSON "local runtime block" both `cg doctor --json` and `cg model status --format json`
    /// embed. Operational metadata only — never an ontology basis lane.
    #[must_use]
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| json!({}))
    }
}

/// Resolve the current installed binary's default runtime report with no active model.
///
/// SCAFFOLD (deliberately wrong): returns a fixed CPU report. Step 4 computes it from the compiled
/// binary capability and the host probe.
#[must_use]
pub fn installed_binary_runtime_report(
    report_id: impl Into<String>,
) -> LocalUnderstandingRuntimeReport {
    let backend = resolve_backend(
        &LocalBackendConfig::new(BinaryCapability::Cpu, ConfiguredBackend::Auto),
        &NoGpuProbe,
    );
    LocalUnderstandingRuntimeReport::from_backend_report(report_id, &backend, None, None)
}

// ----------------------------------------------------------------------------------------------
// Path-scoped fake local-engine seam
// ----------------------------------------------------------------------------------------------

/// A local engine registered for one artifact path, plus the runtime facts a report needs. Held by
/// the catalog GGUF route so a test can drive the production path without a real GGUF.
#[derive(Clone)]
pub struct RegisteredLocalEngine {
    pub engine: Arc<dyn LocalUnderstandingEngine>,
    pub model_id: String,
    pub backend_config: LocalBackendConfig,
    pub probe: Arc<dyn GpuProbe>,
}

impl std::fmt::Debug for RegisteredLocalEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredLocalEngine")
            .field("model_id", &self.model_id)
            .field("backend_config", &self.backend_config)
            .finish_non_exhaustive()
    }
}

type EngineRegistry = Mutex<HashMap<PathBuf, RegisteredLocalEngine>>;

fn engine_registry() -> &'static EngineRegistry {
    static REGISTRY: OnceLock<EngineRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A registration guard. Dropping it removes the path-scoped fake engine, restoring the production
/// route for later runs in the same process.
#[must_use = "hold the guard for the lifetime of the fake registration; dropping it unregisters"]
pub struct LocalEngineRegistration {
    path: PathBuf,
}

impl Drop for LocalEngineRegistration {
    fn drop(&mut self) {
        if let Ok(mut registry) = engine_registry().lock() {
            registry.remove(&self.path);
        }
    }
}

/// Register a fake local engine for one artifact path. Path-scoped and restored on guard drop, so
/// two engines for two paths never leak into each other and no global current-engine hook exists.
pub fn register_local_engine(
    path: impl Into<PathBuf>,
    engine: RegisteredLocalEngine,
) -> LocalEngineRegistration {
    let path = path.into();
    if let Ok(mut registry) = engine_registry().lock() {
        registry.insert(path.clone(), engine);
    }
    LocalEngineRegistration { path }
}

/// The registered fake engine for a path, if any. The production GGUF route consults this before
/// loading a real model.
#[must_use]
pub fn registered_local_engine(path: &Path) -> Option<RegisteredLocalEngine> {
    engine_registry().lock().ok()?.get(path).cloned()
}

// ----------------------------------------------------------------------------------------------
// Shared release-artifact manifest
// ----------------------------------------------------------------------------------------------

/// The audit block recorded for one built artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactAudit {
    pub static_single_binary: bool,
    #[serde(default)]
    pub dynamic_needed: Vec<String>,
    pub cuda_runtime_static: bool,
    pub cublas_static: bool,
    /// `"lazy"` for the CUDA artifact, `"not_applicable"` for the CPU artifact.
    pub host_driver_load: String,
    pub gpu_less_startup_smoke: String,
    pub llama_cpp_under_musl_proof_path: String,
}

/// One artifact row in the shared release manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactEntry {
    pub artifact_name: String,
    pub executable_name: String,
    pub target_os: String,
    pub arch: String,
    pub libc: String,
    pub binary_capability: String,
    pub url_or_path: String,
    pub sha256: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub sidecars: Vec<String>,
    pub git_commit: String,
    pub build_command: String,
    pub audit: ArtifactAudit,
}

/// The shared release-artifact manifest. The installer and CI parse the SAME schema through this
/// type; a fixture-only installer schema is not acceptable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub artifacts: Vec<ArtifactEntry>,
}

/// Typed reasons the acceleration parsers/validators reject input, so error paths name the bad
/// field rather than collapsing into a generic failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccelerationError {
    ManifestParse(String),
    NoArtifactForCapability(String),
    MissingField(String),
    WrongTargetTriple(String),
    WrongLibc(String),
    DynamicCudaDependency(String),
    Sidecar(String),
    HashMismatch,
    NotExecutable,
    ProductOutputMismatch(String),
    MissingPrerequisite(String),
    SkippedNoOptIn(String),
}

impl std::fmt::Display for AccelerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ManifestParse(m) => write!(f, "manifest parse failed: {m}"),
            Self::NoArtifactForCapability(c) => write!(f, "no artifact for capability {c}"),
            Self::MissingField(field) => write!(f, "missing receipt field: {field}"),
            Self::WrongTargetTriple(t) => write!(f, "wrong target triple: {t}"),
            Self::WrongLibc(l) => write!(f, "wrong libc: {l}"),
            Self::DynamicCudaDependency(d) => write!(f, "dynamic CUDA dependency: {d}"),
            Self::Sidecar(s) => write!(f, "runtime sidecar present: {s}"),
            Self::HashMismatch => write!(f, "artifact hash mismatch"),
            Self::NotExecutable => write!(f, "artifact is not executable"),
            Self::ProductOutputMismatch(m) => write!(f, "product output mismatch: {m}"),
            Self::MissingPrerequisite(v) => write!(f, "missing validation prerequisite: {v}"),
            Self::SkippedNoOptIn(r) => write!(f, "validation skipped: {r}"),
        }
    }
}

impl std::error::Error for AccelerationError {}

impl ArtifactManifest {
    /// Parse the shared manifest schema from JSON.
    pub fn parse(json_text: &str) -> Result<Self, AccelerationError> {
        serde_json::from_str(json_text)
            .map_err(|err| AccelerationError::ManifestParse(err.to_string()))
    }

    /// The CPU artifact row, if present.
    #[must_use]
    pub fn cpu(&self) -> Option<&ArtifactEntry> {
        self.artifacts
            .iter()
            .find(|entry| entry.binary_capability == "cpu")
    }

    /// The CUDA artifact row, if present.
    #[must_use]
    pub fn cuda(&self) -> Option<&ArtifactEntry> {
        self.artifacts
            .iter()
            .find(|entry| entry.binary_capability == "cuda")
    }

    /// Select CPU or CUDA from the manifest using the host probe: CUDA when a usable CUDA device is
    /// present, CPU otherwise.
    ///
    /// SCAFFOLD (deliberately wrong): always selects the CPU artifact regardless of probe result,
    /// so the CUDA-selection frozen test fails on assertion. Step 4 selects on the probe.
    pub fn select(&self, _probe: &dyn GpuProbe) -> Result<&ArtifactEntry, AccelerationError> {
        self.cpu()
            .ok_or_else(|| AccelerationError::NoArtifactForCapability("cpu".to_string()))
    }
}

// ----------------------------------------------------------------------------------------------
// Acceleration validation-receipt checker
// ----------------------------------------------------------------------------------------------

/// Validate an acceleration validation receipt against the real `cg` output JSON it references.
/// Rejects a receipt that omits artifact provenance, the product-output hash, local usage, or the
/// backend report, or whose backend/usage disagrees with the referenced product output.
///
/// This slice implements the token/backend-missing rejection: the referenced product output must
/// carry a local usage total AND both backend fields. Full provenance/product-output
/// cross-checking (hash comparison, artifact sha, etc.) is exercised by the deferred
/// `ci_acceleration_artifacts` / live-validation test families and is not implemented here.
pub fn validate_acceleration_receipt(
    receipt: &Value,
    cg_output_json: &Value,
) -> Result<(), AccelerationError> {
    let _ = receipt;
    let usage_present = cg_output_json
        .pointer("/usage/total_tokens")
        .and_then(Value::as_u64)
        .or_else(|| cg_output_json.get("total_tokens").and_then(Value::as_u64))
        .is_some();
    if !usage_present {
        return Err(AccelerationError::MissingField(
            "usage.total_tokens".to_string(),
        ));
    }
    if cg_output_json
        .get("binary_capability")
        .and_then(Value::as_str)
        .is_none()
    {
        return Err(AccelerationError::MissingField(
            "binary_capability".to_string(),
        ));
    }
    if cg_output_json
        .get("active_backend")
        .and_then(Value::as_str)
        .is_none()
    {
        return Err(AccelerationError::MissingField(
            "active_backend".to_string(),
        ));
    }
    Ok(())
}

// ----------------------------------------------------------------------------------------------
// Live validation opt-in gate
// ----------------------------------------------------------------------------------------------

/// The required environment variables for a live e4b validation run.
pub const VALIDATION_ENV_VARS: [&str; 7] = [
    "CG_ACCEL_CPU_ARTIFACT",
    "CG_ACCEL_CUDA_ARTIFACT",
    "CG_ACCEL_GGUF_CACHE",
    "CG_ACCEL_GOLD_SESSION_JSONL",
    "CG_ACCEL_GOLD_SESSION_ID",
    "CG_ACCEL_RECEIPT_DIR",
    "CG_ACCEL_CG_BENCH",
];

/// The outcome of the live-validation opt-in gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationGate {
    /// `CG_ACCEL_VALIDATE` is unset — skip before touching any model cache, gold session, or `cg`.
    SkippedNoOptIn(String),
    /// Opted in but a required prerequisite path/var is missing.
    MissingPrerequisite(String),
    /// Fully opted in; live validation may proceed.
    Ready,
}

/// Which lane's prerequisites to check: CPU needs all but the CUDA artifact; CUDA needs all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLane {
    Cpu,
    Cuda,
}

/// Evaluate the opt-in gate from an environment lookup. Absent `CG_ACCEL_VALIDATE=1` returns a skip
/// BEFORE any prerequisite is read; opted-in-but-incomplete returns a typed missing-prerequisite.
///
/// SCAFFOLD (deliberately wrong): always returns `Ready`, so the skip/missing-prerequisite frozen
/// tests fail on assertions. Step 4 checks the opt-in flag first, then each required path.
pub fn validation_gate(
    _lane: ValidationLane,
    _env: &dyn Fn(&str) -> Option<String>,
) -> ValidationGate {
    ValidationGate::Ready
}
