//! Local-understanding runtime capability: what the installed `cg` binary can do, which backend
//! actually ran, and why a CUDA-capable binary fell back to CPU.
//!
//! This module owns the closed capability/backend/fallback vocabulary, the [`GpuProbe`] seam (a
//! host-driver probe the binary consults at runtime, never at build/config time), the backend
//! resolver that turns `(config, probe result)` into a [`LocalBackendReport`], and the
//! model-load observation seam ([`LoadedEngineObservation`]) that records the GPU layers actually
//! requested of and offloaded by llama.cpp after a real model load.
//!
//! The load-bearing honesty rule: **active CUDA is a load/run result, never a probe/config
//! result.** A report may say `active_backend = Cuda` only when a loaded engine observed non-zero
//! offloaded layers on the same run. The resolver alone (probe + config) can prepare a CUDA load
//! and request GPU layers, but it must not claim active CUDA without an observation.

use serde::Serialize;

/// The CUDA artifact's product name — single-sourced here so the backend report's
/// `recommended_action` and `cg-catalog`'s doctor-facing install action never drift apart.
pub const CUDA_ARTIFACT_NAME: &str = "cg-linux-x86_64-cuda";

/// What the installed binary artifact is capable of — a build-time fact baked into the artifact,
/// reported honestly by the running binary. A CPU artifact never claims it tried CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryCapability {
    /// A CPU-only artifact (static musl single binary). No accelerator backend is compiled in.
    Cpu,
    /// A CUDA artifact (glibc, static CUDA runtime/cuBLAS, lazy host-driver load).
    Cuda,
}

impl BinaryCapability {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }

    /// The build-time capability of *this* compiled binary: `cuda` when the `cuda` feature is
    /// enabled, `cpu` otherwise. Computed from the compiled feature set, never from a probe.
    #[must_use]
    pub fn compiled() -> Self {
        #[cfg(feature = "cuda")]
        {
            Self::Cuda
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::Cpu
        }
    }
}

/// The operator's requested backend: `auto` (use the accelerator when the binary and host allow),
/// or an explicit `cpu`/`cuda` selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfiguredBackend {
    Auto,
    Cpu,
    Cuda,
}

impl ConfiguredBackend {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

/// The backend that actually ran the local-understanding call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveBackend {
    Cpu,
    Cuda,
}

impl ActiveBackend {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

/// The closed, actionable set of reasons a CUDA-capable binary ran on CPU instead. A generic
/// "unavailable" bucket is deliberately not representable — each reason names a different fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackReason {
    /// No compatible NVIDIA GPU / host driver was found.
    NoGpu,
    /// A GPU was present but CUDA initialization failed (driver/runtime mismatch, etc.).
    CudaInitFailed,
    /// A GPU was present and usable, but it did not have enough VRAM for the model.
    InsufficientVram,
    /// The operator explicitly forced CPU (config or env), overriding a usable GPU.
    OperatorForcedCpu,
}

impl FallbackReason {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoGpu => "no_gpu",
            Self::CudaInitFailed => "cuda_init_failed",
            Self::InsufficientVram => "insufficient_vram",
            Self::OperatorForcedCpu => "operator_forced_cpu",
        }
    }
}

/// The result of consulting the host at runtime for accelerator availability. A CUDA artifact runs
/// this after `main`; a CPU artifact never needs it. Doubles produce each variant for tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuProbeResult {
    /// No usable NVIDIA GPU / host driver present.
    NoGpu,
    /// A usable CUDA device with the given total VRAM in bytes.
    UsableCuda { total_vram_bytes: u64 },
    /// A GPU/driver was present but CUDA initialization failed.
    CudaInitFailed,
    /// A GPU was present but has less VRAM than the model needs.
    InsufficientVram {
        available_bytes: u64,
        required_bytes: u64,
    },
}

/// A runtime host-driver probe. The real CUDA build queries the driver lazily; the CPU build uses
/// a probe that always reports [`GpuProbeResult::NoGpu`]. Tests inject deterministic doubles.
pub trait GpuProbe: Send + Sync {
    fn probe(&self) -> GpuProbeResult;
}

/// A probe that always reports no GPU — the CPU artifact's probe, and the safe default.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoGpuProbe;

impl GpuProbe for NoGpuProbe {
    fn probe(&self) -> GpuProbeResult {
        GpuProbeResult::NoGpu
    }
}

/// The inputs to backend resolution: what the binary can do and what the operator asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalBackendConfig {
    pub binary_capability: BinaryCapability,
    pub configured_backend: ConfiguredBackend,
    /// GPU layers to request when CUDA is prepared (`-1` = offload all). Ignored on the CPU path.
    pub requested_gpu_layers_when_cuda: i32,
}

impl LocalBackendConfig {
    #[must_use]
    pub fn new(binary_capability: BinaryCapability, configured_backend: ConfiguredBackend) -> Self {
        Self {
            binary_capability,
            configured_backend,
            requested_gpu_layers_when_cuda: -1,
        }
    }
}

/// What a real model load observed: the GPU layers requested of `LlamaModelParams` and the layers
/// llama.cpp actually offloaded. `observed_gpu_layers` is read from the loaded model, never
/// derived from the request, the probe, the capability, or the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LoadedEngineObservation {
    pub requested_gpu_layers: i32,
    pub observed_gpu_layers: i32,
}

impl LoadedEngineObservation {
    #[must_use]
    pub fn cpu_only() -> Self {
        Self {
            requested_gpu_layers: 0,
            observed_gpu_layers: 0,
        }
    }

    /// True when llama.cpp actually offloaded work to the GPU on this load.
    #[must_use]
    pub fn offloaded(&self) -> bool {
        self.observed_gpu_layers > 0
    }
}

/// The resolved runtime report for one local-understanding run. `active_backend`, `fallback_reason`,
/// and `observed_gpu_layers` are only trustworthy once a load observation is folded in via
/// [`LocalBackendReport::with_load_observation`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LocalBackendReport {
    pub binary_capability: BinaryCapability,
    pub configured_backend: ConfiguredBackend,
    pub active_backend: ActiveBackend,
    /// GPU layers the resolver would request of the loader when preparing a CUDA load; `None` on
    /// the CPU path.
    pub requested_gpu_layers: Option<i32>,
    /// GPU layers the loaded engine reported offloading; `None` until a load has been observed.
    pub observed_gpu_layers: Option<i32>,
    pub fallback_reason: Option<FallbackReason>,
    /// True when the host has a usable GPU that this particular binary cannot use (a CPU artifact
    /// on an NVIDIA host).
    pub gpu_available_but_unusable_by_this_binary: bool,
    /// Present only when `gpu_available_but_unusable_by_this_binary` is true: the exact
    /// user-facing fix (install the CUDA artifact). Names [`CUDA_ARTIFACT_NAME`] so the action and
    /// the artifact stay in sync. A CPU binary never claims it tried CUDA — this field only ever
    /// recommends installing the other artifact, never a probe/config bookkeeping claim.
    pub recommended_action: Option<String>,
}

impl LocalBackendReport {
    /// Fold a real load observation into a resolver report. This is the ONLY way `active_backend`
    /// becomes `Cuda`: the resolver may have requested GPU layers, but active CUDA requires the
    /// loaded engine to have observed non-zero offload. Zero observed offload downgrades to CPU
    /// with `no_gpu` unless a more specific fallback reason was already set.
    #[must_use]
    pub fn with_load_observation(mut self, observation: LoadedEngineObservation) -> Self {
        self.observed_gpu_layers = Some(observation.observed_gpu_layers);
        if observation.offloaded() && self.requested_gpu_layers.is_some_and(|n| n != 0) {
            self.active_backend = ActiveBackend::Cuda;
            self.fallback_reason = None;
        } else {
            self.active_backend = ActiveBackend::Cpu;
            if self.fallback_reason.is_none() && self.binary_capability == BinaryCapability::Cuda {
                self.fallback_reason = Some(FallbackReason::NoGpu);
            }
        }
        self
    }
}

/// Resolve the backend from the binary capability, the operator's configuration, and a probe.
///
/// SCAFFOLD (deliberately wrong): reports CPU for every input and records zero requested GPU
/// layers, regardless of capability, configuration, or probe result. The real implementation
/// (Step 4) resolves CUDA preparation, the CPU-on-GPU-host mismatch, operator-forced CPU, and the
/// distinct fallback reasons.
#[must_use]
pub fn resolve_backend(config: &LocalBackendConfig, probe: &dyn GpuProbe) -> LocalBackendReport {
    let _ = probe;
    LocalBackendReport {
        binary_capability: config.binary_capability,
        configured_backend: config.configured_backend,
        active_backend: ActiveBackend::Cpu,
        requested_gpu_layers: None,
        observed_gpu_layers: None,
        fallback_reason: None,
        gpu_available_but_unusable_by_this_binary: false,
        recommended_action: None,
    }
}
