//! Backend capability and fallback semantics for local-understanding acceleration.
//!
//! The load-bearing contract under test: active CUDA is a load/run result, never a probe/config
//! result. `resolve_backend` alone (binary capability + operator config + host probe) may PREPARE
//! a CUDA load (requesting non-zero GPU layers) but must not claim `active_backend = Cuda` without
//! a folded-in [`LoadedEngineObservation`] that actually offloaded work.

use cg_local_llm::{
    ActiveBackend, BinaryCapability, CUDA_ARTIFACT_NAME, ConfiguredBackend, FallbackReason,
    GpuProbe, GpuProbeResult, LoadedEngineObservation, LocalBackendConfig, resolve_backend,
};

struct FakeProbe(GpuProbeResult);

impl GpuProbe for FakeProbe {
    fn probe(&self) -> GpuProbeResult {
        self.0
    }
}

#[test]
fn cpu_binary_reports_cpu_capability_even_when_probe_sees_nvidia() {
    let config = LocalBackendConfig::new(BinaryCapability::Cpu, ConfiguredBackend::Auto);
    let probe = FakeProbe(GpuProbeResult::UsableCuda {
        total_vram_bytes: 24 * 1024 * 1024 * 1024,
    });

    let report = resolve_backend(&config, &probe);

    assert_eq!(
        report.active_backend,
        ActiveBackend::Cpu,
        "a CPU binary must never report active CUDA"
    );
    assert!(
        report.gpu_available_but_unusable_by_this_binary,
        "a CPU binary on an NVIDIA host must flag the GPU as present-but-unusable"
    );
    assert!(
        report.fallback_reason.is_none(),
        "a CPU binary must not claim it tried CUDA and fell back — it never tried; got {:?}",
        report.fallback_reason
    );
    assert!(
        report.requested_gpu_layers.is_none(),
        "a CPU binary must never request GPU layers"
    );
    let action = report
        .recommended_action
        .as_deref()
        .unwrap_or_else(|| panic!("a CPU binary on a usable-CUDA host must name the fix"));
    assert!(
        action.to_lowercase().contains("cuda"),
        "the recommended action must mention CUDA, got: {action}"
    );
    assert!(
        action.contains(CUDA_ARTIFACT_NAME),
        "the recommended action must name the exact CUDA artifact ({CUDA_ARTIFACT_NAME}), got: {action}"
    );
}

#[test]
fn cuda_binary_auto_uses_cuda_when_probe_and_vram_succeed() {
    let config = LocalBackendConfig::new(BinaryCapability::Cuda, ConfiguredBackend::Auto);
    let probe = FakeProbe(GpuProbeResult::UsableCuda {
        total_vram_bytes: 24 * 1024 * 1024 * 1024,
    });

    let report = resolve_backend(&config, &probe);

    assert_eq!(
        report.requested_gpu_layers,
        Some(config.requested_gpu_layers_when_cuda),
        "a usable CUDA probe must make the resolver prepare a non-zero-layer load"
    );
    assert_ne!(
        report.requested_gpu_layers,
        Some(0),
        "a prepared CUDA load must request non-zero GPU layers"
    );
    assert_eq!(
        report.active_backend,
        ActiveBackend::Cpu,
        "a probe-only report must NOT claim active CUDA before a load observation"
    );
    assert!(
        report.fallback_reason.is_none(),
        "preparing a CUDA load is not itself a fallback"
    );

    // Folding in a real load observation with actual offload is the ONLY way active_backend
    // becomes Cuda.
    let offloaded = report.with_load_observation(LoadedEngineObservation {
        requested_gpu_layers: config.requested_gpu_layers_when_cuda,
        observed_gpu_layers: 32,
    });
    assert_eq!(
        offloaded.active_backend,
        ActiveBackend::Cuda,
        "active CUDA is reachable once the loaded engine reports real offload"
    );
    assert!(offloaded.fallback_reason.is_none());
}

#[test]
fn active_cuda_requires_loaded_engine_to_report_real_offload() {
    let config = LocalBackendConfig::new(BinaryCapability::Cuda, ConfiguredBackend::Auto);
    let probe = FakeProbe(GpuProbeResult::UsableCuda {
        total_vram_bytes: 24 * 1024 * 1024 * 1024,
    });
    let report = resolve_backend(&config, &probe);

    // Zero observed offload: even though the probe succeeded and layers were requested, the
    // loaded engine reports it never actually put anything on the GPU. Active CUDA must be
    // impossible here.
    let zero_offload = report
        .clone()
        .with_load_observation(LoadedEngineObservation {
            requested_gpu_layers: config.requested_gpu_layers_when_cuda,
            observed_gpu_layers: 0,
        });
    assert_eq!(
        zero_offload.active_backend,
        ActiveBackend::Cpu,
        "zero observed GPU layers must never yield active CUDA, regardless of probe/request"
    );
    assert_eq!(zero_offload.observed_gpu_layers, Some(0));
    assert_eq!(
        zero_offload.requested_gpu_layers,
        Some(config.requested_gpu_layers_when_cuda),
        "the request that was made must still be visible on the report even when offload failed"
    );

    // Non-zero observed offload: active CUDA, and both requested and observed layers are present
    // on the same report (the loader-seam join test 2.3 is checking for).
    let real_offload = report.with_load_observation(LoadedEngineObservation {
        requested_gpu_layers: config.requested_gpu_layers_when_cuda,
        observed_gpu_layers: 40,
    });
    assert_eq!(real_offload.active_backend, ActiveBackend::Cuda);
    assert_eq!(
        real_offload.requested_gpu_layers,
        Some(config.requested_gpu_layers_when_cuda)
    );
    assert_eq!(real_offload.observed_gpu_layers, Some(40));
}

#[test]
fn cuda_binary_no_gpu_falls_back_to_cpu_with_no_gpu_reason() {
    let config = LocalBackendConfig::new(BinaryCapability::Cuda, ConfiguredBackend::Auto);
    let probe = FakeProbe(GpuProbeResult::NoGpu);

    let report = resolve_backend(&config, &probe);

    assert_eq!(report.active_backend, ActiveBackend::Cpu);
    assert_eq!(report.fallback_reason, Some(FallbackReason::NoGpu));
    assert!(
        report.requested_gpu_layers.is_none() || report.requested_gpu_layers == Some(0),
        "no usable GPU must never request positive GPU layers, got {:?}",
        report.requested_gpu_layers
    );
}

#[test]
fn cuda_init_failure_and_insufficient_vram_are_distinct_fallback_reasons() {
    let config = LocalBackendConfig::new(BinaryCapability::Cuda, ConfiguredBackend::Auto);

    let init_failed_report = resolve_backend(&config, &FakeProbe(GpuProbeResult::CudaInitFailed));
    assert_eq!(
        init_failed_report.fallback_reason,
        Some(FallbackReason::CudaInitFailed)
    );

    let insufficient_vram_report = resolve_backend(
        &config,
        &FakeProbe(GpuProbeResult::InsufficientVram {
            available_bytes: 2 * 1024 * 1024 * 1024,
            required_bytes: 8 * 1024 * 1024 * 1024,
        }),
    );
    assert_eq!(
        insufficient_vram_report.fallback_reason,
        Some(FallbackReason::InsufficientVram)
    );

    assert_ne!(
        init_failed_report.fallback_reason, insufficient_vram_report.fallback_reason,
        "CUDA init failure and insufficient VRAM must not collapse into the same reason"
    );
}

#[test]
fn operator_forced_cpu_overrides_usable_cuda_and_is_receipted() {
    let config = LocalBackendConfig::new(BinaryCapability::Cuda, ConfiguredBackend::Cpu);
    let probe = FakeProbe(GpuProbeResult::UsableCuda {
        total_vram_bytes: 24 * 1024 * 1024 * 1024,
    });

    let report = resolve_backend(&config, &probe);

    assert_eq!(report.configured_backend, ConfiguredBackend::Cpu);
    assert_eq!(report.active_backend, ActiveBackend::Cpu);
    assert_eq!(
        report.fallback_reason,
        Some(FallbackReason::OperatorForcedCpu)
    );
    assert_eq!(
        report.requested_gpu_layers,
        Some(0),
        "an operator-forced CPU run must request zero GPU layers even with a usable GPU present"
    );
}
