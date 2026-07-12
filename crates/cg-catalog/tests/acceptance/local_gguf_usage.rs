//! Local GGUF usage reconnection: local token/time cost must reach the normal `ModelReply.usage`
//! and usage-receipt path, joined with the SAME run's backend report, without leaking into any
//! ontology basis lane. Drives the production catalog GGUF route (`run_local_gguf_understanding`,
//! `capture_local_understanding_usage`) through a path-scoped fake local engine — never a parallel
//! test-only adapter.

use std::sync::Arc;

use cg_catalog::acceleration::{
    self, BinaryCapability, ConfiguredBackend, LocalBackendConfig, NoGpuProbe,
    RegisteredLocalEngine, register_local_engine, registered_local_engine,
};
use cg_catalog::{capture_local_understanding_usage, run_local_gguf_understanding};
use cg_local_llm::{
    GenerationParams, LoadedEngineObservation, LocalEnforcementLevel, LocalLlmError, LocalOutcome,
    LocalOutputContract, LocalReceipt, LocalReply, LocalUnderstandingEngine,
};
use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::{
    Embedder, EmbedderRegistration, EnforcementLevel, ModelCallOutcome, Store, StoreConfig,
};
use serde_json::Value;

/// A store with a real (deterministic, lexical) text embedder registered — capture writes
/// embeddings for the digest/observations it records, so `default_text_embedder: None` (the
/// shared `common::separate_store` default) fails these tests with `TextEmbedderDisabled`.
fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let embedder = Arc::new(LexicalMockEmbedder::default());
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: dir.path().join("context-graph.db"),
            default_text_embedder: None,
            ..StoreConfig::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("open store with lexical test embedder");
    (dir, store)
}

fn fake_harvest_json(marker: &str) -> String {
    // The observation needs a resolvable entity: the capture harness only falls back to an
    // entity-less observation when SOME entity already exists in the harvest (`fallback_entity`),
    // so an entity-less harvest here would silently drop the observation.
    serde_json::json!({
        "summary": format!("fake local harvest {marker}"),
        "goals": [{"key": "G1", "text": format!("goal for {marker}")}],
        "entities": [{
            "key": "E1",
            "name": marker,
            "path": marker,
            "entity_type": "service",
        }],
        "observations": [{
            "key": "O1",
            "text": format!("observation for {marker}"),
            "entity_key": "E1",
            "observation_type": marker,
        }],
    })
    .to_string()
}

struct FakeLocalEngine {
    text: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    wall_clock_ms: u128,
    load_observation: Option<LoadedEngineObservation>,
}

impl FakeLocalEngine {
    fn captured(
        marker: &str,
        prompt_tokens: usize,
        generated_tokens: usize,
        wall_clock_ms: u128,
    ) -> Self {
        Self {
            text: fake_harvest_json(marker),
            prompt_tokens,
            generated_tokens,
            wall_clock_ms,
            load_observation: None,
        }
    }

    fn tokenless(marker: &str) -> Self {
        Self {
            text: fake_harvest_json(marker),
            prompt_tokens: 0,
            generated_tokens: 0,
            wall_clock_ms: 50,
            load_observation: None,
        }
    }
}

impl LocalUnderstandingEngine for FakeLocalEngine {
    fn understand(
        &self,
        _prompt: &str,
        _contract: &LocalOutputContract,
        _params: &GenerationParams,
    ) -> Result<LocalReply, LocalLlmError> {
        Ok(LocalReply {
            text: self.text.clone(),
            receipt: LocalReceipt {
                level: LocalEnforcementLevel::Native,
                outcome: LocalOutcome::Captured,
                prompt_tokens: self.prompt_tokens,
                generated_tokens: self.generated_tokens,
                wall_clock_ms: self.wall_clock_ms,
            },
        })
    }

    fn load_observation(&self) -> Option<LoadedEngineObservation> {
        self.load_observation
    }
}

fn cpu_auto_config() -> LocalBackendConfig {
    LocalBackendConfig::new(BinaryCapability::Cpu, ConfiguredBackend::Auto)
}

fn registration(engine: FakeLocalEngine, model_id: &str) -> RegisteredLocalEngine {
    RegisteredLocalEngine {
        engine: Arc::new(engine),
        model_id: model_id.to_string(),
        backend_config: cpu_auto_config(),
        probe: Arc::new(NoGpuProbe),
    }
}

#[test]
fn local_gguf_reply_maps_local_receipt_into_model_reply_usage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let artifact_path = dir.path().join("fake-1-1.gguf");
    let _guard = register_local_engine(
        &artifact_path,
        registration(
            FakeLocalEngine::captured("marker-1-1", 1234, 56, 789),
            "fake-model-1-1",
        ),
    );

    let run = run_local_gguf_understanding(
        &artifact_path,
        "prompt text for 1.1",
        LocalOutputContract::Unconstrained,
    )
    .unwrap_or_else(|err| panic!("expected fake-engine run to succeed, got error: {err}"));

    let usage = run
        .reply
        .usage
        .clone()
        .unwrap_or_else(|| panic!("expected ModelReply.usage to be Some, got None"));
    assert_eq!(usage.prompt_tokens, 1234);
    assert_eq!(usage.completion_tokens, 56);
    assert_eq!(usage.total_tokens, 1290);

    let receipt = run.reply.model_call_receipt();
    assert_eq!(receipt.reliability_level, EnforcementLevel::Native);
    assert_eq!(receipt.outcome, ModelCallOutcome::Captured);
}

#[test]
fn local_gguf_engine_registration_is_path_scoped_and_restored() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path_a = dir.path().join("fake-1-2-a.gguf");
    let path_b = dir.path().join("fake-1-2-b.gguf");

    let guard_a = register_local_engine(
        &path_a,
        registration(
            FakeLocalEngine::captured("marker-1-2-a", 100, 10, 111),
            "fake-model-1-2-a",
        ),
    );
    let guard_b = register_local_engine(
        &path_b,
        registration(
            FakeLocalEngine::captured("marker-1-2-b", 200, 20, 222),
            "fake-model-1-2-b",
        ),
    );

    let run_a =
        run_local_gguf_understanding(&path_a, "prompt-a", LocalOutputContract::Unconstrained)
            .unwrap_or_else(|err| panic!("expected fake-a run to succeed, got error: {err}"));
    let run_b =
        run_local_gguf_understanding(&path_b, "prompt-b", LocalOutputContract::Unconstrained)
            .unwrap_or_else(|err| panic!("expected fake-b run to succeed, got error: {err}"));

    assert!(run_a.reply.phrasing.contains("marker-1-2-a"));
    assert!(run_b.reply.phrasing.contains("marker-1-2-b"));
    assert_ne!(run_a.reply.phrasing, run_b.reply.phrasing);
    assert_eq!(
        run_a.reply.usage.as_ref().map(|u| u.prompt_tokens),
        Some(100)
    );
    assert_eq!(
        run_b.reply.usage.as_ref().map(|u| u.prompt_tokens),
        Some(200)
    );
    assert_eq!(
        run_a.runtime_report.model_id.as_deref(),
        Some("fake-model-1-2-a")
    );
    assert_eq!(
        run_b.runtime_report.model_id.as_deref(),
        Some("fake-model-1-2-b")
    );

    drop(guard_a);
    drop(guard_b);

    assert!(
        registered_local_engine(&path_a).is_none(),
        "dropping the registration guard must unregister path_a"
    );
    assert!(
        registered_local_engine(&path_b).is_none(),
        "dropping the registration guard must unregister path_b"
    );

    let after_drop = run_local_gguf_understanding(
        &path_a,
        "prompt-a-again",
        LocalOutputContract::Unconstrained,
    );
    assert!(
        after_drop.is_err(),
        "after dropping the registration, the production route must fall through to a real GGUF \
         load against this fixture path and fail — no global fake leak into later runs"
    );
}

#[test]
fn local_gguf_capture_receipt_reaches_usage_json_and_totals() {
    let (_dir, store) = fresh_store();
    let model_dir = tempfile::tempdir().expect("tempdir");
    let artifact_path = model_dir.path().join("fake-1-3.gguf");
    let _guard = register_local_engine(
        &artifact_path,
        registration(
            FakeLocalEngine::captured("marker-1-3", 1234, 56, 789),
            "fake-model-1-3",
        ),
    );

    let usage_json = capture_local_understanding_usage(
        &store,
        "local-gguf-1-3-ctx",
        "transcript for local gguf capture receipt test",
        &artifact_path,
    )
    .unwrap_or_else(|err| panic!("expected capture to succeed, got error: {err}"));

    assert_eq!(
        usage_json
            .pointer("/calls/0/prompt_tokens")
            .and_then(Value::as_u64),
        Some(1234)
    );
    assert_eq!(
        usage_json
            .pointer("/calls/0/completion_tokens")
            .and_then(Value::as_u64),
        Some(56)
    );
    assert_eq!(
        usage_json
            .pointer("/calls/0/total_tokens")
            .and_then(Value::as_u64),
        Some(1290)
    );
    assert_eq!(
        usage_json
            .pointer("/calls/0/elapsed_ms")
            .and_then(Value::as_u64),
        Some(789)
    );
    assert_eq!(
        usage_json.pointer("/total_tokens").and_then(Value::as_u64),
        Some(1290)
    );

    let context_id = store
        .list_contexts()
        .expect("list_contexts")
        .into_iter()
        .find(|context| context.name == "local-gguf-1-3-ctx")
        .map(|context| context.id);
    let observations = store
        .list_observations(context_id)
        .expect("list_observations");
    assert!(
        observations
            .iter()
            .any(|observation| observation.observation_type == "marker-1-3"),
        "the captured ontology must come from the fake engine's harvest JSON, not a synthesized \
         placeholder; observations were: {:?}",
        observations
            .iter()
            .map(|o| &o.observation_type)
            .collect::<Vec<_>>()
    );
}

#[test]
fn local_gguf_run_receipt_carries_usage_and_backend_from_same_run() {
    let (_dir, store) = fresh_store();
    let model_dir = tempfile::tempdir().expect("tempdir");
    let artifact_path = model_dir.path().join("fake-1-4.gguf");
    let _guard = register_local_engine(
        &artifact_path,
        registration(
            FakeLocalEngine::captured("marker-1-4", 1234, 56, 789),
            "fake-model-1-4",
        ),
    );

    let usage_json = capture_local_understanding_usage(
        &store,
        "local-gguf-1-4-ctx",
        "transcript for backend+usage same-run join test",
        &artifact_path,
    )
    .unwrap_or_else(|err| panic!("expected capture to succeed, got error: {err}"));

    let call = usage_json
        .pointer("/calls/0")
        .unwrap_or_else(|| panic!("expected /calls/0 in usage json: {usage_json}"));
    assert_eq!(
        call.pointer("/binary_capability").and_then(Value::as_str),
        Some("cpu")
    );
    assert_eq!(
        call.pointer("/configured_backend").and_then(Value::as_str),
        Some("auto")
    );
    assert_eq!(
        call.pointer("/active_backend").and_then(Value::as_str),
        Some("cpu")
    );
    assert!(
        call.get("gpu_layers").is_some(),
        "gpu_layers key must be present (number or null) on the same call record: {call}"
    );
    assert!(
        call.get("fallback_reason").is_some(),
        "fallback_reason key must be present (null or a closed reason) on the same call record: {call}"
    );
    assert_eq!(
        call.pointer("/prompt_tokens").and_then(Value::as_u64),
        Some(1234),
        "usage and backend must come from the SAME call record: {call}"
    );
    assert_eq!(
        call.pointer("/generated_tokens").and_then(Value::as_u64),
        Some(56)
    );
    assert_eq!(
        call.pointer("/wall_clock_ms").and_then(Value::as_u64),
        Some(789)
    );
}

#[test]
fn local_gguf_usage_tokens_come_from_local_reply_not_synthesized_timing() {
    let (_dir, store) = fresh_store();
    let model_dir = tempfile::tempdir().expect("tempdir");

    let good_path = model_dir.path().join("fake-1-5-good.gguf");
    let _good_guard = register_local_engine(
        &good_path,
        registration(
            FakeLocalEngine::captured("marker-1-5-good", 4321, 65, 987),
            "fake-model-1-5-good",
        ),
    );
    let good_usage = capture_local_understanding_usage(
        &store,
        "local-gguf-1-5-good-ctx",
        "transcript with distinctive local usage",
        &good_path,
    )
    .unwrap_or_else(|err| panic!("expected good-fake capture to succeed, got error: {err}"));
    assert_eq!(
        good_usage
            .pointer("/calls/0/prompt_tokens")
            .and_then(Value::as_u64),
        Some(4321)
    );
    assert_eq!(
        good_usage
            .pointer("/calls/0/generated_tokens")
            .and_then(Value::as_u64),
        Some(65)
    );
    assert_eq!(
        good_usage
            .pointer("/calls/0/wall_clock_ms")
            .and_then(Value::as_u64),
        Some(987)
    );

    let tokenless_path = model_dir.path().join("fake-1-5-tokenless.gguf");
    let _tokenless_guard = register_local_engine(
        &tokenless_path,
        registration(
            FakeLocalEngine::tokenless("marker-1-5-tokenless"),
            "fake-model-1-5-tokenless",
        ),
    );
    let tokenless_usage = capture_local_understanding_usage(
        &store,
        "local-gguf-1-5-tokenless-ctx",
        "transcript with no local token counts",
        &tokenless_path,
    )
    .unwrap_or_else(|err| panic!("expected tokenless-fake capture to succeed, got error: {err}"));
    assert!(
        tokenless_usage.pointer("/calls/0/prompt_tokens").is_none(),
        "a token-less local reply must never report fabricated prompt_tokens, got: {tokenless_usage}"
    );
    let usage_is_null_somewhere = tokenless_usage
        .pointer("/calls/0/usage")
        .map(Value::is_null)
        .unwrap_or(false)
        || tokenless_usage
            .pointer("/total_tokens")
            .map(Value::is_null)
            .unwrap_or(false);
    assert!(
        usage_is_null_somewhere,
        "a token-less local reply must surface as missing/null usage, never a synthesized \
         timing-only record: {tokenless_usage}"
    );
}

#[test]
fn local_runtime_receipt_is_operational_metadata_not_decision_basis() {
    let (dir, store) = fresh_store();
    let model_dir = tempfile::tempdir().expect("tempdir");
    let artifact_path = model_dir.path().join("fake-1-6.gguf");
    let _guard = register_local_engine(
        &artifact_path,
        registration(
            FakeLocalEngine::captured("marker-1-6", 1111, 22, 333),
            "fake-model-1-6",
        ),
    );

    capture_local_understanding_usage(
        &store,
        "local-gguf-1-6-ctx",
        "transcript for operational-lane barrier test",
        &artifact_path,
    )
    .unwrap_or_else(|err| panic!("expected capture to succeed, got error: {err}"));

    // Reopen the store from the same on-disk path so this reads back committed state, not a live
    // handle's cache. `common::store_at` calls `Store::initialize_fresh`, which WIPES an existing
    // db file — a genuine reopen must use `Store::open` instead.
    drop(store);
    let reopened = Store::open(StoreConfig {
        db_path: dir.path().join("context-graph.db"),
        default_text_embedder: None,
        ..StoreConfig::default()
    })
    .expect("reopen store from the same on-disk path");

    let context_id = reopened
        .list_contexts()
        .expect("list_contexts")
        .into_iter()
        .find(|context| context.name == "local-gguf-1-6-ctx")
        .map(|context| context.id);
    let observations = reopened
        .list_observations(context_id)
        .expect("list_observations");

    let usage_observations: Vec<_> = observations
        .iter()
        .filter(|observation| observation.observation_type == "provider_usage")
        .collect();
    assert!(
        !usage_observations.is_empty(),
        "a provider_usage observation must be persisted after a local-GGUF capture"
    );
    for observation in &usage_observations {
        let usage_value = observation
            .properties
            .get("usage")
            .unwrap_or_else(|| panic!("properties.usage must be present on the usage observation"));
        assert_eq!(
            usage_value
                .pointer("/calls/0/prompt_tokens")
                .and_then(Value::as_u64),
            Some(1111),
            "the persisted usage properties must carry THIS run's real local token counts: {usage_value}"
        );
        assert!(
            observation.observed_properties.is_empty(),
            "observed_properties must stay empty for a runtime/usage observation (basis lane must \
             stay clean)"
        );
        assert!(
            observation.state_delta.is_empty(),
            "state_delta must stay empty for a runtime/usage observation (basis lane must stay clean)"
        );
    }

    let forbidden_fields = [
        "binary_capability",
        "active_backend",
        "fallback_reason",
        "prompt_tokens",
        "generated_tokens",
        "wall_clock_ms",
    ];
    for observation in &observations {
        if observation.observation_type == "provider_usage" {
            continue;
        }
        let dump = serde_json::to_string(&observation.properties).unwrap_or_default();
        for field in forbidden_fields {
            assert!(
                !dump.contains(field),
                "product observation {:?} must not carry the runtime/usage field {field}: {dump}",
                observation.observation_type
            );
        }
    }
}

#[test]
fn local_validation_receipt_validator_rejects_missing_usage_or_backend_fields() {
    let receipt = serde_json::json!({
        "artifact_sha256": "deadbeef",
        "binary_capability": "cpu",
        "active_backend": "cpu",
        "usage": {"total_tokens": 15},
    });

    let valid_output = serde_json::json!({
        "usage": {
            "calls": [{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}],
            "total_tokens": 15,
        },
        "binary_capability": "cpu",
        "active_backend": "cpu",
    });
    assert!(
        acceleration::validate_acceleration_receipt(&receipt, &valid_output).is_ok(),
        "a receipt/output pair with both usage and backend fields present must validate"
    );

    let missing_usage_output = serde_json::json!({
        "binary_capability": "cpu",
        "active_backend": "cpu",
    });
    assert!(
        acceleration::validate_acceleration_receipt(&receipt, &missing_usage_output).is_err(),
        "product output missing local usage must be rejected"
    );

    let missing_backend_output = serde_json::json!({
        "usage": {"total_tokens": 15},
    });
    assert!(
        acceleration::validate_acceleration_receipt(&receipt, &missing_backend_output).is_err(),
        "product output missing backend fields must be rejected"
    );

    let missing_both_output = serde_json::json!({});
    assert!(
        acceleration::validate_acceleration_receipt(&receipt, &missing_both_output).is_err(),
        "product output missing both usage and backend fields must be rejected"
    );
}
