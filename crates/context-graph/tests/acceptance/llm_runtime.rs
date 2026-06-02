use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::OsString;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use cg_catalog::{
    DEGRADED_MARKER, ModelConnectionCatalog, ModelKind, NoopLlmClient, PrincipalAwareConfig,
    UnderstandingRuntime,
};
use chrono::Utc;
use context_graph::embedder::mock::MockEmbedder;
use context_graph::{
    CgError, ContextId, CreateContext, CreateDecision, CreateEntity, CreateIntention, DecisionId,
    DeploymentSurface, Embedder, EmbedderConfig, EmbedderRegistration, EntityType, EvidenceId,
    EvidenceKind, EvidenceRef, InvalidationSeverity, InvalidationStatus, LlmClient,
    ModelInvocation, ModelReply, ObservationId, Principal, RecordObservation,
    RecordObservationResult, RetentionStatus, Store, StoreConfig, UnderstandingConsumer,
    UnderstandingOutcome, UnderstandingRequest,
};
use serde_json::json;
use tempfile::TempDir;

fn developer() -> Principal {
    Principal::Human("developer@example.com".to_string())
}

fn expected_marker(consumer: &UnderstandingConsumer, input: &str) -> String {
    match consumer {
        UnderstandingConsumer::ConversationCondensation => format!("condensed:{input}"),
        UnderstandingConsumer::BlueprintNormalization => format!("blueprint:{input}"),
        UnderstandingConsumer::WebhookEntityResolution => format!("entity:{input}"),
        UnderstandingConsumer::SignatureSemanticJudgment => format!("signature-judgment:{input}"),
    }
}

fn request(
    context_id: ContextId,
    consumer: UnderstandingConsumer,
    deployment: DeploymentSurface,
    input: &str,
) -> UnderstandingRequest {
    UnderstandingRequest {
        context_id,
        consumer,
        deployment,
        input: input.to_string(),
        paid: true,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ObservedInvocation {
    consumer: UnderstandingConsumer,
    deployment: DeploymentSurface,
    input: String,
    resolved_connection: Option<String>,
    resolved_credential: Option<String>,
    cache_path: Option<PathBuf>,
    local_artifacts: Vec<PathBuf>,
}

#[derive(Clone, Copy)]
struct HostedRuntimeSpec<'a> {
    context_id: ContextId,
    connection_name: &'a str,
    model_id: &'a str,
    env_var: &'a str,
    secret: &'a str,
    spend_owner: &'a str,
    consent: bool,
}

#[derive(Debug)]
struct RecordingClient {
    observed: Mutex<Vec<ObservedInvocation>>,
    reply: Option<ModelReply>,
    fail: bool,
}

impl RecordingClient {
    fn input_derived() -> Self {
        Self {
            observed: Mutex::new(Vec::new()),
            reply: None,
            fail: false,
        }
    }

    fn fixed(reply: ModelReply) -> Self {
        Self {
            observed: Mutex::new(Vec::new()),
            reply: Some(reply),
            fail: false,
        }
    }

    fn unavailable() -> Self {
        Self {
            observed: Mutex::new(Vec::new()),
            reply: None,
            fail: true,
        }
    }

    fn observed(&self) -> Vec<ObservedInvocation> {
        self.observed.lock().expect("observed lock").clone()
    }

    fn call_count(&self) -> usize {
        self.observed.lock().expect("observed lock").len()
    }
}

impl LlmClient for RecordingClient {
    async fn understand(&self, invocation: &ModelInvocation) -> context_graph::Result<ModelReply> {
        self.understand_native(invocation)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn understand_native(&self, invocation: &ModelInvocation) -> context_graph::Result<ModelReply> {
        self.observed
            .lock()
            .expect("observed lock")
            .push(ObservedInvocation {
                consumer: invocation.request.consumer.clone(),
                deployment: invocation.request.deployment.clone(),
                input: invocation.request.input.clone(),
                resolved_connection: invocation.resolved_connection.clone(),
                resolved_credential: invocation.resolved_credential.clone(),
                cache_path: invocation.cache_path.clone(),
                local_artifacts: invocation.local_artifacts.clone(),
            });
        if self.fail {
            return Err(CgError::Engine("provider unavailable".to_string()));
        }
        if let Some(reply) = &self.reply {
            return Ok(reply.clone());
        }
        Ok(ModelReply {
            marker: expected_marker(&invocation.request.consumer, &invocation.request.input),
            phrasing: format!("phrased:{}", invocation.request.input),
            advisory_flag: false,
            malformed: false,
        })
    }
}

fn hosted_runtime_for_context(
    spec: HostedRuntimeSpec<'_>,
    client: Arc<RecordingClient>,
) -> UnderstandingRuntime<RecordingClient> {
    let principal = developer();
    let mut config = PrincipalAwareConfig::default();
    config.insert_secret(principal.clone(), spec.env_var, spec.secret);
    config.insert_secret(
        principal.clone(),
        "CG_LLM_API_KEY",
        "wrong-feature-sentinel",
    );
    config.insert_secret(
        principal.clone(),
        "CG_DIGEST_LLM_API_KEY",
        "wrong-digest-sentinel",
    );
    let mut catalog = ModelConnectionCatalog::default();
    catalog.connect_env_var(
        spec.connection_name,
        spec.env_var,
        principal.clone(),
        &mut config,
    );
    catalog.register_model(spec.model_id, ModelKind::LlmClient, spec.connection_name);
    catalog.bind_context_model(spec.context_id, spec.model_id);
    if spec.consent {
        catalog.record_consent(spec.connection_name, spec.spend_owner);
    }
    UnderstandingRuntime::new(catalog, principal, config).with_client(spec.model_id, client)
}

fn connected_hosted_runtime(
    consent: bool,
    client: Arc<RecordingClient>,
) -> (UnderstandingRuntime<RecordingClient>, ContextId) {
    let context_id = ContextId::new_v7();
    (
        hosted_runtime_for_context(
            HostedRuntimeSpec {
                context_id,
                connection_name: "hosted-reasoning",
                model_id: "reasoning-default",
                env_var: "HOSTED_MODEL_KEY",
                secret: "hosted-secret",
                spend_owner: "personal quota",
                consent,
            },
            client,
        ),
        context_id,
    )
}

pub(crate) enum FeatureEnvMode {
    Poisoned,
    Removed,
}

pub(crate) struct FeatureEnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl Drop for FeatureEnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.saved.iter().rev() {
            // SAFETY: These acceptance tests serialize mutations of these test-only
            // feature-key variables through FEATURE_ENV_LOCK and restore them on drop.
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }
}

pub(crate) fn scoped_feature_envs(mode: FeatureEnvMode) -> FeatureEnvGuard {
    static FEATURE_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = FEATURE_ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let keys = ["CG_LLM_API_KEY", "CG_DIGEST_LLM_API_KEY"];
    let saved = keys
        .into_iter()
        .map(|key| (key, std::env::var_os(key)))
        .collect::<Vec<_>>();

    for key in keys {
        // SAFETY: See FeatureEnvGuard::drop; this scoped guard restores the process
        // environment and serializes mutations of these sentinel-only variables.
        unsafe {
            match mode {
                FeatureEnvMode::Poisoned => {
                    std::env::set_var(key, format!("wrong-process-sentinel-{key}"));
                }
                FeatureEnvMode::Removed => {
                    std::env::remove_var(key);
                }
            }
        }
    }

    FeatureEnvGuard { _lock: lock, saved }
}

fn local_runtime_for_context(
    context_id: ContextId,
    model_dir: &Path,
    client: Arc<RecordingClient>,
) -> UnderstandingRuntime<RecordingClient> {
    let mut catalog = ModelConnectionCatalog::default();
    catalog.register_local_model("local-reasoning", ModelKind::LlmClient, model_dir);
    catalog.bind_context_model(context_id, "local-reasoning");
    UnderstandingRuntime::new(catalog, developer(), PrincipalAwareConfig::default())
        .with_client("local-reasoning", client)
}

fn completed(
    result: std::result::Result<UnderstandingOutcome, impl Debug>,
    input: &str,
) -> UnderstandingOutcome {
    result.unwrap_or_else(|err| UnderstandingOutcome {
        marker: format!("error:{err:?}"),
        degraded_marker: None,
        phrasing: None,
        model_invoked: false,
        resolved_connection: None,
        cache_path_consulted: None,
        stored_understanding: Some(input.to_string()),
    })
}

fn assert_degraded(input: &str, outcome: UnderstandingOutcome) {
    assert_eq!(
        outcome,
        UnderstandingOutcome {
            marker: format!("declarative:{input}"),
            degraded_marker: Some(DEGRADED_MARKER.to_string()),
            phrasing: None,
            model_invoked: false,
            resolved_connection: None,
            cache_path_consulted: None,
            stored_understanding: None,
        },
        "degraded operation must complete on the exact visible lesser path"
    );
}

fn observable_without_connection(
    outcome: &UnderstandingOutcome,
) -> (
    String,
    Option<String>,
    Option<String>,
    bool,
    Option<PathBuf>,
    Option<String>,
) {
    (
        outcome.marker.clone(),
        outcome.degraded_marker.clone(),
        outcome.phrasing.clone(),
        outcome.model_invoked,
        outcome.cache_path_consulted.clone(),
        outcome.stored_understanding.clone(),
    )
}

fn write_single_file_model(model_dir: &Path) -> PathBuf {
    std::fs::create_dir_all(model_dir).expect("local model dir");
    let artifact = model_dir.join("model.gguf");
    std::fs::write(&artifact, b"single-file-local-llm").expect("local model artifact");
    artifact
}

fn read_log_without_setup<C: LlmClient>(runtime: &UnderstandingRuntime<C>) -> Vec<String> {
    runtime
        .config()
        .read_log(&developer())
        .into_iter()
        .filter(|env| env != "HOSTED_MODEL_KEY")
        .collect()
}

fn invalidation_signature(
    result: &RecordObservationResult,
) -> Vec<(
    DecisionId,
    InvalidationStatus,
    InvalidationSeverity,
    usize,
    bool,
)> {
    result
        .invalidations
        .iter()
        .map(|invalidation| {
            (
                invalidation.affected_decision_id,
                invalidation.status,
                invalidation.severity,
                invalidation.threatened_intention_ids.len(),
                invalidation.intention_now_un_served,
            )
        })
        .collect()
}

type InvalidationContract = (InvalidationStatus, InvalidationSeverity, usize, bool);
type InvalidationContracts = Vec<InvalidationContract>;

fn record_engine_case(
    label: &str,
    state_delta: BTreeMap<String, Option<serde_json::Value>>,
    client: Option<Arc<RecordingClient>>,
) -> (InvalidationContracts, Option<UnderstandingOutcome>) {
    let dir = TempDir::new().expect("engine tempdir");
    let store = Store::open(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: Some(EmbedderConfig::disabled()),
        ..Default::default()
    })
    .expect("open store");
    let context = store
        .create_context(CreateContext {
            name: format!("{label} context"),
            ..Default::default()
        })
        .expect("create context");
    let entity = store
        .create_entity(CreateEntity {
            entity_type: EntityType::File,
            name: format!("{label} file"),
            context_id: context.id,
            properties: BTreeMap::from([("path".to_string(), json!("src/old.rs"))]),
            ..Default::default()
        })
        .expect("create entity");
    let intention = store
        .create_intention(CreateIntention {
            description: format!("{label} intention"),
            context_id: context.id,
            ..Default::default()
        })
        .expect("create intention");
    store
        .create_decision(CreateDecision {
            decision_type: "architecture".to_string(),
            description: format!("{label} decision"),
            confidence: Some(0.9),
            intention_ids: vec![intention.id],
            based_on_entity_ids: vec![entity.id],
            basis_fields: Some(HashMap::from([(
                entity.id,
                vec!["path".to_string(), "owner".to_string()],
            )])),
            context_id: context.id,
            ..Default::default()
        })
        .expect("create decision");

    let outcome = client.map(|client| {
        let mut runtime = hosted_runtime_for_context(
            HostedRuntimeSpec {
                context_id: context.id,
                connection_name: "hosted-reasoning",
                model_id: "reasoning-default",
                env_var: "HOSTED_MODEL_KEY",
                secret: "hosted-secret",
                spend_owner: "personal quota",
                consent: true,
            },
            client,
        );
        completed(
            runtime.understand(request(
                context.id,
                UnderstandingConsumer::SignatureSemanticJudgment,
                DeploymentSurface::Edge,
                label,
            )),
            label,
        )
    });

    let observation_id = ObservationId::new_v7();
    let result = store
        .record_observation(RecordObservation {
            id: observation_id,
            entity_id: entity.id,
            context_id: context.id,
            observation_type: "change".to_string(),
            source: label.to_string(),
            observed_at: Utc::now(),
            evidence: vec![EvidenceRef {
                id: EvidenceId::new_v7(),
                observation_id,
                context_id: context.id,
                kind: EvidenceKind::StructuredSignal,
                source_ref:
                    "cas://sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .to_string(),
                content_hash: Some(
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .to_string(),
                ),
                captured_at: Some(Utc::now()),
                retention_status: RetentionStatus::RetainedExternal,
                text_projection: Some(label.to_string()),
                ..Default::default()
            }],
            observed_properties: BTreeMap::new(),
            state_delta,
            properties: BTreeMap::new(),
            embeddings: Vec::new(),
        })
        .expect("record observation");
    let normalized = invalidation_signature(&result)
        .into_iter()
        .map(|(_, status, severity, threatened_count, now_unserved)| {
            (status, severity, threatened_count, now_unserved)
        })
        .collect();
    (normalized, outcome)
}

// Use case: a connected reasoning implementation should show up in the catalog as callable understanding, not as an inert embedder label.
#[test]
fn llm_client_implementations_register_as_catalog_entries_of_kind_llm_client() {
    let client = Arc::new(RecordingClient::input_derived());
    let context_id = ContextId::new_v7();
    let mut runtime = hosted_runtime_for_context(
        HostedRuntimeSpec {
            context_id,
            connection_name: "hosted-reasoning",
            model_id: "reasoning-default",
            env_var: "HOSTED_MODEL_KEY",
            secret: "hosted-secret",
            spend_owner: "personal quota",
            consent: true,
        },
        client,
    );
    let whitespace = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "   ",
        )),
        "   ",
    );
    assert_eq!(
        whitespace.marker,
        expected_marker(&UnderstandingConsumer::ConversationCondensation, "   "),
        "degenerate whitespace input must still produce a defined input-derived result"
    );
    let empty = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "",
        )),
        "",
    );
    assert_eq!(
        empty.marker,
        expected_marker(&UnderstandingConsumer::ConversationCondensation, ""),
        "empty input must return a defined typed result, not a panic or fabricated fallback"
    );

    let mut catalog = ModelConnectionCatalog::default();
    let registered = catalog.register_model(
        "reasoning-default",
        ModelKind::LlmClient,
        "hosted-reasoning",
    );

    assert_eq!(
        registered.kind,
        ModelKind::LlmClient,
        "reasoning models must register as LlmClient catalog entries"
    );
    assert_eq!(
        registered.connection_name.as_deref(),
        Some("hosted-reasoning")
    );
    assert_eq!(registered.welded_auth_env_var, None);
    assert_eq!(registered.cache_path, None);

    let no_entry_client = Arc::new(RecordingClient::input_derived());
    let mut no_entry_runtime = UnderstandingRuntime::new(
        ModelConnectionCatalog::default(),
        developer(),
        PrincipalAwareConfig::default(),
    )
    .with_client("reasoning-default", no_entry_client.clone());
    let no_entry = completed(
        no_entry_runtime.understand(request(
            ContextId::new_v7(),
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "registered client without catalog entry",
        )),
        "registered client without catalog entry",
    );
    assert_degraded("registered client without catalog entry", no_entry);
    assert_eq!(
        no_entry_client.call_count(),
        0,
        "a direct in-memory client registry must not make an absent catalog entry callable"
    );

    let wrong_kind_client = Arc::new(RecordingClient::input_derived());
    let wrong_kind_context = ContextId::new_v7();
    let mut wrong_kind_catalog = ModelConnectionCatalog::default();
    wrong_kind_catalog.register_model("reasoning-default", ModelKind::TextEmbedder, "embedder");
    wrong_kind_catalog.bind_context_model(wrong_kind_context, "reasoning-default");
    let mut wrong_kind_runtime = UnderstandingRuntime::new(
        wrong_kind_catalog,
        developer(),
        PrincipalAwareConfig::default(),
    )
    .with_client("reasoning-default", wrong_kind_client.clone());
    let wrong_kind = completed(
        wrong_kind_runtime.understand(request(
            wrong_kind_context,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "embedder row with client",
        )),
        "embedder row with client",
    );
    assert_degraded("embedder row with client", wrong_kind);
    assert_eq!(
        wrong_kind_client.call_count(),
        0,
        "a non-llm catalog kind must not become callable just because a client was registered"
    );
}

// Use case: the caller asks for understanding once; catalog routing decides whether the backed model is hosted or local.
#[test]
fn llm_client_summarize_routes_through_catalog_irrespective_of_hosted_vs_local_candle() {
    let hosted_client = Arc::new(RecordingClient::input_derived());
    let (mut hosted_runtime, hosted_context) =
        connected_hosted_runtime(true, hosted_client.clone());
    let input = "rename auth/session.rs to auth/tokens.rs";
    let hosted = completed(
        hosted_runtime.understand(request(
            hosted_context,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            input,
        )),
        input,
    );

    let tempdir = TempDir::new().expect("cache tempdir");
    let model_dir = tempdir.path().join("local-reasoning");
    let artifact = write_single_file_model(&model_dir);
    let local_client = Arc::new(RecordingClient::input_derived());
    let local_context = ContextId::new_v7();
    let mut local_runtime =
        local_runtime_for_context(local_context, &model_dir, local_client.clone());
    let local = completed(
        local_runtime.understand(request(
            local_context,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            input,
        )),
        input,
    );

    assert_eq!(
        hosted.marker,
        expected_marker(&UnderstandingConsumer::ConversationCondensation, input)
    );
    assert_eq!(
        hosted_client.observed()[0].resolved_connection.as_deref(),
        Some("hosted-reasoning"),
        "hosted route must resolve through the named connection"
    );
    assert_eq!(
        local_client.observed()[0].cache_path.as_ref(),
        Some(&model_dir),
        "local route must load through the catalog cache path"
    );
    assert_eq!(
        local_client.observed()[0].local_artifacts,
        vec![artifact],
        "local route must pass the loaded single-file artifact, not just copy the directory name"
    );
    assert_eq!(
        local.marker, hosted.marker,
        "hosted and local routes must expose the same understanding contract"
    );
}

// Use case: removing a local model from the cache should make the runtime visibly degrade instead of pretending it still understood.
#[test]
fn llm_client_local_candle_implementation_loads_via_catalog_cache_path() {
    let tempdir = TempDir::new().expect("cache tempdir");
    let model_dir = tempdir.path().join("local-reasoning");
    let artifact = write_single_file_model(&model_dir);
    let client = Arc::new(RecordingClient::input_derived());
    let context_id = ContextId::new_v7();
    let mut runtime = local_runtime_for_context(context_id, &model_dir, client.clone());

    let available = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "local cache present",
        )),
        "local cache present",
    );
    assert_eq!(client.observed()[0].cache_path.as_ref(), Some(&model_dir));
    assert_eq!(client.observed()[0].local_artifacts, vec![artifact]);
    assert_eq!(
        available.cache_path_consulted.as_ref(),
        Some(&model_dir),
        "successful local understanding should report the catalog cache path it loaded"
    );

    std::fs::remove_dir_all(&model_dir).expect("remove local model dir");
    let missing = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "local cache missing",
        )),
        "local cache missing",
    );
    assert_degraded("local cache missing", missing);
}

// Use case: hosted calls should resolve the key through the scoped connection reference, never through a global feature variable.
#[test]
fn llm_client_hosted_implementation_pulls_credentials_via_principal_aware_config_not_global_env() {
    let _feature_env_guard = scoped_feature_envs(FeatureEnvMode::Poisoned);
    let client = Arc::new(RecordingClient::input_derived());
    let (mut runtime, context_id) = connected_hosted_runtime(true, client.clone());

    let outcome = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::WebhookEntityResolution,
            DeploymentSurface::Edge,
            "payment-svc-prod deploy event",
        )),
        "payment-svc-prod deploy event",
    );

    assert_eq!(
        outcome.marker,
        expected_marker(
            &UnderstandingConsumer::WebhookEntityResolution,
            "payment-svc-prod deploy event"
        )
    );
    assert_eq!(
        client.observed()[0].resolved_credential.as_deref(),
        Some("hosted-secret"),
        "hosted runtime must pass the credential resolved through the named connection"
    );
    assert_eq!(
        read_log_without_setup(&runtime),
        Vec::<String>::new(),
        "hosted runtime must not consult per-feature credential variables"
    );
    drop(_feature_env_guard);

    let _removed_feature_env_guard = scoped_feature_envs(FeatureEnvMode::Removed);
    let absent_global = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::WebhookEntityResolution,
            DeploymentSurface::Edge,
            "payment-svc-prod deploy event with no global feature key",
        )),
        "payment-svc-prod deploy event with no global feature key",
    );
    assert_eq!(
        absent_global.marker,
        expected_marker(
            &UnderstandingConsumer::WebhookEntityResolution,
            "payment-svc-prod deploy event with no global feature key"
        ),
        "hosted runtime must keep using the named connection when feature globals are absent"
    );
    assert_eq!(
        client.observed()[1].resolved_credential.as_deref(),
        Some("hosted-secret"),
        "absent process feature keys must not change connection credential resolution"
    );
}

// Use case: a background paid call must stop at the consent chokepoint until the user has opted in.
#[test]
fn paid_understand_call_is_refused_unless_consent_is_on_record_for_the_connection() {
    let no_consent_client = Arc::new(RecordingClient::input_derived());
    let (mut no_consent_runtime, context_id) =
        connected_hosted_runtime(false, no_consent_client.clone());

    let no_consent = completed(
        no_consent_runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "background condensation",
        )),
        "background condensation",
    );
    assert_degraded("background condensation", no_consent);
    assert_eq!(
        no_consent_client.call_count(),
        0,
        "paid backend must not be invoked before consent is recorded"
    );

    let consented_client = Arc::new(RecordingClient::input_derived());
    let (mut consented_runtime, consented_context) =
        connected_hosted_runtime(true, consented_client.clone());
    let consented = completed(
        consented_runtime.understand(request(
            consented_context,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "after consent",
        )),
        "after consent",
    );
    assert_eq!(
        consented.marker,
        expected_marker(
            &UnderstandingConsumer::ConversationCondensation,
            "after consent"
        )
    );
    assert_eq!(
        consented_client.call_count(),
        1,
        "consented paid backend should be invoked exactly once for one request"
    );
}

// Use case: consent can be revoked between background jobs, so every paid call must check the live connection state again.
#[test]
fn consent_revoked_between_calls_is_rechecked_on_every_paid_call() {
    let client = Arc::new(RecordingClient::input_derived());
    let (mut runtime, context_id) = connected_hosted_runtime(true, client.clone());

    let first = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "first background condensation",
        )),
        "first background condensation",
    );
    assert_eq!(
        first.marker,
        expected_marker(
            &UnderstandingConsumer::ConversationCondensation,
            "first background condensation"
        )
    );

    runtime.catalog_mut().revoke_consent("hosted-reasoning");
    let second = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "second background condensation",
        )),
        "second background condensation",
    );

    assert_degraded("second background condensation", second);
    assert_eq!(
        client.call_count(),
        1,
        "revoked consent must prevent the next paid backend invocation"
    );

    runtime
        .catalog_mut()
        .record_consent("hosted-reasoning", "personal quota");
    let third = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "third background condensation",
        )),
        "third background condensation",
    );
    assert_eq!(
        third.marker,
        expected_marker(
            &UnderstandingConsumer::ConversationCondensation,
            "third background condensation"
        ),
        "restoring consent must re-enable the next paid backend call"
    );
    assert_eq!(
        client.call_count(),
        2,
        "live consent recheck must allow exactly one call after consent is restored"
    );
}

// Use case: a reachable provider can still return bad output; that must degrade visibly instead of storing garbage as understanding.
#[test]
fn connected_model_returning_error_or_malformed_response_degrades_without_fabricating_success() {
    let client = Arc::new(RecordingClient::fixed(ModelReply {
        marker: "garbled-body".to_string(),
        phrasing: "{not-valid-understanding".to_string(),
        advisory_flag: false,
        malformed: true,
    }));
    let (mut runtime, context_id) = connected_hosted_runtime(true, client.clone());
    let outcome = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::BlueprintNormalization,
            DeploymentSurface::Edge,
            "fault isolation proposal",
        )),
        "fault isolation proposal",
    );

    assert_degraded("fault isolation proposal", outcome.clone());
    assert_eq!(
        client.call_count(),
        1,
        "malformed output must degrade after attempting the connected backend"
    );
    assert_eq!(
        outcome.stored_understanding, None,
        "malformed provider output must not be stored as understanding"
    );
}

// Use case: an embedded agent with no model configured should keep working and clearly say it is on the lesser path.
#[test]
fn no_model_connected_degrades_to_lesser_path_and_operation_completes() {
    let mut runtime = UnderstandingRuntime::<NoopLlmClient>::new(
        ModelConnectionCatalog::default(),
        developer(),
        PrincipalAwareConfig::default(),
    );
    let context_id = ContextId::new_v7();
    let outcome = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "no model configured",
        )),
        "no model configured",
    );

    assert_degraded("no model configured", outcome);
}

// Use case: if the provider is unreachable, webhook resolution should complete via declarative mapping rather than returning an operational failure.
#[test]
fn llm_provider_unavailable_falls_back_to_declarative() {
    let client = Arc::new(RecordingClient::unavailable());
    let (mut runtime, context_id) = connected_hosted_runtime(true, client.clone());
    let outcome = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::WebhookEntityResolution,
            DeploymentSurface::Edge,
            "payment-svc-prod deploy event",
        )),
        "payment-svc-prod deploy event",
    );

    assert_degraded("payment-svc-prod deploy event", outcome);
    assert_eq!(
        client.call_count(),
        1,
        "provider-unavailable fallback must happen after exactly one backend attempt"
    );
}

// Use case: from a user's seat, no consent and no connected model are the same visible lesser path and neither spends money.
#[test]
fn no_consent_outcome_is_observably_identical_to_no_model_outcome() {
    let client = Arc::new(RecordingClient::input_derived());
    let (mut no_consent_runtime, context_id) = connected_hosted_runtime(false, client.clone());
    let no_consent = completed(
        no_consent_runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "same fallback",
        )),
        "same fallback",
    );

    let mut no_model_runtime = UnderstandingRuntime::<NoopLlmClient>::new(
        ModelConnectionCatalog::default(),
        developer(),
        PrincipalAwareConfig::default(),
    );
    let no_model = completed(
        no_model_runtime.understand(request(
            context_id,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "same fallback",
        )),
        "same fallback",
    );

    assert_eq!(
        no_consent, no_model,
        "no-consent and no-model outcomes must be indistinguishable to consumers"
    );
    assert_eq!(
        client.call_count(),
        0,
        "no-consent fallback must not invoke the paid backend"
    );
}

// Use case: the model may phrase a warning, but it must not decide whether a stale-path invalidation fires.
#[test]
fn model_understanding_never_gates_whether_an_invalidation_flag_fires() {
    let fire_baseline = record_engine_case(
        "basis-change-baseline",
        BTreeMap::from([("path".to_string(), Some(json!("src/current.rs")))]),
        None,
    )
    .0;
    let silent_baseline = record_engine_case(
        "non-basis-baseline",
        BTreeMap::from([("description".to_string(), Some(json!("comment only")))]),
        None,
    )
    .0;

    let fire_client = Arc::new(RecordingClient::fixed(ModelReply {
        marker: "model-says-ignore".to_string(),
        phrasing: "model-says-ignore: adversarial phrasing landed".to_string(),
        advisory_flag: false,
        malformed: false,
    }));
    let (fire_with_model, fire_outcome) = record_engine_case(
        "basis-change-with-model",
        BTreeMap::from([("path".to_string(), Some(json!("src/current.rs")))]),
        Some(fire_client),
    );

    let silent_client = Arc::new(RecordingClient::fixed(ModelReply {
        marker: "model-says-invalidate".to_string(),
        phrasing: "model-says-invalidate: adversarial phrasing landed".to_string(),
        advisory_flag: true,
        malformed: false,
    }));
    let (silent_with_model, silent_outcome) = record_engine_case(
        "non-basis-with-model",
        BTreeMap::from([("description".to_string(), Some(json!("comment only")))]),
        Some(silent_client),
    );

    assert_eq!(
        fire_with_model, fire_baseline,
        "basis-field invalidations must match the engine-deterministic baseline even when the model says to ignore them"
    );
    assert_eq!(
        silent_with_model, silent_baseline,
        "non-basis observations must remain silent even when the model asks for a flag"
    );
    assert_eq!(
        silent_baseline,
        Vec::<(InvalidationStatus, InvalidationSeverity, usize, bool)>::new()
    );
    assert_eq!(
        fire_outcome
            .expect("fire phrasing outcome")
            .phrasing
            .as_deref(),
        Some("model-says-ignore: adversarial phrasing landed")
    );
    let silent_outcome = silent_outcome.expect("silent phrasing outcome");
    assert_eq!(
        silent_outcome.phrasing.as_deref(),
        Some("model-says-invalidate: adversarial phrasing landed")
    );
    assert_ne!(
        silent_outcome.marker, "model-says-invalidate",
        "a model verdict marker must not be exposed as the operation's flag authority"
    );

    let fire_opposite_client = Arc::new(RecordingClient::fixed(ModelReply {
        marker: "model-says-invalidate".to_string(),
        phrasing: "model-says-invalidate: opposite verdict phrasing landed".to_string(),
        advisory_flag: true,
        malformed: false,
    }));
    let (fire_with_opposite_model, fire_opposite_outcome) = record_engine_case(
        "basis-change-with-opposite-model",
        BTreeMap::from([("path".to_string(), Some(json!("src/current.rs")))]),
        Some(fire_opposite_client),
    );
    let silent_opposite_client = Arc::new(RecordingClient::fixed(ModelReply {
        marker: "model-says-ignore".to_string(),
        phrasing: "model-says-ignore: opposite verdict phrasing landed".to_string(),
        advisory_flag: false,
        malformed: false,
    }));
    let (silent_with_opposite_model, silent_opposite_outcome) = record_engine_case(
        "non-basis-with-opposite-model",
        BTreeMap::from([("description".to_string(), Some(json!("comment only")))]),
        Some(silent_opposite_client),
    );

    assert_eq!(
        fire_with_opposite_model, fire_with_model,
        "opposite model verdicts must produce byte-identical fire outcomes"
    );
    assert_eq!(
        silent_with_opposite_model, silent_with_model,
        "opposite model verdicts must produce byte-identical silent outcomes"
    );
    assert_eq!(
        fire_opposite_outcome
            .expect("opposite fire phrasing outcome")
            .phrasing
            .as_deref(),
        Some("model-says-invalidate: opposite verdict phrasing landed")
    );
    assert_eq!(
        silent_opposite_outcome
            .expect("opposite silent phrasing outcome")
            .phrasing
            .as_deref(),
        Some("model-says-ignore: opposite verdict phrasing landed")
    );
}

// Use case: condensation, blueprint matching, webhook resolution, and signature judgment should all use one connected model rather than feature-specific keys.
#[test]
fn one_connected_model_serves_all_four_understanding_consumers_without_a_per_feature_key() {
    let _feature_env_guard = scoped_feature_envs(FeatureEnvMode::Poisoned);
    let client = Arc::new(RecordingClient::input_derived());
    let (mut runtime, context_id) = connected_hosted_runtime(true, client.clone());
    let consumers = [
        UnderstandingConsumer::ConversationCondensation,
        UnderstandingConsumer::BlueprintNormalization,
        UnderstandingConsumer::WebhookEntityResolution,
        UnderstandingConsumer::SignatureSemanticJudgment,
    ];
    let mut resolved = BTreeSet::new();
    let mut markers = BTreeSet::new();

    for consumer in consumers {
        let input = format!("shared understanding input for {consumer:?}");
        let outcome = completed(
            runtime.understand(request(
                context_id,
                consumer.clone(),
                DeploymentSurface::Edge,
                &input,
            )),
            &input,
        );
        let marker = outcome.marker.clone();
        assert_eq!(
            marker,
            expected_marker(&consumer, &input),
            "each consumer must produce its own input-derived marker"
        );
        resolved.insert(outcome.resolved_connection.clone());
        markers.insert(marker);
    }

    assert_eq!(
        resolved,
        BTreeSet::from([Some("hosted-reasoning".to_string())]),
        "all consumers must resolve the same named connection"
    );
    assert_eq!(
        client.call_count(),
        4,
        "each consumer should route through the shared model invocation seam"
    );
    assert_eq!(
        client
            .observed()
            .into_iter()
            .map(|call| call.resolved_credential)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([Some("hosted-secret".to_string())]),
        "all consumers must use the connection credential, not feature-specific keys"
    );
    assert_eq!(
        read_log_without_setup(&runtime),
        Vec::<String>::new(),
        "understanding consumers must not read per-feature credential variables"
    );
}

// Use case: moving from the laptop to the team server should not change the understanding result, only which connection backs it.
#[test]
fn same_understanding_behavior_on_cg_and_cg_server_only_the_resolved_connection_differs() {
    let edge_client = Arc::new(RecordingClient::input_derived());
    let server_client = Arc::new(RecordingClient::input_derived());
    let edge_context = ContextId::new_v7();
    let server_context = ContextId::new_v7();
    let mut edge_runtime = hosted_runtime_for_context(
        HostedRuntimeSpec {
            context_id: edge_context,
            connection_name: "personal-reasoning",
            model_id: "edge-model",
            env_var: "EDGE_MODEL_KEY",
            secret: "edge-secret",
            spend_owner: "personal quota",
            consent: true,
        },
        edge_client.clone(),
    );
    let mut server_runtime = hosted_runtime_for_context(
        HostedRuntimeSpec {
            context_id: server_context,
            connection_name: "org-reasoning",
            model_id: "server-model",
            env_var: "ORG_MODEL_KEY",
            secret: "org-secret",
            spend_owner: "org quota",
            consent: true,
        },
        server_client.clone(),
    );

    let edge_status = edge_runtime.catalog().status_for_context(edge_context);
    let server_status = server_runtime.catalog().status_for_context(server_context);
    assert_eq!(
        edge_status.spend_owner.as_deref(),
        Some("personal quota"),
        "edge status must carry the personal payer/consent metadata on the connection"
    );
    assert_eq!(
        server_status.spend_owner.as_deref(),
        Some("org quota"),
        "server status must carry the org payer/consent metadata on the connection before routing"
    );
    assert!(edge_status.consent_recorded);
    assert!(server_status.consent_recorded);

    let edge = completed(
        edge_runtime.understand(request(
            edge_context,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Edge,
            "same conversation",
        )),
        "same conversation",
    );
    let server = completed(
        server_runtime.understand(request(
            server_context,
            UnderstandingConsumer::ConversationCondensation,
            DeploymentSurface::Server,
            "same conversation",
        )),
        "same conversation",
    );

    assert_eq!(
        observable_without_connection(&server),
        observable_without_connection(&edge),
        "edge and server must produce the same observable understanding for the same input"
    );
    assert_eq!(
        edge_client.observed()[0].resolved_connection.as_deref(),
        Some("personal-reasoning")
    );
    assert_eq!(
        server_client.observed()[0].resolved_connection.as_deref(),
        Some("org-reasoning")
    );
    assert_eq!(
        edge_client.observed()[0].resolved_credential.as_deref(),
        Some("edge-secret")
    );
    assert_eq!(
        server_client.observed()[0].resolved_credential.as_deref(),
        Some("org-secret")
    );
}

// Use case: optimized builds must preserve the no-spend chokepoint and the line that keeps model judgment out of flag firing.
#[test]
fn consent_chokepoint_and_determinism_boundary_hold_in_release_profile() {
    let release_fire_baseline = record_engine_case(
        "release-basis-change-baseline",
        BTreeMap::from([("path".to_string(), Some(json!("src/release-current.rs")))]),
        None,
    )
    .0;
    let release_ignore_client = Arc::new(RecordingClient::fixed(ModelReply {
        marker: "release-model-says-ignore".to_string(),
        phrasing: "release model says ignore; deterministic flag must still fire".to_string(),
        advisory_flag: false,
        malformed: false,
    }));
    let (release_fire_with_model, release_outcome) = record_engine_case(
        "release-basis-change-with-adversarial-model",
        BTreeMap::from([("path".to_string(), Some(json!("src/release-current.rs")))]),
        Some(release_ignore_client),
    );
    assert_eq!(
        release_fire_baseline.len(),
        1,
        "release determinism baseline must include the engine-fired invalidation"
    );
    assert_eq!(
        release_fire_with_model, release_fire_baseline,
        "release profile must not let a model verdict suppress an engine-fired invalidation"
    );
    assert_eq!(
        release_outcome
            .expect("release adversarial phrasing outcome")
            .phrasing
            .as_deref(),
        Some("release model says ignore; deterministic flag must still fire"),
        "release determinism check must prove the model ran while staying off the flag path"
    );

    let client = Arc::new(RecordingClient::fixed(ModelReply {
        marker: "release-adversarial".to_string(),
        phrasing: "release phrasing".to_string(),
        advisory_flag: true,
        malformed: false,
    }));
    let (mut runtime, context_id) = connected_hosted_runtime(false, client.clone());
    let outcome = completed(
        runtime.understand(request(
            context_id,
            UnderstandingConsumer::SignatureSemanticJudgment,
            DeploymentSurface::Edge,
            "release profile unchanged observation",
        )),
        "release profile unchanged observation",
    );

    assert_eq!(
        client.call_count(),
        0,
        "release profile must not invoke a paid backend without consent"
    );
    assert_degraded("release profile unchanged observation", outcome.clone());
}

// Use case: the existing local embedder path should still open and produce embeddings without a named connection.
#[test]
fn existing_local_embedder_registration_and_load_unaffected_by_connection_substrate() {
    let dir = TempDir::new().expect("tempdir");
    let db_path = dir.path().join("embedder-regression.db");
    let embedder = MockEmbedder::new("regression_text_space", 4);
    let mut space = embedder.embedding_space();
    space.default_for_evidence_kind = Some(EvidenceKind::Text);
    let first = embedder
        .embed(context_graph::EmbeddingInput::Text(
            "existing local embedder".to_string(),
        ))
        .expect("mock embedder output");
    let repeated = embedder
        .embed(context_graph::EmbeddingInput::Text(
            "existing local embedder".to_string(),
        ))
        .expect("repeat mock embedder output");
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: db_path.clone(),
            default_text_embedder: Some(EmbedderConfig::disabled()),
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space,
            embedder: Arc::new(embedder),
        }],
    )
    .expect("store with local embedder registration");
    let context = store
        .create_context(CreateContext {
            name: "embedder regression".to_string(),
            labels: Vec::new(),
            properties: Default::default(),
        })
        .expect("context create");
    drop(store);

    let reopened_embedder = MockEmbedder::new("regression_text_space", 4);
    let mut reopened_space = reopened_embedder.embedding_space();
    reopened_space.default_for_evidence_kind = Some(EvidenceKind::Text);
    let reopened = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: db_path.clone(),
            default_text_embedder: Some(EmbedderConfig::disabled()),
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: reopened_space,
            embedder: Arc::new(reopened_embedder.clone()),
        }],
    )
    .expect("reopen store with local embedder registration");
    let after_reopen = reopened_embedder
        .embed(context_graph::EmbeddingInput::Text(
            "existing local embedder".to_string(),
        ))
        .expect("mock embedder output after reopen");

    assert_eq!(first.embedding_space_id, "regression_text_space");
    assert_eq!(first.dimension, 4);
    assert_eq!(
        repeated.vector, first.vector,
        "local embedder value must be stable across repeated embeds"
    );
    assert_eq!(
        after_reopen.vector, first.vector,
        "local embedder value must be stable after reopening the store"
    );
    assert_eq!(context.name, "embedder regression");
    assert_eq!(reopened.db_path(), db_path.as_path());
    assert_eq!(DEGRADED_MARKER, "running without a connected model");
}
