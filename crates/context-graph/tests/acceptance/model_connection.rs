use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Arc, Mutex};
use std::thread;

use cg_catalog::{
    CacheStatus, Capability, Catalog, CatalogEntry, CredentialResolution, DEGRADED_MARKER,
    HostedModelConnectionRequest, License, MigrationRecord, MissingConnectionOutcome,
    ModelConnectionCatalog, ModelKind, PrincipalAwareConfig, Source, UnderstandingRuntime,
    UnderstandingRuntimeError,
};
use context_graph::{
    ContextId, DeploymentSurface, LlmClient, ModelInvocation, ModelReply, Principal, ProviderUsage,
    UnderstandingConsumer, UnderstandingRequest,
};
use tempfile::TempDir;

use crate::llm_runtime::{FeatureEnvMode, scoped_feature_envs};

fn developer() -> Principal {
    Principal::Human("developer@example.com".to_string())
}

fn config_with_secret(principal: &Principal, env_var: &str, secret: &str) -> PrincipalAwareConfig {
    let mut config = PrincipalAwareConfig::default();
    config.insert_secret(principal.clone(), env_var, secret);
    config
}

fn open_catalog(dir: &TempDir) -> ModelConnectionCatalog {
    ModelConnectionCatalog::open(dir.path())
}

fn open_catalog_path(path: &Path) -> ModelConnectionCatalog {
    ModelConnectionCatalog::open(path)
}

fn request(context_id: ContextId, input: &str) -> UnderstandingRequest {
    UnderstandingRequest {
        context_id,
        consumer: UnderstandingConsumer::WebhookEntityResolution,
        deployment: DeploymentSurface::Edge,
        input: input.to_string(),
        paid: true,
    }
}

fn spawn_chat_completion_server(body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback provider");
    let address = listener.local_addr().expect("loopback provider address");
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept provider request");
        let mut reader = BufReader::new(stream.try_clone().expect("clone provider stream"));
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            let bytes = reader.read_line(&mut line).expect("read provider header");
            if bytes == 0 || line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                content_length = value.trim().parse::<usize>().unwrap_or(0);
            }
        }
        if content_length > 0 {
            let mut request_body = vec![0; content_length];
            reader
                .read_exact(&mut request_body)
                .expect("read provider request body");
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream
            .write_all(response.as_bytes())
            .expect("write provider response");
    });
    format!("http://{address}/v1")
}

#[test]
fn hosted_chat_completion_runtime_carries_provider_usage() {
    let dir = TempDir::new().expect("catalog dir");
    let context_id = ContextId::new_v7();
    let base_url = spawn_chat_completion_server(
        r#"{
  "id": "chatcmpl-loopback-usage",
  "object": "chat.completion",
  "created": 1766018000,
  "model": "local-vllm/test-model",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "hosted usage survived"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 37,
    "completion_tokens": 11,
    "total_tokens": 48
  }
}"#,
    );
    let mut catalog = open_catalog(&dir);
    catalog
        .connect_hosted_no_auth_result(HostedModelConnectionRequest {
            connection_name: "loopback-usage".to_string(),
            model_id: "loopback-usage-model".to_string(),
            base_url,
            provider: "local-vllm".to_string(),
            provider_model: "test-model".to_string(),
            consent_owner: Some("test quota".to_string()),
        })
        .expect("connect loopback provider");
    catalog.bind_context_model(context_id, "loopback-usage-model");
    let mut runtime = UnderstandingRuntime::<cg_catalog::NoopLlmClient>::new(
        catalog,
        developer(),
        PrincipalAwareConfig::default(),
    );

    let outcome = runtime
        .understand(request(context_id, "return a usage-bearing reply"))
        .expect("hosted understanding succeeds");

    assert_eq!(outcome.phrasing.as_deref(), Some("hosted usage survived"));
    assert_eq!(
        outcome.usage,
        Some(ProviderUsage {
            prompt_tokens: 37,
            completion_tokens: 11,
            total_tokens: 48
        }),
        "the shipped HTTP chat-completions runtime must carry the provider usage block, not only the test capture helper"
    );
}

#[derive(Debug, Default)]
struct ConnectionRecordingClient {
    observed: Mutex<Vec<(Option<String>, Option<String>)>>,
}

impl ConnectionRecordingClient {
    fn observed(&self) -> Vec<(Option<String>, Option<String>)> {
        self.observed.lock().expect("observed lock").clone()
    }
}

impl LlmClient for ConnectionRecordingClient {
    async fn understand(&self, invocation: &ModelInvocation) -> context_graph::Result<ModelReply> {
        self.understand_native(invocation)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn understand_native(&self, invocation: &ModelInvocation) -> context_graph::Result<ModelReply> {
        self.observed.lock().expect("observed lock").push((
            invocation.resolved_connection.clone(),
            invocation.resolved_credential.clone(),
        ));
        Ok(ModelReply {
            marker: format!("understood:{}", invocation.request.input),
            phrasing: format!("phrased:{}", invocation.request.input),
            advisory_flag: false,
            malformed: false,
            usage: None,
        })
    }
}

fn catalog_disk_contains_path(root: &Path, needle: &str) -> bool {
    fn walk(path: &Path, needle: &[u8]) -> bool {
        let Ok(metadata) = std::fs::metadata(path) else {
            return false;
        };
        if metadata.is_file() {
            return std::fs::read(path)
                .map(|bytes| bytes.windows(needle.len()).any(|window| window == needle))
                .unwrap_or(false);
        }
        if !metadata.is_dir() {
            return false;
        }
        std::fs::read_dir(path)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .any(|entry| walk(&entry.path(), needle))
            })
            .unwrap_or(false)
    }

    walk(root, needle.as_bytes())
}

fn catalog_disk_contains(dir: &TempDir, needle: &str) -> bool {
    catalog_disk_contains_path(dir.path(), needle)
}

fn model_connection_child_mode(expected: &str) -> bool {
    std::env::var("CG_MODEL_CONNECTION_DURABILITY_CHILD").as_deref() == Ok(expected)
}

fn child_catalog_root() -> PathBuf {
    PathBuf::from(
        std::env::var("CG_MODEL_CONNECTION_DURABILITY_ROOT").expect("durability child root env"),
    )
}

fn context_id_env(name: &str) -> ContextId {
    ContextId::from(
        uuid::Uuid::parse_str(&std::env::var(name).unwrap_or_else(|_| panic!("{name} env")))
            .unwrap_or_else(|_| panic!("{name} uuid")),
    )
}

fn spawn_durability_child(test_name: &str, mode: &str, root: &Path, extra_env: &[(&str, String)]) {
    let mut command = process::Command::new(std::env::current_exe().expect("current test binary"));
    command
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env("CG_MODEL_CONNECTION_DURABILITY_CHILD", mode)
        .env("CG_MODEL_CONNECTION_DURABILITY_ROOT", root);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let output = command
        .output()
        .expect("spawn fresh model-connection verifier process");
    assert!(
        output.status.success(),
        "model connection durability must be verified from a fresh process, not parent process state; mode={mode}; child_status={:?}; stdout={}; stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_primary_connection_durable_from_fresh_process() {
    let root = child_catalog_root();
    let reopened = open_catalog_path(&root);
    let connection = reopened
        .read_connection("primary-hosted")
        .expect("fresh process reads durable primary connection");
    assert_eq!(connection.credential_reference.env_var, "MY_MODEL_KEY");
    assert_eq!(
        connection.exposed_secret, None,
        "fresh process read-back must expose only the env-var reference"
    );
    let model = reopened
        .read_model("reasoning-default")
        .expect("fresh process reads durable model reference");
    assert_eq!(model.connection_name.as_deref(), Some("primary-hosted"));
    assert_eq!(model.welded_auth_env_var, None);
    assert_eq!(model.exposed_secret, None);
    assert!(
        !catalog_disk_contains_path(&root, "secret-sentinel-hosted-key"),
        "fresh process durability check must still find no raw provider key on disk"
    );
}

fn assert_shared_connection_durable_from_fresh_process() {
    let root = child_catalog_root();
    let principal = developer();
    let mut config = config_with_secret(&principal, "SHARED_MODEL_KEY", "shared-secret");
    let mut reopened = open_catalog_path(&root);
    let reasoning = reopened.resolve_model_connection("reasoning", &principal, &mut config);
    let embedder = reopened.resolve_model_connection("remote-embedder", &principal, &mut config);
    let expected = MissingConnectionOutcome::Resolved(CredentialResolution {
        connection_name: "shared-remote".to_string(),
        env_var: "SHARED_MODEL_KEY".to_string(),
        secret: Some("shared-secret".to_string()),
    });
    assert_eq!(
        reasoning, expected,
        "fresh process must resolve the durable reasoning model through the shared connection"
    );
    assert_eq!(
        embedder, expected,
        "fresh process must resolve the durable remote embedder through the same connection"
    );
    assert_eq!(
        reopened.connection_count(),
        1,
        "fresh process must see one durable shared connection, not process-local model-kind tables"
    );
}

fn assert_rotating_reference_durable_from_fresh_process() {
    let root = child_catalog_root();
    let principal = developer();
    let mut config = config_with_secret(&principal, "ROTATING_MODEL_KEY", "v2-secret");
    let mut reopened = open_catalog_path(&root);
    let resolution = reopened.resolve_credential("rotating-hosted", &principal, &mut config);
    assert_eq!(
        resolution,
        Some(CredentialResolution {
            connection_name: "rotating-hosted".to_string(),
            env_var: "ROTATING_MODEL_KEY".to_string(),
            secret: Some("v2-secret".to_string()),
        }),
        "fresh process must resolve the rotated value through a durable env-var reference"
    );
    assert!(
        !catalog_disk_contains_path(&root, "v2-secret"),
        "fresh process durability check must not find the rotated raw key on disk"
    );
}

fn assert_status_durable_from_fresh_process() {
    let root = child_catalog_root();
    let client_context = context_id_env("CG_MODEL_CONNECTION_CLIENT_CONTEXT");
    let work_context = context_id_env("CG_MODEL_CONNECTION_WORK_CONTEXT");
    let reopened = open_catalog_path(&root);
    let client_status = reopened.status_for_context(client_context);
    let work_status = reopened.status_for_context(work_context);
    assert_eq!(client_status.connection_name.as_deref(), Some("client-a"));
    assert_eq!(client_status.model_id.as_deref(), Some("client-model"));
    assert!(client_status.consent_recorded);
    assert_eq!(client_status.spend_owner.as_deref(), Some("personal quota"));
    assert_eq!(work_status.connection_name.as_deref(), Some("work"));
    assert_eq!(work_status.model_id.as_deref(), Some("work-model"));
    assert!(
        !work_status.consent_recorded,
        "fresh process must observe the durable revoked-consent state"
    );
}

fn assert_migration_durable_from_fresh_process() {
    let root = child_catalog_root();
    let principal = developer();
    let mut config = config_with_secret(&principal, "LEGACY_MODEL_KEY", "legacy-secret");
    let mut reopened = open_catalog_path(&root);
    assert_eq!(
        reopened.migration_records(),
        vec![MigrationRecord {
            model_id: "legacy-reasoning".to_string(),
            from_env_var: "LEGACY_MODEL_KEY".to_string(),
            to_connection_name: "legacy-reasoning-connection".to_string(),
        }],
        "fresh process must read the durable migration record, not a parent process side map"
    );
    let migrated = reopened
        .read_model("legacy-reasoning")
        .expect("fresh process reads migrated model");
    assert_eq!(migrated.welded_auth_env_var, None);
    assert_eq!(
        migrated.connection_name.as_deref(),
        Some("legacy-reasoning-connection")
    );
    assert_eq!(migrated.exposed_secret, None);
    let resolved =
        reopened.resolve_credential("legacy-reasoning-connection", &principal, &mut config);
    assert_eq!(
        resolved,
        Some(CredentialResolution {
            connection_name: "legacy-reasoning-connection".to_string(),
            env_var: "LEGACY_MODEL_KEY".to_string(),
            secret: Some("legacy-secret".to_string()),
        }),
        "fresh process must resolve the migrated durable reference live"
    );
    assert!(
        !catalog_disk_contains_path(&root, "legacy-secret"),
        "fresh process durability check must still find no raw legacy key on disk"
    );
    assert_catalog_model_registration(
        &root,
        "legacy-reasoning",
        ModelKind::LlmClient,
        "legacy-reasoning-connection",
    );
}

fn catalog_entry_with_connection_reference(
    model_id: &str,
    kind: ModelKind,
    connection_name: &str,
) -> CatalogEntry {
    CatalogEntry {
        id: model_id.to_string(),
        kind,
        capability: Capability {
            dimension: None,
            output_shape: Some("understanding".to_string()),
            input_modality: Some("text".to_string()),
        },
        license: License {
            spdx: "Proprietary".to_string(),
            url: "https://provider.example.invalid/license".to_string(),
            requires_manual_acceptance: false,
        },
        source: Source::NamedConnection {
            connection_name: connection_name.to_string(),
        },
        byte_size: 0,
        cache_status: CacheStatus::Cached,
        local_path: None,
        install_instructions: None,
        audit_footnote: None,
        acceptance_recorded: false,
        manual_install_url: None,
        artifact_filenames: Vec::new(),
        per_file_sha256: Vec::new(),
    }
}

fn catalog_entry_with_welded_api_endpoint(
    model_id: &str,
    kind: ModelKind,
    auth_env_var: &str,
) -> CatalogEntry {
    CatalogEntry {
        id: model_id.to_string(),
        kind,
        capability: Capability {
            dimension: None,
            output_shape: Some("understanding".to_string()),
            input_modality: Some("text".to_string()),
        },
        license: License {
            spdx: "Proprietary".to_string(),
            url: "https://provider.example.invalid/license".to_string(),
            requires_manual_acceptance: false,
        },
        source: Source::ApiEndpoint {
            base_url: "https://provider.example.invalid/v1".to_string(),
            provider: "legacy-provider".to_string(),
            model: model_id.to_string(),
            auth_env_var: auth_env_var.to_string(),
        },
        byte_size: 0,
        cache_status: CacheStatus::Cached,
        local_path: None,
        install_instructions: None,
        audit_footnote: None,
        acceptance_recorded: false,
        manual_install_url: None,
        artifact_filenames: Vec::new(),
        per_file_sha256: Vec::new(),
    }
}

fn write_shipped_catalog_entries(root: &Path, entries: Vec<CatalogEntry>) {
    Catalog::open_with_entries_against_root(root, "connection-substrate-test", entries)
        .expect("write shipped catalog entries");
}

fn assert_catalog_model_registration(
    root: &Path,
    model_id: &str,
    kind: ModelKind,
    connection_name: &str,
) {
    let shipped_catalog = Catalog::open_with_scope_against_root(root, "connection-substrate-test")
        .expect("open shipped catalog");
    let entry = shipped_catalog.show(model_id);
    assert!(
        entry.is_ok(),
        "registering a connection-backed model must create the shipped cg-catalog Catalog entry"
    );
    let entry = entry.expect("entry was asserted present");
    assert_eq!(
        entry.kind, kind,
        "the shipped catalog entry must preserve the registered model kind"
    );
    assert!(
        matches!(
            entry.source,
            Source::NamedConnection {
                connection_name: ref catalog_connection_name,
            } if catalog_connection_name == connection_name
        ),
        "the shipped catalog entry must carry the named connection reference and must not keep the legacy welded auth field"
    );
}

// Use case: a developer names the variable that holds a provider key and later inspects the model setup; the key itself must never be printed back.
#[test]
fn credential_lives_on_named_connection_model_entry_carries_no_secret() {
    if model_connection_child_mode("primary-connection") {
        assert_primary_connection_durable_from_fresh_process();
        return;
    }

    let dir = TempDir::new().expect("catalog tempdir");
    let principal = developer();
    let mut config = config_with_secret(&principal, "MY_MODEL_KEY", "secret-sentinel-hosted-key");
    let mut catalog = open_catalog(&dir);

    let connection = catalog.connect_env_var(
        "primary-hosted",
        "MY_MODEL_KEY",
        principal.clone(),
        &mut config,
    );
    let model = catalog.register_model("reasoning-default", ModelKind::LlmClient, "primary-hosted");

    assert_eq!(connection.credential_reference.env_var, "MY_MODEL_KEY");
    assert_eq!(
        connection.exposed_secret, None,
        "connection read-back must expose the env-var reference, never the key value"
    );
    assert_eq!(model.connection_name.as_deref(), Some("primary-hosted"));
    assert_eq!(
        model.welded_auth_env_var, None,
        "hosted model entries must reference a named connection instead of carrying auth"
    );
    assert_eq!(
        model.exposed_secret, None,
        "model read-back must not contain the provider key"
    );
    assert!(
        !catalog_disk_contains(&dir, "secret-sentinel-hosted-key"),
        "the raw provider key must not be persisted in hidden catalog storage"
    );

    let reopened = open_catalog(&dir);
    assert_eq!(
        reopened.read_connection("primary-hosted"),
        Some(connection),
        "named connections must be durable catalog records, not runtime-local memory"
    );
    assert_eq!(
        reopened.read_model("reasoning-default"),
        Some(model),
        "model-to-connection references must survive reopening the catalog"
    );
    assert_catalog_model_registration(
        dir.path(),
        "reasoning-default",
        ModelKind::LlmClient,
        "primary-hosted",
    );
    spawn_durability_child(
        "model_connection::credential_lives_on_named_connection_model_entry_carries_no_secret",
        "primary-connection",
        dir.path(),
        &[],
    );
}

// Use case: the same account connection should serve text embedding and reasoning so the user does not maintain two credential systems.
#[test]
fn same_named_connection_serves_a_remote_embedder_and_the_reasoning_runtime() {
    if model_connection_child_mode("shared-connection") {
        assert_shared_connection_durable_from_fresh_process();
        return;
    }

    let dir = TempDir::new().expect("catalog tempdir");
    let principal = developer();
    let mut config = config_with_secret(&principal, "SHARED_MODEL_KEY", "shared-secret");
    let mut catalog = open_catalog(&dir);

    catalog.connect_env_var(
        "shared-remote",
        "SHARED_MODEL_KEY",
        principal.clone(),
        &mut config,
    );
    catalog.register_model("reasoning", ModelKind::LlmClient, "shared-remote");
    catalog.register_model("remote-embedder", ModelKind::TextEmbedder, "shared-remote");

    let reasoning = catalog.resolve_model_connection("reasoning", &principal, &mut config);
    let embedder = catalog.resolve_model_connection("remote-embedder", &principal, &mut config);

    assert_eq!(
        reasoning,
        MissingConnectionOutcome::Resolved(CredentialResolution {
            connection_name: "shared-remote".to_string(),
            env_var: "SHARED_MODEL_KEY".to_string(),
            secret: Some("shared-secret".to_string()),
        })
    );
    assert_eq!(
        embedder, reasoning,
        "remote embedder and reasoning runtime must resolve the identical named connection"
    );
    assert_eq!(
        catalog.connection_count(),
        1,
        "registering the embedder must not create a second connection surface"
    );
    assert_catalog_model_registration(
        dir.path(),
        "reasoning",
        ModelKind::LlmClient,
        "shared-remote",
    );
    assert_catalog_model_registration(
        dir.path(),
        "remote-embedder",
        ModelKind::TextEmbedder,
        "shared-remote",
    );

    let mut reopened = open_catalog(&dir);
    let reopened_reasoning =
        reopened.resolve_model_connection("reasoning", &principal, &mut config);
    let reopened_embedder =
        reopened.resolve_model_connection("remote-embedder", &principal, &mut config);
    assert_eq!(
        reopened_embedder, reopened_reasoning,
        "shared connection identity must remain shared after reopening the catalog"
    );
    spawn_durability_child(
        "model_connection::same_named_connection_serves_a_remote_embedder_and_the_reasoning_runtime",
        "shared-connection",
        dir.path(),
        &[],
    );

    let catalog_only_dir = TempDir::new().expect("catalog-only tempdir");
    let mut catalog_only_config =
        config_with_secret(&principal, "CATALOG_ONLY_SHARED_KEY", "catalog-only-secret");
    let mut catalog_only = open_catalog(&catalog_only_dir);
    catalog_only.connect_env_var(
        "catalog-only-shared",
        "CATALOG_ONLY_SHARED_KEY",
        principal.clone(),
        &mut catalog_only_config,
    );
    write_shipped_catalog_entries(
        catalog_only_dir.path(),
        vec![
            catalog_entry_with_connection_reference(
                "catalog-only-reasoning",
                ModelKind::LlmClient,
                "catalog-only-shared",
            ),
            catalog_entry_with_connection_reference(
                "catalog-only-embedder",
                ModelKind::TextEmbedder,
                "catalog-only-shared",
            ),
        ],
    );
    let catalog_only_reasoning = catalog_only.resolve_model_connection(
        "catalog-only-reasoning",
        &principal,
        &mut catalog_only_config,
    );
    let catalog_only_embedder = catalog_only.resolve_model_connection(
        "catalog-only-embedder",
        &principal,
        &mut catalog_only_config,
    );
    assert_eq!(
        catalog_only_reasoning,
        MissingConnectionOutcome::Resolved(CredentialResolution {
            connection_name: "catalog-only-shared".to_string(),
            env_var: "CATALOG_ONLY_SHARED_KEY".to_string(),
            secret: Some("catalog-only-secret".to_string()),
        }),
        "model-to-connection routing must read the shipped catalog row, not only a sidecar map"
    );
    assert_eq!(
        catalog_only_embedder, catalog_only_reasoning,
        "both model kinds must resolve through the same shipped catalog connection reference"
    );
}

// Use case: after rotating a provider key in the shell, the connected model should pick up the new value without reconnecting.
#[test]
fn connecting_by_naming_an_env_var_records_a_live_reference_not_a_pasted_key() {
    if model_connection_child_mode("rotating-reference") {
        assert_rotating_reference_durable_from_fresh_process();
        return;
    }

    let dir = TempDir::new().expect("catalog tempdir");
    let principal = developer();
    let mut config = config_with_secret(&principal, "ROTATING_MODEL_KEY", "v1-secret");
    let mut catalog = open_catalog(&dir);

    catalog.connect_env_var(
        "rotating-hosted",
        "ROTATING_MODEL_KEY",
        principal.clone(),
        &mut config,
    );
    assert!(
        !catalog_disk_contains(&dir, "v1-secret"),
        "connecting must persist only the env-var reference, not the initial raw key"
    );
    let first = catalog
        .resolve_credential("rotating-hosted", &principal, &mut config)
        .expect("credential resolution");
    assert_eq!(first.secret.as_deref(), Some("v1-secret"));

    config.insert_secret(principal.clone(), "ROTATING_MODEL_KEY", "v2-secret");
    let second = catalog
        .resolve_credential("rotating-hosted", &principal, &mut config)
        .expect("credential resolution after rotation");

    assert_eq!(
        second,
        CredentialResolution {
            connection_name: "rotating-hosted".to_string(),
            env_var: "ROTATING_MODEL_KEY".to_string(),
            secret: Some("v2-secret".to_string()),
        },
        "credential resolution must follow the live reference, not a connect-time snapshot"
    );

    let mut reopened = open_catalog(&dir);
    let reopened_resolution =
        reopened.resolve_credential("rotating-hosted", &principal, &mut config);
    assert_eq!(
        reopened_resolution,
        Some(second),
        "the live env-var reference must resolve after reopening the catalog"
    );
    assert!(
        !catalog_disk_contains(&dir, "v2-secret"),
        "rotating credentials must not write the replacement raw key into catalog storage"
    );
    spawn_durability_child(
        "model_connection::connecting_by_naming_an_env_var_records_a_live_reference_not_a_pasted_key",
        "rotating-reference",
        dir.path(),
        &[],
    );
}

// Use case: a consultant working in two contexts needs status to show which model and quota each context touches before any background spend happens.
#[test]
fn status_read_shows_active_connection_consent_state_and_which_model_a_context_resolves() {
    if model_connection_child_mode("status") {
        assert_status_durable_from_fresh_process();
        return;
    }

    let dir = TempDir::new().expect("catalog tempdir");
    let principal = developer();
    let mut config = PrincipalAwareConfig::default();
    config.insert_secret(principal.clone(), "KEY_CLIENT_A", "client-a-secret");
    config.insert_secret(principal.clone(), "KEY_WORK", "work-secret");
    let mut catalog = open_catalog(&dir);
    let client_context = ContextId::new_v7();
    let work_context = ContextId::new_v7();

    catalog.connect_env_var("client-a", "KEY_CLIENT_A", principal.clone(), &mut config);
    catalog.connect_env_var("work", "KEY_WORK", principal.clone(), &mut config);
    catalog.register_model("client-model", ModelKind::LlmClient, "client-a");
    catalog.register_model("work-model", ModelKind::LlmClient, "work");
    catalog.bind_context_model(client_context, "client-model");
    catalog.bind_context_model(work_context, "work-model");
    catalog.record_consent("client-a", "personal quota");

    let client_status = catalog.status_for_context(client_context);
    let work_status = catalog.status_for_context(work_context);
    assert_eq!(client_status.connection_name.as_deref(), Some("client-a"));
    assert_eq!(client_status.model_id.as_deref(), Some("client-model"));
    assert!(client_status.consent_recorded);
    assert_eq!(client_status.spend_owner.as_deref(), Some("personal quota"));
    assert_eq!(work_status.connection_name.as_deref(), Some("work"));
    assert_eq!(work_status.model_id.as_deref(), Some("work-model"));
    assert!(
        !work_status.consent_recorded,
        "consent status must be scoped to the context's resolved connection"
    );

    catalog.record_consent("work", "work quota");
    let work_after_consent = catalog.status_for_context(work_context);
    let client_after_work_toggle = catalog.status_for_context(client_context);
    assert!(work_after_consent.consent_recorded);
    assert_eq!(
        work_after_consent.spend_owner.as_deref(),
        Some("work quota")
    );
    assert_eq!(
        client_after_work_toggle.spend_owner.as_deref(),
        Some("personal quota"),
        "recording consent for work must not rewrite the client-a spend owner"
    );

    catalog.revoke_consent("work");
    let revoked_status = catalog.status_for_context(work_context);
    assert!(
        !revoked_status.consent_recorded,
        "revoking consent must update the live connection status before any later paid call"
    );
    let reopened_after_revoke = open_catalog(&dir);
    let reopened_revoked_status = reopened_after_revoke.status_for_context(work_context);
    assert!(
        !reopened_revoked_status.consent_recorded,
        "revoked consent must survive reopening the catalog"
    );
    spawn_durability_child(
        "model_connection::status_read_shows_active_connection_consent_state_and_which_model_a_context_resolves",
        "status",
        dir.path(),
        &[
            (
                "CG_MODEL_CONNECTION_CLIENT_CONTEXT",
                client_context.as_uuid().to_string(),
            ),
            (
                "CG_MODEL_CONNECTION_WORK_CONTEXT",
                work_context.as_uuid().to_string(),
            ),
        ],
    );
    let revoked_client = Arc::new(ConnectionRecordingClient::default());
    let mut revoked_runtime =
        UnderstandingRuntime::new(reopened_after_revoke, principal.clone(), config.clone())
            .with_client("work-model", revoked_client.clone());
    let revoked_understanding = revoked_runtime
        .understand(request(work_context, "revoked work scoped request"))
        .expect("revoked work scoped request completes on lesser path");
    assert_eq!(
        revoked_understanding.degraded_marker.as_deref(),
        Some(DEGRADED_MARKER),
        "reopened revoked consent must keep refusing the paid backend"
    );
    assert_eq!(
        revoked_client.observed(),
        Vec::<(Option<String>, Option<String>)>::new(),
        "a reopened revoked connection must not invoke the backend through stale consent"
    );
    catalog.record_consent("work", "work quota");

    let work_resolution = catalog
        .resolve_credential("work", &principal, &mut config)
        .expect("work credential resolution");
    let client_resolution = catalog
        .resolve_credential("client-a", &principal, &mut config)
        .expect("client credential resolution");
    assert_eq!(work_resolution.secret.as_deref(), Some("work-secret"));
    assert_eq!(client_resolution.secret.as_deref(), Some("client-a-secret"));
    assert_eq!(
        catalog.resolved_connection_log(),
        &["work".to_string(), "client-a".to_string()],
        "a scoped request must resolve only the connection chosen for that context"
    );

    let client = Arc::new(ConnectionRecordingClient::default());
    let mut runtime = UnderstandingRuntime::new(catalog.clone(), principal.clone(), config.clone())
        .with_client("client-model", client.clone())
        .with_client("work-model", client.clone());
    let work = runtime
        .understand(request(work_context, "work scoped request"))
        .expect("work scoped understanding");
    let client_a = runtime
        .understand(request(client_context, "client scoped request"))
        .expect("client scoped understanding");
    assert_eq!(work.marker, "understood:work scoped request");
    assert_eq!(client_a.marker, "understood:client scoped request");
    assert_eq!(
        client.observed(),
        vec![
            (Some("work".to_string()), Some("work-secret".to_string())),
            (
                Some("client-a".to_string()),
                Some("client-a-secret".to_string())
            ),
        ],
        "runtime requests must resolve credentials from the model bound to the request context"
    );

    let reviewer = Principal::Human("reviewer@example.com".to_string());
    config.insert_secret(reviewer.clone(), "KEY_WORK", "reviewer-work-secret");
    let reviewer_client = Arc::new(ConnectionRecordingClient::default());
    let mut reviewer_runtime = UnderstandingRuntime::new(catalog.clone(), reviewer, config.clone())
        .with_client("work-model", reviewer_client.clone());
    reviewer_runtime
        .understand(request(work_context, "reviewer work scoped request"))
        .expect("reviewer work scoped understanding");
    assert_eq!(
        reviewer_client.observed(),
        vec![(
            Some("work".to_string()),
            Some("reviewer-work-secret".to_string())
        )],
        "credential resolution must be scoped by principal as well as context"
    );
}

// Use case: an upgrade from the old hosted shape should leave an auditably migrated connection, not a silent rewrite or a dropped credential.
#[test]
fn migration_from_welded_auth_env_var_is_explicit_observable_and_reads_no_secret() {
    if model_connection_child_mode("migration") {
        assert_migration_durable_from_fresh_process();
        return;
    }

    let dir = TempDir::new().expect("catalog tempdir");
    let principal = developer();
    let mut config = config_with_secret(&principal, "LEGACY_MODEL_KEY", "legacy-secret");
    let mut catalog = open_catalog(&dir);

    write_shipped_catalog_entries(
        dir.path(),
        vec![catalog_entry_with_welded_api_endpoint(
            "legacy-reasoning",
            ModelKind::LlmClient,
            "LEGACY_MODEL_KEY",
        )],
    );
    catalog
        .register_legacy_hosted_model_with_welded_env_var("legacy-reasoning", "LEGACY_MODEL_KEY");
    let records = catalog.migrate_legacy_welded_credentials();

    assert_eq!(
        records,
        vec![MigrationRecord {
            model_id: "legacy-reasoning".to_string(),
            from_env_var: "LEGACY_MODEL_KEY".to_string(),
            to_connection_name: "legacy-reasoning-connection".to_string(),
        }],
        "migration must be observable and identify exactly which reference moved"
    );
    let migrated = catalog
        .read_model("legacy-reasoning")
        .expect("migrated model");
    assert_eq!(
        migrated.welded_auth_env_var, None,
        "the upgraded model entry must no longer carry the credential reference directly"
    );
    assert_eq!(
        migrated.connection_name.as_deref(),
        Some("legacy-reasoning-connection")
    );
    assert_eq!(
        migrated.exposed_secret, None,
        "migration must move only the reference, never the secret value"
    );
    assert_eq!(
        config.read_log(&principal),
        Vec::<String>::new(),
        "migration must not read the provider secret"
    );
    assert!(
        !catalog_disk_contains(&dir, "legacy-secret"),
        "migration must not persist the raw legacy provider key"
    );
    let resolved = catalog
        .resolve_credential("legacy-reasoning-connection", &principal, &mut config)
        .expect("migrated credential resolution");
    assert_eq!(resolved.secret.as_deref(), Some("legacy-secret"));

    let mut reopened = open_catalog(&dir);
    let reopened_resolution =
        reopened.resolve_credential("legacy-reasoning-connection", &principal, &mut config);
    assert_eq!(
        reopened_resolution,
        Some(resolved),
        "migrated references must resolve live after reopening the catalog"
    );
    assert_catalog_model_registration(
        dir.path(),
        "legacy-reasoning",
        ModelKind::LlmClient,
        "legacy-reasoning-connection",
    );
    spawn_durability_child(
        "model_connection::migration_from_welded_auth_env_var_is_explicit_observable_and_reads_no_secret",
        "migration",
        dir.path(),
        &[],
    );

    let context_id = ContextId::new_v7();
    catalog.bind_context_model(context_id, "legacy-reasoning");
    catalog.record_consent("legacy-reasoning-connection", "legacy quota");
    let client = Arc::new(ConnectionRecordingClient::default());
    let mut runtime = UnderstandingRuntime::new(catalog, principal, config)
        .with_client("legacy-reasoning", client.clone());
    runtime
        .understand(request(context_id, "migrated shipped catalog request"))
        .expect("migrated shipped catalog runtime request");
    assert_eq!(
        client.observed(),
        vec![(
            Some("legacy-reasoning-connection".to_string()),
            Some("legacy-secret".to_string())
        )],
        "runtime resolution after migration must use the shipped Catalog legacy entry, not a stale side map"
    );
}

// Use case: a typo in a model's connection name should be fixed loudly, not treated like the optional no-model fallback.
#[test]
fn model_referencing_a_missing_connection_is_a_hard_error_not_a_silent_fallback() {
    let _feature_env_guard = scoped_feature_envs(FeatureEnvMode::Poisoned);
    assert_eq!(
        std::env::var("CG_LLM_API_KEY").as_deref(),
        Ok("wrong-process-sentinel-CG_LLM_API_KEY"),
        "this test must poison the real process-global feature key so a std::env fallback cannot pass invisibly"
    );
    let principal = developer();
    let mut config = PrincipalAwareConfig::default();
    config.insert_secret(
        principal.clone(),
        "CG_LLM_API_KEY",
        "global-config-sentinel",
    );
    let mut catalog = ModelConnectionCatalog::default();
    let context_id = ContextId::new_v7();
    catalog.register_model(
        "dangling-reasoning",
        ModelKind::LlmClient,
        "deleted-connection",
    );
    catalog.bind_context_model(context_id, "dangling-reasoning");

    let result = catalog.resolve_model_connection("dangling-reasoning", &principal, &mut config);
    assert_eq!(
        result,
        MissingConnectionOutcome::HardError {
            model_id: "dangling-reasoning".to_string(),
            connection_name: "deleted-connection".to_string(),
        },
        "a dangling reference is a catalog error, not graceful no-model degradation"
    );
    assert_eq!(
        config.read_log(&principal),
        Vec::<String>::new(),
        "dangling resolution must not fall back to a global feature credential"
    );

    let client = Arc::new(ConnectionRecordingClient::default());
    let mut runtime = UnderstandingRuntime::new(catalog, principal.clone(), config)
        .with_client("dangling-reasoning", client.clone());
    let runtime_result = runtime.understand(request(context_id, "dangling runtime request"));
    assert_eq!(
        runtime_result,
        Err(UnderstandingRuntimeError::DanglingConnection {
            model_id: "dangling-reasoning".to_string(),
            connection_name: "deleted-connection".to_string(),
        }),
        "using a model with a dangling connection must return an explicit configuration error"
    );
    assert_eq!(
        client.observed(),
        Vec::<(Option<String>, Option<String>)>::new(),
        "a dangling connection must not be hidden by invoking the backend through a global key"
    );
}
