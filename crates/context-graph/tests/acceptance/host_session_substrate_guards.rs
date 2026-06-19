use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use cg_catalog::{
    CredentialReference, CredentialSource, DEGRADED_MARKER, ModelConnectionCatalog, ModelKind,
    PrincipalAwareConfig, UnderstandingRuntime,
};
use context_graph::{
    ContextId, DeploymentSurface, FailureBucket, LlmClient, ModelInvocation, ModelReply, Principal,
    UnderstandingConsumer, UnderstandingOutcome, UnderstandingRequest,
};
use tempfile::TempDir;

macro_rules! ok {
    ($expr:expr, $step:literal) => {{
        let result = $expr;
        assert!(result.is_ok(), "{}: expected Ok, got {:?}", $step, result);
        result.unwrap()
    }};
}

fn developer() -> Principal {
    Principal::Human("developer@example.test".to_string())
}

fn request(context_id: ContextId, input: &str) -> UnderstandingRequest {
    UnderstandingRequest {
        context_id,
        consumer: UnderstandingConsumer::ConversationCondensation,
        deployment: DeploymentSurface::Edge,
        input: input.to_string(),
        paid: true,
    }
}

#[derive(Debug, Default)]
struct RecordingClient {
    observed: Mutex<Vec<Option<String>>>,
}

impl RecordingClient {
    fn observed(&self) -> Vec<Option<String>> {
        self.observed.lock().expect("observed lock").clone()
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
            .push(invocation.resolved_connection.clone());
        Ok(ModelReply {
            marker: format!("understood:{}", invocation.request.input),
            phrasing: format!("phrased:{}", invocation.request.input),
            advisory_flag: false,
            malformed: false,
            usage: None,
        })
    }
}

fn degraded(input: &str, failure_bucket: FailureBucket) -> UnderstandingOutcome {
    UnderstandingOutcome {
        marker: format!("declarative:{input}"),
        degraded_marker: Some(DEGRADED_MARKER.to_string()),
        failure_bucket: Some(failure_bucket),
        phrasing: None,
        model_invoked: false,
        resolved_connection: None,
        cache_path_consulted: None,
        stored_understanding: Some(input.to_string()),
        usage: None,
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn locked_package_names(lockfile: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_package = false;
    for raw_line in lockfile.lines() {
        let line = raw_line.trim();
        if line == "[[package]]" {
            in_package = true;
            continue;
        }
        if in_package && let Some(value) = line.strip_prefix("name = ") {
            names.push(value.trim_matches('"').to_string());
            in_package = false;
        }
    }
    names
}

fn default_build_tree(package: &str) -> String {
    let output = Command::new("cargo")
        .args(["tree", "-p", package, "--edges", "normal,build"])
        .current_dir(workspace_root())
        .output()
        .expect("run cargo tree for default build dependency audit");
    assert!(
        output.status.success(),
        "cargo tree for {package} default build failed: status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("cargo tree output utf8")
}

#[test]
fn existing_credential_source_records_decode_unchanged() {
    let dir = TempDir::new().expect("catalog tempdir");
    let principal = developer();
    let mut config = PrincipalAwareConfig::default();
    config.insert_secret(principal.clone(), "MODEL_KEY", "env-secret");
    let mut catalog = ModelConnectionCatalog::open(dir.path());
    catalog.connect_env_var("env-conn", "MODEL_KEY", principal.clone(), &mut config);
    catalog.record_consent("env-conn", "developer quota");
    let env_connection = catalog
        .read_connection("env-conn")
        .expect("env connection after consent");

    let stored_connection = cg_catalog::ConnectionRead {
        name: "stored-conn".to_string(),
        credential_reference: CredentialReference {
            source: CredentialSource::StoredSecret {
                secret_ref: "secret://model".to_string(),
            },
            env_var: String::new(),
        },
        exposed_secret: None,
        consent_recorded: false,
        spend_owner: "unknown".to_string(),
    };
    let path = dir.path().join("cg/model-connections.toml");
    std::fs::create_dir_all(path.parent().expect("state parent")).expect("state dir");
    let fixture = r#"write_sequence = 2
models = []
context_models = []
migration_records = []

[[connections]]
name = "env-conn"
consent_recorded = true
spend_owner = "developer quota"

[connections.credential_reference]
env_var = "MODEL_KEY"

[connections.credential_reference.source.EnvVar]
env_var = "MODEL_KEY"

[[connections]]
name = "stored-conn"
consent_recorded = false
spend_owner = "unknown"

[connections.credential_reference]
env_var = ""

[connections.credential_reference.source.StoredSecret]
secret_ref = "secret://model"
"#;
    std::fs::write(&path, fixture).expect("write catalog fixture");

    let reopened = ModelConnectionCatalog::open(dir.path());
    assert_eq!(reopened.load_error(), None);
    assert_eq!(
        reopened.read_connection("env-conn"),
        Some(env_connection),
        "env-var source and consent must round-trip unchanged"
    );
    assert_eq!(
        reopened.read_connection("stored-conn"),
        Some(stored_connection),
        "stored-secret source must decode unchanged"
    );
}

#[test]
fn paid_call_requires_recorded_consent_and_uses_named_connection() {
    let dir = TempDir::new().expect("catalog tempdir");
    let principal = developer();
    let mut config = PrincipalAwareConfig::default();
    config.insert_secret(principal.clone(), "MODEL_KEY", "secret");
    let context_id = ContextId::new_v7();
    let mut catalog = ModelConnectionCatalog::open(dir.path());
    catalog.connect_env_var("hosted", "MODEL_KEY", principal.clone(), &mut config);
    catalog.register_model("reasoning", ModelKind::LlmClient, "hosted");
    catalog.bind_context_model(context_id, "reasoning");

    let client = Arc::new(RecordingClient::default());
    let mut runtime = UnderstandingRuntime::new(catalog, principal.clone(), config.clone())
        .with_client("reasoning", client.clone());
    let no_consent = ok!(
        runtime.understand(request(context_id, "before consent")),
        "understand without consent"
    );
    assert_eq!(
        no_consent,
        degraded("before consent", FailureBucket::Unauthorized)
    );
    assert_eq!(
        client.observed(),
        Vec::<Option<String>>::new(),
        "unconsented connection must not invoke the paid client"
    );

    runtime
        .catalog_mut()
        .record_consent("hosted", "developer quota");
    let after = ok!(
        runtime.understand(request(context_id, "after consent")),
        "understand after consent"
    );
    assert!(after.model_invoked);
    assert_eq!(after.resolved_connection.as_deref(), Some("hosted"));
    assert_eq!(
        client.observed(),
        vec![Some("hosted".to_string())],
        "consent enables exactly one paid invocation"
    );
}

#[test]
fn missing_connection_degrades_to_typed_marker_not_crash() {
    let client = Arc::new(RecordingClient::default());
    let mut runtime = UnderstandingRuntime::new(
        ModelConnectionCatalog::default(),
        developer(),
        PrincipalAwareConfig::default(),
    )
    .with_client("unregistered", client.clone());
    let outcome = ok!(
        runtime.understand(request(ContextId::new_v7(), "no model configured")),
        "understand without configured model"
    );
    assert_eq!(
        outcome,
        degraded("no model configured", FailureBucket::NotConnected)
    );
    assert_eq!(client.observed(), Vec::<Option<String>>::new());
}

#[test]
fn workspace_default_build_remains_tokio_free_and_sdk_free() {
    let lockfile = std::fs::read_to_string(workspace_root().join("Cargo.lock"))
        .expect("workspace lockfile readable");
    let names = locked_package_names(&lockfile);
    for package in ["context-graph", "cg-catalog"] {
        let tree = default_build_tree(package);
        assert!(
            !tree.lines().any(|line| line.contains("tokio v")),
            "{package} default-build dependency tree must remain tokio-free:\n{tree}"
        );
    }
    let sdk_names = names
        .iter()
        .filter(|name| {
            name.starts_with("anthropic")
                || name.starts_with("async-openai")
                || name.as_str() == "openai"
        })
        .collect::<Vec<_>>();
    assert!(
        sdk_names.is_empty(),
        "model connection support must not add provider SDK crates: {sdk_names:?}"
    );
}
