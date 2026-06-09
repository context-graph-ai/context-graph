use std::fs;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use cg_catalog::{
    CredentialSource, DEGRADED_MARKER, ModelConnectionCatalog, PrincipalAwareConfig,
    UnderstandingRuntime,
};
use context_graph::{
    ContextId, DeploymentSurface, FailureBucket, Principal, UnderstandingConsumer,
    UnderstandingRequest,
};
use serde_json::Value;

const CONNECTION_NAME: &str = "byok-openai-compatible";
const MODEL_ID: &str = "byok-reasoning-model";
const ENV_VAR: &str = "CG_TEST_BYOK_TOKEN";
const STORED_SECRET_REF: &str = "connection:byok-openai-compatible:api_key";

fn developer() -> Principal {
    Principal::Human("developer@example.test".to_string())
}

fn paid_request(context_id: ContextId, input: &str) -> UnderstandingRequest {
    UnderstandingRequest {
        context_id,
        consumer: UnderstandingConsumer::ConversationCondensation,
        deployment: DeploymentSurface::Edge,
        input: input.to_string(),
        paid: true,
    }
}

fn config_with_secret(principal: &Principal, env_var: &str, secret: &str) -> PrincipalAwareConfig {
    let mut config = PrincipalAwareConfig::default();
    config.insert_secret(principal.clone(), env_var, secret);
    config
}

fn catalog_text(root: &Path) -> String {
    fs::read_to_string(root.join("cg/model-connections.toml")).unwrap_or_default()
}

fn write_stored_secret_connection(root: &Path, consent_recorded: bool) {
    let path = root.join("cg/model-connections.toml");
    fs::create_dir_all(path.parent().expect("catalog parent")).expect("catalog parent exists");
    fs::write(
        path,
        format!(
            r#"write_sequence = 1
active_connection_name = "{CONNECTION_NAME}"
models = []
context_models = []
migration_records = []

[[connections]]
name = "{CONNECTION_NAME}"
consent_recorded = {consent_recorded}
spend_owner = "developer@example.test"

[connections.credential_reference]
env_var = ""

[connections.credential_reference.source.StoredSecret]
secret_ref = "{STORED_SECRET_REF}"
"#
        ),
    )
    .expect("stored-secret connection fixture writes");
}

#[derive(Debug)]
struct CapturedRequest {
    authorization: Option<String>,
}

struct OneShotStub {
    base_url: String,
    received: mpsc::Receiver<Option<CapturedRequest>>,
    handle: thread::JoinHandle<()>,
}

impl OneShotStub {
    fn openai_reply(reply: &str) -> Self {
        let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server starts");
        let base_url = crate::common::stub_base_url(&server);
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "content": reply,
                },
            }],
        })
        .to_string();
        let (sender, received) = mpsc::channel();
        let handle = thread::spawn(move || {
            let Some(request) = server.recv_timeout(Duration::from_secs(2)).ok().flatten() else {
                let _ = sender.send(None);
                return;
            };
            let authorization = request
                .headers()
                .iter()
                .find(|header| header.field.equiv("Authorization"))
                .map(|header| header.value.as_str().to_string());
            let _ = sender.send(Some(CapturedRequest { authorization }));
            let content_type =
                tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                    .expect("content type header");
            let response = tiny_http::Response::from_string(body)
                .with_status_code(200)
                .with_header(content_type);
            let _ = request.respond(response);
        });
        Self {
            base_url,
            received,
            handle,
        }
    }

    fn join(self) -> Option<CapturedRequest> {
        let request = self
            .received
            .recv_timeout(Duration::from_secs(3))
            .unwrap_or(None);
        self.handle.join().expect("stub server joins");
        request
    }
}

fn value_contains_string(value: &Value, needle: &str) -> bool {
    match value {
        Value::String(text) => text.contains(needle),
        Value::Array(values) => values
            .iter()
            .any(|value| value_contains_string(value, needle)),
        Value::Object(map) => map
            .iter()
            .any(|(key, value)| key.contains(needle) || value_contains_string(value, needle)),
        _ => false,
    }
}

#[test]
fn supplied_token_lives_on_the_credential_reference_and_never_on_the_model_entry() {
    let temp = tempfile::tempdir().expect("tempdir");
    let secret = "sk-supplied-token-never-on-model";
    let principal = developer();
    let mut config = PrincipalAwareConfig::default();
    config.insert_stored_secret(principal.clone(), STORED_SECRET_REF, secret);
    write_stored_secret_connection(temp.path(), true);
    let stub = OneShotStub::openai_reply("stored secret model reply");
    let mut catalog = ModelConnectionCatalog::open(temp.path());
    catalog.register_hosted_model(
        MODEL_ID,
        CONNECTION_NAME,
        &stub.base_url,
        "openai-compatible",
        "gpt-4.1-mini",
    );
    let context_id = ContextId::new_v7();
    catalog.bind_context_model(context_id, MODEL_ID);

    let connection = catalog
        .read_connection(CONNECTION_NAME)
        .expect("connection should be recorded");
    assert_eq!(
        connection.credential_reference.source,
        CredentialSource::StoredSecret {
            secret_ref: format!("connection:{CONNECTION_NAME}:api_key")
        },
        "a supplied BYOK token should live behind the connection credential reference"
    );
    assert_eq!(connection.exposed_secret, None);
    let resolved = catalog
        .resolve_credential(CONNECTION_NAME, &principal, &mut config)
        .expect("stored credential resolves through the named connection");
    assert_eq!(resolved.secret.as_deref(), Some(secret));
    assert_eq!(
        config.read_log(&principal),
        vec![format!("stored-secret:{STORED_SECRET_REF}")],
        "paid use must resolve the stored credential reference rather than an env var"
    );

    let model = catalog
        .read_model(MODEL_ID)
        .expect("model should be recorded");
    let model_json = serde_json::to_value(&model).expect("model serializes");
    assert!(
        !value_contains_string(&model_json, secret),
        "the hosted model entry must not carry the supplied token: {model_json}"
    );
    assert!(
        !catalog_text(temp.path()).contains(secret),
        "the catalog file must not persist the raw supplied token"
    );

    let mut runtime: UnderstandingRuntime =
        UnderstandingRuntime::new(catalog, principal.clone(), config);
    let outcome = runtime
        .understand(paid_request(context_id, "prove stored BYOK token"))
        .expect("stored-secret BYOK call should complete");
    let request = stub.join().expect("stored-secret BYOK should call stub");
    assert_eq!(
        request.authorization.as_deref(),
        Some("Bearer sk-supplied-token-never-on-model"),
        "stored BYOK token must be used only as an outbound authorization header"
    );
    assert_eq!(
        outcome.phrasing.as_deref(),
        Some("stored secret model reply"),
        "stored-secret BYOK path must return the model reply through the same result surface"
    );
}

#[test]
fn stored_secret_credential_resolution_requires_recorded_consent() {
    let temp = tempfile::tempdir().expect("tempdir");
    let secret = "sk-stored-token-without-consent";
    let principal = developer();
    let mut config = PrincipalAwareConfig::default();
    config.insert_stored_secret(principal.clone(), STORED_SECRET_REF, secret);
    write_stored_secret_connection(temp.path(), false);

    let mut catalog = ModelConnectionCatalog::open(temp.path());
    let resolved = catalog.resolve_credential(CONNECTION_NAME, &principal, &mut config);

    assert!(
        resolved.is_none(),
        "stored BYOK token plaintext must not resolve until paid-use consent is recorded"
    );
    assert_eq!(
        config.read_log(&principal),
        Vec::<String>::new(),
        "refused stored-secret resolution must not read or audit plaintext token access"
    );
}

#[test]
fn list_show_status_emit_no_secret_in_json_or_text_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let secret = "sk-status-secret";
    let principal = developer();
    let mut config = config_with_secret(&principal, ENV_VAR, secret);
    let mut catalog = ModelConnectionCatalog::open(temp.path());
    catalog.connect_env_var(CONNECTION_NAME, ENV_VAR, principal.clone(), &mut config);
    catalog.record_consent(CONNECTION_NAME, "developer@example.test");
    catalog.register_hosted_model(
        MODEL_ID,
        CONNECTION_NAME,
        "https://llm.example.test/v1",
        "openai-compatible",
        "gpt-4.1-mini",
    );
    let context_id = ContextId::new_v7();
    catalog.bind_context_model(context_id, MODEL_ID);

    let list_json = serde_json::to_value(catalog.list_connections()).expect("list serializes");
    let show_json = serde_json::to_value(catalog.read_model(MODEL_ID)).expect("show serializes");
    let status_json =
        serde_json::to_value(catalog.status_for_context(context_id)).expect("status serializes");
    let text_output = format!("{list_json:?}\n{show_json:?}\n{status_json:?}");

    assert!(
        !text_output.contains(secret)
            && !value_contains_string(&list_json, secret)
            && !value_contains_string(&show_json, secret)
            && !value_contains_string(&status_json, secret),
        "list/show/status surfaces must not reveal the resolved endpoint secret: {text_output}"
    );
    assert_eq!(
        status_json.get("state").and_then(Value::as_str),
        Some("ready"),
        "status JSON should expose a stable ready state without printing the secret"
    );
}

#[test]
fn no_connection_configured_falls_back_to_raw_storage_with_typed_not_connected_bucket_and_retains_input()
 {
    let input = "retain this uncondensed user chunk";
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = ModelConnectionCatalog::open(temp.path());
    let mut runtime: UnderstandingRuntime =
        UnderstandingRuntime::new(catalog, developer(), PrincipalAwareConfig::default());

    let outcome = runtime
        .understand(paid_request(ContextId::new_v7(), input))
        .expect("no-connection path returns a degraded outcome");

    assert_eq!(outcome.marker, format!("declarative:{input}"));
    assert_eq!(outcome.degraded_marker.as_deref(), Some(DEGRADED_MARKER));
    assert_eq!(
        outcome.failure_bucket,
        Some(FailureBucket::NotConnected),
        "no configured connection must surface a typed not-connected failure bucket"
    );
    assert_eq!(
        outcome.stored_understanding.as_deref(),
        Some(input),
        "degraded understanding must retain the original input verbatim for retry"
    );
    assert_eq!(
        outcome.phrasing, None,
        "degraded paths must not fabricate model phrasing"
    );
}
