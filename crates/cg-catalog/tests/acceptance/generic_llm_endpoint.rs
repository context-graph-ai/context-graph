use std::collections::BTreeSet;
use std::fs;
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use cg_catalog::{
    DEGRADED_MARKER, ModelConnectionCatalog, NoopLlmClient, PrincipalAwareConfig,
    ProviderAuthScheme, ProviderDialect, ProviderTransport, UnderstandingRuntime,
    load_provider_specs, try_load_provider_specs,
};
use context_graph::{
    ContextId, DeploymentSurface, FailureBucket, OutputContract, Principal, UnderstandingConsumer,
    UnderstandingOutcome, UnderstandingRequest,
};
use serde_json::{Value, json};

fn developer() -> Principal {
    Principal::Human("generic-llm-acceptance@example.com".to_string())
}

fn request_with_consumer(
    context_id: ContextId,
    consumer: UnderstandingConsumer,
    input: &str,
) -> UnderstandingRequest {
    UnderstandingRequest {
        context_id,
        consumer,
        deployment: DeploymentSurface::Server,
        input: input.to_string(),
        paid: true,
    }
}

fn request(context_id: ContextId, input: &str) -> UnderstandingRequest {
    request_with_consumer(
        context_id,
        UnderstandingConsumer::ConversationCondensation,
        input,
    )
}

fn request_with_schema_contract(
    context_id: ContextId,
    input: &str,
    schema: &str,
) -> UnderstandingRequest {
    request(context_id, input).with_output_contract(OutputContract::Schema(schema.to_string()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RecordedRequest {
    method: String,
    path: String,
    authorization: Option<String>,
    content_type: Option<String>,
    body: Value,
    raw_body: String,
}

struct StubServer {
    base_url: String,
    received: mpsc::Receiver<Option<RecordedRequest>>,
    handle: thread::JoinHandle<()>,
    join_timeout: Duration,
}

impl StubServer {
    fn join(self) -> Option<RecordedRequest> {
        let recorded = self
            .received
            .recv_timeout(self.join_timeout)
            .unwrap_or(None);
        self.handle.join().expect("stub server thread joins");
        recorded
    }

    fn receive_without_joining(self) -> Option<RecordedRequest> {
        let StubServer {
            received,
            handle,
            join_timeout,
            ..
        } = self;
        let recorded = received.recv_timeout(join_timeout).unwrap_or(None);
        drop(handle);
        recorded
    }
}

struct SequenceStubServer {
    base_url: String,
    received: mpsc::Receiver<Vec<RecordedRequest>>,
    handle: thread::JoinHandle<()>,
    join_timeout: Duration,
}

impl SequenceStubServer {
    fn join(self) -> Vec<RecordedRequest> {
        let recorded = self
            .received
            .recv_timeout(self.join_timeout)
            .unwrap_or_default();
        self.handle.join().expect("stub server thread joins");
        recorded
    }
}

fn stub_server(status: u16, response_body: Value) -> StubServer {
    stub_server_with_delay_and_wait(
        status,
        Some(response_body.to_string()),
        Duration::ZERO,
        Duration::from_secs(2),
    )
}

fn stub_server_with_raw_body(status: u16, response_body: &'static str) -> StubServer {
    stub_server_with_delay_and_wait(
        status,
        Some(response_body.to_string()),
        Duration::ZERO,
        Duration::from_secs(2),
    )
}

fn stub_server_expecting_no_request(status: u16, response_body: Value) -> StubServer {
    stub_server_with_delay_and_wait(
        status,
        Some(response_body.to_string()),
        Duration::ZERO,
        Duration::from_millis(250),
    )
}

fn stub_server_sequence(response_bodies: Vec<Value>) -> SequenceStubServer {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server starts");
    let base_url = crate::common::stub_base_url(&server);
    let expected_requests = response_bodies.len();
    let (sender, received) = mpsc::channel();
    let handle = thread::spawn(move || {
        let mut recorded = Vec::new();
        for response_body in response_bodies {
            let Some(mut request) = server.recv_timeout(Duration::from_secs(2)).ok().flatten()
            else {
                break;
            };
            recorded.push(capture_request(&mut request));
            let content_type =
                tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                    .expect("content-type header");
            let response = tiny_http::Response::from_string(response_body.to_string())
                .with_status_code(200)
                .with_header(content_type);
            let _ = request.respond(response);
        }
        let _ = sender.send(recorded);
    });

    SequenceStubServer {
        base_url,
        received,
        handle,
        join_timeout: Duration::from_secs((expected_requests as u64 * 3).max(3)),
    }
}

fn silent_timeout_stub_server() -> StubServer {
    stub_server_with_delay_and_wait(200, None, Duration::from_secs(30), Duration::from_secs(2))
}

fn stub_server_with_delay_and_wait(
    status: u16,
    response_body: Option<String>,
    response_delay: Duration,
    request_wait: Duration,
) -> StubServer {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("stub server starts");
    let base_url = crate::common::stub_base_url(&server);
    let (sender, received) = mpsc::channel();
    let handle = thread::spawn(move || {
        let Some(mut request) = server.recv_timeout(request_wait).ok().flatten() else {
            let _ = sender.send(None);
            return;
        };
        let recorded = capture_request(&mut request);
        let _ = sender.send(Some(recorded));
        if !response_delay.is_zero() {
            thread::sleep(response_delay);
        }
        if let Some(response_body) = response_body {
            let content_type =
                tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                    .expect("content-type header");
            let response = tiny_http::Response::from_string(response_body)
                .with_status_code(status)
                .with_header(content_type);
            let _ = request.respond(response);
        }
    });

    StubServer {
        base_url,
        received,
        handle,
        join_timeout: request_wait + Duration::from_secs(1),
    }
}

fn header_value(request: &tiny_http::Request, name: &'static str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.as_str().to_string())
}

fn capture_request(request: &mut tiny_http::Request) -> RecordedRequest {
    let method = request.method().as_str().to_string();
    let path = request.url().to_string();
    let authorization = header_value(request, "Authorization");
    let content_type = header_value(request, "Content-Type");
    let mut raw_body = String::new();
    request
        .as_reader()
        .read_to_string(&mut raw_body)
        .expect("request body reads");
    let body = serde_json::from_str(&raw_body).unwrap_or(Value::Null);
    RecordedRequest {
        method,
        path,
        authorization,
        content_type,
        body,
        raw_body,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpAuthScheme {
    Bearer,
    None,
}

struct RuntimeSpec<'a> {
    root: &'a std::path::Path,
    context_id: ContextId,
    model_id: &'a str,
    connection_name: &'a str,
    base_url: &'a str,
    provider: &'a str,
    provider_model: &'a str,
    auth_scheme: HttpAuthScheme,
    secret: Option<&'a str>,
    consent: bool,
}

fn connected_runtime(spec: RuntimeSpec<'_>) -> UnderstandingRuntime<NoopLlmClient> {
    if spec.auth_scheme == HttpAuthScheme::None {
        assert!(
            spec.secret.is_none(),
            "None-auth HTTP connections must not be constructed with a token"
        );
    }
    let principal = developer();
    let mut config = PrincipalAwareConfig::default();
    let mut catalog = ModelConnectionCatalog::open(spec.root);
    catalog.connect_env_var(
        spec.connection_name,
        if spec.secret.is_some() {
            "GENERIC_LLM_API_KEY"
        } else {
            ""
        },
        principal.clone(),
        &mut config,
    );
    if let Some(secret) = spec.secret {
        config.insert_secret(principal.clone(), "GENERIC_LLM_API_KEY", secret);
    }
    if spec.consent {
        catalog.record_consent(spec.connection_name, "test quota");
    }
    catalog.register_hosted_model(
        spec.model_id,
        spec.connection_name,
        spec.base_url,
        spec.provider,
        spec.provider_model,
    );
    catalog.bind_context_model(spec.context_id, spec.model_id);
    UnderstandingRuntime::<NoopLlmClient>::new(catalog, principal, config)
}

fn empty_request() -> RecordedRequest {
    RecordedRequest {
        method: String::new(),
        path: String::new(),
        authorization: None,
        content_type: None,
        body: Value::Null,
        raw_body: String::new(),
    }
}

fn required_request(recorded: Option<RecordedRequest>, context: &str) -> RecordedRequest {
    assert!(
        recorded.is_some(),
        "{context} should have received exactly one outbound HTTP request"
    );
    recorded.unwrap_or_else(empty_request)
}

fn assert_openai_chat_request(
    recorded: &RecordedRequest,
    expected_path: &str,
    expected_model: &str,
    expected_input: &str,
    expected_authorization: Option<&str>,
) {
    assert_eq!(recorded.method, "POST");
    assert_eq!(recorded.path, expected_path);
    assert_eq!(recorded.authorization.as_deref(), expected_authorization);
    assert!(
        recorded
            .content_type
            .as_deref()
            .is_some_and(|value| value.starts_with("application/json")),
        "OpenAI-compatible requests should send JSON content-type, got {:?}",
        recorded.content_type
    );
    assert_eq!(
        recorded.body.get("model").and_then(Value::as_str),
        Some(expected_model)
    );
    let messages = recorded.body.get("messages").and_then(Value::as_array);
    assert!(
        messages.is_some(),
        "OpenAI body should include messages array: {}",
        recorded.raw_body
    );
    let Some(messages) = messages else {
        return;
    };
    assert!(
        messages.iter().any(|message| {
            message.get("role").and_then(Value::as_str) == Some("user")
                && message
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| content.contains(expected_input))
        }),
        "OpenAI body should carry the understanding input in a user message: {}",
        recorded.raw_body
    );
}

fn assert_success_outcome(
    outcome: &UnderstandingOutcome,
    input: &str,
    expected_phrasing: &str,
    expected_connection: &str,
) {
    assert_eq!(outcome.marker, format!("condensed:{input}"));
    assert_eq!(outcome.degraded_marker, None);
    assert_eq!(outcome.failure_bucket, None);
    assert_eq!(outcome.phrasing.as_deref(), Some(expected_phrasing));
    assert!(outcome.model_invoked);
    assert_eq!(
        outcome.resolved_connection.as_deref(),
        Some(expected_connection)
    );
    assert_eq!(outcome.cache_path_consulted, None);
    assert_eq!(outcome.stored_understanding, None);
}

fn assert_degraded(outcome: &UnderstandingOutcome, input: &str, expected_bucket: FailureBucket) {
    assert_eq!(outcome.marker, format!("declarative:{input}"));
    assert_eq!(outcome.degraded_marker.as_deref(), Some(DEGRADED_MARKER));
    assert_eq!(outcome.failure_bucket, Some(expected_bucket));
    assert_eq!(outcome.phrasing, None);
    assert!(
        !outcome.model_invoked,
        "degraded operation must not count as a successfully invoked model"
    );
    assert_eq!(outcome.cache_path_consulted, None);
    assert_eq!(outcome.stored_understanding.as_deref(), Some(input));
}

fn openai_choice(content: &str) -> Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }]
    })
}

fn dead_base_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind dead endpoint probe");
    let addr = listener.local_addr().expect("dead endpoint address");
    drop(listener);
    format!("http://{addr}")
}

fn sorted_object_keys(value: &Value) -> Vec<String> {
    let mut keys = value
        .as_object()
        .map(|object| object.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    keys.sort();
    keys
}

fn json_kind(value: Option<&Value>) -> &'static str {
    match value {
        Some(Value::Null) => "null",
        Some(Value::Bool(_)) => "bool",
        Some(Value::Number(_)) => "number",
        Some(Value::String(_)) => "string",
        Some(Value::Array(_)) => "array",
        Some(Value::Object(_)) => "object",
        None => "missing",
    }
}

fn message_envelope_shape(value: &Value) -> Vec<(Option<String>, Vec<String>, &'static str)> {
    value
        .get("messages")
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .map(|message| {
                    (
                        message
                            .get("role")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        sorted_object_keys(message),
                        json_kind(message.get("content")),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

const PRIMARY_SCHEMA: &str =
    include_str!("../../../cg-cli/tests/fixtures/fixture_output_schema.json");
const ALT_SCHEMA: &str =
    include_str!("../../../cg-cli/tests/fixtures/fixture_output_schema_alt.json");

fn write_structured_output_override(root: &std::path::Path, provider: &str, style: &str) {
    fs::create_dir_all(root.join("cg")).expect("provider override dir");
    fs::write(
        root.join("cg/providers.toml"),
        format!(
            r#"[providers.{provider}]
structured_output = "{style}"
"#,
        ),
    )
    .expect("write structured-output provider override");
}

fn write_http_timeout_override(root: &std::path::Path, provider: &str, timeout_ms: u64) {
    fs::create_dir_all(root.join("cg")).expect("provider override dir");
    fs::write(
        root.join("cg/providers.toml"),
        format!(
            r#"[providers.{provider}]
timeout_ms = {timeout_ms}
"#,
        ),
    )
    .expect("write HTTP timeout provider override");
}

fn local_vllm_runtime(
    root: &std::path::Path,
    context_id: ContextId,
    server: &StubServer,
    connection_name: &str,
) -> UnderstandingRuntime<NoopLlmClient> {
    connected_runtime(RuntimeSpec {
        root,
        context_id,
        model_id: connection_name,
        connection_name,
        base_url: &server.base_url,
        provider: "local-vllm",
        provider_model: "test-model",
        auth_scheme: HttpAuthScheme::None,
        secret: None,
        consent: true,
    })
}

fn schema_property_names(schema: &Value) -> BTreeSet<String> {
    schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| properties.keys().cloned().collect())
        .unwrap_or_default()
}

fn required_property_names(schema: &Value) -> BTreeSet<String> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .map(|required| {
            required
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn strict_response_format_schema(schema: &Value) -> Value {
    let Some(object) = schema.as_object() else {
        return schema.clone();
    };

    let originally_required = required_property_names(schema);
    let required_was_declared = object.contains_key("required");
    let mut transformed = object.clone();
    if let Some(Value::Object(properties)) = object.get("properties") {
        transformed.insert("additionalProperties".to_string(), Value::Bool(false));
        transformed.insert(
            "properties".to_string(),
            Value::Object(
                properties
                    .iter()
                    .map(|(name, property)| {
                        let property_schema = strict_response_format_schema(property);
                        let property_schema =
                            if required_was_declared && originally_required.contains(name) {
                                property_schema
                            } else {
                                schema_with_null(property_schema)
                            };
                        (name.clone(), property_schema)
                    })
                    .collect(),
            ),
        );
        transformed.insert(
            "required".to_string(),
            Value::Array(properties.keys().cloned().map(Value::String).collect()),
        );
    }
    if let Some(items) = object.get("items") {
        transformed.insert("items".to_string(), strict_response_format_schema(items));
    }

    Value::Object(transformed)
}

fn schema_with_null(mut schema: Value) -> Value {
    if schema_allows_null(&schema) {
        return schema;
    }
    let Value::Object(object) = &mut schema else {
        return json!({ "anyOf": [schema, { "type": "null" }] });
    };
    if let Some(type_value) = object.get_mut("type") {
        match type_value {
            Value::String(type_name) => {
                *type_value = Value::Array(vec![
                    Value::String(type_name.clone()),
                    Value::String("null".to_string()),
                ]);
                return schema;
            }
            Value::Array(types) => {
                types.push(Value::String("null".to_string()));
                return schema;
            }
            _ => {}
        }
    }
    if let Some(items) = object.get_mut("anyOf").and_then(Value::as_array_mut) {
        items.push(json!({ "type": "null" }));
        return schema;
    }
    json!({ "anyOf": [schema, { "type": "null" }] })
}

fn schema_allows_null(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    if let Some(type_value) = object.get("type") {
        match type_value {
            Value::String(type_name) => {
                if type_name == "null" {
                    return true;
                }
            }
            Value::Array(types) => {
                if types.iter().any(|item| item.as_str() == Some("null")) {
                    return true;
                }
            }
            _ => {}
        }
    }
    object
        .get("anyOf")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(schema_allows_null)
}

fn normalize_required_sets(schema: &Value) -> Value {
    match schema {
        Value::Object(object) => {
            let mut normalized = serde_json::Map::new();
            for (key, value) in object {
                if key == "required" {
                    let mut required = value.as_array().cloned().unwrap_or_default();
                    required.sort_by_key(|item| item.as_str().unwrap_or_default().to_owned());
                    normalized.insert(key.clone(), Value::Array(required));
                } else {
                    normalized.insert(key.clone(), normalize_required_sets(value));
                }
            }
            Value::Object(normalized)
        }
        Value::Array(items) => Value::Array(items.iter().map(normalize_required_sets).collect()),
        _ => schema.clone(),
    }
}

fn response_format_schema(body: &Value) -> Option<&Value> {
    body.pointer("/response_format/json_schema/schema")
}

fn assert_strict_response_format_schema(body: &Value, fixture_schema: &Value) {
    assert_eq!(
        body.pointer("/response_format/type")
            .and_then(Value::as_str),
        Some("json_schema"),
        "structured-output response_format.type must request json_schema; body={body}"
    );
    assert_eq!(
        body.pointer("/response_format/json_schema/strict")
            .and_then(Value::as_bool),
        Some(true),
        "structured-output response_format must request strict schema enforcement; body={body}"
    );
    let Some(wire_schema) = response_format_schema(body) else {
        assert!(
            body.pointer("/response_format/json_schema/schema")
                .is_some(),
            "structured-output response_format json_schema.schema is absent from the outbound body: {body}"
        );
        return;
    };
    assert_eq!(
        wire_schema.get("type").and_then(Value::as_str),
        Some("object"),
        "wire schema must be a strict root object: {wire_schema}"
    );
    assert_eq!(
        wire_schema
            .get("additionalProperties")
            .and_then(Value::as_bool),
        Some(false),
        "wire schema must close additional properties: {wire_schema}"
    );
    assert_eq!(
        schema_property_names(wire_schema),
        schema_property_names(fixture_schema),
        "wire schema property names must track the attached fixture schema"
    );
    assert_eq!(
        required_property_names(wire_schema),
        schema_property_names(fixture_schema),
        "strict wire schema must require every attached fixture property"
    );
    let expected_schema = strict_response_format_schema(fixture_schema);
    assert_eq!(
        normalize_required_sets(wire_schema),
        normalize_required_sets(&expected_schema),
        "wire schema must equal the full strict transform of the attached fixture schema, including nested bodies: {wire_schema}"
    );
}

fn assert_schema_contract_request(
    style: &str,
    schema: &str,
    input: &str,
    connection_name: &str,
) -> RecordedRequest {
    let temp = tempfile::tempdir().expect("tempdir");
    write_structured_output_override(temp.path(), "local-vllm", style);
    let context_id = ContextId::new_v7();
    let server = stub_server(200, openai_choice("structured-output wire probe"));
    let mut runtime = local_vllm_runtime(temp.path(), context_id, &server, connection_name);
    let _ = runtime
        .understand(request_with_schema_contract(context_id, input, schema))
        .expect("understanding should complete");
    required_request(server.join(), input)
}

#[test]
fn schema_contract_rides_the_wire_as_response_format_json_schema() {
    let recorded = assert_schema_contract_request(
        "response_format_schema",
        PRIMARY_SCHEMA,
        "primary schema contract",
        "response-format-schema-contract",
    );
    let fixture_schema =
        serde_json::from_str::<Value>(PRIMARY_SCHEMA).expect("primary schema JSON");

    assert_openai_chat_request(
        &recorded,
        "/chat/completions",
        "test-model",
        "primary schema contract",
        None,
    );
    assert_strict_response_format_schema(&recorded.body, &fixture_schema);
}

#[test]
fn guided_json_style_rides_the_wire_under_its_configured_field_name() {
    let recorded = assert_schema_contract_request(
        "guided_json",
        PRIMARY_SCHEMA,
        "guided json schema contract",
        "guided-json-schema-contract",
    );
    let fixture_schema =
        serde_json::from_str::<Value>(PRIMARY_SCHEMA).expect("primary schema JSON");

    assert!(
        recorded.body.get("response_format").is_none(),
        "guided_json endpoints must not receive response_format: {}",
        recorded.raw_body
    );
    assert_eq!(
        recorded.body.get("guided_json"),
        Some(&fixture_schema),
        "guided_json endpoints must receive the attached schema under the configured field name"
    );
}

#[test]
fn wire_schema_tracks_the_attached_contract_not_a_constant() {
    let recorded = assert_schema_contract_request(
        "response_format_schema",
        ALT_SCHEMA,
        "alternate schema contract",
        "response-format-alt-contract",
    );
    let primary_schema =
        serde_json::from_str::<Value>(PRIMARY_SCHEMA).expect("primary schema JSON");
    let alt_schema = serde_json::from_str::<Value>(ALT_SCHEMA).expect("alt schema JSON");

    let Some(wire_schema) = response_format_schema(&recorded.body) else {
        assert!(
            recorded
                .body
                .pointer("/response_format/json_schema/schema")
                .is_some(),
            "structured-output response_format json_schema.schema is absent from the outbound body: {}",
            recorded.raw_body
        );
        return;
    };
    assert_strict_response_format_schema(&recorded.body, &alt_schema);
    assert_eq!(
        wire_schema.pointer("/properties/confidence_note/type"),
        Some(&json!(["string", "null"])),
        "optional top-level fields must be nullable when strict response_format requires every property: {wire_schema}"
    );
    assert_eq!(
        wire_schema.pointer("/properties/memory/properties/source_paths/type"),
        Some(&json!(["array", "null"])),
        "optional nested fields must be nullable when strict response_format requires every property: {wire_schema}"
    );
    assert_eq!(
        schema_property_names(wire_schema),
        schema_property_names(&alt_schema),
        "wire schema must use the attached ALT fixture property set"
    );
    assert_ne!(
        normalize_required_sets(wire_schema),
        normalize_required_sets(&strict_response_format_schema(&primary_schema)),
        "wire schema must not echo the primary fixture as a constant"
    );
}

#[test]
fn unconstrained_capture_sends_no_structured_output_field() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let server = stub_server(200, openai_choice("unconstrained wire probe"));
    let mut runtime =
        local_vllm_runtime(temp.path(), context_id, &server, "unconstrained-contract");
    let _ = runtime
        .understand(request(context_id, "unconstrained request"))
        .expect("understanding should complete");
    let recorded = required_request(server.join(), "unconstrained request");
    let keys = recorded
        .body
        .as_object()
        .expect("request body object")
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();

    assert_eq!(
        keys,
        BTreeSet::from([
            "max_tokens".to_string(),
            "messages".to_string(),
            "model".to_string(),
            "temperature".to_string(),
        ]),
        "unconstrained capture must keep today's four-field OpenAI-compatible body"
    );
    assert!(recorded.body.get("response_format").is_none());
    assert!(recorded.body.get("guided_json").is_none());
    assert!(recorded.body.get("grammar").is_none());
}

#[test]
fn chat_completions_round_trip_returns_model_text_into_model_reply_phrasing() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let input = "condense this policy change";
    let provider_text = "Provider phrasing from OpenAI-compatible chat.";
    let server = stub_server(200, openai_choice(provider_text));
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "generic-openai-round-trip",
        connection_name: "openai-round-trip",
        base_url: &server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-test-generic"),
        consent: true,
    });

    let outcome = runtime
        .understand(request(context_id, input))
        .expect("understanding should complete");
    let recorded = required_request(server.join(), "round trip");

    assert_openai_chat_request(
        &recorded,
        "/chat/completions",
        "gpt-4o-mini",
        input,
        Some("Bearer sk-test-generic"),
    );
    assert_success_outcome(&outcome, input, provider_text, "openai-round-trip");
}

#[test]
fn outbound_request_model_field_equals_connection_provider_model() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let input = "use the user selected model id";
    let server = stub_server(200, openai_choice("selected model phrasing"));
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "selected-understanding-model",
        connection_name: "selected-openai",
        base_url: &server.base_url,
        provider: "openai-compatible",
        provider_model: "user-declared-model-id",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-selected"),
        consent: true,
    });

    let _ = runtime
        .understand(request(context_id, input))
        .expect("understanding should complete");
    let recorded = required_request(server.join(), "model field fidelity");

    assert_openai_chat_request(
        &recorded,
        "/chat/completions",
        "user-declared-model-id",
        input,
        Some("Bearer sk-selected"),
    );
}

#[test]
fn bearer_scheme_sends_authorization_header_none_scheme_sends_no_authorization_header_and_empty_token_is_rejected()
 {
    let temp = tempfile::tempdir().expect("tempdir");
    let bearer_context = ContextId::new_v7();
    let bearer_input = "bearer auth request";
    let bearer_server = stub_server(200, openai_choice("bearer phrasing"));
    let mut bearer_runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id: bearer_context,
        model_id: "bearer-model",
        connection_name: "bearer-connection",
        base_url: &bearer_server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-bearer"),
        consent: true,
    });

    let _ = bearer_runtime
        .understand(request(bearer_context, bearer_input))
        .expect("bearer understanding should complete");
    let bearer_recorded = required_request(bearer_server.join(), "bearer auth");
    assert_openai_chat_request(
        &bearer_recorded,
        "/chat/completions",
        "gpt-4o-mini",
        bearer_input,
        Some("Bearer sk-bearer"),
    );

    let none_context = ContextId::new_v7();
    let none_input = "none auth request";
    let none_server = stub_server(200, openai_choice("none auth phrasing"));
    let mut none_runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id: none_context,
        model_id: "none-auth-model",
        connection_name: "none-auth-connection",
        base_url: &none_server.base_url,
        provider: "local-vllm",
        provider_model: "llama-local",
        auth_scheme: HttpAuthScheme::None,
        secret: None,
        consent: true,
    });

    let _ = none_runtime
        .understand(request(none_context, none_input))
        .expect("none-auth understanding should complete");
    let none_recorded = required_request(none_server.join(), "none auth");
    assert_openai_chat_request(
        &none_recorded,
        "/chat/completions",
        "llama-local",
        none_input,
        None,
    );

    let empty_context = ContextId::new_v7();
    let empty_input = "empty bearer token must not leave process";
    let empty_server = stub_server_expecting_no_request(200, openai_choice("must not be reached"));
    let mut empty_runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id: empty_context,
        model_id: "empty-token-model",
        connection_name: "empty-token-connection",
        base_url: &empty_server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some(""),
        consent: true,
    });
    let empty = empty_runtime
        .understand(request(empty_context, empty_input))
        .expect("empty-token understanding should complete");
    assert_eq!(
        empty_server.join(),
        None,
        "empty Bearer token must be rejected before an HTTP request is sent"
    );
    assert_degraded(&empty, empty_input, FailureBucket::Unauthorized);

    let whitespace_context = ContextId::new_v7();
    let whitespace_input = "whitespace bearer token must not leave process";
    let whitespace_server =
        stub_server_expecting_no_request(200, openai_choice("must not be reached"));
    let mut whitespace_runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id: whitespace_context,
        model_id: "whitespace-token-model",
        connection_name: "whitespace-token-connection",
        base_url: &whitespace_server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some(" \t  "),
        consent: true,
    });
    let whitespace = whitespace_runtime
        .understand(request(whitespace_context, whitespace_input))
        .expect("whitespace-token understanding should complete");
    assert_eq!(
        whitespace_server.join(),
        None,
        "whitespace-only Bearer token must be rejected before an HTTP request is sent"
    );
    assert_degraded(&whitespace, whitespace_input, FailureBucket::Unauthorized);
}

#[test]
fn base_url_is_used_exactly_as_given_no_v1_auto_appended_and_trailing_slash_not_doubled() {
    let temp = tempfile::tempdir().expect("tempdir");
    let input = "preserve the operator base url";
    let cases = [
        (
            "nested-openai-v1",
            "/openai/v1",
            "/openai/v1/chat/completions",
        ),
        ("plain-v1", "/v1", "/v1/chat/completions"),
        ("trailing-v1", "/v1/", "/v1/chat/completions"),
    ];

    for (label, suffix, expected_path) in cases {
        let context_id = ContextId::new_v7();
        let server = stub_server(200, openai_choice("base url phrasing"));
        let base_url = format!("{}{}", server.base_url, suffix);
        let model_id = format!("base-url-model-{label}");
        let connection_name = format!("base-url-connection-{label}");
        let mut runtime = connected_runtime(RuntimeSpec {
            root: temp.path(),
            context_id,
            model_id: &model_id,
            connection_name: &connection_name,
            base_url: &base_url,
            provider: "openai-compatible",
            provider_model: "qwen2.5-coder",
            auth_scheme: HttpAuthScheme::Bearer,
            secret: Some("sk-base-url"),
            consent: true,
        });

        let _ = runtime
            .understand(request(context_id, input))
            .expect("understanding should complete");
        let recorded = required_request(server.join(), label);

        assert_openai_chat_request(
            &recorded,
            expected_path,
            "qwen2.5-coder",
            input,
            Some("Bearer sk-base-url"),
        );
    }
}

#[test]
fn reply_text_is_extracted_via_json_pointer_choices_0_message_content() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let input = "extract only the assistant content";
    let provider_text = "Only this message content should become phrasing.";
    let server = stub_server(
        200,
        json!({
            "id": "chatcmpl-extra",
            "model": "provider-side-model-name-that-must-not-leak",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": provider_text
                    },
                    "finish_reason": "stop",
                    "unexpected_sibling": {
                        "content": "not this content"
                    }
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": "not the second choice"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19
            },
            "content": "not the top-level content"
        }),
    );
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "json-pointer-model",
        connection_name: "json-pointer-connection",
        base_url: &server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4.1-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-json-pointer"),
        consent: true,
    });

    let outcome = runtime
        .understand(request(context_id, input))
        .expect("understanding should complete");
    let _ = required_request(server.join(), "json pointer extraction");

    assert_success_outcome(&outcome, input, provider_text, "json-pointer-connection");
    let phrasing = outcome.phrasing.as_deref().unwrap_or_default();
    assert!(!phrasing.contains("unexpected_sibling"));
    assert!(!phrasing.contains("not the top-level content"));
    assert!(!phrasing.contains("provider-side-model-name"));
    assert!(!phrasing.contains("not the second choice"));
    assert!(!phrasing.contains("total_tokens"));
}

#[test]
fn reply_format_negative_cases_surface_typed_malformed_never_panic() {
    let temp = tempfile::tempdir().expect("tempdir");
    let cases = [
        ("not-json", "{not-json".to_string()),
        (
            "missing-choices",
            json!({"id": "chatcmpl-missing"}).to_string(),
        ),
        ("empty-choices", json!({"choices": []}).to_string()),
        (
            "null-content",
            json!({"choices": [{"message": {"content": Value::Null}}]}).to_string(),
        ),
        (
            "empty-content",
            json!({"choices": [{"message": {"content": ""}}]}).to_string(),
        ),
    ];

    for (case, body) in cases {
        let context_id = ContextId::new_v7();
        let input = format!("retain this input after malformed reply {case}");
        let server = stub_server_with_raw_body(200, Box::leak(body.into_boxed_str()));
        let mut runtime = connected_runtime(RuntimeSpec {
            root: temp.path(),
            context_id,
            model_id: &format!("malformed-model-{case}"),
            connection_name: &format!("malformed-connection-{case}"),
            base_url: &server.base_url,
            provider: "openai-compatible",
            provider_model: "gpt-4o-mini",
            auth_scheme: HttpAuthScheme::Bearer,
            secret: Some("sk-malformed"),
            consent: true,
        });

        let outcome = runtime
            .understand(request(context_id, &input))
            .expect("malformed reply should degrade without panicking");
        let _ = required_request(server.join(), case);

        assert_degraded(&outcome, &input, FailureBucket::ReplyMalformed);
        let diagnostics = runtime.catalog().persistence_errors();
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("malformed provider reply")),
            "malformed 200-OK replies must flow through ModelReply.malformed before degradation; diagnostics={diagnostics:?}"
        );
    }
}

#[test]
fn http_4xx_degrades_to_typed_unauthorized_bucket_and_retains_uncondensed_input() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let input = "protect this user text on auth failure";
    let server = stub_server(
        401,
        json!({
            "error": {
                "message": "bad key",
                "type": "invalid_request_error"
            }
        }),
    );
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "http-4xx-model",
        connection_name: "http-4xx-connection",
        base_url: &server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-expired"),
        consent: true,
    });

    let outcome = runtime
        .understand(request(context_id, input))
        .expect("4xx understanding should complete");
    let recorded = required_request(server.join(), "http 4xx");

    assert_openai_chat_request(
        &recorded,
        "/chat/completions",
        "gpt-4o-mini",
        input,
        Some("Bearer sk-expired"),
    );
    assert_degraded(&outcome, input, FailureBucket::Unauthorized);
}

#[test]
fn http_5xx_degrades_to_typed_unreachable_bucket_and_retains_uncondensed_input() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let input = "retain this input after server failure";
    let server = stub_server(
        503,
        json!({
            "error": {
                "message": "provider overloaded",
                "type": "server_error"
            }
        }),
    );
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "http-5xx-model",
        connection_name: "http-5xx-connection",
        base_url: &server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-server-error"),
        consent: true,
    });

    let outcome = runtime
        .understand(request(context_id, input))
        .expect("5xx understanding should complete");
    let recorded = required_request(server.join(), "http 5xx");

    assert_openai_chat_request(
        &recorded,
        "/chat/completions",
        "gpt-4o-mini",
        input,
        Some("Bearer sk-server-error"),
    );
    assert_degraded(&outcome, input, FailureBucket::Unreachable);
}

#[test]
fn connection_refused_degrades_to_typed_unreachable_bucket_and_retains_uncondensed_input() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let input = "retain this input after connection refused";
    let dead_url = dead_base_url();
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "connection-refused-model",
        connection_name: "connection-refused",
        base_url: &dead_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-refused"),
        consent: true,
    });

    let outcome = runtime
        .understand(request(context_id, input))
        .expect("connection-refused understanding should complete");

    assert_degraded(&outcome, input, FailureBucket::Unreachable);
}

#[test]
fn timeout_hang_degrades_to_typed_unreachable_bucket_within_bounded_time_agent_keeps_running() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let input = "retain this input after timeout";
    let server = silent_timeout_stub_server();
    write_http_timeout_override(temp.path(), "openai-compatible", 1_000);
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "timeout-model",
        connection_name: "timeout-connection",
        base_url: &server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-timeout"),
        consent: true,
    });

    let started = Instant::now();
    let outcome = runtime
        .understand(request(context_id, input))
        .expect("timeout understanding should complete");
    let elapsed = started.elapsed();
    let recorded = required_request(server.receive_without_joining(), "timeout");

    assert_openai_chat_request(
        &recorded,
        "/chat/completions",
        "gpt-4o-mini",
        input,
        Some("Bearer sk-timeout"),
    );
    assert!(
        elapsed < Duration::from_secs(12),
        "timeout must stay within the configured hosted timeout budget; elapsed={elapsed:?}"
    );
    assert_degraded(&outcome, input, FailureBucket::Unreachable);
}

#[test]
fn consent_not_recorded_makes_no_outbound_request_to_the_stub() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let input = "must not spend without consent";
    let server = stub_server_expecting_no_request(200, openai_choice("must not be reached"));
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "no-consent-model",
        connection_name: "no-consent-connection",
        base_url: &server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-blocked"),
        consent: false,
    });

    let outcome = runtime
        .understand(request(context_id, input))
        .expect("missing consent should degrade without panicking");

    assert_eq!(
        server.join(),
        None,
        "paid endpoint calls must not reach the network before consent is recorded"
    );
    assert_degraded(&outcome, input, FailureBucket::Unauthorized);
}

#[test]
fn consent_recorded_drives_exactly_one_outbound_request_to_the_stub() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let input = "spend after consent";
    let server = stub_server(200, openai_choice("consented sentinel"));
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "consented-model",
        connection_name: "consented-connection",
        base_url: &server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-consented"),
        consent: true,
    });

    let _ = runtime
        .understand(request(context_id, input))
        .expect("consented call returns an outcome");
    let recorded = required_request(server.join(), "consented request");

    assert_openai_chat_request(
        &recorded,
        "/chat/completions",
        "gpt-4o-mini",
        input,
        Some("Bearer sk-consented"),
    );
}

#[test]
fn selftest_and_condensation_drive_the_identical_stub_request_shape() {
    let temp = tempfile::tempdir().expect("tempdir");
    let condensation_context = ContextId::new_v7();
    let server = stub_server_sequence(vec![
        openai_choice("condensation"),
        openai_choice("selftest"),
    ]);
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id: condensation_context,
        model_id: "shared-wire-model",
        connection_name: "shared-wire-connection",
        base_url: &server.base_url,
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-shared"),
        consent: true,
    });

    let condensation_outcome = runtime
        .understand(request(
            condensation_context,
            "condense via generic endpoint",
        ))
        .expect("condensation should complete");
    let proof = runtime.prove_host_session_connection(
        "shared-wire-connection",
        "cg selftest proof prompt",
        false,
    );
    let requests = server.join();
    assert_eq!(
        requests.len(),
        2,
        "condensation and selftest must both drive the same named endpoint"
    );
    let condensation = &requests[0];
    let selftest = &requests[1];
    let proof =
        proof.expect("generic BYOK selftest should complete through the runtime proof seam");

    assert_success_outcome(
        &condensation_outcome,
        "condense via generic endpoint",
        "condensation",
        "shared-wire-connection",
    );
    assert_eq!(proof.connection_name, "shared-wire-connection");
    assert_eq!(proof.provider, "openai-compatible");
    assert_eq!(proof.reply, "selftest");
    assert_openai_chat_request(
        condensation,
        "/chat/completions",
        "gpt-4o-mini",
        "condense via generic endpoint",
        Some("Bearer sk-shared"),
    );
    assert_openai_chat_request(
        selftest,
        "/chat/completions",
        "gpt-4o-mini",
        "cg selftest proof prompt",
        Some("Bearer sk-shared"),
    );
    assert_eq!(condensation.method, selftest.method);
    assert_eq!(condensation.path, selftest.path);
    assert_eq!(
        condensation.body.get("model"),
        selftest.body.get("model"),
        "selftest and condensation must share the same OpenAI-compatible request model field"
    );
    assert_eq!(
        condensation.authorization, selftest.authorization,
        "selftest and condensation must use the same credential source"
    );
    assert_eq!(
        sorted_object_keys(&condensation.body),
        sorted_object_keys(&selftest.body),
        "selftest and condensation must keep the same top-level Chat Completions body shape"
    );
    assert_eq!(
        message_envelope_shape(&condensation.body),
        message_envelope_shape(&selftest.body),
        "selftest and condensation must use the same chat message envelope shape, ignoring only prompt text"
    );
}

#[test]
fn honest_connection_does_not_advertise_a_capability_it_cannot_carry() {
    let temp = tempfile::tempdir().expect("tempdir");
    let context_id = ContextId::new_v7();
    let runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "honest-status-model",
        connection_name: "honest-status-connection",
        base_url: "http://127.0.0.1:65535",
        provider: "openai-compatible",
        provider_model: "gpt-4o-mini",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-status"),
        consent: true,
    });

    let status = runtime.catalog().status_for_context(context_id);
    let status_json = serde_json::to_value(&status).expect("status serializes");
    let condition = status_json.get("condition").and_then(Value::as_str);

    assert_eq!(
        status_json.get("state").and_then(Value::as_str),
        Some("ready"),
        "honest health must report the connected status state it is judging: {status_json}"
    );
    assert_eq!(
        status_json.get("provider").and_then(Value::as_str),
        Some("openai-compatible"),
        "honest health must name the exact provider row it is judging: {status_json}"
    );
    assert_eq!(
        status_json.get("endpoint").and_then(Value::as_str),
        Some("http://127.0.0.1:65535"),
        "honest health must name the exact endpoint it is judging: {status_json}"
    );
    assert_eq!(
        status_json.get("provider_model").and_then(Value::as_str),
        Some("gpt-4o-mini"),
        "honest health must name the exact provider model it is judging: {status_json}"
    );
    assert!(
        status_json.get("capabilities").is_none(),
        "generic OpenAI-compatible status must not invent a public capability list; the consuming ingestion plan owns capability requirements: {status_json}"
    );
    assert_eq!(
        condition, None,
        "a plain prompt-in/text-out connection must stay ready without claiming a capability-limited condition: {status_json}"
    );
}

#[test]
fn provider_specs_ship_required_http_rows_and_accept_user_http_override_as_data() {
    let temp = tempfile::tempdir().expect("provider spec tempdir");
    let specs = load_provider_specs(temp.path(), &[]);
    for (provider, auth_scheme) in [
        ("openai", ProviderAuthScheme::Bearer),
        ("nebius", ProviderAuthScheme::Bearer),
        ("openrouter", ProviderAuthScheme::Bearer),
        ("local-vllm", ProviderAuthScheme::None),
    ] {
        let spec = specs.providers.get(provider);
        assert!(
            spec.is_some(),
            "embedded provider specs must include the required generic HTTP row `{provider}`"
        );
        let Some(spec) = spec else {
            continue;
        };
        assert_eq!(
            spec.transport,
            ProviderTransport::Http,
            "{provider} must be declared as HTTP transport, not hard-coded subprocess behavior"
        );
        assert_eq!(
            spec.auth_scheme,
            Some(auth_scheme),
            "{provider} must declare its Bearer vs None auth scheme as data"
        );
        assert_eq!(
            spec.dialect,
            Some(ProviderDialect::OpenaiChatCompletions),
            "{provider} must declare the OpenAI Chat Completions dialect as data"
        );
        assert_eq!(
            spec.reply_path, "/choices/0/message/content",
            "{provider} must declare the assistant-content JSON pointer"
        );
        assert!(
            spec.base_url.as_deref().is_some_and(|url| !url.is_empty()),
            "{provider} must declare a base URL row value instead of relying on hard-coded defaults"
        );
    }

    fs::create_dir_all(temp.path().join("cg")).expect("provider override dir");
    fs::write(
        temp.path().join("cg/providers.toml"),
        r#"[providers.maya-openai-compatible]
transport = "http"
auth_scheme = "bearer"
base_url = "https://maya.example.invalid/openai/v1"
dialect = "openai-chat-completions"
reply_path = "/choices/0/message/content"
timeout_ms = 12000
consent_required = true
"#,
    )
    .expect("write HTTP provider override");
    let override_specs = try_load_provider_specs(temp.path(), &[]);
    assert!(
        override_specs.is_ok(),
        "user HTTP provider overrides must parse as data without subprocess-only fields: {override_specs:?}"
    );
    let custom = override_specs
        .expect("user HTTP provider overrides should parse")
        .providers
        .get("maya-openai-compatible")
        .cloned()
        .expect("HTTP-only user provider override must be retained as provider data");
    assert_eq!(custom.transport, ProviderTransport::Http);
    assert_eq!(custom.auth_scheme, Some(ProviderAuthScheme::Bearer));
    assert_eq!(
        custom.base_url.as_deref(),
        Some("https://maya.example.invalid/openai/v1")
    );
    assert_eq!(custom.dialect, Some(ProviderDialect::OpenaiChatCompletions));
    assert_eq!(custom.reply_path, "/choices/0/message/content");
}

#[test]
fn user_http_provider_override_uses_custom_row_metadata_and_connected_endpoint() {
    let temp = tempfile::tempdir().expect("provider dispatch tempdir");
    let provider_text = "Maya custom OpenAI-compatible row responded.";
    let server = stub_server(
        200,
        json!({
            "id": "chatcmpl-maya-custom",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "wrong default pointer content"
                    }
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": provider_text
                    }
                }
            ]
        }),
    );
    let row_base_url = format!("{}/maya-row/v42", server.base_url);
    let connected_base_url = format!("{}/connected-endpoint/v1", server.base_url);
    fs::create_dir_all(temp.path().join("cg")).expect("provider override dir");
    fs::write(
        temp.path().join("cg/providers.toml"),
        format!(
            r#"[providers.maya-openai-compatible]
transport = "http"
auth_scheme = "bearer"
base_url = "{}"
dialect = "openai-chat-completions"
reply_path = "/choices/1/message/content"
timeout_ms = 5000
consent_required = true
"#,
            row_base_url
        ),
    )
    .expect("write custom HTTP provider override");
    let specs_result = try_load_provider_specs(temp.path(), &[]);
    assert!(
        specs_result.is_ok(),
        "custom OpenAI-compatible provider row should parse as provider data: {specs_result:?}"
    );
    let Ok(specs) = specs_result else {
        return;
    };
    let custom = specs.providers.get("maya-openai-compatible");
    assert!(
        custom.is_some(),
        "custom provider row must be loaded from the HTTP-only override"
    );
    let Some(custom) = custom else {
        return;
    };
    assert_eq!(custom.transport, ProviderTransport::Http);
    assert_eq!(custom.auth_scheme, Some(ProviderAuthScheme::Bearer));
    assert_eq!(
        custom.base_url.as_deref(),
        Some(row_base_url.as_str()),
        "dispatch must have a provider-row endpoint to use, not a hard-coded built-in table"
    );
    assert_eq!(custom.reply_path, "/choices/1/message/content");

    let context_id = ContextId::new_v7();
    let mut runtime = connected_runtime(RuntimeSpec {
        root: temp.path(),
        context_id,
        model_id: "maya-custom-provider-model",
        connection_name: "maya-custom-provider-connection",
        base_url: &connected_base_url,
        provider: "maya-openai-compatible",
        provider_model: "maya-provider-model",
        auth_scheme: HttpAuthScheme::Bearer,
        secret: Some("sk-maya-custom"),
        consent: true,
    });
    let input = "custom provider dispatch";
    let outcome = runtime
        .understand(request(context_id, input))
        .expect("custom provider dispatch returns an outcome");
    let recorded = required_request(
        server.join(),
        "custom provider row metadata must dispatch through the connected endpoint",
    );
    assert_openai_chat_request(
        &recorded,
        "/connected-endpoint/v1/chat/completions",
        "maya-provider-model",
        input,
        Some("Bearer sk-maya-custom"),
    );
    assert_success_outcome(
        &outcome,
        input,
        provider_text,
        "maya-custom-provider-connection",
    );
}
