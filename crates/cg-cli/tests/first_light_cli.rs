mod support;

use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use context_graph::context_graph_pending_push_change_count;
use context_graph::{
    ContextId, CreateDecision, CreateIntention, DecisionId, DecisionStatus, DiscoveryFilter,
    EntityId, IntentionId, IntentionOrigin, IntentionStatus, Store, StoreConfig,
};
use contextdb_core::Value as DbValue;
use contextdb_engine::Database;
use contextdb_server::transport::iroh::IrohServer;
use contextdb_server::{SyncClient, SyncServer};
use serde_json::Value;
use support::{CliHarness, CommandOutput, CompletionBehavior, LoginState, Provider, cg_bin};

const AUTH_INITIAL: &str = "Acme needs password-less login to ship. We found the auth module file at src/auth.py and decided the auth module stays at src/auth.py.";
const AUTH_WITH_JWT: &str = "Acme needs password-less login to ship. We found the auth module file at src/auth.py and decided the auth module stays at src/auth.py. Use JWT with 15-min access tokens.";
const AUTH_RENAME: &str = "Following Tuesday, the auth module file moved from src/auth.py to src/authentication.py. Keep password-less login served by the auth module at src/authentication.py.";
const AUTH_CURRENT_WITH_JWT: &str = "Acme password-less login is now served by the auth module at src/authentication.py. Use JWT with 15-min access tokens.";
const CI_PIPELINE_SESSION: &str = "The delivery team needs a CI pipeline. Run CI on every pull request, block merges when CI fails, and include unit tests, clippy, and release build in .github/workflows/ci.yml.";
const CLIENT_A_SOX_SESSION: &str = "Client A must keep SOX deploy approvals. The publish pipeline is deploy.yml; require SOX approval before promote.";
const CLIENT_B_PUBLISH_SESSION: &str = "Client B must avoid the duplicate-publish race. The publish pipeline is deploy.yml; hold 10 min before promote.";

fn combined(output: &CommandOutput) -> String {
    format!("{}\n{}", output.stdout, output.stderr)
}

fn assert_record_contains(output: &CommandOutput, expected: &str) {
    let text = combined(output);
    assert_eq!(output.code, 0, "{text}");
    assert!(text.contains(expected), "{text}");
    let json = output_json(output).expect("record output json");
    assert_eq!(
        json.pointer("/captured/goals").and_then(Value::as_u64),
        Some(1),
        "{text}"
    );
    assert_eq!(
        json.pointer("/captured/decisions").and_then(Value::as_u64),
        Some(1),
        "{text}"
    );
}

fn assert_record_observation_count(output: &CommandOutput, expected: u64) {
    let text = combined(output);
    let json = output_json(output).expect("record output json");
    assert_eq!(
        json.pointer("/captured/observations")
            .and_then(Value::as_u64),
        Some(expected),
        "{text}"
    );
}

fn assert_record_text(output: &CommandOutput, expected: &str) {
    let text = combined(output);
    assert_eq!(output.code, 0, "{text}");
    assert!(text.contains(expected), "{text}");
}

fn extraction_harness() -> CliHarness {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        "developer@example.test",
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let login = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    assert_eq!(
        login.code,
        0,
        "fake host login failed: {}",
        combined(&login)
    );
    let selftest = harness.run(&["selftest"]);
    assert_eq!(
        selftest.code,
        0,
        "fake host selftest failed: {}",
        combined(&selftest)
    );
    harness
        .providers
        .set_behavior(Provider::Claude, CompletionBehavior::ExtractHarvest);
    harness
}

fn run_in_dir(harness: &CliHarness, cwd: &std::path::Path, args: &[&str]) -> CommandOutput {
    let output = Command::new(cg_bin())
        .args(args)
        .current_dir(cwd)
        .env("CG_DATA_ROOT", harness.data_root.path())
        .env("CG_USE_TEST_LEXICAL_EMBEDDER", "1")
        .env("HOME", harness.data_root.path())
        .env("PATH", harness.providers.path_env())
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .stdin(Stdio::null())
        .output()
        .expect("run cg");
    CommandOutput {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn run_in_dir_with_stdin(
    harness: &CliHarness,
    cwd: &std::path::Path,
    args: &[&str],
    stdin: &str,
) -> CommandOutput {
    let mut child = Command::new(cg_bin())
        .args(args)
        .current_dir(cwd)
        .env("CG_DATA_ROOT", harness.data_root.path())
        .env("CG_USE_TEST_LEXICAL_EMBEDDER", "1")
        .env("HOME", harness.data_root.path())
        .env("PATH", harness.providers.path_env())
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run cg");
    child
        .stdin
        .as_mut()
        .expect("stdin pipe")
        .write_all(stdin.as_bytes())
        .expect("write stdin");
    let output = child.wait_with_output().expect("wait for cg");
    CommandOutput {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn output_json(output: &CommandOutput) -> Option<Value> {
    serde_json::from_str(&output.stdout).ok()
}

fn spawn_one_shot_chat_completion_server(
    assistant_content: Value,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind hosted provider fixture");
    let address = listener.local_addr().expect("hosted provider address");
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept hosted provider request");
        let mut reader = BufReader::new(stream.try_clone().expect("clone provider stream"));
        let mut first_line = String::new();
        reader
            .read_line(&mut first_line)
            .expect("read provider request line");
        let mut request = first_line;
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
            request.push_str(&line);
        }
        if content_length > 0 {
            let mut request_body = vec![0; content_length];
            reader
                .read_exact(&mut request_body)
                .expect("read provider request body");
            request.push_str(&String::from_utf8_lossy(&request_body));
        }
        tx.send(request).expect("send provider request");
        let body = serde_json::json!({
            "id": "chatcmpl-cli-hosted-usage",
            "object": "chat.completion",
            "created": 1766019000,
            "model": "local-vllm/test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_content.to_string(),
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream
            .write_all(response.as_bytes())
            .expect("write provider response");
    });
    (format!("http://{address}/v1"), rx)
}

#[derive(Clone)]
struct ChatCompletionStubResponse {
    status: u16,
    body: Value,
}

impl ChatCompletionStubResponse {
    fn ok_content(content: impl Into<String>, prompt_tokens: u64, completion_tokens: u64) -> Self {
        Self {
            status: 200,
            body: chat_completion_body(
                serde_json::json!({
                    "role": "assistant",
                    "content": content.into(),
                }),
                "stop",
                prompt_tokens,
                completion_tokens,
            ),
        }
    }

    fn raw(status: u16, body: Value) -> Self {
        Self { status, body }
    }
}

struct HostedCapture {
    harness: CliHarness,
    output: CommandOutput,
    requests: Vec<String>,
}

fn chat_completion_body(
    message: Value,
    finish_reason: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Value {
    serde_json::json!({
        "id": "chatcmpl-cli-sequence",
        "object": "chat.completion",
        "created": 1766019000,
        "model": "local-vllm/test-model",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })
}

fn spawn_chat_completion_sequence_server(
    responses: Vec<ChatCompletionStubResponse>,
) -> (String, mpsc::Receiver<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind hosted provider fixture");
    listener
        .set_nonblocking(true)
        .expect("sequence listener nonblocking");
    let address = listener.local_addr().expect("hosted provider address");
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut requests = Vec::new();
        for response in responses {
            let Some(mut stream) = accept_sequence_request(&listener) else {
                break;
            };
            requests.push(read_http_request(&mut stream));
            write_http_response(&mut stream, response.status, &response.body);
        }
        let _ = tx.send(requests);
    });
    (format!("http://{address}/v1"), rx)
}

fn accept_sequence_request(listener: &TcpListener) -> Option<TcpStream> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match listener.accept() {
            Ok((stream, _)) => return Some(stream),
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                if std::time::Instant::now() >= deadline {
                    return None;
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

fn read_http_request(stream: &mut TcpStream) -> String {
    let mut reader = BufReader::new(stream.try_clone().expect("clone provider stream"));
    let mut first_line = String::new();
    reader
        .read_line(&mut first_line)
        .expect("read provider request line");
    let mut request = first_line;
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).expect("read provider header");
        if bytes == 0 || line == "\r\n" {
            request.push_str("\r\n");
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            content_length = value.trim().parse::<usize>().unwrap_or(0);
        }
        request.push_str(&line);
    }
    if content_length > 0 {
        let mut request_body = vec![0; content_length];
        reader
            .read_exact(&mut request_body)
            .expect("read provider request body");
        request.push_str(&String::from_utf8_lossy(&request_body));
    }
    request
}

fn write_http_response(stream: &mut TcpStream, status: u16, body: &Value) {
    let body = body.to_string();
    let reason = if status == 400 { "Bad Request" } else { "OK" };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream
        .write_all(response.as_bytes())
        .expect("write provider response");
}

fn hosted_requests(rx: mpsc::Receiver<Vec<String>>) -> Vec<String> {
    rx.recv_timeout(Duration::from_secs(10)).unwrap_or_default()
}

fn hosted_request_body(request: &str) -> Value {
    let body = request
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .expect("hosted request must include a body separator");
    serde_json::from_str(body).expect("hosted request body JSON")
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
        return serde_json::json!({ "anyOf": [schema, { "type": "null" }] });
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
        items.push(serde_json::json!({ "type": "null" }));
        return schema;
    }
    serde_json::json!({ "anyOf": [schema, { "type": "null" }] })
}

fn schema_allows_null(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return false;
    };
    match object.get("type") {
        Some(Value::String(type_name)) if type_name == "null" => return true,
        Some(Value::Array(types)) if types.iter().any(|item| item.as_str() == Some("null")) => {
            return true;
        }
        _ => {}
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

fn assert_response_format_schema_request(request: &str) {
    let body = hosted_request_body(request);
    let fixture_schema = fixture_value(include_str!("fixtures/fixture_output_schema.json"));
    let expected_schema = strict_response_format_schema(&fixture_schema);
    assert_eq!(
        body.pointer("/response_format/type")
            .and_then(Value::as_str),
        Some("json_schema"),
        "Native-configured request must carry response_format json_schema: {body}"
    );
    assert_eq!(
        body.pointer("/response_format/json_schema/strict")
            .and_then(Value::as_bool),
        Some(true),
        "Native-configured request must ask the endpoint for strict schema enforcement: {body}"
    );
    assert_eq!(
        body.pointer("/response_format/json_schema/schema")
            .map(normalize_required_sets),
        Some(normalize_required_sets(&expected_schema)),
        "Native-configured request must carry the attached schema body: {body}"
    );
}

fn assert_json_object_request(request: &str) {
    let body = hosted_request_body(request);
    assert_eq!(
        body.pointer("/response_format/type")
            .and_then(Value::as_str),
        Some("json_object"),
        "ShapeOnly-configured request must carry response_format json_object: {body}"
    );
}

fn assert_no_structured_output_field_request(request: &str) {
    let body = hosted_request_body(request);
    for field in ["response_format", "guided_json", "grammar"] {
        assert!(
            body.get(field).is_none(),
            "fallback request must drop structured-output field {field}: {body}"
        );
    }
}

fn write_local_vllm_structured_output_override(harness: &CliHarness, style: Option<&str>) {
    let Some(style) = style else {
        return;
    };
    let config_dir = harness.data_root.path().join("cg");
    std::fs::create_dir_all(&config_dir).expect("provider override dir");
    std::fs::write(
        config_dir.join("providers.toml"),
        format!(
            r#"[providers.local-vllm]
structured_output = "{style}"
"#,
        ),
    )
    .expect("write structured-output provider override");
}

fn run_hosted_capture_with_responses(
    style: Option<&str>,
    responses: Vec<ChatCompletionStubResponse>,
    transcript: &str,
) -> HostedCapture {
    let harness = CliHarness::new();
    write_local_vllm_structured_output_override(&harness, style);
    let cwd = tempfile::tempdir().expect("hosted capture cwd");
    let (endpoint, requests) = spawn_chat_completion_sequence_server(responses);
    let connect = harness.run(&[
        "model",
        "connect",
        "local-vllm",
        "--endpoint",
        &endpoint,
        "--model",
        "test-model",
        "--no-auth",
        "--consent",
    ]);
    assert_eq!(connect.code, 0, "{}", combined(&connect));
    let output = run_in_dir_with_stdin(
        &harness,
        cwd.path(),
        &["record", "--understand-now"],
        transcript,
    );
    HostedCapture {
        harness,
        output,
        requests: hosted_requests(requests),
    }
}

fn assert_query_readback_contains(harness: &CliHarness, query: &str, expected: &[&str]) {
    let output = harness.run(&["query", query]);
    let text = combined(&output);
    assert_eq!(output.code, 0, "{text}");
    for needle in expected {
        assert!(
            text.contains(needle),
            "captured memory must be readable through query; missing {needle:?}: {text}"
        );
    }
}

fn fixture_value(raw: &str) -> Value {
    serde_json::from_str(raw).expect("fixture JSON")
}

fn fixture_content(value: &Value) -> String {
    match value.get("content").expect("fixture content") {
        Value::String(content) => content.clone(),
        other => other.to_string(),
    }
}

fn malformed_then_good_responses() -> Vec<ChatCompletionStubResponse> {
    let replies = fixture_value(include_str!("fixtures/malformed_then_good_replies.json"));
    replies
        .as_array()
        .expect("malformed-then-good array")
        .iter()
        .enumerate()
        .map(|(index, item)| {
            ChatCompletionStubResponse::ok_content(
                fixture_content(item),
                400 + index as u64,
                40 + index as u64,
            )
        })
        .collect()
}

fn malformed_then_uncleanable_responses() -> Vec<ChatCompletionStubResponse> {
    let replies = fixture_value(include_str!(
        "fixtures/malformed_then_uncleanable_replies.json"
    ));
    replies
        .as_array()
        .expect("malformed-then-uncleanable array")
        .iter()
        .enumerate()
        .map(|(index, item)| {
            ChatCompletionStubResponse::ok_content(
                fixture_content(item),
                500 + index as u64,
                50 + index as u64,
            )
        })
        .collect()
}

fn good_harvest_response() -> ChatCompletionStubResponse {
    malformed_then_good_responses()
        .pop()
        .expect("good harvest response")
}

fn receipt_call_string(output: &CommandOutput, field: &str) -> Option<String> {
    output_json(output)?
        .pointer(&format!("/usage/calls/0/{field}"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn first_reliability_level(output: &CommandOutput) -> Option<String> {
    receipt_call_string(output, "reliability_level")
}

fn first_outcome(output: &CommandOutput) -> Option<String> {
    receipt_call_string(output, "outcome")
}

fn all_reliability_levels(output: &CommandOutput) -> Vec<String> {
    output_json(output)
        .and_then(|json| {
            json.pointer("/usage/calls")
                .and_then(Value::as_array)
                .map(|calls| {
                    calls
                        .iter()
                        .filter_map(|call| {
                            call.pointer("/reliability_level")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                        })
                        .collect::<Vec<_>>()
                })
        })
        .unwrap_or_default()
}

fn assert_receipt_omits_endpoint_specific_output_names(output: &CommandOutput) {
    let json = output_json(output).expect("record output json");
    let receipt = serde_json::to_string(&json)
        .expect("record output json must render for endpoint vocabulary check")
        .to_ascii_lowercase();
    for vendor_name in [
        "response_format",
        "guided_json",
        "grammar",
        "json-schema",
        "output-schema",
    ] {
        assert!(
            !receipt.contains(vendor_name),
            "receipt must use endpoint-agnostic reliability vocabulary and omit provider field name {vendor_name}: {receipt}"
        );
    }
}

fn assert_floor_harvest_rows(output: &CommandOutput) {
    assert_record_contains(output, "record malformed replies through the repair floor");
    assert_record_contains(output, "keep first-day capture reliable");
    assert_record_observation_count(output, 1);
}

fn assert_observation_stored(output: &CommandOutput, expected: &str) {
    let store =
        persisted_store(store_path_from(&[output])).expect("record output must expose store path");
    let context_id = context_id_from(&[output]).expect("record output must expose context id");
    let observations = store
        .list_observations(Some(context_id))
        .expect("recorded observations");
    assert!(
        observations
            .iter()
            .any(|observation| observation.observation_type == expected),
        "recorded observation {expected:?} must be stored in the engine rows: {observations:?}"
    );
}

fn assert_floor_observation_stored(output: &CommandOutput) {
    assert_observation_stored(
        output,
        "capture door observed a malformed reply before repair",
    );
}

#[test]
fn floor_capture_recovers_from_malformed_reply_and_writes_memory() {
    let HostedCapture {
        harness,
        output,
        requests,
    } = run_hosted_capture_with_responses(
        None,
        malformed_then_good_responses(),
        "A real session says the capture door should repair malformed replies and keep first-day capture reliable.",
    );

    assert_eq!(
        requests.len(),
        2,
        "the shipped floor must fire one repair retry after an off-shape reply"
    );
    assert_floor_harvest_rows(&output);
    assert_query_readback_contains(
        &harness,
        "first-day capture reliable",
        &[
            "record malformed replies through the repair floor",
            "keep first-day capture reliable",
        ],
    );
    assert_floor_observation_stored(&output);
}

#[test]
fn floor_exhaustion_is_not_a_silent_empty_graph() {
    let HostedCapture { output, .. } = run_hosted_capture_with_responses(
        None,
        malformed_then_uncleanable_responses(),
        "A real session asks cg to keep first-day capture reliable even when both floor attempts are off-shape.",
    );
    let json = output_json(&output);
    let silently_empty = output.code == 0
        && json
            .as_ref()
            .and_then(|value| value.pointer("/captured/goals"))
            .and_then(Value::as_u64)
            == Some(0)
        && json
            .as_ref()
            .and_then(|value| value.pointer("/captured/decisions"))
            .and_then(Value::as_u64)
            == Some(0);

    assert!(
        !silently_empty,
        "floor exhaustion must not end as success with an empty graph and no observable failure"
    );
    assert_eq!(
        first_reliability_level(&output).as_deref(),
        Some("Floor"),
        "the exhausted floor must be recorded as the achieved Floor level"
    );
}

#[test]
fn unsupported_field_400_retries_without_field_and_still_captures() {
    let HostedCapture {
        harness,
        output,
        requests,
    } = run_hosted_capture_with_responses(
        Some("response_format_schema"),
        vec![
            ChatCompletionStubResponse::raw(
                400,
                serde_json::json!({
                    "error": {
                        "message": "unsupported response_format field",
                        "type": "invalid_request_error"
                    }
                }),
            ),
            good_harvest_response(),
        ],
        "A real session says unsupported schema fields should fall back and still capture.",
    );

    assert_eq!(
        requests.len(),
        2,
        "the 400 fallback must retry once without the structured-output field"
    );
    assert_response_format_schema_request(&requests[0]);
    assert_no_structured_output_field_request(&requests[1]);
    assert_eq!(
        output.code,
        0,
        "an unsupported structured-output field must fall back instead of hard-erroring: {}",
        combined(&output)
    );
    assert_floor_harvest_rows(&output);
    assert_query_readback_contains(
        &harness,
        "first-day capture reliable",
        &[
            "record malformed replies through the repair floor",
            "keep first-day capture reliable",
        ],
    );
    assert_floor_observation_stored(&output);
    assert_eq!(
        first_reliability_level(&output).as_deref(),
        Some("Floor"),
        "the 400 fallback must record the achieved Floor level, not the configured Native rung"
    );
}

#[test]
fn enforcing_endpoint_whose_reply_needed_repair_records_achieved_not_configured_level() {
    let HostedCapture {
        harness: native_harness,
        output: native_output,
        requests: native_requests,
    } = run_hosted_capture_with_responses(
        Some("response_format_schema"),
        malformed_then_good_responses(),
        "A real session reaches a Native-configured endpoint whose first reply still needs repair.",
    );
    let HostedCapture {
        harness: shape_harness,
        output: shape_output,
        requests: shape_requests,
    } = run_hosted_capture_with_responses(
        Some("json_object"),
        malformed_then_good_responses(),
        "A real session reaches a ShapeOnly-configured endpoint whose first reply still needs repair.",
    );

    assert_eq!(native_requests.len(), 2, "Native case must run repair");
    assert_eq!(shape_requests.len(), 2, "ShapeOnly case must run repair");
    assert_floor_harvest_rows(&native_output);
    assert_floor_harvest_rows(&shape_output);
    assert_query_readback_contains(
        &native_harness,
        "first-day capture reliable",
        &[
            "record malformed replies through the repair floor",
            "keep first-day capture reliable",
        ],
    );
    assert_floor_observation_stored(&native_output);
    assert_query_readback_contains(
        &shape_harness,
        "first-day capture reliable",
        &[
            "record malformed replies through the repair floor",
            "keep first-day capture reliable",
        ],
    );
    assert_floor_observation_stored(&shape_output);
    assert_eq!(
        first_reliability_level(&native_output).as_deref(),
        Some("Floor"),
        "a Native-configured call that needed repair achieved Floor"
    );
    assert_eq!(
        first_reliability_level(&shape_output).as_deref(),
        Some("Floor"),
        "a ShapeOnly-configured call that needed repair achieved Floor"
    );
    assert_response_format_schema_request(&native_requests[0]);
    assert_json_object_request(&shape_requests[0]);
}

#[test]
fn refusal_and_truncation_are_distinct_outcomes_not_malformed() {
    let fixture = fixture_value(include_str!("fixtures/refusal_and_truncation_replies.json"));
    let malformed = fixture.get("malformed").expect("malformed control fixture");
    let repair = fixture.get("repair").expect("repair fixture");
    let HostedCapture {
        output: malformed_output,
        requests: malformed_requests,
        ..
    } = run_hosted_capture_with_responses(
        None,
        vec![
            ChatCompletionStubResponse::ok_content(fixture_content(malformed), 610, 61),
            ChatCompletionStubResponse::ok_content(fixture_content(repair), 611, 62),
        ],
        "A real session produces a malformed control reply before repair.",
    );
    let HostedCapture {
        output: refusal_output,
        requests: refusal_requests,
        ..
    } = run_hosted_capture_with_responses(
        None,
        vec![ChatCompletionStubResponse::raw(
            200,
            fixture
                .get("refusal_body")
                .expect("refusal fixture")
                .clone(),
        )],
        "A real session asks the model to refuse extraction of private credentials.",
    );
    let HostedCapture {
        output: truncation_output,
        requests: truncation_requests,
        ..
    } = run_hosted_capture_with_responses(
        None,
        vec![ChatCompletionStubResponse::raw(
            200,
            fixture
                .get("truncation_body")
                .expect("truncation fixture")
                .clone(),
        )],
        "A real session returns a response capped by max tokens.",
    );

    assert_eq!(
        malformed_requests.len(),
        2,
        "malformed control burns repair"
    );
    assert_eq!(refusal_requests.len(), 1, "refusal must not burn repair");
    assert_eq!(
        truncation_requests.len(),
        1,
        "truncation must not burn repair"
    );
    assert_eq!(
        [
            first_outcome(&malformed_output),
            first_outcome(&refusal_output),
            first_outcome(&truncation_output),
        ],
        [
            Some("malformed".to_string()),
            Some("refusal".to_string()),
            Some("truncation".to_string()),
        ],
        "malformed, refusal, and truncation must be named as pairwise-distinct receipt outcomes"
    );
}

#[test]
fn enforcement_level_native_shapeonly_floor_recorded_per_call() {
    let HostedCapture {
        harness: native_harness,
        output: native_output,
        requests: native_requests,
    } = run_hosted_capture_with_responses(
        Some("response_format_schema"),
        vec![good_harvest_response()],
        "A real session records malformed replies through the repair floor to keep first-day capture reliable under a schema-honoring endpoint.",
    );
    let HostedCapture {
        harness: shape_harness,
        output: shape_output,
        requests: shape_requests,
    } = run_hosted_capture_with_responses(
        Some("json_object"),
        vec![good_harvest_response()],
        "A real session records malformed replies through the repair floor to keep first-day capture reliable under a json_object endpoint.",
    );
    let HostedCapture {
        harness: floor_harness,
        output: floor_output,
        requests: floor_requests,
    } = run_hosted_capture_with_responses(
        None,
        vec![good_harvest_response()],
        "A real session records malformed replies through the repair floor to keep first-day capture reliable under an unconstrained endpoint.",
    );

    assert_eq!(
        native_requests.len(),
        1,
        "Native success should use one call"
    );
    assert_eq!(
        shape_requests.len(),
        1,
        "ShapeOnly success should use one call"
    );
    assert_eq!(floor_requests.len(), 1, "Floor success should use one call");
    assert_response_format_schema_request(&native_requests[0]);
    assert_json_object_request(&shape_requests[0]);
    assert_no_structured_output_field_request(&floor_requests[0]);
    assert_query_readback_contains(
        &native_harness,
        "first-day capture reliable",
        &[
            "record malformed replies through the repair floor",
            "keep first-day capture reliable",
        ],
    );
    assert_floor_observation_stored(&native_output);
    assert_query_readback_contains(
        &shape_harness,
        "first-day capture reliable",
        &[
            "record malformed replies through the repair floor",
            "keep first-day capture reliable",
        ],
    );
    assert_floor_observation_stored(&shape_output);
    assert_query_readback_contains(
        &floor_harness,
        "first-day capture reliable",
        &[
            "record malformed replies through the repair floor",
            "keep first-day capture reliable",
        ],
    );
    assert_floor_observation_stored(&floor_output);
    assert_eq!(
        first_reliability_level(&native_output).as_deref(),
        Some("Native")
    );
    assert_eq!(
        first_reliability_level(&shape_output).as_deref(),
        Some("ShapeOnly")
    );
    assert_eq!(
        first_reliability_level(&floor_output).as_deref(),
        Some("Floor")
    );
}

#[test]
fn installed_cli_default_path_records_floor_enforcement_level() {
    let harness = extraction_harness();
    let output = harness.run_with_stdin(
        &["record", "--understand-now"],
        "Goal G_installed: keep installed CLI capture reliable. Entity E_installed: capture receipt path=receipt.json. Decision D_installed: installed CLI default path lands memory serves G_installed based_on E_installed. Observation O_installed: installed CLI observation lane lands on E_installed.",
    );

    assert_record_contains(&output, "installed CLI default path lands memory");
    assert_record_observation_count(&output, 1);
    assert_query_readback_contains(
        &harness,
        "installed CLI default path lands memory",
        &["installed CLI default path lands memory"],
    );
    assert_observation_stored(
        &output,
        "installed CLI observation lane lands on E_installed",
    );
    assert_eq!(
        first_reliability_level(&output).as_deref(),
        Some("Floor"),
        "the installed CLI default path must report the achieved Floor level until a schema flag is wired"
    );
}

#[test]
fn receipt_reliability_vocabulary_is_endpoint_agnostic() {
    let HostedCapture {
        output: response_format_output,
        ..
    } = run_hosted_capture_with_responses(
        Some("response_format_schema"),
        vec![good_harvest_response()],
        "A real session uses a response_format_schema endpoint.",
    );
    let HostedCapture {
        output: guided_json_output,
        ..
    } = run_hosted_capture_with_responses(
        Some("guided_json"),
        vec![good_harvest_response()],
        "A real session uses a guided_json endpoint.",
    );
    let mut levels = all_reliability_levels(&response_format_output);
    levels.extend(all_reliability_levels(&guided_json_output));
    assert_receipt_omits_endpoint_specific_output_names(&response_format_output);
    assert_receipt_omits_endpoint_specific_output_names(&guided_json_output);

    assert!(
        !levels.is_empty(),
        "fixture captures must emit receipt levels"
    );
    for level in levels {
        assert!(
            matches!(level.as_str(), "Native" | "ShapeOnly" | "Floor"),
            "receipt reliability level must stay in the endpoint-agnostic vocabulary: {level}"
        );
        let lower = level.to_ascii_lowercase();
        assert!(
            ![
                "response_format",
                "guided_json",
                "grammar",
                "json-schema",
                "output-schema"
            ]
            .iter()
            .any(|vendor| lower.contains(vendor)),
            "receipt reliability level must not leak vendor field names: {level}"
        );
    }
}

fn stable_id(output: &CommandOutput, keys: &[&str]) -> Option<String> {
    let json = output_json(output)?;
    let stable_ids = json.get("stable_ids").and_then(Value::as_object)?;
    keys.iter().find_map(|key| {
        stable_ids
            .get(*key)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    })
}

fn json_string_at<'a>(json: &'a Value, paths: &[&str]) -> Option<&'a str> {
    paths
        .iter()
        .find_map(|path| json.pointer(path).and_then(Value::as_str))
}

fn json_u64_at(json: &Value, paths: &[&str]) -> Option<u64> {
    paths
        .iter()
        .find_map(|path| json.pointer(path).and_then(Value::as_u64))
}

fn json_bool_at(json: &Value, paths: &[&str]) -> Option<bool> {
    paths
        .iter()
        .find_map(|path| json.pointer(path).and_then(Value::as_bool))
}

fn json_array_len_at(json: &Value, paths: &[&str]) -> Option<u64> {
    paths.iter().find_map(|path| {
        json.pointer(path)
            .and_then(Value::as_array)
            .map(|items| items.len() as u64)
    })
}

fn completion_records_since(harness: &CliHarness, start: usize) -> Vec<Value> {
    let records = harness.providers.completion_records();
    assert!(
        records.len() >= start,
        "provider completion log shrank: start={start}, records={records:?}"
    );
    records[start..].to_vec()
}

fn provider_usage(record: &Value) -> (u64, u64, u64) {
    let usage = record
        .get("usage")
        .and_then(Value::as_object)
        .expect("fake provider completion log must include exact provider usage");
    let input = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .expect("provider usage input tokens");
    let output = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .expect("provider usage output tokens");
    let total = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .expect("provider usage total tokens");
    assert_eq!(
        total,
        input + output,
        "fake provider usage fixture is invalid"
    );
    (input, output, total)
}

fn assert_cli_capture_usage_receipt(output: &CommandOutput, provider_records: &[Value]) {
    assert!(
        !provider_records.is_empty(),
        "record fixture must drive the shipped provider path"
    );
    let json = output_json(output).expect("record output json");
    let calls = json
        .pointer("/usage/calls")
        .and_then(Value::as_array)
        .expect("record receipt must expose per-model-call usage from the shipped provider path");
    assert_eq!(
        calls.len(),
        provider_records.len(),
        "receipt must account for every provider call made by this record command"
    );
    let mut summed_total = 0;
    for (call, provider_record) in calls.iter().zip(provider_records) {
        let (expected_prompt, expected_completion, expected_total) =
            provider_usage(provider_record);
        let prompt_tokens = call
            .pointer("/prompt_tokens")
            .or_else(|| call.pointer("/input_tokens"))
            .and_then(Value::as_u64)
            .expect("provider call usage must include input/prompt tokens");
        let completion_tokens = call
            .pointer("/completion_tokens")
            .or_else(|| call.pointer("/output_tokens"))
            .and_then(Value::as_u64)
            .expect("provider call usage must include output/completion tokens");
        let total_tokens = call
            .pointer("/total_tokens")
            .and_then(Value::as_u64)
            .expect("provider call usage must include total tokens");
        assert_eq!(
            (prompt_tokens, completion_tokens, total_tokens),
            (expected_prompt, expected_completion, expected_total),
            "record receipt must preserve the exact usage block returned by the fake provider"
        );
        assert_eq!(
            total_tokens,
            prompt_tokens + completion_tokens,
            "per-call total must come from the provider usage block, not a fixture constant"
        );
        summed_total += total_tokens;
    }
    assert_eq!(
        json.pointer("/usage/total_tokens").and_then(Value::as_u64),
        Some(summed_total),
        "record receipt must sum provider usage across calls"
    );
    assert!(
        json.pointer("/usage/elapsed_ms")
            .and_then(Value::as_u64)
            .unwrap_or_default()
            > 0,
        "record receipt must expose elapsed wall-clock time"
    );
}

fn expected_cli_import_chunks(segments: &[String], max_chars: usize) -> u64 {
    let mut buffer = String::new();
    let mut chunks = 0;
    for segment in segments {
        append_expected_import_segment(&mut buffer, segment, max_chars, &mut chunks);
    }
    flush_expected_import_chunk(&mut buffer, &mut chunks);
    chunks
}

fn append_expected_import_segment(
    buffer: &mut String,
    segment: &str,
    max_chars: usize,
    chunks: &mut u64,
) {
    let mut rest = segment;
    while !rest.is_empty() {
        if buffer.len() >= max_chars {
            flush_expected_import_chunk(buffer, chunks);
            continue;
        }
        let remaining = max_chars - buffer.len();
        if rest.len() <= remaining {
            buffer.push_str(rest);
            break;
        }
        let split = expected_import_chunk_boundary(rest, remaining);
        if split == 0 {
            flush_expected_import_chunk(buffer, chunks);
            continue;
        }
        buffer.push_str(&rest[..split]);
        flush_expected_import_chunk(buffer, chunks);
        rest = &rest[split..];
    }
}

fn flush_expected_import_chunk(buffer: &mut String, chunks: &mut u64) {
    if !buffer.trim().is_empty() {
        *chunks += 1;
    }
    buffer.clear();
}

fn expected_import_chunk_boundary(text: &str, max_bytes: usize) -> usize {
    if max_bytes >= text.len() {
        return text.len();
    }
    let mut last_boundary = 0;
    let mut last_whitespace = None;
    for (index, ch) in text.char_indices() {
        let end = index + ch.len_utf8();
        if end > max_bytes {
            break;
        }
        last_boundary = end;
        if ch.is_whitespace() {
            last_whitespace = Some(end);
        }
    }
    if let Some(whitespace) = last_whitespace.filter(|index| *index >= max_bytes / 2) {
        return whitespace;
    }
    if last_boundary > 0 {
        return last_boundary;
    }
    text.char_indices()
        .nth(1)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

fn assert_cli_import_estimate_makes_no_provider_calls(harness: &CliHarness, cwd: &std::path::Path) {
    let session_path = cwd.join("session.jsonl");
    let chunk_window = 64;
    let user_text = "Goal G_est: estimate import. Entity E_est: estimator path=src/estimate.rs. Decision D_est: estimate before spending serves G_est based_on E_est.";
    let assistant_text = "Observation O_est: E_est moved -> src/estimate_v2.rs";
    let parsed_segments = vec![
        format!("User: {user_text}\n"),
        format!("Assistant: {assistant_text}\n"),
    ];
    let expected_chunks = expected_cli_import_chunks(&parsed_segments, chunk_window);
    assert!(
        expected_chunks > 1,
        "estimate fixture must force real chunking"
    );
    std::fs::write(
        &session_path,
        [
            format!(r#"{{"message":{{"role":"user","content":"{user_text}"}}}}"#),
            format!(r#"{{"message":{{"role":"assistant","content":"{assistant_text}"}}}}"#),
        ]
        .join("\n"),
    )
    .expect("write estimate fixture");
    let before = harness.providers.completion_count();
    let estimate = run_in_dir(
        harness,
        cwd,
        &[
            "import",
            "--understand-now",
            session_path.to_str().expect("utf8 session path"),
            "--tenant-id",
            "owner-tenant",
            "--estimate",
            "--chunk-tokens",
            &chunk_window.to_string(),
        ],
    );
    assert_eq!(estimate.code, 0, "{}", combined(&estimate));
    assert_eq!(
        harness.providers.completion_count(),
        before,
        "`cg import --estimate` must not call the provider"
    );
    let json = output_json(&estimate).expect("estimate output json");
    assert!(
        json.pointer("/estimate/chunks").and_then(Value::as_u64) == Some(expected_chunks),
        "estimate must report real chunk count"
    );
    assert_eq!(
        json.pointer("/estimate/resolved_chunk_window")
            .and_then(Value::as_u64),
        Some(chunk_window as u64),
        "estimate must use the explicit chunk window that determines the expected chunk count"
    );
    assert_eq!(
        json.pointer("/estimate/provider_calls")
            .and_then(Value::as_u64),
        Some(0),
        "estimate must explicitly report zero provider calls"
    );
    assert!(
        json.pointer("/estimate/elapsed_ms")
            .and_then(Value::as_u64)
            .unwrap_or_default()
            > 0,
        "estimate must project elapsed wall-clock time"
    );
}

fn store_path_from(outputs: &[&CommandOutput]) -> Option<std::path::PathBuf> {
    outputs.iter().find_map(|output| {
        let json = output_json(output)?;
        json_string_at(
            &json,
            &[
                "/store/db_path",
                "/store_path",
                "/db_path",
                "/context_graph_db",
            ],
        )
        .map(std::path::PathBuf::from)
    })
}

fn context_id_from(outputs: &[&CommandOutput]) -> Option<ContextId> {
    outputs
        .iter()
        .find_map(|output| {
            let json = output_json(output)?;
            json_string_at(
                &json,
                &["/context/id", "/context_id", "/stable_ids/context"],
            )
            .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        })
        .map(ContextId::from)
}

fn memory_counts(output: &CommandOutput) -> Option<(u64, u64, u64)> {
    if let Some(json) = output_json(output) {
        let goals = json
            .pointer("/totals/goals")
            .or_else(|| json.pointer("/goals"))
            .and_then(Value::as_u64);
        let decisions = json
            .pointer("/totals/decisions")
            .or_else(|| json.pointer("/decisions"))
            .and_then(Value::as_u64);
        let observations = json
            .pointer("/totals/observations")
            .or_else(|| json.pointer("/observations"))
            .and_then(Value::as_u64);
        if let (Some(goals), Some(decisions), Some(observations)) = (goals, decisions, observations)
        {
            return Some((goals, decisions, observations));
        }
    }

    let text = combined(output);
    Some((
        count_after_label(&text, "goals:")?,
        count_after_label(&text, "decisions:")?,
        count_after_label(&text, "observations:")?,
    ))
}

struct CliSyncFixture {
    _hub_dir: tempfile::TempDir,
    _hub_store: Store,
    hub_db: Arc<Database>,
    shutdown: Arc<AtomicBool>,
    server_task: tokio::task::JoinHandle<()>,
    remote_url: String,
    tenant_id: String,
}

impl Drop for CliSyncFixture {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.server_task.abort();
    }
}

async fn cli_sync_fixture(label: &str) -> CliSyncFixture {
    let hub_dir = tempfile::tempdir().expect("hub tempdir");
    let tenant_id = format!("tenant-{}", label.replace('_', "-"));
    let hub_store = Store::open(StoreConfig {
        db_path: hub_dir.path().join("hub.db"),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open CLI sync hub store");
    let hub_db = hub_store.sync_database();
    let endpoint = tokio::time::timeout(
        Duration::from_secs(30),
        IrohServer::bind(&format!(
            "iroh:?identity={}",
            hub_dir.path().join("hub.fabric-identity.key").display()
        )),
    )
    .await
    .expect("CLI authenticated hub bind stays bounded")
    .expect("bind CLI authenticated hub");
    let remote_url = endpoint.ticket();
    let server = Arc::new(SyncServer::new(
        Arc::clone(&hub_db),
        &endpoint,
        contextdb_core::TenantId::from(&tenant_id),
    ));
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_task = {
        let server = Arc::clone(&server);
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move { server.run_until(shutdown).await })
    };
    CliSyncFixture {
        _hub_dir: hub_dir,
        _hub_store: hub_store,
        hub_db,
        shutdown,
        server_task,
        remote_url,
        tenant_id,
    }
}

async fn unavailable_authenticated_ticket(label: &str) -> String {
    let root = tempfile::tempdir().expect("unavailable CLI endpoint tempdir");
    let endpoint = tokio::time::timeout(
        Duration::from_secs(30),
        IrohServer::bind(&format!(
            "iroh:?identity={}",
            root.path().join(format!("{label}.key")).display()
        )),
    )
    .await
    .expect("unavailable CLI endpoint bind stays bounded")
    .expect("mint authenticated unavailable CLI ticket");
    let ticket = endpoint.ticket();
    drop(endpoint);
    ticket
}

fn cli_expected_pending_push_change_count(
    store: &Store,
    remote_url: &str,
    tenant_id: &str,
) -> usize {
    let client = SyncClient::new(
        store.sync_database(),
        remote_url,
        contextdb_core::TenantId::from(tenant_id),
    );
    context_graph_pending_push_change_count(&client)
        .expect("count the exact pending push set for the CLI fixture")
}

fn db_table_row_count(db: &Database, table: &str) -> usize {
    db.scan(table, db.snapshot())
        .unwrap_or_else(|err| panic!("scan {table}: {err}"))
        .len()
}

fn schema_version_installed_at(db: &Database) -> contextdb_core::TxId {
    let installed_at = db
        .scan("cg_schema_version", db.snapshot())
        .expect("scan schema version")
        .into_iter()
        .filter_map(|row| match row.values.get("installed_at") {
            Some(DbValue::TxId(tx)) => Some(*tx),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        installed_at.len(),
        1,
        "schema version fixture must have exactly one installed_at value; installed_at={installed_at:?}"
    );
    installed_at[0]
}

fn force_divergent_schema_version(path: &std::path::Path) -> contextdb_core::TxId {
    let store = persisted_store(Some(path.to_path_buf())).expect("open second CLI store");
    let db = store.sync_database();
    let original_installed_at = schema_version_installed_at(db.as_ref());
    let committed = db.committed_watermark();
    let divergent_installed_at = if original_installed_at != committed {
        committed
    } else {
        assert!(
            committed.0 > 0,
            "schema version fixture needs a prior valid txid to create a divergent schema row"
        );
        contextdb_core::TxId(committed.0 - 1)
    };
    assert_ne!(
        original_installed_at, divergent_installed_at,
        "schema version fixture must create a divergent local schema row"
    );
    db.execute(
        "UPDATE cg_schema_version SET installed_at = $installed_at WHERE id = $id",
        &HashMap::from([
            ("id".to_string(), DbValue::Int64(1)),
            (
                "installed_at".to_string(),
                DbValue::TxId(divergent_installed_at),
            ),
        ]),
    )
    .expect("create local divergent schema-version fixture");
    assert_eq!(
        schema_version_installed_at(db.as_ref()),
        divergent_installed_at,
        "fixture must create a local divergent schema version"
    );
    divergent_installed_at
}

fn count_after_label(text: &str, label: &str) -> Option<u64> {
    let tail = text.split_once(label)?.1;
    tail.split_whitespace()
        .next()
        .and_then(|value| value.trim_matches(',').parse().ok())
}

fn stable_uuid_from(outputs: &[&CommandOutput], keys: &[&str]) -> Option<uuid::Uuid> {
    outputs
        .iter()
        .find_map(|output| stable_id(output, keys))
        .and_then(|raw| uuid::Uuid::parse_str(&raw).ok())
}

fn persisted_store(path: Option<std::path::PathBuf>) -> Option<Store> {
    path.and_then(|db_path| {
        Store::open(StoreConfig {
            db_path,
            default_text_embedder: None,
            ..Default::default()
        })
        .ok()
    })
}

fn scoped_persisted_store(path: Option<std::path::PathBuf>, context_label: &str) -> Option<Store> {
    let db_path = path?;
    let admin = persisted_store(Some(db_path.clone()))?;
    let context_id = admin
        .list_contexts()
        .ok()?
        .into_iter()
        .find(|context| context.name == context_label)?
        .id;
    Store::open_with_contexts(
        StoreConfig {
            db_path,
            default_text_embedder: None,
            ..Default::default()
        },
        BTreeSet::from([context_id]),
    )
    .ok()
}

fn persisted_decisions_for_intention(
    store: Option<&Store>,
    intention_id: Option<IntentionId>,
) -> Vec<(DecisionId, String, DecisionStatus)> {
    match (store, intention_id) {
        (Some(store), Some(intention_id)) => store
            .by_intention(intention_id, DiscoveryFilter::default())
            .unwrap_or_default()
            .into_iter()
            .map(|decision| (decision.id, decision.description, decision.status))
            .collect(),
        _ => Vec::new(),
    }
}

fn persisted_decisions_for_entity(
    store: Option<&Store>,
    entity_id: Option<EntityId>,
) -> Vec<(DecisionId, String, DecisionStatus)> {
    match (store, entity_id) {
        (Some(store), Some(entity_id)) => store
            .by_entity_decisions(
                entity_id,
                DiscoveryFilter {
                    include_decision_statuses: Some(vec![
                        DecisionStatus::Active,
                        DecisionStatus::Invalidated,
                        DecisionStatus::Superseded,
                    ]),
                    ..Default::default()
                },
            )
            .unwrap_or_default()
            .into_iter()
            .map(|decision| (decision.id, decision.description, decision.status))
            .collect(),
        _ => Vec::new(),
    }
}

fn persisted_invalidated_decisions(
    store: Option<&Store>,
    context_id: Option<ContextId>,
    decision_id: Option<DecisionId>,
) -> Vec<(DecisionId, Vec<IntentionId>)> {
    match (store, context_id, decision_id) {
        (Some(store), Some(context_id), Some(decision_id)) => store
            .list_invalidations(context_id, Some(decision_id), None)
            .unwrap_or_default()
            .into_iter()
            .map(|row| (row.affected_decision_id, row.threatened_intention_ids))
            .collect(),
        _ => Vec::new(),
    }
}

fn persisted_invalidation_decision_ids(
    store: Option<&Store>,
    context_id: Option<ContextId>,
) -> Vec<DecisionId> {
    let mut ids = match (store, context_id) {
        (Some(store), Some(context_id)) => store
            .list_invalidations(context_id, None, None)
            .unwrap_or_default()
            .into_iter()
            .map(|row| row.affected_decision_id)
            .collect(),
        _ => Vec::new(),
    };
    sort_decision_ids(&mut ids);
    ids
}

fn persisted_context_names(path: Option<&std::path::Path>) -> Vec<String> {
    let Some(path) = path else {
        return Vec::new();
    };
    let Ok(db) = Database::open(path) else {
        return Vec::new();
    };
    let mut names = db
        .scan("contexts", db.snapshot())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| match row.values.get("name") {
            Some(DbValue::Text(name)) => Some(name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn invalidation_decision_ids_from_output(
    output: &CommandOutput,
    array_keys: &[&str],
) -> Vec<DecisionId> {
    let Some(json) = output_json(output) else {
        return Vec::new();
    };
    let Some(rows) = array_keys
        .iter()
        .find_map(|key| json.get(*key).and_then(Value::as_array))
    else {
        return Vec::new();
    };
    let mut ids = rows
        .iter()
        .filter_map(|row| {
            row.get("affected_decision_id")
                .or_else(|| row.get("decision_id"))
                .and_then(Value::as_str)
                .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
                .map(DecisionId::from)
        })
        .collect::<Vec<_>>();
    sort_decision_ids(&mut ids);
    ids
}

fn sort_decision_ids(ids: &mut [DecisionId]) {
    ids.sort_by_key(|id| id.as_uuid());
}

fn seed_decision_text_only_decoy(
    store: Option<&Store>,
    context_id: Option<ContextId>,
) -> Option<DecisionId> {
    let (Some(store), Some(context_id)) = (store, context_id) else {
        return None;
    };
    let decoy_goal = store
        .create_intention(CreateIntention {
            description: "keep decision-text decoy out of digest query".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            ..Default::default()
        })
        .ok()?;
    store
        .create_decision(CreateDecision {
            decision_type: "code_reference".to_string(),
            description: "auth module at src/authentication.py decision-text-only decoy"
                .to_string(),
            confidence: Some(0.99),
            intention_ids: vec![decoy_goal.id],
            context_id,
            ..Default::default()
        })
        .ok()
        .map(|decision| decision.id)
}

fn digest_ids_extracted_from_decision(
    path: Option<&std::path::Path>,
    decision_id: Option<DecisionId>,
) -> Vec<String> {
    let (Some(path), Some(decision_id)) = (path, decision_id) else {
        return Vec::new();
    };
    let Ok(db) = Database::open(path) else {
        return Vec::new();
    };
    let mut ids = db
        .scan("edges", db.snapshot())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| {
            let source_matches = matches!(
                row.values.get("source_id"),
                Some(DbValue::Uuid(source_id)) if *source_id == decision_id.as_uuid()
            );
            let relationship_matches = matches!(
                row.values.get("edge_type"),
                Some(DbValue::Text(edge_type)) if edge_type == "EXTRACTED_FROM"
            );
            match (
                source_matches,
                relationship_matches,
                row.values.get("target_id"),
            ) {
                (true, true, Some(DbValue::Uuid(target_id))) => Some(target_id.to_string()),
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn sorted_present_decision_ids(ids: &[Option<DecisionId>]) -> Vec<DecisionId> {
    let mut present = ids.iter().flatten().copied().collect::<Vec<_>>();
    sort_decision_ids(&mut present);
    present
}

fn no_resolution(output: &CommandOutput) -> bool {
    let text = combined(output);
    if text.contains("not found") || text.contains("no matching") || text.contains("nothing") {
        return true;
    }
    output_json(output)
        .and_then(|json| {
            json.get("decisions")
                .or_else(|| json.get("history"))
                .and_then(Value::as_array)
                .map(Vec::is_empty)
        })
        .unwrap_or(false)
}

#[test]
fn cli_record_observation_emits_invalidations() {
    let harness = extraction_harness();
    let first = harness.run_with_stdin(&["record", "--understand-now"], AUTH_INITIAL);
    let output = harness.run_with_stdin(&["record", "--understand-now"], AUTH_RENAME);
    let store_path = store_path_from(&[&first, &output]);
    let store = persisted_store(store_path.clone());
    let context_id = context_id_from(&[&first, &output]);
    let old_decision =
        stable_uuid_from(&[&first, &output], &["D_old", "auth module at src/auth.py"])
            .map(DecisionId::from);
    let goal = stable_uuid_from(
        &[&first, &output],
        &["G1", "ship password-less login for acme"],
    )
    .map(IntentionId::from);

    assert_eq!(
        (
            output.code,
            store_path.is_some(),
            context_id.is_some(),
            old_decision.is_some(),
            persisted_invalidated_decisions(store.as_ref(), context_id, old_decision),
        ),
        (
            0,
            true,
            true,
            true,
            old_decision
                .zip(goal)
                .map(|(old, goal)| vec![(old, vec![goal])])
                .unwrap_or_default(),
        )
    );
    assert!(combined(&output).contains("src/authentication.py"));
    assert!(combined(&output).contains("ship password-less login"));
}

#[test]
fn cli_query_via_digest() {
    let harness = extraction_harness();
    let first = harness.run_with_stdin(&["record", "--understand-now"], AUTH_INITIAL);
    let second = harness.run_with_stdin(&["record", "--understand-now"], AUTH_RENAME);
    let store_path = store_path_from(&[&first, &second]);
    let store = persisted_store(store_path.clone());
    let context_id = context_id_from(&[&first, &second]);
    let decoy = seed_decision_text_only_decoy(store.as_ref(), context_id);
    drop(store);
    let output = harness.run(&["query", "where is the auth module"]);
    let text = combined(&output);
    let store = persisted_store(store_path.clone());
    let goal = stable_uuid_from(
        &[&first, &second, &output],
        &["G1", "ship password-less login for acme"],
    )
    .map(IntentionId::from);
    let current = stable_uuid_from(
        &[&first, &second, &output],
        &["D_new", "auth module at src/authentication.py"],
    )
    .map(DecisionId::from);
    let persisted = persisted_decisions_for_intention(store.as_ref(), goal);
    drop(store);
    let provenance_digest_ids = digest_ids_extracted_from_decision(store_path.as_deref(), current);

    assert_eq!(
        (
            output.code,
            store_path.is_some(),
            decoy.is_some(),
            goal.is_some(),
            current.is_some(),
            !provenance_digest_ids.is_empty(),
            provenance_digest_ids
                .iter()
                .any(|digest_id| text.contains(digest_id)),
            text.contains("EXTRACTED_FROM"),
            !text.contains("decision-text-only decoy"),
            persisted
                .iter()
                .map(|(id, description, status)| (*id, description.as_str(), *status))
                .collect::<Vec<_>>(),
        ),
        (
            0,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            current
                .map(|id| {
                    vec![(
                        id,
                        "auth module at src/authentication.py",
                        DecisionStatus::Active,
                    )]
                })
                .unwrap_or_default(),
        )
    );
    assert!(text.contains("auth module at src/authentication.py"));
    assert!(text.contains("serves goal: ship password-less login for acme"));
    assert!(!text.contains("prior: auth module at src/auth.py [invalidated"));
    assert!(!text.contains("use JWT with 15-min access tokens [superseded]"));
}

#[test]
fn cli_status() {
    let harness = extraction_harness();
    let empty = harness.run(&["status"]);
    harness.run_with_stdin(&["record", "--understand-now"], AUTH_WITH_JWT);
    let first = harness.run(&["status"]);
    harness.run_with_stdin(&["record", "--understand-now"], CI_PIPELINE_SESSION);
    let second = harness.run(&["status"]);

    assert!(combined(&empty).contains("goals: 0"));
    assert!(combined(&first).contains("goals: 1"));
    assert!(combined(&first).contains("decisions: 2"));
    assert!(combined(&first).contains("observations: 1"));
    assert!(combined(&second).contains("goals: 2"));
    assert!(combined(&second).contains("decisions: 4"));
    assert!(combined(&second).contains("observations: 5"));
}

#[test]
fn cli_status_idempotent_on_same_session_recaptured() {
    let harness = extraction_harness();
    let first_capture = harness.run_with_stdin(&["record", "--understand-now"], AUTH_INITIAL);
    let first = harness.run(&["status"]);
    let second_capture = harness.run_with_stdin(&["record", "--understand-now"], AUTH_INITIAL);
    let second = harness.run(&["status"]);
    // The decision id is emitted by the FIRST capture. The identical second capture is a tier-0
    // NO-OP: the verbatim digest already exists, so it is skipped and nothing is re-understood —
    // the strongest idempotency (recapture writes zero new rows). Proven against the persisted
    // store: after both captures the intention still holds exactly ONE decision, with the same id
    // the first capture emitted (never duplicated or replaced).
    let first_decision_id =
        stable_uuid_from(&[&first_capture], &["D_old", "auth module at src/auth.py"])
            .map(DecisionId::from);
    let intention_id = stable_uuid_from(
        &[&first_capture],
        &["G1", "ship password-less login for acme"],
    )
    .map(IntentionId::from);
    let store = persisted_store(store_path_from(&[
        &first_capture,
        &first,
        &second_capture,
        &second,
    ]));
    let persisted_decision_ids = persisted_decisions_for_intention(store.as_ref(), intention_id)
        .into_iter()
        .map(|(id, _, _)| id)
        .collect::<Vec<_>>();
    drop(store);

    assert_eq!(
        (
            memory_counts(&first),
            memory_counts(&second),
            first_decision_id.is_some(),
            persisted_decision_ids,
        ),
        (
            Some((1, 1, 1)),
            Some((1, 1, 1)),
            true,
            first_decision_id.into_iter().collect::<Vec<_>>(),
        )
    );
}

#[test]
fn cli_by_intention() {
    let harness = extraction_harness();
    let capture = harness.run_with_stdin(&["record", "--understand-now"], AUTH_CURRENT_WITH_JWT);
    let intention_id = stable_id(&capture, &["ship password-less login for acme", "G1"]);
    let store_path = store_path_from(&[&capture]);
    let store = persisted_store(store_path.clone());
    let persisted_intention = intention_id
        .as_ref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(IntentionId::from);
    let persisted = persisted_decisions_for_intention(store.as_ref(), persisted_intention);
    drop(store);
    let intention_arg = intention_id
        .as_deref()
        .unwrap_or("__missing_intention_id__");
    let walk = harness.run(&["by-intention", intention_arg]);
    let wrong_id = format!("{intention_arg}-wrong");
    let wrong = harness.run(&["by-intention", wrong_id.as_str()]);
    let text = combined(&walk);

    let mut persisted_decisions = persisted
        .iter()
        .map(|(_, description, status)| (description.as_str(), *status))
        .collect::<Vec<_>>();
    persisted_decisions.sort_by_key(|row| row.0);

    assert_eq!(
        (
            intention_id.as_ref().map(|_| "present"),
            store_path.is_some(),
            persisted_decisions,
        ),
        (
            Some("present"),
            true,
            vec![
                (
                    "auth module at src/authentication.py",
                    DecisionStatus::Active,
                ),
                ("use JWT with 15-min access tokens", DecisionStatus::Active),
            ],
        )
    );
    assert!(text.contains("goal: ship password-less login for acme"));
    assert!(text.contains("auth module at src/authentication.py"));
    assert!(text.contains("use JWT with 15-min access tokens"));
    assert!(!text.contains("store refresh tokens in Redis"));
    assert!(no_resolution(&wrong));
}

#[test]
fn cli_by_entity() {
    let harness = extraction_harness();
    let first = harness.run_with_stdin(&["record", "--understand-now"], AUTH_INITIAL);
    let second = harness.run_with_stdin(&["record", "--understand-now"], AUTH_RENAME);
    let entity_id = stable_id(&first, &["auth module file", "E_auth_module"]);
    let store_path = store_path_from(&[&first, &second]);
    let store = persisted_store(store_path.clone());
    let persisted_entity = entity_id
        .as_ref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(EntityId::from);
    let persisted = persisted_decisions_for_entity(store.as_ref(), persisted_entity);
    drop(store);
    let entity_arg = entity_id.as_deref().unwrap_or("__missing_entity_id__");
    let walk = harness.run(&["by-entity", entity_arg]);
    let wrong_id = format!("{entity_arg}-wrong");
    let wrong = harness.run(&["by-entity", wrong_id.as_str()]);
    let text = combined(&walk);

    assert_eq!(
        (
            entity_id.as_ref().map(|_| "present"),
            store_path.is_some(),
            persisted
                .iter()
                .map(|(_, description, status)| (description.as_str(), *status))
                .collect::<Vec<_>>(),
        ),
        (
            Some("present"),
            true,
            vec![
                (
                    "auth module at src/authentication.py",
                    DecisionStatus::Active
                ),
                ("auth module at src/auth.py", DecisionStatus::Invalidated),
            ],
        )
    );
    assert!(text.contains("auth module file: src/authentication.py"));
    assert!(text.contains("auth module at src/authentication.py"));
    assert!(text.contains("auth module at src/auth.py [invalidated"));
    assert!(text.contains("E_auth_module path changed src/auth.py -> src/authentication.py"));
    assert!(!text.contains("Redis"));
    let walk_json: Value = serde_json::from_str(&walk.stdout).expect("by-entity JSON");
    assert_eq!(
        walk_json
            .pointer("/entity/entity_type")
            .and_then(Value::as_str),
        Some("File"),
        "by-entity must expose the stored entity type for end-to-end assertions"
    );
    assert!(no_resolution(&wrong));
}

#[test]
fn cli_by_entity_renders_non_auth_entity_history_from_stored_rows() {
    let harness = extraction_harness();
    let capture = harness.run_with_stdin(
        &["record", "--understand-now"],
        "Webhook ingestion must stay reliable. The webhook worker file is src/webhook.rs; route webhooks through the signed worker and note the worker path is src/webhook.rs.",
    );
    let entity_id =
        stable_id(&capture, &["E_worker"]).unwrap_or_else(|| "__missing_entity_id__".to_string());
    let walk = harness.run(&["by-entity", &entity_id]);
    let text = combined(&walk);

    assert!(text.contains("webhook worker file: src/webhook.rs"));
    assert!(text.contains("route webhooks through signed worker [current]"));
    assert!(text.contains("E_worker path observed at src/webhook.rs"));
    assert!(!text.contains("auth module"));
    assert!(!text.contains("src/authentication.py"));
}

#[test]
fn cli_walk_verbs_do_not_cross_context_by_default() {
    let harness = extraction_harness();
    let client_a = harness.run_with_stdin(
        &["record", "--understand-now", "--context", "client-a"],
        AUTH_INITIAL,
    );
    harness.run_with_stdin(
        &["record", "--understand-now", "--context", "client-b"],
        "Client B memory must stay separate, so use isolated deploy notes.",
    );
    let intention_id = stable_id(&client_a, &["ship password-less login for acme", "G1"])
        .unwrap_or_else(|| "__missing_intention_id__".to_string());
    let entity_id = stable_id(&client_a, &["auth module file", "E_auth_module"])
        .unwrap_or_else(|| "__missing_entity_id__".to_string());
    let wrong_intention = harness.run(&["by-intention", "--context", "client-b", &intention_id]);
    let wrong_entity = harness.run(&["by-entity", "--context", "client-b", &entity_id]);

    assert!(no_resolution(&wrong_intention));
    assert!(no_resolution(&wrong_entity));
}

#[test]
fn cli_impact_analysis() {
    let harness = extraction_harness();
    harness.run_with_stdin(&["record", "--understand-now"], AUTH_INITIAL);
    harness.run_with_stdin(&["record", "--understand-now"], AUTH_RENAME);
    let output = harness.run(&["impact"]);
    let text = combined(&output);

    assert!(text.contains("auth module at src/auth.py [INVALIDATED]"));
    assert!(text.contains("based on: src/auth.py -> src/authentication.py"));
    assert!(text.contains("served goal: ship password-less login"));
    assert!(text.contains(
        "goal still served by current decision \"auth module at src/authentication.py\""
    ));
}

#[test]
fn cli_impact_goal_still_served_when_another_decision_serves() {
    let harness = extraction_harness();
    harness.run_with_stdin(
        &["record", "--understand-now"],
        "Rollout must stay available. The primary rollout basis is src/d1.py and fallback rollout basis is src/d2.py; D1 uses the primary basis and D2 uses the fallback basis.",
    );
    harness.run_with_stdin(
        &["record", "--understand-now"],
        "The primary rollout basis moved from src/d1.py to src/d1_v2.py.",
    );
    let first = harness.run(&["impact"]);
    harness.run_with_stdin(
        &["record", "--understand-now"],
        "The fallback rollout basis moved from src/d2.py to src/d2_v2.py.",
    );
    let second = harness.run(&["impact"]);

    assert!(combined(&first).contains("goal still served by current decision \"D2\""));
    assert!(!combined(&first).contains("goal unserved"));
    assert!(combined(&second).contains("goal unserved"));
}

#[test]
fn cli_invalidation_lifecycle() {
    let harness = extraction_harness();
    harness.run_with_stdin(&["record", "--understand-now"], AUTH_INITIAL);
    harness.run_with_stdin(&["record", "--understand-now"], AUTH_RENAME);
    let output = harness.run(&["impact"]);
    let text = combined(&output);

    assert!(text.contains("auth module at src/auth.py [INVALIDATED]"));
    assert!(text.contains("auth module at src/authentication.py"));
    assert!(!text.contains("resolved"));
}

#[test]
fn cli_changed_surfaces_cross_session_invalidation() {
    let harness = extraction_harness();
    harness.run_with_stdin(&["record", "--understand-now"], AUTH_INITIAL);
    harness.run_with_stdin(&["record", "--understand-now"], AUTH_RENAME);
    let changed = harness.run(&["changed"]);
    let second = harness.run(&["changed"]);

    assert!(combined(&changed).contains("newly INVALIDATED"));
    assert!(combined(&changed).contains("auth module at src/auth.py"));
    assert!(combined(&changed).contains("ship password-less login"));
    assert!(combined(&second).contains("nothing newly stale"));
}

#[test]
fn impact_standing_set_vs_changed_recent_delta_are_distinct_observables() {
    let harness = extraction_harness();
    let before = harness.run_with_stdin(
        &["record", "--understand-now"],
        "Track the before file at src/before.py so the before path stays visible.",
    );
    let before_invalidation = harness.run_with_stdin(
        &["record", "--understand-now"],
        "The before file moved from src/before.py to src/before_v2.py.",
    );
    assert_record_contains(&before, "src/before.py");
    assert_record_text(&before_invalidation, "src/before_v2.py");
    let cut = harness.run_with_stdin(
        &["record", "--understand-now"],
        "Track the cut file at src/cut.py so the cut path stays visible.",
    );
    let cut_invalidation = harness.run_with_stdin(
        &["record", "--understand-now"],
        "The cut file moved from src/cut.py to src/cut_v2.py.",
    );
    assert_record_contains(&cut, "src/cut.py");
    assert_record_text(&cut_invalidation, "src/cut_v2.py");
    let cutoff_changed = harness.run(&["changed"]);
    let after = harness.run_with_stdin(
        &["record", "--understand-now"],
        "Track the after file at src/after.py so the after path stays visible.",
    );
    let after_invalidation = harness.run_with_stdin(
        &["record", "--understand-now"],
        "The after file moved from src/after.py to src/after_v2.py.",
    );
    assert_record_contains(&after, "src/after.py");
    assert_record_text(&after_invalidation, "src/after_v2.py");
    let changed = harness.run(&["changed"]);
    let repeated = harness.run(&["changed"]);
    let impact = harness.run(&["impact"]);
    let store_path = store_path_from(&[
        &before,
        &before_invalidation,
        &cut,
        &cut_invalidation,
        &cutoff_changed,
        &after,
        &after_invalidation,
        &changed,
        &impact,
    ]);
    let store = persisted_store(store_path.clone());
    let context_id = context_id_from(&[
        &before,
        &before_invalidation,
        &cut,
        &cut_invalidation,
        &after,
        &after_invalidation,
        &changed,
        &impact,
    ]);
    let before_id =
        stable_uuid_from(&[&before, &cutoff_changed, &impact], &["D_before"]).map(DecisionId::from);
    let cut_id =
        stable_uuid_from(&[&cut, &cutoff_changed, &impact], &["D_cut"]).map(DecisionId::from);
    let after_id =
        stable_uuid_from(&[&after, &changed, &impact], &["D_after"]).map(DecisionId::from);
    let expected_initial_delta = sorted_present_decision_ids(&[before_id, cut_id]);
    let expected_recent_delta = sorted_present_decision_ids(&[after_id]);
    let expected_standing = sorted_present_decision_ids(&[before_id, cut_id, after_id]);

    assert_eq!(
        (
            store_path.is_some(),
            context_id.is_some(),
            before_id.is_some(),
            cut_id.is_some(),
            after_id.is_some(),
            persisted_invalidation_decision_ids(store.as_ref(), context_id),
            invalidation_decision_ids_from_output(&cutoff_changed, &["changed"]),
            invalidation_decision_ids_from_output(&changed, &["changed"]),
            invalidation_decision_ids_from_output(&repeated, &["changed"]),
            invalidation_decision_ids_from_output(&impact, &["standing_invalidations"]),
        ),
        (
            true,
            true,
            true,
            true,
            true,
            expected_standing.clone(),
            expected_initial_delta,
            expected_recent_delta,
            Vec::<DecisionId>::new(),
            expected_standing,
        )
    );
}

#[test]
fn cli_query_and_impact_emit_stable_ids_consumed_by_walk_verbs() {
    let harness = extraction_harness();
    let first = harness.run_with_stdin(&["record", "--understand-now"], AUTH_INITIAL);
    let second = harness.run_with_stdin(&["record", "--understand-now"], AUTH_RENAME);
    let query = harness.run(&["query", "where is the auth module"]);
    let impact = harness.run(&["impact"]);
    let intention_id = stable_id(&query, &["ship password-less login for acme", "G1"]);
    let entity_id = stable_id(&impact, &["auth module file", "E_auth_module"]);
    let store_path = store_path_from(&[&first, &second, &query, &impact]);
    let store = persisted_store(store_path.clone());
    let persisted_intention = intention_id
        .as_ref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(IntentionId::from);
    let persisted_entity = entity_id
        .as_ref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(EntityId::from);
    let by_intention_rows = persisted_decisions_for_intention(store.as_ref(), persisted_intention);
    let by_entity_rows = persisted_decisions_for_entity(store.as_ref(), persisted_entity);
    drop(store);
    let intention_arg = intention_id
        .as_deref()
        .unwrap_or("__missing_intention_id__");
    let entity_arg = entity_id.as_deref().unwrap_or("__missing_entity_id__");
    let by_intention = harness.run(&["by-intention", intention_arg]);
    let by_entity = harness.run(&["by-entity", entity_arg]);
    let wrong_intention_id = format!("{intention_arg}-wrong");
    let wrong_entity_id = format!("{entity_arg}-wrong");
    let wrong_intention = harness.run(&["by-intention", wrong_intention_id.as_str()]);
    let wrong_entity = harness.run(&["by-entity", wrong_entity_id.as_str()]);

    assert_eq!(
        (
            intention_id.is_some(),
            entity_id.is_some(),
            store_path.is_some(),
            by_intention_rows
                .iter()
                .map(|(_, description, _)| description.as_str())
                .collect::<Vec<_>>(),
            by_entity_rows
                .iter()
                .map(|(_, description, _)| description.as_str())
                .collect::<Vec<_>>(),
        ),
        (
            true,
            true,
            true,
            vec!["auth module at src/authentication.py"],
            vec![
                "auth module at src/authentication.py",
                "auth module at src/auth.py",
            ],
        )
    );
    assert!(combined(&by_intention).contains("ship password-less login for acme"));
    assert!(combined(&by_entity).contains("auth module file"));
    assert!(no_resolution(&wrong_intention));
    assert!(no_resolution(&wrong_entity));
}

#[test]
fn hosted_record_usage_receipt_flows_through_connected_extractor() {
    let harness = CliHarness::new();
    let cwd = tempfile::tempdir().expect("hosted record cwd");
    let transcript = "Goal G_hosted: show hosted usage. Entity E_hosted: receipt file path=receipt.rs. Decision D_hosted: hosted usage flows through connected extractor serves G_hosted based_on E_hosted.";
    let (endpoint, provider_requests) = spawn_one_shot_chat_completion_server(
        serde_json::json!({
            "summary": "hosted usage receipt",
            "goals": [{
                "key": "G_hosted",
                "text": "show hosted usage"
            }],
            "entities": [{
                "key": "E_hosted",
                "name": "receipt file",
                "path": "receipt.rs",
                "entity_type": "file"
            }],
            "decisions": [{
                "key": "D_hosted",
                "text": "hosted usage flows through connected extractor",
                "goal_key": "G_hosted",
                "entity_key": "E_hosted",
                "relation": null
            }],
            "observations": []
        }),
        3217,
        233,
    );
    let connect = harness.run(&[
        "model",
        "connect",
        "local-vllm",
        "--endpoint",
        &endpoint,
        "--model",
        "test-model",
        "--no-auth",
        "--consent",
    ]);
    assert_eq!(connect.code, 0, "{}", combined(&connect));

    let record = run_in_dir_with_stdin(
        &harness,
        cwd.path(),
        &["record", "--understand-now"],
        transcript,
    );
    let request = provider_requests
        .recv_timeout(Duration::from_secs(5))
        .expect("record must call the configured hosted provider through ConnectedModelExtractor");
    assert!(
        request.contains("hosted usage flows through connected extractor"),
        "provider request must be the production record prompt, not a test-only usage helper: {request}"
    );
    assert_eq!(record.code, 0, "{}", combined(&record));
    let json = output_json(&record).expect("record output json");
    assert_eq!(
        json.pointer("/captured/decisions").and_then(Value::as_u64),
        Some(1),
        "the hosted provider reply must still drive normal capture"
    );
    let calls = json
        .pointer("/usage/calls")
        .and_then(Value::as_array)
        .expect(
            "record receipt must expose usage from the shipped hosted ConnectedModelExtractor path",
        );
    assert_eq!(
        calls.len(),
        1,
        "single-chunk record must report exactly the hosted model call it made"
    );
    let call = &calls[0];
    assert_eq!(
        (
            call.pointer("/prompt_tokens").and_then(Value::as_u64),
            call.pointer("/completion_tokens").and_then(Value::as_u64),
            call.pointer("/total_tokens").and_then(Value::as_u64),
            json.pointer("/usage/total_tokens").and_then(Value::as_u64),
        ),
        (Some(3217), Some(233), Some(3450), Some(3450)),
        "the receipt must preserve the provider usage block carried by UnderstandingRuntime through ConnectedModelExtractor"
    );
}

#[test]
fn default_capture_context_follows_tenant_not_working_directory_cli_path() {
    let harness = extraction_harness();
    let second_machine = extraction_harness();
    let dir_x = tempfile::tempdir().expect("dir x");
    let dir_y = tempfile::tempdir().expect("dir y");
    let dir_z = tempfile::tempdir().expect("dir z");
    assert_cli_import_estimate_makes_no_provider_calls(&harness, dir_x.path());
    harness.providers.set_behavior(
        Provider::Claude,
        CompletionBehavior::SleepThenExtractHarvest,
    );
    let progress = harness.run_with_stdin_observing_provider_before_exit(
        &["record", "--understand-now", "--tenant-id", "progress-tenant"],
        "Goal G_progress: report shipped CLI progress. Entity E_progress: progress file path=progress.rs. Decision D_progress: CLI capture progress is visible serves G_progress based_on E_progress.",
        "capture progress: chunk 1/",
        Provider::Claude,
    );
    assert_eq!(progress.output.code, 0, "{}", combined(&progress.output));
    assert!(
        progress.observed_before_exit,
        "record/import progress must be observable before the shipped cg command exits; output={}",
        combined(&progress.output)
    );
    assert!(
        progress.observed_while_provider_running,
        "record/import progress must be emitted while the provider call is still running, not reconstructed after capture; output={}",
        combined(&progress.output)
    );
    assert!(
        combined(&progress.output).contains("CLI capture progress is visible"),
        "the progress-observed capture must still persist the memory it was recording"
    );
    harness
        .providers
        .set_behavior(Provider::Claude, CompletionBehavior::ExtractHarvest);
    let completions_before_first = harness.providers.completion_records().len();
    let first = run_in_dir_with_stdin(
        &harness,
        dir_x.path(),
        &["record", "--understand-now", "--tenant-id", "owner-tenant"],
        "Goal G1: converge by tenant. Entity E1: auth module path=src/auth.py. Decision D1: tenant memory lands together serves G1 based_on E1.",
    );
    let first_provider_records = completion_records_since(&harness, completions_before_first);
    let completions_before_second = harness.providers.completion_records().len();
    let second = run_in_dir_with_stdin(
        &harness,
        dir_y.path(),
        &["record", "--understand-now", "--tenant-id", "owner-tenant"],
        "Goal G2: converge by tenant again. Entity E2: config file path=config/app.toml. Decision D2: second tenant capture lands together serves G2 based_on E2.",
    );
    let second_provider_records = completion_records_since(&harness, completions_before_second);
    let owner_status_x = run_in_dir(
        &harness,
        dir_x.path(),
        &["status", "--tenant-id", "owner-tenant"],
    );
    let owner_status_y = run_in_dir(
        &harness,
        dir_y.path(),
        &["status", "--tenant-id", "owner-tenant"],
    );
    let other_tenant = run_in_dir_with_stdin(
        &harness,
        dir_x.path(),
        &[
            "record",
            "--understand-now",
            "--tenant-id",
            "other-owner-tenant",
        ],
        "Goal G3: isolate other tenant. Entity E3: billing file path=billing.toml. Decision D3: other tenant stays separate serves G3 based_on E3.",
    );
    let explicit = run_in_dir_with_stdin(
        &harness,
        dir_x.path(),
        &[
            "record",
            "--understand-now",
            "--tenant-id",
            "owner-tenant",
            "--context",
            "named-client",
        ],
        "Goal G4: explicit context. Entity E4: override file path=override.toml. Decision D4: explicit context stays isolated serves G4 based_on E4.",
    );
    let completions_before_second_machine = second_machine.providers.completion_records().len();
    let same_tenant_second_machine = run_in_dir_with_stdin(
        &second_machine,
        dir_z.path(),
        &["record", "--understand-now", "--tenant-id", "owner-tenant"],
        "Goal G7: converge from another machine. Entity E7: other machine path=machine.rs. Decision D7: tenant CLI memory from a second machine lands in the same context serves G7 based_on E7.",
    );
    let second_machine_provider_records =
        completion_records_since(&second_machine, completions_before_second_machine);

    for output in [
        &first,
        &second,
        &other_tenant,
        &explicit,
        &same_tenant_second_machine,
    ] {
        assert_eq!(output.code, 0, "{}", combined(output));
    }
    assert_cli_capture_usage_receipt(&first, &first_provider_records);
    assert_cli_capture_usage_receipt(&second, &second_provider_records);
    assert_cli_capture_usage_receipt(
        &same_tenant_second_machine,
        &second_machine_provider_records,
    );

    let first_context = context_id_from(&[&first]).expect("first record context id");
    let second_context = context_id_from(&[&second]).expect("second record context id");
    let other_context = context_id_from(&[&other_tenant]).expect("other tenant context id");
    let explicit_context = context_id_from(&[&explicit]).expect("explicit context id");
    let second_machine_context =
        context_id_from(&[&same_tenant_second_machine]).expect("second machine context id");

    assert_eq!(
        first_context, second_context,
        "same tenant must resolve to the same context across working directories"
    );
    assert_eq!(
        first_context, second_machine_context,
        "same tenant must resolve to the same context across independent CLI data roots"
    );
    for output in [&owner_status_x, &owner_status_y] {
        assert_eq!(output.code, 0, "{}", combined(output));
    }
    assert_eq!(context_id_from(&[&owner_status_x]), Some(first_context));
    assert_eq!(context_id_from(&[&owner_status_y]), Some(first_context));
    let owner_counts = memory_counts(&owner_status_x).expect("owner status counts");
    assert_eq!(
        (owner_counts.0, owner_counts.1),
        (2, 2),
        "tenant status must see both captures from both working directories"
    );
    assert_ne!(first_context, other_context);
    assert_ne!(first_context, explicit_context);

    let context_names = persisted_context_names(store_path_from(&[&first]).as_deref());
    assert!(
        context_names.iter().any(|name| name == "owner-tenant"),
        "tenant id itself must be the persisted default context name; contexts={context_names:?}"
    );
    assert!(
        !context_names
            .iter()
            .any(|name| name.starts_with("context:")),
        "default context must not be the working-directory hash; contexts={context_names:?}"
    );
    let second_machine_context_names =
        persisted_context_names(store_path_from(&[&same_tenant_second_machine]).as_deref());
    assert!(
        second_machine_context_names
            .iter()
            .any(|name| name == "owner-tenant"),
        "second machine must persist the tenant id itself as the default context name; contexts={second_machine_context_names:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sync_cli_push_pull_status_reports_remote_tenant_watermarks_and_pending() {
    let unavailable_remote = unavailable_authenticated_ticket("offline-cli-hub").await;
    let offline_harness = extraction_harness();
    let offline_record = offline_harness.run_with_stdin(
        &[
            "record", "--understand-now",
            "--remote-server",
            unavailable_remote.as_str(),
            "--tenant-id",
            "offline-cli-tenant",
        ],
        "Goal G0: offline sync must not block capture. Entity E0: offline file path=offline.rs. Decision D0: offline CLI capture survives serves G0 based_on E0.",
    );
    assert_record_contains(&offline_record, "offline CLI capture survives");
    let offline_query = offline_harness.run(&[
        "query",
        "--tenant-id",
        "offline-cli-tenant",
        "offline capture survives",
    ]);
    assert_eq!(offline_query.code, 0, "{}", combined(&offline_query));
    let offline_query_text = combined(&offline_query);
    assert!(
        offline_query_text.contains("offline CLI capture survives")
            && offline_query_text.contains("offline.rs"),
        "remote-configured capture must persist locally when the hub is unreachable: {offline_query_text}"
    );

    let fixture =
        cli_sync_fixture("sync_cli_push_pull_status_reports_remote_tenant_watermarks_and_pending")
            .await;
    let harness = extraction_harness();
    let record = harness.run_with_stdin(
        &[
            "record", "--understand-now",
            "--remote-server",
            fixture.remote_url.as_str(),
            "--tenant-id",
            fixture.tenant_id.as_str(),
        ],
        "Goal G1: sync from the CLI. Entity E1: laptop path=src/sync.rs. Decision D1: operator sync commands carry memory serves G1 based_on E1.",
    );
    assert_eq!(record.code, 0, "{}", combined(&record));
    let store_path = store_path_from(&[&record]).expect("record output should include store path");
    let expected_pending = {
        let store = persisted_store(Some(store_path.clone())).expect("open CLI store after record");
        let local_audit_rows = db_table_row_count(store.sync_database().as_ref(), "audit_log");
        assert!(
            local_audit_rows > 0,
            "fixture must produce local audit rows so the CLI sync path proves audit_log stays local"
        );
        let count = cli_expected_pending_push_change_count(
            &store,
            fixture.remote_url.as_str(),
            fixture.tenant_id.as_str(),
        ) as u64;
        assert!(
            count > 1,
            "fixture must create a real non-boolean pending count"
        );
        count
    };

    let pending_status = harness.run(&[
        "sync",
        "status",
        "--remote-server",
        fixture.remote_url.as_str(),
        "--tenant-id",
        fixture.tenant_id.as_str(),
    ]);
    assert_eq!(pending_status.code, 0, "{}", combined(&pending_status));
    let pending_json = output_json(&pending_status).expect("pending sync status JSON");
    assert_eq!(
        json_string_at(&pending_json, &["/sync/remote_url", "/remote_url"]),
        Some(fixture.remote_url.as_str())
    );
    assert_eq!(
        json_string_at(&pending_json, &["/sync/tenant_id", "/tenant_id"]),
        Some(fixture.tenant_id.as_str())
    );
    let pending_count = json_u64_at(
        &pending_json,
        &["/sync/pending_push_count", "/pending_push_count"],
    )
    .expect("pending push count");
    assert_eq!(
        pending_count, expected_pending,
        "pending status must report the exact live pending count, not a boolean sentinel: {pending_json}"
    );
    assert_eq!(
        json_bool_at(
            &pending_json,
            &["/sync/sync_client_constructed", "/sync_client_constructed"]
        ),
        Some(true)
    );

    let push = harness.run(&[
        "sync",
        "push",
        "--remote-server",
        fixture.remote_url.as_str(),
        "--tenant-id",
        fixture.tenant_id.as_str(),
    ]);
    assert_eq!(push.code, 0, "{}", combined(&push));
    assert_eq!(
        db_table_row_count(fixture.hub_db.as_ref(), "audit_log"),
        0,
        "shipped CLI sync push must register the production table contract and keep audit_log out of the hub"
    );
    let expected_drained = {
        let store = scoped_persisted_store(Some(store_path), &fixture.tenant_id)
            .expect("open scoped CLI store after push");
        cli_expected_pending_push_change_count(
            &store,
            fixture.remote_url.as_str(),
            fixture.tenant_id.as_str(),
        ) as u64
    };
    let pushed_status = harness.run(&[
        "sync",
        "status",
        "--remote-server",
        fixture.remote_url.as_str(),
        "--tenant-id",
        fixture.tenant_id.as_str(),
    ]);
    assert_eq!(pushed_status.code, 0, "{}", combined(&pushed_status));
    let pushed_json = output_json(&pushed_status).expect("pushed sync status JSON");
    assert!(
        json_u64_at(&pushed_json, &["/sync/push_watermark", "/push_watermark"]).unwrap_or(0) > 0,
        "push watermark must advance after CLI push: {pushed_json}"
    );
    assert_eq!(
        json_u64_at(
            &pushed_json,
            &["/sync/pending_push_count", "/pending_push_count"]
        ),
        Some(expected_drained)
    );
    assert_eq!(expected_drained, 0);
    let hub_schema_installed_at = schema_version_installed_at(fixture.hub_db.as_ref());

    let second_edge = CliHarness::new();
    let second_edge_store_path = second_edge.data_root.path().join("context-graph.db");
    let divergent_schema_installed_at = force_divergent_schema_version(&second_edge_store_path);
    let pull = second_edge.run(&[
        "sync",
        "pull",
        "--remote-server",
        fixture.remote_url.as_str(),
        "--tenant-id",
        fixture.tenant_id.as_str(),
    ]);
    assert_eq!(pull.code, 0, "{}", combined(&pull));
    let pull_json = output_json(&pull).expect("pull sync JSON");
    assert_eq!(
        json_array_len_at(
            &pull_json,
            &["/sync/apply_result/conflicts", "/apply_result/conflicts"]
        ),
        Some(0),
        "cg sync pull must report the real ApplyResult from the declared two-way KEEP FIRST schema policy: {pull_json}"
    );
    assert!(
        json_u64_at(
            &pull_json,
            &[
                "/sync/apply_result/applied_rows",
                "/apply_result/applied_rows"
            ]
        )
        .unwrap_or(0)
            > 0,
        "the declared KEEP FIRST pull must apply hub rows to the divergent edge: {pull_json}"
    );
    {
        let post_pull_db =
            Database::open(&second_edge_store_path).expect("open second edge after pull");
        assert_eq!(
            schema_version_installed_at(&post_pull_db),
            hub_schema_installed_at,
            "cg sync pull must converge the divergent schema row to the hub's accepted KEEP FIRST value"
        );
        assert_ne!(
            schema_version_installed_at(&post_pull_db),
            divergent_schema_installed_at,
            "the deliberately divergent local schema row must not survive a two-way KEEP FIRST pull"
        );
    }
    let pulled_query = second_edge.run(&[
        "query",
        "--tenant-id",
        fixture.tenant_id.as_str(),
        "operator sync commands",
    ]);
    assert_eq!(pulled_query.code, 0, "{}", combined(&pulled_query));
    let pulled_text = combined(&pulled_query);
    assert!(
        pulled_text.contains("operator sync commands carry memory")
            && pulled_text.contains("src/sync.rs"),
        "cg sync pull must materialize pushed memory on a second CLI edge, not just return success: {pulled_text}"
    );
}

#[test]
fn cli_query_cross_context_isolation_default_and_opt_in() {
    let harness = extraction_harness();
    let client_a = harness.run_with_stdin(
        &["record", "--understand-now", "--context", "client-a"],
        CLIENT_A_SOX_SESSION,
    );
    let client_b = harness.run_with_stdin(
        &["record", "--understand-now", "--context", "client-b"],
        CLIENT_B_PUBLISH_SESSION,
    );
    assert_record_contains(&client_a, "SOX approval required before promote");
    assert_record_contains(&client_b, "hold 10 min before promote");
    let isolated = harness.run(&["query", "--context", "client-b", "duplicate publish"]);
    let crossed = harness.run(&["query", "--cross-context", "client-a,client-b", "publish"]);

    assert!(
        combined(&isolated).contains("hold 10 min before promote"),
        "{}",
        combined(&isolated)
    );
    assert!(!combined(&isolated).contains("SOX"));
    assert!(combined(&crossed).contains("client-a"));
    assert!(combined(&crossed).contains("client-b"));
}

#[test]
fn cross_context_isolation_holds_on_explicit_context_in_shared_directory() {
    let harness = extraction_harness();
    let client_a = harness.run_with_stdin(
        &["record", "--understand-now", "--context", "client-a"],
        CLIENT_A_SOX_SESSION,
    );
    let client_b = harness.run_with_stdin(
        &["record", "--understand-now", "--context", "client-b"],
        CLIENT_B_PUBLISH_SESSION,
    );
    assert_record_contains(&client_a, "SOX approval required before promote");
    assert_record_contains(&client_b, "hold 10 min before promote");
    let output = harness.run(&["query", "--context", "client-b", "duplicate publish"]);

    assert!(combined(&output).contains("client-b"));
    assert!(
        combined(&output).contains("hold 10 min before promote"),
        "{}",
        combined(&output)
    );
    assert!(!combined(&output).contains("SOX"));
}

#[test]
fn read_commands_do_not_create_missing_contexts() {
    let harness = extraction_harness();
    let missing_id = uuid::Uuid::now_v7().to_string();
    let query = harness.run(&["query", "--context", "typo-client", "anything"]);
    let impact = harness.run(&["impact", "--context", "typo-client"]);
    let changed = harness.run(&["changed", "--context", "typo-client"]);
    let by_intention = harness.run(&[
        "by-intention",
        "--context",
        "typo-client",
        missing_id.as_str(),
    ]);
    let by_entity = harness.run(&["by-entity", "--context", "typo-client", missing_id.as_str()]);
    let store_path = store_path_from(&[&query, &impact, &changed, &by_intention, &by_entity]);

    assert!(combined(&query).contains("no matching memory"));
    assert!(combined(&impact).contains("nothing stale"));
    assert!(combined(&changed).contains("nothing newly stale"));
    assert!(no_resolution(&by_intention));
    assert!(no_resolution(&by_entity));
    assert_eq!(
        persisted_context_names(store_path.as_deref()),
        Vec::<String>::new()
    );
}

#[test]
fn cli_first_capture_load_is_observable_not_silent() {
    let harness = extraction_harness();
    let first = harness.run_with_stdin(&["record", "--understand-now"], "first auth capture");
    let second = harness.run_with_stdin(&["record", "--understand-now"], "second auth capture");

    assert!(combined(&first).contains("loading bundled text embedder"));
    assert!(!combined(&second).contains("loading bundled text embedder"));
}
