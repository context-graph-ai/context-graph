mod support;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cg_catalog::{CredentialSource, ModelConnectionCatalog};
use serde_json::{Map, Value, json};
use tempfile::TempDir;

use support::{
    CommandOutput, base64_basic, catalog_disk_contains, cg_bin, parse_single_json,
    parse_single_json_object,
};

const CONNECTION_NAME: &str = "byok-openai";
const PROVIDER: &str = "openai";
const PROVIDER_MODEL: &str = "gpt-byok-acceptance";
const ENV_KEY: &str = "CG_BYOK_ACCEPTANCE_KEY";
const LOCAL_CONNECTION_NAME: &str = "byok-local-vllm";
const LOCAL_PROVIDER: &str = "local-vllm";
const LOCAL_PROVIDER_MODEL: &str = "llama-local-acceptance";

struct ByokHarness {
    data_root: TempDir,
}

impl ByokHarness {
    fn new() -> Self {
        Self {
            data_root: TempDir::new().expect("cg data root"),
        }
    }

    fn root(&self) -> &Path {
        self.data_root.path()
    }

    fn run(&self, args: &[&str]) -> CommandOutput {
        self.run_with_env(args, &[])
    }

    fn run_with_env(&self, args: &[&str], envs: &[(&str, &str)]) -> CommandOutput {
        let mut command = Command::new(cg_bin());
        command
            .args(args)
            .env("CG_DATA_ROOT", self.root())
            .env("HOME", self.root())
            .env("PATH", "/usr/bin:/bin")
            .env_remove("OPENAI_API_KEY")
            .env_remove("CODEX_API_KEY")
            .env_remove("ANTHROPIC_API_KEY")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (name, value) in envs {
            command.env(name, value);
        }
        let output = command.output().expect("run compiled cg binary");
        CommandOutput {
            code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }
    }
}

#[derive(Clone, Debug)]
struct CapturedRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: String,
}

struct OpenAiStub {
    base_url: String,
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
    shutdown: Option<mpsc::Sender<()>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl OpenAiStub {
    fn chat(reply: &str) -> Self {
        let response = json!({
            "id": unique_token("chatcmpl"),
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply,
                },
                "finish_reason": "stop",
            }],
        });
        Self::new(200, response.to_string())
    }

    fn new(status: u16, body: String) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local HTTP stub");
        let addr = listener.local_addr().expect("local HTTP stub address");
        listener
            .set_nonblocking(true)
            .expect("set local HTTP stub nonblocking");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let thread_requests = Arc::clone(&requests);
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            loop {
                if shutdown_rx.try_recv().is_ok() {
                    break;
                }
                match listener.accept() {
                    Ok((stream, _)) => {
                        handle_http_request(stream, Arc::clone(&thread_requests), status, &body);
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            base_url: format!("http://{addr}/openai/v1"),
            requests,
            shutdown: Some(shutdown_tx),
            handle: Some(handle),
        }
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn request_count(&self) -> usize {
        self.requests.lock().expect("request log").len()
    }

    fn requests(&self) -> Vec<CapturedRequest> {
        self.requests.lock().expect("request log").clone()
    }
}

impl Drop for OpenAiStub {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn handle_http_request(
    mut stream: TcpStream,
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
    status: u16,
    body: &str,
) {
    if let Some(request) = read_http_request(&mut stream) {
        requests.lock().expect("request log").push(request);
    }
    let reason = if status == 200 { "OK" } else { "Error" };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
}

fn read_http_request(stream: &mut TcpStream) -> Option<CapturedRequest> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut chunk = [0_u8; 512];
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..read]);
        if let Some(position) = find_header_end(&bytes) {
            break position;
        }
        if bytes.len() > 64 * 1024 {
            return None;
        }
    };
    let header_text = String::from_utf8_lossy(&bytes[..header_end]).into_owned();
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or_default().to_string();
    let path = request_parts.next().unwrap_or_default().to_string();
    let mut headers = BTreeMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.to_ascii_lowercase(), value.trim().to_string());
        }
    }
    let content_length = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let body_start = header_end + 4;
    let mut body = bytes.get(body_start..).unwrap_or_default().to_vec();
    while body.len() < content_length {
        let mut chunk = vec![0_u8; content_length - body.len()];
        stream.read_exact(&mut chunk).ok()?;
        body.extend_from_slice(&chunk);
    }
    body.truncate(content_length);
    Some(CapturedRequest {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn unique_token(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    format!("{prefix}-{nanos}-{}", std::process::id())
}

fn dead_base_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind dead endpoint probe");
    let addr = listener.local_addr().expect("dead endpoint address");
    drop(listener);
    format!("http://{addr}/openai/v1")
}

fn byok_connect_args(endpoint: &str) -> Vec<&str> {
    vec![
        "model",
        "connect",
        "--connection",
        CONNECTION_NAME,
        "--provider",
        PROVIDER,
        "--endpoint",
        endpoint,
        "--model",
        PROVIDER_MODEL,
        "--env-key",
        ENV_KEY,
        "--format",
        "json",
    ]
}

fn byok_connect_api_key_args<'a>(endpoint: &'a str, api_key: &'a str) -> Vec<&'a str> {
    vec![
        "model",
        "connect",
        "--connection",
        CONNECTION_NAME,
        "--provider",
        PROVIDER,
        "--endpoint",
        endpoint,
        "--model",
        PROVIDER_MODEL,
        "--api-key",
        api_key,
        "--format",
        "json",
    ]
}

fn local_vllm_connect_args(endpoint: &str) -> Vec<&str> {
    vec![
        "model",
        "connect",
        "--connection",
        LOCAL_CONNECTION_NAME,
        "--provider",
        LOCAL_PROVIDER,
        "--endpoint",
        endpoint,
        "--model",
        LOCAL_PROVIDER_MODEL,
        "--no-auth",
        "--format",
        "json",
    ]
}

#[test]
fn cli_connect_accepts_plan_public_positional_and_token_home_flags() {
    let harness = ByokHarness::new();
    let stub = OpenAiStub::chat("public cli connect must not call endpoint");
    let secret = unique_token("byok-secret");

    let output = harness.run_with_env(
        &[
            "model",
            "connect",
            CONNECTION_NAME,
            "--provider",
            PROVIDER,
            "--endpoint",
            stub.base_url(),
            "--model",
            PROVIDER_MODEL,
            "--env",
            ENV_KEY,
            "--consent",
            "--format",
            "json",
        ],
        &[(ENV_KEY, &secret)],
    );

    let object = assert_json_success(&output, "cg model connect public env syntax");
    assert_eq!(
        object.get("connection").and_then(Value::as_str),
        Some(CONNECTION_NAME)
    );
    assert_eq!(
        object.get("provider").and_then(Value::as_str),
        Some(PROVIDER)
    );
    assert_eq!(object.get("env_key").and_then(Value::as_str), Some(ENV_KEY));
    assert_no_secret_leak(&output, harness.root(), &secret);
    assert_eq!(
        stub.request_count(),
        0,
        "public connect syntax must record configuration without spending a provider call"
    );

    let stored_connection = unique_token("byok-stored-public");
    let stored_model = unique_token("gpt-byok-public");
    let stored = harness.run(&[
        "model",
        "connect",
        &stored_connection,
        "--provider",
        PROVIDER,
        "--endpoint",
        stub.base_url(),
        "--model",
        &stored_model,
        "--api-key",
        &secret,
        "--store",
        "--consent",
        "--format",
        "json",
    ]);
    let stored_object = assert_json_success(&stored, "cg model connect public store syntax");
    assert_eq!(
        stored_object
            .get("credential_source")
            .and_then(Value::as_str),
        Some("stored_secret")
    );
    assert_no_secret_leak(&stored, harness.root(), &secret);
    assert_eq!(
        stub.request_count(),
        0,
        "--store connect must not spend a provider call"
    );

    let local_stub = OpenAiStub::chat("local provider positional");
    let local = harness.run(&[
        "model",
        "connect",
        LOCAL_PROVIDER,
        "--endpoint",
        local_stub.base_url(),
        "--model",
        LOCAL_PROVIDER_MODEL,
        "--no-auth",
        "--consent",
        "--format",
        "json",
    ]);
    let local_object = assert_json_success(&local, "cg model connect provider positional syntax");
    assert_eq!(
        local_object.get("connection").and_then(Value::as_str),
        Some(LOCAL_PROVIDER)
    );
    assert_eq!(
        local_object.get("provider").and_then(Value::as_str),
        Some(LOCAL_PROVIDER)
    );
    assert_eq!(
        local_object.get("auth_scheme").and_then(Value::as_str),
        Some("none")
    );
    assert_eq!(
        local_stub.request_count(),
        0,
        "provider positional connect must not spend a provider call"
    );
}

fn assert_json_success(output: &CommandOutput, context: &str) -> Map<String, Value> {
    assert_eq!(
        output.code, 0,
        "{context} must exit zero, got code={} stdout={:?} stderr={:?}",
        output.code, output.stdout, output.stderr
    );
    let parsed = parse_single_json_object(&output.stdout);
    assert!(
        parsed.is_ok(),
        "{context} must emit exactly one JSON object, got stdout={:?} stderr={:?}",
        output.stdout,
        output.stderr
    );
    parsed.unwrap_or_default()
}

fn assert_json_condition(
    output: &CommandOutput,
    expected_code: i32,
    expected_condition: &str,
    expected_bucket: Option<&str>,
    context: &str,
) -> Map<String, Value> {
    assert_eq!(
        output.code, expected_code,
        "{context} must exit {expected_code}, got code={} stdout={:?} stderr={:?}",
        output.code, output.stdout, output.stderr
    );
    let parsed = parse_single_json_object(&output.stdout);
    assert!(
        parsed.is_ok(),
        "{context} must emit exactly one JSON object, got stdout={:?} stderr={:?}",
        output.stdout,
        output.stderr
    );
    let object = parsed.unwrap_or_default();
    assert_eq!(
        object.get("condition").and_then(Value::as_str),
        Some(expected_condition),
        "{context} must expose condition `{expected_condition}`"
    );
    if let Some(expected_bucket) = expected_bucket {
        assert_eq!(
            object.get("bucket").and_then(Value::as_str),
            Some(expected_bucket),
            "{context} must expose bucket `{expected_bucket}`"
        );
    }
    object
}

fn assert_no_secret_leak(output: &CommandOutput, data_root: &Path, secret: &str) {
    assert!(
        !output.stdout.contains(secret) && !output.stderr.contains(secret),
        "secret value must never be printed; stdout={:?} stderr={:?}",
        output.stdout,
        output.stderr
    );
    assert!(
        !catalog_disk_contains(data_root, secret)
            && !catalog_disk_contains(data_root, &base64_basic(secret)),
        "secret value must never be written to the catalog or local cg state"
    );
}

fn assert_cli_success_no_secret(
    output: &CommandOutput,
    data_root: &Path,
    secret: &str,
    context: &str,
) {
    assert_eq!(
        output.code, 0,
        "{context} must exit zero, got code={} stdout={:?} stderr={:?}",
        output.code, output.stdout, output.stderr
    );
    assert_no_secret_leak(output, data_root, secret);
}

fn assert_list_names_byok_connection(output: &CommandOutput, endpoint: &str, secret: &str) {
    assert_eq!(
        output.code, 0,
        "cg model list must exit zero, got code={} stdout={:?} stderr={:?}",
        output.code, output.stdout, output.stderr
    );
    let parsed = parse_single_json(&output.stdout);
    assert!(
        parsed.is_ok(),
        "cg model list --format json must emit valid JSON, got stdout={:?} stderr={:?}",
        output.stdout,
        output.stderr
    );
    let rendered = parsed.unwrap_or(Value::Null).to_string();
    for needle in [CONNECTION_NAME, PROVIDER, PROVIDER_MODEL, endpoint] {
        assert!(
            rendered.contains(needle),
            "cg model list JSON must include `{needle}` for the named BYOK connection, got {rendered}"
        );
    }
    assert!(
        !rendered.contains(secret),
        "cg model list JSON must not print the stored secret"
    );
}

fn assert_status_names_byok_connection(
    output: &CommandOutput,
    endpoint: &str,
    secret: &str,
    context: &str,
) -> Map<String, Value> {
    let object = assert_json_success(output, context);
    for (field, expected) in [
        ("connection", CONNECTION_NAME),
        ("provider", PROVIDER),
        ("model", PROVIDER_MODEL),
        ("endpoint", endpoint),
    ] {
        assert_eq!(
            object.get(field).and_then(Value::as_str),
            Some(expected),
            "{context} must expose {field}={expected:?} for scriptable BYOK status, got {object:?}"
        );
    }
    let rendered = Value::Object(object.clone()).to_string();
    assert!(
        !rendered.contains(secret),
        "{context} must not print the stored secret"
    );
    object
}

fn assert_openai_request(request: &CapturedRequest, endpoint_path: &str, secret: &str) {
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, endpoint_path);
    let expected_auth = format!("Bearer {secret}");
    assert_eq!(
        request.headers.get("authorization").map(String::as_str),
        Some(expected_auth.as_str())
    );
    assert!(
        !request.body.contains(secret),
        "request body must not contain the API key"
    );
    let body = serde_json::from_str::<Value>(&request.body).expect("OpenAI request JSON body");
    assert_eq!(
        body.get("model").and_then(Value::as_str),
        Some(PROVIDER_MODEL)
    );
    assert!(
        body.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| !messages.is_empty()),
        "OpenAI-compatible request must carry a non-empty messages array"
    );
}

fn assert_no_auth_openai_request(
    request: &CapturedRequest,
    endpoint_path: &str,
    expected_model: &str,
) {
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, endpoint_path);
    assert_eq!(
        request.headers.get("authorization"),
        None,
        "no-auth local-vllm connections must omit Authorization entirely"
    );
    let body = serde_json::from_str::<Value>(&request.body).expect("OpenAI request JSON body");
    assert_eq!(
        body.get("model").and_then(Value::as_str),
        Some(expected_model)
    );
    assert!(
        body.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| !messages.is_empty()),
        "OpenAI-compatible request must carry a non-empty messages array"
    );
}

#[test]
fn cli_connect_registers_named_connection_with_chosen_provider_and_own_url() {
    let harness = ByokHarness::new();
    let stub = OpenAiStub::chat("connect should not call the endpoint");
    let secret = unique_token("byok-secret");
    let mut args = byok_connect_args(stub.base_url());
    args.push("--consent");

    let output = harness.run_with_env(&args, &[(ENV_KEY, &secret)]);

    let object = assert_json_success(&output, "cg model connect");
    assert_eq!(
        object.get("connection").and_then(Value::as_str),
        Some(CONNECTION_NAME)
    );
    assert_eq!(
        object.get("provider").and_then(Value::as_str),
        Some(PROVIDER)
    );
    assert_eq!(
        object.get("endpoint").and_then(Value::as_str),
        Some(stub.base_url())
    );
    assert_eq!(
        object.get("model").and_then(Value::as_str),
        Some(PROVIDER_MODEL)
    );
    assert_eq!(object.get("env_key").and_then(Value::as_str), Some(ENV_KEY));
    assert_no_secret_leak(&output, harness.root(), &secret);
    assert_eq!(
        stub.request_count(),
        0,
        "connect records configuration and consent but must not spend a live HTTP call"
    );

    let catalog = ModelConnectionCatalog::open(harness.root());
    let connection = catalog
        .read_connection(CONNECTION_NAME)
        .expect("connect records named connection");
    assert_eq!(connection.name, CONNECTION_NAME);
    assert_eq!(connection.exposed_secret, None);
    assert!(connection.consent_recorded);
    assert_eq!(
        connection.credential_reference.source,
        CredentialSource::EnvVar {
            env_var: ENV_KEY.to_string()
        }
    );
    let model = catalog
        .read_model(PROVIDER_MODEL)
        .expect("connect registers hosted model");
    assert_eq!(model.connection_name.as_deref(), Some(CONNECTION_NAME));
    let endpoint = model.hosted_endpoint.expect("hosted endpoint metadata");
    assert_eq!(endpoint.base_url, stub.base_url());
    assert_eq!(endpoint.provider, PROVIDER);
    assert_eq!(endpoint.provider_model, PROVIDER_MODEL);

    let local_stub = OpenAiStub::chat("local-vllm-selftest");
    let mut local_args = local_vllm_connect_args(local_stub.base_url());
    local_args.push("--consent");
    let local_connect = harness.run(&local_args);
    let local_object = assert_json_success(&local_connect, "cg model connect local-vllm no-auth");
    assert_eq!(
        local_object.get("connection").and_then(Value::as_str),
        Some(LOCAL_CONNECTION_NAME)
    );
    assert_eq!(
        local_object.get("provider").and_then(Value::as_str),
        Some(LOCAL_PROVIDER)
    );
    assert_eq!(
        local_object.get("endpoint").and_then(Value::as_str),
        Some(local_stub.base_url())
    );
    assert_eq!(
        local_object.get("model").and_then(Value::as_str),
        Some(LOCAL_PROVIDER_MODEL)
    );
    assert_eq!(
        local_object.get("auth_scheme").and_then(Value::as_str),
        Some("none"),
        "local-vllm CLI connect must record an explicit None auth scheme"
    );
    assert_eq!(
        local_stub.request_count(),
        0,
        "local-vllm connect records configuration and consent but must not spend a live HTTP call"
    );
    let local_selftest = harness.run(&[
        "model",
        "selftest",
        "--connection",
        LOCAL_CONNECTION_NAME,
        "--format",
        "json",
    ]);
    let local_selftest_object =
        assert_json_success(&local_selftest, "cg model selftest local-vllm no-auth");
    assert_eq!(
        local_selftest_object.get("reply").and_then(Value::as_str),
        Some("local-vllm-selftest"),
        "local-vllm selftest must return the actual stub reply"
    );
    let local_request = local_stub
        .requests()
        .pop()
        .expect("local-vllm selftest sends one request");
    assert_no_auth_openai_request(
        &local_request,
        "/openai/v1/chat/completions",
        LOCAL_PROVIDER_MODEL,
    );
}

#[test]
fn cli_connect_then_status_then_selftest_success_state_exit_zero_and_json_stable() {
    let harness = ByokHarness::new();
    let stub = OpenAiStub::chat("status-ready-sentinel");
    let secret = unique_token("byok-secret");
    let mut args = byok_connect_api_key_args(stub.base_url(), &secret);
    args.push("--consent");
    let connect = harness.run(&args);
    let connect_object = assert_json_success(&connect, "cg model connect with supplied API key");
    assert_eq!(
        connect_object
            .get("credential_source")
            .and_then(Value::as_str),
        Some("stored_secret"),
        "pasted BYOK tokens must become stored credential references, not env-var links"
    );
    assert_no_secret_leak(&connect, harness.root(), &secret);
    let catalog = ModelConnectionCatalog::open(harness.root());
    let connection = catalog
        .read_connection(CONNECTION_NAME)
        .expect("api-key connect records named connection");
    assert_eq!(connection.exposed_secret, None);
    assert!(matches!(
        connection.credential_reference.source,
        CredentialSource::StoredSecret { .. }
    ));
    let show_json = harness.run(&["model", "show", CONNECTION_NAME, "--format", "json"]);
    let show_object = assert_json_success(&show_json, "cg model show named BYOK connection");
    assert_eq!(
        show_object.get("connection").and_then(Value::as_str),
        Some(CONNECTION_NAME)
    );
    assert_eq!(
        show_object.get("provider").and_then(Value::as_str),
        Some(PROVIDER)
    );
    assert_eq!(
        show_object.get("endpoint").and_then(Value::as_str),
        Some(stub.base_url())
    );
    assert_eq!(
        show_object.get("model").and_then(Value::as_str),
        Some(PROVIDER_MODEL)
    );
    assert_eq!(
        show_object.get("credential_source").and_then(Value::as_str),
        Some("stored_secret")
    );
    assert_no_secret_leak(&show_json, harness.root(), &secret);

    let list_json = harness.run(&["model", "list", "--format", "json"]);
    assert_list_names_byok_connection(&list_json, stub.base_url(), &secret);
    assert_no_secret_leak(&list_json, harness.root(), &secret);
    let list_text = harness.run(&["model", "list", "--format", "text"]);
    assert_cli_success_no_secret(
        &list_text,
        harness.root(),
        &secret,
        "cg model list --format text",
    );
    for needle in [CONNECTION_NAME, PROVIDER, PROVIDER_MODEL, stub.base_url()] {
        assert!(
            list_text.stdout.contains(needle),
            "cg model list text must include `{needle}` for the named BYOK connection, got {:?}",
            list_text.stdout
        );
    }

    let show_text = harness.run(&["model", "show", CONNECTION_NAME, "--format", "text"]);
    assert_cli_success_no_secret(
        &show_text,
        harness.root(),
        &secret,
        "cg model show --format text",
    );
    let status_json = harness.run(&[
        "model",
        "status",
        "--connection",
        CONNECTION_NAME,
        "--format",
        "json",
    ]);
    assert_status_names_byok_connection(
        &status_json,
        stub.base_url(),
        &secret,
        "cg model status --format json before proof",
    );
    assert_no_secret_leak(&status_json, harness.root(), &secret);
    let status_text = harness.run(&[
        "model",
        "status",
        "--connection",
        CONNECTION_NAME,
        "--format",
        "text",
    ]);
    assert_cli_success_no_secret(
        &status_text,
        harness.root(),
        &secret,
        "cg model status --format text before proof",
    );
    for needle in [CONNECTION_NAME, PROVIDER, PROVIDER_MODEL, stub.base_url()] {
        assert!(
            status_text.stdout.contains(needle),
            "cg model status text must include `{needle}` for the named BYOK connection, got {:?}",
            status_text.stdout
        );
    }

    let selftest = harness.run(&[
        "model",
        "selftest",
        "--connection",
        CONNECTION_NAME,
        "--format",
        "json",
    ]);
    let selftest_object = assert_json_success(&selftest, "cg model selftest before ready status");
    assert_eq!(
        selftest_object.get("reply").and_then(Value::as_str),
        Some("status-ready-sentinel"),
        "selftest JSON must show the actual OpenAI-compatible stub reply"
    );
    assert_eq!(
        selftest_object.get("connection").and_then(Value::as_str),
        Some(CONNECTION_NAME)
    );
    assert_no_secret_leak(&selftest, harness.root(), &secret);
    assert_eq!(
        stub.request_count(),
        1,
        "consented selftest must spend exactly one live HTTP call"
    );
    let request = stub
        .requests()
        .pop()
        .expect("consented selftest sends one request");
    assert_openai_request(&request, "/openai/v1/chat/completions", &secret);

    let ready = harness.run(&[
        "model",
        "status",
        "--connection",
        CONNECTION_NAME,
        "--format",
        "json",
    ]);
    let ready_object = assert_status_names_byok_connection(
        &ready,
        stub.base_url(),
        &secret,
        "ready cg model status",
    );
    assert_eq!(
        ready_object.get("state").and_then(Value::as_str),
        Some("ready")
    );
    assert_eq!(ready_object.get("degraded"), Some(&json!(false)));
    assert_eq!(ready_object.get("last_error"), Some(&Value::Null));
    assert_no_secret_leak(&ready, harness.root(), &secret);

    let dead_endpoint = dead_base_url();
    let mut reconnect_args = byok_connect_api_key_args(&dead_endpoint, &secret);
    reconnect_args.push("--consent");
    let reconnect = harness.run(&reconnect_args);
    assert_json_success(&reconnect, "cg model reconnect to unreachable endpoint");
    let degraded_selftest = harness.run(&[
        "model",
        "selftest",
        "--connection",
        CONNECTION_NAME,
        "--format",
        "json",
    ]);
    assert_json_condition(
        &degraded_selftest,
        6,
        "live_call_failed",
        Some("unreachable"),
        "unreachable cg model selftest",
    );
    let degraded = harness.run(&[
        "model",
        "status",
        "--connection",
        CONNECTION_NAME,
        "--format",
        "json",
    ]);
    let degraded_object = assert_json_condition(
        &degraded,
        6,
        "live_call_failed",
        Some("unreachable"),
        "degraded cg model status",
    );
    assert_eq!(
        degraded_object.get("state").and_then(Value::as_str),
        Some("degraded")
    );
    assert_eq!(degraded_object.get("degraded"), Some(&json!(true)));
    let last_error = degraded_object
        .get("last_error")
        .and_then(Value::as_object)
        .expect("degraded status carries last_error object");
    assert_eq!(
        last_error.get("condition").and_then(Value::as_str),
        Some("live_call_failed")
    );
    assert_eq!(
        last_error.get("bucket").and_then(Value::as_str),
        Some("unreachable")
    );
    assert_no_secret_leak(&degraded, harness.root(), &secret);
}

#[test]
fn cli_model_status_after_passing_selftest_refreshes_live_reachability() {
    let harness = ByokHarness::new();
    let stub = OpenAiStub::chat("status-refresh-sentinel");
    let endpoint = stub.base_url().to_string();
    let secret = unique_token("byok-secret");
    let mut args = byok_connect_api_key_args(&endpoint, &secret);
    args.push("--consent");
    let connect = harness.run(&args);
    assert_json_success(&connect, "cg model connect before stale-status probe");

    let selftest = harness.run(&[
        "model",
        "selftest",
        "--connection",
        CONNECTION_NAME,
        "--format",
        "json",
    ]);
    assert_json_success(&selftest, "cg model selftest before stale-status probe");
    assert_eq!(
        stub.request_count(),
        1,
        "initial selftest must spend exactly one live provider call"
    );

    drop(stub);

    let status = harness.run(&[
        "model",
        "status",
        "--connection",
        CONNECTION_NAME,
        "--format",
        "json",
    ]);
    let object = assert_json_condition(
        &status,
        6,
        "live_call_failed",
        Some("unreachable"),
        "cg model status after previously passing endpoint goes down",
    );
    assert_eq!(
        object.get("state").and_then(Value::as_str),
        Some("degraded")
    );
    assert_eq!(object.get("connection_works"), Some(&json!(false)));
    assert_eq!(
        object.get("proof_status").and_then(Value::as_str),
        Some("failed")
    );
    assert_eq!(
        object.get("endpoint").and_then(Value::as_str),
        Some(endpoint.as_str())
    );
    assert_no_secret_leak(&status, harness.root(), &secret);
}

#[test]
fn cli_status_when_not_connected_is_a_distinct_not_connected_condition() {
    let harness = ByokHarness::new();
    let output = harness.run(&["model", "status", "--format", "json"]);

    assert_json_condition(
        &output,
        4,
        "not_connected",
        Some("not_connected"),
        "not-connected cg model status",
    );
}

#[test]
fn cli_selftest_without_consent_is_a_distinct_consent_refused_condition_and_zero_requests() {
    let harness = ByokHarness::new();
    let stub = OpenAiStub::chat("no consent should not call the endpoint");
    let secret = unique_token("byok-secret");
    let connect = harness.run_with_env(&byok_connect_args(stub.base_url()), &[(ENV_KEY, &secret)]);
    let connect_object = assert_json_success(&connect, "cg model connect without consent");
    assert_eq!(
        connect_object.get("connection").and_then(Value::as_str),
        Some(CONNECTION_NAME)
    );
    assert_no_secret_leak(&connect, harness.root(), &secret);
    let catalog = ModelConnectionCatalog::open(harness.root());
    let connection = catalog
        .read_connection(CONNECTION_NAME)
        .expect("connect without consent records named connection");
    assert!(
        !connection.consent_recorded,
        "cg model connect without --consent must not record spend consent"
    );

    let output = harness.run_with_env(
        &[
            "model",
            "selftest",
            "--connection",
            CONNECTION_NAME,
            "--format",
            "json",
        ],
        &[(ENV_KEY, &secret)],
    );

    assert_json_condition(
        &output,
        5,
        "consent_not_recorded",
        Some("auth"),
        "selftest without consent",
    );
    assert_eq!(
        stub.request_count(),
        0,
        "missing consent must stop before opening an HTTP connection"
    );
    assert_no_secret_leak(&output, harness.root(), &secret);
}

#[test]
fn cli_selftest_against_unreachable_endpoint_is_a_distinct_unreachable_condition() {
    let harness = ByokHarness::new();
    let secret = unique_token("byok-secret");
    let dead_endpoint = dead_base_url();
    let mut connect_args = byok_connect_args(&dead_endpoint);
    connect_args.push("--consent");
    let connect = harness.run_with_env(&connect_args, &[(ENV_KEY, &secret)]);
    assert_json_success(&connect, "cg model connect to unreachable endpoint");

    let output = harness.run_with_env(
        &[
            "model",
            "selftest",
            "--connection",
            CONNECTION_NAME,
            "--format",
            "json",
        ],
        &[(ENV_KEY, &secret)],
    );

    let object = assert_json_condition(
        &output,
        6,
        "live_call_failed",
        Some("unreachable"),
        "selftest against unreachable endpoint",
    );
    assert_eq!(object.get("degraded"), Some(&json!(true)));
    assert_no_secret_leak(&output, harness.root(), &secret);
}
