mod support;

use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, mpsc};
use std::thread;
use std::time::Duration;

use context_graph::{
    ContextId, CreateDecision, CreateIntention, DecisionId, DecisionStatus, DiscoveryFilter,
    EntityId, IntentionId, IntentionOrigin, IntentionStatus, Store, StoreConfig,
    context_graph_pending_push_change_count, context_graph_sync_conflict_policies,
};
use contextdb_core::Value as DbValue;
use contextdb_engine::Database;
use contextdb_server::{SyncClient, SyncServer};
use serde_json::Value;
use support::{CliHarness, CommandOutput, CompletionBehavior, LoginState, Provider, cg_bin};
use testcontainers::core::{IntoContainerPort, Mount, WaitFor};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, GenericImage, ImageExt};
use tokio::sync::OwnedSemaphorePermit;

const AUTH_INITIAL: &str = "Acme needs password-less login to ship. We found the auth module file at src/auth.py and decided the auth module stays at src/auth.py.";
const AUTH_WITH_JWT: &str = "Acme needs password-less login to ship. We found the auth module file at src/auth.py and decided the auth module stays at src/auth.py. Use JWT with 15-min access tokens.";
const AUTH_RENAME: &str = "Following Tuesday, the auth module file moved from src/auth.py to src/authentication.py. Keep password-less login served by the auth module at src/authentication.py.";
const AUTH_CURRENT_WITH_JWT: &str = "Acme password-less login is now served by the auth module at src/authentication.py. Use JWT with 15-min access tokens.";
const CI_PIPELINE_SESSION: &str = "The delivery team needs a CI pipeline. Run CI on every pull request, block merges when CI fails, and include unit tests, clippy, and release build in .github/workflows/ci.yml.";
const CLIENT_A_SOX_SESSION: &str = "Client A must keep SOX deploy approvals. The publish pipeline is deploy.yml; require SOX approval before promote.";
const CLIENT_B_PUBLISH_SESSION: &str = "Client B must avoid the duplicate-publish race. The publish pipeline is deploy.yml; hold 10 min before promote.";

static CLI_NATS_TEST_LOCK: LazyLock<Arc<tokio::sync::Semaphore>> =
    LazyLock::new(|| Arc::new(tokio::sync::Semaphore::new(1)));

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

struct CliNatsFixture {
    _lock: OwnedSemaphorePermit,
    _container: ContainerAsync<GenericImage>,
    nats_url: String,
}

async fn start_cli_nats_or_environment_error() -> CliNatsFixture {
    let lock = CLI_NATS_TEST_LOCK
        .clone()
        .acquire_owned()
        .await
        .expect("NATS fixture lock should not close");
    let conf = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../context-graph/tests/common/nats.conf")
        .to_string_lossy()
        .into_owned();
    let image = GenericImage::new("nats", "latest")
        .with_exposed_port(4222.tcp())
        .with_wait_for(WaitFor::message_on_stderr("Server is ready"));
    let request = image
        .with_mount(Mount::bind_mount(conf, "/etc/nats/nats.conf"))
        .with_cmd(["--js", "--config", "/etc/nats/nats.conf"]);
    let container = request
        .start()
        .await
        .expect("environment ERROR: Docker/NATS substrate is required for this CLI sync test");
    let nats_port = container
        .get_host_port_ipv4(4222.tcp())
        .await
        .expect("NATS port should be mapped");
    CliNatsFixture {
        _lock: lock,
        _container: container,
        nats_url: format!("nats://127.0.0.1:{nats_port}"),
    }
}

struct CliSyncFixture {
    _nats: CliNatsFixture,
    _hub_dir: tempfile::TempDir,
    _hub_store: Store,
    hub_db: Arc<Database>,
    shutdown: Arc<AtomicBool>,
    server_task: tokio::task::JoinHandle<()>,
    nats_url: String,
    tenant_id: String,
}

impl Drop for CliSyncFixture {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.server_task.abort();
    }
}

async fn cli_sync_fixture(label: &str) -> CliSyncFixture {
    let nats = start_cli_nats_or_environment_error().await;
    let hub_dir = tempfile::tempdir().expect("hub tempdir");
    let tenant_id = format!("tenant-{}", label.replace('_', "-"));
    let hub_store = Store::open(StoreConfig {
        db_path: hub_dir.path().join("hub.db"),
        default_text_embedder: None,
        ..Default::default()
    })
    .expect("open CLI sync hub store");
    let hub_db = hub_store.sync_database();
    let server = Arc::new(SyncServer::new(
        Arc::clone(&hub_db),
        &nats.nats_url,
        &tenant_id,
        context_graph_sync_conflict_policies(),
    ));
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_task = {
        let server = Arc::clone(&server);
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move { server.run_until(shutdown).await })
    };
    tokio::time::sleep(Duration::from_millis(500)).await;
    let nats_url = nats.nats_url.clone();
    CliSyncFixture {
        _nats: nats,
        _hub_dir: hub_dir,
        _hub_store: hub_store,
        hub_db,
        shutdown,
        server_task,
        nats_url,
        tenant_id,
    }
}

fn cli_expected_pending_push_change_count(store: &Store, nats_url: &str, tenant_id: &str) -> usize {
    let client = SyncClient::new(store.sync_database(), nats_url, tenant_id);
    context_graph_pending_push_change_count(client.db(), client.push_watermark())
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

fn force_write_once_schema_conflict(path: &std::path::Path) -> contextdb_core::TxId {
    let store = persisted_store(Some(path.to_path_buf())).expect("open second CLI store");
    let db = store.sync_database();
    let original_installed_at = schema_version_installed_at(db.as_ref());
    let committed = db.committed_watermark();
    let conflict_installed_at = if original_installed_at != committed {
        committed
    } else {
        assert!(
            committed.0 > 0,
            "schema version fixture needs a prior valid txid to create a divergent write-once row"
        );
        contextdb_core::TxId(committed.0 - 1)
    };
    assert_ne!(
        original_installed_at, conflict_installed_at,
        "schema version fixture must create a divergent local write-once row"
    );
    db.execute(
        "UPDATE cg_schema_version SET installed_at = $installed_at WHERE id = $id",
        &HashMap::from([
            ("id".to_string(), DbValue::Int64(1)),
            (
                "installed_at".to_string(),
                DbValue::TxId(conflict_installed_at),
            ),
        ]),
    )
    .expect("create local write-once conflict fixture");
    assert_eq!(
        schema_version_installed_at(db.as_ref()),
        conflict_installed_at,
        "fixture must create a local write-once schema conflict"
    );
    conflict_installed_at
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
    let first = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let output = harness.run_with_stdin(&["record"], AUTH_RENAME);
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
    let first = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let second = harness.run_with_stdin(&["record"], AUTH_RENAME);
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
    harness.run_with_stdin(&["record"], AUTH_WITH_JWT);
    let first = harness.run(&["status"]);
    harness.run_with_stdin(&["record"], CI_PIPELINE_SESSION);
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
    let first_capture = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let first = harness.run(&["status"]);
    let second_capture = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let second = harness.run(&["status"]);
    let first_decision_id = stable_uuid_from(
        &[&first_capture, &first],
        &["D_old", "auth module at src/auth.py"],
    );
    let second_decision_id = stable_uuid_from(
        &[&second_capture, &second],
        &["D_old", "auth module at src/auth.py"],
    );

    assert_eq!(
        (
            memory_counts(&first),
            memory_counts(&second),
            first_decision_id.is_some(),
            second_decision_id.is_some(),
            first_decision_id == second_decision_id,
        ),
        (Some((1, 1, 1)), Some((1, 1, 1)), true, true, true)
    );
}

#[test]
fn cli_by_intention() {
    let harness = extraction_harness();
    let capture = harness.run_with_stdin(&["record"], AUTH_CURRENT_WITH_JWT);
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
    persisted_decisions.sort_by(|left, right| left.0.cmp(right.0));

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
    let first = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let second = harness.run_with_stdin(&["record"], AUTH_RENAME);
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
    assert!(no_resolution(&wrong));
}

#[test]
fn cli_by_entity_renders_non_auth_entity_history_from_stored_rows() {
    let harness = extraction_harness();
    let capture = harness.run_with_stdin(
        &["record"],
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
    let client_a = harness.run_with_stdin(&["record", "--context", "client-a"], AUTH_INITIAL);
    harness.run_with_stdin(
        &["record", "--context", "client-b"],
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
    harness.run_with_stdin(&["record"], AUTH_INITIAL);
    harness.run_with_stdin(&["record"], AUTH_RENAME);
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
        &["record"],
        "Rollout must stay available. The primary rollout basis is src/d1.py and fallback rollout basis is src/d2.py; D1 uses the primary basis and D2 uses the fallback basis.",
    );
    harness.run_with_stdin(
        &["record"],
        "The primary rollout basis moved from src/d1.py to src/d1_v2.py.",
    );
    let first = harness.run(&["impact"]);
    harness.run_with_stdin(
        &["record"],
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
    harness.run_with_stdin(&["record"], AUTH_INITIAL);
    harness.run_with_stdin(&["record"], AUTH_RENAME);
    let output = harness.run(&["impact"]);
    let text = combined(&output);

    assert!(text.contains("auth module at src/auth.py [INVALIDATED]"));
    assert!(text.contains("auth module at src/authentication.py"));
    assert!(!text.contains("resolved"));
}

#[test]
fn cli_changed_surfaces_cross_session_invalidation() {
    let harness = extraction_harness();
    harness.run_with_stdin(&["record"], AUTH_INITIAL);
    harness.run_with_stdin(&["record"], AUTH_RENAME);
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
        &["record"],
        "Track the before file at src/before.py so the before path stays visible.",
    );
    let before_invalidation = harness.run_with_stdin(
        &["record"],
        "The before file moved from src/before.py to src/before_v2.py.",
    );
    assert_record_contains(&before, "src/before.py");
    assert_record_text(&before_invalidation, "src/before_v2.py");
    let cut = harness.run_with_stdin(
        &["record"],
        "Track the cut file at src/cut.py so the cut path stays visible.",
    );
    let cut_invalidation = harness.run_with_stdin(
        &["record"],
        "The cut file moved from src/cut.py to src/cut_v2.py.",
    );
    assert_record_contains(&cut, "src/cut.py");
    assert_record_text(&cut_invalidation, "src/cut_v2.py");
    let cutoff_changed = harness.run(&["changed"]);
    let after = harness.run_with_stdin(
        &["record"],
        "Track the after file at src/after.py so the after path stays visible.",
    );
    let after_invalidation = harness.run_with_stdin(
        &["record"],
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
    let first = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let second = harness.run_with_stdin(&["record"], AUTH_RENAME);
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

    let record = run_in_dir_with_stdin(&harness, cwd.path(), &["record"], transcript);
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
        &["record", "--tenant-id", "progress-tenant"],
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
        &["record", "--tenant-id", "owner-tenant"],
        "Goal G1: converge by tenant. Entity E1: auth module path=src/auth.py. Decision D1: tenant memory lands together serves G1 based_on E1.",
    );
    let first_provider_records = completion_records_since(&harness, completions_before_first);
    let completions_before_second = harness.providers.completion_records().len();
    let second = run_in_dir_with_stdin(
        &harness,
        dir_y.path(),
        &["record", "--tenant-id", "owner-tenant"],
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
        &["record", "--tenant-id", "other-owner-tenant"],
        "Goal G3: isolate other tenant. Entity E3: billing file path=billing.toml. Decision D3: other tenant stays separate serves G3 based_on E3.",
    );
    let explicit = run_in_dir_with_stdin(
        &harness,
        dir_x.path(),
        &[
            "record",
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
        &["record", "--tenant-id", "owner-tenant"],
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
    let offline_harness = extraction_harness();
    let offline_record = offline_harness.run_with_stdin(
        &[
            "record",
            "--remote-server",
            "nats://127.0.0.1:1",
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
            "record",
            "--remote-server",
            fixture.nats_url.as_str(),
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
            fixture.nats_url.as_str(),
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
        fixture.nats_url.as_str(),
        "--tenant-id",
        fixture.tenant_id.as_str(),
    ]);
    assert_eq!(pending_status.code, 0, "{}", combined(&pending_status));
    let pending_json = output_json(&pending_status).expect("pending sync status JSON");
    assert_eq!(
        json_string_at(&pending_json, &["/sync/remote_url", "/remote_url"]),
        Some(fixture.nats_url.as_str())
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
        fixture.nats_url.as_str(),
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
            fixture.nats_url.as_str(),
            fixture.tenant_id.as_str(),
        ) as u64
    };
    let pushed_status = harness.run(&[
        "sync",
        "status",
        "--remote-server",
        fixture.nats_url.as_str(),
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

    let second_edge = CliHarness::new();
    let second_edge_store_path = second_edge.data_root.path().join("context-graph.db");
    let local_schema_installed_at = force_write_once_schema_conflict(&second_edge_store_path);
    let pull = second_edge.run(&[
        "sync",
        "pull",
        "--remote-server",
        fixture.nats_url.as_str(),
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
        "cg sync pull must report the real ApplyResult from the production per-table policy contract; a default ServerWins pull records a conflict for the write-once schema row: {pull_json}"
    );
    assert!(
        json_u64_at(
            &pull_json,
            &[
                "/sync/apply_result/skipped_rows",
                "/apply_result/skipped_rows"
            ]
        )
        .unwrap_or(0)
            > 0,
        "write-once schema conflict fixture must exercise the InsertIfNotExists skip path: {pull_json}"
    );
    {
        let post_pull_db =
            Database::open(&second_edge_store_path).expect("open second edge after pull");
        assert_eq!(
            schema_version_installed_at(&post_pull_db),
            local_schema_installed_at,
            "cg sync pull must preserve the local divergent write-once schema row while reporting zero conflicts"
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
    let client_a =
        harness.run_with_stdin(&["record", "--context", "client-a"], CLIENT_A_SOX_SESSION);
    let client_b = harness.run_with_stdin(
        &["record", "--context", "client-b"],
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
    let client_a =
        harness.run_with_stdin(&["record", "--context", "client-a"], CLIENT_A_SOX_SESSION);
    let client_b = harness.run_with_stdin(
        &["record", "--context", "client-b"],
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
    let first = harness.run_with_stdin(&["record"], "first auth capture");
    let second = harness.run_with_stdin(&["record"], "second auth capture");

    assert!(combined(&first).contains("loading bundled text embedder"));
    assert!(!combined(&second).contains("loading bundled text embedder"));
}
