#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tempfile::TempDir;

pub struct CommandOutput {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

pub struct ObservedCommandOutput {
    pub output: CommandOutput,
    pub observed_before_exit: bool,
    pub observed_while_provider_running: bool,
}

pub struct CliHarness {
    pub data_root: TempDir,
    pub providers: FakeProviderRig,
}

impl CliHarness {
    pub fn new() -> Self {
        Self {
            data_root: TempDir::new().expect("cg data root"),
            providers: FakeProviderRig::new(),
        }
    }

    pub fn run(&self, args: &[&str]) -> CommandOutput {
        run_cg(
            args,
            self.data_root.path(),
            &self.providers.path_env(),
            None,
        )
    }

    pub fn run_with_stdin(&self, args: &[&str], stdin: &str) -> CommandOutput {
        run_cg(
            args,
            self.data_root.path(),
            &self.providers.path_env(),
            Some(stdin),
        )
    }

    pub fn run_with_stdin_observing_provider_before_exit(
        &self,
        args: &[&str],
        stdin: &str,
        needle: &str,
        provider: Provider,
    ) -> ObservedCommandOutput {
        run_cg_observing_provider_before_exit_inner(
            args,
            self.data_root.path(),
            &self.providers.path_env(),
            Some(stdin),
            needle,
            &self.providers.completion_started_marker(provider),
            &self.providers.completion_exited_marker(provider),
        )
    }

    pub fn run_observing_provider_before_exit(
        &self,
        args: &[&str],
        needle: &str,
        provider: Provider,
    ) -> ObservedCommandOutput {
        run_cg_observing_provider_before_exit(
            args,
            self.data_root.path(),
            &self.providers.path_env(),
            needle,
            &self.providers.completion_started_marker(provider),
            &self.providers.completion_exited_marker(provider),
        )
    }

    pub fn run_writing_stdin_after_output(
        &self,
        args: &[&str],
        needle: &str,
        stdin_after: &str,
    ) -> ObservedCommandOutput {
        run_cg_writing_stdin_after_output(
            args,
            self.data_root.path(),
            &self.providers.path_env(),
            needle,
            stdin_after,
        )
    }
}

pub fn cg_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_cg"))
}

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

pub fn run_cg(
    args: &[&str],
    data_root: &Path,
    path_env: &str,
    stdin: Option<&str>,
) -> CommandOutput {
    let mut command = Command::new(cg_bin());
    command
        .args(args)
        .env("CG_DATA_ROOT", data_root)
        .env("CG_USE_TEST_LEXICAL_EMBEDDER", "1")
        .env("HOME", data_root)
        .env("PATH", path_env)
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().expect("spawn compiled cg binary");
    if let Some(stdin) = stdin
        && let Some(mut child_stdin) = child.stdin.take()
        && let Err(err) = child_stdin.write_all(stdin.as_bytes())
    {
        assert_eq!(err.kind(), ErrorKind::BrokenPipe, "write cg stdin: {err}");
    }
    let output = child.wait_with_output().expect("collect cg output");
    CommandOutput {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

pub fn run_cg_observing_provider_before_exit(
    args: &[&str],
    data_root: &Path,
    path_env: &str,
    needle: &str,
    provider_started_marker: &Path,
    provider_exited_marker: &Path,
) -> ObservedCommandOutput {
    run_cg_observing_provider_before_exit_inner(
        args,
        data_root,
        path_env,
        None,
        needle,
        provider_started_marker,
        provider_exited_marker,
    )
}

fn run_cg_observing_provider_before_exit_inner(
    args: &[&str],
    data_root: &Path,
    path_env: &str,
    stdin: Option<&str>,
    needle: &str,
    provider_started_marker: &Path,
    provider_exited_marker: &Path,
) -> ObservedCommandOutput {
    for marker in [provider_started_marker, provider_exited_marker] {
        match fs::remove_file(marker) {
            Ok(()) => {}
            Err(err) if err.kind() == ErrorKind::NotFound => {}
            Err(err) => panic!("remove stale provider marker {}: {err}", marker.display()),
        }
    }
    let mut command = Command::new(cg_bin());
    command
        .args(args)
        .env("CG_DATA_ROOT", data_root)
        .env("CG_USE_TEST_LEXICAL_EMBEDDER", "1")
        .env("HOME", data_root)
        .env("PATH", path_env)
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().expect("spawn compiled cg binary");
    if let Some(stdin) = stdin
        && let Some(mut child_stdin) = child.stdin.take()
        && let Err(err) = child_stdin.write_all(stdin.as_bytes())
    {
        assert_eq!(err.kind(), ErrorKind::BrokenPipe, "write cg stdin: {err}");
    }
    let stdout = child.stdout.take().expect("take cg stdout");
    let stderr = child.stderr.take().expect("take cg stderr");
    let (tx, rx) = mpsc::channel();
    let stdout_handle = spawn_stream_reader(stdout, tx.clone());
    let stderr_handle = spawn_stream_reader(stderr, tx);
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut observed_before_exit = false;
    let mut observed_while_provider_running = false;
    let status = loop {
        while let Ok(chunk) = rx.try_recv() {
            if chunk.contains(needle) {
                observed_before_exit = true;
                if provider_started_marker.exists() && !provider_exited_marker.exists() {
                    observed_while_provider_running = true;
                }
            }
        }
        if let Some(status) = child.try_wait().expect("poll cg child") {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().expect("kill timed-out cg child");
            break child.wait().expect("collect killed cg child");
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_handle.join().expect("join stdout reader");
    let stderr = stderr_handle.join().expect("join stderr reader");
    ObservedCommandOutput {
        output: CommandOutput {
            code: status.code().unwrap_or(-1),
            stdout,
            stderr,
        },
        observed_before_exit,
        observed_while_provider_running,
    }
}

pub fn run_cg_writing_stdin_after_output(
    args: &[&str],
    data_root: &Path,
    path_env: &str,
    needle: &str,
    stdin_after: &str,
) -> ObservedCommandOutput {
    let mut command = Command::new(cg_bin());
    command
        .args(args)
        .env("CG_DATA_ROOT", data_root)
        .env("CG_USE_TEST_LEXICAL_EMBEDDER", "1")
        .env("HOME", data_root)
        .env("PATH", path_env)
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().expect("spawn compiled cg binary");
    let mut stdin = Some(child.stdin.take().expect("take cg stdin"));
    let stdout = child.stdout.take().expect("take cg stdout");
    let stderr = child.stderr.take().expect("take cg stderr");
    let (tx, rx) = mpsc::channel();
    let stdout_handle = spawn_stream_reader(stdout, tx.clone());
    let stderr_handle = spawn_stream_reader(stderr, tx);
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut observed_before_exit = false;
    let mut held_stdin = None;
    let status = loop {
        while let Ok(chunk) = rx.try_recv() {
            if chunk.contains(needle) {
                observed_before_exit = true;
            }
        }
        if observed_before_exit && let Some(mut child_stdin) = stdin.take() {
            child_stdin
                .write_all(stdin_after.as_bytes())
                .expect("write delayed cg stdin");
            held_stdin = Some(child_stdin);
        }
        if let Some(status) = child.try_wait().expect("poll cg child") {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().expect("kill timed-out cg child");
            break child.wait().expect("collect killed cg child");
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_handle.join().expect("join stdout reader");
    let stderr = stderr_handle.join().expect("join stderr reader");
    drop(held_stdin);
    ObservedCommandOutput {
        output: CommandOutput {
            code: status.code().unwrap_or(-1),
            stdout,
            stderr,
        },
        observed_before_exit,
        observed_while_provider_running: false,
    }
}

fn spawn_stream_reader<R: Read + Send + 'static>(
    stream: R,
    tx: mpsc::Sender<String>,
) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        let mut output = String::new();
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    output.push_str(&line);
                    let _ = tx.send(line);
                }
                Err(_) => break,
            }
        }
        output
    })
}

pub fn parse_single_json(stdout: &str) -> Result<Value, String> {
    serde_json::from_str::<Value>(stdout).map_err(|err| err.to_string())
}

pub fn parse_single_json_object(stdout: &str) -> Result<serde_json::Map<String, Value>, String> {
    match parse_single_json(stdout)? {
        Value::Object(map) => Ok(map),
        other => Err(format!("expected JSON object, got {other:?}")),
    }
}

pub fn unique_label(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    format!("{prefix}-{nanos}-{}@example.test", std::process::id())
}

#[derive(Clone, Copy)]
pub enum Provider {
    Claude,
    Codex,
    Agy,
}

impl Provider {
    pub fn binary_name(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Agy => "agy",
        }
    }
}

#[derive(Clone, Copy)]
pub enum LoginState {
    Valid,
    Expired,
}

impl LoginState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::Expired => "expired",
        }
    }
}

#[derive(Clone, Copy)]
pub enum CompletionBehavior {
    Succeed,
    ExtractHarvest,
    SleepThenExtractHarvest,
    Fail,
    BillingRequired,
    RateLimit,
    SleepThenSucceed,
    Hang,
    Empty,
}

impl CompletionBehavior {
    fn as_str(self) -> &'static str {
        match self {
            Self::Succeed => "succeed",
            Self::ExtractHarvest => "extract-harvest",
            Self::SleepThenExtractHarvest => "sleep-then-extract-harvest",
            Self::Fail => "fail",
            Self::BillingRequired => "billing-required",
            Self::RateLimit => "rate-limit",
            Self::SleepThenSucceed => "sleep-then-succeed",
            Self::Hang => "hang",
            Self::Empty => "empty",
        }
    }
}

pub struct FakeProviderRig {
    _root: TempDir,
    pub root_path: PathBuf,
    pub bin_dir: PathBuf,
    pub fixture_dir: PathBuf,
    pub log_path: PathBuf,
}

impl FakeProviderRig {
    pub fn new() -> Self {
        let root = TempDir::new().expect("fake provider root");
        let root_path = root.path().to_path_buf();
        let bin_dir = root_path.join("bin");
        let fixture_dir = root_path.join("fixtures");
        fs::create_dir_all(&bin_dir).expect("fake bin dir");
        fs::create_dir_all(&fixture_dir).expect("fake fixture dir");
        let rig = Self {
            _root: root,
            root_path,
            bin_dir,
            fixture_dir,
            log_path: PathBuf::new(),
        };
        let log_path = rig.fixture_dir.join("invocations.jsonl");
        let rig = Self { log_path, ..rig };
        rig.write_driver();
        rig
    }

    pub fn install(
        &self,
        provider: Provider,
        account_label: &str,
        login_state: LoginState,
        behavior: CompletionBehavior,
    ) {
        let provider_dir = self.fixture_dir.join(provider.binary_name());
        fs::create_dir_all(&provider_dir).expect("provider fixture dir");
        write_file(provider_dir.join("account_label"), account_label);
        write_file(provider_dir.join("login_state"), login_state.as_str());
        write_file(provider_dir.join("completion_behavior"), behavior.as_str());
        write_file(provider_dir.join("auth_probe_drivable"), "true");
        write_file(
            provider_dir.join("expired_marker"),
            &format!("expired-marker-{account_label}"),
        );
        write_file(
            provider_dir.join("failure_marker"),
            &format!("failure-marker-{account_label}"),
        );
        write_file(provider_dir.join("secret"), &unique_label("secret"));
        match provider {
            Provider::Claude => {
                write_file(
                    provider_dir.join("help_flags"),
                    "--output-format\n--mcp-config\n--strict-mcp-config\n--tools\n",
                );
            }
            Provider::Codex => {
                write_file(provider_dir.join("help_flags"), "exec\nlogin\nlogout\n");
                write_file(
                    provider_dir.join("exec_help_flags"),
                    "--output-last-message\n-C\n--skip-git-repo-check\n--ignore-user-config\n--ephemeral\n",
                );
            }
            Provider::Agy => {
                write_file(provider_dir.join("help_flags"), "--print\n");
            }
        }
        write_file(provider_dir.join("version"), "fixture-provider 2026.6.4");
        self.write_wrapper(provider);
    }

    pub fn remove_provider(&self, provider: Provider) {
        let path = self.bin_dir.join(provider.binary_name());
        if path.exists() {
            fs::remove_file(path).expect("remove fake provider");
        }
    }

    pub fn set_behavior(&self, provider: Provider, behavior: CompletionBehavior) {
        write_file(
            self.fixture_dir
                .join(provider.binary_name())
                .join("completion_behavior"),
            behavior.as_str(),
        );
    }

    pub fn set_auth_probe_drivable(&self, provider: Provider, drivable: bool) {
        write_file(
            self.fixture_dir
                .join(provider.binary_name())
                .join("auth_probe_drivable"),
            if drivable { "true" } else { "false" },
        );
    }

    pub fn set_auth_probe_hangs(&self, provider: Provider) {
        write_file(
            self.fixture_dir
                .join(provider.binary_name())
                .join("auth_probe_behavior"),
            "hang",
        );
    }

    pub fn set_auth_probe_unsupported(&self, provider: Provider) {
        write_file(
            self.fixture_dir
                .join(provider.binary_name())
                .join("auth_probe_behavior"),
            "unsupported",
        );
    }

    pub fn set_help_probe_hangs(&self, provider: Provider) {
        write_file(
            self.fixture_dir
                .join(provider.binary_name())
                .join("help_probe_behavior"),
            "hang",
        );
    }

    pub fn set_version_probe_fails(&self, provider: Provider) {
        write_file(
            self.fixture_dir
                .join(provider.binary_name())
                .join("version_probe_behavior"),
            "fail",
        );
    }

    pub fn set_auth_probe_stderr(&self, provider: Provider, warning: &str) {
        write_file(
            self.fixture_dir
                .join(provider.binary_name())
                .join("auth_stderr"),
            warning,
        );
    }

    pub fn set_login_state(&self, provider: Provider, login_state: LoginState) {
        write_file(
            self.fixture_dir
                .join(provider.binary_name())
                .join("login_state"),
            login_state.as_str(),
        );
    }

    pub fn write_secret(&self, provider: Provider, secret: &str) {
        write_file(
            self.fixture_dir.join(provider.binary_name()).join("secret"),
            secret,
        );
    }

    pub fn account_label(&self, provider: Provider) -> Option<String> {
        fs::read_to_string(
            self.fixture_dir
                .join(provider.binary_name())
                .join("account_label"),
        )
        .ok()
    }

    pub fn path_env(&self) -> String {
        format!("{}:/usr/bin:/bin", self.bin_dir.display())
    }

    pub fn completion_records(&self) -> Vec<Value> {
        fs::read_to_string(&self.log_path)
            .ok()
            .map(|text| {
                text.lines()
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .filter(|record| {
                        record.get("mode").and_then(Value::as_str) == Some("completion")
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn completion_count(&self) -> usize {
        self.completion_records().len()
    }

    pub fn probe_records(&self) -> Vec<Value> {
        fs::read_to_string(&self.log_path)
            .ok()
            .map(|text| {
                text.lines()
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .filter(|record| {
                        matches!(
                            record.get("mode").and_then(Value::as_str),
                            Some("probe") | Some("auth-status-probe")
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn probe_count(&self) -> usize {
        self.probe_records().len()
    }

    pub fn auth_status_probe_count(&self) -> usize {
        self.probe_records()
            .into_iter()
            .filter(|record| {
                record.get("mode").and_then(Value::as_str) == Some("auth-status-probe")
            })
            .count()
    }

    pub fn auth_status_probe_count_for(&self, provider: Provider) -> usize {
        self.probe_records()
            .into_iter()
            .filter(|record| {
                record.get("mode").and_then(Value::as_str) == Some("auth-status-probe")
                    && record.get("provider").and_then(Value::as_str)
                        == Some(provider.binary_name())
            })
            .count()
    }

    pub fn last_completion_provider(&self) -> Option<String> {
        self.completion_records()
            .last()
            .and_then(|record| record.get("provider"))
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    pub fn last_completion_prompt(&self) -> Option<String> {
        self.completion_records()
            .last()
            .and_then(|record| record.get("prompt"))
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    pub fn expected_reply(&self, provider: Provider) -> Option<String> {
        let secret =
            fs::read_to_string(self.fixture_dir.join(provider.binary_name()).join("secret"))
                .ok()?;
        let prompt = self.last_completion_prompt()?;
        Some(secret_reply(secret.trim(), &prompt))
    }

    pub fn completion_started_marker(&self, provider: Provider) -> PathBuf {
        self.fixture_dir
            .join(provider.binary_name())
            .join("completion_started")
    }

    pub fn completion_exited_marker(&self, provider: Provider) -> PathBuf {
        self.fixture_dir
            .join(provider.binary_name())
            .join("completion_exited")
    }

    fn write_wrapper(&self, provider: Provider) {
        let script = format!(
            "#!/bin/sh\npython3 {} {} {} \"$@\"\n",
            shell_quote(self.fixture_dir.join("fake_provider.py")),
            shell_quote(provider.binary_name()),
            shell_quote(&self.fixture_dir),
        );
        let path = self.bin_dir.join(provider.binary_name());
        write_file(&path, &script);
        make_executable(&path);
    }

    fn write_driver(&self) {
        let driver = r#"
import json
import os
import pathlib
import sys
import time

provider = sys.argv[1]
root = pathlib.Path(sys.argv[2])
fixture = root / provider
args = sys.argv[3:]
log_path = root / "invocations.jsonl"

def read(name, default=""):
    path = fixture / name
    if path.exists():
        return path.read_text().strip()
    return default

TRACKED_ENV = [
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
]

def completion_usage(reply):
    output_tokens = len(reply.split())
    return {
        "input_tokens": 13,
        "output_tokens": output_tokens,
        "total_tokens": 13 + output_tokens,
    }

def expected_completion_usage(prompt):
    behavior = read("completion_behavior", "succeed")
    if behavior in ("extract-harvest", "sleep-then-extract-harvest"):
        return completion_usage(skill_proposal_json(prompt) or harvest_json(prompt))
    if behavior in ("succeed", "sleep-then-succeed"):
        return completion_usage(proof(read("secret", "secret"), prompt))
    return None

def write_log(mode, prompt=""):
    record = {
        "provider": provider,
        "mode": mode,
        "argv": args,
        "prompt": prompt,
        "env": {name: os.environ[name] for name in TRACKED_ENV if name in os.environ},
    }
    if mode == "completion":
        usage = expected_completion_usage(prompt)
        if usage is not None:
            record["usage"] = usage
    with log_path.open("a") as out:
        out.write(json.dumps(record, sort_keys=True) + "\n")

def mark(name):
    (fixture / name).write_text(str(time.time()))

def proof(secret, prompt):
    value = 14695981039346656037
    for byte in (secret + "\0" + prompt).encode():
        value ^= byte
        value = (value * 1099511628211) & 0xffffffffffffffff
    return "proof-%016x" % value

def capture_transcript(prompt):
    marker = "Transcript:\n"
    if marker in prompt:
        transcript = prompt.split(marker, 1)[1]
        start = "<cg_transcript>\n"
        end = "\n</cg_transcript>"
        if start in transcript and end in transcript:
            transcript = transcript.split(start, 1)[1].split(end, 1)[0]
        return capture_next_session_chunk(transcript)
    return capture_next_session_chunk(prompt)

def capture_next_session_chunk(text):
    marker = "Next session chunk:\n"
    if marker in text:
        return text.rsplit(marker, 1)[1]
    return text

def clean_statement(statement):
    statement = statement.strip().strip(".")
    for role in ("User:", "Assistant:", "System:"):
        if statement.lower().startswith(role.lower()):
            return statement[len(role):].strip()
    return statement

def parse_label(statement, label, ordinal):
    statement = clean_statement(statement)
    lower = statement.lower()
    label_lower = label.lower()
    if lower.startswith(label_lower + ":") or lower.startswith(label_lower + " "):
        rest = statement[len(label) + 1:].strip()
    else:
        return None
    if ":" in rest:
        maybe_key, body = rest.split(":", 1)
        key = maybe_key.split()[0] if maybe_key.split() else ""
        if key.startswith(label[0]):
            return key, body.strip()
    parts = rest.split(None, 1)
    if parts and parts[0].startswith(label[0]):
        return parts[0], parts[1].strip() if len(parts) > 1 else rest
    return "%s%d" % (label[0], ordinal), rest

def path_from_text(text):
    if "path=" in text:
        return text.split("path=", 1)[1].split()[0].rstrip(",")
    for token in text.split():
        if "/" in token or ".toml" in token or ".yml" in token:
            return token.rstrip(",")
    return None

def entity_type(path):
    if ".toml" in path:
        return "configuration"
    if "deploy" in path or "workflow" in path:
        return "pipeline"
    if "/" in path or path.endswith(".py") or path.endswith(".rs"):
        return "file"
    return "service"

def explicit_entity_type(text):
    if "type=" not in text:
        return None
    return text.split("type=", 1)[1].split()[0].rstrip(",").lower()

def entity_seed(key, text):
    path = path_from_text(text)
    if not path:
        if "auth module" in text:
            path = "src/auth.py"
        elif "redis" in text.lower():
            path = "config/redis.toml"
        else:
            path = text
    if key == "E_auth_module":
        name = "auth module file"
    elif key == "E_redis":
        name = "redis config"
    else:
        name = text.split(" path=", 1)[0].strip().strip(",")
    return {"key": key, "name": name.split(" type=", 1)[0].strip(), "path": path, "entity_type": explicit_entity_type(text) or entity_type(path)}

def decision_seed(key, text, entities):
    decision_text = text.split(" serves ", 1)[0].strip()
    known = {
        "auth module at src/auth.py": "D_old",
        "auth module at src/authentication.py": "D_new",
        "use JWT with 15-min access tokens": "D_jwt",
        "store refresh tokens in Redis": "D_other",
    }
    key = known.get(decision_text, key)
    goal_key = None
    if " serves " in text:
        goal_key = text.split(" serves ", 1)[1].split()[0].rstrip(",")
    entity_key = None
    if "based_on " in text:
        entity_key = text.split("based_on ", 1)[1].split()[0].rstrip(",")
    lower = text.lower()
    if not entity_key:
        for entity in entities:
            if entity["path"].lower() in lower:
                entity_key = entity["key"]
                break
    if not entity_key and ("src/auth.py" in lower or "src/authentication.py" in lower):
        entity_key = "E_auth_module"
    if not entity_key and "redis" in lower:
        entity_key = "E_redis"
    return {"key": key, "text": decision_text, "goal_key": goal_key, "entity_key": entity_key}

def observation_seed(key, text, entities):
    lower = text.lower()
    first = text.split()[0].rstrip(":") if text.split() else ""
    entity_key = first if first.startswith("E") else None
    if not entity_key:
        for entity in entities:
            if entity["path"].lower() in lower:
                entity_key = entity["key"]
                break
    if not entity_key and "auth module" in lower:
        entity_key = "E_auth_module"
    if not entity_key and "redis" in lower:
        entity_key = "E_redis"
    path_change = None
    if "->" in text:
        path_change = text.split("->", 1)[1].split()[0].rstrip(",")
    known = {
        "auth module file observed at src/auth.py": "O_auth_path",
        "auth module file observed at src/authentication.py": "O_auth_current",
    }
    return {"key": known.get(text, key), "text": text, "entity_key": entity_key, "path_change": path_change}

def infer_entities(harvest):
    keys = {entity["key"] for entity in harvest["entities"]}
    all_text = " ".join([d["text"] for d in harvest["decisions"]] + [o["text"] for o in harvest["observations"]])
    if "E_auth_module" not in keys and ("src/auth.py" in all_text or "src/authentication.py" in all_text):
        harvest["entities"].append({"key":"E_auth_module","name":"auth module file","path":"src/auth.py","entity_type":"file"})
    if "E_redis" not in keys and "redis" in all_text.lower():
        harvest["entities"].append({"key":"E_redis","name":"redis config","path":"config/redis.toml","entity_type":"configuration"})

def compact(text):
    return " ".join(text.split())

def skill_proposal_json(prompt):
    lower = prompt.lower()
    if "skill proposal task" not in lower:
        return None
    if "entity name: context-graph" in lower and "source type: file" in lower and "target type: repository" in lower:
        return json.dumps({
            "description": "retype context-graph from File to Repository when future captures mention this entity",
            "rule": "when an entity named context-graph is typed File -> Repository",
        }, sort_keys=True)
    return json.dumps({
        "description": "retype the corrected entity when future captures mention it",
        "rule": "when an entity named context-graph is typed File -> Repository",
    }, sort_keys=True)

def harvest_json(prompt):
    transcript = capture_transcript(prompt)
    lower = transcript.lower()
    if "password-less login" in lower and "auth module file at src/auth.py" in lower and "src/authentication.py" not in lower:
        decisions = [{"key":"D_old","text":"auth module at src/auth.py","goal_key":"G1","entity_key":"E_auth_module"}]
        if "jwt" in lower and "15-min" in lower:
            decisions.append({"key":"D_jwt","text":"use JWT with 15-min access tokens","goal_key":"G1","entity_key":"E_auth_module"})
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G1","text":"ship password-less login for acme"}],
            "entities": [{"key":"E_auth_module","name":"auth module file","path":"src/auth.py","entity_type":"file"}],
            "decisions": decisions,
            "observations": [{"key":"O_auth_path","text":"auth module file observed at src/auth.py","entity_key":"E_auth_module","path_change":None}],
        }, sort_keys=True)
    if "password-less login" in lower and "auth module at src/authentication.py" in lower and "jwt" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G1","text":"ship password-less login for acme"}],
            "entities": [{"key":"E_auth_module","name":"auth module file","path":"src/authentication.py","entity_type":"file"}],
            "decisions": [
                {"key":"D_new","text":"auth module at src/authentication.py","goal_key":"G1","entity_key":"E_auth_module"},
                {"key":"D_jwt","text":"use JWT with 15-min access tokens","goal_key":"G1","entity_key":"E_auth_module"},
            ],
            "observations": [],
        }, sort_keys=True)
    if "moved from src/auth.py to src/authentication.py" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G1","text":"ship password-less login for acme"}],
            "entities": [{"key":"E_auth_module","name":"auth module file","path":"src/auth.py","entity_type":"file"}],
            "observations": [{"key":"O_auth_path_change","text":"E_auth_module path changed src/auth.py -> src/authentication.py","entity_key":"E_auth_module","path_change":"src/authentication.py"}],
            "decisions": [{"key":"D_new","text":"auth module at src/authentication.py","goal_key":"G1","entity_key":"E_auth_module"}],
        }, sort_keys=True)
    if "password-less login" in lower and "15-min access tokens" in lower and "auth migration notes" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [
                {"key":"G1","text":"ship password-less login for acme"},
                {"key":"G_auth_audit","text":"keep audit trail for auth migrations"},
            ],
            "entities": [{"key":"E_auth_module","name":"auth module file","path":"src/auth.py","entity_type":"file"}],
            "decisions": [
                {"key":"D_old","text":"auth module at src/auth.py","goal_key":"G1","entity_key":"E_auth_module"},
                {"key":"D_jwt","text":"use JWT with 15-min access tokens","goal_key":"G1","entity_key":"E_auth_module"},
                {"key":"D_auth_docs","text":"write auth migration notes to docs/auth-migration.md","goal_key":"G_auth_audit","entity_key":None},
            ],
            "observations": [
                {"key":"O_auth_path","text":"auth module file observed at src/auth.py","entity_key":"E_auth_module","path_change":None},
                {"key":"O_auth_tests","text":"auth tests cover password-less login","entity_key":"E_auth_module","path_change":None},
                {"key":"O_jwt_refresh","text":"JWT refresh token TTL remains 7 days","entity_key":"E_auth_module","path_change":None},
                {"key":"O_auth_audit","text":"audit trail lives in docs/auth-migration.md","entity_key":"E_auth_module","path_change":None},
            ],
        }, sort_keys=True)
    if "reduce checkout latency" in lower and "product catalog" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G_checkout_latency","text":"reduce checkout latency"}],
            "entities": [{"key":"E_checkout","name":"checkout service","path":"checkout service","entity_type":"service"}],
            "decisions": [{"key":"D_checkout_cache","text":"cache product catalog in memory","goal_key":"G_checkout_latency","entity_key":"E_checkout"}],
            "observations": [{"key":"O_checkout_latency","text":"checkout latency p95 dropped to 220ms","entity_key":"E_checkout","path_change":None}],
        }, sort_keys=True)
    if "migrating the auth handler" in lower and "src/authentication.py" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G_auth_handler","text":"migrate auth handler"}],
            "entities": [{"key":"E_auth_module","name":"auth module file","path":"src/authentication.py","entity_type":"file"}],
            "decisions": [{"key":"D_auth_handler","text":"replace auth handler with src/authentication.py","goal_key":"G_auth_handler","entity_key":"E_auth_module"}],
            "observations": [],
        }, sort_keys=True)
    if "ci pipeline" in lower and "pull request" in lower and "release build" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G_ci","text":"set up CI pipeline"}],
            "entities": [{"key":"E_ci","name":"CI pipeline","path":".github/workflows/ci.yml","entity_type":"pipeline"}],
            "decisions": [
                {"key":"D_ci_1","text":"run CI on every pull request","goal_key":"G_ci","entity_key":"E_ci"},
                {"key":"D_ci_2","text":"block merges when CI fails","goal_key":"G_ci","entity_key":"E_ci"},
            ],
            "observations": [
                {"key":"O_ci_1","text":"CI workflow file exists","entity_key":"E_ci","path_change":None},
                {"key":"O_ci_2","text":"unit tests run in CI","entity_key":"E_ci","path_change":None},
                {"key":"O_ci_3","text":"clippy runs in CI","entity_key":"E_ci","path_change":None},
                {"key":"O_ci_4","text":"release build runs in CI","entity_key":"E_ci","path_change":None},
            ],
        }, sort_keys=True)
    if "project x" in lower and "local context store" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G_project","text":"keep project x memory local"}],
            "entities": [],
            "decisions": [{"key":"D_project","text":"use local context store","goal_key":"G_project","entity_key":None}],
            "observations": [],
        }, sort_keys=True)
    if "context-graph" in lower and "type=file" in lower:
        goal_key = "G_context_graph"
        entity_key = "E_context_graph"
        decision_key = "D_context_graph"
        for token in transcript.replace(":", " ").split():
            if token.startswith("G_") or token.startswith("G"):
                goal_key = token.strip(",.")
                break
        for token in transcript.replace(":", " ").split():
            if token.startswith("E_") or token.startswith("E"):
                entity_key = token.strip(",.")
                break
        for token in transcript.replace(":", " ").split():
            if token.startswith("D_") or token.startswith("D"):
                decision_key = token.strip(",.")
                break
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key": goal_key, "text": "keep the context-graph repository classified"}],
            "entities": [{"key": entity_key, "name": "context-graph", "path": "context-graph", "entity_type": "file"}],
            "decisions": [{"key": decision_key, "text": "treat context-graph as the repository root", "goal_key": goal_key, "entity_key": entity_key}],
            "observations": [],
        }, sort_keys=True)
    if "webhook ingestion" in lower and "signed worker" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G_webhook","text":"keep webhook ingestion reliable"}],
            "entities": [{"key":"E_worker","name":"webhook worker file","path":"src/webhook.rs","entity_type":"file"}],
            "decisions": [{"key":"D_webhook","text":"route webhooks through signed worker","goal_key":"G_webhook","entity_key":"E_worker"}],
            "observations": [{"key":"O_worker","text":"E_worker path observed at src/webhook.rs","entity_key":"E_worker","path_change":None}],
        }, sort_keys=True)
    if "client b memory" in lower and "isolated deploy notes" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G_client_b","text":"keep client b memory separate"}],
            "entities": [],
            "decisions": [{"key":"D_client_b","text":"use isolated deploy notes","goal_key":"G_client_b","entity_key":None}],
            "observations": [],
        }, sort_keys=True)
    if "rollout must stay available" in lower and "src/d1.py" in lower and "src/d2.py" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G_rollout","text":"keep rollout available"}],
            "entities": [
                {"key":"E_d1","name":"primary rollout basis","path":"src/d1.py","entity_type":"file"},
                {"key":"E_d2","name":"fallback rollout basis","path":"src/d2.py","entity_type":"file"},
            ],
            "decisions": [
                {"key":"D1","text":"D1","goal_key":"G_rollout","entity_key":"E_d1"},
                {"key":"D2","text":"D2","goal_key":"G_rollout","entity_key":"E_d2"},
            ],
            "observations": [],
        }, sort_keys=True)
    if "primary rollout basis moved from src/d1.py to src/d1_v2.py" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [],
            "entities": [{"key":"E_d1","name":"primary rollout basis","path":"src/d1.py","entity_type":"file"}],
            "decisions": [],
            "observations": [{"key":"O_d1","text":"E_d1 path changed src/d1.py -> src/d1_v2.py","entity_key":"E_d1","path_change":"src/d1_v2.py"}],
        }, sort_keys=True)
    if "fallback rollout basis moved from src/d2.py to src/d2_v2.py" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [],
            "entities": [{"key":"E_d2","name":"fallback rollout basis","path":"src/d2.py","entity_type":"file"}],
            "decisions": [],
            "observations": [{"key":"O_d2","text":"E_d2 path changed src/d2.py -> src/d2_v2.py","entity_key":"E_d2","path_change":"src/d2_v2.py"}],
        }, sort_keys=True)
    for label, before, after in (
        ("before", "src/before.py", "src/before_v2.py"),
        ("cut", "src/cut.py", "src/cut_v2.py"),
        ("after", "src/after.py", "src/after_v2.py"),
    ):
        if ("track the %s file" % label) in lower and before in lower:
            return json.dumps({
                "summary": compact(transcript),
                "goals": [{"key":"G_%s" % label,"text":"keep %s path tracked" % label}],
                "entities": [{"key":"E_%s" % label,"name":"%s file" % label,"path":before,"entity_type":"file"}],
                "decisions": [{"key":"D_%s" % label,"text":before,"goal_key":"G_%s" % label,"entity_key":"E_%s" % label}],
                "observations": [{"key":"O_%s_initial" % label,"text":"E_%s path observed at %s" % (label, before),"entity_key":"E_%s" % label,"path_change":None}],
            }, sort_keys=True)
        if ("%s file moved from %s to %s" % (label, before, after)) in lower:
            return json.dumps({
                "summary": compact(transcript),
                "goals": [{"key":"G_%s" % label,"text":"keep %s path tracked" % label}],
                "entities": [{"key":"E_%s" % label,"name":"%s file" % label,"path":before,"entity_type":"file"}],
                "decisions": [{"key":"D_%s_new" % label,"text":after,"goal_key":"G_%s" % label,"entity_key":"E_%s" % label}],
                "observations": [{"key":"O_%s" % label,"text":"E_%s path changed %s -> %s" % (label, before, after),"entity_key":"E_%s" % label,"path_change":after}],
            }, sort_keys=True)
    if "avoid the duplicate-publish race" in lower and "hold 10 min before promote" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G_publish","text":"avoid the duplicate-publish race"}],
            "entities": [{"key":"E_publish","name":"publish pipeline","path":"deploy.yml","entity_type":"pipeline"}],
            "decisions": [{"key":"D_publish","text":"hold 10 min before promote","goal_key":"G_publish","entity_key":"E_publish"}],
            "observations": [],
        }, sort_keys=True)
    if "sox deploy approvals" in lower and "approval before promote" in lower:
        return json.dumps({
            "summary": compact(transcript),
            "goals": [{"key":"G_sox","text":"keep SOX deploy approvals"}],
            "entities": [{"key":"E_publish","name":"publish pipeline","path":"deploy.yml","entity_type":"pipeline"}],
            "decisions": [{"key":"D_sox","text":"SOX approval required before promote","goal_key":"G_sox","entity_key":"E_publish"}],
            "observations": [],
        }, sort_keys=True)
    harvest = {"summary": " ".join(transcript.split()), "goals": [], "entities": [], "decisions": [], "observations": []}
    statements = []
    for line in transcript.splitlines():
        statements.extend(line.split(". "))
    for statement in statements:
        statement = statement.strip().strip(".")
        if not statement:
            continue
        parsed = parse_label(statement, "Goal", len(harvest["goals"]) + 1)
        if parsed:
            key, text = parsed
            harvest["goals"].append({"key": key, "text": text})
            continue
        parsed = parse_label(statement, "Entity", len(harvest["entities"]) + 1)
        if parsed:
            key, text = parsed
            harvest["entities"].append(entity_seed(key, text))
            continue
        parsed = parse_label(statement, "Decision", len(harvest["decisions"]) + 1)
        if parsed:
            key, text = parsed
            harvest["decisions"].append(decision_seed(key, text, harvest["entities"]))
            continue
        parsed = parse_label(statement, "Observation", len(harvest["observations"]) + 1)
        if parsed:
            key, text = parsed
            harvest["observations"].append(observation_seed(key, text, harvest["entities"]))
    infer_entities(harvest)
    return json.dumps(harvest, sort_keys=True)

def claude_envelope(reply, is_error=False, status=None):
    usage = completion_usage(reply)
    return {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "api_error_status": status,
        "result": reply,
        "usage": usage,
        "modelUsage": {"claude-sonnet-4-20250514": {"inputTokens": usage["input_tokens"], "outputTokens": usage["output_tokens"]}},
    }

if "--help" in args:
    write_log("probe")
    if read("help_probe_behavior", "normal") == "hang":
        time.sleep(120)
    if provider == "codex" and args and args[0] == "exec":
        print(read("exec_help_flags", "--output-last-message\n-C"))
    elif provider == "agy":
        print(read("help_flags", "--print\n"), file=sys.stderr)
    else:
        print(read("help_flags", "--output-format\n--mcp-config\n--strict-mcp-config\n--tools"))
    sys.exit(0)

if "--version" in args:
    write_log("probe")
    if read("version_probe_behavior", "normal") == "fail":
        print("version probe failed", file=sys.stderr)
        sys.exit(2)
    if provider == "agy":
        print(read("version", "fixture-provider 0"), file=sys.stderr)
    else:
        print(read("version", "fixture-provider 0"))
    sys.exit(0)

if "auth" in args and "status" in args:
    write_log("auth-status-probe")
    probe_behavior = read("auth_probe_behavior", "normal")
    if probe_behavior == "hang":
        time.sleep(120)
    if probe_behavior == "unsupported":
        print("unknown command: auth status", file=sys.stderr)
        sys.exit(2)
    if read("auth_probe_drivable", "true") != "true":
        print(json.dumps({"ok": False, "error": "non_interactive_probe_failed"}))
        sys.exit(1)
    if read("login_state", "valid") == "expired":
        print(json.dumps({"ok": False, "error": read("expired_marker", "expired")}))
        sys.exit(1)
    print(json.dumps({"ok": True, "account": read("account_label", "dev@example.test"), "apiProvider": "firstParty"}))
    warning = read("auth_stderr")
    if warning:
        print(warning, file=sys.stderr)
    sys.exit(0)

prompt = sys.stdin.read()
if provider == "agy":
    if "--print" not in args:
        write_log("completion", prompt)
        mark("completion_started")
        mark("completion_exited")
        print("flag needs an argument: -print", file=sys.stderr)
        sys.exit(2)
    print_index = args.index("--print")
    if print_index + 1 >= len(args):
        write_log("completion", prompt)
        mark("completion_started")
        mark("completion_exited")
        print("flag needs an argument: -print", file=sys.stderr)
        sys.exit(2)
    prompt = args[print_index + 1]
variadic_failure = False
positionals = []
skip_next = False
for idx, arg in enumerate(args):
    if skip_next:
        skip_next = False
        continue
    if provider == "claude" and arg in ("--tools", "--mcp-config"):
        if idx + 1 < len(args):
            skip_next = True
        variadic_failure = True
        continue
    if arg.startswith("--tools=") or arg.startswith("--mcp-config="):
        continue
    if arg.startswith("-"):
        continue
    positionals.append(arg)

if provider == "claude" and positionals:
    prompt = " ".join(positionals)

write_log("completion", prompt)
mark("completion_started")

if variadic_failure:
    mark("completion_exited")
    print("MCP config not found: prompt was consumed by a variadic flag", file=sys.stderr)
    sys.exit(1)

if read("login_state", "valid") == "expired":
    message = read("expired_marker", "expired")
    mark("completion_exited")
    if provider == "claude":
        print(json.dumps(claude_envelope(message, True, 401)))
    else:
        print(message, file=sys.stderr)
    sys.exit(1)

behavior = read("completion_behavior", "succeed")
if behavior == "hang":
    time.sleep(120)
if behavior in ("sleep-then-succeed", "sleep-then-extract-harvest"):
    time.sleep(1)
if behavior == "billing-required":
    mark("completion_exited")
    if provider == "claude":
        print(json.dumps(claude_envelope("Your Claude account billing or quota must be updated before this request can run.", True, 402)))
    else:
        print("billing or quota must be updated", file=sys.stderr)
    sys.exit(1)
if behavior == "rate-limit":
    mark("completion_exited")
    if provider == "claude":
        print(json.dumps(claude_envelope("Rate limit exceeded for the Claude host session quota. Please retry later.", True, 429)))
    else:
        print("rate limit exceeded", file=sys.stderr)
    sys.exit(1)
if behavior == "fail":
    mark("completion_exited")
    print(read("failure_marker", "fixture failure"), file=sys.stderr)
    sys.exit(1)
if behavior == "empty":
    mark("completion_exited")
    sys.exit(0)

if behavior in ("extract-harvest", "sleep-then-extract-harvest"):
    reply = skill_proposal_json(prompt) or harvest_json(prompt)
    mark("completion_exited")
    if provider == "claude":
        print(json.dumps(claude_envelope(reply)))
    elif provider == "codex":
        output_path = None
        for idx, arg in enumerate(args):
            if arg == "--output-last-message" and idx + 1 < len(args):
                output_path = args[idx + 1]
            elif arg.startswith("--output-last-message="):
                output_path = arg.split("=", 1)[1]
        if output_path:
            pathlib.Path(output_path).write_text(reply)
        else:
            print(reply)
    else:
        print(reply)
    sys.exit(0)

reply = proof(read("secret", "secret"), prompt)
mark("completion_exited")
if provider == "claude":
    if "stream-json" in " ".join(args):
        print(json.dumps({"type": "system", "subtype": "init", "apiKeySource": "none"}))
    print(json.dumps(claude_envelope(reply)))
elif provider == "codex":
    output_path = None
    for idx, arg in enumerate(args):
        if arg == "--output-last-message" and idx + 1 < len(args):
            output_path = args[idx + 1]
        elif arg.startswith("--output-last-message="):
            output_path = arg.split("=", 1)[1]
    if output_path:
        pathlib.Path(output_path).write_text(reply)
    else:
        print(reply)
else:
    print(reply)
"#;
        write_file(
            self.fixture_dir.join("fake_provider.py"),
            driver.trim_start(),
        );
    }
}

pub fn secret_reply(secret: &str, prompt: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in secret
        .as_bytes()
        .iter()
        .copied()
        .chain([0])
        .chain(prompt.as_bytes().iter().copied())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("proof-{hash:016x}")
}

pub fn run_variadic_fake(argv: &[&str], stdin: &str) -> CommandOutput {
    let rig = FakeProviderRig::new();
    rig.install(
        Provider::Claude,
        &unique_label("dev"),
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let mut command = Command::new(rig.bin_dir.join("claude"));
    command
        .args(argv)
        .env("PATH", rig.path_env())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut attempts = 0;
    let mut child = loop {
        match command.spawn() {
            Ok(child) => break child,
            Err(err) if err.raw_os_error() == Some(26) && attempts < 20 => {
                attempts += 1;
                thread::sleep(Duration::from_millis(10));
            }
            Err(err) => panic!("spawn fake claude: {err}"),
        }
    };
    if let Some(mut child_stdin) = child.stdin.take()
        && let Err(err) = child_stdin.write_all(stdin.as_bytes())
    {
        assert_eq!(
            err.kind(),
            ErrorKind::BrokenPipe,
            "write fake claude stdin: {err}"
        );
    }
    let output = child.wait_with_output().expect("fake claude output");
    CommandOutput {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

pub fn catalog_disk_contains(root: &Path, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let needle = needle.as_bytes();
    walk_files(root)
        .into_iter()
        .filter_map(|path| fs::read(path).ok())
        .any(|bytes| bytes.windows(needle.len()).any(|window| window == needle))
}

pub fn base64_basic(input: &str) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut out = String::new();
    let mut chunks = bytes.chunks(3);
    for chunk in chunks.by_ref() {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub fn file_inventory(root: &Path) -> BTreeMap<PathBuf, (u64, u64)> {
    walk_files(root)
        .into_iter()
        .filter_map(|path| {
            let relative = path.strip_prefix(root).ok()?.to_path_buf();
            let bytes = fs::read(&path).ok()?;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut hasher);
            Some((relative, (bytes.len() as u64, hasher.finish())))
        })
        .collect()
}

pub fn write_file(path: impl AsRef<Path>, contents: &str) {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("file parent dir");
    }
    fs::write(path, contents).expect("write fixture file");
}

fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return out;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk_files(&path));
        } else if path.is_file() {
            out.push(path);
        }
    }
    out.sort();
    out
}

fn shell_quote(value: impl AsRef<Path>) -> String {
    let value = value.as_ref().display().to_string();
    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = fs::metadata(path).expect("script metadata").permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).expect("script permissions");
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) {}
