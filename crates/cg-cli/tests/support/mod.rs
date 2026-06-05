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
    let mut command = Command::new(cg_bin());
    command
        .args(args)
        .env("CG_DATA_ROOT", data_root)
        .env("HOME", data_root)
        .env("PATH", path_env)
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().expect("spawn compiled cg binary");
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

def write_log(mode, prompt=""):
    record = {
        "provider": provider,
        "mode": mode,
        "argv": args,
        "prompt": prompt,
        "env": {name: os.environ[name] for name in TRACKED_ENV if name in os.environ},
    }
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

def claude_envelope(reply, is_error=False, status=None):
    return {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "api_error_status": status,
        "result": reply,
        "usage": {"input_tokens": 13, "output_tokens": len(reply.split())},
        "modelUsage": {"claude-sonnet-4-20250514": {"inputTokens": 13, "outputTokens": len(reply.split())}},
    }

if "--help" in args:
    write_log("probe")
    if read("help_probe_behavior", "normal") == "hang":
        time.sleep(120)
    if provider == "codex" and args and args[0] == "exec":
        print(read("exec_help_flags", "--output-last-message\n-C"))
    else:
        print(read("help_flags", "--output-format\n--mcp-config\n--strict-mcp-config\n--tools"))
    sys.exit(0)

if "--version" in args:
    write_log("probe")
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
if behavior == "sleep-then-succeed":
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
