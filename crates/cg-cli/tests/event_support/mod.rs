//! Shared harness for the event-history CLI journeys.
//!
//! Every run drives the compiled `cg` binary with an isolated data root, an isolated home whose
//! `.claude/projects` holds independently authored synthetic Claude Code history, and an explicit
//! model cache root. Non-interactive runs start in a new session so they can never reach the
//! developer's terminal; interactive runs get a real pseudo-terminal as their controlling terminal.

#![allow(dead_code)]

use std::fs;
use std::io::{Read, Write};
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

pub const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
pub const NOMIC_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";

pub struct Output {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Own one asynchronous fixture process until it is reaped. Failure cleanup targets only the
/// child returned by its spawn, including when an assertion unwinds before a rendezvous release.
pub struct TestChild {
    child: Child,
    reaped: bool,
}

impl TestChild {
    pub fn new(child: Child) -> Self {
        Self {
            child,
            reaped: false,
        }
    }

    pub fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn kill(&mut self) -> std::io::Result<()> {
        self.child.kill()
    }

    pub fn wait(&mut self) -> std::io::Result<ExitStatus> {
        let status = self.child.wait()?;
        self.reaped = true;
        Ok(status)
    }

    pub fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        let status = self.child.try_wait()?;
        self.reaped |= status.is_some();
        Ok(status)
    }

    pub fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        drop(self.child.stdin.take());
        // Drain both pipes concurrently while retaining the Child in this owner: delegating to
        // Child::wait_with_output would consume it before an I/O failure could run cleanup.
        let stdout_reader = self.child.stdout.take().map(|mut pipe| {
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                pipe.read_to_end(&mut bytes).map(|_| bytes)
            })
        });
        let mut stderr = Vec::new();
        let stderr_result = match self.child.stderr.take() {
            Some(mut pipe) => pipe.read_to_end(&mut stderr).map(|_| ()),
            None => Ok(()),
        };
        if stderr_result.is_err() {
            // Make the stdout reader finish before joining it on this failure path.
            let _ = self.kill();
            let _ = self.wait();
        }
        let stdout = match stdout_reader {
            Some(reader) => reader
                .join()
                .map_err(|_| std::io::Error::other("fixture stdout reader panicked"))??,
            None => Vec::new(),
        };
        stderr_result?;
        let status = self.wait()?;
        Ok(std::process::Output {
            status,
            stdout,
            stderr,
        })
    }
}

impl Drop for TestChild {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

struct StateEvents {
    _directory: tempfile::TempDir,
    path: PathBuf,
    receiver: std::os::unix::net::UnixDatagram,
}

impl StateEvents {
    fn new() -> Self {
        let directory = tempfile::tempdir_in("/tmp").expect("short state event directory");
        let path = directory.path().join("state.sock");
        let receiver = std::os::unix::net::UnixDatagram::bind(&path).expect("state event socket");
        Self {
            _directory: directory,
            path,
            receiver,
        }
    }
}

thread_local! { static STATE_EVENTS: StateEvents = StateEvents::new(); }

pub fn state_event_path() -> PathBuf {
    STATE_EVENTS.with(|events| events.path.clone())
}

pub fn write_test_file(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> std::io::Result<()> {
    cg_node_fs::write_test_file_for_test(path, contents)
}

/// Recheck authoritative state after an explicit fixture write or worker completion.
/// The old allowance arguments remain for callers; the bundle runner owns failure deadlines.
pub fn wait_for_state<T, E>(
    _allowance: Duration,
    _interval: Duration,
    mut observe: impl FnMut() -> Result<Option<T>, E>,
) -> Result<Option<T>, E> {
    loop {
        if let Some(value) = observe()? {
            return Ok(Some(value));
        }
        STATE_EVENTS.with(|events| events.receiver.recv(&mut [0]).expect("fixture state event"));
    }
}

impl Output {
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.stdout).unwrap_or_else(|error| {
            panic!(
                "expected one JSON object on stdout: {error}; code={} stdout={} stderr={}",
                self.code, self.stdout, self.stderr
            )
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "code={} stdout={} stderr={}",
            self.code, self.stdout, self.stderr
        )
    }
}

/// One isolated machine: a home, a data root and a model cache root.
pub struct Machine {
    _root: tempfile::TempDir,
    pub root: PathBuf,
    pub home: PathBuf,
    pub data_root: PathBuf,
    pub model_cache: PathBuf,
    state_events_path: PathBuf,
}

impl Machine {
    /// A machine whose model cache is empty, so any model preflight finds nothing installed.
    pub fn without_model() -> Self {
        let root = tempfile::tempdir().expect("isolated machine root");
        let base = fs::canonicalize(root.path()).expect("canonical machine root");
        let home = base.join("home");
        fs::create_dir_all(home.join(".claude").join("projects")).expect("claude history root");
        let model_cache = base.join("empty-model-cache");
        fs::create_dir_all(&model_cache).expect("empty model cache");
        Self {
            state_events_path: state_event_path(),
            data_root: base.join("data-root"),
            home,
            model_cache,
            root: base,
            _root: root,
        }
    }

    /// A machine that uses the already-installed pinned Nomic artifact. No download is allowed.
    pub fn with_nomic() -> Self {
        let mut machine = Self::without_model();
        machine.model_cache = existing_nomic_cache_root();
        machine
    }

    pub fn history_root(&self) -> PathBuf {
        self.home.join(".claude").join("projects")
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_root.join("context-graph.db")
    }

    /// Create a project working directory with the given name under this machine.
    pub fn project(&self, name: &str) -> PathBuf {
        let path = self.root.join("work").join(name);
        fs::create_dir_all(&path).expect("project directory");
        fs::canonicalize(&path).expect("canonical project")
    }

    fn command(&self, cwd: &Path, args: &[&str]) -> Command {
        self.command_using(Path::new(env!("CARGO_BIN_EXE_cg")), cwd, args)
    }

    fn command_using(&self, binary: &Path, cwd: &Path, args: &[&str]) -> Command {
        let mut command = Command::new(binary);
        command
            .args(args)
            .current_dir(cwd)
            .env("CG_DATA_ROOT", &self.data_root)
            .env("CG_TEST_STATE_EVENTS", &self.state_events_path)
            .env("HOME", &self.home)
            .env("CG_MODEL_CACHE_ROOT", &self.model_cache)
            .env_remove("CLAUDE_CONFIG_DIR")
            .env_remove("XDG_CACHE_HOME")
            .env_remove("OPENAI_API_KEY")
            .env_remove("ANTHROPIC_API_KEY")
            .env_remove("CODEX_API_KEY")
            .env_remove("CG_USE_TEST_LEXICAL_EMBEDDER")
            .env_remove("CG_NO_UNDERSTAND_KICK");
        command
    }

    /// Run cg with no controlling terminal at all.
    pub fn run(&self, cwd: &Path, args: &[&str]) -> Output {
        let mut command = self.command(cwd, args);
        Self::run_command(&mut command)
    }

    /// Run cg with one internal test environment value while preserving the ordinary journey.
    pub fn run_with_env(&self, cwd: &Path, args: &[&str], key: &str, value: &Path) -> Output {
        let mut command = self.command(cwd, args);
        command.env(key, value);
        Self::run_command(&mut command)
    }

    /// Start one ordinary CLI journey while the caller observes an existing internal rendezvous.
    /// The command uses the same isolated roots and session setup as `run_with_env`.
    pub fn spawn_with_env(&self, cwd: &Path, args: &[&str], key: &str, value: &Path) -> TestChild {
        self.spawn_with_envs(cwd, args, &[(key, value)])
    }

    /// Start a journey with multiple existing internal rendezvous directories.
    pub fn spawn_with_envs(&self, cwd: &Path, args: &[&str], envs: &[(&str, &Path)]) -> TestChild {
        let mut command = self.command(cwd, args);
        for (key, value) in envs {
            command.env(key, value);
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // SAFETY: setsid is async-signal-safe and only detaches the child from any terminal.
        unsafe {
            command.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
        TestChild::new(command.spawn().expect("start the compiled cg binary"))
    }

    fn run_command(command: &mut Command) -> Output {
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // SAFETY: setsid is async-signal-safe and only detaches the child from any terminal.
        unsafe {
            command.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
        let output = command.output().expect("run the compiled cg binary");
        Output {
            code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }
    }

    /// Run cg with a pseudo-terminal as its controlling terminal and stdin, answering each prompt
    /// fragment with the paired reply. stdout and stderr stay pipes, so the terminal transcript is
    /// separate from the command's structured output.
    pub fn run_interactive(
        &self,
        cwd: &Path,
        args: &[&str],
        answers: &[(&str, &str)],
    ) -> (Output, String) {
        self.run_interactive_using(
            Path::new(env!("CARGO_BIN_EXE_cg")),
            cwd,
            args,
            answers,
            &mut |_| {},
        )
    }

    /// [`Machine::run_interactive`] with an explicit binary, calling `before_reply` with each
    /// prompt's index after the prompt appears and before it is answered.
    pub fn run_interactive_using(
        &self,
        binary: &Path,
        cwd: &Path,
        args: &[&str],
        answers: &[(&str, &str)],
        before_reply: &mut dyn FnMut(usize),
    ) -> (Output, String) {
        let (master, slave) = open_pty();
        let mut command = self.command_using(binary, cwd, args);
        command
            .stdin(Stdio::from(slave.try_clone().expect("clone pty slave")))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // SAFETY: setsid and the TIOCSCTTY ioctl are async-signal-safe; they make the inherited
        // pseudo-terminal the child's controlling terminal.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::ioctl(0, libc::TIOCSCTTY as _, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command.spawn().expect("spawn cg on a pseudo-terminal");
        // The command still holds the slave as the child's stdin; release every parent copy so the
        // master reports end of input once the child exits.
        drop(command);
        drop(slave);

        let transcript = Arc::new((
            Mutex::new((String::new(), false)),
            std::sync::Condvar::new(),
        ));
        let reader_transcript = Arc::clone(&transcript);
        let mut master_reader = fs::File::from(master.try_clone().expect("clone pty master"));
        let reader = std::thread::spawn(move || {
            let mut buffer = [0_u8; 1024];
            loop {
                let read = master_reader.read(&mut buffer);
                let mut state = reader_transcript.0.lock().expect("transcript lock");
                match read {
                    Ok(0) | Err(_) => {
                        state.1 = true;
                        reader_transcript.1.notify_all();
                        break;
                    }
                    Ok(read) => state.0.push_str(&String::from_utf8_lossy(&buffer[..read])),
                }
                reader_transcript.1.notify_all();
            }
        });
        let mut master_writer = fs::File::from(master);
        let mut seen = 0_usize;
        for (index, (prompt, reply)) in answers.iter().enumerate() {
            let mut state = transcript.0.lock().expect("transcript lock");
            while !state.1 && !state.0[seen..].contains(prompt) {
                state = transcript.1.wait(state).expect("terminal output event");
            }
            let end = state.0[seen..]
                .find(prompt)
                .map(|index| seen + index + prompt.len())
                .unwrap_or_else(|| {
                    panic!(
                        "cg closed its terminal before prompt {prompt:?}: {:?}",
                        state.0
                    )
                });
            drop(state);
            seen = end;
            before_reply(index);
            master_writer
                .write_all(reply.as_bytes())
                .expect("answer the terminal prompt");
        }
        let mut stdout = String::new();
        let mut stderr = String::new();
        child
            .stdout
            .take()
            .expect("stdout pipe")
            .read_to_string(&mut stdout)
            .expect("read stdout");
        child
            .stderr
            .take()
            .expect("stderr pipe")
            .read_to_string(&mut stderr)
            .expect("read stderr");
        let status = child.wait().expect("wait for cg");
        drop(master_writer);
        let _ = reader.join();
        let transcript = transcript.0.lock().expect("transcript lock").0.clone();
        (
            Output {
                code: status.code().unwrap_or(-1),
                stdout,
                stderr,
            },
            transcript,
        )
    }

    /// Poll `cg status` until the named Context's search is ready.
    pub fn wait_for_search_ready(&self, cwd: &Path, context: &str) -> Value {
        let mut last_status = None;
        let ready = wait_for_state(Duration::from_secs(600), Duration::from_millis(250), || {
            let status = self.run(cwd, &["status", "--context", context]);
            let ready = if status.code == 0 {
                let value = status.json();
                if value
                    .pointer("/history/search/status")
                    .and_then(Value::as_str)
                    == Some("ready")
                {
                    Some(value)
                } else {
                    None
                }
            } else {
                None
            };
            last_status = Some(status);
            Ok::<_, ()>(ready)
        })
        .expect("status observation cannot fail");
        ready.unwrap_or_else(|| {
            panic!(
                "search never became ready for {context}; last status {}",
                last_status
                    .expect("at least one status observation")
                    .describe()
            )
        })
    }

    /// The kernel wakes this acquire after the worker releases its lock, even on SIGTERM.
    pub fn wait_for_search_worker_exit(&self) {
        let lock_path = self.data_root.join("event-search-worker.lock");
        let _released = cg_node_fs::wait_for_worker_lock_for_test(&lock_path)
            .expect("detached worker released its lock");
    }
}

fn open_pty() -> (OwnedFd, OwnedFd) {
    let mut master = 0;
    let mut slave = 0;
    // SAFETY: openpty writes two new descriptors into the provided integers.
    let result = unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(result, 0, "openpty: {}", std::io::Error::last_os_error());
    // SAFETY: both descriptors were just returned by openpty and are owned here.
    unsafe { (OwnedFd::from_raw_fd(master), OwnedFd::from_raw_fd(slave)) }
}

pub fn existing_nomic_cache_root() -> PathBuf {
    let cache_root = cg_catalog::default_cache_root();
    let model_dir = cache_root.join(NOMIC_MODEL_ID);
    for file in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "REVISION",
    ] {
        assert!(
            model_dir.join(file).is_file(),
            "this journey requires the existing pinned Nomic artifact at {}; missing {file}. \
             No lexical or downloaded embedding fallback is allowed",
            model_dir.display()
        );
    }
    let revision = fs::read_to_string(model_dir.join("REVISION")).expect("read Nomic revision");
    assert_eq!(revision.trim(), NOMIC_REVISION, "pinned Nomic revision");
    cache_root
}

/// Claude Code's project-store name for a working directory: every character outside ASCII
/// letters and digits becomes `-`.
pub fn claude_project_key(project: &Path) -> String {
    project
        .to_str()
        .expect("unicode project path")
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect()
}

/// One synthetic Claude Code session transcript for `project`, one user/assistant exchange per
/// entry of `turns`. Returns the file path.
pub fn write_session(
    history_root: &Path,
    project: &Path,
    session_id: &str,
    turns: &[(&str, &str)],
) -> PathBuf {
    let directory = history_root.join(claude_project_key(project));
    fs::create_dir_all(&directory).expect("project history directory");
    let path = directory.join(format!("{session_id}.jsonl"));
    let mut lines = String::new();
    for (index, (user, assistant)) in turns.iter().enumerate() {
        let user_record = serde_json::json!({
            "type": "user",
            "uuid": uuid::Uuid::new_v4().to_string(),
            "sessionId": session_id,
            "cwd": project.display().to_string(),
            "timestamp": format!("2026-09-01T09:{:02}:00Z", index * 2),
            "message": {"role": "user", "content": user}
        });
        let assistant_record = serde_json::json!({
            "type": "assistant",
            "uuid": uuid::Uuid::new_v4().to_string(),
            "sessionId": session_id,
            "cwd": project.display().to_string(),
            "timestamp": format!("2026-09-01T09:{:02}:30Z", index * 2),
            "message": {
                "id": format!("{session_id}-m{index}"),
                "role": "assistant",
                "content": [{"type": "text", "text": assistant}]
            }
        });
        lines.push_str(&user_record.to_string());
        lines.push('\n');
        lines.push_str(&assistant_record.to_string());
        lines.push('\n');
    }
    fs::write(&path, lines).expect("write session transcript");
    path
}

/// A helper-agent transcript filed under its parent session, the way Claude Code stores it.
pub fn write_helper_transcript(
    history_root: &Path,
    project: &Path,
    parent_session: &str,
    agent: &str,
    text: &str,
) -> PathBuf {
    let directory = history_root
        .join(claude_project_key(project))
        .join(parent_session)
        .join("subagents");
    fs::create_dir_all(&directory).expect("helper transcript directory");
    let path = directory.join(format!("agent-{agent}.jsonl"));
    let record = serde_json::json!({
        "type": "user",
        "uuid": uuid::Uuid::new_v4().to_string(),
        "sessionId": parent_session,
        "agentId": agent,
        "isSidechain": true,
        "cwd": project.display().to_string(),
        "timestamp": "2026-09-01T10:00:00Z",
        "message": {"role": "user", "content": text}
    });
    fs::write(&path, format!("{record}\n")).expect("write helper transcript");
    path
}

/// Every file below `root` with its bytes, for proving a command left a directory untouched.
pub fn fingerprint(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut found = Vec::new();
    if !root.exists() {
        return found;
    }
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).expect("read directory") {
            let entry = entry.expect("directory entry");
            let path = entry.path();
            if entry.file_type().expect("entry type").is_dir() {
                found.push((path.clone(), Vec::new()));
                pending.push(path);
            } else {
                found.push((path.clone(), fs::read(&path).unwrap_or_default()));
            }
        }
    }
    found.sort_by(|left, right| left.0.cmp(&right.0));
    found
}

pub fn str_at<'a>(value: &'a Value, pointer: &str) -> &'a str {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{pointer} must be a string in {value}"))
}

pub fn u64_at(value: &Value, pointer: &str) -> u64 {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("{pointer} must be a count in {value}"))
}
