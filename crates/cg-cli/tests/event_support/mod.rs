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
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

pub const NOMIC_MODEL_ID: &str = "nomic-embed-text-v1.5";
pub const NOMIC_REVISION: &str = "e9b6763023c676ca8431644204f50c2b100d9aab";

pub struct Output {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
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

        let transcript = Arc::new(Mutex::new(String::new()));
        let reader_transcript = Arc::clone(&transcript);
        let mut master_reader = fs::File::from(master.try_clone().expect("clone pty master"));
        let reader = std::thread::spawn(move || {
            let mut buffer = [0_u8; 1024];
            loop {
                match master_reader.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => reader_transcript
                        .lock()
                        .expect("transcript lock")
                        .push_str(&String::from_utf8_lossy(&buffer[..read])),
                }
            }
        });
        let mut master_writer = fs::File::from(master);
        let mut seen = 0_usize;
        for (index, (prompt, reply)) in answers.iter().enumerate() {
            let deadline = Instant::now() + Duration::from_secs(60);
            loop {
                let found = {
                    let text = transcript.lock().expect("transcript lock");
                    text[seen..]
                        .find(prompt)
                        .map(|index| seen + index + prompt.len())
                };
                if let Some(end) = found {
                    seen = end;
                    break;
                }
                if Instant::now() > deadline {
                    let _ = child.kill();
                    panic!(
                        "cg never wrote the terminal prompt {prompt:?}; transcript so far: {:?}",
                        transcript.lock().expect("transcript lock")
                    );
                }
                std::thread::sleep(Duration::from_millis(20));
            }
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
        let transcript = transcript.lock().expect("transcript lock").clone();
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
        let deadline = Instant::now() + Duration::from_secs(600);
        loop {
            let status = self.run(cwd, &["status", "--context", context]);
            if status.code == 0 {
                let value = status.json();
                if value
                    .pointer("/history/search/status")
                    .and_then(Value::as_str)
                    == Some("ready")
                {
                    return value;
                }
            }
            if Instant::now() > deadline {
                panic!(
                    "search never became ready for {context}; last status {}",
                    status.describe()
                );
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    /// Poll until the detached search worker process has exited: its per-data-root lock can be
    /// taken again. The worker is another process, so its lock is the state there is to observe.
    pub fn wait_for_search_worker_exit(&self) {
        let lock_path = self.data_root.join("event-search-worker.lock");
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if cg_node_fs::acquire_worker_lock(&lock_path).is_ok() {
                return;
            }
            if Instant::now() > deadline {
                panic!(
                    "the detached search worker still holds {} after 120 s",
                    lock_path.display()
                );
            }
            std::thread::sleep(Duration::from_millis(100));
        }
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
            std::ptr::null(),
            std::ptr::null(),
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
