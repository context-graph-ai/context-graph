//! `cg status`, `cg find` and `cg show` answer while another process is writing the store.
//!
//! ## The promise
//!
//! An agent asks what its memory holds while an import is still running. It is entitled to an
//! answer: these three commands read, they never write, so who is holding the store writable is
//! none of their business. The answer must be the SAME one the idle store gives — not a refusal, a
//! wait, or a degraded reading — and asking must leave nothing behind: no file created beside the
//! store, no byte of the store changed. A writable open would fail both halves at once, because
//! contextdb is single-owner (a second writable open is refused with the busy condition, exit 4)
//! and because a writable open rewrites the store file on the way in.
//!
//! ## What is proved here, through the compiled binary
//!
//! - On a store nobody holds, all three commands answer and the whole data root is byte-for-byte
//!   what it was: the same files, the same bytes, nothing added.
//! - With a live writer holding that same store, all three answer again, and each answer is
//!   IDENTICAL to the one the idle store gave; the data root is again unchanged. The busy exit code
//!   never appears and neither does any sentence that tells an operator the engine went wrong.
//!   Both halves share one fixture, because building a searchable store means a real Nomic
//!   projection and the gate pays for each one.
//! - With the search projection still pending — captured events, nothing indexed yet — `cg status`
//!   under that same live writer reports the pending projection rather than failing, and `cg find`
//!   gives its ordinary not-searchable-yet answer rather than a busy refusal.
//!
//! The writer is a real second PROCESS taking the store the way a running import takes it, and the
//! harness advances only when that process has printed that it holds it. Nothing here sleeps and
//! nothing here decides anything from elapsed time.
//!
//! The paused-projection-worker rendezvous in `crates/cg-cli/src/history/worker.rs` is not used:
//! it compiles only under `event-search-test-support`, which the repository gate does not enable,
//! so a proof built on it would not run in the gate. A store left with its projection pending is
//! the same condition reached with what every build carries.

#![cfg(unix)]

mod event_support;
#[path = "support/store_contention.rs"]
mod store_contention;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

use event_support::{existing_nomic_cache_root, fingerprint, write_session};
use store_contention::{HoldingChild, READS_AS_A_BROKEN_ENGINE, runtime_root, store_path_in};

/// The Context every command in this file names, so no command depends on a project binding.
const CONTEXT: &str = "ledger";

/// contextdb's exit code for a store another process is writing. Its appearance on any command
/// here would mean that command asked for a writable open.
const STORE_BUSY_EXIT: i32 = 4;

/// The writer-holder child this binary re-invokes. It holds the store and waits to be released.
#[test]
fn a_writer_holder_child() {
    store_contention::run_writer_holder_child_if_asked();
}

struct CgRun {
    code: i32,
    stdout: String,
    stderr: String,
}

impl CgRun {
    fn everything_said(&self) -> String {
        format!(
            "code={} stdout={} stderr={}",
            self.code, self.stdout, self.stderr
        )
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.stdout).unwrap_or_else(|error| {
            panic!(
                "expected one JSON object: {error}; {}",
                self.everything_said()
            )
        })
    }
}

/// One `cg` invocation against `data_root`, with the pinned Nomic artifact already on disk so the
/// search projection is the shipped one rather than a test stand-in.
fn run_cg(args: &[&str], data_root: &Path, runtime: &Path) -> CgRun {
    let output = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_cg")))
        .args(args)
        .current_dir(data_root)
        .env("CG_DATA_ROOT", data_root)
        .env("HOME", data_root)
        .env("CG_MODEL_CACHE_ROOT", existing_nomic_cache_root())
        .env("XDG_RUNTIME_DIR", runtime)
        .env_remove("CG_USE_TEST_LEXICAL_EMBEDDER")
        .env_remove("CLAUDE_CONFIG_DIR")
        .env_remove("XDG_CACHE_HOME")
        .env_remove("OPENAI_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("CODEX_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("run the compiled cg binary");
    CgRun {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// A data root holding a store the shipped binary imported one synthetic session into. `search`
/// is passed straight to `cg import --search`, so a caller chooses between a ready projection and
/// one that is still pending.
struct Fixture {
    _data_root: tempfile::TempDir,
    _runtime: tempfile::TempDir,
    data_root: PathBuf,
    runtime: PathBuf,
}

impl Fixture {
    fn imported(search: &str) -> Self {
        let runtime = runtime_root();
        let data_root = tempfile::Builder::new()
            .prefix("cgd")
            .tempdir_in("/tmp")
            .expect("a cg data root");
        let project = data_root.path().join("work");
        std::fs::create_dir_all(&project).expect("a project directory");
        let project = std::fs::canonicalize(&project).expect("canonical project directory");
        let history_root = data_root.path().join(".claude").join("projects");
        std::fs::create_dir_all(&history_root).expect("a claude history root");
        let session = write_session(
            &history_root,
            &project,
            "adoption-reader-session",
            &[
                (
                    "why did we choose Iroh instead of NATS?",
                    "We chose Iroh for direct peer transport with no broker in the middle.",
                ),
                (
                    "what does the work ledger keep?",
                    "The work ledger keeps append-only rows the fabric slice consumes.",
                ),
            ],
        );
        let fixture = Self {
            data_root: data_root.path().to_path_buf(),
            runtime: runtime.path().to_path_buf(),
            _data_root: data_root,
            _runtime: runtime,
        };
        let imported = fixture.run(&[
            "import",
            session.to_str().expect("unicode session path"),
            "--context",
            CONTEXT,
            "--search",
            search,
        ]);
        assert_eq!(
            imported.code,
            0,
            "the fixture import must succeed: {}",
            imported.everything_said()
        );
        assert!(
            store_path_in(&fixture.data_root).exists(),
            "the fixture import must leave a committed store: {}",
            imported.everything_said()
        );
        fixture
    }

    fn run(&self, args: &[&str]) -> CgRun {
        run_cg(args, &self.data_root, &self.runtime)
    }

    fn hold_as_writer(&self) -> HoldingChild {
        let mut writer = HoldingChild::hold_as_writer(
            &store_path_in(&self.data_root),
            &self.runtime,
            "a_writer_holder_child",
        );
        writer.wait_until_holding_as_writer();
        writer
    }

    /// Every file under the data root and its bytes: what a read must not change.
    fn image(&self) -> Vec<(PathBuf, Vec<u8>)> {
        fingerprint(&self.data_root)
    }

    /// The passage `cg find` returns for the fixture's own question.
    fn a_passage_id(&self) -> String {
        let found = self.run(&["find", "why Iroh", "--context", CONTEXT, "--format", "json"]);
        assert_eq!(
            found.code,
            0,
            "the fixture's search must answer before a passage can be shown: {}",
            found.everything_said()
        );
        found.json()["hits"]
            .as_array()
            .and_then(|hits| hits.first())
            .and_then(|hit| hit["passage_id"].as_str())
            .unwrap_or_else(|| panic!("cg find returned no passage: {}", found.stdout))
            .to_owned()
    }
}

/// Nothing a refused-as-busy command would say ever appears on a read-only answer.
fn assert_never_reads_as_busy_or_broken(run: &CgRun, command: &str) {
    assert_ne!(
        run.code,
        STORE_BUSY_EXIT,
        "`cg {command}` was refused as busy, which only a writable open can be: {}",
        run.everything_said()
    );
    let said = run.everything_said().to_lowercase();
    for fragment in READS_AS_A_BROKEN_ENGINE {
        assert!(
            !said.contains(fragment),
            "`cg {command}` told the operator the engine went wrong (`{fragment}`): {}",
            run.everything_said()
        );
    }
}

#[test]
fn the_read_only_commands_answer_an_idle_store_and_a_held_one_without_changing_a_byte() {
    let fixture = Fixture::imported("wait");
    let passage = fixture.a_passage_id();
    let commands: Vec<(&str, Vec<&str>)> = vec![
        (
            "status",
            vec!["status", "--context", CONTEXT, "--format", "json"],
        ),
        (
            "find",
            vec!["find", "why Iroh", "--context", CONTEXT, "--format", "json"],
        ),
        (
            "show",
            vec!["show", &passage, "--context", CONTEXT, "--format", "json"],
        ),
    ];

    // Nobody is holding the store: each command answers, and the whole data root is what it was.
    let before_idle = fixture.image();
    let idle: Vec<Value> = commands
        .iter()
        .map(|(command, args)| {
            let run = fixture.run(args);
            assert_eq!(
                run.code,
                0,
                "`cg {command}` must answer an idle store: {}",
                run.everything_said()
            );
            assert_never_reads_as_busy_or_broken(&run, command);
            run.json()
        })
        .collect();
    assert_eq!(
        before_idle,
        fixture.image(),
        "reading an idle store created nothing and changed nothing under its data root"
    );

    // A second process now holds the store writable, the way a running import holds it.
    let writer = fixture.hold_as_writer();
    let before_held = fixture.image();
    for ((command, args), idle_answer) in commands.iter().zip(&idle) {
        let run = fixture.run(args);
        assert_never_reads_as_busy_or_broken(&run, command);
        assert_eq!(
            run.code,
            0,
            "`cg {command}` must answer a store a writer is holding: {}",
            run.everything_said()
        );
        assert_eq!(
            &run.json(),
            idle_answer,
            "`cg {command}` answered a held store differently from the idle store"
        );
    }
    let after_held = fixture.image();
    writer.release();
    assert_eq!(
        before_held, after_held,
        "reading a held store created nothing and changed nothing under its data root"
    );
}

#[test]
fn status_reports_a_pending_projection_under_a_live_writer_rather_than_failing() {
    let fixture = Fixture::imported("defer");

    let writer = fixture.hold_as_writer();
    let before = fixture.image();

    let status = fixture.run(&["status", "--context", CONTEXT, "--format", "json"]);
    assert_never_reads_as_busy_or_broken(&status, "status");
    assert_eq!(
        status.code,
        0,
        "`cg status` must answer while an import's projection is still owed: {}",
        status.everything_said()
    );
    let answered = status.json();
    assert_eq!(
        answered["history"]["captured_events"].as_u64(),
        Some(4),
        "status reports the events the import captured: {}",
        status.stdout
    );
    let search = &answered["history"]["search"];
    assert_ne!(
        search["status"].as_str(),
        Some("ready"),
        "this fixture's projection is deliberately still owed: {}",
        status.stdout
    );
    assert_eq!(
        search["pending_events"].as_u64(),
        Some(4),
        "status names the events the projection still owes: {}",
        status.stdout
    );

    // The same route, asked for something it cannot answer yet: still its own answer, never a
    // refusal about who is holding the store.
    let found = fixture.run(&["find", "why Iroh", "--context", CONTEXT, "--format", "json"]);
    assert_never_reads_as_busy_or_broken(&found, "find");
    assert_eq!(
        found.json()["state"].as_str(),
        Some("history_search_not_ready"),
        "cg find answers that the history is not searchable yet: {}",
        found.everything_said()
    );

    let after = fixture.image();
    writer.release();
    assert_eq!(
        before, after,
        "reading a held store created nothing and changed nothing under its data root"
    );
}
