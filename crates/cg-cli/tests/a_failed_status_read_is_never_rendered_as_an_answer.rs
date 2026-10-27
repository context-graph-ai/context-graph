//! Every read `cg status` makes after the store is open must reach the operator as itself — a
//! failure when it failed, an absence when the store really holds nothing.
//!
//! ## The promise
//!
//! Opening the store is only the first thing `cg status` does. Seven reads follow it, and each one
//! has a rendering that is an ANSWER: `retention_days: null` says "no retention is configured",
//! `last_understanding_run` absent says "understanding has never run", a fabric line with zero
//! copies says "sharing is configured and nothing is queued", `"fabric_jobs":[]` says "nothing was
//! ever submitted", and zero counts say "this store is empty". Every one of those is a statement a
//! developer and an agent act on. A read that FAILED says none of them, and rendering it as one of
//! them is not a missing feature — it is a false statement about somebody's memory.
//!
//! ## How the failure is produced
//!
//! A post-open read failing on demand is not reachable from outside the process, so `cg-cli` ships
//! a fault-injection door behind the `status-fault-injection` feature, which nothing but the
//! package's own dev-dependency on itself turns on (`injected_status_read_failure`,
//! crates/cg-cli/src/main.rs:3972-3985). `CG_FAIL_STATUS_METRIC_READ=<branch>` makes exactly that
//! one read refuse and leaves the open untouched. This file drives all SEVEN branches through the
//! real binary and pins both sides of the promise: a refused read is a refusal, and an honest
//! absence is still a success.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::tier0::{Tier0CaptureRequest, capture_tier0};
use context_graph::{Embedder, EmbedderRegistration, Store, StoreConfig};

/// The environment switch the fault-injection door reads.
const FAULT_ENV: &str = "CG_FAIL_STATUS_METRIC_READ";

/// Every post-open status read that can be refused, with what has to be in the store for the read
/// to be reached at all. Keeping the whole set here — rather than one test per branch — is what
/// makes a NEW unguarded read visible: it will not be in this list, and the roster assertion below
/// says so.
// Read-only status omits retention and raw metrics, and the engine does not expose queue or
// fabric job-state rows to read-only clients.
const REFUSABLE_READS: [&str; 3] = ["understanding_run", "fabric_membership", "counts"];

/// The context label `cg` uses when none is named, so the fixture's captured rows are the ones a
/// bare `cg status` looks for.
const DEFAULT_CONTEXT: &str = "default";

/// Renderings that are ANSWERS about the store. None of them may appear beside a refusal: each one
/// tells a reader something about their memory that a failed read did not establish.
const READS_AS_AN_ANSWER: [&str; 10] = [
    "\"memory\":",
    "\"totals\":",
    "\"understanding_queue\":",
    "\"raw_footprint\":",
    "\"fabric_jobs\":",
    "\"goals\":",
    "\"decisions\":",
    "\"observations\":",
    "\"last_understanding_run\":",
    "goals: 0 decisions: 0 observations: 0",
];

struct CgRun {
    code: i32,
    said: String,
}

fn run_cg(args: &[&str], data_root: &Path, failing_read: Option<&str>) -> CgRun {
    let mut command = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_cg")));
    command
        .args(args)
        .env("CG_DATA_ROOT", data_root)
        .env("CG_USE_TEST_LEXICAL_EMBEDDER", "1")
        .env("HOME", data_root)
        .env_remove(FAULT_ENV)
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(failing_read) = failing_read {
        command.env(FAULT_ENV, failing_read);
    }
    let output = command.output().expect("run the compiled cg binary");
    CgRun {
        code: output.status.code().unwrap_or(-1),
        said: format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    }
}

/// A data root holding a committed store. `captured` decides whether the default context exists
/// (the four reads below `list_contexts` are reached only when it does); `fabric` decides whether
/// a `fabric.toml` is present (the membership read is reached only when it is).
fn data_root(captured: bool, fabric: bool) -> tempfile::TempDir {
    let root = tempfile::Builder::new()
        .prefix("cgf")
        .tempdir_in("/tmp")
        .expect("a cg data root");

    let embedder = std::sync::Arc::new(LexicalMockEmbedder::default());
    let store = Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: root.path().join("context-graph.db"),
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("create the fixture store the way cg creates it");

    if captured {
        let window = 40usize;
        let mut transcript = String::new();
        for unit in 0..8usize {
            let mut chunk = format!("User: note-{unit} ");
            while chunk.len() < window {
                chunk.push('x');
            }
            transcript.push_str(&chunk);
        }
        capture_tier0(
            &store,
            Tier0CaptureRequest {
                context_label: DEFAULT_CONTEXT.to_string(),
                session_ref: "fault-fixture".to_string(),
                transcript,
                source: "test".to_string(),
                chunk_window: window,
            },
        )
        .expect("model-free capture into the default context");
    }
    drop(store);

    if fabric {
        let directory = root.path().join("cg");
        fs::create_dir_all(&directory).expect("create the cg config directory");
        fs::write(directory.join("fabric.toml"), "[fabric]\nsharing = true\n")
            .expect("write the fixture fabric config");
    }
    root
}

/// Every refusable read is one the fixture can actually reach. A read added to the binary without
/// a fault branch, or a branch renamed, shows up here rather than silently going unpinned.
#[test]
fn every_refusable_status_read_is_reached_by_the_fixture() {
    let root = data_root(true, true);
    for read in REFUSABLE_READS {
        let refused = run_cg(&["status"], root.path(), Some(read));
        assert_ne!(
            refused.code, 0,
            "`{read}` is in the refusable roster but the fixture never reaches it — the branch is \
             unreachable, renamed, or the fixture no longer sets the store up for it: {}",
            refused.said
        );
    }
}

#[test]
fn a_refused_status_read_is_reported_as_a_refusal_and_never_as_an_answer() {
    let root = data_root(true, true);

    for read in REFUSABLE_READS {
        let refused = run_cg(&["status"], root.path(), Some(read));
        let said = &refused.said;

        assert_ne!(
            refused.code, 0,
            "the `{read}` read failed, so `cg status` must not exit as a success — the exit code \
             is what a supervisor and a shell script act on: {said}"
        );
        assert!(
            said.contains("\"ok\":false"),
            "the `{read}` refusal must say plainly that it is not an answer: {said}"
        );
        assert!(
            said.contains(&format!("injected failure of the {read} status read")),
            "the refusal must name the read that actually failed — a caller told only that \
             `something` failed cannot tell a missing setting from a broken ledger: {said}"
        );
        assert!(
            said.contains(&root.path().join("context-graph.db").display().to_string()),
            "the refusal names the store it is about: {said}"
        );

        for rendering in READS_AS_AN_ANSWER {
            assert!(
                !said.contains(rendering),
                "the `{read}` read failed, so `{rendering}` must not appear beside it: every one \
                 of those renderings is a STATEMENT about the store that this run never \
                 established: {said}"
            );
        }
    }
}

#[test]
fn a_store_that_really_holds_nothing_still_answers_with_its_absences() {
    // No capture and no fabric config: every absence below is the truth about this store, and the
    // honesty this file asks for must not turn any of them into a refusal.
    let root = data_root(false, false);
    let answered = run_cg(&["status"], root.path(), None);
    let said = &answered.said;

    assert_eq!(
        answered.code, 0,
        "a store that was read and holds nothing is a successful answer, not a failure: {said}"
    );
    assert!(
        said.contains("\"contexts\":[]"),
        "a store with no Context answers with an empty Context list, never invented zeros: {said}"
    );
    assert!(
        said.contains("\"last_understanding_run\":null"),
        "understanding that never ran is an ANSWER — `Ok(None)` from the setting read stays an \
         absence and does not become a refusal: {said}"
    );
    assert!(
        !said.contains("injected"),
        "nothing was injected into this run: {said}"
    );
}

#[test]
fn a_captured_store_answers_with_its_real_totals_when_no_read_refuses() {
    let root = data_root(true, true);
    let answered = run_cg(&["status"], root.path(), None);
    let said = &answered.said;

    assert_eq!(
        answered.code, 0,
        "every read succeeded, so `cg status` answers: {said}"
    );
    assert!(
        said.contains("fabric: sharing on"),
        "the fabric posture is rendered when a config is present: {said}"
    );
}

/// The door is a test door. A branch name nothing matches must change nothing, or the variable
/// would be a way to break a status rather than a way to pin one.
#[test]
fn an_unmatched_branch_name_leaves_the_status_alone() {
    let root = data_root(true, true);
    let answered = run_cg(&["status"], root.path(), Some("not-a-status-read"));
    assert_eq!(
        answered.code, 0,
        "no read is named `not-a-status-read`, so nothing refuses: {}",
        answered.said
    );
    assert!(
        answered.said.contains("fabric: sharing on"),
        "the ordinary answer is unchanged: {}",
        answered.said
    );
}
