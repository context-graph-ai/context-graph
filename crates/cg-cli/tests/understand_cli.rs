//! CLI acceptance tests for  `cg understand` — the queue drainer.
//!
//! The contract: model-free `cg record`/`cg import` leave a pending backlog and write zero graph; a
//! later `cg understand` (foreground) or `cg understand --worker` (the detached post-capture kick)
//! drains that backlog through the connected model into structured Intention/Decision/Entity/
//! Observation memory, stopping at the budget ceiling, with exactly one worker per data root.

mod support;

use std::fs;

use contextdb_core::Value;
use contextdb_engine::Database;
use fs4::fs_std::FileExt;
use serde_json::Value as JsonValue;
use support::{CliHarness, CommandOutput, CompletionBehavior, LoginState, Provider};

/// Read the structured graph a store persisted, as sorted description lists — the comparable shape
/// modulo row ids. Used to prove `--understand-now` and model-free-then-`cg understand` are ONE path.
fn graph_snapshot(harness: &CliHarness, context: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let status = harness.run(&["status", "--context", context]);
    let json = output_json(&status);
    let db_path = json
        .pointer("/store/db_path")
        .or_else(|| json.pointer("/store_path"))
        .or_else(|| json.pointer("/db_path"))
        .and_then(JsonValue::as_str)
        .unwrap_or_else(|| panic!("db path absent from status JSON: {json}"));
    let db = Database::open(db_path).expect("open store");
    let snapshot = db.snapshot();
    let column = |table: &str, column: &str| -> Vec<String> {
        let mut values = db
            .scan(table, snapshot)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|row| match row.values.get(column) {
                Some(Value::Text(text)) => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        values.sort();
        values
    };
    (
        column("intentions", "description"),
        column("decisions", "description"),
        column("observations", "observation_type"),
    )
}

fn combined(output: &CommandOutput) -> String {
    format!("{}\n{}", output.stdout, output.stderr)
}

fn output_json(output: &CommandOutput) -> JsonValue {
    serde_json::from_str(&output.stdout)
        .unwrap_or_else(|err| panic!("cg output was not JSON ({err}): {}", output.stdout))
}

/// A harness with a fake Claude host session connected and driving ExtractHarvest replies.
fn connected_harness() -> CliHarness {
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
    assert_eq!(login.code, 0, "fake host login failed: {}", login.stderr);
    harness
        .providers
        .set_behavior(Provider::Claude, CompletionBehavior::ExtractHarvest);
    harness
}

const PARITY_TRANSCRIPT: &str = "Goal G1: keep the ingest path tracked. Entity E1: src/ingest.rs path=src/ingest.rs. Decision D1: keep ingest at src/ingest.rs serves G1 based_on E1.";

fn status_json(harness: &CliHarness, context: &str) -> JsonValue {
    let output = harness.run(&["status", "--context", context]);
    output_json(&output)
}

fn goals(status: &JsonValue) -> u64 {
    status
        .pointer("/goals")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0)
}

#[test]
fn record_then_understand_structures_the_graph() {
    let harness = connected_harness();
    let record = harness.run_with_stdin(
        &["record", "--no-kick", "--context", "loop"],
        PARITY_TRANSCRIPT,
    );
    assert_eq!(
        record.code,
        0,
        "model-free record failed: {}",
        combined(&record)
    );

    // model-free capture wrote zero graph and left a pending backlog.
    let before = status_json(&harness, "loop");
    assert_eq!(goals(&before), 0, "model-free capture writes no graph");

    // Drain the backlog through the connected model.
    let understand = harness.run(&["understand", "--context", "loop"]);
    assert_eq!(
        understand.code,
        0,
        "understand failed: {}",
        combined(&understand)
    );
    let drained = output_json(&understand);
    assert!(
        drained
            .pointer("/understood")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0)
            >= 1,
        "understand structured at least one digest: {}",
        understand.stdout
    );

    // The graph is now structured and the backlog is drained.
    let after = status_json(&harness, "loop");
    assert!(goals(&after) >= 1, "understanding produced a goal: {after}");
    assert!(
        after
            .pointer("/decisions")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0)
            >= 1,
        "understanding produced a decision: {after}"
    );
}

#[test]
fn understand_worker_drains_then_repeat_is_noop() {
    let harness = connected_harness();
    harness.run_with_stdin(
        &["record", "--no-kick", "--context", "loop"],
        PARITY_TRANSCRIPT,
    );
    let first = harness.run(&["understand", "--worker"]);
    assert_eq!(first.code, 0, "worker failed: {}", combined(&first));
    let after_first = harness.providers.completion_count();
    assert!(
        after_first >= 1,
        "the worker drove the model at least once (completions={after_first})"
    );

    // Queue empty → a second worker run drives no further completions.
    let second = harness.run(&["understand", "--worker"]);
    assert_eq!(second.code, 0);
    assert_eq!(
        harness.providers.completion_count(),
        after_first,
        "a drained queue is a no-op"
    );
}

#[test]
fn worker_is_a_silent_noop_without_a_model() {
    // No model connected. model-free capture still works; the worker is a silent no-op.
    let harness = CliHarness::new();
    let record = harness.run_with_stdin(
        &["record", "--no-kick", "--context", "loop"],
        PARITY_TRANSCRIPT,
    );
    assert_eq!(
        record.code,
        0,
        "model-free record failed: {}",
        combined(&record)
    );
    let worker = harness.run(&["understand", "--worker"]);
    assert_eq!(worker.code, 0, "worker exits 0 with no model");
    assert!(
        worker.stdout.trim().is_empty(),
        "the worker is silent: {}",
        worker.stdout
    );
}

#[test]
fn worker_lock_is_exclusive() {
    let harness = connected_harness();
    harness.run_with_stdin(
        &["record", "--no-kick", "--context", "loop"],
        PARITY_TRANSCRIPT,
    );
    // Hold the per-data-root worker lock from the test, exactly as a running worker would.
    let lock_path = harness.data_root.path().join("understand-worker.lock");
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .expect("open lock file");
    assert!(
        FileExt::try_lock_exclusive(&lock).expect("lock"),
        "test acquires the lock first"
    );

    // A worker that cannot get the lock exits 0 and drains nothing.
    let contended = harness.run(&["understand", "--worker"]);
    assert_eq!(contended.code, 0, "a contended worker exits 0");
    assert_eq!(
        harness.providers.completion_count(),
        0,
        "the contended worker drove no model calls"
    );

    // Release the lock; now a worker drains.
    FileExt::unlock(&lock).expect("unlock");
    drop(lock);
    let drained = harness.run(&["understand", "--worker"]);
    assert_eq!(drained.code, 0);
}

#[test]
fn budget_ceiling_stops_the_drain() {
    let harness = connected_harness();
    // Two distinct sessions → two pending digests.
    harness.run_with_stdin(
        &[
            "record",
            "--no-kick",
            "--context",
            "loop",
            "--session",
            "s1",
        ],
        PARITY_TRANSCRIPT,
    );
    harness.run_with_stdin(
        &[
            "record",
            "--no-kick",
            "--context",
            "loop",
            "--session",
            "s2",
        ],
        "Goal G2: keep the auth path tracked. Decision D2: keep auth at src/auth.py serves G2.",
    );
    // A 1-token ceiling: the first digest runs (spend starts at zero), then the ceiling trips.
    let understand = harness.run(&["understand", "--context", "loop", "--budget", "1"]);
    assert_eq!(
        understand.code,
        0,
        "understand failed: {}",
        combined(&understand)
    );
    let drained = output_json(&understand);
    assert_eq!(
        drained.pointer("/understood").and_then(JsonValue::as_u64),
        Some(1),
        "the budget stops the drain after the first digest: {}",
        understand.stdout
    );
}

const GENERATION_TIME_SKILL: &str = "# description: guidance about the ingest path\n## rule\nalways classify the ingest path decision as durable\n";

#[test]
fn generation_time_steering_reaches_the_extraction_prompt() {
    let harness = connected_harness();
    // Author a generation-time steering rule whose description matches the ingest transcript.
    let skill_path = harness.data_root.path().join("ingest-steer.md");
    fs::write(&skill_path, GENERATION_TIME_SKILL).expect("write skill file");
    let add = harness.run(&[
        "skills",
        "add",
        "--strategy",
        "generation-time",
        skill_path.to_str().unwrap(),
    ]);
    assert_eq!(add.code, 0, "skills add failed: {}", combined(&add));

    harness.run_with_stdin(
        &["record", "--no-kick", "--context", "loop"],
        PARITY_TRANSCRIPT,
    );
    let understand = harness.run(&["understand", "--context", "loop"]);
    assert_eq!(
        understand.code,
        0,
        "understand failed: {}",
        combined(&understand)
    );

    // The steering block was appended to the extraction prompt the model actually received.
    let saw_steering = harness.providers.completion_records().iter().any(|record| {
        record
            .get("prompt")
            .and_then(JsonValue::as_str)
            .map(|prompt| {
                prompt.contains("Apply these learned corrections")
                    && prompt.contains("always classify the ingest path decision as durable")
            })
            .unwrap_or(false)
    });
    assert!(
        saw_steering,
        "the generation-time steering block reached the extraction prompt"
    );
}

#[test]
fn skills_add_rejects_an_unknown_strategy() {
    let harness = CliHarness::new();
    let skill_path = harness.data_root.path().join("steer.md");
    fs::write(&skill_path, GENERATION_TIME_SKILL).expect("write skill file");
    let add = harness.run(&[
        "skills",
        "add",
        "--strategy",
        "teleport",
        skill_path.to_str().unwrap(),
    ]);
    assert_ne!(add.code, 0, "an unknown strategy is rejected loudly");
    assert!(
        combined(&add).contains("unknown --strategy"),
        "the error names the unknown strategy: {}",
        combined(&add)
    );
}

const EQUIVALENCE_TRANSCRIPT: &str = "Acme needs password-less login to ship. We found the auth module file at src/auth.py and decided the auth module stays at src/auth.py. Use JWT with 15-min access tokens for the login flow.";

/// The convergence contract: understanding reaches the store through ONE implementation. Capturing
/// input X with `--understand-now` (model-free capture + a FOREGROUND drain of the just-captured
/// digests) produces the SAME structured graph as model-free-capturing X and then running
/// `cg understand` (the background/foreground drainer over the backlog) — same intentions,
/// decisions, and observations, modulo row ids. If the two paths ever diverged, one of them would
/// be a second, duplicate understanding pipeline.
#[test]
fn understand_now_and_tier0_then_understand_produce_the_same_graph() {
    // Path A: `--understand-now` (model-free capture, then a blocking foreground drain).
    let eager = connected_harness();
    let eager_capture = eager.run_with_stdin(
        &["record", "--understand-now", "--context", "loop"],
        EQUIVALENCE_TRANSCRIPT,
    );
    assert_eq!(
        eager_capture.code,
        0,
        "--understand-now failed: {}",
        combined(&eager_capture)
    );

    // Path B: default model-free capture, then the `cg understand` drainer.
    let deferred = connected_harness();
    let deferred_capture = deferred.run_with_stdin(
        &["record", "--no-kick", "--context", "loop"],
        EQUIVALENCE_TRANSCRIPT,
    );
    assert_eq!(
        deferred_capture.code,
        0,
        "model-free record failed: {}",
        combined(&deferred_capture)
    );
    let understand = deferred.run(&["understand", "--context", "loop"]);
    assert_eq!(
        understand.code,
        0,
        "cg understand failed: {}",
        combined(&understand)
    );

    let eager_graph = graph_snapshot(&eager, "loop");
    let deferred_graph = graph_snapshot(&deferred, "loop");
    assert!(
        !eager_graph.0.is_empty() && !eager_graph.1.is_empty(),
        "both paths must actually structure a graph, not silently no-op: {eager_graph:?}"
    );
    assert_eq!(
        eager_graph, deferred_graph,
        "--understand-now and model-free + cg understand must produce the SAME structured graph \
         (one understanding path), modulo ids"
    );
}
