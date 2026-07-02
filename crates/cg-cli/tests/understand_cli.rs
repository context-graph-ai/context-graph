//! CLI acceptance tests for WP2 `cg understand` — the queue drainer.
//!
//! The contract: tier-0 `cg record`/`cg import` leave a pending backlog and write zero graph; a
//! later `cg understand` (foreground) or `cg understand --worker` (the detached post-capture kick)
//! drains that backlog through the connected model into structured Intention/Decision/Entity/
//! Observation memory, stopping at the budget ceiling, with exactly one worker per data root.

mod support;

use std::fs;

use fs4::fs_std::FileExt;
use serde_json::Value as JsonValue;
use support::{CliHarness, CommandOutput, CompletionBehavior, LoginState, Provider};

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

fn pending(status: &JsonValue) -> u64 {
    status
        .pointer("/understanding_queue/pending")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0)
}

fn structured(status: &JsonValue) -> u64 {
    status
        .pointer("/understanding_queue/structured")
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
        "tier-0 record failed: {}",
        combined(&record)
    );

    // Tier-0 capture wrote zero graph and left a pending backlog.
    let before = status_json(&harness, "loop");
    assert_eq!(goals(&before), 0, "tier-0 capture writes no graph");
    assert!(
        pending(&before) >= 1,
        "the digest is queued for understanding"
    );

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
    assert_eq!(pending(&after), 0, "the backlog is drained");
    assert!(
        structured(&after) >= 1,
        "the queue row is marked structured"
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
    assert!(structured(&status_json(&harness, "loop")) >= 1);

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
    // No model connected. Tier-0 capture still works; the worker is a silent no-op.
    let harness = CliHarness::new();
    let record = harness.run_with_stdin(
        &["record", "--no-kick", "--context", "loop"],
        PARITY_TRANSCRIPT,
    );
    assert_eq!(
        record.code,
        0,
        "tier-0 record failed: {}",
        combined(&record)
    );
    let pending_before = pending(&status_json(&harness, "loop"));
    assert!(pending_before >= 1);

    let worker = harness.run(&["understand", "--worker"]);
    assert_eq!(worker.code, 0, "worker exits 0 with no model");
    assert!(
        worker.stdout.trim().is_empty(),
        "the worker is silent: {}",
        worker.stdout
    );
    assert_eq!(
        pending(&status_json(&harness, "loop")),
        pending_before,
        "nothing was drained without a model"
    );
}

#[test]
fn worker_lock_is_exclusive() {
    let harness = connected_harness();
    harness.run_with_stdin(
        &["record", "--no-kick", "--context", "loop"],
        PARITY_TRANSCRIPT,
    );
    let pending_before = pending(&status_json(&harness, "loop"));
    assert!(pending_before >= 1);

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
    assert_eq!(
        pending(&status_json(&harness, "loop")),
        pending_before,
        "the contended worker drained nothing"
    );

    // Release the lock; now a worker drains.
    FileExt::unlock(&lock).expect("unlock");
    drop(lock);
    let drained = harness.run(&["understand", "--worker"]);
    assert_eq!(drained.code, 0);
    assert!(structured(&status_json(&harness, "loop")) >= 1);
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
    assert!(pending(&status_json(&harness, "loop")) >= 2);

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
    assert!(
        pending(&status_json(&harness, "loop")) >= 1,
        "a digest remains pending under the ceiling"
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
