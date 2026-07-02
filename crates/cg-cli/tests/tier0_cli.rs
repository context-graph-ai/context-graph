//! CLI acceptance tests for WP1 tier-0 deterministic capture.
//!
//! `cg record` / `cg import` are the deterministic, free front door: zero model calls even when a
//! model is connected, an honest zero-cost estimate, a purge command (`cg forget`), and `cg status`
//! surfacing the understanding-queue backlog and raw footprint. Tests pass `--no-kick` so the
//! detached background worker is not spawned during the test run.

mod support;

use serde_json::Value as JsonValue;
use support::{CliHarness, CommandOutput, write_file};

fn combined(output: &CommandOutput) -> String {
    format!("{}\n{}", output.stdout, output.stderr)
}

fn output_json(output: &CommandOutput) -> JsonValue {
    serde_json::from_str(&output.stdout)
        .unwrap_or_else(|err| panic!("cg output was not JSON ({err}): {}", output.stdout))
}

#[test]
fn record_default_is_tier0_zero_model_calls() {
    let harness = CliHarness::new();
    let output = harness.run(&[
        "record",
        "--no-kick",
        "--context",
        "loop",
        "User: we chose sqlite for the local cache",
    ]);
    assert_eq!(output.code, 0, "record failed: {}", combined(&output));
    let json = output_json(&output);
    assert_eq!(
        json.pointer("/tier").and_then(JsonValue::as_str),
        Some("tier-0")
    );
    assert_eq!(
        json.pointer("/receipt/model_calls")
            .and_then(JsonValue::as_u64),
        Some(0),
        "tier-0 record makes zero model calls"
    );
    assert_eq!(
        json.pointer("/receipt/cost_usd")
            .and_then(JsonValue::as_f64),
        Some(0.0)
    );
    assert!(
        json.pointer("/receipt/digests")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0)
            >= 1,
        "record wrote at least one verbatim digest"
    );
    assert!(
        json.pointer("/receipt/pending")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0)
            >= 1,
        "the digest is queued for understanding"
    );
}

#[test]
fn import_estimate_reports_tier0_zero_cost() {
    let harness = CliHarness::new();
    let session = harness.data_root.path().join("session.jsonl");
    write_file(
        &session,
        "{\"message\":{\"role\":\"user\",\"content\":\"adopt kafka for the event bus\"}}\n{\"message\":{\"role\":\"assistant\",\"content\":\"kafka it is, i will wire the producer\"}}\n",
    );
    let output = harness.run(&[
        "import",
        "--estimate",
        "--context",
        "loop",
        session.to_str().unwrap(),
    ]);
    assert_eq!(output.code, 0, "estimate failed: {}", combined(&output));
    let json = output_json(&output);
    assert_eq!(
        json.pointer("/tier").and_then(JsonValue::as_str),
        Some("tier-0")
    );
    assert_eq!(
        json.pointer("/estimate/model_calls")
            .and_then(JsonValue::as_u64),
        Some(0),
        "estimate is zero-outbound"
    );
    assert_eq!(
        json.pointer("/estimate/cost_usd")
            .and_then(JsonValue::as_f64),
        Some(0.0)
    );
    assert!(
        json.pointer("/estimate/chunks")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0)
            >= 1
    );
}

#[test]
fn status_reports_queue_counts_and_raw_footprint() {
    let harness = CliHarness::new();
    let capture = harness.run(&[
        "record",
        "--no-kick",
        "--context",
        "loop",
        "User: migrate the api gateway to oauth2 tokens",
    ]);
    assert_eq!(capture.code, 0, "record failed: {}", combined(&capture));

    let output = harness.run(&["status", "--context", "loop"]);
    assert_eq!(output.code, 0, "status failed: {}", combined(&output));
    let json = output_json(&output);
    assert_eq!(
        json.pointer("/understanding_queue/pending")
            .and_then(JsonValue::as_u64),
        Some(1),
        "status reports the pending understanding backlog"
    );
    assert!(
        json.pointer("/raw_footprint/chunks")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0)
            >= 1,
        "status reports the kept raw footprint"
    );
    assert!(
        json.pointer("/raw_footprint/bytes")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0)
            > 0
    );
}

#[test]
fn forget_purges_raw_and_marks_queue_skipped() {
    let harness = CliHarness::new();
    let capture = harness.run(&[
        "record",
        "--no-kick",
        "--context",
        "loop",
        "User: pick terraform for provisioning the fleet",
    ]);
    assert_eq!(capture.code, 0, "record failed: {}", combined(&capture));

    let forget = harness.run(&["forget", "--context", "loop"]);
    assert_eq!(forget.code, 0, "forget failed: {}", combined(&forget));
    let json = output_json(&forget);
    assert!(
        json.pointer("/forget/chunks_purged")
            .and_then(JsonValue::as_u64)
            .unwrap_or(0)
            >= 1,
        "forget purged the kept raw chunk"
    );

    // The raw footprint is now empty and the queue row is skipped, not pending.
    let status = harness.run(&["status", "--context", "loop"]);
    let status_json = output_json(&status);
    assert_eq!(
        status_json
            .pointer("/raw_footprint/chunks")
            .and_then(JsonValue::as_u64),
        Some(0),
        "raw purge is complete"
    );
    assert_eq!(
        status_json
            .pointer("/understanding_queue/pending")
            .and_then(JsonValue::as_u64),
        Some(0)
    );
    assert_eq!(
        status_json
            .pointer("/understanding_queue/skipped")
            .and_then(JsonValue::as_u64),
        Some(1)
    );

    // Idempotent second purge.
    let again = harness.run(&["forget", "--context", "loop"]);
    let again_json = output_json(&again);
    assert_eq!(
        again_json
            .pointer("/forget/chunks_purged")
            .and_then(JsonValue::as_u64),
        Some(0)
    );
}

#[test]
fn forget_set_retention_days_is_recorded_in_status() {
    let harness = CliHarness::new();
    let set = harness.run(&["forget", "--set-retention-days", "14"]);
    assert_eq!(set.code, 0, "set retention failed: {}", combined(&set));
    assert_eq!(
        output_json(&set)
            .pointer("/raw_retention_days")
            .and_then(JsonValue::as_u64),
        Some(14)
    );

    let status = harness.run(&["status", "--context", "loop"]);
    assert_eq!(
        output_json(&status)
            .pointer("/raw_footprint/retention_days")
            .and_then(JsonValue::as_u64),
        Some(14),
        "status surfaces the configured raw retention window"
    );
}
