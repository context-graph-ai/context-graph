//! CLI acceptance tests for WP1 tier-0 deterministic capture.
//!
//! `cg record` / `cg import` are the deterministic, free front door: zero model calls even when a
//! model is connected, an honest zero-cost estimate, a purge command (`cg forget`), and `cg status`
//! surfacing the understanding-queue backlog and raw footprint. Tests pass `--no-kick` so the
//! detached background worker is not spawned during the test run.

mod support;

use serde_json::Value as JsonValue;
use support::{CliHarness, CommandOutput};

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
