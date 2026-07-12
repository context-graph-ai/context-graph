//! Fabric first-light — duplicate delivery, claim races, and exactly-once guarantees.
//!
//! Result exactly-once lives in `work_results` (worker side); apply exactly-once lives in the
//! retained write-once completion marker (owner side). Keeping them separate is what lets execution
//! and apply live on different machines. The ledger is append-only; cg issues no UPDATE.

use context_graph::fabric::{
    self, ApplyOutcome, FabricCapture, FabricPollOutcome, FabricWorkerConfig,
};

use crate::fabric_common::{
    CountingExtractor, FabricNode, context_id, fabric_fixture, memory_total, push_then_pull,
    tier0_request, work_row_count,
};

const SESSION: &str = "User: Goal G1: resume across machines. Assistant: Decision D1: keep the auth module at src/auth.py.";

#[test]
fn applying_same_result_twice_is_a_noop() {
    let node = FabricNode::open("apply_twice");
    fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture");
    let ctx = context_id(&node.store, "daily-loop").expect("context");

    let extractor = CountingExtractor::new();
    let first = fabric::apply_result(&node.store, "job-1", &extractor).expect("apply once");
    assert!(
        matches!(first, ApplyOutcome::Applied(_)),
        "the first apply writes understanding onto the anchor digest"
    );
    let after_first = memory_total(&node.store, ctx);

    let second = fabric::apply_result(&node.store, "job-1", &extractor).expect("apply again");
    assert_eq!(
        second,
        ApplyOutcome::AlreadyApplied,
        "re-applying the identical result is a no-op"
    );
    assert_eq!(
        memory_total(&node.store, ctx),
        after_first,
        "re-applying adds no memory (write-once completion marker holds)"
    );
}

#[test]
fn recapturing_identical_content_creates_no_duplicate_job() {
    let node = FabricNode::open("recapture_dedup");
    fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("first capture");
    fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("identical recapture");

    assert_eq!(
        work_row_count(&node.store, "work_jobs"),
        1,
        "byte-identical content yields exactly one understanding job, not two"
    );
}

#[tokio::test]
async fn result_arriving_after_local_execution_does_not_duplicate() {
    let node = FabricNode::open("late_result");
    fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture");
    let ctx = context_id(&node.store, "daily-loop").expect("context");

    let extractor = CountingExtractor::new();
    let drained = fabric::drain_local_worker(&node.store, &extractor, "local-node", None)
        .expect("local drain");
    assert_eq!(drained, 1, "the job executes and applies locally");
    let after_local = memory_total(&node.store, ctx);

    // A late remote result for the same job arrives; it must be a no-op.
    let outcome = fabric::apply_result(&node.store, "job-1", &extractor).expect("late apply");
    assert_eq!(
        outcome,
        ApplyOutcome::AlreadyApplied,
        "a late remote result for an already-applied job is a no-op"
    );
    assert_eq!(
        memory_total(&node.store, ctx),
        after_local,
        "no duplicate memory from the late result"
    );
}

#[tokio::test]
async fn two_workers_racing_one_claim_each() {
    let fixture = fabric_fixture("two_workers_race").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on edge1");
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;

    let ext1 = CountingExtractor::new();
    let ext2 = CountingExtractor::new();
    let config1 = FabricWorkerConfig::for_node("edge-1");
    let config2 = FabricWorkerConfig::for_node("edge-2");
    let out1 = fabric::run_worker_once(
        &fixture.edge1_client,
        &fixture.edge1.store,
        &ext1,
        &config1,
        1_000,
    )
    .expect("edge1 poll");
    let out2 = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &ext2,
        &config2,
        1_000,
    )
    .expect("edge2 poll");

    let executed = [&out1, &out2]
        .iter()
        .filter(|o| matches!(o, FabricPollOutcome::Executed { .. }))
        .count();
    assert_eq!(
        executed, 1,
        "exactly one edge wins the claim and executes the job"
    );
    assert_eq!(
        ext1.call_count() + ext2.call_count(),
        1,
        "the model double is invoked exactly once across both racing workers"
    );
}

#[tokio::test]
async fn worker_crash_mid_job_lease_expires_then_second_attempt_succeeds() {
    let fixture = fabric_fixture("crash_lease").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on edge1");
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    // Edge A claims then "dies" without pushing a result.
    let ext_a = CountingExtractor::new();
    let config_a = FabricWorkerConfig::for_node("edge-a");
    let _ = fabric::run_worker_once(
        &fixture.edge1_client,
        &fixture.edge1.store,
        &ext_a,
        &config_a,
        1_000,
    );

    // Advance the injected clock past the lease TTL; edge B claims and completes.
    let after_lease = 1_000 + config_a.lease_ms + 1;
    let ext_b = CountingExtractor::new();
    let config_b = FabricWorkerConfig::for_node("edge-b");
    let out_b = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &ext_b,
        &config_b,
        after_lease,
    )
    .expect("edge B poll after lease expiry");
    assert!(
        matches!(out_b, FabricPollOutcome::Executed { .. }),
        "a second attempt claims and completes after the lease expires"
    );

    push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;
    let ctx = context_id(&fixture.edge1.store, "daily-loop").expect("context");
    let statuses =
        fabric::job_statuses(&fixture.edge1.store, ctx, after_lease + 10).expect("statuses");
    assert!(
        statuses
            .iter()
            .any(|s| s.state == "done" && s.executor_node_id.as_deref() == Some("edge-b")),
        "the job is done exactly once with edge B as executor"
    );
}

#[tokio::test]
async fn offline_drained_job_not_reclaimed_remotely_after_reconnect() {
    let fixture = fabric_fixture("offline_reconnect").await;
    // Edge1 captures and drains its own job while offline (no sync yet).
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on edge1");
    let ext1 = CountingExtractor::new();
    let drained = fabric::drain_local_worker(&fixture.edge1.store, &ext1, "edge-1", None)
        .expect("offline local drain");
    assert_eq!(
        drained, 1,
        "edge1 drains and applies its own job while offline"
    );

    // Reconnect and push; edge2 polls the hub.
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    let ext2 = CountingExtractor::new();
    let config2 = FabricWorkerConfig::for_node("edge-2");
    let out2 = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &ext2,
        &config2,
        5_000,
    )
    .expect("edge2 poll after reconnect");
    assert_eq!(
        out2,
        FabricPollOutcome::NoWork,
        "a completed job is not re-offered after reconnect — exactly-once holds across offline→online"
    );
    assert_eq!(
        ext1.call_count() + ext2.call_count(),
        1,
        "the model double is invoked exactly once across both edges"
    );
}

#[tokio::test]
async fn work_tables_are_append_only_no_update() {
    let fixture = fabric_fixture("append_only").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on edge1");
    assert_eq!(
        work_row_count(&fixture.edge1.store, "work_jobs"),
        1,
        "capture writes exactly one job row"
    );
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    // Two claim attempts append two claim rows — a state change is a new row, never an UPDATE.
    let ext = CountingExtractor::new();
    let config = FabricWorkerConfig::for_node("edge-2");
    let _ = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &ext,
        &config,
        1_000,
    );
    let _ = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &ext,
        &config,
        2_000,
    );
    assert!(
        work_row_count(&fixture.edge2.store, "work_claims") >= 1,
        "each claim attempt appends a new claim row rather than updating a prior one"
    );
}
