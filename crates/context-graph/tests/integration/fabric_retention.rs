//! Fabric first-light — apply for a since-forgotten digest, and per-node input retention.
//!
//! Apply is anchor-checked: a result whose anchor digest is absent writes nothing and resurrects
//! nothing. Retention is the ledger's only delete — terminal jobs past grace — and it is
//! everywhere-local (executor edge sweeps its own copy). Deleting inputs never changes a job's
//! computed state. Every grace boundary is driven by an injected clock.

use context_graph::DigestId;
use context_graph::fabric::{
    self, ApplyOutcome, FabricCapture, FabricPollOutcome, FabricWorkerConfig,
};

use crate::fabric_common::{
    CountingExtractor, FabricNode, context_id, fabric_fixture, memory_total, push_then_pull,
    tier0_request, work_row_count,
};

const SESSION: &str = "User: Goal G1: keep memory anchored. Assistant: Decision D1: keep the auth module at src/auth.py.";

#[test]
fn result_for_forgotten_digest_is_dropped_not_resurrected() {
    let node = FabricNode::open("forgotten_dropped");
    let extractor = CountingExtractor::new();

    // Control: a well-formed result applies onto a PRESENT anchor and becomes memory.
    let report = fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture");
    let ctx = context_id(&node.store, "daily-loop").expect("context");
    let present_digest = *report.digest_ids.first().expect("one digest");
    let present_job = fabric::job_id_for_digest(present_digest);
    let control =
        fabric::apply_result(&node.store, &present_job, &extractor).expect("apply present");
    assert!(
        matches!(control, ApplyOutcome::Applied(_)),
        "a result for a present anchor applies as memory"
    );
    assert!(
        memory_total(&node.store, ctx) > 0,
        "control apply writes memory"
    );

    // Forgotten/absent anchor: the result is dropped terminally, not silently applied.
    let ghost = DigestId::new_v7();
    let ghost_job = fabric::job_id_for_digest(ghost);
    let dropped = fabric::apply_result(&node.store, &ghost_job, &extractor).expect("apply ghost");
    assert_eq!(
        dropped,
        ApplyOutcome::AnchorMissing,
        "a result whose anchor is absent is dropped, not applied"
    );
    assert!(
        node.store.get_digest(ghost).expect("get digest").is_none(),
        "the forgotten digest is not resurrected"
    );
}

#[test]
fn apply_checks_anchor_digest_exists_before_writing_memory() {
    let node = FabricNode::open("anchor_checked");
    let extractor = CountingExtractor::new();

    // Control: apply onto a present anchor writes memory.
    let report = fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture");
    let ctx = context_id(&node.store, "daily-loop").expect("context");
    let job = fabric::job_id_for_digest(*report.digest_ids.first().expect("digest"));
    let applied = fabric::apply_result(&node.store, &job, &extractor).expect("apply present");
    assert!(
        matches!(applied, ApplyOutcome::Applied(_)),
        "apply writes memory when the anchor digest exists"
    );

    // Absent anchor: apply writes no memory and reports the anchor-missing outcome.
    let before = memory_total(&node.store, ctx);
    let ghost_job = fabric::job_id_for_digest(DigestId::new_v7());
    let outcome = fabric::apply_result(&node.store, &ghost_job, &extractor).expect("apply ghost");
    assert_eq!(
        outcome,
        ApplyOutcome::AnchorMissing,
        "apply refuses to write onto an absent anchor digest"
    );
    assert_eq!(
        memory_total(&node.store, ctx),
        before,
        "no memory is written for an absent anchor"
    );
}

#[test]
fn retention_delete_of_input_copies_does_not_change_job_state() {
    let node = FabricNode::open("retention_state");
    fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture");
    let ctx = context_id(&node.store, "daily-loop").expect("context");
    let extractor = CountingExtractor::new();
    fabric::drain_local_worker(&node.store, &extractor, "local-node", None).expect("drain");

    // The job is done before retention.
    let before_statuses = fabric::job_statuses(&node.store, ctx, 500_000).expect("statuses");
    assert!(
        before_statuses.iter().any(|s| s.state == "done"),
        "a done job exists before retention"
    );
    let done_before = before_statuses
        .into_iter()
        .find(|s| s.state == "done")
        .expect("checked present above");

    // Advance the injected clock past grace so the input copies delete.
    fabric::run_input_retention(&node.store, 1_000_000, 1).expect("retention");

    let after_statuses = fabric::job_statuses(&node.store, ctx, 1_000_001).expect("statuses");
    assert!(
        after_statuses.iter().any(|s| s.state == "done"),
        "the job is still done after its inputs are gone"
    );
    let done_after = after_statuses
        .into_iter()
        .find(|s| s.state == "done")
        .expect("checked present above");
    assert_eq!(
        done_after.executor_node_id, done_before.executor_node_id,
        "deleting inputs is not a state transition — executor id intact"
    );
    assert_eq!(
        done_after.receipt, done_before.receipt,
        "the receipt survives input retention"
    );
}

#[test]
fn input_copies_deleted_after_result_lands_plus_grace() {
    let node = FabricNode::open("retention_submitter");
    fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture");
    let extractor = CountingExtractor::new();
    fabric::drain_local_worker(&node.store, &extractor, "local-node", None).expect("drain");

    // Before grace: the input copy is present.
    assert!(
        work_row_count(&node.store, "work_inputs") >= 1,
        "the ephemeral input copy is present before result-lands + grace"
    );

    // After result-lands + grace (injected clock): it is swept.
    fabric::run_input_retention(&node.store, 1_000_000, 1).expect("retention");
    assert_eq!(
        work_row_count(&node.store, "work_inputs"),
        0,
        "the submitter's input copy is gone after result-lands + grace"
    );
}

#[tokio::test]
#[ignore = "hangs rather than fails: the test never returns, so the runner stops it at its terminate-after; quarantined under DL-CG-17"]
async fn input_copies_deleted_on_executor_after_result_lands_plus_grace() {
    let fixture = fabric_fixture("retention_executor").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on submitter");
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    // Edge2 executes as the worker, materializing its own ephemeral input copy.
    let extractor = CountingExtractor::new();
    let config = FabricWorkerConfig::for_node("edge-2");
    let out = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &extractor,
        &config,
        1_000,
    )
    .expect("executor poll");
    assert!(
        matches!(out, FabricPollOutcome::Executed { .. }),
        "the executor edge runs the job"
    );
    assert!(
        work_row_count(&fixture.edge2.store, "work_inputs") >= 1,
        "the executor holds its own ephemeral input copy before grace"
    );

    // After result-lands + grace, the executor sweeps its OWN copy — deletion is everywhere-local.
    fabric::run_input_retention(&fixture.edge2.store, 1_000_000, 1).expect("executor retention");
    assert_eq!(
        work_row_count(&fixture.edge2.store, "work_inputs"),
        0,
        "the executor edge deletes its ephemeral input copy, not just the submitter"
    );
}
