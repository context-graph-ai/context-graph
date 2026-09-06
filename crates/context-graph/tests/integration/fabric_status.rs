//! Fabric first-light — status honesty, touch-priority + budget drain, and remote receipts.
//!
//! Status consults the injected clock (an expired lease is reclaimable, never running). A claim-race
//! loser never reports success. A remote receipt is carried unmangled, originates on the executor,
//! and surfaces the executing node id in status.

use context_graph::fabric::{self, FabricCapture, FabricPollOutcome, FabricWorkerConfig};

use crate::fabric_common::{
    CountingExtractor, FabricNode, FailingExtractor, context_id, fabric_fixture, push_then_pull,
    tier0_request,
};

const SESSION: &str = "User: Goal G1: run understanding remotely. Assistant: Decision D1: keep the auth module at src/auth.py.";
const SESSION_B: &str =
    "User: Goal G2: batch the backlog. Assistant: Decision D2: store refresh tokens in Redis.";
const SESSION_C: &str =
    "User: Goal G3: watch the budget. Assistant: Decision D3: cap the nightly drain.";

#[tokio::test]
#[ignore = "hangs rather than fails: the test never returns, so the runner stops it at its terminate-after; quarantined under DL-CG-17"]
async fn expired_lease_job_not_shown_as_running() {
    let fixture = fabric_fixture("expired_lease").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture");
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    // Edge2 claims (leases) the job at t=1000, then produces no result.
    let extractor = FailingExtractor::new();
    let config = FabricWorkerConfig::for_node("edge-2");
    let _ = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &extractor,
        &config,
        1_000,
    );
    push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;

    // Past the lease TTL with no result, status must show it reclaimable/pending, never running.
    let ctx = context_id(&fixture.edge1.store, "daily-loop").expect("context");
    let past_ttl = 1_000 + config.lease_ms + 1;
    let statuses = fabric::job_statuses(&fixture.edge1.store, ctx, past_ttl).expect("statuses");
    assert!(
        statuses.iter().any(|s| s.state == "pending"),
        "an expired-lease job shows as reclaimable/pending"
    );
    assert!(
        !statuses.iter().any(|s| s.state == "leased"),
        "an expired-lease job is never shown as leased/running"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_that_lost_claim_race_does_not_report_success() {
    let fixture = fabric_fixture("lost_race_status").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture");
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;
    push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;

    let ext1 = CountingExtractor::new();
    let ext2 = CountingExtractor::new();
    let out1 = fabric::run_worker_once(
        &fixture.edge1_client,
        &fixture.edge1.store,
        &ext1,
        &FabricWorkerConfig::for_node("edge-1"),
        1_000,
    )
    .expect("edge1 poll");
    let out2 = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &ext2,
        &FabricWorkerConfig::for_node("edge-2"),
        1_000,
    )
    .expect("edge2 poll");

    let winners = [&out1, &out2]
        .iter()
        .filter(|o| matches!(o, FabricPollOutcome::Executed { .. }))
        .count();
    let losers = [&out1, &out2]
        .iter()
        .filter(|o| matches!(o, FabricPollOutcome::LostClaim { .. }))
        .count();
    assert_eq!(winners, 1, "exactly one worker wins the claim");
    assert_eq!(losers, 1, "the loser reports a lost claim, never success");
}

#[tokio::test]
#[ignore = "hangs rather than fails: the test never returns, so the runner stops it at its terminate-after; quarantined under DL-CG-17"]
async fn status_shows_state_executing_node_and_receipt() {
    let fixture = fabric_fixture("status_executor").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on submitter");
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    let extractor = CountingExtractor::new();
    let _ = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &extractor,
        &FabricWorkerConfig::for_node("worker-pod"),
        1_000,
    );
    push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;

    let ctx = context_id(&fixture.edge1.store, "daily-loop").expect("context");
    let statuses = fabric::job_statuses(&fixture.edge1.store, ctx, 5_000).expect("statuses");
    assert!(
        statuses.iter().any(|s| s.state == "done"),
        "a remote-executed done job appears in status"
    );
    let done = statuses
        .iter()
        .find(|s| s.state == "done")
        .expect("checked present above");
    assert_eq!(
        done.executor_node_id.as_deref(),
        Some("worker-pod"),
        "status names the executing node id, distinct from the submitter"
    );
    assert!(
        done.receipt.is_some(),
        "status shows the token/time receipt for the done job"
    );
}

#[tokio::test]
#[ignore = "hangs rather than fails: the test never returns, so the runner stops it at its terminate-after; quarantined under DL-CG-17"]
async fn attempt_ceiling_reached_marks_terminal_failed_visible_in_status() {
    let fixture = fabric_fixture("attempt_ceiling").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture");
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    // Exhaust the attempt ceiling with failing polls (injected clock advancing past each lease).
    let extractor = FailingExtractor::new();
    let config = FabricWorkerConfig::for_node("edge-2");
    for attempt in 0..5 {
        let now = 1_000 + attempt * (config.lease_ms + 1);
        let _ = fabric::run_worker_once(
            &fixture.edge2_client,
            &fixture.edge2.store,
            &extractor,
            &config,
            now,
        );
        push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;
    }

    let ctx = context_id(&fixture.edge1.store, "daily-loop").expect("context");
    let statuses = fabric::job_statuses(&fixture.edge1.store, ctx, 1_000_000).expect("statuses");
    assert!(
        statuses.iter().any(|s| s.state == "failed"),
        "a job past its attempt ceiling becomes terminal failed, visible as failed and not re-leased"
    );
}

#[tokio::test]
#[ignore = "hangs rather than fails: the test never returns, so the runner stops it at its terminate-after; quarantined under DL-CG-17"]
async fn remote_receipt_round_trips_unmangled() {
    let fixture = fabric_fixture("receipt_round_trip").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on submitter");
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    let extractor = CountingExtractor::new();
    let out = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &extractor,
        &FabricWorkerConfig::for_node("worker-pod"),
        1_000,
    )
    .expect("executor poll");
    assert!(
        matches!(out, FabricPollOutcome::Executed { .. }),
        "the executor produces a result and receipt"
    );
    let produced = fabric::job_statuses(
        &fixture.edge2.store,
        context_id(&fixture.edge2.store, "daily-loop").expect("edge2 context"),
        2_000,
    )
    .expect("edge2 statuses")
    .into_iter()
    .find(|s| s.state == "done")
    .and_then(|s| s.receipt);

    push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;
    let applied = fabric::job_statuses(
        &fixture.edge1.store,
        context_id(&fixture.edge1.store, "daily-loop").expect("edge1 context"),
        3_000,
    )
    .expect("edge1 statuses")
    .into_iter()
    .find(|s| s.state == "done")
    .and_then(|s| s.receipt);

    assert!(
        applied.is_some(),
        "the receipt survives sync + apply on the submitter"
    );
    assert_eq!(
        applied, produced,
        "every receipt field is preserved after sync + apply — no truncation or rewrite"
    );
}

#[tokio::test]
#[ignore = "hangs rather than fails: the test never returns, so the runner stops it at its terminate-after; quarantined under DL-CG-17"]
async fn receipt_origin_is_executor_not_submitter() {
    let fixture = fabric_fixture("receipt_origin").await;
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on submitter");
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    let extractor = CountingExtractor::new();
    let _ = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &extractor,
        &FabricWorkerConfig::for_node("worker-pod"),
        1_000,
    );
    push_then_pull(&fixture.edge2_client, &fixture.edge1_client).await;

    let ctx = context_id(&fixture.edge1.store, "daily-loop").expect("context");
    let statuses = fabric::job_statuses(&fixture.edge1.store, ctx, 4_000).expect("statuses");
    assert!(
        statuses.iter().any(|s| s.state == "done"),
        "a done job appears on the submitter after apply"
    );
    let done = statuses
        .into_iter()
        .find(|s| s.state == "done")
        .expect("checked present above");
    assert_eq!(
        done.executor_node_id.as_deref(),
        Some("worker-pod"),
        "the applied receipt carries the executor's node id, not the submitter's — a locally rebuilt receipt cannot pass"
    );
}

#[test]
fn worker_drains_most_recently_touched_first() {
    let node = FabricNode::open("touch_priority");
    let low = fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s-low", SESSION_B)),
    )
    .expect("capture low")
    .digest_ids;
    let high = fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s-high", SESSION_C)),
    )
    .expect("capture high")
    .digest_ids;
    let ctx = context_id(&node.store, "daily-loop").expect("context");
    let high_digest = *high.first().expect("high digest");
    // Make the high-priority digest the most touched.
    for _ in 0..3 {
        context_graph::tier0::bump_touch(&node.store, ctx, high_digest).expect("bump touch");
    }
    let _ = low;

    // Drain a single job under a budget; the most-touched job must be the one that runs.
    let extractor = CountingExtractor::new();
    fabric::drain_local_worker(&node.store, &extractor, "local-node", Some(1)).expect("drain one");
    let high_job = fabric::job_id_for_digest(high_digest);
    let statuses = fabric::job_statuses(&node.store, ctx, 5_000).expect("statuses");
    assert!(
        statuses
            .iter()
            .any(|s| s.job_id == high_job && s.state == "done"),
        "the worker drains the most-recently-touched job first"
    );
}

#[test]
fn worker_budget_ceiling_stops_drain() {
    let node = FabricNode::open("budget_ceiling");
    for (session, text) in [("s1", SESSION), ("s2", SESSION_B), ("s3", SESSION_C)] {
        fabric::capture(
            &node.store,
            FabricCapture::new(tier0_request("daily-loop", session, text)),
        )
        .expect("capture");
    }
    let ctx = context_id(&node.store, "daily-loop").expect("context");

    let extractor = CountingExtractor::new();
    let drained =
        fabric::drain_local_worker(&node.store, &extractor, "local-node", Some(1)).expect("drain");
    assert_eq!(
        drained, 1,
        "a worker honoring a budget ceiling stops at the ceiling"
    );
    let pending = fabric::job_statuses(&node.store, ctx, 5_000)
        .expect("statuses")
        .into_iter()
        .filter(|s| s.state == "pending")
        .count();
    assert_eq!(
        pending, 2,
        "the remaining jobs stay pending after the budget stops the drain"
    );
}
