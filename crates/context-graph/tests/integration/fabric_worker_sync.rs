//! Fabric first-light — the remote-worker self-sufficiency regression, pinning the live two-root
//! defect: a real remote worker must PULL the tenant's ledger before it polls (so it sees a job
//! another node submitted) and PUSH its result afterwards (so the outcome reaches the submitter
//! without waiting for the next claim's piggyback).
//!
//! Every other fabric worker test pre-pulls onto the worker edge in the fixture (`push_then_pull`
//! before invoking the worker), which masked BOTH halves of this defect: the inner
//! `run_worker_once` scans only rows already in its own local db, so a worker that never pulled saw
//! no job (real rehearsal: a worker polled for minutes while the job stayed `pending`), and it never
//! pushed its recorded result, so the submitter never saw `done`. These tests invoke ONLY the
//! self-sufficient `run_worker_poll` on the worker edge, with NO fixture pre-pull on that edge, and
//! assert the round-trip closes.

use context_graph::fabric::{
    self, FabricCapture, FabricPollOutcome, FabricWorkerConfig, run_worker_poll,
};
use std::time::Duration;

use crate::fabric_common::{
    FabricFixture, FabricNode, context_id, fabric_fixture, tier0_request,
    unavailable_authenticated_client, work_row_count,
};

const SESSION: &str = "User: Goal G1: resume across machines. Assistant: Decision D1: keep the auth module at src/auth.py.";

/// Push one edge's pending ledger rows to the hub WITHOUT pulling anything back onto it — the
/// submitter's own outbound push. Deliberately not `push_then_pull`: the whole point of these tests
/// is that the WORKER edge does no fixture pull of its own.
async fn push_only(client: &contextdb_server::SyncClient) {
    for _ in 0..32 {
        if !client
            .has_pending_push_changes()
            .expect("pending push check")
        {
            break;
        }
        tokio::time::timeout(Duration::from_secs(30), client.push())
            .await
            .expect("authenticated push stays bounded")
            .expect("push round");
    }
    assert!(
        !client
            .has_pending_push_changes()
            .expect("pending push check"),
        "push did not drain within 32 authenticated rounds"
    );
}

/// Pull the hub onto one edge WITHOUT any worker involvement — the digest owner's own `cg sync pull`.
async fn pull_only(client: &contextdb_server::SyncClient) {
    for _ in 0..32 {
        let before = client.pull_watermark();
        tokio::time::timeout(Duration::from_secs(30), client.pull_default())
            .await
            .expect("authenticated pull stays bounded")
            .expect("pull round");
        if client.pull_watermark() == before {
            return;
        }
    }
    panic!("pull did not converge within 32 authenticated rounds");
}

#[tokio::test]
async fn remote_worker_pulls_the_job_then_pushes_the_result_without_a_fixture_prepull() {
    let fixture: FabricFixture = fabric_fixture("remote_worker_self_sufficient").await;

    // Edge1 is the SUBMITTER + digest owner: capture the session, then push its job to the hub. No
    // pull onto edge2 (the worker) — that is exactly what the frozen fixtures did and what masked the
    // defect. Edge2 must discover the job on its OWN pull inside run_worker_poll.
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on edge1");
    push_only(&fixture.edge1_client).await;

    // The worker edge has never seen this job: nothing has been pulled onto it. The inner
    // `run_worker_once` alone would scan an empty local ledger and report NoWork — the masked defect.
    assert_eq!(
        work_row_count(&fixture.edge2.store, "work_jobs"),
        0,
        "the worker edge has NOT pre-pulled the job — this is the unmasked starting state"
    );

    // Invoke ONLY the self-sufficient poll on the worker edge. It must pull the job from the hub,
    // claim it, execute it, record the result, and push that result back — all in this one call.
    let ext = crate::fabric_common::CountingExtractor::new();
    let config = FabricWorkerConfig::for_node("worker-edge");
    let outcome = run_worker_poll(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &ext,
        &config,
        1_000,
    )
    .expect("worker poll");
    assert!(
        matches!(outcome, FabricPollOutcome::Executed { .. }),
        "the remote worker pulled the peer's job and executed it (got {outcome:?})"
    );
    assert_eq!(
        ext.call_count(),
        1,
        "the model ran exactly once on the worker edge"
    );

    // The digest owner (edge1) now pulls — and ONLY pulls. The worker did NOT `cg sync push`
    // afterwards; the result must already be on the hub because run_worker_poll pushed it. If the
    // result-push were fixture-masked, edge1 would see no `work_results` row here.
    pull_only(&fixture.edge1_client).await;

    let ctx = context_id(&fixture.edge1.store, "daily-loop").expect("context on submitter");
    let statuses = fabric::job_statuses(&fixture.edge1.store, ctx, 2_000).expect("statuses");
    assert!(
        statuses
            .iter()
            .any(|s| s.state == "done" && s.executor_node_id.as_deref() == Some("worker-edge")),
        "the submitter sees the job DONE with the remote worker as executor, purely from its own \
         pull — the worker's result reached the hub without any explicit worker-side push \
         (statuses: {statuses:?})"
    );
    assert!(
        work_row_count(&fixture.edge1.store, "work_results") >= 1,
        "the executor's result row reached the submitter's ledger"
    );
}

#[tokio::test]
async fn worker_poll_with_unreachable_hub_degrades_quietly_offline() {
    // A genuinely hub-less persisted-identity node uses a valid authenticated ticket whose endpoint
    // has gone away. The self-sufficient poll must treat both round trips as best-effort and still
    // execute the locally present job through the offline claim convention.
    let node = FabricNode::open("offline_worker");
    fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on the offline node");

    let client =
        unavailable_authenticated_client(node.store.sync_database(), "tenant-alpha", &node.db_path)
            .await;

    let ext = crate::fabric_common::CountingExtractor::new();
    let config = FabricWorkerConfig::for_node("solo-worker");
    let outcome = run_worker_poll(&client, &node.store, &ext, &config, 1_000)
        .expect("worker poll must not error when the hub is unreachable — it degrades to offline");
    assert!(
        matches!(outcome, FabricPollOutcome::Executed { .. }),
        "with the hub down, the worker still claims + executes its own locally-present job through \
         the offline single-node convention (got {outcome:?})"
    );
    assert_eq!(ext.call_count(), 1, "the model ran exactly once");
    assert!(
        work_row_count(&node.store, "work_results") >= 1,
        "the result is recorded locally even though it could not be pushed to the down hub"
    );
}
