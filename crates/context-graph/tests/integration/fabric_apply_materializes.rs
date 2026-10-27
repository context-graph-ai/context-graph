//! A pulled worker result materializes memory on the submitter without a model call.
//!
//! After a sync pull, the library sweep commits the worker's structured output as memory.
//! Repeating the sweep leaves the stored result unchanged.

use context_graph::fabric::{
    self, ApplyOutcome, FabricCapture, FabricPollOutcome, FabricWorkerConfig,
};
use std::time::Duration;

use crate::fabric_common::{
    CountingExtractor, FabricFixture, context_id, fabric_fixture, memory_total, tier0_request,
};

const SESSION: &str = "User: Goal G1: resume across machines. Assistant: Decision D1: keep the auth module at src/auth.py.";

/// Push one edge's pending ledger rows to the hub, no pull back — the submitter's own outbound push.
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

/// Pull the hub onto one edge, no worker involvement — the digest owner's own `cg sync pull`.
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

#[tokio::test(flavor = "multi_thread")]
async fn pulled_worker_result_materializes_memory_on_submitter() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture: FabricFixture = fabric_fixture("apply_materializes").await;

    // Submitter (edge1) captures a session and pushes its understanding job to the hub.
    fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("capture", "s1", SESSION)),
    )
    .expect("capture on submitter");
    push_only(&fixture.edge1_client).await;

    // Worker (edge2) runs the self-sufficient poll: it pulls the job, executes it, and pushes the
    // structured result + receipt back. The model runs exactly once, on the worker.
    let worker_ext = CountingExtractor::new();
    let out = fabric::run_worker_poll(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &worker_ext,
        &FabricWorkerConfig::for_node("worker-b"),
        1_000,
    )
    .expect("worker poll");
    assert!(
        matches!(out, FabricPollOutcome::Executed { .. }),
        "the worker executed the pulled job (got {out:?})"
    );
    assert_eq!(
        worker_ext.call_count(),
        1,
        "the model ran once, on the worker"
    );

    // Before the submitter pulls, its graph carries NO structured memory — model-free capture wrote only
    // the verbatim digest; understanding is still remote.
    let ctx = context_id(&fixture.edge1.store, "capture").expect("context on submitter");
    assert_eq!(
        memory_total(&fixture.edge1.store, ctx),
        0,
        "no structured memory on the submitter before it pulls and applies"
    );

    // The submitter does ONLY a `cg sync pull` (no worker, no explicit apply call), then the sweep the
    // CLI's pull path runs. The test never calls `apply_result`.
    pull_only(&fixture.edge1_client).await;
    let applied = fabric::apply_pulled_results(&fixture.edge1.store).expect("apply sweep");
    assert_eq!(
        applied.len(),
        1,
        "the sweep materialized the one arrived worker result"
    );

    // The worker's STRUCTURED OUTPUT is now memory on the submitter — with no model on the submitter.
    assert!(
        memory_total(&fixture.edge1.store, ctx) > 0,
        "the submitter's graph materialized the worker's structured output"
    );
    let after_first = memory_total(&fixture.edge1.store, ctx);

    // A second pull + sweep is a no-op: the write-once completion marker holds, so no duplicate memory.
    pull_only(&fixture.edge1_client).await;
    let again = fabric::apply_pulled_results(&fixture.edge1.store).expect("second sweep");
    assert!(
        again.is_empty(),
        "a second sweep applies nothing — the completion marker holds"
    );
    assert_eq!(
        memory_total(&fixture.edge1.store, ctx),
        after_first,
        "re-sweeping adds no memory"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn apply_commits_the_worker_payload_and_never_re_extracts_on_the_submitter() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let fixture: FabricFixture = fabric_fixture("apply_payload_not_model").await;

    let digests = fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("capture", "s1", SESSION)),
    )
    .expect("capture on submitter")
    .digest_ids;
    let digest = *digests.first().expect("one digest");
    push_only(&fixture.edge1_client).await;

    let worker_ext = CountingExtractor::new();
    fabric::run_worker_poll(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &worker_ext,
        &FabricWorkerConfig::for_node("worker-b"),
        1_000,
    )
    .expect("worker poll");
    pull_only(&fixture.edge1_client).await;

    // Applying a worker's structured result commits that payload without invoking the submitter's
    // model. The spy records any unexpected extraction call.
    let submitter_spy = CountingExtractor::new();
    let job_id = fabric::job_id_for_digest(digest);
    let outcome = fabric::apply_result(&fixture.edge1.store, &job_id, &submitter_spy)
        .expect("apply on submitter");
    assert!(
        matches!(outcome, ApplyOutcome::Applied(_)),
        "the worker's recorded payload applies onto the anchor digest (got {outcome:?})"
    );
    assert_eq!(
        submitter_spy.call_count(),
        0,
        "apply committed the worker's structured output — the submitter's model was never invoked"
    );
    let ctx = context_id(&fixture.edge1.store, "capture").expect("context");
    assert!(
        memory_total(&fixture.edge1.store, ctx) > 0,
        "memory materialized from the worker payload, not from a submitter-side re-extraction"
    );
}
