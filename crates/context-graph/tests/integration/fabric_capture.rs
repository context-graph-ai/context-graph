//! Fabric capture atomically writes the digest, ledger job, and input copies.
//!
//! One $0 capture must commit the verbatim digest, exactly one understanding-job row, and its input
//! copies TOGETHER in one atomic model-free transaction. model-free's shipped output (verbatim digest,
//! local embedding, kept raw) is unchanged. Inputs travel as refs, never inlined.

use context_graph::fabric::{self, FabricCapture, FabricConfig};

use crate::fabric_common::{
    ArmedCaptureFault, CountingExtractor, FabricNode, acquire_worker_lock_at, context_id,
    memory_total, tier0_request, work_row_count,
};

const TRANSCRIPT: &str = "User: Goal G1: ship password-less login. Assistant: Decision D1: keep the auth module at src/auth.py.";

#[test]
fn capture_writes_digest_job_and_inputs_in_one_transaction() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let node = FabricNode::open("cap_one_txn");
    let extractor = CountingExtractor::new();
    let report = fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("capture", "s1", TRANSCRIPT)),
    )
    .expect("fabric capture");

    assert_eq!(
        report.digests_written, 1,
        "one chunk yields one verbatim digest"
    );
    assert_eq!(
        work_row_count(&node.store, "work_jobs"),
        1,
        "capture writes exactly one understanding-job row anchored on that digest"
    );
    assert!(
        work_row_count(&node.store, "work_inputs") >= 1,
        "capture copies the understanding input into the ledger"
    );
    assert_eq!(
        extractor.call_count(),
        0,
        "capture is model-free — the executor runs the model later, elsewhere"
    );
}

#[test]
fn capture_abort_between_digest_and_job_strands_nothing() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let node = FabricNode::open("cap_abort");
    let fault = ArmedCaptureFault::armed();
    let mut capture = FabricCapture::new(tier0_request("capture", "s1", TRANSCRIPT));
    capture.fault = Some(&fault);

    let result = fabric::capture(&node.store, capture);
    assert!(
        result.is_err(),
        "a fault after the digest write must abort the whole capture transaction"
    );

    // All-or-nothing: nothing from the aborted capture survives.
    assert_eq!(
        work_row_count(&node.store, "digests"),
        0,
        "no digest row survives an aborted capture"
    );
    assert_eq!(
        work_row_count(&node.store, "work_jobs"),
        0,
        "no job row survives an aborted capture"
    );
    assert_eq!(
        work_row_count(&node.store, "work_inputs"),
        0,
        "no input row survives an aborted capture"
    );
}

#[test]
fn capture_makes_zero_model_calls_and_costs_zero() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let node = FabricNode::open("cap_zero_cost");
    let config = FabricConfig {
        hub_endpoint: Some("ticket-abc".to_string()),
        tenant: Some("tenant-alpha".to_string()),
        ..Default::default()
    };
    let mut capture = FabricCapture::new(tier0_request("capture", "s1", TRANSCRIPT));
    capture.config = Some(&config);

    let report = fabric::capture(&node.store, capture).expect("fabric capture");
    assert_eq!(
        report.model_calls, 0,
        "capture invokes no model even with a fabric configured"
    );
    assert!(
        report.digests_written >= 1,
        "capture still writes the verbatim digest"
    );
}

#[test]
fn capture_still_produces_verbatim_digest_and_local_embedding() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let node = FabricNode::open("cap_verbatim");
    let report = fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("capture", "s1", TRANSCRIPT)),
    )
    .expect("fabric capture");

    let digest_id = *report.digest_ids.first().expect("one digest written");
    let digest = node
        .store
        .get_digest(digest_id)
        .expect("read digest")
        .expect("digest exists");
    assert_eq!(
        digest.summary,
        context_graph::tier0::salience_clean(TRANSCRIPT),
        "the digest carries the verbatim salience-cleaned text, unchanged by the fabric"
    );
    assert!(
        digest.vector_text.is_some(),
        "capture still writes a local embedding for the digest"
    );
    let ctx = context_id(&node.store, "capture").expect("context exists");
    let footprint = context_graph::tier0::raw_footprint(&node.store, ctx).expect("raw footprint");
    assert!(
        footprint.chunk_count >= 1,
        "capture still keeps the raw chunk"
    );
}

#[tokio::test]
async fn offline_single_machine_drains_its_own_jobs_locally() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let node = FabricNode::open("offline_drain");
    let config = FabricConfig {
        hub_endpoint: Some("unreachable-hub".to_string()),
        tenant: Some("tenant-alpha".to_string()),
        ..Default::default()
    };
    let mut capture = FabricCapture::new(tier0_request("capture", "s1", TRANSCRIPT));
    capture.config = Some(&config);
    fabric::capture(&node.store, capture).expect("fabric capture");

    let extractor = CountingExtractor::new();
    let drained = fabric::drain_local_worker(&node.store, &extractor, "local-node", None)
        .expect("local worker drain");
    assert_eq!(
        drained, 1,
        "a hub-unreachable machine still drains its own job locally"
    );

    let ctx = context_id(&node.store, "capture").expect("context exists");
    assert!(
        memory_total(&node.store, ctx) > 0,
        "local drain produces memory even with the hub unreachable"
    );
    let statuses = fabric::job_statuses(&node.store, ctx, 10_000).expect("job statuses");
    assert!(
        statuses
            .iter()
            .any(|s| s.state == "done" && s.executor_node_id.as_deref() == Some("local-node")),
        "status names the local node as the executor of its own job"
    );
}

#[test]
fn inputs_travel_as_refs_never_inlined() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let node = FabricNode::open("inputs_as_refs");
    fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("capture", "s1", TRANSCRIPT)),
    )
    .expect("fabric capture");

    assert_eq!(
        work_row_count(&node.store, "work_jobs"),
        1,
        "capture writes a job that references its input by id"
    );
    // The job row must reference input by id, never embed the raw conversation bytes.
    let db = node.store.sync_database();
    let job_rows = db.scan("work_jobs", db.snapshot()).expect("scan work_jobs");
    for row in &job_rows {
        let rendered = format!("{:?}", row.values);
        assert!(
            !rendered.contains(TRANSCRIPT),
            "the ledger job row must not embed inline conversation bytes: {rendered}"
        );
    }
}

#[test]
fn worker_lockfile_second_worker_is_noop_when_held() {
    let _alongside = crate::runs_alone::alongside_siblings();
    let node = FabricNode::open("worker_lock");
    let root = node.data_root();
    let first = acquire_worker_lock_at(&root);
    assert!(
        first.is_some(),
        "the first worker acquires the per-data-root lock"
    );
    let second = acquire_worker_lock_at(&root);
    assert!(
        second.is_none(),
        "a second worker is a no-op while the lock is held"
    );
    drop(first);
}
