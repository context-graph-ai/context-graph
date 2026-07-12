//! Fabric first-light — apply atomicity: the write-once completion marker commits in the SAME
//! transaction as the structured graph it guards, so a crash between the graph write and the marker
//! is impossible. Proven by a fault fired at the marker boundary rolling BOTH back — before the fix
//! the graph committed on its own transaction, so it survived a fault at the marker step.

use context_graph::fabric::{self, ApplyOutcome, FabricCapture};

use crate::fabric_common::{
    ArmedCaptureFault, CountingExtractor, FabricNode, context_id, memory_total, tier0_request,
};

const SESSION: &str = "User: Goal G1: resume across machines. Assistant: Decision D1: keep the auth module at src/auth.py.";

#[test]
fn apply_fault_at_marker_rolls_back_graph_and_marker_together() {
    let node = FabricNode::open("apply_atomicity");
    fabric::capture(
        &node.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture");
    let ctx = context_id(&node.store, "daily-loop").expect("context");
    assert_eq!(
        memory_total(&node.store, ctx),
        0,
        "no structured memory exists before understanding is applied"
    );

    // Apply with a fault armed at the completion-marker boundary: the graph rows are staged, then the
    // fault fires before the marker write and commit.
    let extractor = CountingExtractor::new();
    let fault = ArmedCaptureFault::armed();
    let result = fabric::apply_result_with_fault(&node.store, "job-1", &extractor, &fault);
    assert!(
        result.is_err(),
        "a fault at the completion-marker boundary aborts the apply"
    );

    // Shared fate: the structured graph the apply staged is rolled back WITH the marker. Before the
    // fix the graph committed on its own transaction, so it would have survived this fault and left a
    // graph-without-marker — the exact split this test forbids.
    assert_eq!(
        memory_total(&node.store, ctx),
        0,
        "the structured graph did not commit — it rolled back together with the completion marker"
    );

    // The apply is fully retryable: a clean re-apply (no fault) now writes the graph and marks it
    // structured — the earlier abort left nothing half-applied to trip the write-once guard.
    let clean = fabric::apply_result(&node.store, "job-1", &extractor).expect("clean re-apply");
    assert!(
        matches!(clean, ApplyOutcome::Applied(_)),
        "the retry applies the result onto the anchor digest"
    );
    assert!(
        memory_total(&node.store, ctx) > 0,
        "the retry writes the structured graph the faulted apply rolled back"
    );

    // Idempotent after success: the write-once completion marker (committed with the graph) holds.
    let again =
        fabric::apply_result(&node.store, "job-1", &extractor).expect("re-apply after success");
    assert_eq!(
        again,
        ApplyOutcome::AlreadyApplied,
        "the committed completion marker makes a second apply a no-op"
    );
}
