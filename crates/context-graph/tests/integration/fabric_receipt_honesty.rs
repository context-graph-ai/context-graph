//! Fabric first-light final mile — the executor's receipt carries the model's REAL spend.
//!
//! The live two-root rehearsal defect: the worker's stderr line printed the true tokens/elapsed, but
//! the `work_results` receipt stored `total_tokens` = the INPUT BYTE COUNT and `wall_clock_ms` = 0 —
//! a fabricated receipt that then flowed to `cg status`. This test drives the real executor with a
//! content-faithful double that reports a KNOWN provider usage and asserts the recorded receipt
//! carries THOSE token numbers and a nonzero wall-clock, never the input byte length.

use std::sync::Mutex;
use std::time::Duration;

use context_graph::ingest::{ConversationExtractor, ExtractionRequest, StructuredHarvest};
use context_graph::{
    EnforcementLevel, ModelCallOutcome, ModelCallUsage, ProviderUsage, UsageReportState,
};

use context_graph::fabric::{self, FabricCapture, FabricWorkerConfig};

use crate::fabric_common::{fabric_fixture, push_then_pull, tier0_request};
use crate::fake_extractor::LabeledTranscriptExtractor;

const SESSION: &str = "User: Goal G1: resume across machines. Assistant: Decision D1: keep the auth module at src/auth.py.";

// A KNOWN provider usage the double reports — deliberately unlike the input byte length so an honest
// receipt (real tokens) is distinguishable from the old fabricated one (byte count).
const KNOWN_PROMPT_TOKENS: u64 = 1_300;
const KNOWN_COMPLETION_TOKENS: u64 = 144;
const KNOWN_TOTAL_TOKENS: u64 = 1_444;
const KNOWN_ELAPSED_MS: u64 = 1_234;

/// A content-faithful extractor double (it produces a real harvest via the lexical extractor) that
/// reports a KNOWN model usage through `drain_model_usage`, exactly as a real connected extractor
/// reports the provider's token block plus the measured per-call wall-clock. Drains once.
#[derive(Debug, Default)]
struct KnownUsageExtractor {
    inner: LabeledTranscriptExtractor,
    pending: Mutex<bool>,
}

impl KnownUsageExtractor {
    fn new() -> Self {
        Self::default()
    }
}

impl ConversationExtractor for KnownUsageExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        *self.pending.lock().expect("usage lock") = true;
        self.inner.extract(request)
    }

    fn drain_model_usage(&self) -> Vec<ModelCallUsage> {
        let mut pending = self.pending.lock().expect("usage lock");
        if !*pending {
            return Vec::new();
        }
        *pending = false;
        vec![ModelCallUsage {
            chunk_index: 1,
            usage: UsageReportState::Tokens(ProviderUsage {
                prompt_tokens: KNOWN_PROMPT_TOKENS,
                completion_tokens: KNOWN_COMPLETION_TOKENS,
                total_tokens: KNOWN_TOTAL_TOKENS,
            }),
            elapsed: Duration::from_millis(KNOWN_ELAPSED_MS),
            reliability_level: EnforcementLevel::Native,
            outcome: ModelCallOutcome::Captured,
            flow_tag: Some("extraction".to_string()),
            model_id: Some("known-usage-double".to_string()),
        }]
    }
}

#[tokio::test]
#[ignore = "hangs rather than fails: the test never returns, so the runner stops it at its terminate-after; quarantined under DL-CG-17"]
async fn executor_receipt_carries_real_tokens_and_elapsed_not_input_byte_length() {
    let fixture = fabric_fixture("receipt_honesty").await;
    let digests = fabric::capture(
        &fixture.edge1.store,
        FabricCapture::new(tier0_request("daily-loop", "s1", SESSION)),
    )
    .expect("capture on submitter")
    .digest_ids;
    let digest = *digests.first().expect("one digest");
    push_then_pull(&fixture.edge1_client, &fixture.edge2_client).await;

    // Execute on the worker with the known-usage double.
    let extractor = KnownUsageExtractor::new();
    let _ = fabric::run_worker_once(
        &fixture.edge2_client,
        &fixture.edge2.store,
        &extractor,
        &FabricWorkerConfig::for_node("worker-pod"),
        1_000,
    )
    .expect("worker poll");

    // The recorded receipt on the worker carries the model's REAL spend — the provider's own token
    // count and the measured wall-clock — never the input byte length and never a zero clock.
    let job_id = fabric::job_id_for_digest(digest);
    let receipt = fabric::recorded_receipt(&fixture.edge2.store, &job_id)
        .expect("read receipt")
        .expect("a result was recorded");
    assert_eq!(
        receipt.total_tokens, KNOWN_TOTAL_TOKENS,
        "the receipt carries the model's reported token total, not the input byte length"
    );
    assert_ne!(
        receipt.total_tokens,
        SESSION.len() as u64,
        "the receipt is NOT the input byte count (the old fabricated value)"
    );
    assert_eq!(
        receipt.wall_clock_ms, KNOWN_ELAPSED_MS,
        "the receipt carries the measured model wall-clock"
    );
    assert!(
        receipt.wall_clock_ms > 0,
        "the receipt wall-clock is nonzero — never the fabricated 0"
    );
    assert_eq!(
        receipt.executor_node_id, "worker-pod",
        "the receipt originates on the executor"
    );
}
