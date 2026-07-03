//! The bench's capture driver and store read-back (criterion C7). The driver goes through the
//! PRODUCTION library entry (`run_capture` — the same terminal path the understanding drain
//! calls via `run_capture_into_digest`); the gap-report harness's side execution path is the
//! documented false-green history this replaces. Real full-bench runs drive the actual `cg`
//! binaries; this library driver is the deterministic-double seam for CI.

use context_graph::Store;
use context_graph::harness::{CaptureHarnessConfig, CaptureHarnessReport, run_capture};
use context_graph::ingest::{CaptureReport, CaptureRequest, ConversationExtractor};

use crate::scoring::ExtractedSession;

/// Drive one transcript through the production capture path (the caller owns the store: real
/// embedder for measurement runs, deterministic mock in CI). Returns the per-chunk reports.
pub fn run_transcript_through_capture(
    _store: &Store,
    _context_label: &str,
    _transcript: &str,
    _extractor: &dyn ConversationExtractor,
    _config: &CaptureHarnessConfig,
) -> Result<Vec<(CaptureReport, CaptureHarnessReport)>, String> {
    todo!("cg-correctness C7: bench capture driver not implemented yet")
}

/// Read a context's extraction back from the store as the session-level shape the frozen
/// scoring consumes: active intentions, decisions with their SERVES attachment resolved to the
/// serving intention's text, entities with types, observations.
pub fn collect_extracted_session(
    _store: &Store,
    _context_label: &str,
) -> Result<ExtractedSession, String> {
    todo!("cg-correctness C7: bench store read-back not implemented yet")
}
