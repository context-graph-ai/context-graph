//! Connected-by-default capture context on the shared extraction path.
//!
//! Capturing with only a tenant supplied (no explicit context) from two different
//! working directories must land both captures in the SAME tenant context — the
//! working directory plays no role. This drives the capture path and reads the
//! persisted rows back at the store boundary, so a plain in-memory map could not
//! satisfy it: the context_id is a property of the committed graph.

use context_graph::harness::{CaptureHarnessConfig, default_capture_context, run_capture};
use context_graph::ingest::CaptureRequest;
use context_graph::{ContextId, Store, StoreConfig};
use contextdb_core::Value as DbValue;

use super::action_test_extractors::SpanFaithfulExtractor;
use super::fake_extractor::open_store_with_lexical_embedder;

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn capture_under_tenant(
    store: &Store,
    tenant_id: &str,
    cwd: &str,
    transcript: &str,
    stored_decision: &str,
) -> ContextId {
    let label = default_capture_context(tenant_id, cwd);
    let request = CaptureRequest {
        context_label: label.clone(),
        transcript: transcript.to_string(),
        model_available: true,
    };
    run_capture(
        store,
        request,
        &SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("capture succeeds");
    persisted_decision_context(store, stored_decision)
}

/// Read the committed graph back and return the context_id the decision landed in.
fn persisted_decision_context(store: &Store, decision_text: &str) -> ContextId {
    let db = store.sync_database();
    let rows = db.scan("decisions", db.snapshot()).expect("scan decisions");
    rows.iter()
        .find_map(
            |row| match (row.values.get("description"), row.values.get("context_id")) {
                (Some(DbValue::Text(description)), Some(DbValue::Uuid(context_id)))
                    if description == decision_text =>
                {
                    Some(ContextId::from(*context_id))
                }
                _ => None,
            },
        )
        .unwrap_or_else(|| panic!("decision `{decision_text}` must persist with a context_id"))
}

#[test]
fn capture_context_defaults_to_tenant_not_working_directory() {
    let (_dir, store) = fresh_store();

    let cwd_a = "/home/dev/project-a";
    let cwd_b = "/srv/builds/project-b";

    let decision_a = "share one tenant memory from the first directory";
    let decision_b = "reuse the same tenant context from the second directory";
    let context_from_a = capture_under_tenant(
        &store,
        "owner-tenant",
        cwd_a,
        &format!("Goal G1: converge by tenant\nDecision D1: {decision_a} serves G1"),
        decision_a,
    );
    let context_from_b = capture_under_tenant(
        &store,
        "owner-tenant",
        cwd_b,
        &format!("Goal G1: converge by tenant again\nDecision D2: {decision_b} serves G1"),
        decision_b,
    );

    assert_eq!(
        context_from_a, context_from_b,
        "same tenant captured from two working directories must converge on one context"
    );
}
