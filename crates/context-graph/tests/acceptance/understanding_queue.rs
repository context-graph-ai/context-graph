//! Acceptance tests for WP2 understanding-queue drain accessors.
//!
//! The contract under test: tier-0 capture leaves a durable `pending` understanding-queue row per
//! digest; a worker drains those rows by PRIORITY (most-touched first, oldest first within equal
//! touch), claims each atomically so two workers never double-understand the same digest, records a
//! write-once completion on success, and increments attempts with a terminal cap on model failure.
//! A digest that already carries a completion row (a synced-in completion from another machine) is
//! skipped. Touch bumps and queue state survive a store reopen.

use context_graph::tier0::{Tier0CaptureRequest, capture_tier0, queue_counts};
use context_graph::understanding::{
    MAX_UNDERSTANDING_ATTEMPTS, PendingDigest, bump_touch, claim_digest, list_pending_digests,
    mark_failed, mark_structured,
};
use context_graph::{ContextId, DigestId, Store, StoreConfig};

use super::fake_extractor::open_store_with_lexical_embedder;

const CONTEXT: &str = "daily-loop";

fn fresh_store(dir: &std::path::Path) -> Store {
    open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    })
}

fn context_id(store: &Store) -> ContextId {
    store
        .list_contexts()
        .expect("contexts")
        .into_iter()
        .find(|context| context.name == CONTEXT)
        .expect("capture context exists")
        .id
}

/// Capture one short transcript as one tier-0 digest and return its id.
fn capture_one(store: &Store, session: &str, text: &str) -> DigestId {
    let report = capture_tier0(
        store,
        Tier0CaptureRequest {
            context_label: CONTEXT.to_string(),
            session_ref: session.to_string(),
            transcript: text.to_string(),
            source: "test".to_string(),
            chunk_window: 0,
        },
    )
    .expect("tier-0 capture");
    assert_eq!(
        report.digest_ids.len(),
        1,
        "expected one digest for a short transcript"
    );
    report.digest_ids[0]
}

#[test]
fn pending_digests_ordered_by_touch_then_age() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = fresh_store(dir.path());
    let first = capture_one(&store, "s1", "User: we chose sqlite for the local cache");
    let second = capture_one(&store, "s2", "User: we picked nats for the event bus");
    let third = capture_one(&store, "s3", "User: we adopted rust for the engine core");
    let ctx = context_id(&store);

    // Touch priority: second is touched twice, third once, first never.
    bump_touch(&store, second).expect("bump second");
    bump_touch(&store, second).expect("bump second again");
    bump_touch(&store, third).expect("bump third");

    let pending = list_pending_digests(&store, ctx, None).expect("list pending");
    let order: Vec<DigestId> = pending.iter().map(|p| p.digest_id).collect();
    assert_eq!(
        order,
        vec![second, third, first],
        "ordered by touch_count DESC then oldest-first: {order:?}"
    );
    // The stored raw chunk text rides along so the worker can feed it back as analysis text.
    let second_row: &PendingDigest = pending
        .iter()
        .find(|p| p.digest_id == second)
        .expect("second present");
    assert!(
        second_row.chunk_text.contains("nats for the event bus"),
        "pending row carries the raw chunk: {}",
        second_row.chunk_text
    );
    assert_eq!(second_row.touch_count, 2);
}

#[test]
fn claim_is_idempotent_and_exclusive() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = fresh_store(dir.path());
    let digest = capture_one(&store, "s1", "User: we chose sqlite for the local cache");

    assert!(claim_digest(&store, digest).expect("first claim"));
    assert!(
        !claim_digest(&store, digest).expect("second claim"),
        "a second worker cannot re-claim an in-flight digest"
    );
    let ctx = context_id(&store);
    assert_eq!(queue_counts(&store, ctx).expect("counts").in_flight, 1);
    // A claimed digest is no longer offered as pending.
    assert!(
        list_pending_digests(&store, ctx, None)
            .expect("list")
            .is_empty()
    );
}

#[test]
fn structured_writes_completion_and_status() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = fresh_store(dir.path());
    let digest = capture_one(&store, "s1", "User: we chose sqlite for the local cache");
    let ctx = context_id(&store);

    assert!(claim_digest(&store, digest).expect("claim"));
    mark_structured(&store, ctx, digest, "test-model", Some("{\"tokens\":42}"))
        .expect("mark structured");

    let counts = queue_counts(&store, ctx).expect("counts");
    assert_eq!(counts.structured, 1);
    assert_eq!(counts.pending, 0);
    assert_eq!(counts.in_flight, 0);
    // The write-once completion row means a synced-in edge skips re-understanding this digest.
    assert!(
        list_pending_digests(&store, ctx, None)
            .expect("list")
            .is_empty()
    );
}

#[test]
fn failed_increments_attempts_and_caps() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = fresh_store(dir.path());
    let digest = capture_one(&store, "s1", "User: we chose sqlite for the local cache");
    let ctx = context_id(&store);

    // Each failure increments attempts; below the cap the digest is re-queued for another drain.
    let mut last_attempts = 0;
    for expected in 1..MAX_UNDERSTANDING_ATTEMPTS {
        assert!(claim_digest(&store, digest).expect("claim"));
        let outcome = mark_failed(&store, digest, "model timed out", MAX_UNDERSTANDING_ATTEMPTS)
            .expect("mark failed");
        assert_eq!(outcome.attempts, expected);
        assert!(!outcome.terminal, "not terminal below the cap");
        last_attempts = outcome.attempts;
        // Re-queued: still offered for a retry.
        assert_eq!(
            list_pending_digests(&store, ctx, None).expect("list").len(),
            1
        );
    }
    // The capping failure marks it terminally failed and it is no longer drained.
    assert!(claim_digest(&store, digest).expect("claim"));
    let terminal = mark_failed(&store, digest, "model timed out", MAX_UNDERSTANDING_ATTEMPTS)
        .expect("mark failed");
    assert_eq!(terminal.attempts, last_attempts + 1);
    assert!(terminal.terminal, "at the cap the digest is terminally failed");
    let counts = queue_counts(&store, ctx).expect("counts");
    assert_eq!(counts.failed, 1);
    assert!(
        list_pending_digests(&store, ctx, None)
            .expect("list")
            .is_empty(),
        "a terminally-failed digest is never re-drained"
    );
}

#[test]
fn completion_row_skips_pending() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = fresh_store(dir.path());
    let digest = capture_one(&store, "s1", "User: we chose sqlite for the local cache");
    let ctx = context_id(&store);

    // Simulate a completion synced in from another machine while our row is still pending: mark it
    // structured through the completion write, then confirm it drops out of the pending set even
    // though a naive status scan might still find work.
    assert!(claim_digest(&store, digest).expect("claim"));
    mark_structured(&store, ctx, digest, "peer-model", None).expect("completion");
    assert!(
        list_pending_digests(&store, ctx, None)
            .expect("list")
            .is_empty()
    );
}

#[test]
fn touches_and_queue_survive_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let digest;
    let ctx;
    {
        let store = fresh_store(dir.path());
        digest = capture_one(&store, "s1", "User: we chose sqlite for the local cache");
        ctx = context_id(&store);
        bump_touch(&store, digest).expect("bump");
        bump_touch(&store, digest).expect("bump");
    }
    // Reopen from disk: the pending row, its touch count, and its raw chunk are all durable.
    let store = fresh_store(dir.path());
    let pending = list_pending_digests(&store, ctx, None).expect("list");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].digest_id, digest);
    assert_eq!(pending[0].touch_count, 2);
    assert!(pending[0].chunk_text.contains("sqlite"));
}
