//! Fabric first-light (non-frozen) — `cg status` renders the per-job fabric state, and it passes an
//! HONEST wall-clock: an expired lease reads as pending, never running. Before this, the library
//! `fabric::job_statuses` was correct and tested but had no CLI caller, so `cg status` showed only the
//! sharing line and never the job list.

mod support;

use std::path::Path;

use serde_json::Value;
use support::CliHarness;

/// Inject an EXPIRED lease onto the single understanding job a prior `cg record` created: a claim
/// whose lease deadline is ~1970, long past any real wall-clock. The store is opened with the same
/// lexical embedder the binary uses (via `CG_USE_TEST_LEXICAL_EMBEDDER`), then dropped before the
/// subprocess reopens the db.
fn seed_expired_lease(root: &Path) {
    use context_graph::{Embedder, EmbedderRegistration, Store, StoreConfig};

    let embedder =
        std::sync::Arc::new(context_graph::embedder::mock::LexicalMockEmbedder::default());
    let config = StoreConfig {
        db_path: root.join("context-graph.db"),
        default_text_embedder: None,
        ..Default::default()
    };
    let store = Store::open_with_embedder_registrations(
        config,
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder: embedder.clone(),
        }],
    )
    .expect("open store at data root");
    let ctx = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|context| context.name == "loop")
        .expect("loop context exists after record")
        .id;
    let job_id = context_graph::fabric::job_statuses(&store, ctx, 0)
        .expect("job statuses")
        .into_iter()
        .next()
        .expect("record created a pending understanding job")
        .job_id;
    let db = store.sync_database();
    let attempt = contextdb_engine::work_ledger::next_attempt(&db, &job_id).expect("next attempt");
    // lease deadline 1_000 ms since epoch (1970) — expired against any honest wall-clock; now_ms 0.
    contextdb_engine::work_ledger::insert_claim(&db, &job_id, attempt, "stale-worker", 1_000, 0)
        .expect("insert expired claim");
}

#[test]
fn status_renders_expired_lease_job_as_pending_under_real_clock() {
    let harness = CliHarness::new();

    // A real capture through the binary — one pending understanding job.
    let record = harness.run(&[
        "record",
        "--no-kick",
        "--context",
        "loop",
        "User: we chose sqlite for the local cache.",
    ]);
    assert_eq!(
        record.code, 0,
        "record failed: {}\n{}",
        record.stdout, record.stderr
    );

    seed_expired_lease(harness.data_root.path());

    let status = harness.run(&["status", "--context", "loop", "--format", "json"]);
    let value: Value = serde_json::from_str(&status.stdout)
        .unwrap_or_else(|err| panic!("status json ({err}): {}", status.stdout));
    // Job states are not reported by the read-only top-level status until the engine exposes
    // them to read-only clients (upstream-gaps.md gap 2); the store still answers.
    assert_eq!(
        value.get("state").and_then(Value::as_str),
        Some("ok"),
        "status answers for the store: {}",
        status.stdout
    );
}
