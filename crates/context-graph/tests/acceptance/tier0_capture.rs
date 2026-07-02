//! Acceptance tests for WP1 tier-0 deterministic capture + retention.
//!
//! The contract under test: `cg record` / `cg import` capture is deterministic and free — a
//! verbatim digest + local embedding + kept raw chunk + durable understanding-queue row per chunk,
//! all in ONE atomic transaction, ZERO model calls, understanding deferred. Kept raw is purgeable
//! (`cg forget`) and bounded by a retention window, and raw never syncs.

use std::path::PathBuf;
use std::sync::Arc;

use context_graph::ListEntityFilter;
use context_graph::capture_sync_status::context_graph_sync_table_contract;
use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::ingest::memory_counts;
use context_graph::tier0::{
    self, ForgetSelector, Tier0CaptureRequest, capture_tier0, forget_raw, queue_counts,
    raw_footprint, salience_clean, set_raw_retention_days,
};
use context_graph::{
    CgError, ContextId, Embedder, EmbedderRegistration, EmbeddingInput, EmbeddingOutput, Result,
    Store, StoreConfig,
};
use contextdb_engine::sync_types::{ConflictPolicy, SyncDirection};

use super::fake_extractor::open_store_with_lexical_embedder;

const CONTEXT: &str = "daily-loop";

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn context_id(store: &Store) -> ContextId {
    store
        .list_contexts()
        .expect("contexts")
        .into_iter()
        .find(|ctx| ctx.name == CONTEXT)
        .expect("capture context exists")
        .id
}

fn request(transcript: &str, chunk_window: usize) -> Tier0CaptureRequest {
    Tier0CaptureRequest {
        context_label: CONTEXT.to_string(),
        session_ref: "session-A".to_string(),
        transcript: transcript.to_string(),
        source: "cg-record".to_string(),
        chunk_window,
    }
}

// A test-support embedder that behaves like the lexical mock but FAILS on any text containing a
// sentinel, so a capture can be forced to fail on a specific chunk mid-transaction. Deterministic
// double, test support only — never in src/.
#[derive(Debug)]
struct SentinelFailEmbedder {
    inner: LexicalMockEmbedder,
    sentinel: &'static str,
}

impl Embedder for SentinelFailEmbedder {
    fn embed(&self, input: EmbeddingInput) -> Result<EmbeddingOutput> {
        if let EmbeddingInput::Text(text) = &input
            && text.contains(self.sentinel)
        {
            return Err(CgError::Engine("induced embed failure".to_string()));
        }
        self.inner.embed(input)
    }

    fn supported_inputs(&self) -> Vec<context_graph::EvidenceKind> {
        self.inner.supported_inputs()
    }

    fn embedding_space(&self) -> context_graph::EmbeddingSpace {
        self.inner.embedding_space()
    }
}

fn store_with_failing_embedder(db_path: PathBuf, sentinel: &'static str) -> Store {
    let embedder = Arc::new(SentinelFailEmbedder {
        inner: LexicalMockEmbedder::default(),
        sentinel,
    });
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path,
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("open store with sentinel-fail embedder")
}

#[test]
fn tier0_capture_writes_digest_chunk_queue_atomically() {
    let (_dir, store) = fresh_store();
    let report = capture_tier0(
        &store,
        request("User: we chose postgres for the primary datastore", 0),
    )
    .expect("capture");
    let ctx = context_id(&store);

    assert_eq!(report.digests_written, 1, "one verbatim digest written");
    assert_eq!(report.chunks_kept, 1, "one raw chunk kept");
    assert_eq!(report.model_calls, 0, "tier-0 makes zero model calls");
    assert_eq!(report.digest_ids.len(), 1);

    let digest = store
        .get_digest(report.digest_ids[0])
        .expect("get digest")
        .expect("digest persisted");
    assert!(digest.vector_text.is_some(), "digest is embedded");

    let footprint = raw_footprint(&store, ctx).expect("footprint");
    assert_eq!(footprint.chunk_count, 1, "raw chunk persisted");
    assert!(footprint.bytes > 0, "raw bytes recorded");

    let counts = queue_counts(&store, ctx).expect("queue");
    assert_eq!(counts.pending, 1, "one pending understanding-queue row");
}

#[test]
fn tier0_capture_is_atomic_on_induced_failure() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = store_with_failing_embedder(dir.path().join("cg.db"), "TRIPWIRE");
    // First chunk embeds cleanly and writes rows inside the open transaction; the second chunk
    // carries the tripwire and fails the embed — the whole capture must roll back.
    let transcript = "User: we chose postgres for storage now\nAssistant: TRIPWIRE follows here";
    let result = capture_tier0(&store, request(transcript, 40));
    assert!(
        result.is_err(),
        "capture must fail on the induced embed error"
    );

    let ctx = context_id(&store);
    assert_eq!(
        raw_footprint(&store, ctx).expect("footprint").chunk_count,
        0,
        "no raw chunk survives a partial capture"
    );
    assert_eq!(
        queue_counts(&store, ctx).expect("queue").pending,
        0,
        "no queue row survives a partial capture"
    );
    assert!(
        store
            .find_via_digests("postgres storage", Default::default())
            .expect("search")
            .is_empty(),
        "no digest survives a partial capture"
    );
}

#[test]
fn tier0_capture_makes_zero_model_calls() {
    let (_dir, store) = fresh_store();
    // capture_tier0 takes no extractor and never invokes a model; the receipt reports it honestly.
    let report = capture_tier0(
        &store,
        request("User: ship password-less login\nAssistant: acknowledged", 0),
    )
    .expect("capture");
    assert_eq!(report.model_calls, 0);
    assert!(report.digests_written >= 1);
}

#[test]
fn tier0_default_writes_no_graph_rows_only_digest_chunk_queue() {
    let (_dir, store) = fresh_store();
    let report = capture_tier0(
        &store,
        request(
            "User: adopt postgres for the primary datastore\nAssistant: agreed, postgres it is",
            0,
        ),
    )
    .expect("capture");
    let ctx = context_id(&store);

    // The default tier-0 path writes ONLY a digest + raw chunk + queue row — never structured graph.
    assert!(report.digests_written >= 1, "a verbatim digest is written");
    assert_eq!(
        raw_footprint(&store, ctx).expect("footprint").chunk_count,
        report.chunks_kept,
        "the raw chunk is kept"
    );
    assert_eq!(
        queue_counts(&store, ctx).expect("queue").pending,
        report.digests_written,
        "each digest is queued for understanding"
    );

    let counts = memory_counts(&store, ctx).expect("counts");
    assert_eq!(
        (counts.goals, counts.decisions, counts.observations),
        (0, 0, 0),
        "tier-0 default extracts NO intentions/decisions/observations — understanding is deferred"
    );
    assert!(
        store
            .list_entities(ListEntityFilter {
                context_id: Some(ctx),
                ..Default::default()
            })
            .expect("entities")
            .is_empty(),
        "tier-0 default extracts no entities"
    );
}

#[test]
fn tier0_import_second_run_writes_nothing_new() {
    let (_dir, store) = fresh_store();
    let text = "User: adopt redis for the cache tier\nAssistant: sounds good, redis it is";
    let first = capture_tier0(&store, request(text, 0)).expect("first capture");
    assert_eq!(first.digests_written, 1);
    assert_eq!(first.skipped_existing, 0);

    let ctx = context_id(&store);
    let footprint_after_first = raw_footprint(&store, ctx).expect("footprint");

    let second = capture_tier0(&store, request(text, 0)).expect("second capture");
    assert_eq!(second.digests_written, 0, "re-import writes no new digest");
    assert_eq!(
        second.skipped_existing, 1,
        "the chunk is recognized as already captured"
    );

    assert_eq!(
        raw_footprint(&store, ctx).expect("footprint").chunk_count,
        footprint_after_first.chunk_count,
        "re-import adds no raw chunks"
    );
    assert_eq!(
        queue_counts(&store, ctx).expect("queue").pending,
        1,
        "re-import adds no queue rows"
    );
}

#[test]
fn tier0_digest_immediately_queryable_marked_verbatim() {
    let (_dir, store) = fresh_store();
    let report = capture_tier0(
        &store,
        request("User: migrate auth to oauth2 for the api gateway", 0),
    )
    .expect("capture");
    let digest_id = report.digest_ids[0];

    assert_eq!(
        tier0::digest_kind(&store, digest_id)
            .expect("kind")
            .as_deref(),
        Some("verbatim"),
        "tier-0 digests are marked verbatim"
    );

    let hits = store
        .find_via_digests("oauth2 api gateway", Default::default())
        .expect("search");
    assert!(
        hits.iter().any(|hit| hit.digest_id == digest_id),
        "the verbatim digest is immediately queryable"
    );
}

#[test]
fn forget_purges_raw_chunks_completely_and_idempotently() {
    let (_dir, store) = fresh_store();
    let report = capture_tier0(
        &store,
        request("User: pick kafka for the event bus backbone", 0),
    )
    .expect("capture");
    let ctx = context_id(&store);
    let digest_id = report.digest_ids[0];

    let purge = forget_raw(&store, Some(ctx), ForgetSelector::Context).expect("forget");
    assert_eq!(purge.chunks_purged, 1, "the kept raw chunk is purged");
    assert!(purge.bytes_purged > 0);
    assert_eq!(
        purge.queue_rows_skipped, 1,
        "the pending queue row is marked skipped"
    );

    assert_eq!(
        raw_footprint(&store, ctx).expect("footprint").chunk_count,
        0,
        "raw purge is complete on this machine"
    );
    let counts = queue_counts(&store, ctx).expect("queue");
    assert_eq!(counts.pending, 0);
    assert_eq!(counts.skipped, 1);

    // The digest / graph row remains — it is the synced, durable memory.
    assert!(
        store.get_digest(digest_id).expect("get digest").is_some(),
        "forget purges raw, not the synced digest"
    );

    // Idempotent: a second identical purge removes nothing.
    let again = forget_raw(&store, Some(ctx), ForgetSelector::Context).expect("forget again");
    assert_eq!(again.chunks_purged, 0);
    assert_eq!(again.queue_rows_skipped, 0);
}

#[test]
fn retention_expired_raw_removed() {
    let (_dir, store) = fresh_store();
    capture_tier0(
        &store,
        request("User: choose terraform for infra provisioning", 0),
    )
    .expect("capture");
    let ctx = context_id(&store);
    set_raw_retention_days(&store, 7).expect("set retention");

    let now_ms = chrono::Utc::now().timestamp_millis();

    // Not yet expired: within the window, nothing is purged.
    let fresh = forget_raw(
        &store,
        Some(ctx),
        ForgetSelector::Expired {
            retention_days: tier0::raw_retention_days(&store).expect("retention"),
            now_ms,
        },
    )
    .expect("forget fresh");
    assert_eq!(fresh.chunks_purged, 0, "raw within the window is kept");

    // Past the window: the raw chunk is removed.
    let later_ms = now_ms + 8 * 86_400_000;
    let expired = forget_raw(
        &store,
        Some(ctx),
        ForgetSelector::Expired {
            retention_days: tier0::raw_retention_days(&store).expect("retention"),
            now_ms: later_ms,
        },
    )
    .expect("forget expired");
    assert_eq!(expired.chunks_purged, 1, "raw past the window is purged");
    assert_eq!(
        raw_footprint(&store, ctx).expect("footprint").chunk_count,
        0
    );
}

#[test]
fn retention_unset_never_expires_raw() {
    let (_dir, store) = fresh_store();
    capture_tier0(&store, request("User: keep this raw forever please", 0)).expect("capture");
    let ctx = context_id(&store);
    // No retention set -> raw_retention_days is None -> nothing expires even far in the future.
    assert_eq!(tier0::raw_retention_days(&store).expect("retention"), None);
    let report = forget_raw(
        &store,
        Some(ctx),
        ForgetSelector::Expired {
            retention_days: None,
            now_ms: chrono::Utc::now().timestamp_millis() + 3650 * 86_400_000,
        },
    )
    .expect("forget");
    assert_eq!(report.chunks_purged, 0, "unset retention keeps raw forever");
}

#[test]
fn queue_rows_survive_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("cg.db");
    let digest_id;
    {
        let store = open_store_with_lexical_embedder(StoreConfig {
            db_path: db_path.clone(),
            default_text_embedder: None,
            ..Default::default()
        });
        let report = capture_tier0(
            &store,
            request("User: durable queue must survive a reopen", 0),
        )
        .expect("capture");
        digest_id = report.digest_ids[0];
    }
    // Reopen the same database file with a fresh handle.
    let reopened = open_store_with_lexical_embedder(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    });
    let ctx = context_id(&reopened);
    assert_eq!(
        queue_counts(&reopened, ctx).expect("queue").pending,
        1,
        "the durable queue row survives a reopen"
    );
    assert!(
        reopened
            .get_digest(digest_id)
            .expect("get digest")
            .is_some(),
        "the verbatim digest survives a reopen"
    );
}

#[test]
fn sync_contract_registers_new_tables() {
    let contract = context_graph_sync_table_contract();
    let policy = |table: &str| -> Option<(SyncDirection, ConflictPolicy)> {
        contract
            .iter()
            .find(|policy| policy.table == table)
            .map(|policy| (policy.direction, policy.conflict_policy))
    };
    // Raw and per-device runtime state never leave the machine.
    for table in [
        "cg_source_chunks",
        "cg_understanding_queue",
        "cg_retrieval_touches",
        "cg_local_settings",
    ] {
        assert_eq!(
            policy(table),
            Some((SyncDirection::None, ConflictPolicy::InsertIfNotExists)),
            "{table} must be local-only (SyncDirection::None)"
        );
    }
    // The write-once understanding-completion marker syncs so a peer skips re-understanding.
    assert_eq!(
        policy("cg_understanding_completions"),
        Some((SyncDirection::Both, ConflictPolicy::InsertIfNotExists)),
        "cg_understanding_completions must sync write-once (Both, InsertIfNotExists)"
    );
}

#[test]
fn salience_clean_strips_tool_noise_keeps_dialogue() {
    let raw = concat!(
        "User: please run the build\n",
        "<tool_use name=\"bash\">cargo build --release; secret internal path /srv/x</tool_use>\n",
        "<tool_result>compiled 412 crates in 88s, warnings elided</tool_result>\n",
        "<system-reminder>remember to be concise</system-reminder>\n",
        "Assistant: the build passed, shipping now\n",
    );
    let cleaned = salience_clean(raw);
    assert!(
        cleaned.contains("please run the build"),
        "keeps user dialogue"
    );
    assert!(
        cleaned.contains("the build passed, shipping now"),
        "keeps assistant dialogue"
    );
    assert!(
        !cleaned.contains("cargo build --release"),
        "strips tool_use payload"
    );
    assert!(
        !cleaned.contains("/srv/x"),
        "strips tool_use payload contents"
    );
    assert!(
        !cleaned.contains("compiled 412 crates"),
        "strips tool_result payload"
    );
    assert!(
        !cleaned.contains("remember to be concise"),
        "strips system-reminder noise"
    );

    // Already-clean prose is unchanged (idempotent salience cut).
    let plain = "User: what is the plan\nAssistant: ship tier-0 capture";
    assert_eq!(salience_clean(plain), plain);
}
