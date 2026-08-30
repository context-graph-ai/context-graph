//! The understanding-queue counts describe the queue — not every digest the context holds.
//!
//! Public contract: `cg status`'s five queue counts answer "what understanding work is there?",
//! and the work is what capture put on the shipped work ledger. A digest carrying NO ledger job is
//! not work: it is not pending, not in flight, not structured, not failed, not skipped. Digests
//! reach a store this way in normal running — the `digests` table syncs between machines while the
//! work ledger and the kept raw do not — so a machine that syncs a peer's memory acquires digests
//! it never queued, and a bare digest count would inflate its backlog forever.
//!
//! The counts and the drain door must also agree. `list_pending_digests` is what a worker is
//! actually offered, and it refuses a digest with no job. An operator who reads "6 pending" and
//! runs the worker must get 6 digests to understand, not 6 minus however many arrived by sync;
//! a pending total the drain cannot deliver is a status that lies about the backlog.
//!
//! Exclusion is a property of the QUEUE, not of the digest: the same job-less digests still carry
//! kept raw bytes, and the raw-source footprint still counts them. A fix that made these digests
//! invisible everywhere would break that, so both totals are asserted over the same store.

#[path = "common/large_status_fixture.rs"]
mod large_status_fixture;

use std::collections::HashMap;

use context_graph::ingest::split_transcript_into_windows;
use context_graph::schema::DIGEST_VECTOR_DIMENSION;
use context_graph::tier0::{
    QueueCounts, RawFootprint, Tier0CaptureRequest, capture_tier0, queue_counts, raw_footprint,
};
use context_graph::understanding::{claim_digest, list_pending_digests, mark_structured};
use context_graph::{ContextId, DigestId, Store};
use contextdb_core::Value as DbValue;
use contextdb_engine::work_ledger::job_snapshot;

use large_status_fixture::open_store;

/// Bytes each captured window covers.
const CHUNK_WINDOW: usize = 40;

/// Digests captured through the production door — every one of them queued.
const CAPTURED_DIGESTS: usize = 6;

/// Raw bytes each directly-seeded digest keeps. Distinct, so a byte total over them cannot be a
/// row count times a constant.
const SEEDED_CHUNK_BYTES: [usize; 2] = [37, 61];

/// A store holding both kinds of digest: the ones capture queued, and the ones that are only
/// content.
struct MixedStore {
    context_id: ContextId,
    /// The digests capture queued, in capture order.
    queued: Vec<DigestId>,
    /// Raw bytes capture kept for `queued`.
    queued_bytes: u64,
    /// A digest with kept raw and no ledger job at all.
    bare: DigestId,
    /// A digest with kept raw, no ledger job, and a write-once understanding completion.
    completed: DigestId,
}

/// One capture window's worth of text: exactly [`CHUNK_WINDOW`] bytes, each block naming its own
/// unit so nothing dedups as already-captured content.
fn transcript() -> String {
    let mut text = String::with_capacity(CAPTURED_DIGESTS * CHUNK_WINDOW);
    for unit in 0..CAPTURED_DIGESTS {
        let mut block = format!("User: queued note {unit} ");
        while block.len() < CHUNK_WINDOW {
            block.push('x');
        }
        assert_eq!(
            block.len(),
            CHUNK_WINDOW,
            "a fixture block must fill one capture window exactly"
        );
        text.push_str(&block);
    }
    text
}

/// The work-ledger job id tier-0 capture submits for a digest.
fn understanding_job_id(digest: DigestId) -> String {
    format!("cg.understanding.{}", digest.as_uuid())
}

/// Capture [`CAPTURED_DIGESTS`] digests through the production door, then put two digests in the
/// same context that capture never queued.
fn fill(store: &Store) -> MixedStore {
    let text = transcript();
    let windows = split_transcript_into_windows(&text, CHUNK_WINDOW);
    assert_eq!(
        windows.len(),
        CAPTURED_DIGESTS,
        "the fixture transcript must cut into one window per intended digest"
    );

    let report = capture_tier0(
        store,
        Tier0CaptureRequest {
            context_label: "loop".to_string(),
            session_ref: "membership-fixture".to_string(),
            transcript: text,
            source: "test".to_string(),
            chunk_window: CHUNK_WINDOW,
        },
    )
    .expect("tier-0 capture of the queued digests");
    assert_eq!(
        (
            report.chunks_kept,
            report.skipped_existing,
            report.skipped_noise
        ),
        (CAPTURED_DIGESTS, 0, 0),
        "capture must keep every fixture window and skip none"
    );
    let queued = report.digest_ids;
    assert_eq!(
        queued.len(),
        CAPTURED_DIGESTS,
        "tier-0 writes one digest per kept chunk"
    );

    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|context| context.name == "loop")
        .expect("the capture context exists")
        .id;

    // The naming has to be right before an absent job proves anything: a digest capture DID queue
    // is found under exactly this id.
    let db = store.sync_database();
    assert!(
        job_snapshot(&db, &understanding_job_id(queued[0]))
            .expect("read a ledger job")
            .is_some(),
        "the fixture must name the understanding job capture actually submitted"
    );

    // Whatever kind capture stamps on the digest it writes is the kind the queue's universe is
    // scoped to, so the seeded digests are stamped with it rather than with a guess.
    let kind = captured_kind(store, queued[0]);

    let bare =
        seed_content_only_digest(store, context_id, &kind, "peer-bare", SEEDED_CHUNK_BYTES[0]);
    let completed = seed_content_only_digest(
        store,
        context_id,
        &kind,
        "peer-understood",
        SEEDED_CHUNK_BYTES[1],
    );
    mark_structured(store, context_id, completed, "peer-model", None)
        .expect("record a write-once understanding completion for a digest with no job");

    for digest in [bare, completed] {
        assert!(
            job_snapshot(&db, &understanding_job_id(digest))
                .expect("read a ledger job")
                .is_none(),
            "a seeded digest must carry no work-ledger job at all"
        );
    }
    assert!(
        completion_exists(store, completed),
        "the second seeded digest must carry the completion this proof is about"
    );
    assert!(
        !completion_exists(store, bare),
        "the first seeded digest must be bare"
    );

    MixedStore {
        context_id,
        queued,
        queued_bytes: u64::try_from(report.raw_bytes).expect("a capture byte total fits"),
        bare,
        completed,
    }
}

/// The `kind` capture stamped on a digest it wrote.
fn captured_kind(store: &Store, digest: DigestId) -> String {
    let rows = store
        .sync_database()
        .execute(
            "SELECT kind FROM digests WHERE id = $id",
            &HashMap::from([("id".to_string(), DbValue::Uuid(digest.as_uuid()))]),
        )
        .expect("read a captured digest's kind")
        .rows;
    match rows.first().and_then(|row| row.first()) {
        Some(DbValue::Text(kind)) => kind.clone(),
        other => panic!("a captured digest carries a text kind, not {other:?}"),
    }
}

/// Write a digest and its kept raw chunk straight into the store, with no work-ledger job — the
/// shape a digest arrives in when it syncs from a peer that queued it and this machine did not.
fn seed_content_only_digest(
    store: &Store,
    context_id: ContextId,
    kind: &str,
    label: &str,
    chunk_bytes: usize,
) -> DigestId {
    let digest = DigestId::new_v7();
    let content = "z".repeat(chunk_bytes);
    let db = store.sync_database();
    let tx = db.begin().expect("open a transaction for a seeded digest");
    db.execute_in_tx(
        tx,
        "INSERT INTO digests (id, source, session_ref, summary, kind, vector_text, dimension, context_id, properties, created_at) VALUES ($id, $source, $session_ref, $summary, $kind, $vector_text, $dimension, $context_id, $properties, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(digest.as_uuid())),
            ("source".to_string(), DbValue::Text("test".to_string())),
            (
                "session_ref".to_string(),
                DbValue::Text(label.to_string()),
            ),
            ("summary".to_string(), DbValue::Text(content.clone())),
            ("kind".to_string(), DbValue::Text(kind.to_string())),
            (
                "vector_text".to_string(),
                DbValue::Vector(vec![0.0; DIGEST_VECTOR_DIMENSION]),
            ),
            (
                "dimension".to_string(),
                DbValue::Int64(DIGEST_VECTOR_DIMENSION as i64),
            ),
            (
                "context_id".to_string(),
                DbValue::Uuid(context_id.as_uuid()),
            ),
            (
                "properties".to_string(),
                DbValue::Json(serde_json::json!({})),
            ),
            ("created_at".to_string(), DbValue::TxId(tx)),
        ]),
    )
    .expect("seed a digest with no ledger job");
    db.execute_in_tx(
        tx,
        "INSERT INTO cg_source_chunks (id, digest_id, context_id, seq, content, content_hash, byte_len, created_at, created_at_ms) VALUES ($id, $digest_id, $context_id, $seq, $content, $content_hash, $byte_len, $created_at, $created_at_ms)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(digest.to_string())),
            ("digest_id".to_string(), DbValue::Text(digest.to_string())),
            (
                "context_id".to_string(),
                DbValue::Uuid(context_id.as_uuid()),
            ),
            ("seq".to_string(), DbValue::Int64(0)),
            ("content".to_string(), DbValue::Text(content)),
            (
                "content_hash".to_string(),
                DbValue::Text(format!("seeded-{label}")),
            ),
            (
                "byte_len".to_string(),
                DbValue::Int64(i64::try_from(chunk_bytes).expect("a chunk length fits")),
            ),
            ("created_at".to_string(), DbValue::TxId(tx)),
            ("created_at_ms".to_string(), DbValue::Int64(0)),
        ]),
    )
    .expect("seed the kept raw of a digest with no ledger job");
    db.commit(tx).expect("commit a seeded digest");
    digest
}

/// Whether a digest carries a write-once understanding completion.
fn completion_exists(store: &Store, digest: DigestId) -> bool {
    !store
        .sync_database()
        .execute(
            "SELECT id FROM cg_understanding_completions WHERE id = $id",
            &HashMap::from([("id".to_string(), DbValue::Text(digest.to_string()))]),
        )
        .expect("read the understanding completions")
        .rows
        .is_empty()
}

/// Every job the counts placed in a bucket.
fn counted(counts: &QueueCounts) -> usize {
    counts.pending + counts.in_flight + counts.structured + counts.failed + counts.skipped
}

/// The counts agree with the drain door about what is pending, and neither seeded digest is in any
/// bucket or on any worker's offer.
fn assert_queue_holds_only_capture_work(store: &Store, fixture: &MixedStore, counts: &QueueCounts) {
    assert_eq!(
        counted(counts),
        CAPTURED_DIGESTS,
        "the queue counts must total the digests capture queued ({CAPTURED_DIGESTS}), not every \
         digest the context holds: {counts:?}"
    );
    let offered = list_pending_digests(store, fixture.context_id, None)
        .expect("the drain door's own answer about what is pending");
    assert_eq!(
        offered.len(),
        counts.pending,
        "a worker must be offered exactly the digests status calls pending"
    );
    for seeded in [fixture.bare, fixture.completed] {
        assert!(
            !offered.iter().any(|pending| pending.digest_id == seeded),
            "the drain door must not offer a digest with no ledger job"
        );
    }
}

#[test]
fn a_digest_with_no_ledger_job_is_in_no_queue_bucket() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store(&dir.path().join("cg.db"));
    let fixture = fill(&store);

    let counts = queue_counts(&store, fixture.context_id).expect("queue counts");
    assert_eq!(
        counts,
        QueueCounts {
            pending: CAPTURED_DIGESTS,
            in_flight: 0,
            structured: 0,
            failed: 0,
            skipped: 0,
        },
        "a digest with no ledger job is not queue work — neither the bare one (which must not read \
         as pending) nor the completed one (which must not read as structured)"
    );
    assert_queue_holds_only_capture_work(&store, &fixture, &counts);

    // The same digests are still content: their kept raw is part of the context's footprint.
    let seeded_bytes: u64 = SEEDED_CHUNK_BYTES
        .iter()
        .map(|bytes| u64::try_from(*bytes).expect("a chunk length fits"))
        .sum();
    assert_eq!(
        raw_footprint(&store, fixture.context_id).expect("raw footprint"),
        RawFootprint {
            chunk_count: CAPTURED_DIGESTS + SEEDED_CHUNK_BYTES.len(),
            bytes: fixture.queued_bytes + seeded_bytes,
        },
        "a digest that is out of the queue still keeps raw bytes, and the footprint must count them"
    );
}

#[test]
fn queued_digests_keep_moving_between_buckets_while_job_less_digests_stay_out() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store(&dir.path().join("cg.db"));
    let fixture = fill(&store);

    assert!(
        claim_digest(&store, fixture.queued[0]).expect("claim a pending digest"),
        "a digest capture queued is claimable"
    );
    let after_claim = queue_counts(&store, fixture.context_id).expect("queue counts after a claim");
    assert_eq!(
        after_claim,
        QueueCounts {
            pending: CAPTURED_DIGESTS - 1,
            in_flight: 1,
            structured: 0,
            failed: 0,
            skipped: 0,
        },
        "a claim moves one queued digest from pending to in flight and adds nothing"
    );
    assert_queue_holds_only_capture_work(&store, &fixture, &after_claim);

    mark_structured(
        &store,
        fixture.context_id,
        fixture.queued[1],
        "test-model",
        None,
    )
    .expect("record a write-once understanding completion for a queued digest");
    let after_completion =
        queue_counts(&store, fixture.context_id).expect("queue counts after a completion");
    assert_eq!(
        after_completion,
        QueueCounts {
            pending: CAPTURED_DIGESTS - 2,
            in_flight: 1,
            structured: 1,
            failed: 0,
            skipped: 0,
        },
        "a completion on a QUEUED digest is the one that counts as structured — the identical \
         completion on a job-less digest is not"
    );
    assert_queue_holds_only_capture_work(&store, &fixture, &after_completion);
}
