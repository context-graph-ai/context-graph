//! A store whose one context holds more raw source-chunk rows than an uncapped read may examine.
//!
//! Shared by the library proof (`context-graph/tests/status_totals_are_bounded_on_a_large_store.rs`)
//! and the command-line proof (`cg-cli/tests/status_is_bounded.rs`) so both assert against the SAME
//! arithmetic. Everything is written through the production model-free capture door; the expected
//! totals are computed from what was handed to it, using the same public windowing function capture
//! itself uses, and cross-checked against capture's own receipts.

#![allow(dead_code)]

use std::path::Path;

use context_graph::embedder::mock::LexicalMockEmbedder;
use context_graph::ingest::split_transcript_into_windows;
use context_graph::tier0::{ForgetSelector, Tier0CaptureRequest, capture_tier0, forget_raw};
use context_graph::understanding::{
    MAX_UNDERSTANDING_ATTEMPTS, claim_digest, mark_failed, mark_structured,
};
use context_graph::{ContextId, DigestId, Embedder, EmbedderRegistration, Store, StoreConfig};
use contextdb_engine::Database;
use contextdb_engine::work_ledger::{
    ClaimInsert, JobSpec, cancel_job, insert_claim, job_snapshot, record_failure, record_result,
    submit_job,
};

/// The capture context every fixture row lands in.
pub const CONTEXT: &str = "loop";

/// Bytes each captured digest covers. Small, so a modest transcript makes a large table.
pub const CHUNK_WINDOW: usize = 40;

/// More rows than the shipped `work` ceiling (50,000) admits, so an uncapped scan of this table
/// cannot answer at all and a correct total can only have come from the bounded door.
pub const TARGET_CHUNKS: usize = 62_000;

/// Chunks per capture transaction. One transaction for all 62,000 would be a fixture that proves
/// nothing about status; capture is called the way a real recorder calls it, many times.
pub const CHUNKS_PER_BATCH: usize = 2_000;

/// What was actually written, and therefore exactly what `cg status` must report back.
pub struct LargeStore {
    pub context_id: ContextId,
    /// Raw source-chunk rows kept in `context_id`.
    pub chunk_count: usize,
    /// The sum of every kept chunk's `byte_len`.
    pub raw_bytes: u64,
    /// Understanding jobs written, one per digest — all pending until a state is forced on them.
    pub digest_ids: Vec<DigestId>,
}

/// Open a store at `db_path` with the same lexical embedder the `cg` binary uses under
/// `CG_USE_TEST_LEXICAL_EMBEDDER`, so the command-line proof can reopen what this wrote.
pub fn open_store(db_path: &Path) -> Store {
    let embedder = std::sync::Arc::new(LexicalMockEmbedder::default());
    Store::open_with_embedder_registrations(
        StoreConfig {
            db_path: db_path.to_path_buf(),
            default_text_embedder: None,
            ..Default::default()
        },
        vec![EmbedderRegistration {
            space: embedder.embedding_space(),
            embedder,
        }],
    )
    .expect("open the large-status fixture store")
}

/// One capture window's worth of text: exactly [`CHUNK_WINDOW`] bytes, so the splitter's
/// char-boundary cut lands on it whole.
///
/// Each block carries its own unit number, so no two windows are ever the same content and nothing
/// dedups as an already-captured digest. Each ends in a varying run of spaces, which `trim` takes
/// off, so the kept chunks differ in `byte_len` and a correct byte total cannot be a row count
/// times a constant.
fn block(unit: usize) -> String {
    let trailing = 1 + unit % 7;
    let mut text = format!("User: note-{unit} ");
    assert!(
        text.len() + trailing <= CHUNK_WINDOW,
        "a fixture block's identifying head must fit inside one capture window"
    );
    while text.len() + trailing < CHUNK_WINDOW {
        text.push('x');
    }
    for _ in 0..(CHUNK_WINDOW - text.len()) {
        text.push(' ');
    }
    text
}

/// `units` consecutive blocks, starting at `first_unit`.
fn transcript(first_unit: usize, units: usize) -> String {
    let mut text = String::with_capacity(units * CHUNK_WINDOW);
    for unit in first_unit..first_unit + units {
        text.push_str(&block(unit));
    }
    text
}

/// What capturing one context put in the store.
pub struct CapturedContext {
    pub context_id: ContextId,
    /// Raw source-chunk rows kept in this context.
    pub chunk_count: usize,
    /// The sum of every kept chunk's `byte_len`.
    pub raw_bytes: u64,
    /// One understanding job per kept chunk, all pending until a state is forced on them.
    pub digest_ids: Vec<DigestId>,
}

/// Capture into `label` until it holds at least `target_chunks` raw chunks, starting the block
/// numbering at `first_unit` so two contexts never generate the same content and nothing dedups.
fn capture_into(
    store: &Store,
    label: &str,
    first_unit: usize,
    target_chunks: usize,
) -> CapturedContext {
    let mut chunk_count = 0usize;
    let mut raw_bytes = 0u64;
    let mut digest_ids: Vec<DigestId> = Vec::new();
    let mut next_unit = first_unit;
    let mut batch = 0usize;

    while chunk_count < target_chunks {
        // One block is one window, but the exact count still comes from the splitter rather
        // than from a guess about how capture will cut the text.
        let units = CHUNKS_PER_BATCH.min(target_chunks - chunk_count);
        let text = transcript(next_unit, units);
        next_unit += units;

        let windows = split_transcript_into_windows(&text, CHUNK_WINDOW);
        let expected_chunks = windows.len();
        let expected_bytes: u64 = windows
            .iter()
            .map(|window| u64::try_from(window.len()).expect("a window length fits a byte total"))
            .sum();

        let report = capture_tier0(
            store,
            Tier0CaptureRequest {
                context_label: label.to_string(),
                session_ref: format!("fixture-{label}-{batch}"),
                transcript: text.clone(),
                source: "test".to_string(),
                chunk_window: CHUNK_WINDOW,
            },
        )
        .expect("model-free capture of a fixture batch");

        assert_eq!(
            report.skipped_existing, 0,
            "batch {batch} of {label} generated content that had already been captured — the fixture must not dedup"
        );
        assert_eq!(
            report.skipped_noise, 0,
            "batch {batch} of {label} generated a window that cleaned down to nothing"
        );
        assert_eq!(
            report.chunks_kept, expected_chunks,
            "capture kept a different number of chunks than the windowing it was handed"
        );
        assert_eq!(
            u64::try_from(report.raw_bytes).expect("a capture byte total fits"),
            expected_bytes,
            "capture kept a different byte total than the windowing it was handed"
        );

        chunk_count += expected_chunks;
        raw_bytes += expected_bytes;
        digest_ids.extend(report.digest_ids);
        batch += 1;
    }

    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|context| context.name == label)
        .expect("the fixture capture context exists")
        .id;

    assert_eq!(
        digest_ids.len(),
        chunk_count,
        "model-free writes one digest per kept chunk"
    );

    CapturedContext {
        context_id,
        chunk_count,
        raw_bytes,
        digest_ids,
    }
}

/// Fill `store` through the production model-free capture door until the context holds more than
/// [`TARGET_CHUNKS`] raw chunks, returning the totals computed from what was handed in.
pub fn fill(store: &Store) -> LargeStore {
    let captured = capture_into(store, CONTEXT, 0, TARGET_CHUNKS);
    assert!(
        captured.chunk_count > 50_000,
        "the fixture must hold more rows ({}) than the shipped `work` ceiling admits",
        captured.chunk_count
    );
    LargeStore {
        context_id: captured.context_id,
        chunk_count: captured.chunk_count,
        raw_bytes: captured.raw_bytes,
        digest_ids: captured.digest_ids,
    }
}

/// The node a fixture stands in for when it records what a worker did.
pub const FIXTURE_WORKER: &str = "fixture-worker";

/// Digests claimed through the library's own drain door, left in flight.
pub const CLAIMED_THROUGH_CG: usize = 150;
/// Digests failed through the library's own drain door, up to the attempt cap.
pub const FAILED_THROUGH_CG: usize = 50;
/// Digests whose kept raw is purged, which cancels their still-pending understanding job.
pub const PURGED_THROUGH_CG: usize = 150;
/// Digests carrying a write-once completion and no ledger row at all.
pub const COMPLETED_THROUGH_CG: usize = 150;
/// Digests a worker claimed and is still holding.
pub const CLAIMED_BY_A_WORKER: usize = 900;
/// Digests a worker finished, recorded as a ledger result.
pub const RESULTED_BY_A_WORKER: usize = 600;
/// Digests a worker failed at every attempt the job allows.
pub const FAILED_BY_A_WORKER: usize = 500;
/// Digests a worker's job was called off for.
pub const CANCELLED_BY_A_WORKER: usize = 500;
/// Digests carrying BOTH a ledger result and a write-once completion — counted once, as structured.
pub const RESULTED_AND_COMPLETED: usize = 150;

/// The states forced onto a slice of one context's understanding jobs, as the counts `cg status`
/// must then report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForcedStates {
    pub in_flight: usize,
    pub failed: usize,
    pub skipped: usize,
    pub structured: usize,
    /// How many of the context's jobs carry any ledger state or completion at all.
    pub stateful_jobs: usize,
    /// Raw chunks the purge removed, and the bytes it reported removing with them.
    pub chunks_purged: usize,
    pub bytes_purged: u64,
}

/// The work-ledger job id model-free capture submits for a digest.
///
/// A worker recording a claim, a result, a failure or a cancellation names the job it did the work
/// for, so a fixture standing in for that worker has to name it the same way.
/// [`force_state_spread`] proves the name is right before it writes anything.
fn understanding_job_id(digest: DigestId) -> String {
    format!("cg.understanding.{digest}")
}

/// Give more than one read budget of `fixture`'s understanding jobs real ledger state, spread
/// across every state a job can be in.
///
/// Counting per state has to stay correct at volume, so the states here are thousands of jobs deep
/// rather than a handful: a count that quietly stopped at the first page, or that took its answer
/// from one budget's worth of rows, cannot survive this spread. Two writers produce it, because both
/// exist in production — the library's own drain door (claim, fail, purge, complete) and a worker
/// recording straight onto the shipped work ledger (claim, result, failure, cancellation).
/// [`RESULTED_AND_COMPLETED`] digests carry both a ledger result and a write-once completion, which
/// is one digest in one state — structured — not two.
pub fn force_state_spread(store: &Store, fixture: &LargeStore) -> ForcedStates {
    let db = store.sync_database();
    let first = *fixture
        .digest_ids
        .first()
        .expect("the fixture captured at least one digest");
    assert!(
        job_snapshot(&db, &understanding_job_id(first))
            .expect("read a ledger job")
            .is_some(),
        "the fixture must name the understanding job capture actually submitted"
    );

    let mut digests = fixture.digest_ids.iter().copied();
    let mut take = |count: usize| -> Vec<DigestId> { (&mut digests).take(count).collect() };

    for digest in take(CLAIMED_THROUGH_CG) {
        assert!(
            claim_digest(store, digest).expect("claim a pending digest"),
            "a pending digest is claimable"
        );
    }

    for digest in take(FAILED_THROUGH_CG) {
        for attempt in 1..=MAX_UNDERSTANDING_ATTEMPTS {
            assert!(
                claim_digest(store, digest).expect("claim before recording a failure"),
                "a digest below the attempt cap is claimable again"
            );
            let failure = mark_failed(store, digest, "fixture failure", MAX_UNDERSTANDING_ATTEMPTS)
                .expect("record a model failure");
            assert_eq!(
                failure.terminal,
                attempt == MAX_UNDERSTANDING_ATTEMPTS,
                "a digest turns terminally failed at the attempt cap and not before"
            );
        }
    }

    let mut chunks_purged = 0usize;
    let mut bytes_purged = 0u64;
    for digest in take(PURGED_THROUGH_CG) {
        let report = forget_raw(
            store,
            Some(fixture.context_id),
            ForgetSelector::Digest(digest),
        )
        .expect("purge one digest's kept raw");
        assert_eq!(
            report.queue_rows_skipped, 1,
            "purging the raw of an un-understood digest cancels its understanding job"
        );
        chunks_purged += report.chunks_purged;
        bytes_purged += report.bytes_purged;
    }

    for digest in take(COMPLETED_THROUGH_CG) {
        mark_structured(store, fixture.context_id, digest, "test-model", None)
            .expect("record a write-once understanding completion");
    }

    for digest in take(CLAIMED_BY_A_WORKER) {
        hold_a_claim(&db, &understanding_job_id(digest));
    }
    for digest in take(RESULTED_BY_A_WORKER) {
        record_a_result(&db, &understanding_job_id(digest));
    }
    for digest in take(FAILED_BY_A_WORKER) {
        fail_every_attempt(
            &db,
            &understanding_job_id(digest),
            MAX_UNDERSTANDING_ATTEMPTS,
        );
    }
    for digest in take(CANCELLED_BY_A_WORKER) {
        cancel_the_job(&db, &understanding_job_id(digest));
    }
    for digest in take(RESULTED_AND_COMPLETED) {
        record_a_result(&db, &understanding_job_id(digest));
        mark_structured(store, fixture.context_id, digest, "test-model", None)
            .expect("record a write-once understanding completion over a ledger result");
    }

    ForcedStates {
        in_flight: CLAIMED_THROUGH_CG + CLAIMED_BY_A_WORKER,
        failed: FAILED_THROUGH_CG + FAILED_BY_A_WORKER,
        skipped: PURGED_THROUGH_CG + CANCELLED_BY_A_WORKER,
        structured: COMPLETED_THROUGH_CG + RESULTED_BY_A_WORKER + RESULTED_AND_COMPLETED,
        stateful_jobs: CLAIMED_THROUGH_CG
            + FAILED_THROUGH_CG
            + PURGED_THROUGH_CG
            + COMPLETED_THROUGH_CG
            + CLAIMED_BY_A_WORKER
            + RESULTED_BY_A_WORKER
            + FAILED_BY_A_WORKER
            + CANCELLED_BY_A_WORKER
            + RESULTED_AND_COMPLETED,
        chunks_purged,
        bytes_purged,
    }
}

/// A worker holds this job: one claim, leased until the worker says otherwise.
fn hold_a_claim(db: &Database, job_id: &str) {
    assert_eq!(
        insert_claim(db, job_id, 1, FIXTURE_WORKER, i64::MAX, 0).expect("record a worker claim"),
        ClaimInsert::Inserted,
        "an untouched job's first claim is the fixture's"
    );
}

/// A worker finished this job: the ledger carries its result.
fn record_a_result(db: &Database, job_id: &str) {
    record_result(
        db,
        job_id,
        1,
        FIXTURE_WORKER,
        b"{}",
        serde_json::json!({ "fixture": true }),
        0,
    )
    .expect("record a worker result");
}

/// A worker failed this job at every attempt it allows, so it is terminally failed.
fn fail_every_attempt(db: &Database, job_id: &str, max_attempts: u32) {
    for attempt in 1..=i64::from(max_attempts) {
        record_failure(db, job_id, attempt, FIXTURE_WORKER, "fixture failure", 0)
            .expect("record a worker failure");
    }
}

/// This job was called off before any result came back.
fn cancel_the_job(db: &Database, job_id: &str) {
    cancel_job(db, job_id, FIXTURE_WORKER, Some("fixture cancellation"), 0)
        .expect("record a worker cancellation");
}

/// Capture contexts that have nothing to do with the one status is asked about.
pub const CROWD_CONTEXTS: [&str; 2] = ["sidebar", "archive"];
/// Raw chunks — and understanding jobs — each unrelated context contributes.
pub const CROWD_CHUNKS_PER_CONTEXT: usize = 1_500;
/// A work class that is not context-graph's, writing to the same ledger tables.
pub const UNRELATED_WORK_CLASS: &str = "fixture.unrelated";
/// Jobs the unrelated work class submits.
pub const UNRELATED_JOBS: usize = 1_200;
/// Attempts an unrelated job allows before its failures make it terminal.
pub const UNRELATED_MAX_ATTEMPTS: u32 = 3;

/// Block numbering the crowd starts at, past everything the requested context used, so no
/// unrelated capture ever generates content the requested context already holds.
const CROWD_FIRST_UNIT: usize = 1_000_000;

/// Jobs a worker claims and keeps holding, in each unrelated context and in the unrelated class.
pub const CROWD_CLAIMED: usize = 400;
/// Jobs a worker finishes there.
pub const CROWD_RESULTED: usize = 300;
/// Jobs a worker fails to the attempt cap there.
pub const CROWD_FAILED: usize = 200;
/// Jobs called off there.
pub const CROWD_CANCELLED: usize = 200;

/// What else the store holds while status is asked about one context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerCrowd {
    /// The unrelated capture contexts.
    pub contexts: Vec<ContextId>,
    /// Digests — and understanding jobs — held by those contexts.
    pub foreign_digests: usize,
    /// Jobs submitted under a work class that is not context-graph's.
    pub unrelated_jobs: usize,
    /// Claim, result, failure and cancellation rows written for work in neither the requested
    /// context nor context-graph's own work class.
    pub foreign_ledger_rows: usize,
}

/// Fill the same ledger tables with work the requested context has nothing to do with.
///
/// Two things share those tables in production: other capture contexts on the same machine, and
/// whatever else the embedding application submits — a work class context-graph never looks at. A
/// status total for ONE context has to be unmoved by both. Its counts must not change, and no read
/// it makes may grow because the machine has been busy elsewhere, so this writes thousands of rows
/// of each and the proof re-asks the same question afterwards.
pub fn crowd_the_ledger(store: &Store) -> LedgerCrowd {
    let db = store.sync_database();
    let mut contexts = Vec::new();
    let mut foreign_digests = 0usize;
    let mut foreign_ledger_rows = 0usize;

    for (position, label) in CROWD_CONTEXTS.iter().enumerate() {
        let captured = capture_into(
            store,
            label,
            CROWD_FIRST_UNIT + position * 2 * CROWD_CHUNKS_PER_CONTEXT,
            CROWD_CHUNKS_PER_CONTEXT,
        );
        foreign_digests += captured.chunk_count;
        contexts.push(captured.context_id);
        let jobs: Vec<String> = captured
            .digest_ids
            .iter()
            .map(|digest| understanding_job_id(*digest))
            .collect();
        foreign_ledger_rows += spread_worker_state(&db, &jobs, MAX_UNDERSTANDING_ATTEMPTS);
    }

    let mut unrelated: Vec<String> = Vec::with_capacity(UNRELATED_JOBS);
    for index in 0..UNRELATED_JOBS {
        let job_id = format!("{UNRELATED_WORK_CLASS}.{index}");
        let spec = JobSpec::builder(
            job_id.clone(),
            UNRELATED_WORK_CLASS,
            "compact",
            FIXTURE_WORKER,
        )
        .max_attempts(i64::from(UNRELATED_MAX_ATTEMPTS))
        .build();
        submit_job(&db, &spec, &[] as &[&[u8]]).expect("submit a job of an unrelated work class");
        unrelated.push(job_id);
    }
    foreign_ledger_rows += spread_worker_state(&db, &unrelated, UNRELATED_MAX_ATTEMPTS);

    LedgerCrowd {
        contexts,
        foreign_digests,
        unrelated_jobs: UNRELATED_JOBS,
        foreign_ledger_rows,
    }
}

/// Claim, finish, fail and call off a slice of `jobs`, returning how many ledger rows that wrote.
fn spread_worker_state(db: &Database, jobs: &[String], max_attempts: u32) -> usize {
    assert!(
        jobs.len() >= CROWD_CLAIMED + CROWD_RESULTED + CROWD_FAILED + CROWD_CANCELLED,
        "the crowd needs {} jobs to put in every state and was handed {}",
        CROWD_CLAIMED + CROWD_RESULTED + CROWD_FAILED + CROWD_CANCELLED,
        jobs.len()
    );
    let mut jobs = jobs.iter();
    let mut take = |count: usize| -> Vec<String> { (&mut jobs).take(count).cloned().collect() };

    for job_id in take(CROWD_CLAIMED) {
        hold_a_claim(db, &job_id);
    }
    for job_id in take(CROWD_RESULTED) {
        record_a_result(db, &job_id);
    }
    for job_id in take(CROWD_FAILED) {
        fail_every_attempt(db, &job_id, max_attempts);
    }
    for job_id in take(CROWD_CANCELLED) {
        cancel_the_job(db, &job_id);
    }

    CROWD_CLAIMED
        + CROWD_RESULTED
        + CROWD_FAILED * usize::try_from(max_attempts).expect("an attempt cap fits")
        + CROWD_CANCELLED
}
