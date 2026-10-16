//! model-free deterministic capture: the default `cg record` / `cg import` front door.
//!
//! model-free capture writes a verbatim digest, its local embedding, the kept raw source chunk, and a
//! durable understanding job on the shipped work ledger (`work_jobs` + its copied `work_inputs`) —
//! all in ONE atomic transaction, with ZERO model calls. The model never runs on this path;
//! understanding (structuring a chunk into the ontology) is deferred to a later worker that drains the
//! ledger. This is the "day one import everything free; overnight structure locally" contract:
//! capture is instant and costless, so a developer can fill cg with a session's history without
//! paying a token or waiting on a provider.
//!
//! Raw source chunks NEVER leave the machine (`cg_source_chunks` syncs `None`) — the graph is the
//! durable, synced asset; raw is local understanding substrate. Because raw is kept, this module
//! also owns the purge path (`forget_raw`) and the retention window (`raw_retention_days`): keeping
//! raw without a way to delete it is not shippable.

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use contextdb_core::{Value as DbValue, VectorIndexRef};
use contextdb_engine::work_ledger::{
    self, InputRef, JobSpec, add_job_input_in_tx, job_state, job_state_counts, submit_job_in_tx,
};
use sha2::{Digest as _, Sha256};

use crate::embedding_space::BuiltinSpaceRegistry;
use crate::fabric::CaptureFault;
use crate::ingest::split_transcript_into_windows;
use crate::status_reads::{StatusReads, aggregate_u64};
use crate::understanding::MAX_UNDERSTANDING_ATTEMPTS;
use crate::{
    CgError, Context, ContextId, CreateContext, DigestId, EmbeddingId, EvidenceProducer, Result,
    Store,
};

pub use crate::status_reads::{ReadObservation, StatusReadObserver, UnobservedReads};

/// The `kind` model-free capture stamps on every digest it writes, in the same transaction as that
/// digest's understanding job. It is what makes "the digests this context queued for understanding"
/// answerable as one aggregate.
pub(crate) const VERBATIM_KIND: &str = "verbatim";

/// The work-ledger class every model-free understanding job is submitted under. A worker advertises this
/// as a requirement tag to become eligible.
pub(crate) const UNDERSTANDING_WORK_CLASS: &str = "cg.understanding";
/// The requirement tag a capable understanding worker must advertise to claim an understanding job.
pub(crate) const UNDERSTANDING_REQUIREMENT_TAG: &str = "class:cg.understanding";
/// The stable local submitter/executor node id for a single-node install (submitter, worker, and
/// digest owner are the same machine).
pub(crate) const LOCAL_NODE_ID: &str = "cg-local";

/// Default byte window a model-free capture splits a transcript into when the caller does not resolve
/// one. Each window becomes one verbatim digest (the retrieval granularity).
pub const TIER0_DEFAULT_CHUNK_WINDOW: usize = 16_000;

const RAW_RETENTION_DAYS_KEY: &str = "raw_retention_days";
const MILLIS_PER_DAY: i64 = 86_400_000;

/// A request to capture a conversation deterministically, with no model involvement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tier0CaptureRequest {
    /// The capture context (tenant/blueprint handle). Rows are always stored under this context.
    pub context_label: String,
    /// Groups this capture's digests as one session; carried on every digest's `session_ref`.
    pub session_ref: String,
    /// The conversation text to capture.
    pub transcript: String,
    /// Provenance label written to each digest's `source` column (e.g. "cg-record", "cg-import").
    pub source: String,
    /// Byte window each digest covers. Falls back to [`TIER0_DEFAULT_CHUNK_WINDOW`] when zero.
    pub chunk_window: usize,
}

impl Tier0CaptureRequest {
    fn resolved_window(&self) -> usize {
        if self.chunk_window == 0 {
            TIER0_DEFAULT_CHUNK_WINDOW
        } else {
            self.chunk_window
        }
    }
}

/// The deterministic-capture receipt. Understanding is deferred, so there are no harvest counts
/// here — only the model-free truth: digests written, raw kept, what was skipped, and the always-zero
/// model cost.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tier0CaptureReport {
    /// Verbatim digests written this capture (one per kept chunk).
    pub digests_written: usize,
    /// Raw source chunks kept this capture (equal to `digests_written`).
    pub chunks_kept: usize,
    /// Chunks whose deterministic digest already existed (idempotent re-capture) — no rows written.
    pub skipped_existing: usize,
    /// Chunks that cleaned down to nothing (pure tool noise) — no digest, no raw kept.
    pub skipped_noise: usize,
    /// Raw bytes kept this capture.
    pub raw_bytes: usize,
    /// Pending understanding jobs in the capture context after this capture (ledger job state).
    pub pending_after: usize,
    /// Always zero — model-free never calls a model.
    pub model_calls: usize,
    /// The session handle these digests were grouped under.
    pub session_ref: String,
    /// The ids of the digests written this capture (skips excluded).
    pub digest_ids: Vec<DigestId>,
}

/// Understanding-job counts for one context, by state (computed over the work ledger).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueueCounts {
    pub pending: usize,
    pub in_flight: usize,
    pub structured: usize,
    pub failed: usize,
    pub skipped: usize,
}

/// The raw-source footprint for one context: how many chunks are kept and their total bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RawFootprint {
    pub chunk_count: usize,
    pub bytes: u64,
}

/// What a [`forget_raw`] call should purge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgetSelector {
    /// All raw chunks kept for one digest.
    Digest(DigestId),
    /// All raw chunks whose digest was captured under this session handle.
    Session(String),
    /// Every raw chunk in the capture context.
    Context,
    /// Every raw chunk visible to the store handle (all contexts).
    AllRaw,
    /// Raw chunks older than the retention window. A `None` window (retention unset) purges nothing.
    Expired {
        retention_days: Option<u64>,
        now_ms: i64,
    },
}

/// The result of a purge: what was removed and which queue rows were marked skipped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ForgetReport {
    pub chunks_purged: usize,
    pub bytes_purged: u64,
    pub queue_rows_skipped: usize,
}

/// The deterministic salience cut: strip tool_use / tool_result payloads and harness noise
/// (system reminders, teammate messages, local-command blocks), keeping user + assistant dialogue.
///
/// This is a pure function — the summary a verbatim digest is embedded from. The kept raw chunk
/// retains the FULL slice (tool noise and all); this is only the cleaned rendering. Cleaning is
/// deterministic and idempotent: already-clean prose passes through unchanged.
pub fn salience_clean(raw: &str) -> String {
    let without_tool_use = strip_prefixed_tag_blocks(raw, "<tool_use", "</tool_use>");
    let without_tool_result =
        strip_prefixed_tag_blocks(&without_tool_use, "<tool_result", "</tool_result>");
    let without_system = strip_prefixed_tag_blocks(
        &without_tool_result,
        "<system-reminder",
        "</system-reminder>",
    );
    let without_teammate =
        strip_prefixed_tag_blocks(&without_system, "<teammate-message", "</teammate-message>");
    let without_local = strip_local_command_blocks(&without_teammate);
    collapse_blank_lines(&without_local)
}

/// Capture a conversation deterministically: verbatim digest + embedding + raw chunk + one ledger
/// understanding job per chunk, all committed in ONE atomic transaction, ZERO model calls. Re-running
/// the same capture is idempotent — a chunk whose deterministic digest already exists is skipped,
/// writing no new rows.
pub fn capture_tier0(store: &Store, request: Tier0CaptureRequest) -> Result<Tier0CaptureReport> {
    capture_tier0_inner(store, request, false, None)
}

/// The fabric-aware model-free capture entry point. `source_opt_out` keeps this source's understanding
/// input local (submitted as an `InputRef::local_path` with no copied `work_inputs` row); `fault` is a
/// test-only seam that aborts the capture transaction after the digest write to prove all-or-nothing
/// rollback. Production callers pass `false` / `None`.
pub(crate) fn capture_tier0_inner(
    store: &Store,
    request: Tier0CaptureRequest,
    source_opt_out: bool,
    fault: Option<&dyn CaptureFault>,
) -> Result<Tier0CaptureReport> {
    let transcript = request.transcript.trim();
    let context = context_by_label(store, &request.context_label)?;
    let mut report = Tier0CaptureReport {
        session_ref: request.session_ref.clone(),
        ..Default::default()
    };
    if transcript.is_empty() {
        report.pending_after = queue_counts(store, context.id)?.pending;
        return Ok(report);
    }

    let window = request.resolved_window();
    let chunks: Vec<String> = split_transcript_into_windows(transcript, window)
        .into_iter()
        .map(str::to_string)
        .collect();

    let db = store.internal_db();
    let tx = db.begin().map_err(map_engine)?;
    let outcome = write_tier0_chunks(
        store,
        tx,
        context.id,
        &request,
        &chunks,
        source_opt_out,
        fault,
        &mut report,
    );
    match outcome {
        Ok(()) => {
            db.commit(tx).map_err(map_engine)?;
        }
        Err(err) => {
            // Any failure mid-capture rolls back the whole transaction: no digest, chunk, queue, or
            // embedding rows survive a partial capture.
            let _ = db.rollback(tx);
            return Err(err);
        }
    }
    report.pending_after = queue_counts(store, context.id)?.pending;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn write_tier0_chunks(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
    request: &Tier0CaptureRequest,
    chunks: &[String],
    source_opt_out: bool,
    fault: Option<&dyn CaptureFault>,
    report: &mut Tier0CaptureReport,
) -> Result<()> {
    let now_ms = Utc::now().timestamp_millis();
    let mut seen: std::collections::HashSet<uuid::Uuid> = std::collections::HashSet::new();
    let mut seq: i64 = 0;
    for chunk in chunks {
        let content_hash = hex_sha256(chunk);
        let digest_uuid =
            deterministic_digest_uuid(context_id, &request.session_ref, &content_hash);
        // In-batch duplicate (an identical chunk twice in one capture) — keep it once.
        if !seen.insert(digest_uuid) {
            report.skipped_existing += 1;
            continue;
        }
        // Idempotent re-capture: a committed digest with this id already exists.
        if digest_exists(store, context_id, digest_uuid)? {
            report.skipped_existing += 1;
            continue;
        }
        let summary = salience_clean(chunk);
        if summary.trim().is_empty() {
            // Pure tool noise cleans to nothing — no digest to attach raw to; drop it.
            report.skipped_noise += 1;
            continue;
        }
        seq += 1;
        let digest_id = DigestId::from(digest_uuid);
        write_verbatim_digest(store, tx, context_id, digest_id, request, &summary)?;
        write_source_chunk(
            store,
            tx,
            context_id,
            digest_id,
            seq,
            chunk,
            &content_hash,
            now_ms,
        )?;
        // Test-only rollback seam: fire after the digest/raw write, before the ledger job/input
        // writes, so an abort here leaves NOTHING behind — the whole capture transaction rolls back.
        if let Some(fault) = fault {
            fault.fail_after_digest()?;
        }
        submit_understanding_job(
            store,
            tx,
            context_id,
            digest_id,
            chunk,
            source_opt_out,
            now_ms,
        )?;
        report.digests_written += 1;
        report.chunks_kept += 1;
        report.raw_bytes += chunk.len();
        report.digest_ids.push(digest_id);
    }
    Ok(())
}

fn write_verbatim_digest(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
    digest_id: DigestId,
    request: &Tier0CaptureRequest,
    summary: &str,
) -> Result<()> {
    let vector = store.embed_digest_text(summary)?;
    let db = store.internal_db();
    let row_id = db
        .insert_row(
            tx,
            "digests",
            HashMap::from([
                ("id".to_string(), DbValue::Uuid(digest_id.as_uuid())),
                ("source".to_string(), DbValue::Text(request.source.clone())),
                (
                    "session_ref".to_string(),
                    DbValue::Text(request.session_ref.clone()),
                ),
                ("summary".to_string(), DbValue::Text(summary.to_string())),
                ("kind".to_string(), DbValue::Text(VERBATIM_KIND.to_string())),
                ("vector_text".to_string(), DbValue::Vector(vector.clone())),
                (
                    "dimension".to_string(),
                    DbValue::Int64(crate::schema::DIGEST_VECTOR_DIMENSION as i64),
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
        .map_err(map_engine)?;
    let embedding_id = EmbeddingId::new_v7();
    db.execute_in_tx(
        tx,
        "INSERT INTO digest_embedding_handles (id, context_id, embedding_space_id, target_id, produced_by, producer_system, created_at) VALUES ($id, $context_id, $embedding_space_id, $target_id, $produced_by, $producer_system, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(embedding_id.as_uuid())),
            (
                "context_id".to_string(),
                DbValue::Uuid(context_id.as_uuid()),
            ),
            (
                "embedding_space_id".to_string(),
                DbValue::Text(BuiltinSpaceRegistry::TEXT_DEFAULT_ID.to_string()),
            ),
            ("target_id".to_string(), DbValue::Uuid(digest_id.as_uuid())),
            (
                "produced_by".to_string(),
                DbValue::Json(
                    serde_json::to_value(EvidenceProducer::default())
                        .map_err(|err| CgError::Engine(format!("producer serialize: {err}")))?,
                ),
            ),
            ("producer_system".to_string(), DbValue::Text(String::new())),
            ("created_at".to_string(), DbValue::TxId(tx)),
        ]),
    )
    .map_err(map_engine)?;
    db.insert_vector(
        tx,
        VectorIndexRef::new("digests", "vector_text"),
        row_id,
        vector,
    )
    .map_err(map_engine)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_source_chunk(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
    digest_id: DigestId,
    seq: i64,
    content: &str,
    content_hash: &str,
    now_ms: i64,
) -> Result<()> {
    store
        .internal_db()
        .execute_in_tx(
            tx,
            "INSERT INTO cg_source_chunks (id, digest_id, context_id, seq, content, content_hash, byte_len, created_at, created_at_ms) VALUES ($id, $digest_id, $context_id, $seq, $content, $content_hash, $byte_len, $created_at, $created_at_ms)",
            &HashMap::from([
                ("id".to_string(), DbValue::Text(digest_id.to_string())),
                (
                    "digest_id".to_string(),
                    DbValue::Text(digest_id.to_string()),
                ),
                (
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                ),
                ("seq".to_string(), DbValue::Int64(seq)),
                ("content".to_string(), DbValue::Text(content.to_string())),
                (
                    "content_hash".to_string(),
                    DbValue::Text(content_hash.to_string()),
                ),
                ("byte_len".to_string(), DbValue::Int64(content.len() as i64)),
                ("created_at".to_string(), DbValue::TxId(tx)),
                ("created_at_ms".to_string(), DbValue::Int64(now_ms)),
            ]),
        )
        .map_err(map_engine)?;
    Ok(())
}

/// The deterministic 1:1 ledger job id for a model-free digest. Re-importing identical content yields the
/// same digest and thus the same job id, so `work_jobs`' insert-if-not-exists dedups the resubmit.
pub(crate) fn understanding_job_id(digest_id: DigestId) -> String {
    format!("cg.understanding.{}", digest_id.as_uuid())
}

/// Submit one chunk's understanding as a work-ledger job on the caller's OPEN capture transaction, so
/// the job (and its ephemeral input copy) commits atomically with the verbatim digest — one $0
/// model-free transaction, no second commit. The requirement tags carry the worker capability
/// contract; priority is the digest's local, never-synced retrieval-touch count. The job's
/// `output_schema` is unset.
///
/// Per-source opt-out is a ref-kind swap, not a policy language: an opted-out source submits its input
/// as `InputRef::local_path` (no copied `work_inputs` row), which the shipped movement policy refuses
/// for any non-submitter regardless of the global sharing knob. A default source submits
/// `InputRef::ledger_input` and copies the verbatim chunk into `work_inputs` — the only thing a remote
/// worker reads. Either way the job row references its input by id and never inlines the bytes.
fn submit_understanding_job(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
    digest_id: DigestId,
    chunk: &str,
    source_opt_out: bool,
    now_ms: i64,
) -> Result<()> {
    let job_id = understanding_job_id(digest_id);
    let priority = retrieval_touch_count_in(store, digest_id)? as i64;
    let input_ref = if source_opt_out {
        InputRef::local_path(&format!("cg_source_chunks/{digest_id}"))
    } else {
        InputRef::ledger_input()
    };
    let spec = JobSpec::builder(
        job_id.clone(),
        UNDERSTANDING_WORK_CLASS,
        "understand",
        LOCAL_NODE_ID,
    )
    .requirement_tags(vec![UNDERSTANDING_REQUIREMENT_TAG.to_string()])
    .input_refs(vec![input_ref])
    .priority(priority)
    .max_attempts(i64::from(MAX_UNDERSTANDING_ATTEMPTS))
    .provenance(Some(
        serde_json::json!({ "context_id": context_id.as_uuid() }),
    ))
    .submitted_at_ms(now_ms)
    .build();
    submit_job_in_tx(store.internal_db(), tx, &spec).map_err(map_engine)?;
    if !source_opt_out {
        // The understanding input: the verbatim chunk text, referenced from the job — never inlined.
        add_job_input_in_tx(store.internal_db(), tx, &job_id, 0, chunk.as_bytes())
            .map_err(map_engine)?;
    }
    Ok(())
}

/// Count understanding jobs in one context, by state, through the bounded reading door.
///
/// What an operator is entitled to: the numbers are CORRECT on a store of any size, and asking for
/// them costs the same whether the machine has done a day's work or a year's. A context holding
/// hundreds of thousands of captured digests is far past what one read may examine, so this walks
/// the context's own digests one bounded page at a time and keeps only the running counts and the
/// page in hand — never a set of identities that grows with the store, never a collected table, and
/// never a read whose size follows the ledger rather than the page in hand.
///
/// A digest is in the understanding queue only once its job is on the work ledger, so per page the
/// first read asks the ledger which of that page's job ids it actually holds. A digest with no job
/// row is not queued at all — the drain path skips it for the same reason
/// ([`crate::understanding::list_pending_digests`]) — so status never advertises it, in any bucket.
///
/// Of the digests that ARE enqueued, the ones carrying a write-once completion are found with one
/// more membership read over that page's digest ids, and the rest have their state decided by the
/// work ledger's own batched answer ([`job_state_counts`]) for exactly those jobs. The ledger owns
/// what a job's state IS; this only counts. A completion wins over whatever the ledger says, so a
/// digest that is both completed and Done is counted once, as structured. Pending is what the ledger
/// holds a job row for and has nothing live or terminal to say about, which falls out of the page's
/// arithmetic rather than a scan. The batched counts cannot report an id the ledger disowns here —
/// the membership read has already removed every one of them — and were such a count to appear it is
/// excluded as well, so no bucket ever grows from a job that does not exist.
pub fn queue_counts(store: &Store, context_id: ContextId) -> Result<QueueCounts> {
    queue_counts_observed(store, context_id, &UnobservedReads)
}

/// [`queue_counts`], telling `observer` about every bounded read it makes.
///
/// The same code path production takes: what an observer is told is what a plain
/// [`queue_counts`] does.
pub fn queue_counts_observed(
    store: &Store,
    context_id: ContextId,
    observer: &dyn StatusReadObserver,
) -> Result<QueueCounts> {
    let db = store.internal_db();
    let reads = StatusReads::open(store, observer)?;
    let mut queued_params = HashMap::from([(
        "context_id".to_string(),
        DbValue::Uuid(context_id.as_uuid()),
    )]);
    queued_params.insert("kind".to_string(), DbValue::Text(VERBATIM_KIND.to_string()));

    let mut counts = QueueCounts::default();
    // Every digest model-free capture wrote in this context, in bounded pages. `kind` is what capture
    // stamps on the digest it writes in the same transaction as the job, so this is the queue's
    // candidate universe, and it is already scoped to the context — no job identity from another
    // context is ever read, whatever else the machine has queued.
    reads.for_each_page(
        "SELECT id FROM digests WHERE context_id = $context_id AND kind = $kind",
        &queued_params,
        |rows| {
            let digests: Vec<DigestId> = rows
                .iter()
                .filter_map(|row| match row.first() {
                    Some(DbValue::Uuid(id)) => Some(DigestId::from(*id)),
                    _ => None,
                })
                .collect();
            if digests.is_empty() {
                return Ok(());
            }
            // Only a digest whose job reached the ledger is in the queue. One membership read for
            // the whole page decides that, and everything below counts only what survives it.
            let page_jobs: Vec<String> = digests
                .iter()
                .map(|digest| understanding_job_id(*digest))
                .collect();
            let enqueued = keys_present(&reads, "work_jobs", "job_id", &page_jobs)?;
            let queued: Vec<(String, String)> = digests
                .iter()
                .zip(page_jobs)
                .filter(|(_, job_id)| enqueued.contains(job_id))
                .map(|(digest, job_id)| (digest.to_string(), job_id))
                .collect();
            if queued.is_empty() {
                return Ok(());
            }
            let digest_keys: Vec<String> = queued
                .iter()
                .map(|(digest_text, _)| digest_text.clone())
                .collect();
            let completed =
                keys_present(&reads, "cg_understanding_completions", "id", &digest_keys)?;
            counts.structured += completed.len();
            let jobs: Vec<&String> = queued
                .iter()
                .filter(|(digest_text, _)| !completed.contains(digest_text))
                .map(|(_, job_id)| job_id)
                .collect();
            if jobs.is_empty() {
                return Ok(());
            }
            let states = job_state_counts(db, &jobs, ledger_clock_ms()).map_err(map_engine)?;
            counts.in_flight += states.leased;
            counts.structured += states.done;
            counts.failed += states.failed;
            counts.skipped += states.cancelled;
            // A job the ledger holds a row for and has nothing live or terminal to say about is
            // queued and untouched. An id the ledger disowns cannot reach here, and is excluded if
            // it ever does.
            counts.pending += states.pending;
            Ok(())
        },
    )?;
    Ok(counts)
}

/// Which of one page's keys `table` actually holds a row for, under `key_column`.
///
/// One bounded read for the whole page: the page's keys are asked about together with a keyed `IN`,
/// so the cost follows the page in hand and not the size of the table being asked. This is how a
/// status total learns both which digests reached the work ledger and which carry a write-once
/// completion, without ever reading either table through.
fn keys_present(
    reads: &StatusReads<'_>,
    table: &str,
    key_column: &str,
    keys: &[String],
) -> Result<HashSet<String>> {
    let mut params = HashMap::with_capacity(keys.len());
    let mut placeholders = Vec::with_capacity(keys.len());
    for (position, key) in keys.iter().enumerate() {
        let name = format!("key{position}");
        params.insert(name.clone(), DbValue::Text(key.clone()));
        placeholders.push(format!("${name}"));
    }
    let statement = format!(
        "SELECT {key_column} FROM {table} WHERE {key_column} IN ({})",
        placeholders.join(", ")
    );
    let mut present = HashSet::with_capacity(keys.len());
    reads.for_each_page(&statement, &params, |rows| {
        for row in rows {
            if let Some(DbValue::Text(key)) = row.first() {
                present.insert(key.clone());
            }
        }
        Ok(())
    })?;
    Ok(present)
}

/// The advisory wall-clock the local single-node claim model reads. Local claims take an unexpiring
/// lease (`i64::MAX`), so any finite clock keeps them live.
pub(crate) fn ledger_clock_ms() -> i64 {
    0
}

/// The digests in a context, each with a monotonic ordering key (the digest's `created_at` TxId), so
/// callers can order by capture age deterministically.
pub(crate) fn context_digests(
    store: &Store,
    context_id: ContextId,
) -> Result<Vec<(DigestId, i128)>> {
    let result = store.read_query(
        "SELECT id, created_at FROM digests WHERE context_id = $context_id",
        &HashMap::from([(
            "context_id".to_string(),
            DbValue::Uuid(context_id.as_uuid()),
        )]),
    )?;
    Ok(result
        .rows
        .iter()
        .filter_map(|row| match row.first() {
            Some(DbValue::Uuid(id)) => {
                Some((DigestId::from(*id), row.get(1).map(order_key).unwrap_or(0)))
            }
            _ => None,
        })
        .collect())
}

pub(crate) fn job_exists(store: &Store, job_id: &str) -> Result<bool> {
    let result = store.read_query(
        "SELECT job_id FROM work_jobs WHERE job_id = $job_id",
        &HashMap::from([("job_id".to_string(), DbValue::Text(job_id.to_string()))]),
    )?;
    Ok(!result.rows.is_empty())
}

pub(crate) fn completion_exists_local(store: &Store, digest_text: &str) -> Result<bool> {
    let result = store.read_query(
        "SELECT id FROM cg_understanding_completions WHERE id = $id",
        &HashMap::from([("id".to_string(), DbValue::Text(digest_text.to_string()))]),
    )?;
    Ok(!result.rows.is_empty())
}

/// Bump the retrieval-touch counter for one digest: increment `touch_count` and stamp
/// `last_touched_at`, inserting the row at count 1 on first touch. This is the capture signal
/// that a captured memory actually answered a query — the priority hint the understanding worker
/// reads to structure the memories a developer keeps reaching for first. Machine-local (the
/// `cg_retrieval_touches` table never syncs — per-device firing stays local), and committed in its
/// own transaction so a touch never rides inside a capture or retrieval read.
///
/// This is a read-then-increment, not an atomic bump: two concurrent hooks on the same data root
/// could both read N and write N+1 (a lost update). That is deliberately acceptable — the touch
/// count is only a drain-PRIORITY hint, never a correctness input, and it never syncs, so a
/// occasional under-count only slightly reorders which memory is structured first.
pub fn bump_touch(store: &Store, context_id: ContextId, digest_id: DigestId) -> Result<()> {
    let db = store.internal_db();
    let tx = db.begin().map_err(map_engine)?;
    let outcome = (|| -> Result<()> {
        // The engine has no `col = col + 1` UPDATE form, so read the current count then upsert.
        let current = retrieval_touch_count_in(store, digest_id)?;
        let updated = db
            .execute_in_tx(
                tx,
                "UPDATE cg_retrieval_touches SET touch_count = $touch_count, last_touched_at = $last_touched_at WHERE digest_id = $digest_id",
                &HashMap::from([
                    (
                        "touch_count".to_string(),
                        DbValue::Int64((current + 1) as i64),
                    ),
                    ("last_touched_at".to_string(), DbValue::TxId(tx)),
                    (
                        "digest_id".to_string(),
                        DbValue::Text(digest_id.to_string()),
                    ),
                ]),
            )
            .map_err(map_engine)?;
        if updated.rows_affected == 0 {
            db.execute_in_tx(
                tx,
                "INSERT INTO cg_retrieval_touches (digest_id, context_id, touch_count, last_touched_at) VALUES ($digest_id, $context_id, $touch_count, $last_touched_at)",
                &HashMap::from([
                    (
                        "digest_id".to_string(),
                        DbValue::Text(digest_id.to_string()),
                    ),
                    (
                        "context_id".to_string(),
                        DbValue::Uuid(context_id.as_uuid()),
                    ),
                    ("touch_count".to_string(), DbValue::Int64(1)),
                    ("last_touched_at".to_string(), DbValue::TxId(tx)),
                ]),
            )
            .map_err(map_engine)?;
        }
        Ok(())
    })();
    match outcome {
        Ok(()) => db.commit(tx).map_err(map_engine)?,
        Err(err) => {
            let _ = db.rollback(tx);
            return Err(err);
        }
    }
    Ok(())
}

/// The retrieval-touch count for one digest (0 if never touched).
pub fn retrieval_touch_count(store: &Store, digest_id: DigestId) -> Result<u64> {
    retrieval_touch_count_in(store, digest_id)
}

fn retrieval_touch_count_in(store: &Store, digest_id: DigestId) -> Result<u64> {
    let result = store.read_query(
        "SELECT touch_count FROM cg_retrieval_touches WHERE digest_id = $digest_id",
        &HashMap::from([(
            "digest_id".to_string(),
            DbValue::Text(digest_id.to_string()),
        )]),
    )?;
    let count = result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::Int64(count)) => Some((*count).max(0) as u64),
        _ => None,
    });
    Ok(count.unwrap_or(0))
}

/// The `kind` marker on a digest (e.g. "verbatim" for a model-free digest), or `None` if absent.
pub fn digest_kind(store: &Store, digest_id: DigestId) -> Result<Option<String>> {
    let result = store.read_query(
        "SELECT kind FROM digests WHERE id = $id",
        &HashMap::from([("id".to_string(), DbValue::Uuid(digest_id.as_uuid()))]),
    )?;
    Ok(result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::Text(kind)) => Some(kind.clone()),
        _ => None,
    }))
}

/// The raw-source footprint (kept chunks + total bytes) for one context.
///
/// One aggregate, paged to completion through the bounded reading door: a context holding more raw
/// chunks than a single read may examine still gets an exact count and an exact byte total, because
/// each fetch spends its own work budget and the running totals survive between them.
pub fn raw_footprint(store: &Store, context_id: ContextId) -> Result<RawFootprint> {
    raw_footprint_observed(store, context_id, &UnobservedReads)
}

/// [`raw_footprint`], telling `observer` about every bounded read it makes.
///
/// The same code path production takes: what an observer is told is what a plain [`raw_footprint`]
/// does.
pub fn raw_footprint_observed(
    store: &Store,
    context_id: ContextId,
    observer: &dyn StatusReadObserver,
) -> Result<RawFootprint> {
    let reads = StatusReads::open(store, observer)?;
    let total = reads.total(
        "SELECT COUNT(*), SUM(byte_len) FROM cg_source_chunks WHERE context_id = $context_id",
        &HashMap::from([(
            "context_id".to_string(),
            DbValue::Uuid(context_id.as_uuid()),
        )]),
    )?;
    Ok(RawFootprint {
        chunk_count: usize::try_from(aggregate_u64(&total, 0)).unwrap_or(usize::MAX),
        bytes: aggregate_u64(&total, 1),
    })
}

/// Purge kept raw source chunks and mark their still-pending queue rows `skipped`. Raw purge is
/// COMPLETE on this machine because raw never syncs. Digests and derived graph rows remain — they
/// are the synced, durable memory. Idempotent: a second identical purge removes nothing.
pub fn forget_raw(
    store: &Store,
    capture_context_id: Option<ContextId>,
    selector: ForgetSelector,
) -> Result<ForgetReport> {
    let targets = select_chunk_targets(store, capture_context_id, &selector)?;
    let mut report = ForgetReport::default();
    if targets.is_empty() {
        return Ok(report);
    }
    let db = store.internal_db();
    let tx = db.begin().map_err(map_engine)?;
    let now_ms = Utc::now().timestamp_millis();
    let outcome = (|| -> Result<()> {
        for target in &targets {
            db.execute_in_tx(
                tx,
                "DELETE FROM cg_source_chunks WHERE id = $id",
                &HashMap::from([("id".to_string(), DbValue::Text(target.chunk_id.clone()))]),
            )
            .map_err(map_engine)?;
            report.chunks_purged += 1;
            report.bytes_purged += target.byte_len;
            // Purging the raw makes an un-understood job un-runnable, so cancel it on the ledger.
            // Only a still-pending job is
            // cancelled (a completed job keeps its result); a re-purge is a no-op.
            let Some(digest_id) = parse_digest_uuid(&target.digest_id) else {
                continue;
            };
            let job_id = understanding_job_id(DigestId::from(digest_id));
            if !job_exists(store, &job_id)? {
                continue;
            }
            if completion_exists_local(store, &target.digest_id)? {
                continue;
            }
            if job_state(db, &job_id, ledger_clock_ms()).map_err(map_engine)?
                != work_ledger::JobState::Pending
            {
                continue;
            }
            db.execute_in_tx(
                tx,
                "INSERT INTO work_cancellations (job_id, requested_by, reason, cancelled_at) VALUES ($job_id, $requested_by, $reason, $cancelled_at)",
                &HashMap::from([
                    ("job_id".to_string(), DbValue::Text(job_id.clone())),
                    (
                        "requested_by".to_string(),
                        DbValue::Text(LOCAL_NODE_ID.to_string()),
                    ),
                    (
                        "reason".to_string(),
                        DbValue::Text("raw purged by cg forget".to_string()),
                    ),
                    ("cancelled_at".to_string(), DbValue::Timestamp(now_ms)),
                ]),
            )
            .map_err(map_engine)?;
            report.queue_rows_skipped += 1;
        }
        Ok(())
    })();
    match outcome {
        Ok(()) => db.commit(tx).map_err(map_engine)?,
        Err(err) => {
            let _ = db.rollback(tx);
            return Err(err);
        }
    }
    Ok(report)
}

struct ChunkTarget {
    chunk_id: String,
    digest_id: String,
    byte_len: u64,
}

fn select_chunk_targets(
    store: &Store,
    capture_context_id: Option<ContextId>,
    selector: &ForgetSelector,
) -> Result<Vec<ChunkTarget>> {
    let (clause, params) = match selector {
        ForgetSelector::Digest(id) => (
            "digest_id = $digest_id".to_string(),
            HashMap::from([("digest_id".to_string(), DbValue::Text(id.to_string()))]),
        ),
        ForgetSelector::Session(session_ref) => {
            // cg_source_chunks carries no session_ref; resolve session -> digest ids first.
            let digest_ids = digest_ids_for_session(store, capture_context_id, session_ref)?;
            if digest_ids.is_empty() {
                return Ok(Vec::new());
            }
            return collect_targets_for_digests(store, &digest_ids);
        }
        ForgetSelector::Context => {
            let context_id = require_context(capture_context_id)?;
            (
                "context_id = $context_id".to_string(),
                HashMap::from([(
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                )]),
            )
        }
        ForgetSelector::AllRaw => (String::new(), HashMap::new()),
        ForgetSelector::Expired {
            retention_days,
            now_ms,
        } => {
            let Some(days) = retention_days else {
                // Retention unset means keep raw forever — nothing expires.
                return Ok(Vec::new());
            };
            let cutoff = now_ms - (*days as i64) * MILLIS_PER_DAY;
            (
                "created_at_ms < $cutoff".to_string(),
                HashMap::from([("cutoff".to_string(), DbValue::Int64(cutoff))]),
            )
        }
    };
    query_chunk_targets(store, &clause, params)
}

fn query_chunk_targets(
    store: &Store,
    clause: &str,
    params: HashMap<String, DbValue>,
) -> Result<Vec<ChunkTarget>> {
    let sql = if clause.is_empty() {
        "SELECT id, digest_id, byte_len FROM cg_source_chunks".to_string()
    } else {
        format!("SELECT id, digest_id, byte_len FROM cg_source_chunks WHERE {clause}")
    };
    let result = store.read_query(&sql, &params)?;
    Ok(result
        .rows
        .iter()
        .filter_map(|row| match row.as_slice() {
            [
                DbValue::Text(chunk_id),
                DbValue::Text(digest_id),
                DbValue::Int64(byte_len),
            ] => Some(ChunkTarget {
                chunk_id: chunk_id.clone(),
                digest_id: digest_id.clone(),
                byte_len: (*byte_len).max(0) as u64,
            }),
            _ => None,
        })
        .collect())
}

fn collect_targets_for_digests(store: &Store, digest_ids: &[String]) -> Result<Vec<ChunkTarget>> {
    let mut targets = Vec::new();
    for digest_id in digest_ids {
        targets.extend(query_chunk_targets(
            store,
            "digest_id = $digest_id",
            HashMap::from([("digest_id".to_string(), DbValue::Text(digest_id.clone()))]),
        )?);
    }
    Ok(targets)
}

fn digest_ids_for_session(
    store: &Store,
    capture_context_id: Option<ContextId>,
    session_ref: &str,
) -> Result<Vec<String>> {
    let mut params = HashMap::from([(
        "session_ref".to_string(),
        DbValue::Text(session_ref.to_string()),
    )]);
    let mut clause = "session_ref = $session_ref".to_string();
    if let Some(context_id) = capture_context_id {
        clause.push_str(" AND context_id = $context_id");
        params.insert(
            "context_id".to_string(),
            DbValue::Uuid(context_id.as_uuid()),
        );
    }
    let sql = format!("SELECT id FROM digests WHERE {clause}");
    let result = store.read_query(&sql, &params)?;
    Ok(result
        .rows
        .iter()
        .filter_map(|row| match row.first() {
            Some(DbValue::Uuid(id)) => Some(id.to_string()),
            _ => None,
        })
        .collect())
}

fn require_context(context_id: Option<ContextId>) -> Result<ContextId> {
    context_id.ok_or_else(|| CgError::Engine("forget requires a capture context".to_string()))
}

/// Read a local setting value (machine-local, never synced).
pub fn get_local_setting(store: &Store, key: &str) -> Result<Option<String>> {
    local_setting(store.as_graph_rows(), key)
}

/// [`get_local_setting`] through the read-only reader; the reader must declare
/// `cg_local_settings` (as [`crate::event_history_reader_declaration`] does).
pub fn get_local_setting_read_only(
    reader: &crate::ConsumerReader,
    key: &str,
) -> Result<Option<String>> {
    local_setting(&reader.graph(), key)
}

/// The one local-setting read, authored once over the shared row seam.
fn local_setting(
    rows: &dyn crate::store::graph_rows::GraphRows,
    key: &str,
) -> Result<Option<String>> {
    let result = rows.query_rows(
        "SELECT value FROM cg_local_settings WHERE key = $key",
        &HashMap::from([("key".to_string(), DbValue::Text(key.to_string()))]),
    )?;
    Ok(result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::Text(value)) => Some(value.clone()),
        _ => None,
    }))
}

/// Upsert a local setting value (machine-local, never synced).
pub fn set_local_setting(store: &Store, key: &str, value: &str) -> Result<()> {
    let db = store.internal_db();
    let tx = db.begin().map_err(map_engine)?;
    let outcome = (|| -> Result<()> {
        let updated = db
            .execute_in_tx(
                tx,
                "UPDATE cg_local_settings SET value = $value, updated_at = $updated_at WHERE key = $key",
                &HashMap::from([
                    ("key".to_string(), DbValue::Text(key.to_string())),
                    ("value".to_string(), DbValue::Text(value.to_string())),
                    ("updated_at".to_string(), DbValue::TxId(tx)),
                ]),
            )
            .map_err(map_engine)?;
        if updated.rows_affected == 0 {
            db.execute_in_tx(
                tx,
                "INSERT INTO cg_local_settings (key, value, updated_at) VALUES ($key, $value, $updated_at)",
                &HashMap::from([
                    ("key".to_string(), DbValue::Text(key.to_string())),
                    ("value".to_string(), DbValue::Text(value.to_string())),
                    ("updated_at".to_string(), DbValue::TxId(tx)),
                ]),
            )
            .map_err(map_engine)?;
        }
        Ok(())
    })();
    match outcome {
        Ok(()) => db.commit(tx).map_err(map_engine)?,
        Err(err) => {
            let _ = db.rollback(tx);
            return Err(err);
        }
    }
    Ok(())
}

/// The configured raw-retention window in days, or `None` (keep raw forever).
pub fn raw_retention_days(store: &Store) -> Result<Option<u64>> {
    Ok(get_local_setting(store, RAW_RETENTION_DAYS_KEY)?
        .and_then(|value| value.trim().parse().ok()))
}

/// Set the raw-retention window in days.
pub fn set_raw_retention_days(store: &Store, days: u64) -> Result<()> {
    set_local_setting(store, RAW_RETENTION_DAYS_KEY, &days.to_string())
}

// --- deterministic identity + hashing helpers -------------------------------------------------

fn deterministic_digest_uuid(
    context_id: ContextId,
    session_ref: &str,
    content_hash: &str,
) -> uuid::Uuid {
    let mut hasher = Sha256::new();
    hasher.update(context_id.as_uuid().as_bytes());
    hasher.update(b"\x1e");
    hasher.update(session_ref.as_bytes());
    hasher.update(b"\x1e");
    hasher.update(content_hash.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    uuid::Uuid::from_bytes(bytes)
}

fn digest_exists(store: &Store, context_id: ContextId, digest_uuid: uuid::Uuid) -> Result<bool> {
    let result = store.read_query(
        "SELECT id FROM digests WHERE id = $id AND context_id = $context_id",
        &HashMap::from([
            ("id".to_string(), DbValue::Uuid(digest_uuid)),
            (
                "context_id".to_string(),
                DbValue::Uuid(context_id.as_uuid()),
            ),
        ]),
    )?;
    Ok(!result.rows.is_empty())
}

fn parse_digest_uuid(digest_text: &str) -> Option<uuid::Uuid> {
    uuid::Uuid::parse_str(digest_text).ok()
}

/// A monotonic ordering key from a row's `created_at`/`enqueued_at` column (a strictly-increasing
/// TxId per committed capture), so callers can order by capture age deterministically.
fn order_key(value: &DbValue) -> i128 {
    match value {
        DbValue::TxId(tx) => i128::from(tx.0),
        DbValue::Int64(value) => i128::from(*value),
        _ => 0,
    }
}

fn hex_sha256(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn context_by_label(store: &Store, label: &str) -> Result<Context> {
    if let Some(existing) = store
        .list_contexts()?
        .into_iter()
        .find(|ctx| ctx.name == label)
    {
        return Ok(existing);
    }
    store.create_context(CreateContext {
        name: label.to_string(),
        ..Default::default()
    })
}

pub(crate) fn map_engine(err: contextdb_core::Error) -> CgError {
    CgError::Engine(format!("unexpected engine error: {err}"))
}

// --- salience-clean text helpers --------------------------------------------------------------

/// Strip `<prefix ...>...</close>` blocks (the open tag may carry attributes). Mirrors the import
/// parser's noise stripping so model-free cleaning recognizes the same harness markers.
fn strip_prefixed_tag_blocks(text: &str, open_prefix: &str, close: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(open_prefix) {
        out.push_str(&rest[..start]);
        let tag = &rest[start..];
        let Some(open_end) = tag.find('>') else {
            break;
        };
        let after_open = &tag[open_end + 1..];
        if let Some(close_start) = after_open.find(close) {
            rest = &after_open[close_start + close.len()..];
        } else {
            rest = "";
        }
    }
    out.push_str(rest);
    out
}

fn strip_local_command_blocks(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<local-command-") {
        out.push_str(&rest[..start]);
        let tag = &rest[start..];
        let Some(open_end) = tag.find('>') else {
            break;
        };
        let open_tag = &tag[..=open_end];
        let suffix = open_tag
            .trim_start_matches("<local-command-")
            .trim_end_matches('>');
        let close = format!("</local-command-{suffix}>");
        let after_open = &tag[open_end + 1..];
        if let Some(close_start) = after_open.find(&close) {
            rest = &after_open[close_start + close.len()..];
        } else {
            rest = "";
        }
    }
    out.push_str(rest);
    out
}

fn collapse_blank_lines(text: &str) -> String {
    let mut lines = Vec::new();
    let mut previous_blank = false;
    for line in text.lines() {
        let trimmed = line.trim_end();
        let is_blank = trimmed.trim().is_empty();
        if is_blank && previous_blank {
            continue;
        }
        lines.push(trimmed.to_string());
        previous_blank = is_blank;
    }
    lines.join("\n").trim().to_string()
}
