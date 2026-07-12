//! Understanding drain accessors: the store-level seam a worker uses to turn the durable tier-0
//! backlog into structured memory.
//!
//! Tier-0 capture is deterministic and free; it submits one understanding job onto the shipped work
//! ledger per digest (state `pending`) plus the raw source chunk in `cg_source_chunks`. A worker
//! later drains that backlog through a real model: it lists the pending digests by PRIORITY
//! (most-retrieved first, oldest first within equal touch), CLAIMS each one atomically (a ledger
//! claim, `pending → leased`) so two concurrent workers never understand the same digest twice, runs
//! extraction over the stored raw chunk, then records the outcome — a write-once completion on
//! success (the ledger job's computed state reads `structured`), or a recorded failure that re-queues
//! below the attempt cap and turns the job terminally `failed` at the cap.
//!
//! The private `cg_understanding_queue` table is retired: the ledger's computed job state
//! (`work_jobs`/`work_claims`/`work_failures`/`work_cancellations`, plus the retained write-once
//! `cg_understanding_completions` marker) subsumes the queue's pending/in_flight/attempts/status role,
//! so there is no second source of truth to rot. These are pure store accessors: no model, no
//! filesystem, no process control. The worker loop, the lockfile, and the connected-model extractor
//! live in the CLI (the surface caller); vigil, which embeds cg as a library, can drive the same
//! drain in-process by calling these directly.

use std::collections::HashMap;

use contextdb_core::Value as DbValue;
use contextdb_engine::work_ledger::{
    ClaimInsert, JobState, insert_claim, job_state, next_attempt, record_failure,
};

use crate::tier0::{
    LOCAL_NODE_ID, context_digests, job_exists, ledger_clock_ms, map_engine, understanding_job_id,
};
use crate::{ContextId, DigestId, Result, Store};

/// The default cap on model attempts per digest. After this many failures a digest stays `failed`
/// and is never drained again, so a persistently-unparseable chunk cannot wedge the queue.
pub const MAX_UNDERSTANDING_ATTEMPTS: u32 = 3;

/// A pending digest offered to the worker, in priority order, carrying the stored raw chunk the
/// worker feeds back to the model as analysis text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingDigest {
    pub digest_id: DigestId,
    /// The verbatim raw slice kept at capture — what understanding re-analyzes into the ontology.
    pub chunk_text: String,
    /// How many times this digest has been surfaced by retrieval (the priority signal).
    pub touch_count: u64,
}

/// The outcome of recording a model failure against a claimed digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFailure {
    /// The digest's attempt count after this failure.
    pub attempts: u32,
    /// Whether this failure reached the cap, so the digest is now terminally `failed`.
    pub terminal: bool,
}

/// List the pending digests for one context in drain PRIORITY order: highest `touch_count` first,
/// then oldest captured first within equal touch. Each row carries the stored raw chunk text. A
/// digest that already has a completion row (a synced-in completion from another machine, or a local
/// success) is skipped even if the ledger job would otherwise read pending. `limit` caps how many are
/// returned (the full backlog when `None`).
pub fn list_pending_digests(
    store: &Store,
    context_id: ContextId,
    limit: Option<usize>,
) -> Result<Vec<PendingDigest>> {
    let db = store.internal_db();
    let mut candidates: Vec<(i128, u64, PendingDigest)> = Vec::new();
    for (digest_id, capture_order) in context_digests(store, context_id)? {
        let digest_text = digest_id.to_string();
        let job_id = understanding_job_id(digest_id);
        // Only digests captured onto the ledger and still in the pending state are drainable.
        if !job_exists(store, &job_id)? {
            continue;
        }
        if completion_exists(store, &digest_text)? {
            continue;
        }
        if job_state(db, &job_id, ledger_clock_ms()).map_err(map_engine)? != JobState::Pending {
            continue;
        }
        let Some(chunk_text) = chunk_text_for_digest(store, &digest_text)? else {
            // No raw chunk kept (e.g. purged by `cg forget`): nothing to understand.
            continue;
        };
        let touch_count = touch_count_for_digest(store, &digest_text)?;
        candidates.push((
            capture_order,
            touch_count,
            PendingDigest {
                digest_id,
                chunk_text,
                touch_count,
            },
        ));
    }

    // Priority: touch_count DESC, then oldest captured first (capture order ASC) within equal touch.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut out: Vec<PendingDigest> = candidates
        .into_iter()
        .map(|(_, _, pending)| pending)
        .collect();
    if let Some(limit) = limit {
        out.truncate(limit);
    }
    Ok(out)
}

/// Claim a pending digest for understanding via a ledger claim (`pending → leased`). Returns `true`
/// when this call won the claim, `false` when the digest was not pending (already claimed by another
/// worker, already structured, failed, or purged) — the ledger's claim primary key makes the claim
/// safe against a concurrent second worker.
pub fn claim_digest(store: &Store, digest_id: DigestId) -> Result<bool> {
    let db = store.internal_db();
    let job_id = understanding_job_id(digest_id);
    if !job_exists(store, &job_id)? {
        return Ok(false);
    }
    // Only a pending job is claimable — a leased (already in-flight), done, failed, or cancelled job
    // is refused, so a second concurrent worker never re-claims work in flight.
    if job_state(db, &job_id, ledger_clock_ms()).map_err(map_engine)? != JobState::Pending {
        return Ok(false);
    }
    let attempt = next_attempt(db, &job_id).map_err(map_engine)?;
    // The local claim takes an unexpiring lease: the single-node claim is sticky until the worker
    // records success or failure (the fabric worker slice injects a real lease clock for expiry).
    match insert_claim(
        db,
        &job_id,
        attempt,
        LOCAL_NODE_ID,
        i64::MAX,
        ledger_clock_ms(),
    )
    .map_err(map_engine)?
    {
        ClaimInsert::Inserted => Ok(true),
        ClaimInsert::AlreadyHeld { .. } => Ok(false),
    }
}

/// Record a successful understanding of a claimed digest: write the write-once completion row (keyed
/// by digest id, so a re-run or a synced-in peer completion never double-inserts). Once it exists the
/// ledger job's computed state reads `structured`. The completion is a LOCAL apply marker (it does not
/// sync — the structured graph rows are what sync); it guards against re-applying the same result.
pub fn mark_structured(
    store: &Store,
    context_id: ContextId,
    digest_id: DigestId,
    model: &str,
    receipt_json: Option<&str>,
) -> Result<()> {
    let db = store.internal_db();
    let tx = db.begin().map_err(map_engine)?;
    let digest_text = digest_id.to_string();
    let receipt = match receipt_json {
        Some(json) => DbValue::Text(json.to_string()),
        None => DbValue::Null,
    };
    let outcome = (|| -> Result<()> {
        // Write-once: only insert the completion when one does not already exist for this digest.
        // The completion is the durable, synced apply-guard; the ledger job's computed state reads
        // `structured` once this row exists (a synced-in completion or a local success), so a peer
        // that already understood the digest skips it.
        if !completion_exists(store, &digest_text)? {
            db.execute_in_tx(
                tx,
                "INSERT INTO cg_understanding_completions (id, context_id, model, receipt, completed_at) VALUES ($id, $context_id, $model, $receipt, $completed_at)",
                &HashMap::from([
                    ("id".to_string(), DbValue::Text(digest_text.clone())),
                    (
                        "context_id".to_string(),
                        DbValue::Uuid(context_id.as_uuid()),
                    ),
                    ("model".to_string(), DbValue::Text(model.to_string())),
                    ("receipt".to_string(), receipt.clone()),
                    ("completed_at".to_string(), DbValue::TxId(tx)),
                ]),
            )
            .map_err(map_engine)?;
        }
        Ok(())
    })();
    match outcome {
        Ok(()) => db.commit(tx).map_err(map_engine),
        Err(err) => {
            let _ = db.rollback(tx);
            Err(err)
        }
    }
}

/// Record a model failure against a claimed digest: increment `attempts` and stamp `last_error`.
/// Below `max_attempts` the digest is re-queued (`in_flight → pending`) for a later retry; at the cap
/// it is left terminally `failed` so a persistently-failing chunk cannot wedge the drain. Returns the
/// new attempt count and whether the cap was reached.
pub fn mark_failed(
    store: &Store,
    digest_id: DigestId,
    error: &str,
    max_attempts: u32,
) -> Result<QueueFailure> {
    let db = store.internal_db();
    let job_id = understanding_job_id(digest_id);
    // Record the failure against the attempt this worker holds (the highest claim attempt). The
    // failure row releases the lease: below the cap the job's computed state falls back to `pending`
    // for a retry; at the cap (`failures >= max_attempts`) it turns terminally `failed`.
    let attempt = highest_claim_attempt(store, &job_id)?;
    record_failure(
        db,
        &job_id,
        attempt,
        LOCAL_NODE_ID,
        error,
        ledger_clock_ms(),
    )
    .map_err(map_engine)?;
    let new_attempts = failure_count(store, &job_id)?;
    let terminal = new_attempts >= max_attempts.max(1);
    Ok(QueueFailure {
        attempts: new_attempts,
        terminal,
    })
}

/// The highest attempt number any claim row records for a job — the attempt a just-claimed worker
/// currently holds.
fn highest_claim_attempt(store: &Store, job_id: &str) -> Result<i64> {
    let result = store
        .internal_db()
        .execute(
            "SELECT attempt FROM work_claims WHERE job_id = $job_id",
            &HashMap::from([("job_id".to_string(), DbValue::Text(job_id.to_string()))]),
        )
        .map_err(map_engine)?;
    Ok(result
        .rows
        .iter()
        .filter_map(|row| match row.first() {
            Some(DbValue::Int64(attempt)) => Some(*attempt),
            Some(DbValue::Timestamp(attempt)) => Some(*attempt),
            _ => None,
        })
        .max()
        .unwrap_or(1))
}

/// How many failures have been recorded for a job — the attempt count the retired queue tracked in
/// its `attempts` column.
fn failure_count(store: &Store, job_id: &str) -> Result<u32> {
    let result = store
        .internal_db()
        .execute(
            "SELECT attempt FROM work_failures WHERE job_id = $job_id",
            &HashMap::from([("job_id".to_string(), DbValue::Text(job_id.to_string()))]),
        )
        .map_err(map_engine)?;
    Ok(result.rows.len() as u32)
}

// Retrieval-touch bumping is owned by `tier0::bump_touch` (the injection/query path calls it); this
// module reads touch counts for drain priority but does not write them.

// --- internal read helpers --------------------------------------------------------------------

fn completion_exists(store: &Store, digest_text: &str) -> Result<bool> {
    let result = store
        .internal_db()
        .execute(
            "SELECT id FROM cg_understanding_completions WHERE id = $id",
            &HashMap::from([("id".to_string(), DbValue::Text(digest_text.to_string()))]),
        )
        .map_err(map_engine)?;
    Ok(!result.rows.is_empty())
}

fn chunk_text_for_digest(store: &Store, digest_text: &str) -> Result<Option<String>> {
    let result = store
        .internal_db()
        .execute(
            "SELECT content FROM cg_source_chunks WHERE digest_id = $digest_id",
            &HashMap::from([(
                "digest_id".to_string(),
                DbValue::Text(digest_text.to_string()),
            )]),
        )
        .map_err(map_engine)?;
    Ok(result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::Text(content)) => Some(content.clone()),
        _ => None,
    }))
}

fn touch_count_for_digest(store: &Store, digest_text: &str) -> Result<u64> {
    let result = store
        .internal_db()
        .execute(
            "SELECT touch_count FROM cg_retrieval_touches WHERE digest_id = $digest_id",
            &HashMap::from([(
                "digest_id".to_string(),
                DbValue::Text(digest_text.to_string()),
            )]),
        )
        .map_err(map_engine)?;
    Ok(result
        .rows
        .first()
        .and_then(|row| match row.first() {
            Some(DbValue::Int64(count)) => Some((*count).max(0) as u64),
            _ => None,
        })
        .unwrap_or(0))
}
