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
//! A local drain owns each digest it claims through the shared background-work claim
//! ([`crate::background_work`]): the ledger records the attempt, and the shared claim is what keeps
//! two local drains apart and proves the claiming drain is alive. The claim lasts until the drain
//! records the digest's result or failure, or its process exits. A drain that dies mid-digest —
//! killed, crashed, the machine restarted — leaves the digest interrupted, and the next drain, from
//! any process, offers and claims it at once instead of waiting on a lease.
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

use crate::store::background_claim::{
    ClaimKey, Taken, abandon_kept, fence_kept_in_tx, finish_kept_in_tx, kept_token, release_kept,
};
use crate::tier0::{
    LOCAL_NODE_ID, context_digests, job_exists, ledger_clock_ms, map_engine, understanding_job_id,
};
use crate::{BackgroundWork, BackgroundWorkState, CgError, ContextId, DigestId, Result, Store};

fn digest_claim(context_id: ContextId, digest_id: DigestId) -> ClaimKey {
    ClaimKey::new(
        BackgroundWork::Understanding,
        context_id,
        digest_id.as_uuid(),
    )
}

/// The Context a captured digest belongs to, or `None` when it no longer exists.
fn digest_context(store: &Store, digest_id: DigestId) -> Result<Option<ContextId>> {
    let result = store.read_query(
        "SELECT context_id FROM digests WHERE id = $id",
        &HashMap::from([("id".to_string(), DbValue::Uuid(digest_id.as_uuid()))]),
    )?;
    Ok(result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::Uuid(context_id)) => Some(ContextId::from(*context_id)),
        _ => None,
    }))
}

/// Whether a digest's ledger state lets a local drain claim it: pending, or leased by this machine
/// under a shared claim whose drain is no longer alive (interrupted, or released without a result).
fn locally_claimable(
    store: &Store,
    context_id: ContextId,
    digest_id: DigestId,
    state: &JobState,
) -> Result<bool> {
    Ok(match state {
        JobState::Pending => true,
        JobState::Leased { node_id, .. } if node_id == LOCAL_NODE_ID => {
            digest_claim(context_id, digest_id).state(store)? != BackgroundWorkState::Running
        }
        _ => false,
    })
}

/// Fence a graph write into `digest_id` on this process's claim of it, when it holds one, so a
/// drain that has lost the digest to another can never publish its understanding. A write by a
/// process that holds no claim (a synced-in result being applied) is not fenced here.
pub(crate) fn fence_held_digest_in_tx(
    store: &Store,
    tx: crate::TxId,
    digest_id: DigestId,
) -> Result<()> {
    if kept_token(store.db_path(), digest_id.as_uuid()).is_none() {
        return Ok(());
    }
    let Some(context_id) = digest_context(store, digest_id)? else {
        return Err(lost_digest(digest_id));
    };
    match fence_kept_in_tx(&digest_claim(context_id, digest_id), store, tx)? {
        Some(false) => Err(lost_digest(digest_id)),
        None | Some(true) => Ok(()),
    }
}

/// Give up this process's claim of a digest after the transaction that recorded its result
/// committed. The claim row was deleted by that transaction; this drops the kernel lock.
pub(crate) fn digest_result_committed(store: &Store, digest_id: DigestId) {
    release_kept(store.db_path(), digest_id.as_uuid());
}

fn lost_digest(digest_id: DigestId) -> CgError {
    CgError::Engine(format!(
        "understanding of digest {} belongs to another drain now, so this drain's result was not recorded",
        digest_id.as_uuid()
    ))
}

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
/// success) is skipped even if the ledger job would otherwise read pending. A digest a local drain
/// was understanding when it died is offered again at once; one a live drain holds is not. `limit`
/// caps how many are returned (the full backlog when `None`).
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
        let state = job_state(db, &job_id, ledger_clock_ms()).map_err(map_engine)?;
        if !locally_claimable(store, context_id, digest_id, &state)? {
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

/// Claim a digest for understanding. Returns `true` when this call won it, `false` when it is not
/// claimable here: already understood, failed, cancelled or purged, leased by another machine, or
/// held by a live local drain.
///
/// The shared background-work claim is taken first: it keeps two local drains apart and proves this
/// one is alive for as long as it holds the digest. The ledger then records the attempt
/// (`pending → leased`), which is what other machines see. A digest a dead local drain left leased
/// is claimed again at once under the next attempt number; no lease has to run out.
pub fn claim_digest(store: &Store, digest_id: DigestId) -> Result<bool> {
    let db = store.internal_db();
    let job_id = understanding_job_id(digest_id);
    if !job_exists(store, &job_id)? {
        return Ok(false);
    }
    let Some(context_id) = digest_context(store, digest_id)? else {
        return Ok(false);
    };
    let state = job_state(db, &job_id, ledger_clock_ms()).map_err(map_engine)?;
    if !matches!(state, JobState::Pending)
        && !matches!(&state, JobState::Leased { node_id, .. } if node_id == LOCAL_NODE_ID)
    {
        return Ok(false);
    }
    let tx = store.begin_tx()?;
    let claim = match digest_claim(context_id, digest_id).take_in_tx(store, tx) {
        Ok(Taken::Held(claim)) => claim,
        // A live drain holds it, or another drain claimed it while this one was reading.
        Ok(Taken::Live(_) | Taken::Moved) => {
            let _ = db.rollback(tx);
            return Ok(false);
        }
        Err(error) => {
            let _ = db.rollback(tx);
            return Err(error);
        }
    };
    match db.commit(tx) {
        Ok(()) => {}
        // Another local drain claimed the digest first.
        Err(
            contextdb_core::Error::ConditionalUpdateConflict { .. }
            | contextdb_core::Error::UniqueViolation { .. },
        ) => return Ok(false),
        Err(error) => return Err(map_engine(error)),
    }
    let ledger = next_attempt(db, &job_id).and_then(|attempt| {
        insert_claim(
            db,
            &job_id,
            attempt,
            LOCAL_NODE_ID,
            i64::MAX,
            ledger_clock_ms(),
        )
    });
    match ledger {
        Ok(ClaimInsert::Inserted) => {
            // The claim outlives this call: the drain records the digest's result or failure in a
            // later one.
            claim.keep()?;
            Ok(true)
        }
        Ok(ClaimInsert::AlreadyHeld { .. }) => {
            claim.release(store)?;
            Ok(false)
        }
        // The background claim committed but the ledger refused the attempt: give the claim back
        // so the digest reads as pending, not as interrupted by a drain that never ran.
        Err(error) => {
            let _ = claim.release(store);
            Err(map_engine(error))
        }
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
    let outcome = mark_structured_in_tx(store, tx, context_id, digest_id, model, receipt_json);
    match outcome {
        Ok(()) => match db.commit(tx) {
            Ok(()) => {
                digest_result_committed(store, digest_id);
                Ok(())
            }
            Err(contextdb_core::Error::ConditionalUpdateConflict { .. }) => {
                Err(lost_digest(digest_id))
            }
            Err(error) => Err(map_engine(error)),
        },
        Err(err) => {
            let _ = db.rollback(tx);
            Err(err)
        }
    }
}

/// Write the write-once completion row on the CALLER's open transaction, so an apply's completion
/// marker commits atomically with the structured graph it guards (the apply path stages the graph
/// mutation and this marker on ONE `CaptureTxn` — a crash cannot leave the graph written but the
/// marker missing). Semantics match [`mark_structured`]: write-once (skipped when a completion
/// already exists), keyed by digest id, local-only (never syncs).
pub(crate) fn mark_structured_in_tx(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
    digest_id: DigestId,
    model: &str,
    receipt_json: Option<&str>,
) -> Result<()> {
    let db = store.internal_db();
    // A drain that holds this digest ends its claim in the same transaction as the result, and is
    // refused if another drain has taken the digest over.
    if finish_kept_in_tx(&digest_claim(context_id, digest_id), store, tx)? == Some(false) {
        return Err(lost_digest(digest_id));
    }
    let digest_text = digest_id.to_string();
    let receipt = match receipt_json {
        Some(json) => DbValue::Text(json.to_string()),
        None => DbValue::Null,
    };
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
    // Record the failure against the highest claim attempt, which is the attempt this drain took: a
    // local lease never runs out, so no other machine claims over it. The failure row ends that
    // attempt's ledger lease: below the cap the job falls back to `pending` for a retry; at the cap
    // (`failures >= max_attempts`) it turns terminally `failed`. Only then is the shared claim
    // released, so no other drain claims it in between.
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
    match digest_context(store, digest_id)? {
        Some(context_id) => abandon_kept(&digest_claim(context_id, digest_id), store)?,
        None => release_kept(store.db_path(), digest_id.as_uuid()),
    }
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
    let result = store.read_query(
        "SELECT attempt FROM work_claims WHERE job_id = $job_id",
        &HashMap::from([("job_id".to_string(), DbValue::Text(job_id.to_string()))]),
    )?;
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
    let result = store.read_query(
        "SELECT attempt FROM work_failures WHERE job_id = $job_id",
        &HashMap::from([("job_id".to_string(), DbValue::Text(job_id.to_string()))]),
    )?;
    Ok(result.rows.len() as u32)
}

// Retrieval-touch bumping is owned by `tier0::bump_touch` (the injection/query path calls it); this
// module reads touch counts for drain priority but does not write them.

// --- internal read helpers --------------------------------------------------------------------

fn completion_exists(store: &Store, digest_text: &str) -> Result<bool> {
    let result = store.read_query(
        "SELECT id FROM cg_understanding_completions WHERE id = $id",
        &HashMap::from([("id".to_string(), DbValue::Text(digest_text.to_string()))]),
    )?;
    Ok(!result.rows.is_empty())
}

fn chunk_text_for_digest(store: &Store, digest_text: &str) -> Result<Option<String>> {
    let result = store.read_query(
        "SELECT content FROM cg_source_chunks WHERE digest_id = $digest_id",
        &HashMap::from([(
            "digest_id".to_string(),
            DbValue::Text(digest_text.to_string()),
        )]),
    )?;
    Ok(result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::Text(content)) => Some(content.clone()),
        _ => None,
    }))
}

fn touch_count_for_digest(store: &Store, digest_text: &str) -> Result<u64> {
    let result = store.read_query(
        "SELECT touch_count FROM cg_retrieval_touches WHERE digest_id = $digest_id",
        &HashMap::from([(
            "digest_id".to_string(),
            DbValue::Text(digest_text.to_string()),
        )]),
    )?;
    Ok(result
        .rows
        .first()
        .and_then(|row| match row.first() {
            Some(DbValue::Int64(count)) => Some((*count).max(0) as u64),
            _ => None,
        })
        .unwrap_or(0))
}
