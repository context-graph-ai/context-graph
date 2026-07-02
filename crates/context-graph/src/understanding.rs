//! Understanding-queue drain accessors: the store-level seam a worker uses to turn the durable
//! tier-0 backlog into structured memory.
//!
//! Tier-0 capture is deterministic and free; it leaves one `pending` row per digest in
//! `cg_understanding_queue` plus the raw source chunk in `cg_source_chunks`. A worker later drains
//! that backlog through a real model: it lists the pending digests by PRIORITY (most-retrieved
//! first, oldest first within equal touch), CLAIMS each one atomically (`pending → in_flight`) so
//! two concurrent workers never understand the same digest twice, runs extraction over the stored
//! raw chunk, then records the outcome — `structured` plus a write-once completion on success, or an
//! attempt increment with a terminal cap on model failure.
//!
//! These are pure store accessors: no model, no filesystem, no process control. The worker loop, the
//! lockfile, and the connected-model extractor live in the CLI (the surface caller); vigil, which
//! embeds cg as a library, can drive the same drain in-process by calling these directly.

use std::collections::HashMap;

use contextdb_core::Value as DbValue;

use crate::tier0::map_engine;
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
/// then oldest enqueued first within equal touch. Each row carries the stored raw chunk text. A
/// digest that already has a completion row (a synced-in completion from another machine, or a local
/// success) is skipped even if its local queue row still reads `pending`. `limit` caps how many are
/// returned (the full backlog when `None`).
pub fn list_pending_digests(
    store: &Store,
    context_id: ContextId,
    limit: Option<usize>,
) -> Result<Vec<PendingDigest>> {
    let db = store.internal_db();
    let rows = db
        .execute(
            "SELECT digest_id, enqueued_at FROM cg_understanding_queue WHERE context_id = $context_id AND status = $status",
            &HashMap::from([
                (
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                ),
                ("status".to_string(), DbValue::Text("pending".to_string())),
            ]),
        )
        .map_err(map_engine)?;

    let mut candidates: Vec<(i128, u64, PendingDigest)> = Vec::new();
    for row in &rows.rows {
        let digest_text = match row.first() {
            Some(DbValue::Text(id)) => id.clone(),
            _ => continue,
        };
        let enqueued_order = row.get(1).map(order_key).unwrap_or(0);
        if completion_exists(store, &digest_text)? {
            continue;
        }
        let Some(chunk_text) = chunk_text_for_digest(store, &digest_text)? else {
            // No raw chunk kept (e.g. purged by `cg forget`): nothing to understand.
            continue;
        };
        let digest_id = match parse_digest_id(&digest_text) {
            Some(id) => id,
            None => continue,
        };
        let touch_count = touch_count_for_digest(store, &digest_text)?;
        candidates.push((
            enqueued_order,
            touch_count,
            PendingDigest {
                digest_id,
                chunk_text,
                touch_count,
            },
        ));
    }

    // Priority: touch_count DESC, then oldest enqueued first (enqueue order ASC) within equal touch.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut out: Vec<PendingDigest> = candidates.into_iter().map(|(_, _, pending)| pending).collect();
    if let Some(limit) = limit {
        out.truncate(limit);
    }
    Ok(out)
}

/// Atomically claim a pending digest for understanding (`pending → in_flight`). Returns `true` when
/// this call won the claim, `false` when the digest was not pending (already claimed by another
/// worker, already structured, or purged) — the guard on `status = 'pending'` makes the claim safe
/// against a concurrent second worker.
pub fn claim_digest(store: &Store, digest_id: DigestId) -> Result<bool> {
    let db = store.internal_db();
    let tx = db.begin().map_err(map_engine)?;
    let outcome = db.execute_in_tx(
        tx,
        "UPDATE cg_understanding_queue SET status = 'in_flight', updated_at = $updated_at WHERE digest_id = $digest_id AND status = 'pending'",
        &HashMap::from([
            (
                "digest_id".to_string(),
                DbValue::Text(digest_id.to_string()),
            ),
            ("updated_at".to_string(), DbValue::TxId(tx)),
        ]),
    );
    match outcome {
        Ok(result) => {
            db.commit(tx).map_err(map_engine)?;
            Ok(result.rows_affected == 1)
        }
        Err(err) => {
            let _ = db.rollback(tx);
            Err(map_engine(err))
        }
    }
}

/// Record a successful understanding of a claimed digest: flip the queue row to `structured` and
/// write the write-once completion row (keyed by digest id, so a re-run or a synced-in peer
/// completion never double-inserts). The completion syncs (`InsertIfNotExists`), so every edge learns
/// this digest is understood and skips it.
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
        db.execute_in_tx(
            tx,
            "UPDATE cg_understanding_queue SET status = 'structured', updated_at = $updated_at WHERE digest_id = $digest_id",
            &HashMap::from([
                ("digest_id".to_string(), DbValue::Text(digest_text.clone())),
                ("updated_at".to_string(), DbValue::TxId(tx)),
            ]),
        )
        .map_err(map_engine)?;
        // Write-once: only insert the completion when one does not already exist for this digest.
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
    let digest_text = digest_id.to_string();
    let attempts = current_attempts(store, &digest_text)?;
    let new_attempts = attempts.saturating_add(1);
    let terminal = new_attempts >= max_attempts.max(1);
    let status = if terminal { "failed" } else { "pending" };
    let tx = db.begin().map_err(map_engine)?;
    let outcome = db.execute_in_tx(
        tx,
        "UPDATE cg_understanding_queue SET status = $status, attempts = $attempts, last_error = $last_error, updated_at = $updated_at WHERE digest_id = $digest_id AND status = 'in_flight'",
        &HashMap::from([
            ("status".to_string(), DbValue::Text(status.to_string())),
            ("attempts".to_string(), DbValue::Int64(i64::from(new_attempts))),
            ("last_error".to_string(), DbValue::Text(error.to_string())),
            ("updated_at".to_string(), DbValue::TxId(tx)),
            ("digest_id".to_string(), DbValue::Text(digest_text)),
        ]),
    );
    match outcome {
        Ok(_) => {
            db.commit(tx).map_err(map_engine)?;
            Ok(QueueFailure {
                attempts: new_attempts,
                terminal,
            })
        }
        Err(err) => {
            let _ = db.rollback(tx);
            Err(map_engine(err))
        }
    }
}

/// Increment a digest's retrieval touch count (the drain-priority signal). Idempotent-safe under
/// re-open: the count is persisted. Public so the injection/query path can bump a digest each time it
/// is surfaced, promoting frequently-retrieved memory to the front of the understanding queue.
pub fn bump_touch(store: &Store, digest_id: DigestId) -> Result<()> {
    let db = store.internal_db();
    let digest_text = digest_id.to_string();
    let current = touch_count_for_digest(store, &digest_text)?;
    let context_id = context_id_for_digest(store, &digest_text)?;
    let tx = db.begin().map_err(map_engine)?;
    let outcome = (|| -> Result<()> {
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
                    ("digest_id".to_string(), DbValue::Text(digest_text.clone())),
                ]),
            )
            .map_err(map_engine)?;
        if updated.rows_affected == 0 {
            let Some(context_id) = context_id else {
                // No digest/context to attribute the touch to; nothing to bump.
                return Ok(());
            };
            db.execute_in_tx(
                tx,
                "INSERT INTO cg_retrieval_touches (digest_id, context_id, touch_count, last_touched_at) VALUES ($digest_id, $context_id, $touch_count, $last_touched_at)",
                &HashMap::from([
                    ("digest_id".to_string(), DbValue::Text(digest_text.clone())),
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
        Ok(()) => db.commit(tx).map_err(map_engine),
        Err(err) => {
            let _ = db.rollback(tx);
            Err(err)
        }
    }
}

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
            &HashMap::from([("digest_id".to_string(), DbValue::Text(digest_text.to_string()))]),
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
            &HashMap::from([("digest_id".to_string(), DbValue::Text(digest_text.to_string()))]),
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

fn current_attempts(store: &Store, digest_text: &str) -> Result<u32> {
    let result = store
        .internal_db()
        .execute(
            "SELECT attempts FROM cg_understanding_queue WHERE digest_id = $digest_id",
            &HashMap::from([("digest_id".to_string(), DbValue::Text(digest_text.to_string()))]),
        )
        .map_err(map_engine)?;
    Ok(result
        .rows
        .first()
        .and_then(|row| match row.first() {
            Some(DbValue::Int64(attempts)) => Some((*attempts).max(0) as u32),
            _ => None,
        })
        .unwrap_or(0))
}

fn context_id_for_digest(store: &Store, digest_text: &str) -> Result<Option<ContextId>> {
    let result = store
        .internal_db()
        .execute(
            "SELECT context_id FROM cg_understanding_queue WHERE digest_id = $digest_id",
            &HashMap::from([("digest_id".to_string(), DbValue::Text(digest_text.to_string()))]),
        )
        .map_err(map_engine)?;
    Ok(result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::Uuid(id)) => Some(ContextId::from(*id)),
        _ => None,
    }))
}

fn parse_digest_id(digest_text: &str) -> Option<DigestId> {
    uuid::Uuid::parse_str(digest_text).ok().map(DigestId::from)
}

fn order_key(value: &DbValue) -> i128 {
    match value {
        DbValue::TxId(tx) => i128::from(tx.0),
        DbValue::Int64(value) => i128::from(*value),
        _ => 0,
    }
}
