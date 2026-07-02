//! Tier-0 deterministic capture: the default `cg record` / `cg import` front door.
//!
//! Tier-0 capture writes a verbatim digest, its local embedding, the kept raw source chunk, and a
//! durable understanding-queue row — all in ONE atomic transaction, with ZERO model calls. The
//! model never runs on this path; understanding (structuring a chunk into the ontology) is deferred
//! to a later worker that drains the queue. This is the "day one import everything free; overnight
//! structure locally" contract: capture is instant and costless, so a developer can fill cg with a
//! session's history without paying a token or waiting on a provider.
//!
//! Raw source chunks NEVER leave the machine (`cg_source_chunks` syncs `None`) — the graph is the
//! durable, synced asset; raw is local understanding substrate. Because raw is kept, this module
//! also owns the purge path (`forget_raw`) and the retention window (`raw_retention_days`): keeping
//! raw without a way to delete it is not shippable.

use std::collections::HashMap;

use chrono::Utc;
use contextdb_core::{Value as DbValue, VectorIndexRef};
use sha2::{Digest as _, Sha256};

use crate::embedding_space::BuiltinSpaceRegistry;
use crate::ingest::split_transcript_into_windows;
use crate::{
    CgError, Context, ContextId, CreateContext, DigestId, EmbeddingId, EvidenceProducer, Result,
    Store,
};

/// Default byte window a tier-0 capture splits a transcript into when the caller does not resolve
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
/// here — only the tier-0 truth: digests written, raw kept, what was skipped, and the always-zero
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
    /// Pending understanding-queue rows in the capture context after this capture.
    pub pending_after: usize,
    /// Always zero — tier-0 never calls a model.
    pub model_calls: usize,
    /// The session handle these digests were grouped under.
    pub session_ref: String,
    /// The ids of the digests written this capture (skips excluded).
    pub digest_ids: Vec<DigestId>,
}

/// Understanding-queue counts for one context, by state.
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

/// Capture a conversation deterministically: verbatim digest + embedding + raw chunk + queue row
/// per chunk, all committed in ONE atomic transaction, ZERO model calls. Re-running the same
/// capture is idempotent — a chunk whose deterministic digest already exists is skipped, writing no
/// new rows.
pub fn capture_tier0(store: &Store, request: Tier0CaptureRequest) -> Result<Tier0CaptureReport> {
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
    let outcome = write_tier0_chunks(store, tx, context.id, &request, &chunks, &mut report);
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

fn write_tier0_chunks(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
    request: &Tier0CaptureRequest,
    chunks: &[String],
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
        write_queue_row(store, tx, context_id, digest_id)?;
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
                ("kind".to_string(), DbValue::Text("verbatim".to_string())),
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

fn write_queue_row(
    store: &Store,
    tx: crate::TxId,
    context_id: ContextId,
    digest_id: DigestId,
) -> Result<()> {
    store
        .internal_db()
        .execute_in_tx(
            tx,
            "INSERT INTO cg_understanding_queue (digest_id, context_id, status, attempts, last_error, enqueued_at, updated_at) VALUES ($digest_id, $context_id, $status, $attempts, $last_error, $enqueued_at, $updated_at)",
            &HashMap::from([
                (
                    "digest_id".to_string(),
                    DbValue::Text(digest_id.to_string()),
                ),
                (
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                ),
                ("status".to_string(), DbValue::Text("pending".to_string())),
                ("attempts".to_string(), DbValue::Int64(0)),
                ("last_error".to_string(), DbValue::Null),
                ("enqueued_at".to_string(), DbValue::TxId(tx)),
                ("updated_at".to_string(), DbValue::TxId(tx)),
            ]),
        )
        .map_err(map_engine)?;
    Ok(())
}

/// Count understanding-queue rows in one context, by state.
pub fn queue_counts(store: &Store, context_id: ContextId) -> Result<QueueCounts> {
    // The engine has no GROUP BY / aggregate support, so tally the status column in Rust.
    let result = store
        .internal_db()
        .execute(
            "SELECT status FROM cg_understanding_queue WHERE context_id = $context_id",
            &HashMap::from([(
                "context_id".to_string(),
                DbValue::Uuid(context_id.as_uuid()),
            )]),
        )
        .map_err(map_engine)?;
    let mut counts = QueueCounts::default();
    for row in &result.rows {
        let status = match row.first() {
            Some(DbValue::Text(status)) => status.as_str(),
            _ => continue,
        };
        match status {
            "pending" => counts.pending += 1,
            "in_flight" => counts.in_flight += 1,
            "structured" => counts.structured += 1,
            "failed" => counts.failed += 1,
            "skipped" => counts.skipped += 1,
            _ => {}
        }
    }
    Ok(counts)
}

/// The `kind` marker on a digest (e.g. "verbatim" for a tier-0 digest), or `None` if absent.
pub fn digest_kind(store: &Store, digest_id: DigestId) -> Result<Option<String>> {
    let result = store
        .internal_db()
        .execute(
            "SELECT kind FROM digests WHERE id = $id",
            &HashMap::from([("id".to_string(), DbValue::Uuid(digest_id.as_uuid()))]),
        )
        .map_err(map_engine)?;
    Ok(result.rows.first().and_then(|row| match row.first() {
        Some(DbValue::Text(kind)) => Some(kind.clone()),
        _ => None,
    }))
}

/// The raw-source footprint (kept chunks + total bytes) for one context.
pub fn raw_footprint(store: &Store, context_id: ContextId) -> Result<RawFootprint> {
    // The engine has no aggregate support, so tally row bytes in Rust.
    let result = store
        .internal_db()
        .execute(
            "SELECT byte_len FROM cg_source_chunks WHERE context_id = $context_id",
            &HashMap::from([(
                "context_id".to_string(),
                DbValue::Uuid(context_id.as_uuid()),
            )]),
        )
        .map_err(map_engine)?;
    let mut footprint = RawFootprint::default();
    for row in &result.rows {
        footprint.chunk_count += 1;
        if let Some(DbValue::Int64(bytes)) = row.first() {
            footprint.bytes += (*bytes).max(0) as u64;
        }
    }
    Ok(footprint)
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
            let updated = db
                .execute_in_tx(
                    tx,
                    "UPDATE cg_understanding_queue SET status = 'skipped', updated_at = $updated_at WHERE digest_id = $digest_id AND status = 'pending'",
                    &HashMap::from([
                        (
                            "digest_id".to_string(),
                            DbValue::Text(target.digest_id.clone()),
                        ),
                        ("updated_at".to_string(), DbValue::TxId(tx)),
                    ]),
                )
                .map_err(map_engine)?;
            report.queue_rows_skipped += updated.rows_affected as usize;
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
    let result = store
        .internal_db()
        .execute(&sql, &params)
        .map_err(map_engine)?;
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
    let result = store
        .internal_db()
        .execute(&sql, &params)
        .map_err(map_engine)?;
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
    let result = store
        .internal_db()
        .execute(
            "SELECT value FROM cg_local_settings WHERE key = $key",
            &HashMap::from([("key".to_string(), DbValue::Text(key.to_string()))]),
        )
        .map_err(map_engine)?;
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
    let result = store
        .internal_db()
        .execute(
            "SELECT id FROM digests WHERE id = $id AND context_id = $context_id",
            &HashMap::from([
                ("id".to_string(), DbValue::Uuid(digest_uuid)),
                (
                    "context_id".to_string(),
                    DbValue::Uuid(context_id.as_uuid()),
                ),
            ]),
        )
        .map_err(map_engine)?;
    Ok(!result.rows.is_empty())
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

fn map_engine(err: contextdb_core::Error) -> CgError {
    CgError::Engine(format!("unexpected engine error: {err}"))
}

// --- salience-clean text helpers --------------------------------------------------------------

/// Strip `<prefix ...>...</close>` blocks (the open tag may carry attributes). Mirrors the import
/// parser's noise stripping so tier-0 cleaning recognizes the same harness markers.
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
