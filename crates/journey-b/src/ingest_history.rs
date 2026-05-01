//! Plan 22 cg-import-chat-history + Plan 16 cg-digest-pipeline stubs.
//!
//! Per Track 2 stub policy: "stub reads a tiny fixture export"; Plan 16
//! "stub calls a no-op Observer/Reflector that emits one Digest per session".
//!
//! Acceptance contracts mirrored:
//! - `digest_create_and_retrieve_persist` — Digest row persists across reopen.
//! - `digest_dedup_four_way_classification` — re-importing the same session
//!   produces zero new digests; we model this with a `summary_hash UNIQUE`
//!   composite; the second insert returns a duplicate-key error which we
//!   swallow (the same shape the cg crate uses today).

use crate::store::{params, Store};
use crate::types::{ContextId, DigestId};
use contextdb_core::Value;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ChatHistoryFixture {
    pub session_id: String,
    pub turns: Vec<String>,
}

pub fn fixture_with_auth_decision() -> ChatHistoryFixture {
    ChatHistoryFixture {
        session_id: "claude-code-session-2026-04-15-001".into(),
        turns: vec![
            "User: I need to add token validation to the api routes.".into(),
            "Claude: Use auth.py for token validation — it's the existing helper.".into(),
            "User: ok let's go with that.".into(),
        ],
    }
}

/// Run the no-op Observer/Reflector. Produces one Digest per session.
/// Mirrors `digest_create_and_retrieve_persist`. Re-importing the same
/// session collapses on `summary_hash UNIQUE` — mirrors
/// `digest_dedup_four_way_classification`.
pub fn import_chat_history(
    store: &Store,
    fix: &ChatHistoryFixture,
    ctx: ContextId,
) -> ImportResult {
    let summary = format!("session-summary::{}", fix.session_id);
    let summary_hash = format!("{:x}", hash_str(&summary));
    let id = Uuid::now_v7();
    let vec = store.embedder.embed(&summary);

    // Pre-insert SELECT — engine gap N1 ConcurrentUniqueConstraintEnforcement
    // surfaces in the SEQUENTIAL form too: a duplicate INSERT against a
    // UNIQUE column returns `Ok` with rows_affected=0 (the v0.3.4 duplicate
    // insert no-op semantic), not `Err::UniqueViolation`. The cg crate's
    // `intention_identity_collision_rejected` pattern uses pre-INSERT
    // SELECT to detect — we mirror it here. The contention case (multiple
    // sessions importing the same export concurrently) is the unsolved part
    // of N1 and is flagged in findings.
    let existing = store
        .db
        .execute(
            "SELECT id FROM digests WHERE summary_hash = $sh",
            &params(vec![("sh", Value::Text(summary_hash.clone()))]),
        )
        .unwrap();
    if !existing.rows.is_empty() {
        return ImportResult {
            digest_id: None,
            duplicate: true,
        };
    }

    let tx = store.db.begin();
    let res = store.db.execute_in_tx(
        tx,
        "INSERT INTO digests (id, source, session_ref, summary, summary_hash, context_id, created_at, summary_vector)
         VALUES ($id, 'claude-code', $sref, $sum, $sh, $cx, $t, $v)",
        &params(vec![
            ("id", Value::Uuid(id)),
            ("sref", Value::Text(fix.session_id.clone())),
            ("sum", Value::Text(summary)),
            ("sh", Value::Text(summary_hash)),
            ("cx", Value::Uuid(ctx)),
            ("t", Value::TxId(tx)),
            ("v", Value::Vector(vec)),
        ]),
    );
    match res {
        Ok(_) => {
            store.db.commit(tx).unwrap();
            ImportResult {
                digest_id: Some(id),
                duplicate: false,
            }
        }
        Err(_e) => {
            let _ = store.db.rollback(tx);
            ImportResult {
                digest_id: None,
                duplicate: true,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImportResult {
    pub digest_id: Option<DigestId>,
    pub duplicate: bool,
}

fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
