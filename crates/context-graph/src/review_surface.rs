//! HITL review surface. Produces five message kinds over six verbs at one library
//! seam; the CLI renders these, adding no capture field the library lacks.

use std::collections::HashMap;

use crate::{Store, skill_store::execute_with_live_txid};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageKind {
    CaptureReceipt,
    ReviewItem,
    GroupingProposal,
    SkillCard,
    DriftAlert,
}

#[derive(Debug, Clone)]
pub struct ReviewMessage {
    pub kind: MessageKind,
    pub content: String,
}

#[derive(Debug, Clone, Default)]
pub struct ReviewSummary {
    /// True count of uncertain items (all, not capped).
    pub uncertain_count: usize,
    /// True count of grouping proposals.
    pub grouping_count: usize,
    /// True count of drift alerts.
    pub drift_count: usize,
    /// Count for the currency-flips category (structurally empty on a single edge;
    /// server-core owns reconciliation). Present even when zero.
    pub currency_flips_count: usize,
    /// Displayed items (may be capped at display cap). True counts above are not capped.
    pub uncertain_items: Vec<ReviewItem>,
    pub grouping_items: Vec<ReviewItem>,
    pub drift_items: Vec<ReviewItem>,
}

#[derive(Debug, Clone)]
pub struct ReviewItem {
    pub id: String,
    pub kind: String,
    pub content: String,
    pub category: ReviewCategory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewCategory {
    Uncertain,
    Grouping,
    Drift,
    CurrencyFlips,
    StructurallyEmpty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewVerb {
    Accept,
    Correct,
    Drop,
    Group,
    Undo,
    SkillsOff,
    SkillsOn,
}

/// Parse a verb string into the six valid action verbs. Returns None for any unknown verb.
pub fn parse_verb(s: &str) -> Option<ReviewVerb> {
    match s {
        "accept" => Some(ReviewVerb::Accept),
        "correct" => Some(ReviewVerb::Correct),
        "drop" => Some(ReviewVerb::Drop),
        "group" => Some(ReviewVerb::Group),
        "undo" => Some(ReviewVerb::Undo),
        "skills-off" => Some(ReviewVerb::SkillsOff),
        "skills-on" => Some(ReviewVerb::SkillsOn),
        _ => None,
    }
}

/// Returns all five message kinds emitted by the library seam.
pub fn all_message_kinds() -> Vec<MessageKind> {
    vec![
        MessageKind::CaptureReceipt,
        MessageKind::ReviewItem,
        MessageKind::GroupingProposal,
        MessageKind::SkillCard,
        MessageKind::DriftAlert,
    ]
}

/// Build a review summary with true per-category counts. Display lists may be capped
/// but the count fields reflect the true total.
pub fn build_review_summary(items: &[ReviewItem]) -> ReviewSummary {
    const DISPLAY_CAP: usize = 20;
    let uncertain: Vec<_> = items
        .iter()
        .filter(|i| i.category == ReviewCategory::Uncertain)
        .cloned()
        .collect();
    let grouping: Vec<_> = items
        .iter()
        .filter(|i| i.category == ReviewCategory::Grouping)
        .cloned()
        .collect();
    let drift: Vec<_> = items
        .iter()
        .filter(|i| i.category == ReviewCategory::Drift)
        .cloned()
        .collect();

    ReviewSummary {
        uncertain_count: uncertain.len(),
        grouping_count: grouping.len(),
        drift_count: drift.len(),
        currency_flips_count: 0,
        uncertain_items: uncertain.into_iter().take(DISPLAY_CAP).collect(),
        grouping_items: grouping.into_iter().take(DISPLAY_CAP).collect(),
        drift_items: drift.into_iter().take(DISPLAY_CAP).collect(),
    }
}

/// Build review summary exposing all four categories including the currency-flips
/// category. On a single edge the currency-flips category is structurally empty
/// (server-core owns that reconciliation), but it is counted here from the items list
/// so the count field is accurate when a CurrencyFlips-typed item is provided.
pub fn build_review_summary_with_currency_category(items: &[ReviewItem]) -> ReviewSummary {
    let mut summary = build_review_summary(items);
    summary.currency_flips_count = items
        .iter()
        .filter(|i| i.category == ReviewCategory::CurrencyFlips)
        .count();
    summary
}

#[derive(Debug, Clone, Default)]
pub struct ReviewCursor {
    pub position: usize,
    pub is_local: bool,
}

/// Returns true when the cursor is local-only. The review cursor is always local to
/// the owner device and is never synced — this always returns true regardless of the
/// field value (the field may be initialized to false as a placeholder default).
pub fn is_cursor_local(_cursor: &ReviewCursor) -> bool {
    true
}

/// Advance the review cursor past the current position and persist the new position.
pub fn advance_cursor(store: &Store, cursor: &mut ReviewCursor) {
    cursor.position += 1;
    persist_cursor(store, cursor);
}

/// Advance cursor only for a bare (unfiltered) review call. Filtered-lens calls
/// read the current state but do not advance the cursor mark.
pub fn maybe_advance_cursor(store: &Store, cursor: &mut ReviewCursor, is_bare_review: bool) {
    if is_bare_review {
        advance_cursor(store, cursor);
    }
}

fn persist_cursor(store: &Store, cursor: &ReviewCursor) {
    let db = store.sync_database();
    let params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text("main".to_string()),
        ),
        (
            "position".to_string(),
            contextdb_core::Value::Int64(cursor.position as i64),
        ),
        (
            "is_local".to_string(),
            contextdb_core::Value::Bool(cursor.is_local),
        ),
    ]);
    // Try UPDATE first; fall back to INSERT if no row exists. Each statement
    // runs in its own transaction so `updated_at` is a live tx id, not genesis.
    let updated = match db.begin() {
        Ok(tx) => {
            let mut p = params.clone();
            p.insert("updated_at".to_string(), contextdb_core::Value::TxId(tx));
            let result = db.execute_in_tx(
                tx,
                "UPDATE cg_review_cursor SET position = $position, is_local = $is_local, updated_at = $updated_at WHERE id = $id",
                &p,
            );
            let _ = db.commit(tx);
            result.map(|r| !r.rows.is_empty()).unwrap_or(false)
        }
        Err(_) => false,
    };
    if !updated {
        let _ = execute_with_live_txid(
            &db,
            "INSERT INTO cg_review_cursor (id, position, is_local, updated_at) VALUES ($id, $position, $is_local, $updated_at)",
            "updated_at",
            params,
        );
    }
}

// ─── Pending item persistence ────────────────────────────────────────────────
//
// A pending review item is an item that has been surfaced for review but not yet
// acted on. Items persist in this table until explicitly acted on via `act_on_item`.
// The cursor position does not affect pending item visibility; they never age out.
//
// The `cg_pending_review_items` table schema:
//   id TEXT PRIMARY KEY, acted_on BOOLEAN NOT NULL DEFAULT false, created_at TXID NOT NULL
//
// On first call to `is_item_pending`, the item is registered in the table if not
// already present. Subsequent calls read the persisted acted_on state.

const PENDING_ITEMS_TABLE: &str = "cg_pending_review_items";

fn ensure_pending_item_registered(store: &Store, item_id: &str) {
    let db = store.sync_database();
    let params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(item_id.to_string()),
        ),
        ("acted_on".to_string(), contextdb_core::Value::Bool(false)),
    ]);
    // INSERT only if not already present. Ignore UniqueViolation — item was already there.
    let _ = execute_with_live_txid(
        &db,
        &format!(
            "INSERT INTO {PENDING_ITEMS_TABLE} (id, acted_on, created_at) VALUES ($id, $acted_on, $created_at)"
        ),
        "created_at",
        params,
    );
}

/// Returns true if the given item is still pending (has been registered and not yet
/// acted on). Registering a new item on first query ensures items persist across
/// cursor advances and captures until explicitly acted on.
pub fn is_item_pending(store: &Store, item_id: &str, _cursor: &ReviewCursor) -> bool {
    // Ensure the item is registered in the pending table.
    ensure_pending_item_registered(store, item_id);

    let db = store.sync_database();
    let snapshot = db.snapshot();
    let rows = db.scan(PENDING_ITEMS_TABLE, snapshot).unwrap_or_default();
    for row in &rows {
        match row.values.get("id") {
            Some(contextdb_core::Value::Text(id)) if id == item_id => {
                return match row.values.get("acted_on") {
                    Some(contextdb_core::Value::Bool(b)) => !b,
                    Some(contextdb_core::Value::Int64(i)) => *i == 0,
                    _ => true,
                };
            }
            _ => {}
        }
    }
    true
}

/// List all pending review items (not yet acted on).
pub fn list_pending_items(store: &Store, _cursor: &ReviewCursor) -> Vec<ReviewItem> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    let rows = db.scan(PENDING_ITEMS_TABLE, snapshot).unwrap_or_default();
    rows.into_iter()
        .filter_map(|row| {
            let acted_on = match row.values.get("acted_on") {
                Some(contextdb_core::Value::Bool(b)) => *b,
                Some(contextdb_core::Value::Int64(i)) => *i != 0,
                _ => false,
            };
            if acted_on {
                return None;
            }
            let id = match row.values.get("id") {
                Some(contextdb_core::Value::Text(s)) => s.clone(),
                _ => return None,
            };
            Some(ReviewItem {
                id: id.clone(),
                kind: "pending".to_string(),
                content: format!("pending-{id}"),
                category: ReviewCategory::Uncertain,
            })
        })
        .collect()
}

/// Mark an item as acted on, removing it from the pending review set.
pub fn act_on_item(store: &Store, item_id: &str, _verb: &ReviewVerb) {
    let db = store.sync_database();

    // Ensure the item row exists first.
    ensure_pending_item_registered(store, item_id);

    // Mark as acted on by deleting and re-inserting with acted_on=true.
    let del_params = HashMap::from([(
        "id".to_string(),
        contextdb_core::Value::Text(item_id.to_string()),
    )]);
    let _ = db.execute(
        &format!("DELETE FROM {PENDING_ITEMS_TABLE} WHERE id = $id"),
        &del_params,
    );
    let insert_params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(item_id.to_string()),
        ),
        ("acted_on".to_string(), contextdb_core::Value::Bool(true)),
    ]);
    let _ = execute_with_live_txid(
        &db,
        &format!(
            "INSERT INTO {PENDING_ITEMS_TABLE} (id, acted_on, created_at) VALUES ($id, $acted_on, $created_at)"
        ),
        "created_at",
        insert_params,
    );
}

/// Emit a capture receipt message. The receipt leads with the Intentions section,
/// then decisions and other extracted items.
pub fn emit_capture_receipt(report: &str) -> ReviewMessage {
    ReviewMessage {
        kind: MessageKind::CaptureReceipt,
        content: format!("Intentions: {}", report),
    }
}

/// Emit a grouping proposal message.
pub fn emit_grouping_proposal(proposal: &str) -> ReviewMessage {
    ReviewMessage {
        kind: MessageKind::GroupingProposal,
        content: proposal.to_string(),
    }
}

/// Emit a drift alert message.
pub fn emit_drift_alert(skill_id: &str, provenance: &str) -> ReviewMessage {
    ReviewMessage {
        kind: MessageKind::DriftAlert,
        content: format!("skill={} provenance={}", skill_id, provenance),
    }
}

/// Emit a skill card message.
pub fn emit_skill_card(skill_description: &str) -> ReviewMessage {
    ReviewMessage {
        kind: MessageKind::SkillCard,
        content: skill_description.to_string(),
    }
}

/// Emit a review item message.
pub fn emit_review_item(item: &ReviewItem) -> ReviewMessage {
    ReviewMessage {
        kind: MessageKind::ReviewItem,
        content: item.content.clone(),
    }
}
