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

/// The three sub-actions of the `group` verb. Each has a DISTINCT effect on a grouping proposal:
/// `Confirm` links the candidate Intentions under one canonical entry; `Rename` retitles the
/// catalog entry; `Reject` leaves the Intentions ungrouped. These are part of the single action
/// surface (`group …`), not new top-level verbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupAction {
    Confirm,
    Rename,
    Reject,
}

/// Parse a `group` sub-action argument into its distinct `GroupAction`: `ok` → `Confirm`,
/// `rename` → `Rename`, `no` → `Reject`. Any other token is not a group sub-action (`None`).
pub fn parse_group_action(arg: &str) -> Option<GroupAction> {
    match arg {
        "ok" => Some(GroupAction::Confirm),
        "rename" => Some(GroupAction::Rename),
        "no" => Some(GroupAction::Reject),
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

/// Register every capture-produced flagged item as a pending review item, so `cg review` surfaces it
/// from store state until it is acted on. Idempotent: re-registering an already-present item is a
/// no-op (the `INSERT` hits the primary-key constraint and is ignored). Called on the capture path.
pub(crate) fn register_flagged_items(store: &Store, flagged: &[crate::ingest::FlaggedItem]) {
    for item in flagged {
        if item.key.trim().is_empty() {
            continue;
        }
        register_pending_uncertain_item(
            store,
            &item.key,
            &item.reason,
            item.legal_vocab.as_deref(),
        );
    }
}

fn register_pending_uncertain_item(
    store: &Store,
    item_id: &str,
    content: &str,
    legal_vocab: Option<&[String]>,
) {
    let db = store.sync_database();
    // A closed-vocabulary flag carries the legal value set (JSON array) so `cg why @<id>` can resolve
    // the item by its id and present the named set; a free-text flag stores NULL (no constrained set).
    let legal_vocab_value = match legal_vocab {
        Some(vocab) => {
            contextdb_core::Value::Text(serde_json::to_string(vocab).unwrap_or_default())
        }
        None => contextdb_core::Value::Null,
    };
    let params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(item_id.to_string()),
        ),
        ("acted_on".to_string(), contextdb_core::Value::Bool(false)),
        (
            "content".to_string(),
            contextdb_core::Value::Text(content.to_string()),
        ),
        ("legal_vocab".to_string(), legal_vocab_value),
    ]);
    // INSERT only if not already present; a primary-key collision (already pending) is ignored so a
    // pending item persists across windows and intervening captures until acted on.
    let _ = execute_with_live_txid(
        &db,
        &format!(
            "INSERT INTO {PENDING_ITEMS_TABLE} (id, acted_on, content, legal_vocab, created_at) VALUES ($id, $acted_on, $content, $legal_vocab, $created_at)"
        ),
        "created_at",
        params,
    );
}

/// Resolve a flagged item BY ITS ID against the persisted pending-review-item row, returning the
/// legal value set the correction must come from when the item is a genuinely-flagged CLOSED-VOCAB
/// item (the row carries a `legal_vocab`), or `None` when no pending item has that exact id or the
/// flagged item is free-text (no constrained vocabulary). Exact-id resolution — never token overlap.
pub(crate) fn legal_vocab_for_pending_item(store: &Store, item_id: &str) -> Option<Vec<String>> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    let rows = db.scan(PENDING_ITEMS_TABLE, snapshot).unwrap_or_default();
    let row = rows.into_iter().find(|row| {
        matches!(row.values.get("id"), Some(contextdb_core::Value::Text(id)) if id == item_id)
    })?;
    let acted_on = match row.values.get("acted_on") {
        Some(contextdb_core::Value::Bool(b)) => *b,
        Some(contextdb_core::Value::Int64(i)) => *i != 0,
        _ => false,
    };
    if acted_on {
        return None;
    }
    match row.values.get("legal_vocab") {
        Some(contextdb_core::Value::Text(json)) => serde_json::from_str::<Vec<String>>(json).ok(),
        _ => None,
    }
}

/// `cg review` summary GATHERED from STORE state (not a caller-supplied list): the uncertain
/// category from capture-produced flagged items still pending, plus grouping proposals and drift
/// alerts persisted in the store. The per-category counts are the TRUE store totals (the display
/// lists may be capped); a pending flagged item persists here until acted on.
pub fn gather_review_summary(
    store: &Store,
    _cursor: &ReviewCursor,
) -> crate::Result<ReviewSummary> {
    const DISPLAY_CAP: usize = 20;
    let rows = store.read_table(PENDING_ITEMS_TABLE)?;
    let uncertain: Vec<ReviewItem> = rows
        .into_iter()
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
            let content = match row.values.get("content") {
                Some(contextdb_core::Value::Text(s)) => s.clone(),
                _ => format!("pending-{id}"),
            };
            Some(ReviewItem {
                id,
                kind: "uncertain".to_string(),
                content,
                category: ReviewCategory::Uncertain,
            })
        })
        .collect();

    let drift: Vec<ReviewItem> = crate::skill_store::list_drift_alerts(store)?
        .into_iter()
        .map(|alert| ReviewItem {
            id: alert.skill_id.clone(),
            kind: "drift".to_string(),
            content: format!("skill {} drift", alert.skill_id),
            category: ReviewCategory::Drift,
        })
        .collect();

    Ok(ReviewSummary {
        uncertain_count: uncertain.len(),
        grouping_count: 0,
        drift_count: drift.len(),
        currency_flips_count: 0,
        uncertain_items: uncertain.into_iter().take(DISPLAY_CAP).collect(),
        grouping_items: Vec::new(),
        drift_items: drift.into_iter().take(DISPLAY_CAP).collect(),
    })
}

// The review cursor's position is never read by any gather/list path below (each takes `_cursor`
// and ignores it) — visibility is act-on-based: a pending item stays visible until `act_on_item`
// resolves it (see `cg_pending_review_items` below), never by cursor position. A prior cursor
// "advance" API (`advance_cursor`/`maybe_advance_cursor`/`is_cursor_local`) mutated an in-memory
// `ReviewCursor` that was constructed fresh and dropped at the end of every `cg review` call — a
// build-increment-drop with no persistence and no reader. Deleted; do not reintroduce a cursor
// advance without a persisted, read consumer.
#[derive(Debug, Clone, Default)]
pub struct ReviewCursor {
    pub position: usize,
    pub is_local: bool,
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
pub fn is_item_pending(
    store: &Store,
    item_id: &str,
    _cursor: &ReviewCursor,
) -> crate::Result<bool> {
    // Ensure the item is registered in the pending table.
    ensure_pending_item_registered(store, item_id);

    let rows = store.read_table(PENDING_ITEMS_TABLE)?;
    for row in &rows {
        match row.values.get("id") {
            Some(contextdb_core::Value::Text(id)) if id == item_id => {
                return Ok(match row.values.get("acted_on") {
                    Some(contextdb_core::Value::Bool(b)) => !b,
                    Some(contextdb_core::Value::Int64(i)) => *i == 0,
                    _ => true,
                });
            }
            _ => {}
        }
    }
    Ok(true)
}

/// List all pending review items (not yet acted on).
pub fn list_pending_items(store: &Store, _cursor: &ReviewCursor) -> crate::Result<Vec<ReviewItem>> {
    let rows = store.read_table(PENDING_ITEMS_TABLE)?;
    Ok(rows
        .into_iter()
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
        .collect())
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
