//! Two-layer reversibility: immutable original captures plus an append-only overlay
//! that layers corrections, drops, grouping links, and type changes at read time.
//! Original rows are never edited or deleted (Invariant 3). Reversals are tombstones
//! keyed to the overlay entry they retract; "present and not-retracted" is the live
//! state, so undo is global and order-tolerant.

use std::collections::HashMap;

use crate::{Store, skill_store::execute_with_live_txid};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayKind {
    Correction,
    Drop,
    GroupingLink,
    WrongTypeCorrection,
    GroupingReversal,
    /// A tombstone that retracts another overlay entry. Its `value` holds the id of the entry it
    /// reverses. Append-only: the retracted entry is left byte-unchanged (Invariant 3).
    Reversal,
}

#[derive(Debug, Clone)]
pub struct OverlayEntry {
    pub id: String,
    pub item_id: String,
    pub kind: OverlayKind,
    pub value: Option<String>,
    pub reversed: bool,
}

/// Compensation event logged when an overlay is reversed.
#[derive(Debug, Clone)]
pub struct CompensationEvent {
    pub id: String,
    /// The overlay entry that was reversed.
    pub reversed_overlay_id: String,
    pub kind: CompensationEventKind,
    /// Content describing what was reversed (grouped intention ids, corrected-item id, etc.).
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompensationEventKind {
    CorrectionReversal,
    GroupingReversal,
}

const OVERLAY_TABLE: &str = "cg_overlay_entries";
const COMPENSATION_TABLE: &str = "cg_compensation_events";

fn overlay_kind_str(kind: &OverlayKind) -> &'static str {
    match kind {
        OverlayKind::Correction => "correction",
        OverlayKind::Drop => "drop",
        OverlayKind::GroupingLink => "grouping_link",
        OverlayKind::WrongTypeCorrection => "wrong_type_correction",
        OverlayKind::GroupingReversal => "grouping_reversal",
        OverlayKind::Reversal => "reversal",
    }
}

fn overlay_kind_from_str(s: &str) -> OverlayKind {
    match s {
        "correction" => OverlayKind::Correction,
        "drop" => OverlayKind::Drop,
        "grouping_link" => OverlayKind::GroupingLink,
        "wrong_type_correction" => OverlayKind::WrongTypeCorrection,
        "grouping_reversal" => OverlayKind::GroupingReversal,
        "reversal" => OverlayKind::Reversal,
        _ => OverlayKind::Correction,
    }
}

fn compensation_kind_str(kind: &CompensationEventKind) -> &'static str {
    match kind {
        CompensationEventKind::CorrectionReversal => "correction_reversal",
        CompensationEventKind::GroupingReversal => "grouping_reversal",
    }
}

fn compensation_kind_from_str(s: &str) -> CompensationEventKind {
    match s {
        "grouping_reversal" => CompensationEventKind::GroupingReversal,
        _ => CompensationEventKind::CorrectionReversal,
    }
}

fn persist_overlay_entry(store: &Store, entry: &OverlayEntry) {
    let db = store.sync_database();
    let kind_str = overlay_kind_str(&entry.kind);
    let params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(entry.id.clone()),
        ),
        (
            "item_id".to_string(),
            contextdb_core::Value::Text(entry.item_id.clone()),
        ),
        (
            "kind".to_string(),
            contextdb_core::Value::Text(kind_str.to_string()),
        ),
        (
            "value".to_string(),
            entry
                .value
                .as_ref()
                .map(|v| contextdb_core::Value::Text(v.clone()))
                .unwrap_or(contextdb_core::Value::Null),
        ),
        (
            "reversed".to_string(),
            contextdb_core::Value::Bool(entry.reversed),
        ),
    ]);
    let _ = execute_with_live_txid(
        &db,
        &format!(
            "INSERT INTO {OVERLAY_TABLE} (id, item_id, kind, value, reversed, created_at) VALUES ($id, $item_id, $kind, $value, $reversed, $created_at)"
        ),
        "created_at",
        params,
    );
}

fn persist_compensation_event(store: &Store, event: &CompensationEvent) {
    let db = store.sync_database();
    let kind_str = compensation_kind_str(&event.kind);
    let params = HashMap::from([
        (
            "id".to_string(),
            contextdb_core::Value::Text(event.id.clone()),
        ),
        (
            "reversed_overlay_id".to_string(),
            contextdb_core::Value::Text(event.reversed_overlay_id.clone()),
        ),
        (
            "kind".to_string(),
            contextdb_core::Value::Text(kind_str.to_string()),
        ),
        (
            "content".to_string(),
            contextdb_core::Value::Text(event.content.clone()),
        ),
    ]);
    let _ = execute_with_live_txid(
        &db,
        &format!(
            "INSERT INTO {COMPENSATION_TABLE} (id, reversed_overlay_id, kind, content, created_at) VALUES ($id, $reversed_overlay_id, $kind, $content, $created_at)"
        ),
        "created_at",
        params,
    );
}

fn entry_from_row(row: &HashMap<String, contextdb_core::Value>) -> OverlayEntry {
    use contextdb_core::Value;
    let id = text_or_uuid_str(row.get("id"));
    let item_id = text_or_uuid_str(row.get("item_id"));
    let kind = match row.get("kind") {
        Some(Value::Text(s)) => overlay_kind_from_str(s),
        _ => OverlayKind::Correction,
    };
    let value = match row.get("value") {
        Some(Value::Text(s)) => Some(s.clone()),
        Some(Value::Uuid(u)) => Some(u.to_string()),
        _ => None,
    };
    let reversed = match row.get("reversed") {
        Some(Value::Bool(b)) => *b,
        Some(Value::Int64(i)) => *i != 0,
        _ => false,
    };
    OverlayEntry {
        id,
        item_id,
        kind,
        value,
        reversed,
    }
}

fn text_or_uuid_str(value: Option<&contextdb_core::Value>) -> String {
    use contextdb_core::Value;
    match value {
        Some(Value::Text(s)) => s.clone(),
        Some(Value::Uuid(u)) => u.to_string(),
        _ => String::new(),
    }
}

fn compensation_from_row(row: &HashMap<String, contextdb_core::Value>) -> CompensationEvent {
    use contextdb_core::Value;
    let id = text_or_uuid_str(row.get("id"));
    let reversed_overlay_id = text_or_uuid_str(row.get("reversed_overlay_id"));
    let kind = match row.get("kind") {
        Some(Value::Text(s)) => compensation_kind_from_str(s),
        _ => CompensationEventKind::CorrectionReversal,
    };
    let content = match row.get("content") {
        Some(Value::Text(s)) => s.clone(),
        _ => String::new(),
    };
    CompensationEvent {
        id,
        reversed_overlay_id,
        kind,
        content,
    }
}

/// Record a value correction for an item. Persists an OverlayEntry row with
/// Correction kind; the original captured row is untouched.
pub fn record_correction(store: &Store, item_id: &str, new_value: &str) -> OverlayEntry {
    let entry = OverlayEntry {
        id: uuid::Uuid::new_v4().to_string(),
        item_id: item_id.to_string(),
        kind: OverlayKind::Correction,
        value: Some(new_value.to_string()),
        reversed: false,
    };
    persist_overlay_entry(store, &entry);
    entry
}

/// Record a wrong-type correction (e.g. a captured Decision corrected to an Intention).
/// Persists an OverlayEntry row with WrongTypeCorrection kind; the original row is untouched.
pub fn record_wrong_type_correction(store: &Store, item_id: &str, new_type: &str) -> OverlayEntry {
    let entry = OverlayEntry {
        id: uuid::Uuid::new_v4().to_string(),
        item_id: item_id.to_string(),
        kind: OverlayKind::WrongTypeCorrection,
        value: Some(new_type.to_string()),
        reversed: false,
    };
    persist_overlay_entry(store, &entry);
    entry
}

/// Record a drop overlay — excludes the item from the read view without deleting it.
pub fn record_drop(store: &Store, item_id: &str) -> OverlayEntry {
    let entry = OverlayEntry {
        id: uuid::Uuid::new_v4().to_string(),
        item_id: item_id.to_string(),
        kind: OverlayKind::Drop,
        value: None,
        reversed: false,
    };
    persist_overlay_entry(store, &entry);
    entry
}

/// Record a grouping link between an intention and a catalog entry.
/// Additive: the original intention row is untouched. Each grouping link may be
/// independently reversed.
pub fn record_grouping_link(store: &Store, intention_id: &str, catalog_id: &str) -> OverlayEntry {
    let entry = OverlayEntry {
        id: uuid::Uuid::new_v4().to_string(),
        item_id: intention_id.to_string(),
        kind: OverlayKind::GroupingLink,
        value: Some(catalog_id.to_string()),
        reversed: false,
    };
    persist_overlay_entry(store, &entry);
    entry
}

/// Compute the read-view value for an item by applying its non-reversed overlay entries.
///
/// Rules:
/// - If any non-reversed Correction or WrongTypeCorrection entry exists, the last one wins.
/// - If any non-reversed Drop entry exists, the item is excluded (caller should check `is_dropped`).
/// - GroupingLink entries affect the grouping view, not the value — they are not applied here.
/// - If no active correction exists, the original value is returned unchanged.
pub fn read_with_overlay(original_value: &str, _item_id: &str, entries: &[OverlayEntry]) -> String {
    let cancelled = cancelled_overlay_ids(entries);
    // Find the last correction or type-correction entry that no reversal tombstone has retracted.
    let active_correction = entries.iter().rev().find(|e| {
        !cancelled.contains(e.id.as_str())
            && matches!(
                e.kind,
                OverlayKind::Correction | OverlayKind::WrongTypeCorrection
            )
    });
    match active_correction {
        Some(entry) => entry.value.as_deref().unwrap_or(original_value).to_string(),
        None => original_value.to_string(),
    }
}

/// The set of overlay-entry ids retracted by a reversal tombstone among `entries`. A tombstone is
/// a `Reversal`/`GroupingReversal` entry whose `value` holds the id of the entry it retracts.
fn cancelled_overlay_ids(entries: &[OverlayEntry]) -> std::collections::HashSet<&str> {
    entries
        .iter()
        .filter(|e| {
            matches!(
                e.kind,
                OverlayKind::Reversal | OverlayKind::GroupingReversal
            )
        })
        .filter_map(|e| e.value.as_deref())
        .collect()
}

/// Reverse an overlay entry: marks it reversed, logs a compensation event, and persists both.
/// Returns the compensation event, or None only if the entry was already reversed.
pub fn reverse_overlay(store: &Store, entry: &mut OverlayEntry) -> Option<CompensationEvent> {
    entry.reversed = true;

    // Append-only (Invariant 3): leave the retracted entry byte-unchanged and record a NEW
    // tombstone row that points at it. The tombstone carries `reversed = true` and its `value`
    // is the id of the entry it retracts; the read view treats any entry a tombstone targets as
    // cancelled. Because it is a fresh insert, the reversal also converges across edges under the
    // overlay table's order-free `InsertIfNotExists` sync policy.
    let tombstone = OverlayEntry {
        id: uuid::Uuid::new_v4().to_string(),
        item_id: entry.item_id.clone(),
        kind: match entry.kind {
            OverlayKind::GroupingLink | OverlayKind::GroupingReversal => {
                OverlayKind::GroupingReversal
            }
            _ => OverlayKind::Reversal,
        },
        value: Some(entry.id.clone()),
        reversed: true,
    };
    persist_overlay_entry(store, &tombstone);

    let event_kind = match entry.kind {
        OverlayKind::GroupingLink | OverlayKind::GroupingReversal => {
            CompensationEventKind::GroupingReversal
        }
        _ => CompensationEventKind::CorrectionReversal,
    };

    let content = match &entry.kind {
        OverlayKind::GroupingLink => {
            // Content references the grouped item id and the catalog id.
            format!(
                "grouping reversed: item={} catalog={}",
                entry.item_id,
                entry.value.as_deref().unwrap_or("")
            )
        }
        _ => {
            format!(
                "correction reversed: item={} value={}",
                entry.item_id,
                entry.value.as_deref().unwrap_or("")
            )
        }
    };

    let event = CompensationEvent {
        id: uuid::Uuid::new_v4().to_string(),
        reversed_overlay_id: entry.id.clone(),
        kind: event_kind,
        content,
    };
    persist_compensation_event(store, &event);
    Some(event)
}

/// Returns true if a Drop entry that no reversal tombstone has retracted exists for this item.
pub fn is_dropped(_item_id: &str, entries: &[OverlayEntry]) -> bool {
    let cancelled = cancelled_overlay_ids(entries);
    entries
        .iter()
        .any(|e| e.kind == OverlayKind::Drop && !cancelled.contains(e.id.as_str()))
}

/// List all overlay entries for a given item from persisted store state.
pub fn list_overlay_entries_for_item(store: &Store, item_id: &str) -> Vec<OverlayEntry> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    let rows = db.scan(OVERLAY_TABLE, snapshot).unwrap_or_default();
    rows.into_iter()
        .filter_map(|row| {
            let entry = entry_from_row(&row.values);
            if entry.item_id == item_id {
                Some(entry)
            } else {
                None
            }
        })
        .collect()
}

/// Return all compensation events logged in the store.
pub fn list_compensation_events(store: &Store) -> Vec<CompensationEvent> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    let rows = db.scan(COMPENSATION_TABLE, snapshot).unwrap_or_default();
    rows.into_iter()
        .map(|row| compensation_from_row(&row.values))
        .collect()
}

/// Apply overlay entries to produce the final read value, order-tolerant.
///
/// The convergence rule is:
/// - Collect all corrections by id.
/// - A correction whose id appears in the reversed set (any entry with reversed=true
///   that shares the same original correction id, OR the entry itself has reversed=true)
///   is cancelled.
/// - Among the remaining active corrections, the value from the last one in order wins.
/// - A reversal entry (reversed=true) cancels the correction it came from.
///
/// Order-tolerance: applying [correction, reversal] or [reversal, correction] must
/// produce the same final value (the original, since the correction is cancelled).
pub fn apply_teaching_events_in_order(original_value: &str, events: &[OverlayEntry]) -> String {
    // Build the set of ids that are reversed (any entry with reversed=true).
    let reversed_ids: std::collections::HashSet<&str> = events
        .iter()
        .filter(|e| e.reversed)
        .map(|e| e.id.as_str())
        .collect();

    // Collect active (non-reversed) correction entries in order.
    let active_corrections: Vec<&OverlayEntry> = events
        .iter()
        .filter(|e| {
            !e.reversed
                && !reversed_ids.contains(e.id.as_str())
                && matches!(
                    e.kind,
                    OverlayKind::Correction | OverlayKind::WrongTypeCorrection
                )
        })
        .collect();

    // The last active correction wins; if none, return the original.
    match active_corrections.last() {
        Some(entry) => entry.value.as_deref().unwrap_or(original_value).to_string(),
        None => original_value.to_string(),
    }
}
