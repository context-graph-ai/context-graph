//! Placeholder two-layer overlay. The correct implementation keeps originals immutable
//! and layers corrections as additive overlay rows merged at read time. Every function
//! here returns a default/empty value; assertions fail until the real implementation is in place.

use crate::Store;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayKind {
    Correction,
    Drop,
    GroupingLink,
    WrongTypeCorrection,
    GroupingReversal,
}

#[derive(Debug, Clone)]
pub struct OverlayEntry {
    pub id: String,
    pub item_id: String,
    pub kind: OverlayKind,
    pub value: Option<String>,
    pub reversed: bool,
}

/// Compensation / reversal event logged when an overlay is reversed.
/// Placeholder: `reverse_overlay` does not yet create this event.
#[derive(Debug, Clone)]
pub struct CompensationEvent {
    pub id: String,
    /// The overlay entry that was reversed.
    pub reversed_overlay_id: String,
    pub kind: CompensationEventKind,
    /// Content describing what was reversed (e.g. grouped intention ids, corrected-item id).
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompensationEventKind {
    CorrectionReversal,
    GroupingReversal,
}

/// Placeholder: returns a new entry but does NOT actually persist anything or create an overlay.
pub fn record_correction(_store: &Store, item_id: &str, new_value: &str) -> OverlayEntry {
    OverlayEntry {
        id: format!("{}", uuid::Uuid::new_v4()),
        item_id: item_id.to_string(),
        kind: OverlayKind::Correction,
        value: Some(new_value.to_string()),
        reversed: false,
    }
}

/// Record a wrong-type correction (e.g. Decision → Intention).
/// Placeholder: returns an entry but does NOT persist, log a teaching event, or overlay on read.
pub fn record_wrong_type_correction(_store: &Store, item_id: &str, new_type: &str) -> OverlayEntry {
    OverlayEntry {
        id: format!("{}", uuid::Uuid::new_v4()),
        item_id: item_id.to_string(),
        kind: OverlayKind::WrongTypeCorrection,
        value: Some(new_type.to_string()),
        reversed: false,
    }
}

/// Record a drop overlay.
/// Placeholder: returns an entry but does NOT persist it; `is_dropped` will still return false.
pub fn record_drop(_store: &Store, item_id: &str) -> OverlayEntry {
    OverlayEntry {
        id: format!("{}", uuid::Uuid::new_v4()),
        item_id: item_id.to_string(),
        kind: OverlayKind::Drop,
        value: None,
        reversed: false,
    }
}

/// Record a grouping link between an intention and a catalog entry.
/// Placeholder: returns an entry but does NOT persist any link.
pub fn record_grouping_link(_store: &Store, intention_id: &str, catalog_id: &str) -> OverlayEntry {
    OverlayEntry {
        id: format!("{}", uuid::Uuid::new_v4()),
        item_id: intention_id.to_string(),
        kind: OverlayKind::GroupingLink,
        value: Some(catalog_id.to_string()),
        reversed: false,
    }
}

/// Placeholder: always returns the original value (ignores any overlay entries).
pub fn read_with_overlay(
    original_value: &str,
    _item_id: &str,
    _entries: &[OverlayEntry],
) -> String {
    original_value.to_string()
}

/// Placeholder: marks reversed=true but does NOT log a compensation event.
/// The returned `CompensationEvent` is None until the real implementation is in place.
pub fn reverse_overlay(_store: &Store, entry: &mut OverlayEntry) -> Option<CompensationEvent> {
    entry.reversed = true;
    // Placeholder: no compensation/reversal event is logged.
    None
}

/// Placeholder: always returns false. The real implementation checks the overlay entries for a drop record.
pub fn is_dropped(_item_id: &str, _entries: &[OverlayEntry]) -> bool {
    false
}

/// List all overlay entries for an item (corrections, drops, grouping links).
/// Placeholder: always returns empty. The real implementation reads persisted entries.
pub fn list_overlay_entries_for_item(_store: &Store, _item_id: &str) -> Vec<OverlayEntry> {
    vec![]
}

/// Return all compensation events logged.
/// Placeholder: always returns empty. The real implementation reads persisted events.
pub fn list_compensation_events(_store: &Store) -> Vec<CompensationEvent> {
    vec![]
}

/// Apply out-of-order teaching events and return the final read value.
/// Placeholder: always returns the original value unchanged (no convergence).
pub fn apply_teaching_events_in_order(original_value: &str, _events: &[OverlayEntry]) -> String {
    original_value.to_string()
}
