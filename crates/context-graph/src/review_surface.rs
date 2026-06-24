//! Placeholder HITL review surface. Every function returns a default or incorrect value
//! until the real implementation is in place.

use crate::Store;

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
    /// Structurally-empty category for currency-flips (server-core owns this).
    /// Placeholder: always 0; the category must be present even at 0.
    pub currency_flips_count: usize,
    /// Displayed items (may be capped). Placeholder: reports capped length instead of the true count.
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

/// Placeholder: accepts any unknown verb (returns Accept for anything unrecognized instead of None).
/// The real implementation returns None for unrecognized verbs.
pub fn parse_verb(s: &str) -> Option<ReviewVerb> {
    match s {
        "accept" => Some(ReviewVerb::Accept),
        "correct" => Some(ReviewVerb::Correct),
        "drop" => Some(ReviewVerb::Drop),
        "group" => Some(ReviewVerb::Group),
        "undo" => Some(ReviewVerb::Undo),
        "skills-off" => Some(ReviewVerb::SkillsOff),
        "skills-on" => Some(ReviewVerb::SkillsOn),
        // Placeholder: accepts unknown verbs (real impl returns None for unrecognized input).
        _ => Some(ReviewVerb::Accept),
    }
}

/// The 5 emitted message kinds. Placeholder: returns only 4 (DriftAlert is missing).
pub fn all_message_kinds() -> Vec<MessageKind> {
    vec![
        MessageKind::CaptureReceipt,
        MessageKind::ReviewItem,
        MessageKind::GroupingProposal,
        MessageKind::SkillCard,
        // Placeholder: DriftAlert is not yet included; the real implementation returns all 5.
    ]
}

/// Placeholder: uses capped count instead of the true count, and omits currency_flips from the output.
pub fn build_review_summary(items: &[ReviewItem]) -> ReviewSummary {
    let cap = 5;
    ReviewSummary {
        // Placeholder: uses capped count instead of the true count.
        uncertain_count: items
            .iter()
            .filter(|i| i.category == ReviewCategory::Uncertain)
            .count()
            .min(cap),
        grouping_count: items
            .iter()
            .filter(|i| i.category == ReviewCategory::Grouping)
            .count()
            .min(cap),
        drift_count: items
            .iter()
            .filter(|i| i.category == ReviewCategory::Drift)
            .count()
            .min(cap),
        // Placeholder: currency_flips is always 0. The category must be present even at 0.
        currency_flips_count: 0,
        uncertain_items: items
            .iter()
            .filter(|i| i.category == ReviewCategory::Uncertain)
            .take(cap)
            .cloned()
            .collect(),
        grouping_items: items
            .iter()
            .filter(|i| i.category == ReviewCategory::Grouping)
            .take(cap)
            .cloned()
            .collect(),
        drift_items: items
            .iter()
            .filter(|i| i.category == ReviewCategory::Drift)
            .take(cap)
            .cloned()
            .collect(),
    }
}

/// Build review summary exposing all four categories including the structurally-empty currency-flips.
/// Placeholder: delegates to `build_review_summary` — currency_flips is still 0 and not
/// explicitly set as a distinct category. The real impl must include it at `currency_flips_count: 0`.
pub fn build_review_summary_with_currency_category(items: &[ReviewItem]) -> ReviewSummary {
    build_review_summary(items)
}

#[derive(Debug, Clone, Default)]
pub struct ReviewCursor {
    pub position: usize,
    pub is_local: bool,
}

/// Placeholder: always returns false. The review cursor is always local-only; real impl returns true.
pub fn is_cursor_local(_cursor: &ReviewCursor) -> bool {
    false
}

/// Advance the review cursor past the current position.
/// Placeholder: no-op.
pub fn advance_cursor(_store: &Store, _cursor: &mut ReviewCursor) {}

/// Advance cursor only for a bare (unfiltered) review call.
/// Placeholder: always advances, even for filtered lens calls. The real implementation
/// advances only when `is_bare_review` is true.
pub fn maybe_advance_cursor(_store: &Store, cursor: &mut ReviewCursor, _is_bare_review: bool) {
    cursor.position += 1;
}

/// Returns true if the given item is still pending (not acted on).
/// Placeholder: always returns false.
pub fn is_item_pending(_store: &Store, _item_id: &str, _cursor: &ReviewCursor) -> bool {
    false
}

/// List items pending review (not yet acted on).
/// Placeholder: always returns empty.
pub fn list_pending_items(_store: &Store, _cursor: &ReviewCursor) -> Vec<ReviewItem> {
    vec![]
}

/// Mark an item as acted on (removes from pending).
/// Placeholder: does nothing.
pub fn act_on_item(_store: &Store, _item_id: &str, _verb: &ReviewVerb) {}

/// Emit a capture receipt message leading with intentions.
/// Placeholder: prefixes with "Decisions:" instead of "Intentions:". The real implementation leads with intentions.
pub fn emit_capture_receipt(report: &str) -> ReviewMessage {
    ReviewMessage {
        kind: MessageKind::CaptureReceipt,
        content: format!("Decisions: {}", report),
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
