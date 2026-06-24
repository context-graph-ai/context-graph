//! Placeholder skill store types and operations. Every function returns empty or zero values
//! until the real store-backed implementation is in place.

use serde::{Deserialize, Serialize};

use crate::Store;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillProvenance {
    HumanCorrection,
    LlmAuthored,
    HumanAuthored,
}

#[derive(Debug, Clone)]
pub struct LearnedSkill {
    pub id: String,
    pub description: String,
    pub rule: String,
    pub provenance: SkillProvenance,
    pub active: bool,
    pub held_for_review: bool,
    pub fire_right: u32,
    pub fire_wrong: u32,
}

#[derive(Debug, Clone)]
pub struct TeachingEvent {
    pub id: String,
    /// The skill this teaching event is associated with (empty string if none yet).
    pub skill_id: String,
    pub corrected_item_id: String,
    pub kind: TeachingEventKind,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TeachingEventKind {
    Correction,
    Reversal,
    GroupingReversal,
    WrongTypeCorrection,
}

#[derive(Debug, Clone)]
pub struct FiringTrace {
    pub id: String,
    pub skill_id: String,
    pub capture_id: String,
    /// None = outcome unknown; Some(true) = correct fire; Some(false) = wrong fire.
    pub correct: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct DriftAlert {
    pub skill_id: String,
    pub provenance: SkillProvenance,
    /// Whether this alert is still active (false after the alert is resolved).
    pub active: bool,
}

pub struct HeldOutResult {
    pub score: f64,
}

/// Result of a skill promotion attempt.
#[derive(Debug, Clone)]
pub struct PromotionResult {
    /// Whether the promotion succeeded (held-out gate was beaten, not just tied).
    pub promoted: bool,
    /// The skill produced, if promoted.
    pub skill: Option<LearnedSkill>,
    /// The score achieved on the held-out set (for testing tie detection).
    pub heldout_score: f64,
}

/// A parse-only result for a human-authored skill file.
#[derive(Debug, Clone)]
pub struct ParseSkillResult {
    pub accepted: bool,
    /// If not accepted, the parse error message.
    pub error: Option<String>,
    /// The parsed skill, if accepted.
    pub skill: Option<LearnedSkill>,
}

/// Outcome of applying a `cg fix` correction to an item.
#[derive(Debug, Clone)]
pub struct CorrectionResult {
    /// Whether the item was corrected (overlay applied).
    pub applied: bool,
    /// The teaching event logged for this correction (None until the store-write path is implemented).
    pub teaching_event: Option<TeachingEvent>,
    /// Whether a new auto-applying skill was created. Placeholder: always false.
    pub new_skill_created: bool,
}

/// Placeholder: always returns None. The real implementation promotes when aligned labels beat the held-out gate.
pub fn try_promote_skill(
    _labels: &[TeachingEvent],
    _heldout: &[TeachingEvent],
) -> Option<LearnedSkill> {
    None
}

/// Placeholder: returns 0.0. The real implementation evaluates the skill against the held-out label set.
pub fn evaluate_on_heldout(_skill: &LearnedSkill, _heldout: &[TeachingEvent]) -> HeldOutResult {
    HeldOutResult { score: 0.0 }
}

/// Promotion with explicit tie-detection. Placeholder: always returns not-promoted with score 0.0.
pub fn try_promote_skill_with_result(
    _labels: &[TeachingEvent],
    _heldout: &[TeachingEvent],
    _incumbent_score: Option<f64>,
) -> PromotionResult {
    PromotionResult {
        promoted: false,
        skill: None,
        heldout_score: 0.0,
    }
}

/// Parse a human-authored skill file. Placeholder: accepts any non-empty content without checking
/// structure; a malformed file is also accepted (no parse error returned). The real implementation
/// validates the SKILL.md structure and rejects malformed input.
pub fn parse_human_authored_skill(content: &str) -> ParseSkillResult {
    if content.trim().is_empty() {
        ParseSkillResult {
            accepted: false,
            error: Some("empty skill file".to_string()),
            skill: None,
        }
    } else {
        // Placeholder: accepts without checking structure; real impl parses SKILL.md shape.
        ParseSkillResult {
            accepted: true,
            error: None,
            skill: Some(LearnedSkill {
                id: uuid::Uuid::new_v4().to_string(),
                description: content.lines().next().unwrap_or("").to_string(),
                rule: content.to_string(),
                provenance: SkillProvenance::HumanAuthored,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            }),
        }
    }
}

/// Apply a `cg fix` correction: logs a teaching-event label and applies an overlay.
/// Placeholder: applies a correction but does NOT log a teaching event (event is None).
pub fn apply_correction(
    _store: &Store,
    item_id: &str,
    new_value: &str,
    skill_id: Option<&str>,
) -> CorrectionResult {
    let _ = (item_id, new_value, skill_id);
    // Placeholder: does not log a teaching event.
    CorrectionResult {
        applied: true,
        teaching_event: None, // Placeholder: real impl returns the logged TeachingEvent.
        new_skill_created: false,
    }
}

/// Placeholder: returns empty. The real implementation reads from the cg_learned_skills table.
pub fn list_skills_in_store(_store: &Store) -> Vec<LearnedSkill> {
    vec![]
}

/// Placeholder: returns empty. The real implementation reads from the cg_teaching_events table.
pub fn list_teaching_events_in_store(_store: &Store) -> Vec<TeachingEvent> {
    vec![]
}

/// Placeholder: never alerts. The real implementation raises an alert when the firing record turns bad.
pub fn check_drift(_skill: &LearnedSkill, _traces: &[FiringTrace]) -> Option<DriftAlert> {
    None
}

/// Returns all drift alerts currently active.
/// Placeholder: always empty. The real implementation deduplicates by skill id and persists alerts.
pub fn list_drift_alerts(_store: &Store) -> Vec<DriftAlert> {
    vec![]
}

/// Record a firing trace entry for a skill that just fired.
/// Placeholder: returns a struct but does not persist the trace.
pub fn record_firing(_store: &Store, _skill_id: &str, _capture_id: &str) -> FiringTrace {
    FiringTrace {
        id: uuid::Uuid::new_v4().to_string(),
        skill_id: _skill_id.to_string(),
        capture_id: _capture_id.to_string(),
        correct: None,
    }
}

/// Record a ground-truth label for a firing (correct or incorrect).
/// Placeholder: does not update any stored record.
pub fn record_firing_label(_store: &Store, _firing_id: &str, _correct: bool) {}

/// Export a skill to its SKILL.md text representation.
/// Placeholder: returns a non-conforming string; the real export uses "# description:" headers.
pub fn export_skill(skill: &LearnedSkill) -> String {
    format!(
        "# Skill: {}\n## Rule\n{}\n## Provenance\n{:?}\n",
        skill.description, skill.rule, skill.provenance
    )
}

/// Enable or disable the global skills master switch.
/// Placeholder: does nothing (state is not persisted, always returns false).
pub fn set_skills_enabled(_store: &Store, _enabled: bool) -> bool {
    false // Placeholder: real impl returns the previous state.
}

/// Enable or disable the review-hold (opt-in `cg skills review`).
/// Placeholder: does nothing.
pub fn set_review_hold(_store: &Store, _active: bool) {}

/// Acknowledge a held skill, allowing it to auto-apply from now on.
/// Placeholder: does nothing — the skill remains held.
pub fn acknowledge_skill(_store: &Store, _skill_id: &str) {}

/// Table name constants needed for sync policy tests.
pub const SKILL_TABLE: &str = "cg_learned_skills";
pub const TEACHING_EVENT_TABLE: &str = "cg_teaching_events";
pub const FIRING_TRACE_TABLE: &str = "cg_skill_firing_trace";
pub const REVIEW_CURSOR_TABLE: &str = "cg_review_cursor";
