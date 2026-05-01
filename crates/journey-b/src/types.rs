//! Journey-local type aliases. We keep these minimal — the spike isn't a
//! reimplementation of `context-graph::types`; it carries only what the
//! journey moments need.

use uuid::Uuid;

pub type IntentionId = Uuid;
pub type DecisionId = Uuid;
pub type EntityId = Uuid;
pub type ObservationId = Uuid;
pub type InvalidationId = Uuid;
pub type DigestId = Uuid;
pub type ContextId = Uuid;

/// Severity per `impact_analysis_direct_invalidation_by_engine` (cg-acceptance-tests.md).
/// Confidence ≥ 0.8 → Critical, ≥ 0.5 → Warning, < 0.5 → Info.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Critical,
    Warning,
    Info,
}

impl Severity {
    pub fn from_confidence(confidence: f32) -> Self {
        if confidence >= 0.8 {
            Severity::Critical
        } else if confidence >= 0.5 {
            Severity::Warning
        } else {
            Severity::Info
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidationStatus {
    Pending,
    Acknowledged,
    Dismissed,
    Resolved,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasisDiff {
    pub entity_id: EntityId,
    pub field: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: serde_json::Value,
    pub cascade_from: Option<DecisionId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Invalidation {
    pub id: InvalidationId,
    pub status: InvalidationStatus,
    pub affected_decision_id: DecisionId,
    pub trigger_observation_id: ObservationId,
    pub basis_diff: BasisDiff,
    pub severity: Severity,
}

#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub decision_id: DecisionId,
    pub similarity: f32,
    /// `effective_confidence = confidence × success_rate`.
    /// Per `effective_confidence_null_confidence_defaults_to_success_rate`:
    /// confidence=None → multiplier 1.0. Per `effective_confidence_formula_no_outcome`:
    /// no outcome → success_rate 1.0.
    /// Per `semantic_search_effective_confidence_ranking`: an Invalidated
    /// decision's effective_confidence collapses (rank-policy formula
    /// deprioritises it OR a `decision_status` filter excludes it).
    pub effective_confidence: Option<f32>,
    pub status_at_query_time: DecisionStatusSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionStatusSnapshot {
    Active,
    Invalidated,
    Archived,
    Superseded,
}
