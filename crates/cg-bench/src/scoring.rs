//! The frozen scoring method (criteria commit `99d99ee`).
//!
//! Hierarchical, intention-first, session-level. Matching is embedding-similarity banded by the
//! two frozen thresholds; the gray zone is ROUTED OUT for adjudication (fixed-prompt judge +
//! owner sample), never silently decided here.

use crate::gold::GoldSession;

/// Top-1 similarity at or above this is an automatic match. FROZEN.
pub const AUTO_MATCH_THRESHOLD: f32 = 0.75;
/// Top-1 similarity below this is an automatic non-match. FROZEN.
pub const AUTO_NON_MATCH_THRESHOLD: f32 = 0.55;

/// Similarity seam: production runs use the bundled nomic embedder; tests use deterministic
/// doubles (this trait tests the banding/aggregation logic, never model quality — model quality
/// is the owner live smoke).
pub trait SimilarityProvider {
    fn similarity(&self, a: &str, b: &str) -> f32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchBand {
    Match,
    GrayZone,
    NonMatch,
}

/// Band a similarity score per the frozen thresholds: `>= 0.75` match, `< 0.55` non-match,
/// `[0.55, 0.75)` gray zone (adjudicated externally).
pub fn band(_similarity: f32) -> MatchBand {
    todo!("cg-correctness C1: threshold banding not implemented yet")
}

/// What an understanding run produced for one session, AFTER session-level reconciliation —
/// intentions are session-level objects here by construction (per-chunk scoring cannot express
/// the intention question).
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedSession {
    pub intentions: Vec<String>,
    pub decisions: Vec<ExtractedDecision>,
    pub entities: Vec<ExtractedEntity>,
    pub observations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedDecision {
    pub statement: String,
    /// The goal this decision claims to serve (the extracted trunk attachment), if any.
    pub goal: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CategoryScore {
    pub matched: usize,
    pub gold_total: usize,
    /// Gold statements nothing matched — the omission record.
    pub missed: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionScore {
    /// Session-level: did any extracted intention match a gold intention?
    pub intention_captured: bool,
    /// Attachment is scored ONLY over matched decisions whose gold row has an attachment.
    pub attachment_correct: usize,
    pub attachment_total: usize,
    pub decisions: CategoryScore,
    /// Extracted decisions collapse into similarity groups before precision; a duplicate group
    /// counts once.
    pub decision_groups: usize,
    pub decision_noise_groups: usize,
    /// Fraction of extracted decision items that were duplicates of another extracted item.
    pub duplication_rate: f32,
    pub entities: CategoryScore,
    pub entity_type_correct: usize,
    /// Pairs routed to adjudication (extracted statement, gold statement) — never auto-decided.
    pub gray_zone: Vec<(String, String)>,
}

/// Score one session's extraction against its gold, per the frozen method.
pub fn score_session(
    _gold: &GoldSession,
    _extracted: &ExtractedSession,
    _sim: &dyn SimilarityProvider,
) -> SessionScore {
    todo!("cg-correctness C1: session scoring not implemented yet")
}
