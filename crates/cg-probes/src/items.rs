//! Probe item construction — the deterministic layer between corpus and verifier. P1 items are
//! EXHAUSTIVE: positives are the gold evidence spans (with structural turn context — the
//! containing turn plus its predecessor, a content-blind rule applied uniformly), negatives are
//! the authored non-decisions plus every turn carrying no gold span. Basis items judge claims
//! against the session's oracle packet (gold spans only — no retrieval, no scanning). P2
//! evaluation primitives attribute failure to judgment (span present in a window, scored low)
//! vs segmentation (span never isolated into any window).

use crate::corpus::{SessionRecord, Turn};
use crate::windows::Window;

/// One decision-vs-discussion judgment item.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgmentItem {
    pub session_id: String,
    pub calibration: bool,
    /// The text the verifier judges (span/turn with its structural context).
    pub premise: String,
    /// Gold: is this a durable decision span?
    pub positive: bool,
    /// What this item is, for reports (e.g. "gold-span", "non-decision", "turn").
    pub label: String,
}

/// The structural-context premise: the turn at `index` prefixed by its immediately preceding
/// turn (uniform, content-blind — cross-turn decisions get their antecedent without cue rules).
pub fn judgment_premise(turns: &[Turn], index: usize) -> String {
    let _ = (turns, index);
    todo!("verifier runtime impl commit")
}

/// Exhaustive decision-vs-discussion items over a corpus: one positive per gold decision span,
/// one negative per authored non-decision, one negative per turn containing neither a gold span
/// nor an authored negative quote.
pub fn decision_judgment_items(sessions: &[SessionRecord]) -> Vec<JudgmentItem> {
    let _ = sessions;
    todo!("verifier runtime impl commit")
}

/// The session's oracle evidence packet: every gold decision's evidence span, joined.
pub fn oracle_packet(session: &SessionRecord) -> String {
    let _ = session;
    todo!("verifier runtime impl commit")
}

/// One basis-support judgment item: is `claim` supported by `doc` (the oracle packet)?
#[derive(Debug, Clone, PartialEq)]
pub struct BasisItem {
    pub session_id: String,
    pub calibration: bool,
    pub doc: String,
    pub claim: String,
    /// Gold: supported (a real decision statement) vs unsupported (authored hard negative, or a
    /// cross-session stand-in when the session has no authored negatives).
    pub supported: bool,
}

/// Basis items per session: supported = each gold decision statement vs the session packet;
/// unsupported = authored `unsupported_claims`, or — when a session has none (the dev corpus) —
/// the chronologically NEXT session's decision statements (wrapping), judged against THIS
/// session's packet. Deterministic, no sampling choices.
pub fn basis_items(sessions: &[SessionRecord]) -> Vec<BasisItem> {
    let _ = sessions;
    todo!("verifier runtime impl commit")
}

/// Which windows carry at least one gold span (precision over windows reuses the frozen sweep
/// math on these flags).
pub fn window_gold_flags(windows: &[Window], gold_quotes: &[String]) -> Vec<bool> {
    let _ = (windows, gold_quotes);
    todo!("verifier runtime impl commit")
}

/// Gold-span recall at a threshold: a span is recalled iff SOME window containing it scores at
/// or above the threshold. Returns (recalled, total).
pub fn recall_at(
    windows: &[Window],
    scores: &[f32],
    gold_quotes: &[String],
    threshold: f32,
) -> (usize, usize) {
    let _ = (windows, scores, gold_quotes, threshold);
    todo!("verifier runtime impl commit")
}

/// Indices of gold spans contained in NO window at all — segmentation failures (window shapes,
/// not models), the P2 diagnostic split.
pub fn segmentation_failures(windows: &[Window], gold_quotes: &[String]) -> Vec<usize> {
    let _ = (windows, gold_quotes);
    todo!("verifier runtime impl commit")
}
