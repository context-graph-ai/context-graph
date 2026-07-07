//! Probe item construction — the deterministic layer between corpus and verifier. P1 items are
//! EXHAUSTIVE: positives are the gold evidence spans (with structural turn context — the
//! containing turn plus its predecessor, a content-blind rule applied uniformly), negatives are
//! the authored non-decisions plus every turn carrying no gold span. Basis items judge claims
//! against the session's oracle packet (gold spans only — no retrieval, no scanning). P2
//! evaluation primitives attribute failure to judgment (span present in a window, scored low)
//! vs segmentation (span never isolated into any window).

use crate::corpus::{SessionRecord, Turn};
use crate::windows::Window;
use cg_bench::scoring::SimilarityProvider;

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
    let current = turns[index].rendered();
    if index == 0 {
        current
    } else {
        format!("{}\n{}", turns[index - 1].rendered(), current)
    }
}

/// Item construction report: the items plus what the restatement exclusion removed.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionItems {
    pub items: Vec<JudgmentItem>,
    /// Turns excluded from negatives because they statement-match (frozen band, Match) a gold
    /// decision — assistant restatements are dedup events, not false positives (working-doc D8).
    pub restatement_negatives_excluded: usize,
    /// Turns in the gray band against some gold decision: excluded from items entirely.
    pub gray_negatives_excluded: usize,
}

/// Exhaustive decision-vs-discussion items over a corpus: one positive per gold decision span,
/// one negative per authored non-decision, one negative per turn containing no gold span AND
/// not statement-matching any gold decision (the frozen matcher decides; no cue rules).
pub fn decision_judgment_items(
    sessions: &[SessionRecord],
    sim: &dyn SimilarityProvider,
) -> DecisionItems {
    let _ = (sessions, sim);
    todo!("D8/D10 corrective impl commit")
}

/// The session's oracle evidence packet: the DISTINCT turns containing the gold decision
/// spans, in turn order (gold-anchored context — still no retrieval, no scanning); a span found
/// in no turn contributes the bare quote (working-doc D10).
pub fn oracle_packet(session: &SessionRecord) -> String {
    let _ = session;
    todo!("D8/D10 corrective impl commit")
}

/// The attachment premise for a decision: the same structural-context rule as P1-a (containing
/// turn + predecessor, working-doc D7/D9); unlocatable evidence falls back to the quote, a
/// decision with no quote falls back to its statement.
pub fn attachment_premise(session: &SessionRecord, decision_index: usize) -> String {
    let _ = (session, decision_index);
    todo!("D8/D10 corrective impl commit")
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
    let mut items = Vec::new();
    for (index, session) in sessions.iter().enumerate() {
        let doc = oracle_packet(session);
        if doc.is_empty() {
            continue;
        }
        for decision in &session.gold.decisions {
            items.push(BasisItem {
                session_id: session.id.clone(),
                calibration: session.calibration,
                doc: doc.clone(),
                claim: decision.statement.clone(),
                supported: true,
            });
        }
        let unsupported: Vec<String> = if session.gold.unsupported_claims.is_empty() {
            if sessions.len() < 2 {
                Vec::new()
            } else {
                let next = &sessions[(index + 1) % sessions.len()];
                next.gold
                    .decisions
                    .iter()
                    .map(|d| d.statement.clone())
                    .collect()
            }
        } else {
            session
                .gold
                .unsupported_claims
                .iter()
                .map(|c| c.claim.clone())
                .collect()
        };
        for claim in unsupported {
            items.push(BasisItem {
                session_id: session.id.clone(),
                calibration: session.calibration,
                doc: doc.clone(),
                claim,
                supported: false,
            });
        }
    }
    items
}

/// Which windows carry at least one gold span (precision over windows reuses the frozen sweep
/// math on these flags).
pub fn window_gold_flags(windows: &[Window], gold_quotes: &[String]) -> Vec<bool> {
    use crate::windows::window_contains;
    windows
        .iter()
        .map(|w| gold_quotes.iter().any(|q| window_contains(w, q)))
        .collect()
}

/// Gold-span recall at a threshold: a span is recalled iff SOME window containing it scores at
/// or above the threshold. Returns (recalled, total).
pub fn recall_at(
    windows: &[Window],
    scores: &[f32],
    gold_quotes: &[String],
    threshold: f32,
) -> (usize, usize) {
    use crate::windows::window_contains;
    let recalled = gold_quotes
        .iter()
        .filter(|quote| {
            windows
                .iter()
                .zip(scores)
                .any(|(w, &s)| s >= threshold && window_contains(w, quote))
        })
        .count();
    (recalled, gold_quotes.len())
}

/// Indices of gold spans contained in NO window at all — segmentation failures (window shapes,
/// not models), the P2 diagnostic split.
pub fn segmentation_failures(windows: &[Window], gold_quotes: &[String]) -> Vec<usize> {
    use crate::windows::window_contains;
    gold_quotes
        .iter()
        .enumerate()
        .filter(|(_, quote)| !windows.iter().any(|w| window_contains(w, quote)))
        .map(|(i, _)| i)
        .collect()
}
