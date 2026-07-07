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
    let current = turns[index].rendered();
    if index == 0 {
        current
    } else {
        format!("{}\n{}", turns[index - 1].rendered(), current)
    }
}

/// Exhaustive decision-vs-discussion items over a corpus: one positive per gold decision span,
/// one negative per authored non-decision, one negative per turn containing neither a gold span
/// nor an authored negative quote.
pub fn decision_judgment_items(sessions: &[SessionRecord]) -> Vec<JudgmentItem> {
    use crate::windows::normalize_ws;
    let mut items = Vec::new();
    for session in sessions {
        let normalized_turns: Vec<String> = session
            .turns
            .iter()
            .map(|t| normalize_ws(&t.rendered()))
            .collect();
        let containing_turn = |quote: &str| -> Option<usize> {
            let quote = normalize_ws(quote);
            if quote.is_empty() {
                return None;
            }
            normalized_turns.iter().position(|t| t.contains(&quote))
        };
        let mut used_turns: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for decision in &session.gold.decisions {
            let Some(quote) = &decision.evidence_quote else {
                continue;
            };
            let (premise, label) = match containing_turn(quote) {
                Some(idx) => {
                    used_turns.insert(idx);
                    (judgment_premise(&session.turns, idx), "gold-span")
                }
                None => (quote.clone(), "gold-span-fallback"),
            };
            items.push(JudgmentItem {
                session_id: session.id.clone(),
                calibration: session.calibration,
                premise,
                positive: true,
                label: label.to_string(),
            });
        }
        for negative in &session.gold.non_decisions {
            let (premise, label) = match containing_turn(&negative.quote) {
                Some(idx) => {
                    used_turns.insert(idx);
                    (judgment_premise(&session.turns, idx), "non-decision")
                }
                None => (negative.quote.clone(), "non-decision-fallback"),
            };
            items.push(JudgmentItem {
                session_id: session.id.clone(),
                calibration: session.calibration,
                premise,
                positive: false,
                label: label.to_string(),
            });
        }
        for idx in 0..session.turns.len() {
            if used_turns.contains(&idx) {
                continue;
            }
            items.push(JudgmentItem {
                session_id: session.id.clone(),
                calibration: session.calibration,
                premise: judgment_premise(&session.turns, idx),
                positive: false,
                label: "turn".to_string(),
            });
        }
    }
    items
}

/// The session's oracle evidence packet: every gold decision's evidence span, joined.
pub fn oracle_packet(session: &SessionRecord) -> String {
    session
        .gold
        .decisions
        .iter()
        .filter_map(|d| d.evidence_quote.as_deref())
        .collect::<Vec<_>>()
        .join("\n")
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
