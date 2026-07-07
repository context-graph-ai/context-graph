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
    use crate::windows::normalize_ws;
    use cg_bench::scoring::{MatchBand, band};
    let mut items = Vec::new();
    let (mut restatement_excluded, mut gray_excluded) = (0usize, 0usize);
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
        for (idx, turn) in session.turns.iter().enumerate() {
            if used_turns.contains(&idx) {
                continue;
            }
            // Restatement exclusion (D8): a turn that statement-matches any gold decision on
            // the frozen band is a dedup event, never a negative; gray-band turns are excluded
            // from items entirely and counted.
            let best_band = session
                .gold
                .decisions
                .iter()
                .map(|d| band(sim.similarity(&turn.text, &d.statement)))
                .fold(MatchBand::NonMatch, |acc, b| match (acc, b) {
                    (MatchBand::Match, _) | (_, MatchBand::Match) => MatchBand::Match,
                    (MatchBand::GrayZone, _) | (_, MatchBand::GrayZone) => MatchBand::GrayZone,
                    _ => MatchBand::NonMatch,
                });
            match best_band {
                MatchBand::Match => {
                    restatement_excluded += 1;
                    continue;
                }
                MatchBand::GrayZone => {
                    gray_excluded += 1;
                    continue;
                }
                MatchBand::NonMatch => {}
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
    DecisionItems {
        items,
        restatement_negatives_excluded: restatement_excluded,
        gray_negatives_excluded: gray_excluded,
    }
}

/// The session's oracle evidence packet: the DISTINCT turns containing the gold decision
/// spans, in turn order (gold-anchored context — still no retrieval, no scanning); a span found
/// in no turn contributes the bare quote (working-doc D10).
pub fn oracle_packet(session: &SessionRecord) -> String {
    use crate::windows::normalize_ws;
    let normalized_turns: Vec<String> = session
        .turns
        .iter()
        .map(|t| normalize_ws(&t.rendered()))
        .collect();
    let mut turn_indices: Vec<usize> = Vec::new();
    let mut orphan_quotes: Vec<&str> = Vec::new();
    for decision in &session.gold.decisions {
        let Some(quote) = decision.evidence_quote.as_deref() else {
            continue;
        };
        let needle = normalize_ws(quote);
        match normalized_turns
            .iter()
            .position(|t| !needle.is_empty() && t.contains(&needle))
        {
            Some(idx) => {
                if !turn_indices.contains(&idx) {
                    turn_indices.push(idx);
                }
            }
            None => orphan_quotes.push(quote),
        }
    }
    turn_indices.sort_unstable();
    let mut parts: Vec<String> = turn_indices
        .into_iter()
        .map(|idx| session.turns[idx].rendered())
        .collect();
    parts.extend(orphan_quotes.into_iter().map(str::to_string));
    parts.join("\n")
}

/// The attachment premise for a decision: the same structural-context rule as P1-a (containing
/// turn + predecessor, working-doc D7/D9); unlocatable evidence falls back to the quote, a
/// decision with no quote falls back to its statement.
pub fn attachment_premise(session: &SessionRecord, decision_index: usize) -> String {
    use crate::windows::normalize_ws;
    let Some(decision) = session.gold.decisions.get(decision_index) else {
        return String::new();
    };
    let Some(quote) = decision.evidence_quote.as_deref() else {
        return decision.statement.clone();
    };
    let needle = normalize_ws(quote);
    let containing = session
        .turns
        .iter()
        .position(|t| !needle.is_empty() && normalize_ws(&t.rendered()).contains(&needle));
    match containing {
        Some(idx) => judgment_premise(&session.turns, idx),
        None => quote.to_string(),
    }
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
