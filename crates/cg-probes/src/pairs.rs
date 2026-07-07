//! Attachment pair generation (P1-b / P3-known) and attachment scoring. Distractors are
//! EXHAUSTIVE — every decision is paired with every intention in the corpus roster — so
//! precision is real, never sampled. A decision with no serving intention pairs with the whole
//! roster with no gold pair: the correct behavior is abstention.

use crate::corpus::SessionRecord;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub struct AttachPair {
    pub session_id: String,
    pub decision_index: usize,
    pub decision_statement: String,
    pub decision_evidence: Option<String>,
    pub intention_key: String,
    pub intention_statement: String,
    pub is_gold: bool,
}

/// The corpus intention roster: (key, statement) deduplicated by key in first-occurrence order.
pub fn intention_roster(sessions: &[SessionRecord]) -> Vec<(String, String)> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut roster = Vec::new();
    for session in sessions {
        for intention in &session.gold.intentions {
            if seen.insert(intention.key.clone()) {
                roster.push((intention.key.clone(), intention.statement.clone()));
            }
        }
    }
    roster
}

/// Every gold decision × every roster intention.
pub fn attachment_pairs(sessions: &[SessionRecord]) -> Vec<AttachPair> {
    let roster = intention_roster(sessions);
    let mut pairs = Vec::new();
    for session in sessions {
        for (decision_index, decision) in session.gold.decisions.iter().enumerate() {
            for (key, statement) in &roster {
                pairs.push(AttachPair {
                    session_id: session.id.clone(),
                    decision_index,
                    decision_statement: decision.statement.clone(),
                    decision_evidence: decision.evidence_quote.clone(),
                    intention_key: key.clone(),
                    intention_statement: statement.clone(),
                    is_gold: decision.serves.as_deref() == Some(key.as_str()),
                });
            }
        }
    }
    pairs
}

/// P3-known scoring: `(chosen, gold)` per decision, where None = abstained / no serving
/// intention. Correct iff equal (abstaining on a none-fits case is correct; abstaining on an
/// attached case, or attaching a none-fits case, is wrong).
pub fn score_known_attachment(choices: &[(Option<String>, Option<String>)]) -> f32 {
    if choices.is_empty() {
        return 0.0;
    }
    let correct = choices
        .iter()
        .filter(|(chosen, gold)| chosen == gold)
        .count();
    correct as f32 / choices.len() as f32
}
