//! Attachment pair generation (P1-b / P3-known) and attachment scoring. Distractors are
//! EXHAUSTIVE — every decision is paired with every intention in the corpus roster — so
//! precision is real, never sampled. A decision with no serving intention pairs with the whole
//! roster with no gold pair: the correct behavior is abstention.

use crate::corpus::SessionRecord;

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
    let _ = sessions;
    todo!("probe impl commit")
}

/// Every gold decision × every roster intention.
pub fn attachment_pairs(sessions: &[SessionRecord]) -> Vec<AttachPair> {
    let _ = sessions;
    todo!("probe impl commit")
}

/// P3-known scoring: `(chosen, gold)` per decision, where None = abstained / no serving
/// intention. Correct iff equal (abstaining on a none-fits case is correct; abstaining on an
/// attached case, or attaching a none-fits case, is wrong).
pub fn score_known_attachment(choices: &[(Option<String>, Option<String>)]) -> f32 {
    let _ = choices;
    todo!("probe impl commit")
}
