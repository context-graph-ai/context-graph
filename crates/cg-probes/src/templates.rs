//! Hypothesis templates — MEASURED CONTENT, not deterministic contract (the harness test wall's
//! stated boundary: prompt/template content gets no content-assertion tests). Frozen at
//! calibration time and held identical across every candidate model, threshold, and corpus;
//! any post-verdict change is an owner decision. The application MECHANISM (max over the set)
//! is test-walled via the scoring machinery.

/// Decision-vs-discussion hypotheses; an item's score is the MAX over the set.
pub const DECISION_TEMPLATES: &[&str] = &[
    "A definite decision, commitment, or rule was made here.",
    "The speaker committed to a specific course of action.",
];

/// Attachment hypothesis; `{goal}` is replaced with the intention statement.
pub const ATTACHMENT_TEMPLATE: &str =
    "This was decided in order to achieve the following goal: {goal}";

pub fn attachment_hypothesis(intention_statement: &str) -> String {
    ATTACHMENT_TEMPLATE.replace("{goal}", intention_statement)
}

/// The gray-zone judge prompt: the cg-correctness frozen prompt with the one pre-registered
/// generalization ("software-engineering work session" → "work session") so the identical fixed
/// prompt serves both corpora. Temperature 0, hosted.
pub const JUDGE_PROMPT: &str = "You judge whether two statements from a work session describe \
THE SAME decision/goal, allowing paraphrase, or different ones. Answer with exactly one word: \
SAME or DIFFERENT.\n\nStatement A: {a}\n\nStatement B: {b}\n\nAnswer:";

pub const JUDGE_MODEL: &str = "anthropic/claude-haiku-4.5";
