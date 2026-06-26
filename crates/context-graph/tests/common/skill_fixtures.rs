//! Test builders for skill-store types.
//! These are not in prod; they live in test support and create realistic skill/teaching-event shapes.

use context_graph::skill_store::{LearnedSkill, SkillProvenance, TeachingEvent, TeachingEventKind};

/// A skill inferred by the LLM from correction patterns (LlmAuthored provenance).
pub fn llm_authored_skill(id: &str, description: &str) -> LearnedSkill {
    LearnedSkill {
        id: id.to_string(),
        description: description.to_string(),
        rule: format!("auto-inferred: {}", description),
        provenance: SkillProvenance::LlmAuthored,
        active: true,
        held_for_review: false,
        fire_right: 0,
        fire_wrong: 0,
    }
}

/// A correction teaching event: user corrected `item_id` to a new text.
pub fn correction_event(
    id: &str,
    skill_id: &str,
    item_id: &str,
    corrected_to: &str,
) -> TeachingEvent {
    TeachingEvent {
        id: id.to_string(),
        skill_id: skill_id.to_string(),
        corrected_item_id: item_id.to_string(),
        kind: TeachingEventKind::Correction,
        content: corrected_to.to_string(),
    }
}

/// A reversal teaching event: user reversed a previously applied correction.
pub fn reversal_event(id: &str, skill_id: &str, item_id: &str) -> TeachingEvent {
    TeachingEvent {
        id: id.to_string(),
        skill_id: skill_id.to_string(),
        corrected_item_id: item_id.to_string(),
        kind: TeachingEventKind::Reversal,
        content: "".to_string(),
    }
}

/// Build a set of correction labels (teaching events) that are enough to pass the held-out gate.
/// Returns (positive_labels, heldout_labels). The default teaching events include 5 positive and 1
/// held-out — enough for the gate to promote a skill in the CORRECT implementation (which computes
/// precision, not count). The placeholder always returns None from `try_promote_skill`.
pub fn gate_passing_labels(skill_id: &str) -> (Vec<TeachingEvent>, Vec<TeachingEvent>) {
    let training = vec![
        correction_event("ev1", skill_id, "item1", "technology_choice"),
        correction_event("ev2", skill_id, "item2", "technology_choice"),
        correction_event("ev3", skill_id, "item3", "technology_choice"),
        correction_event("ev4", skill_id, "item4", "technology_choice"),
    ];
    let heldout = vec![correction_event(
        "ev5",
        skill_id,
        "item5",
        "technology_choice",
    )];
    (training, heldout)
}

/// Build a GENERAL candidate's labels: training corrections that all assert the same target AND a
/// disjoint held-out slice whose ground-truth target is the SAME — the rule holds on unseen events,
/// so the held-out gate must pass it. Returns (training, heldout). Content-faithful: the held-out
/// events agree with what the training rule predicts.
pub fn general_candidate_labels(skill_id: &str) -> (Vec<TeachingEvent>, Vec<TeachingEvent>) {
    let training = vec![
        correction_event("gen1", skill_id, "g1", "technology_choice"),
        correction_event("gen2", skill_id, "g2", "technology_choice"),
        correction_event("gen3", skill_id, "g3", "technology_choice"),
        correction_event("gen4", skill_id, "g4", "technology_choice"),
    ];
    // Held-out (unseen) events whose ground truth ALSO is technology_choice → the rule generalizes.
    let heldout = vec![
        correction_event("genh1", skill_id, "g5", "technology_choice"),
        correction_event("genh2", skill_id, "g6", "technology_choice"),
    ];
    (training, heldout)
}

/// Build an OVERFIT candidate's labels: training corrections that all assert one target (fitting the
/// training set perfectly) but a disjoint held-out slice whose ground-truth target is DIFFERENT —
/// the rule memorizes the training content and FAILS to generalize, so the held-out gate must reject
/// it. Returns (training, heldout). Content-faithful: the held-out events disagree with what the
/// training rule predicts, which is exactly what overfitting looks like on unseen data.
pub fn overfit_candidate_labels(skill_id: &str) -> (Vec<TeachingEvent>, Vec<TeachingEvent>) {
    let training = vec![
        correction_event("ovf1", skill_id, "o1", "technology_choice"),
        correction_event("ovf2", skill_id, "o2", "technology_choice"),
        correction_event("ovf3", skill_id, "o3", "technology_choice"),
        correction_event("ovf4", skill_id, "o4", "technology_choice"),
    ];
    // Held-out events whose ground truth is code_reference, NOT technology_choice → the training
    // rule predicts the wrong value on every unseen event (overfit, fails to generalize).
    let heldout = vec![
        correction_event("ovfh1", skill_id, "o5", "code_reference"),
        correction_event("ovfh2", skill_id, "o6", "code_reference"),
    ];
    (training, heldout)
}

/// Build correction labels that should NOT pass the gate (too many reversals for precision).
pub fn gate_failing_labels(skill_id: &str) -> (Vec<TeachingEvent>, Vec<TeachingEvent>) {
    let training = vec![
        correction_event("ef1", skill_id, "item1", "technology_choice"),
        reversal_event("ef2", skill_id, "item1"), // reversed → wrong
        correction_event("ef3", skill_id, "item2", "technology_choice"),
        reversal_event("ef4", skill_id, "item2"), // reversed → wrong
    ];
    let heldout = vec![
        reversal_event("ef5", skill_id, "item3"), // wrong on heldout too
    ];
    (training, heldout)
}
