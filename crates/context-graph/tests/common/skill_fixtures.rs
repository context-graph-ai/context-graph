//! Test builders for skill-store types.
//! These are not in prod; they live in test support and create realistic skill/teaching-event shapes.

use context_graph::skill_store::{LearnedSkill, SkillProvenance};

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
