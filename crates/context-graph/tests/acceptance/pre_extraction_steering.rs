//! Acceptance tests for WP2 apply layer 2 — the generation-time steering seam.
//!
//! The contract: at the queue-drain call site, matched `PreExtractionGuidance` rules plus retrieved
//! few-shot correction examples are assembled into a steering block appended to the extraction
//! prompt, HARD-CAPPED and deduped (the injected-corpus-decay guard). A deterministic router sends
//! structural/type corrections to the post-generation repair layer and omission/judgment corrections
//! to this generation-time layer. The block is empty when the skills master switch is off.

use context_graph::skill_store::{
    ApplyLayer, LearnedSkill, MAX_STEERING_ITEMS, SkillApplyStrategy, SkillProvenance,
    TeachingEventKind, assemble_pre_extraction_steering, persist_skill, route_teaching_event,
    set_skills_enabled,
};
use context_graph::{Store, StoreConfig};

use super::fake_extractor::open_store_with_lexical_embedder;

fn fresh_store(dir: &std::path::Path) -> Store {
    open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    })
}

fn guidance_skill(id: &str, description: &str, rule: &str) -> LearnedSkill {
    LearnedSkill {
        id: id.to_string(),
        description: description.to_string(),
        rule: rule.to_string(),
        effect: None,
        apply_strategy: SkillApplyStrategy::PreExtractionGuidance,
        provenance: SkillProvenance::HumanAuthored,
        active: true,
        held_for_review: false,
        fire_right: 0,
        fire_wrong: 0,
    }
}

#[test]
fn router_sends_corrections_to_generation_time_and_type_fixes_to_post_generation() {
    assert_eq!(
        route_teaching_event(&TeachingEventKind::Correction),
        ApplyLayer::GenerationTime,
        "an omission/judgment correction steers the model at generation time"
    );
    assert_eq!(
        route_teaching_event(&TeachingEventKind::WrongTypeCorrection),
        ApplyLayer::PostGeneration,
        "a structural type fix is replayed by the deterministic post-generation layer"
    );
    assert_eq!(
        route_teaching_event(&TeachingEventKind::Reversal),
        ApplyLayer::PostGeneration
    );
    assert_eq!(
        route_teaching_event(&TeachingEventKind::GroupingReversal),
        ApplyLayer::PostGeneration
    );
}

#[test]
fn steering_block_caps_and_dedups_matched_guidance() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = fresh_store(dir.path());
    // Seven distinct matching guidance rules plus one duplicate rule — all share the token "sqlite".
    for index in 0..7 {
        assert!(
            persist_skill(
                &store,
                &guidance_skill(
                    &format!("guide-{index}"),
                    "guidance about the sqlite cache",
                    &format!("prefer the sqlite cache decision variant {index}"),
                ),
            )
            .expect("read succeeds")
        );
    }
    assert!(
        persist_skill(
            &store,
            &guidance_skill(
                "guide-dup",
                "guidance about the sqlite cache",
                "prefer the sqlite cache decision variant 0",
            ),
        )
        .expect("read succeeds")
    );

    let chunk = "User: we chose the sqlite cache for the local store";
    let block = assemble_pre_extraction_steering(&store, chunk, true, false)
        .expect("read succeeds")
        .expect("matched guidance produces a steering block");
    let item_count = block.matches("\n- ").count();
    assert!(
        item_count <= MAX_STEERING_ITEMS,
        "the steering block is capped at {MAX_STEERING_ITEMS} items, got {item_count}: {block}"
    );
    assert!(item_count >= 1, "at least one guidance item lands: {block}");
    assert!(
        block.contains("Apply these learned corrections"),
        "the block leads with the steering instruction: {block}"
    );
    // The duplicate rule appears only once.
    assert_eq!(
        block.matches("sqlite cache decision variant 0").count(),
        1,
        "duplicate rules are deduped: {block}"
    );
}

#[test]
fn steering_is_empty_when_skills_are_off_or_nothing_matches() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = fresh_store(dir.path());
    assert!(
        persist_skill(
            &store,
            &guidance_skill(
                "guide-1",
                "guidance about the sqlite cache",
                "prefer the sqlite cache decision",
            ),
        )
        .expect("read succeeds")
    );
    let chunk = "User: we chose the sqlite cache for the local store";

    // Master switch off → no steering, even with a matching rule.
    set_skills_enabled(&store, false);
    assert!(
        assemble_pre_extraction_steering(&store, chunk, true, false)
            .expect("read succeeds")
            .is_none()
    );

    // Back on, but an unrelated chunk matches nothing.
    set_skills_enabled(&store, true);
    assert!(
        assemble_pre_extraction_steering(
            &store,
            "User: we picked postgres for durability",
            true,
            false
        )
        .expect("read succeeds")
        .is_none()
    );
    // The caller can also gate it off directly.
    assert!(
        assemble_pre_extraction_steering(&store, chunk, false, false)
            .expect("read succeeds")
            .is_none()
    );
    // And a matching chunk with skills on does produce a block.
    assert!(
        assemble_pre_extraction_steering(&store, chunk, true, false)
            .expect("read succeeds")
            .is_some()
    );
}
