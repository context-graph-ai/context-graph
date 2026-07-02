//! Acceptance tests for WP2 apply layer 1 — the generalized post-generation repair effects.
//!
//! Beyond entity retype, a learned skill can recategorize a decision's `decision_type`, drop a
//! wrongly-captured item, or add a tag — applied deterministically post-harvest on the same capture
//! seam as retype. A skill this build cannot run (an unknown apply strategy) is withheld verbatim,
//! never coerced, and capture proceeds unchanged. Matching is literal-text keyed (identity-based
//! matching is a named deferred item).

use context_graph::ingest::{
    CaptureRequest, ConversationExtractor, DecisionSeed, ExtractionRequest, GoalSeed,
    ModelExtractedField, StructuredHarvest, memory_counts, record_conversation_with_extractor,
};
use context_graph::skill_store::{
    LearnedSkill, SkillApplyStrategy, SkillEffectEnvelope, SkillProvenance, persist_skill,
};
use context_graph::{ContextId, DecisionId, Store, StoreConfig};

use super::fake_extractor::open_store_with_lexical_embedder;

const TRANSCRIPT: &str = "we chose redis for the shared cache to cut latency";

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("cg.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

/// A deterministic extractor that yields one goal and one redis-cache decision typed
/// `infrastructure`, so a recategorize / drop / tag repair has a concrete item to act on.
struct RedisCacheExtractor;

impl ConversationExtractor for RedisCacheExtractor {
    fn extract(&self, _request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        Ok(StructuredHarvest {
            summary: Some("redis cache decision".to_string()),
            goals: vec![GoalSeed {
                key: "G1".to_string(),
                text: "cut cache latency".to_string(),
                context_id: None,
            }],
            decisions: vec![DecisionSeed {
                key: "D1".to_string(),
                text: "use redis for the shared cache".to_string(),
                goal_key: Some("G1".to_string()),
                decision_type: Some(ModelExtractedField::from("infrastructure")),
                ..Default::default()
            }],
            ..Default::default()
        })
    }
}

fn capture(store: &Store, context: &str) -> context_graph::ingest::CaptureReport {
    record_conversation_with_extractor(
        store,
        CaptureRequest {
            context_label: context.to_string(),
            transcript: TRANSCRIPT.to_string(),
            model_available: true,
        },
        &RedisCacheExtractor,
    )
    .expect("capture succeeded")
}

fn decision_id_from_stable(
    report: &context_graph::ingest::CaptureReport,
    key: &str,
) -> Option<DecisionId> {
    report
        .stable_ids
        .get(key)
        .and_then(|s| uuid::Uuid::parse_str(s).ok())
        .map(DecisionId::from)
}

fn context_id(store: &Store, name: &str) -> ContextId {
    store
        .list_contexts()
        .expect("contexts")
        .into_iter()
        .find(|c| c.name == name)
        .expect("context")
        .id
}

fn skill(id: &str, effect: SkillEffectEnvelope, strategy: SkillApplyStrategy) -> LearnedSkill {
    LearnedSkill {
        id: id.to_string(),
        description: "repair the redis cache decision".to_string(),
        rule: "learned repair for the redis decision".to_string(),
        effect: Some(effect),
        apply_strategy: strategy,
        provenance: SkillProvenance::HumanAuthored,
        active: true,
        held_for_review: false,
        fire_right: 0,
        fire_wrong: 0,
    }
}

#[test]
fn recategorize_decision_effect_applies_post_harvest() {
    let (_dir, store) = fresh_store();
    assert!(persist_skill(
        &store,
        &skill(
            "recat-1",
            SkillEffectEnvelope::RecategorizeDecision {
                decision_matcher: "redis".to_string(),
                from_category: "infrastructure".to_string(),
                to_category: "technology_choice".to_string(),
            },
            SkillApplyStrategy::PostHarvestRetype,
        ),
    ));
    let report = capture(&store, "recat");
    let decision = store
        .get_decision(decision_id_from_stable(&report, "D1").expect("D1"))
        .expect("get decision")
        .expect("decision exists");
    assert_eq!(
        decision.decision_type, "technology_choice",
        "the learned recategorize repair changed the decision_type"
    );
}

#[test]
fn add_tag_effect_applies_post_harvest() {
    let (_dir, store) = fresh_store();
    assert!(persist_skill(
        &store,
        &skill(
            "tag-1",
            SkillEffectEnvelope::AddTag {
                item_matcher: "redis".to_string(),
                tag: "reviewed".to_string(),
            },
            SkillApplyStrategy::PostHarvestRetype,
        ),
    ));
    let report = capture(&store, "tag");
    let decision = store
        .get_decision(decision_id_from_stable(&report, "D1").expect("D1"))
        .expect("get decision")
        .expect("decision exists");
    assert!(
        decision.tags.iter().any(|t| t == "reviewed"),
        "the learned add-tag repair tagged the decision: {:?}",
        decision.tags
    );
}

#[test]
fn drop_item_effect_removes_the_matched_decision() {
    let (_dir, store) = fresh_store();
    assert!(persist_skill(
        &store,
        &skill(
            "drop-1",
            SkillEffectEnvelope::DropItem {
                item_matcher: "redis".to_string(),
            },
            SkillApplyStrategy::PostHarvestRetype,
        ),
    ));
    capture(&store, "drop");
    let counts = memory_counts(&store, context_id(&store, "drop")).expect("counts");
    assert_eq!(
        counts.decisions, 0,
        "the learned drop repair removed the wrongly-captured decision"
    );
}

#[test]
fn unknown_apply_strategy_is_withheld_and_capture_is_unchanged() {
    let (_dir, store) = fresh_store();
    // A known DROP effect carried under an apply strategy this build does not implement must be
    // withheld verbatim — never coerced into running — so the decision is NOT dropped.
    assert!(persist_skill(
        &store,
        &skill(
            "withheld-1",
            SkillEffectEnvelope::DropItem {
                item_matcher: "redis".to_string(),
            },
            SkillApplyStrategy::Unknown("generation-time-v2".to_string()),
        ),
    ));
    capture(&store, "withheld");
    let counts = memory_counts(&store, context_id(&store, "withheld")).expect("counts");
    assert_eq!(
        counts.decisions, 1,
        "a withheld skill never runs; capture is unchanged"
    );
}
