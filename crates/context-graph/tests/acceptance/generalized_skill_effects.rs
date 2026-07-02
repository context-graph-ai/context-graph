//! Apply layer 1 — the generalized post-generation repair store.
//!
//! The learned-skill store started with ONE effect kind: retype an entity. This suite drives the
//! three additional deterministic post-generation kinds a `cg fix`-class correction can teach, each
//! promoted from N aligned corrections of THAT KIND and each applied on a later capture through the
//! SAME terminal path the understanding-queue drainer uses (`run_capture`):
//!   - recategorize a decision's category (`decision_type`),
//!   - drop a harvested item,
//!   - add a tag to a harvested item.
//! Plus the forward-compatible withhold contract: a matched post-harvest skill this build cannot
//! apply is withheld (never applied, never counted), surfaced as a diagnostic, and capture proceeds.
//!
//! These are real store/capture paths — no deterministic double lives in `src/`. The corrections are
//! logged as teaching events; promotion reuses the existing threshold-aligned gate, generalized to
//! key on the correction KIND rather than entity-only. The learned effect is then applied on the next
//! real capture and observed by reading the STORED graph back.

use context_graph::harness::{CaptureHarnessConfig, run_capture};
use context_graph::ingest::CaptureRequest;
use context_graph::skill_store::{
    LearnedSkill, SkillApplyStrategy, SkillEffectEnvelope, SkillProvenance, apply_correction,
    list_skills_in_store, persist_skill,
};
use context_graph::{DecisionId, Store, StoreConfig};

use super::fake_extractor::open_store_with_lexical_embedder;
use super::refactor_extractor::SpanFaithfulExtractor;

fn fresh_store(tag: &str) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join(format!("{tag}.db")),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn capture(store: &Store, context_label: &str, transcript: &str) -> context_graph::ingest::CaptureReport {
    let (report, _harness) = run_capture(
        store,
        CaptureRequest {
            context_label: context_label.to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        &SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("capture must succeed");
    report
}

fn capture_with_harness(
    store: &Store,
    context_label: &str,
    transcript: &str,
) -> (
    context_graph::ingest::CaptureReport,
    context_graph::harness::CaptureHarnessReport,
) {
    run_capture(
        store,
        CaptureRequest {
            context_label: context_label.to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        &SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("capture must succeed")
}

fn decision_id(report: &context_graph::ingest::CaptureReport, key: &str) -> Option<DecisionId> {
    report
        .stable_ids
        .get(key)
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(DecisionId::from)
}

/// Teach the same correction N times through the real correction entrypoint. A single correction is
/// only a lesson; the threshold path promotes on the Nth aligned correction of that kind.
fn teach(store: &Store, content: &str, times: usize) -> context_graph::skill_store::CorrectionResult {
    let mut last = None;
    for index in 0..times {
        let item_id = format!("corrected-item-{index}");
        last = Some(apply_correction(store, &item_id, content, None));
    }
    last.expect("at least one correction")
}

// ── Recategorize a decision's category ─────────────────────────────────────────

#[test]
fn decision_recategorize_correction_promotes_and_autoapplies_on_capture() {
    let (_dir, store) = fresh_store("recategorize");
    // Three aligned corrections that a decision matching "postgresql" typed technology_choice should
    // be recategorized to infrastructure.
    let content = "recategorize-decision postgresql :: technology_choice -> infrastructure";
    let promoted = teach(&store, content, 3);
    assert!(
        promoted.new_skill_created,
        "the third aligned recategorize correction must promote a learned skill"
    );

    let skills = list_skills_in_store(&store);
    let learned = skills
        .iter()
        .find(|skill| {
            matches!(
                &skill.effect,
                Some(SkillEffectEnvelope::RecategorizeDecision { to_category, .. })
                    if to_category.eq_ignore_ascii_case("infrastructure")
            )
        })
        .expect("promotion must persist a RecategorizeDecision skill");
    assert_eq!(learned.apply_strategy, SkillApplyStrategy::PostHarvestRetype);

    // A later capture whose decision matches: the stored decision_type must be the learned category.
    let (report, harness) = capture_with_harness(
        &store,
        "recat-capture",
        "Goal G1: choose the storage engine\nDecision D1: use postgresql serves G1",
    );
    let stored = store
        .get_decision(decision_id(&report, "D1").expect("D1 id"))
        .expect("get_decision")
        .expect("decision stored");
    assert_eq!(
        stored.decision_type, "infrastructure",
        "the learned recategorize skill must rewrite the stored decision_type"
    );
    assert!(
        harness.applied_skill_ids.contains(&learned.id),
        "the recategorize skill must be recorded as applied"
    );
}

// ── Drop a harvested item ──────────────────────────────────────────────────────

#[test]
fn drop_correction_promotes_and_autoapplies_on_capture() {
    let (_dir, store) = fresh_store("drop");
    let content = "drop-item redis";
    let promoted = teach(&store, content, 3);
    assert!(
        promoted.new_skill_created,
        "three aligned drop corrections must promote a learned skill"
    );
    let skills = list_skills_in_store(&store);
    let learned = skills
        .iter()
        .find(|skill| matches!(&skill.effect, Some(SkillEffectEnvelope::DropItem { .. })))
        .expect("promotion must persist a DropItem skill");

    let (report, harness) = capture_with_harness(
        &store,
        "drop-capture",
        "Goal G1: pick a cache\nDecision D1: use postgresql serves G1\nDecision D2: use redis serves G1",
    );
    assert!(
        decision_id(&report, "D1").is_some(),
        "the non-matching decision must survive capture"
    );
    assert!(
        decision_id(&report, "D2").is_none(),
        "the matching decision must be dropped before storage"
    );
    assert!(
        harness.applied_skill_ids.contains(&learned.id),
        "the drop skill must be recorded as applied"
    );
}

// ── Add a tag to a harvested item ──────────────────────────────────────────────

#[test]
fn add_tag_correction_promotes_and_autoapplies_on_capture() {
    let (_dir, store) = fresh_store("addtag");
    let content = "add-tag postgresql :: reviewed";
    let promoted = teach(&store, content, 3);
    assert!(
        promoted.new_skill_created,
        "three aligned add-tag corrections must promote a learned skill"
    );
    let skills = list_skills_in_store(&store);
    let learned = skills
        .iter()
        .find(|skill| {
            matches!(
                &skill.effect,
                Some(SkillEffectEnvelope::AddTag { tag, .. }) if tag == "reviewed"
            )
        })
        .expect("promotion must persist an AddTag skill");

    let (report, _harness) = capture_with_harness(
        &store,
        "addtag-capture",
        "Goal G1: choose the storage engine\nDecision D1: use postgresql serves G1",
    );
    let stored = store
        .get_decision(decision_id(&report, "D1").expect("D1 id"))
        .expect("get_decision")
        .expect("decision stored");
    assert!(
        stored.tags.iter().any(|tag| tag == "reviewed"),
        "the learned add-tag skill must add the tag to the stored decision: {:?}",
        stored.tags
    );
    let _ = learned;
}

// ── Withhold contract: a matched effect this build cannot apply ────────────────

#[test]
fn unapplicable_matched_skill_is_withheld_not_applied_and_capture_proceeds() {
    let (_dir, store) = fresh_store("withhold");
    // A post-harvest skill whose effect this build cannot derive (an unrecognized/absent effect kind
    // deserializes to None, and the rule is not a retype rule). It matches the chunk by description,
    // so selection reaches it — and must WITHHOLD it, never apply it, and let capture succeed.
    let skill = LearnedSkill {
        id: "skill-unknown-gizmo".to_string(),
        description: "handle the gizmo widget in some future way".to_string(),
        rule: "an effect rule this build does not implement for gizmo".to_string(),
        effect: None,
        apply_strategy: SkillApplyStrategy::PostHarvestRetype,
        provenance: SkillProvenance::LlmAuthored,
        active: true,
        held_for_review: false,
        fire_right: 0,
        fire_wrong: 0,
    };
    assert!(persist_skill(&store, &skill), "seed skill must persist");

    let (_report, harness) =
        capture_with_harness(&store, "withhold-capture", "Goal G1: ship the gizmo widget today");
    assert!(
        harness.withheld_skill_ids.contains(&skill.id),
        "an unapplicable matched skill must be withheld and surfaced: {:?}",
        harness.withheld_skill_ids
    );
    assert!(
        !harness.applied_skill_ids.contains(&skill.id),
        "a withheld skill must never be counted as applied"
    );
}
