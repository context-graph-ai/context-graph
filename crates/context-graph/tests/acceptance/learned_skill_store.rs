// Learned-skill store and SKILL.md parser tests.
//
// Pure-logic tests: assert the SKILL.md parse/export contract — no store.

use std::collections::HashMap;

use context_graph::skill_store::{
    LearnedSkill, SkillApplyStrategy, SkillEffectEnvelope, SkillProvenance, TeachingEventKind,
    acknowledge_skill, apply_correction, export_skill, list_skills_in_store,
    parse_human_authored_skill, persist_skill, select_pre_extraction_guidance_skills_for_chunk,
    select_skills_for_chunk, set_review_hold, set_skills_enabled,
};
use context_graph::{Store, StoreConfig, TxId};
use contextdb_core::Value as DbValue;

use super::fake_extractor::open_store_with_lexical_embedder;
use super::skill_fixtures::llm_authored_skill;

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("skill_f.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn insert_skill_row(
    store: &Store,
    id: &str,
    description: &str,
    rule: &str,
    provenance: &str,
    active: bool,
    held_for_review: bool,
) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_learned_skills (id, description, rule, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("description".to_string(), DbValue::Text(description.to_string())),
            ("rule".to_string(), DbValue::Text(rule.to_string())),
            ("provenance".to_string(), DbValue::Text(provenance.to_string())),
            ("active".to_string(), DbValue::Bool(active)),
            ("held_for_review".to_string(), DbValue::Bool(held_for_review)),
            ("fire_right".to_string(), DbValue::Int64(0)),
            ("fire_wrong".to_string(), DbValue::Int64(0)),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("insert skill row");
}

/// Set a skill row's `apply_strategy` column directly to an arbitrary raw string, exactly as an
/// older edge would see it after syncing a newer edge's row that names a strategy this build does
/// not implement. Call after `insert_skill_row`.
fn set_apply_strategy_raw(store: &Store, id: &str, apply_strategy: &str) {
    let db = store.sync_database();
    db.execute(
        "UPDATE cg_learned_skills SET apply_strategy = $apply_strategy WHERE id = $id",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            (
                "apply_strategy".to_string(),
                DbValue::Text(apply_strategy.to_string()),
            ),
        ]),
    )
    .expect("set apply_strategy on skill row");
}

fn scan_skill_table(store: &Store) -> Vec<HashMap<String, DbValue>> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    db.scan("cg_learned_skills", snapshot)
        .expect("scan cg_learned_skills")
        .into_iter()
        .map(|row| row.values)
        .collect()
}

fn retype_effect(entity_name: &str, from_type: &str, to_type: &str) -> SkillEffectEnvelope {
    SkillEffectEnvelope::RetypeEntity {
        entity_name: entity_name.to_string(),
        from_type: from_type.to_string(),
        to_type: to_type.to_string(),
    }
}

#[test]
fn learned_skill_is_inspectable_skill_md_shaped_row_with_provenance() {
    let (_dir, store) = fresh_store();

    // Insert one skill of each provenance directly into the real table.
    for (id, provenance) in [
        ("skill-hc", "human-correction"),
        ("skill-la", "llm-authored"),
        ("skill-ha", "human-authored"),
    ] {
        insert_skill_row(
            &store,
            id,
            &format!("desc for {id}"),
            &format!("rule for {id}"),
            provenance,
            true,
            false,
        );
    }

    let skills = list_skills_in_store(&store).expect("read succeeds");
    assert_eq!(
        skills.len(),
        3,
        "list_skills_in_store must return all 3 persisted skills; got {}",
        skills.len()
    );

    // Each row must have a readable description and rule.
    for skill in &skills {
        assert!(
            !skill.description.is_empty(),
            "skill description must not be empty"
        );
        assert!(!skill.rule.is_empty(), "skill rule must not be empty");
    }

    // Provenance values must span all three kinds.
    let provenances: Vec<_> = skills.iter().map(|s| s.provenance.clone()).collect();
    assert!(
        provenances.contains(&SkillProvenance::HumanCorrection),
        "must include HumanCorrection provenance"
    );
    assert!(
        provenances.contains(&SkillProvenance::LlmAuthored),
        "must include LlmAuthored provenance"
    );
    assert!(
        provenances.contains(&SkillProvenance::HumanAuthored),
        "must include HumanAuthored provenance"
    );
}

#[test]
fn legacy_skill_row_derives_effect_and_default_strategy_from_rule() {
    let (_dir, store) = fresh_store();
    insert_skill_row(
        &store,
        "sk-legacy-retype",
        "retype context-graph file to repository",
        "when an entity named context-graph is typed File → Repository",
        "human-correction",
        true,
        false,
    );

    let skill = list_skills_in_store(&store)
        .expect("read succeeds")
        .into_iter()
        .find(|skill| skill.id == "sk-legacy-retype")
        .expect("legacy skill row must load");
    assert_eq!(
        skill.effect,
        Some(retype_effect("context-graph", "File", "Repository")),
        "a legacy row without an effect envelope must derive the structured retype effect from rule"
    );
    assert_eq!(
        skill.apply_strategy,
        SkillApplyStrategy::PostHarvestRetype,
        "a legacy row without an apply_strategy tag must default to the current post-harvest path"
    );
}

#[test]
fn pre_extraction_strategy_is_not_silently_applied_by_post_harvest_retype() {
    use context_graph::harness::{CaptureHarnessConfig, run_capture};
    use context_graph::ingest::CaptureRequest;

    let (_dir, store) = fresh_store();
    let transcript = "Goal G1: remember the project\nEntity E1: context-graph type=file\nDecision D1: keep one repo serves G1";
    assert!(
        persist_skill(
            &store,
            &LearnedSkill {
                id: "sk-pre-extraction".to_string(),
                description: "context-graph repository guidance".to_string(),
                rule: "when an entity named context-graph is typed File → Repository".to_string(),
                effect: Some(retype_effect("context-graph", "File", "Repository")),
                apply_strategy: SkillApplyStrategy::PreExtractionGuidance,
                provenance: SkillProvenance::LlmAuthored,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            }
        )
        .expect("read succeeds"),
        "pre-extraction fixture skill must persist"
    );

    let pre = select_pre_extraction_guidance_skills_for_chunk(&store, transcript, true, false)
        .expect("read succeeds");
    assert!(
        pre.iter().any(|skill| skill.id == "sk-pre-extraction"),
        "the pre-extraction selector must find the generation-time skill"
    );
    let post = select_skills_for_chunk(&store, transcript, true, false).expect("read succeeds");
    assert!(
        post.iter().all(|skill| skill.id != "sk-pre-extraction"),
        "the post-harvest selector must not silently apply a pre-extraction strategy skill"
    );

    let (_report, harness) = run_capture(
        &store,
        CaptureRequest {
            context_label: "pre-extraction-fold".to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("capture with pre-extraction skill must still succeed");
    assert!(
        harness.applied_skill_ids.is_empty(),
        "post-harvest capture must withhold the pre-extraction strategy skill"
    );
    assert_eq!(
        entity_type_named(&store, "context-graph"),
        Some(context_graph::EntityType::File),
        "withholding the pre-extraction strategy must leave the post-harvest entity type unchanged"
    );
}

// regression (fold-seam withhold contract): a stored row whose apply_strategy is a string this build does
// not implement (as an older edge would see after syncing a newer edge's rule) must never be
// coerced to an implemented strategy, must never apply, must never be counted in
// `applied_skill_ids`, and must surface as a named diagnostic on the capture harness report so
// capture never silently drops it.
#[test]
fn unknown_apply_strategy_is_withheld_not_applied_and_surfaces_diagnostic() {
    use context_graph::harness::{CaptureHarnessConfig, run_capture};
    use context_graph::ingest::CaptureRequest;

    let (_dir, store) = fresh_store();
    let transcript = "Goal G1: remember the project\nEntity E1: context-graph type=file\nDecision D1: keep one repo serves G1";

    insert_skill_row(
        &store,
        "sk-future-strategy",
        "retype a context-graph file entity to a repository",
        "when an entity named context-graph is typed File → Repository",
        "human-correction",
        true,
        false,
    );
    set_apply_strategy_raw(&store, "sk-future-strategy", "future-strategy");

    // Store round-trip: the raw strategy string must survive a read exactly, never coerced.
    let reloaded = list_skills_in_store(&store)
        .expect("read succeeds")
        .into_iter()
        .find(|skill| skill.id == "sk-future-strategy")
        .expect("row with an unsupported apply_strategy must still be readable");
    assert_eq!(
        reloaded.apply_strategy,
        SkillApplyStrategy::Unknown("future-strategy".to_string()),
        "an unknown apply_strategy string must be preserved verbatim on read, never coerced to an \
         implemented strategy; got {:?}",
        reloaded.apply_strategy
    );

    let (report, harness) = run_capture(
        &store,
        CaptureRequest {
            context_label: "future-strategy-fold".to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("capture must succeed even though the matching rule carries an unsupported strategy");

    assert!(
        harness
            .applied_skill_ids
            .iter()
            .all(|id| id != "sk-future-strategy"),
        "an unsupported apply strategy must never be applied; applied_skill_ids={:?}",
        harness.applied_skill_ids
    );
    assert!(
        harness
            .withheld_skill_ids
            .iter()
            .any(|id| id == "sk-future-strategy"),
        "an unsupported apply strategy must surface as a withheld diagnostic naming the skill id; \
         withheld_skill_ids={:?}",
        harness.withheld_skill_ids
    );
    assert_eq!(
        entity_type_named(&store, "context-graph"),
        Some(context_graph::EntityType::File),
        "withholding an unsupported-strategy rule must leave the harvested entity type unchanged"
    );
    assert!(
        report.wrote_digest || !report.stable_ids.is_empty(),
        "capture must never fail because of a withheld rule"
    );
}

#[test]
fn a_correction_is_a_label_not_a_skill() {
    let (_dir, store) = fresh_store();

    // Apply a single correction (no skill linked yet).
    let result =
        apply_correction(&store, "item-abc", "technology_choice", None).expect("read succeeds");

    // The correction should report as applied.
    assert!(result.applied, "correction must report as applied");

    assert!(
        result.teaching_event.is_some(),
        "a correction must log a teaching event; teaching_event was None"
    );

    let ev = result.teaching_event.as_ref().unwrap();
    assert_eq!(
        ev.corrected_item_id, "item-abc",
        "teaching event must reference the corrected item"
    );
    assert_eq!(
        ev.kind,
        TeachingEventKind::Correction,
        "teaching event kind must be Correction"
    );

    // No auto-applying skill must have been created.
    assert!(
        !result.new_skill_created,
        "a single correction must NOT create a new auto-applying skill"
    );

    // The skill table must remain empty — no promotion from one correction.
    let rows = scan_skill_table(&store);
    assert!(
        rows.is_empty(),
        "skill table must be empty after a single correction; found {} rows",
        rows.len()
    );
}

#[test]
fn human_authored_skill_bypasses_promotion_gate_parses_only() {
    // A well-formed SKILL.md shaped file.
    let well_formed = "# description: prefer technology_choice for library decisions\n## rule\nwhen the decision text mentions a library or framework → classify as technology_choice\n";
    let result = parse_human_authored_skill(well_formed);
    assert!(result.accepted, "a well-formed skill file must be accepted");
    let skill = result
        .skill
        .as_ref()
        .expect("accepted skill must be present");
    assert_eq!(
        skill.provenance,
        SkillProvenance::HumanAuthored,
        "human-authored skill must carry HumanAuthored provenance"
    );
    assert!(
        result.error.is_none(),
        "accepted skill must carry no parse error"
    );

    // A malformed file (missing required sections).
    let malformed = "not a skill file at all @@@";
    let bad = parse_human_authored_skill(malformed);
    assert!(
        !bad.accepted,
        "a malformed skill file must be rejected with a parse error"
    );
    assert!(
        bad.error.is_some(),
        "a malformed rejection must carry a named parse error message"
    );
    let error_msg = bad.error.as_deref().unwrap_or("");
    assert!(
        !error_msg.is_empty(),
        "parse error message must not be empty"
    );
    assert!(
        bad.skill.is_none(),
        "a rejected parse must not produce a skill"
    );
}

#[test]
fn human_authored_skill_whose_rule_would_fail_heldout_is_still_accepted() {
    let would_fail_gate = "# description: always classify as code_reference\n## rule\nwhen true → classify as code_reference\n";

    let result = parse_human_authored_skill(would_fail_gate);
    assert!(
        result.accepted,
        "a well-formed human-authored skill file must be accepted even if its rule would fail the gate"
    );

    let skill = result.skill.as_ref().expect("accepted skill present");
    assert_eq!(
        skill.provenance,
        SkillProvenance::HumanAuthored,
        "human-authored provenance must be set without running the promotion gate"
    );

    assert!(
        !skill.description.starts_with('#'),
        "description must be extracted cleanly from the `# description:` header, not include the `#`; got: {:?}",
        skill.description
    );
    assert!(
        !skill.description.is_empty(),
        "description must not be empty after extraction"
    );

    assert!(
        result.error.is_none(),
        "accepted human-authored skill parsing must not report an error"
    );
}

#[test]
fn exported_edited_skill_pushed_back_becomes_human_authored() {
    let original = llm_authored_skill("sk-export", "detect technology_choice decisions");

    let exported = export_skill(&original);
    assert!(
        !exported.is_empty(),
        "exported skill must produce non-empty text"
    );

    assert!(
        exported.starts_with("# description:"),
        "exported skill must start with `# description:` header; got: {:?}",
        exported.lines().next().unwrap_or("")
    );

    assert!(
        !exported.contains("## Provenance"),
        "exported skill must not embed a `## Provenance` section; provenance is reconstructed on re-add"
    );

    // Simulate edit: prepend a human comment.
    let edited = format!("# edited by owner\n{exported}");

    let re_added = parse_human_authored_skill(&edited);
    assert!(re_added.accepted, "re-added edited skill must be accepted");
    let skill = re_added.skill.as_ref().expect("re-added skill present");
    // Load-bearing: provenance must be HumanAuthored, NOT LlmAuthored.
    assert_eq!(
        skill.provenance,
        SkillProvenance::HumanAuthored,
        "exported-then-edited-then-re-added skill must carry HumanAuthored provenance"
    );
}

// A receipt lists a skill as applied only when its retype effect touches a harvested entity.
// Firing without an effect may remain in the trace but is excluded from applied_skill_ids.
#[test]
fn matched_but_effectless_skill_is_not_reported_applied() {
    use context_graph::harness::{CaptureHarnessConfig, run_capture};
    use context_graph::ingest::CaptureRequest;

    let (_dir, store) = fresh_store();
    insert_skill_row(
        &store,
        "sk-effectless",
        "retype a context-graph file entity to a repository",
        "when an entity named context-graph is typed File → Repository",
        "human-correction",
        true,
        false,
    );

    // The transcript mentions context-graph (so the skill matches and is selected) but declares
    // NO entity — there is nothing for the retype to change.
    let transcript = "Goal G1: track the context-graph rollout\nDecision D1: document the context-graph release notes serves G1";
    let (_report, harness) = run_capture(
        &store,
        CaptureRequest {
            context_label: "effectless-ctx".to_string(),
            transcript: transcript.to_string(),
            model_available: true,
        },
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("capture with a matched-but-effectless skill must still succeed");

    assert!(
        harness.applied_skill_ids.is_empty(),
        "a skill that produced no persisted effect must not be reported as applied, got {:?}",
        harness.applied_skill_ids
    );
}

// Test 8 (store-state): a promoted skill must auto-apply on a REAL matching capture, changing the
// captured result, with no approval step. The proof is a DIFF against a no-skill baseline capture
// of the same transcript: the two results must differ exactly by the skill's effect.
//
#[test]
fn promoted_skill_auto_applies_when_description_matches_no_approval_click() {
    use context_graph::harness::{CaptureHarnessConfig, run_capture};
    use context_graph::ingest::CaptureRequest;

    // The transcript names an entity whose type the skill is meant to correct (File → Repository).
    let transcript = "Goal G1: remember the project\nEntity E1: context-graph type=file\nDecision D1: keep one repo serves G1";
    let req = |label: &str| CaptureRequest {
        context_label: label.to_string(),
        transcript: transcript.to_string(),
        model_available: true,
    };

    // Baseline: a fresh store with NO learned skill. Capture and read the entity type.
    let (_bdir, baseline_store) = fresh_store();
    let (_br, baseline_hr) = run_capture(
        &baseline_store,
        req("baseline-ctx"),
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("baseline capture");
    let baseline_type = entity_type_named(&baseline_store, "context-graph");

    // With a promoted, active, matching skill in the store. Its effect: retype the File entity to
    // Repository when its description matches the chunk.
    let (_sdir, skill_store) = fresh_store();
    insert_skill_row(
        &skill_store,
        "sk-auto",
        "retype a context-graph file entity to a repository",
        "when an entity named context-graph is typed File → Repository",
        "human-correction",
        true,
        false,
    );
    let (_sr, skill_hr) = run_capture(
        &skill_store,
        req("skill-ctx"),
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("with-skill capture");
    let with_skill_type = entity_type_named(&skill_store, "context-graph");

    // The promoted skill must be reported as applied (no approval click required).
    assert!(
        skill_hr.applied_skill_ids.iter().any(|id| id == "sk-auto"),
        "the promoted matching skill must auto-apply (appear in applied_skill_ids) with no approval \
         step; got {:?}",
        skill_hr.applied_skill_ids
    );
    // The captured RESULT must differ from the no-skill baseline by exactly the skill's effect.
    assert_ne!(
        with_skill_type, baseline_type,
        "a promoted matching skill must change the captured result vs a no-skill baseline \
         (baseline={baseline_type:?}, with_skill={with_skill_type:?})"
    );
    // The no-skill baseline must NOT have applied any skill.
    assert!(
        baseline_hr.applied_skill_ids.is_empty(),
        "the no-skill baseline must apply nothing"
    );
}

/// Read the stored entity type for the entity whose name contains `name_fragment`, in any context.
fn entity_type_named(store: &Store, name_fragment: &str) -> Option<context_graph::EntityType> {
    store
        .list_entities(context_graph::ListEntityFilter::default())
        .expect("list entities")
        .into_iter()
        .find(|e| e.name.contains(name_fragment))
        .map(|e| e.entity_type)
}

#[test]
fn learned_skill_is_visible_via_skills_listing() {
    let (_dir, store) = fresh_store();

    for i in 0..3usize {
        insert_skill_row(
            &store,
            &format!("sk-list-{i}"),
            &format!("skill {i}"),
            &format!("rule {i}"),
            "human-correction",
            true,
            false,
        );
    }

    let skills = list_skills_in_store(&store).expect("read succeeds");
    assert_eq!(
        skills.len(),
        3,
        "listing must surface all 3 skills; got {}",
        skills.len()
    );
    for skill in &skills {
        assert_eq!(
            skill.provenance,
            SkillProvenance::HumanCorrection,
            "provenance must be HumanCorrection"
        );
    }
}

// Test 10 (store-state): a skill that WAS applying must STOP applying after `cg skills off`, while
// capture still succeeds — established by a real capture diff, not merely toggling a flag. With
// skills ON the skill's effect must be present; with skills OFF the same matching capture must NOT
// show the effect; ON again restores it.
//
#[test]
fn skills_off_master_switch_stops_application_capture_still_works() {
    use context_graph::harness::{CaptureHarnessConfig, run_capture};
    use context_graph::ingest::CaptureRequest;

    let transcript = "Goal G1: remember the project\nEntity E1: context-graph type=file\nDecision D1: keep one repo serves G1";
    let req = |label: &str| CaptureRequest {
        context_label: label.to_string(),
        transcript: transcript.to_string(),
        model_available: true,
    };

    // A store holding a promoted, active, matching skill whose effect retypes the entity.
    let (_dir, store) = fresh_store();
    insert_skill_row(
        &store,
        "sk-switch",
        "retype a context-graph file entity to a repository",
        "when an entity named context-graph is typed File → Repository",
        "human-correction",
        true,
        false,
    );

    // The no-skill baseline type (capture in a fresh, skill-free store).
    let (_bdir, baseline_store) = fresh_store();
    run_capture(
        &baseline_store,
        req("switch-baseline"),
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("baseline capture");
    let baseline_type = entity_type_named(&baseline_store, "context-graph");

    // Skills ON: the effect must be present (differs from the no-skill baseline).
    let _ = set_skills_enabled(&store, true);
    let (_on_r, on_hr) = run_capture(
        &store,
        req("switch-on-ctx"),
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig {
            skills_enabled: true,
            ..Default::default()
        },
    )
    .expect("skills-on capture");
    let on_type = entity_type_named(&store, "context-graph");
    assert_ne!(
        on_type, baseline_type,
        "with skills ON, the matching skill must apply and change the captured result vs baseline \
         (on={on_type:?}, baseline={baseline_type:?})"
    );
    assert!(
        on_hr.applied_skill_ids.iter().any(|id| id == "sk-switch"),
        "with skills ON, the matching skill must appear in applied_skill_ids"
    );

    // Skills OFF: the SAME matching capture must NOT show the effect, and capture still succeeds.
    let _ = set_skills_enabled(&store, false);
    let (off_report, off_hr) = run_capture(
        &store,
        req("switch-off-ctx"),
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig {
            skills_enabled: false,
            ..Default::default()
        },
    )
    .expect("skills-off capture must still succeed");
    let off_type = entity_type_named_in(&store, "context-graph", "switch-off-ctx");
    assert_eq!(
        off_type, baseline_type,
        "with skills OFF, a skill that WAS applying must STOP — the captured result must match the \
         no-skill baseline (off={off_type:?}, baseline={baseline_type:?})"
    );
    assert!(
        off_hr.applied_skill_ids.is_empty(),
        "with skills OFF, applied_skill_ids must be empty"
    );
    assert!(
        off_report.wrote_digest || !off_report.stable_ids.is_empty(),
        "capture must still produce a full receipt with skills OFF"
    );

    // ON again: applies again.
    let _ = set_skills_enabled(&store, true);
    let (_on2_r, on2_hr) = run_capture(
        &store,
        req("switch-on-again-ctx"),
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig {
            skills_enabled: true,
            ..Default::default()
        },
    )
    .expect("skills-on-again capture");
    assert!(
        on2_hr.applied_skill_ids.iter().any(|id| id == "sk-switch"),
        "turning skills back ON must re-apply the skill"
    );
}

/// Read the stored entity type for an entity by name within a specific context label.
fn entity_type_named_in(
    store: &Store,
    name_fragment: &str,
    context_label: &str,
) -> Option<context_graph::EntityType> {
    let context_id = store
        .list_contexts()
        .ok()?
        .into_iter()
        .find(|c| c.name == context_label)
        .map(|c| c.id);
    store
        .list_entities(context_graph::ListEntityFilter {
            context_id,
            ..Default::default()
        })
        .expect("list entities")
        .into_iter()
        .find(|e| e.name.contains(name_fragment))
        .map(|e| e.entity_type)
}

// LIVE PATH: the review-hold gate must gate the actual capture path. With `cg skills review` OFF
// (default), a freshly promoted matching skill auto-applies on a REAL capture; with it ON, a newly
// promoted skill is HELD (does NOT apply on the live capture) until acknowledged, while the
// promotion is still recorded. an incorrect implementation's run_capture applies no skills at all → the baseline and
// post-acknowledge "effect present" arms fail (honest regression).
#[test]
fn cg_skills_review_on_holds_a_freshly_promoted_skill_until_acknowledged() {
    use context_graph::harness::{CaptureHarnessConfig, run_capture};
    use context_graph::ingest::CaptureRequest;

    // A transcript whose entity the matching skill retypes (File → Repository) when applied.
    let transcript = "Goal G1: remember the project\nEntity E1: context-graph type=file\nDecision D1: keep one repo serves G1";
    let req = |label: &str| CaptureRequest {
        context_label: label.to_string(),
        transcript: transcript.to_string(),
        model_available: true,
    };

    // The no-skill baseline entity type (a fresh, skill-free store).
    let (_bdir, baseline_store) = fresh_store();
    run_capture(
        &baseline_store,
        req("review-baseline"),
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("baseline capture");
    let baseline_type = entity_type_named(&baseline_store, "context-graph");

    // --- review OFF (default): a promoted matching skill auto-applies on a REAL capture. ---
    let (_dir_off, store_off) = fresh_store();
    set_review_hold(&store_off, false);
    insert_skill_row(
        &store_off,
        "sk-review-off",
        "retype a context-graph file entity to a repository",
        "when an entity named context-graph is typed File → Repository",
        "human-correction",
        true,  // active
        false, // not held
    );
    let (_ro, off_hr) = run_capture(
        &store_off,
        req("review-off-ctx"),
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig {
            review_mode: false,
            ..Default::default()
        },
    )
    .expect("review-off capture");
    let off_type = entity_type_named_in(&store_off, "context-graph", "review-off-ctx");
    assert_ne!(
        off_type, baseline_type,
        "with `cg skills review` OFF, a promoted matching skill must auto-apply on the live capture \
         (effect differs from the no-skill baseline); an implementation that applies no skills (off={off_type:?}, \
         baseline={baseline_type:?})"
    );
    assert!(
        off_hr
            .applied_skill_ids
            .iter()
            .any(|id| id == "sk-review-off"),
        "review-off: the matching skill must appear in applied_skill_ids"
    );

    // --- review ON: a newly promoted skill is HELD on the live capture path until acknowledged. ---
    let (_dir_on, store_on) = fresh_store();
    set_review_hold(&store_on, true);
    // Freshly promoted under review-on → recorded held_for_review, must NOT apply on the live path.
    insert_skill_row(
        &store_on,
        "sk-review-held",
        "retype a context-graph file entity to a repository",
        "when an entity named context-graph is typed File → Repository",
        "human-correction",
        false, // not active yet (held)
        true,  // held_for_review
    );
    // Promotion is still recorded (the skill is listed even while held).
    assert_eq!(
        list_skills_in_store(&store_on)
            .expect("read succeeds")
            .len(),
        1,
        "review-on: the freshly promoted skill must still be recorded/listed even while held"
    );
    let (_rh, held_hr) = run_capture(
        &store_on,
        req("review-held-ctx"),
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig {
            review_mode: true,
            ..Default::default()
        },
    )
    .expect("review-held capture");
    let held_type = entity_type_named_in(&store_on, "context-graph", "review-held-ctx");
    assert_eq!(
        held_type, baseline_type,
        "with `cg skills review` ON, a held skill must NOT apply on the live capture (effect equals \
         the no-skill baseline); held={held_type:?}, baseline={baseline_type:?}"
    );
    assert!(
        held_hr.applied_skill_ids.is_empty(),
        "review-on held: applied_skill_ids must be empty (the held skill did not apply)"
    );

    // --- acknowledge → the skill now applies on a subsequent live capture. ---
    acknowledge_skill(&store_on, "sk-review-held");
    let (_ra, ack_hr) = run_capture(
        &store_on,
        req("review-ack-ctx"),
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig {
            review_mode: true,
            ..Default::default()
        },
    )
    .expect("post-acknowledge capture");
    let ack_type = entity_type_named_in(&store_on, "context-graph", "review-ack-ctx");
    assert_ne!(
        ack_type, baseline_type,
        "after acknowledgement, the skill must apply on a subsequent live capture (effect differs \
         from baseline); an implementation that applies no skills (ack={ack_type:?}, baseline={baseline_type:?})"
    );
    assert!(
        ack_hr
            .applied_skill_ids
            .iter()
            .any(|id| id == "sk-review-held"),
        "post-acknowledge: the skill must appear in applied_skill_ids"
    );
}
