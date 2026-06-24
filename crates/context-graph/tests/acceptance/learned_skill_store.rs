// Learned-skill store and promotion tests.
//
// Store-state tests: fail under a HashMap because assertions read persisted rows
// from the real cg_learned_skills / cg_teaching_events tables. The placeholder
// implementations in skill_store.rs return empty lists, None, or wrong provenance.
//
// Pure-logic tests: assert the function contract (Some/None + score) — no store.

use std::collections::HashMap;

use context_graph::skill_store::{
    SkillProvenance, TeachingEventKind, acknowledge_skill, apply_correction, export_skill,
    list_skills_in_store, parse_human_authored_skill, set_review_hold, set_skills_enabled,
    try_promote_skill, try_promote_skill_with_result,
};
use context_graph::{Store, StoreConfig, TxId};
use contextdb_core::Value as DbValue;

use super::fake_extractor::open_store_with_lexical_embedder;
use super::skill_fixtures::{gate_failing_labels, gate_passing_labels, llm_authored_skill};

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

fn scan_skill_table(store: &Store) -> Vec<HashMap<String, DbValue>> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    db.scan("cg_learned_skills", snapshot)
        .expect("scan cg_learned_skills")
        .into_iter()
        .map(|row| row.values)
        .collect()
}

// Test 1 (store-state): reads persisted rows; the placeholder returns an empty list.
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

    // The placeholder returns empty — this assertion fails.
    let skills = list_skills_in_store(&store);
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

// Test 2 (store-state): the placeholder apply_correction returns teaching_event = None.
#[test]
fn a_correction_is_a_label_not_a_skill() {
    let (_dir, store) = fresh_store();

    // Apply a single correction (no skill linked yet).
    let result = apply_correction(&store, "item-abc", "technology_choice", None);

    // The correction should report as applied.
    assert!(result.applied, "correction must report as applied");

    // A teaching event must have been logged — the placeholder returns None.
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

// Test 3 (pure-logic): the placeholder try_promote_skill always returns None.
#[test]
fn multiple_aligned_labels_promote_a_skill_single_correction_does_not() {
    let skill_id = "sk-promote-test";
    let (training, heldout) = gate_passing_labels(skill_id);

    // Single label must NOT promote.
    let single = try_promote_skill(&training[..1], &heldout);
    assert!(
        single.is_none(),
        "a single correction label must NOT promote a skill"
    );

    // Multiple aligned labels + heldout should promote.
    let promoted = try_promote_skill(&training, &heldout);
    // The placeholder returns None here.
    assert!(
        promoted.is_some(),
        "multiple aligned labels that beat heldout must produce a promoted skill"
    );

    let skill = promoted.unwrap();
    // The promoted skill's description and rule must derive from the labels (not a placeholder).
    assert!(
        !skill.description.is_empty(),
        "promoted skill description must not be empty"
    );
    assert!(
        !skill.rule.is_empty(),
        "promoted skill rule must not be empty"
    );
    // Provenance must reflect correction-origin.
    assert!(
        matches!(
            skill.provenance,
            SkillProvenance::HumanCorrection | SkillProvenance::LlmAuthored
        ),
        "promoted skill provenance must be HumanCorrection or LlmAuthored, got {:?}",
        skill.provenance
    );
}

// Test 4 (pure-logic): the placeholder try_promote_skill_with_result always returns promoted=false.
// A tie must not promote; this test distinguishes tie from outright failure.
// The first assertion (gate-passing labels with no incumbent MUST promote) fails on the placeholder.
#[test]
fn candidate_that_ties_prior_best_on_heldout_is_not_promoted() {
    let skill_id = "sk-tie-test";
    let (training, heldout) = gate_passing_labels(skill_id);

    // Gate-passing labels with no incumbent must produce a promoted result.
    // The placeholder returns promoted=false.
    let first = try_promote_skill_with_result(&training, &heldout, None);
    assert!(
        first.promoted,
        "gate-passing labels with no incumbent must promote a skill; placeholder returns promoted=false"
    );
    assert!(
        first.heldout_score > 0.0,
        "the established incumbent score must be positive; placeholder returns 0.0"
    );
    let incumbent_score = first.heldout_score;

    // Run with the SAME labels against the incumbent — should be a tie → no promotion.
    let tie = try_promote_skill_with_result(&training, &heldout, Some(incumbent_score));
    assert!(
        !tie.promoted,
        "a candidate that ties the incumbent score must NOT be promoted"
    );
    assert_eq!(
        tie.heldout_score, incumbent_score,
        "tied candidate score must equal incumbent score"
    );

    // Gate-failing labels also must not promote.
    let (failing_training, failing_heldout) = gate_failing_labels("sk-fail");
    let failing = try_promote_skill_with_result(&failing_training, &failing_heldout, None);
    assert!(
        !failing.promoted,
        "a candidate that fails the heldout check must NOT be promoted"
    );
}

// Test 5 (pure-logic): the placeholder parse_human_authored_skill accepts any non-empty content.
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
    // The placeholder accepts any non-empty content.
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

// Test 6 (pure-logic): the placeholder parse_human_authored_skill uses the raw first line
// (including the `#` prefix) as the description. The real implementation strips the
// `# description:` header to extract a clean description.
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

    // The SKILL.md parser must extract the clean description (without the `# description:` header).
    // The placeholder uses `content.lines().next()` → description starts with `#`.
    // This assertion fails because the placeholder does not strip the header.
    assert!(
        !skill.description.starts_with('#'),
        "description must be extracted cleanly from the `# description:` header, not include the `#`; got: {:?}",
        skill.description
    );
    assert!(
        !skill.description.is_empty(),
        "description must not be empty after extraction"
    );

    // Verify the gate WOULD fail these labels, confirming the gate-bypass is real.
    let (failing_training, failing_heldout) = gate_failing_labels("sk-would-fail");
    let gate_result = try_promote_skill(&failing_training, &failing_heldout);
    // The human-authored path was accepted even though gate would return None.
    assert!(
        gate_result.is_none(),
        "the gate-failing labels must not promote a skill (gate-bypass proof)"
    );
}

// Test 7 (pure-logic): the placeholder export_skill returns "# Skill: {desc}\n## Rule\n..."
// which is not SKILL.md-shaped. The correct export uses "# description: {desc}" as the first
// header and omits the "## Provenance" section (provenance is reconstructed on re-add).
#[test]
fn exported_edited_skill_pushed_back_becomes_human_authored() {
    let original = llm_authored_skill("sk-export", "detect technology_choice decisions");

    let exported = export_skill(&original);
    assert!(
        !exported.is_empty(),
        "exported skill must produce non-empty text"
    );

    // The exported SKILL.md must start with the `# description:` header (not `# Skill:`).
    // The placeholder emits "# Skill: ...".
    assert!(
        exported.starts_with("# description:"),
        "exported skill must start with `# description:` header; placeholder uses `# Skill:`; got: {:?}",
        exported.lines().next().unwrap_or("")
    );

    // A real SKILL.md does not embed a `## Provenance` section.
    // The placeholder emits `## Provenance\n{prov:?}`.
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

// Test 8 (store-state): the placeholder list_skills_in_store returns empty.
#[test]
fn promoted_skill_auto_applies_when_description_matches_no_approval_click() {
    let (_dir, store) = fresh_store();

    // Insert a promoted (active, not held) skill.
    insert_skill_row(
        &store,
        "sk-auto",
        "detect technology_choice decisions",
        "when decision mentions library → technology_choice",
        "human-correction",
        true,
        false,
    );

    let skills = list_skills_in_store(&store);
    // The placeholder returns empty.
    assert_eq!(
        skills.len(),
        1,
        "promoted active skill must be visible in the listing without an approval click"
    );
    let skill = &skills[0];
    assert!(skill.active, "auto-applying skill must be active");
    assert!(
        !skill.held_for_review,
        "auto-applying skill must NOT be held for review"
    );
}

// Test 9 (store-state): the placeholder list_skills_in_store returns empty.
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

    let skills = list_skills_in_store(&store);
    // The placeholder returns empty.
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

// Test 10 (store-state): the placeholder set_skills_enabled does nothing.
#[test]
fn skills_off_master_switch_stops_application_capture_still_works() {
    let (_dir, store) = fresh_store();

    // Insert one active skill.
    insert_skill_row(
        &store,
        "sk-switch",
        "switch test skill",
        "any rule",
        "human-correction",
        true,
        false,
    );

    // Skills are on: listing must show active skill.
    let skills_on = list_skills_in_store(&store);
    // The placeholder returns empty.
    assert_eq!(
        skills_on.len(),
        1,
        "before switch-off, listing must surface the active skill"
    );

    // Turn off skills master switch.
    let _ = set_skills_enabled(&store, false);

    // After switch-off: apply_correction must still work (capture still works).
    let correction_result = apply_correction(&store, "item-x", "corrected", None);
    assert!(
        correction_result.applied,
        "capture/correction must still work after skills switch-off"
    );

    // Turn skills back on.
    let _ = set_skills_enabled(&store, true);
}

// Test 11 (store-state): the placeholder set_review_hold and acknowledge_skill do nothing.
#[test]
fn cg_skills_review_on_holds_a_freshly_promoted_skill_until_acknowledged() {
    let (_dir, store) = fresh_store();

    // Insert a skill held for review (active=false, held_for_review=true).
    insert_skill_row(
        &store,
        "sk-held",
        "held skill",
        "candidate rule",
        "human-correction",
        false,
        true,
    );

    // With review-hold on: the skill must be in the listing but held.
    set_review_hold(&store, true);
    let skills_before_ack = list_skills_in_store(&store);
    // The placeholder returns empty.
    assert_eq!(
        skills_before_ack.len(),
        1,
        "held skill must still be recorded in the listing even when held"
    );
    let held = &skills_before_ack[0];
    assert!(
        held.held_for_review,
        "skill must be held_for_review before acknowledgement"
    );
    assert!(
        !held.active,
        "held skill must not be active (auto-apply blocked) before acknowledgement"
    );

    // Acknowledge the skill — the placeholder does nothing.
    acknowledge_skill(&store, "sk-held");

    // After acknowledge: the skill must no longer be held.
    let skills_after_ack = list_skills_in_store(&store);
    // The placeholder still returns empty.
    assert_eq!(
        skills_after_ack.len(),
        1,
        "acknowledged skill must still appear in listing"
    );
    let acknowledged = &skills_after_ack[0];
    assert!(
        !acknowledged.held_for_review,
        "skill must not be held_for_review after acknowledgement"
    );
    assert!(
        acknowledged.active,
        "skill must become active after acknowledgement"
    );
}
