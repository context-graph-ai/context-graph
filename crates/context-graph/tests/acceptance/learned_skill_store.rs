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
    try_promote_skill, try_promote_skill_with_gate_counter, try_promote_skill_with_result,
};
use context_graph::{Store, StoreConfig, TxId};
use contextdb_core::Value as DbValue;

use super::fake_extractor::open_store_with_lexical_embedder;
use super::skill_fixtures::{
    gate_failing_labels, gate_passing_labels, general_candidate_labels, llm_authored_skill,
    overfit_candidate_labels,
};

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
    // The promoted skill's IDENTITY must DERIVE from the aligned labels' content — value equality
    // against the labels, not merely non-empty. The gate_passing_labels all correct to
    // "technology_choice", so the derived rule/description must reference that target.
    assert!(
        skill.rule.contains("technology_choice"),
        "promoted skill rule must derive from the aligned labels' content (\"technology_choice\"); \
         a placeholder rule that does not reference the labels fails; got: {:?}",
        skill.rule
    );
    assert!(
        skill.description.contains("technology_choice"),
        "promoted skill description must derive from the aligned labels' content; got: {:?}",
        skill.description
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

// The held-out gate (Surface F) must catch OVERFITTING: a candidate that fits its training labels
// but FAILS a disjoint held-out slice must NOT be promoted, while a GENERAL rule that holds on the
// held-out slice must — the gate discriminates generalization, not training fit. A gate-invocation
// counter proves the verdict came from the held-out slice.
//
// SCAFFOLD STUB the test fails on: try_promote_skill_with_gate_counter judges on TRAINING fit and
// never runs the held-out gate (counter stays 0), so the overfit candidate IS promoted and the
// "gate was invoked" assertion fails.
#[test]
fn overfit_rule_is_not_promoted_while_a_general_rule_is() {
    let (general_training, general_heldout) = general_candidate_labels("sk-general");
    let (overfit_training, overfit_heldout) = overfit_candidate_labels("sk-overfit");

    // The general candidate must be promoted (its rule holds on the held-out slice).
    let general = try_promote_skill_with_gate_counter(&general_training, &general_heldout);
    assert!(
        general.promoted,
        "a GENERAL rule that holds on the held-out slice must be promoted; wrong stub may promote \
         it but the overfit/gate assertions below still fail"
    );

    // The overfit candidate must NOT be promoted (its rule fails the disjoint held-out slice).
    let overfit = try_promote_skill_with_gate_counter(&overfit_training, &overfit_heldout);
    assert!(
        !overfit.promoted,
        "an OVERFIT rule (fits training, fails the held-out slice) must NOT be promoted; \
         wrong stub judges on training fit and promotes it"
    );

    // The held-out gate must have been EVALUATED on the learned path — proving the verdict came from
    // the held-out slice, not training fit. The count rides the PromotionResult as a plain usize
    // (no app-side global/atomic in src/); the wrong stub never runs the gate → it stays 0.
    assert!(
        general.heldout_gate_invocations > 0,
        "the held-out gate must be evaluated on the general candidate (heldout_gate_invocations > 0), \
         proving generalization was measured on the held-out slice; wrong stub never runs it (0)"
    );
    assert!(
        overfit.heldout_gate_invocations > 0,
        "the held-out gate must be evaluated on the overfit candidate too (heldout_gate_invocations \
         > 0); wrong stub never runs it (0)"
    );

    // The promoted general rule's CONTENT must derive from the general labels (value equality).
    let general_skill = general
        .skill
        .expect("general candidate must yield a skill when promoted");
    assert!(
        general_skill.rule.contains("technology_choice"),
        "the promoted general rule must derive from the general labels' content; got: {:?}",
        general_skill.rule
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

// Test 8 (store-state): a promoted skill must auto-apply on a REAL matching capture, changing the
// captured result, with no approval step. The proof is a DIFF against a no-skill baseline capture
// of the same transcript: the two results must differ exactly by the skill's effect.
//
// SCAFFOLD STUB the test fails on: run_capture does not read skills at all (applied_skill_ids is
// hardcoded empty and no skill ever changes a captured value), so the with-skill and no-skill
// captures are identical and applied_skill_ids stays empty → both assertions fail (RED).
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
        &super::refactor_extractor::SpanFaithfulExtractor,
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
        &super::refactor_extractor::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("with-skill capture");
    let with_skill_type = entity_type_named(&skill_store, "context-graph");

    // The promoted skill must be reported as applied (no approval click required).
    assert!(
        skill_hr.applied_skill_ids.iter().any(|id| id == "sk-auto"),
        "the promoted matching skill must auto-apply (appear in applied_skill_ids) with no approval \
         step; wrong stub leaves applied_skill_ids empty, got {:?}",
        skill_hr.applied_skill_ids
    );
    // The captured RESULT must differ from the no-skill baseline by exactly the skill's effect.
    assert_ne!(
        with_skill_type, baseline_type,
        "a promoted matching skill must change the captured result vs a no-skill baseline; \
         wrong stub ignores skills on the capture path so both captures are identical \
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

// Test 10 (store-state): a skill that WAS applying must STOP applying after `cg skills off`, while
// capture still succeeds — established by a real capture diff, not merely toggling a flag. With
// skills ON the skill's effect must be present; with skills OFF the same matching capture must NOT
// show the effect; ON again restores it.
//
// SCAFFOLD STUB the test fails on: run_capture never applies skills (no skill-application seam), so
// the effect is absent even with skills ON → the "effect present with skills ON" assertion fails.
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
        &super::refactor_extractor::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("baseline capture");
    let baseline_type = entity_type_named(&baseline_store, "context-graph");

    // Skills ON: the effect must be present (differs from the no-skill baseline).
    let _ = set_skills_enabled(&store, true);
    let (_on_r, on_hr) = run_capture(
        &store,
        req("switch-on-ctx"),
        &super::refactor_extractor::SpanFaithfulExtractor,
        &CaptureHarnessConfig {
            skills_enabled: true,
            ..Default::default()
        },
    )
    .expect("skills-on capture");
    let on_type = entity_type_named(&store, "context-graph");
    assert_ne!(
        on_type, baseline_type,
        "with skills ON, the matching skill must apply and change the captured result vs baseline; \
         wrong stub never applies skills (on={on_type:?}, baseline={baseline_type:?})"
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
        &super::refactor_extractor::SpanFaithfulExtractor,
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
        &super::refactor_extractor::SpanFaithfulExtractor,
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
// promotion is still recorded. HEAD's run_capture applies no skills at all → the baseline and
// post-acknowledge "effect present" arms fail (honest RED).
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
        &super::refactor_extractor::SpanFaithfulExtractor,
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
        &super::refactor_extractor::SpanFaithfulExtractor,
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
         (effect differs from the no-skill baseline); HEAD applies no skills (off={off_type:?}, \
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
        list_skills_in_store(&store_on).len(),
        1,
        "review-on: the freshly promoted skill must still be recorded/listed even while held"
    );
    let (_rh, held_hr) = run_capture(
        &store_on,
        req("review-held-ctx"),
        &super::refactor_extractor::SpanFaithfulExtractor,
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
        &super::refactor_extractor::SpanFaithfulExtractor,
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
         from baseline); HEAD applies no skills (ack={ack_type:?}, baseline={baseline_type:?})"
    );
    assert!(
        ack_hr
            .applied_skill_ids
            .iter()
            .any(|id| id == "sk-review-held"),
        "post-acknowledge: the skill must appear in applied_skill_ids"
    );
}
