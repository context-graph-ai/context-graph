// Drift safeguard acceptance tests (4 tests).
//
// All four tests are store-state tests: they read persisted rows from
// cg_skill_firing_trace / cg_learned_skills and assert the right-vs-wrong
// distinction. Firing labels persist per trace, and drift detection exposes
// alerts for deteriorating skill records.

use std::collections::HashMap;

use context_graph::skill_store::{
    FiringTrace, LearnedSkill, SkillApplyStrategy, SkillProvenance, check_drift, list_drift_alerts,
    record_firing_label,
};
use context_graph::{Store, StoreConfig, TxId};
use contextdb_core::Value as DbValue;

use super::fake_extractor::open_store_with_lexical_embedder;

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("skill_h.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn insert_skill(store: &Store, id: &str, provenance: &str) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_learned_skills (id, description, rule, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("description".to_string(), DbValue::Text(format!("skill {id}"))),
            ("rule".to_string(), DbValue::Text(format!("rule {id}"))),
            ("provenance".to_string(), DbValue::Text(provenance.to_string())),
            ("active".to_string(), DbValue::Bool(true)),
            ("held_for_review".to_string(), DbValue::Bool(false)),
            ("fire_right".to_string(), DbValue::Int64(0)),
            ("fire_wrong".to_string(), DbValue::Int64(0)),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("insert skill for drift test");
}

fn insert_firing_trace(
    store: &Store,
    id: &str,
    skill_id: &str,
    capture_id: &str,
    correct: Option<bool>,
) {
    let db = store.sync_database();
    let correct_val = match correct {
        Some(true) => DbValue::Int64(1),
        Some(false) => DbValue::Int64(0),
        None => DbValue::Null,
    };
    db.execute(
        "INSERT INTO cg_skill_firing_trace (id, skill_id, capture_id, correct, created_at) VALUES ($id, $skill_id, $capture_id, $correct, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("skill_id".to_string(), DbValue::Text(skill_id.to_string())),
            ("capture_id".to_string(), DbValue::Text(capture_id.to_string())),
            ("correct".to_string(), correct_val),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("insert firing trace for drift test");
}

fn count_trace_with_correct(store: &Store, skill_id: &str, correct_val: Option<bool>) -> usize {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    db.scan("cg_skill_firing_trace", snapshot)
        .expect("scan firing trace")
        .into_iter()
        .filter(|row| {
            let matches_skill = matches!(
                row.values.get("skill_id"),
                Some(DbValue::Text(s)) if s == skill_id
            );
            let matches_correct = match correct_val {
                Some(true) => matches!(row.values.get("correct"), Some(DbValue::Int64(1))),
                Some(false) => matches!(row.values.get("correct"), Some(DbValue::Int64(0))),
                None => matches!(row.values.get("correct"), Some(DbValue::Null) | None),
            };
            matches_skill && matches_correct
        })
        .count()
}

fn get_skill_row(store: &Store, skill_id: &str) -> Option<HashMap<String, DbValue>> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    db.point_lookup(
        "cg_learned_skills",
        "id",
        &DbValue::Text(skill_id.to_string()),
        snapshot,
    )
    .expect("skill point lookup")
    .map(|row| row.values)
}

fn make_skill(id: &str, provenance: SkillProvenance) -> LearnedSkill {
    LearnedSkill {
        id: id.to_string(),
        description: format!("skill {id}"),
        rule: format!("rule {id}"),
        effect: None,
        apply_strategy: SkillApplyStrategy::PostHarvestRetype,
        provenance,
        active: true,
        held_for_review: false,
        fire_right: 0,
        fire_wrong: 0,
    }
}

fn load_traces_for_skill(store: &Store, skill_id: &str) -> Vec<FiringTrace> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    db.scan("cg_skill_firing_trace", snapshot)
        .expect("scan trace")
        .into_iter()
        .filter(|row| {
            matches!(
                row.values.get("skill_id"),
                Some(DbValue::Text(s)) if s == skill_id
            )
        })
        .map(|row| {
            let id = match row.values.get("id") {
                Some(DbValue::Text(t)) => t.clone(),
                _ => panic!("id must be Text"),
            };
            let skill_id = match row.values.get("skill_id") {
                Some(DbValue::Text(t)) => t.clone(),
                _ => panic!("skill_id must be Text"),
            };
            let capture_id = match row.values.get("capture_id") {
                Some(DbValue::Text(t)) => t.clone(),
                _ => panic!("capture_id must be Text"),
            };
            let correct = match row.values.get("correct") {
                Some(DbValue::Int64(1)) => Some(true),
                Some(DbValue::Int64(0)) => Some(false),
                _ => None,
            };
            FiringTrace {
                id,
                skill_id,
                capture_id,
                correct,
            }
        })
        .collect()
}

// Firing traces and their correctness labels persist per skill; accumulated
// wrong firings produce a drift alert.
#[test]
fn per_skill_right_vs_wrong_record_exists_not_a_bare_fire_count() {
    let (_dir, store) = fresh_store();
    insert_skill(&store, "sk-rw", "human-correction");
    let skill = make_skill("sk-rw", SkillProvenance::HumanCorrection);

    // Insert many CORRECT-only firing traces.
    for i in 0..6usize {
        insert_firing_trace(
            &store,
            &format!("ft-right-{i}"),
            "sk-rw",
            &format!("cap-r-{i}"),
            Some(true),
        );
    }

    let correct_traces = load_traces_for_skill(&store, "sk-rw");
    assert_eq!(correct_traces.len(), 6, "must read 6 persisted traces");

    // All-correct traces: drift verdict must NOT move toward bad.
    let alert_correct_only = check_drift(&skill, &correct_traces);
    assert!(
        alert_correct_only.is_none(),
        "correct-only firing record must NOT trigger a drift alert"
    );

    // Now test with wrong-only traces on a separate skill.
    let (_dir2, store2) = fresh_store();
    insert_skill(&store2, "sk-bad", "human-correction");
    let bad_skill = make_skill("sk-bad", SkillProvenance::HumanCorrection);

    for i in 0..5usize {
        insert_firing_trace(
            &store2,
            &format!("ft-wrong-{i}"),
            "sk-bad",
            &format!("cap-w-{i}"),
            Some(false),
        );
    }

    let wrong_traces = load_traces_for_skill(&store2, "sk-bad");

    // Assert the stored record distinguishes right vs wrong by counting.
    let right_count = wrong_traces
        .iter()
        .filter(|t| t.correct == Some(true))
        .count();
    let wrong_count = wrong_traces
        .iter()
        .filter(|t| t.correct == Some(false))
        .count();
    assert_eq!(right_count, 0, "no correct traces in the wrong-only set");
    assert_eq!(
        wrong_count, 5,
        "five wrong traces must be in the wrong-only set"
    );

    let alert_wrong = check_drift(&bad_skill, &wrong_traces);
    assert!(
        alert_wrong.is_some(),
        "a skill with many wrong fires must trigger a drift alert; check_drift returned None"
    );
}

#[test]
fn skill_fires_on_two_captures_with_opposite_correct_labels_attributes_one_right_one_wrong() {
    let (_dir, store) = fresh_store();
    insert_skill(&store, "sk-attr", "human-correction");

    let firing_p_id = "ft-correct-p";
    let firing_q_id = "ft-wrong-q";
    // Insert both traces with no label yet (correct=None / unlabeled).
    insert_firing_trace(&store, firing_p_id, "sk-attr", "cap-p", None);
    insert_firing_trace(&store, firing_q_id, "sk-attr", "cap-q", None);

    record_firing_label(&store, firing_p_id, true);
    record_firing_label(&store, firing_q_id, false);

    let right_count = count_trace_with_correct(&store, "sk-attr", Some(true));
    let wrong_count = count_trace_with_correct(&store, "sk-attr", Some(false));

    assert_eq!(
        right_count, 1,
        "exactly one trace must be attributed as correct (P) after record_firing_label; got {right_count}"
    );
    assert_eq!(
        wrong_count, 1,
        "exactly one trace must be attributed as wrong (Q) after record_firing_label; got {wrong_count}"
    );

    // Swap: re-label P as wrong and Q as correct.
    record_firing_label(&store, firing_p_id, false);
    record_firing_label(&store, firing_q_id, true);

    let right_after_swap = count_trace_with_correct(&store, "sk-attr", Some(true));
    let wrong_after_swap = count_trace_with_correct(&store, "sk-attr", Some(false));

    // Per-firing attribution: swap must flip the counts.
    assert_eq!(
        right_after_swap, 1,
        "after swap, Q (now correct) must be the only right trace; got {right_after_swap}"
    );
    assert_eq!(
        wrong_after_swap, 1,
        "after swap, P (now wrong) must be the only wrong trace; got {wrong_after_swap}"
    );

    // ORDER-INDEPENDENCE: the per-firing attribution must be by FIRING IDENTITY, not insertion
    // position. Run the SAME ground truth (P correct, Q wrong) but record the labels in the OTHER
    // order (Q-then-P) on a fresh store; the attribution must be identical, so a stub that infers
    // right/wrong from insertion position (e.g. "first labeled is right") fails this ordering.
    let (_dir2, store2) = fresh_store();
    insert_skill(&store2, "sk-attr", "human-correction");
    insert_firing_trace(&store2, firing_p_id, "sk-attr", "cap-p", None);
    insert_firing_trace(&store2, firing_q_id, "sk-attr", "cap-q", None);
    // Label Q FIRST (wrong), then P (correct) — opposite recording order, same ground truth.
    record_firing_label(&store2, firing_q_id, false);
    record_firing_label(&store2, firing_p_id, true);

    // P must still be the right firing and Q the wrong one — by identity, regardless of order.
    let p_correct = matches!(count_trace_correct_value(&store2, firing_p_id), Some(true));
    let q_wrong = matches!(count_trace_correct_value(&store2, firing_q_id), Some(false));
    assert!(
        p_correct,
        "P must be attributed RIGHT by firing identity regardless of recording order (Q-then-P)"
    );
    assert!(
        q_wrong,
        "Q must be attributed WRONG by firing identity regardless of recording order (Q-then-P)"
    );
    assert_eq!(
        count_trace_with_correct(&store2, "sk-attr", Some(true)),
        1,
        "exactly one right trace in the reversed-order run"
    );
    assert_eq!(
        count_trace_with_correct(&store2, "sk-attr", Some(false)),
        1,
        "exactly one wrong trace in the reversed-order run"
    );
}

/// Read the `correct` value of a specific firing trace row by its id.
fn count_trace_correct_value(store: &Store, firing_id: &str) -> Option<bool> {
    let db = store.sync_database();
    let snapshot = db.snapshot();
    db.point_lookup(
        "cg_skill_firing_trace",
        "id",
        &DbValue::Text(firing_id.to_string()),
        snapshot,
    )
    .expect("trace point lookup")
    .and_then(|row| match row.values.get("correct") {
        Some(DbValue::Int64(1)) => Some(true),
        Some(DbValue::Int64(0)) => Some(false),
        _ => None,
    })
}

// A deteriorating skill record produces one persisted alert for either provenance;
// repeated checks do not re-emit it or alert on a healthy co-resident skill.
#[test]
fn skill_whose_record_turns_bad_is_flagged_once_for_any_provenance() {
    for provenance_str in &["human-authored", "llm-authored"] {
        let (_dir, store) = fresh_store();
        let prov_enum = if *provenance_str == "human-authored" {
            SkillProvenance::HumanAuthored
        } else {
            SkillProvenance::LlmAuthored
        };

        insert_skill(&store, "sk-target", provenance_str);
        insert_skill(&store, "sk-healthy", "human-correction");

        // Drive the target skill's record bad: many wrong fires.
        for i in 0..6usize {
            insert_firing_trace(
                &store,
                &format!("ft-bad-{i}-{provenance_str}"),
                "sk-target",
                &format!("cap-bad-{i}"),
                Some(false),
            );
        }
        // Healthy skill: only correct fires.
        for i in 0..4usize {
            insert_firing_trace(
                &store,
                &format!("ft-good-{i}-{provenance_str}"),
                "sk-healthy",
                &format!("cap-good-{i}"),
                Some(true),
            );
        }

        let bad_traces = load_traces_for_skill(&store, "sk-target");
        let good_traces = load_traces_for_skill(&store, "sk-healthy");

        let bad_skill = LearnedSkill {
            id: "sk-target".to_string(),
            description: "target skill".to_string(),
            rule: "rule".to_string(),
            effect: None,
            apply_strategy: SkillApplyStrategy::PostHarvestRetype,
            provenance: prov_enum.clone(),
            active: true,
            held_for_review: false,
            fire_right: 0,
            fire_wrong: 0,
        };
        let healthy_skill = make_skill("sk-healthy", SkillProvenance::HumanCorrection);

        // check_drift for the target must return Some(DriftAlert).
        let target_alert = check_drift(&bad_skill, &bad_traces);
        assert!(
            target_alert.is_some(),
            "bad-record skill must trigger a drift alert (provenance={provenance_str}); check_drift returned None"
        );

        let alert = target_alert.unwrap();
        // Alert must name the correct provenance.
        assert_eq!(
            alert.provenance, prov_enum,
            "drift alert provenance must match the bad skill's provenance"
        );
        // Alert must identify the specific bad skill.
        assert_eq!(
            alert.skill_id, "sk-target",
            "drift alert must identify the specific bad skill"
        );

        // Healthy co-resident skill must NOT be alerted.
        let healthy_alert = check_drift(&healthy_skill, &good_traces);
        assert!(
            healthy_alert.is_none(),
            "healthy co-resident skill must NOT be alerted (provenance={provenance_str})"
        );

        // Repeat check_drift N times — alert must remain on the same skill (deduped).
        for _ in 0..4 {
            let repeat_alert = check_drift(&bad_skill, &bad_traces);
            assert!(
                repeat_alert.is_some(),
                "repeated check_drift must still return Some for a bad skill"
            );
            assert_eq!(
                repeat_alert.unwrap().skill_id,
                "sk-target",
                "repeated drift alert must still identify sk-target"
            );
        }

        let all_alerts = list_drift_alerts(&store).expect("read succeeds");
        let target_alerts: Vec<_> = all_alerts
            .iter()
            .filter(|a| a.skill_id == "sk-target")
            .collect();
        assert_eq!(
            target_alerts.len(),
            1,
            "must have exactly one drift alert for sk-target (deduped); got {} (provenance={provenance_str})",
            target_alerts.len()
        );
        let healthy_alerts: Vec<_> = all_alerts
            .iter()
            .filter(|a| a.skill_id == "sk-healthy")
            .collect();
        assert!(
            healthy_alerts.is_empty(),
            "healthy skill must NOT appear in drift alerts (provenance={provenance_str})"
        );
    }
}

// A drift alert informs the caller while the skill remains active.
#[test]
fn drift_alert_never_auto_disables_the_skill() {
    let (_dir, store) = fresh_store();
    insert_skill(&store, "sk-no-autodis", "human-correction");

    // Drive the skill's record bad.
    for i in 0..6usize {
        insert_firing_trace(
            &store,
            &format!("ft-dis-{i}"),
            "sk-no-autodis",
            &format!("cap-dis-{i}"),
            Some(false),
        );
    }

    let bad_skill = make_skill("sk-no-autodis", SkillProvenance::HumanCorrection);
    let traces = load_traces_for_skill(&store, "sk-no-autodis");

    // Trigger the drift alert.
    let alert = check_drift(&bad_skill, &traces);
    assert!(
        alert.is_some(),
        "bad-record skill must produce a drift alert before testing no-auto-disable"
    );
    assert!(
        alert.unwrap().active,
        "drift alert must initially be active (it is an informational flag)"
    );

    // After the alert, the skill row must still be active (not auto-disabled).
    let skill_row = get_skill_row(&store, "sk-no-autodis");
    assert!(
        skill_row.is_some(),
        "skill row must still exist after drift alert"
    );
    let is_active = matches!(skill_row.unwrap().get("active"), Some(DbValue::Bool(true)));
    assert!(
        is_active,
        "drift alert must NOT auto-disable the skill; the skill must remain active"
    );

    // Only an explicit undo (setting active=false) would disable it.
    {
        let db = store.sync_database();
        db.execute(
            "UPDATE cg_learned_skills SET active = $active WHERE id = $id",
            &HashMap::from([
                ("id".to_string(), DbValue::Text("sk-no-autodis".to_string())),
                ("active".to_string(), DbValue::Bool(false)),
            ]),
        )
        .expect("explicit undo disables skill");
    }
    let skill_after_undo = get_skill_row(&store, "sk-no-autodis");
    let is_active_after = matches!(
        skill_after_undo.unwrap().get("active"),
        Some(DbValue::Bool(true))
    );
    assert!(
        !is_active_after,
        "only an explicit undo must disable the skill"
    );
}

/// Insert an active, matching learned skill with a controllable description (test seeding).
/// Insert a skill whose description matches the drift tests' capture chunk AND whose rule carries
/// a real, derivable retype effect (the fold-seam withhold contract excludes an effect-less row
/// from applying/firing regardless of matching strategy — a placeholder non-retype rule string
/// is withheld, not fired, defeating these firing/drift tests' whole point).
fn insert_matching_skill(store: &Store, id: &str, description: &str) {
    let db = store.sync_database();
    db.execute(
        "INSERT INTO cg_learned_skills (id, description, rule, provenance, active, held_for_review, fire_right, fire_wrong, created_at) VALUES ($id, $description, $rule, $provenance, $active, $held_for_review, $fire_right, $fire_wrong, $created_at)",
        &HashMap::from([
            ("id".to_string(), DbValue::Text(id.to_string())),
            ("description".to_string(), DbValue::Text(description.to_string())),
            (
                "rule".to_string(),
                DbValue::Text(
                    "when an entity named context-graph is typed File → Repository".to_string(),
                ),
            ),
            ("provenance".to_string(), DbValue::Text("human-correction".to_string())),
            ("active".to_string(), DbValue::Bool(true)),
            ("held_for_review".to_string(), DbValue::Bool(false)),
            ("fire_right".to_string(), DbValue::Int64(0)),
            ("fire_wrong".to_string(), DbValue::Int64(0)),
            ("created_at".to_string(), DbValue::TxId(TxId(1))),
        ]),
    )
    .expect("insert matching skill");
}

// Capture-recorded drift input: drift's real input is capture-recorded firings. When a promoted skill applies
// during a REAL capture, a `cg_skill_firing_trace` row must be recorded for that skill+capture (on
// the capture path, not hand-inserted); a later wrong-label correction then makes `list_drift_alerts`
// — reading the CAPTURE-produced traces — surface it. an incorrect implementation's run_capture applies no skills and
// records no firing → both arms fail (honest regression).
#[test]
fn skill_application_records_a_firing_trace_on_the_capture_path() {
    use context_graph::harness::{CaptureHarnessConfig, run_capture};
    use context_graph::ingest::CaptureRequest;
    use context_graph::skill_store::record_firing_label;

    let (_dir, store) = fresh_store();

    // Promote a matching skill (its description matches the captured chunk).
    insert_matching_skill(
        &store,
        "sk-fires",
        "retype a context-graph file entity to a repository",
    );

    // Run a REAL capture in which the skill applies.
    let (_report, harness_report) = run_capture(
        &store,
        CaptureRequest {
            context_label: "drift-firing-ctx".to_string(),
            transcript:
                "Goal G1: remember the project\nEntity E1: context-graph type=file\nDecision D1: keep one repo serves G1"
                    .to_string(),
            model_available: true,
        },
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("capture where the skill applies");

    // The skill must be reported applied AND a firing trace recorded ON THE CAPTURE PATH for it.
    assert!(
        harness_report
            .applied_skill_ids
            .iter()
            .any(|id| id == "sk-fires"),
        "the matching skill must apply on the live capture; an implementation that applies no skills (applied={:?})",
        harness_report.applied_skill_ids
    );
    let traces = load_traces_for_skill(&store, "sk-fires");
    assert!(
        !traces.is_empty(),
        "a cg_skill_firing_trace row must be recorded on the capture path for the applied skill; \
         an incorrect implementation's run_capture records none (it sets applied_skill_ids but never records a firing)"
    );

    // A wrong-label correction against that capture-produced firing must let drift surface it.
    let firing = &traces[0];
    record_firing_label(&store, &firing.id, false);
    record_firing_label(&store, &firing.id, false);
    record_firing_label(&store, &firing.id, false);
    let alerts = list_drift_alerts(&store).expect("read succeeds");
    assert!(
        alerts.iter().any(|a| a.skill_id == "sk-fires"),
        "after wrong-label corrections on the CAPTURE-produced firing, list_drift_alerts must \
         surface the skill — drift's real input is capture-recorded firings; got {:?}",
        alerts.iter().map(|a| &a.skill_id).collect::<Vec<_>>()
    );
}
