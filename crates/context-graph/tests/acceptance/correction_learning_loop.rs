// End-to-end learning-loop tests for the reopened cg-llm-client-refactor correction.
//
// These tests deliberately do not seed cg_learned_skills and do not call the pure promotion
// helpers as proof. They drive the product path: real capture -> correction overlay +
// teaching event -> persisted learned skill -> later run_capture auto-application -> drift
// from the capture-produced firing trace.

use std::collections::{BTreeMap, HashMap};

use context_graph::harness::{CaptureHarnessConfig, run_capture};
use context_graph::ingest::{CaptureReport, CaptureRequest};
use context_graph::skill_store::{
    LearnedSkill, SkillApplyStrategy, SkillEffectEnvelope, SkillProvenance, apply_correction,
    list_drift_alerts, list_skills_in_store, list_teaching_events_in_store, persist_skill,
    record_firing_label,
};
use context_graph::{Entity, EntityType, ListEntityFilter, Store, StoreConfig};
use contextdb_core::Value as DbValue;

use super::fake_extractor::open_store_with_lexical_embedder;

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("correction_learning_loop.db"),
        default_text_embedder: None,
        ..Default::default()
    });
    (dir, store)
}

fn retype_effect(entity_name: &str, from_type: &str, to_type: &str) -> SkillEffectEnvelope {
    SkillEffectEnvelope::RetypeEntity {
        entity_name: entity_name.to_string(),
        from_type: from_type.to_string(),
        to_type: to_type.to_string(),
    }
}

fn repository_transcript() -> String {
    [
        "Goal G1: keep the context-graph repository classified",
        "Entity E1: context-graph type=file",
        "Decision D1: treat context-graph as the repository root serves G1",
    ]
    .join("\n")
}

fn repository_with_ingest_transcript() -> String {
    [
        "Goal G1: keep the context-graph repository classified with ingest",
        "Entity E1: context-graph type=file",
        "Entity E2: ingest type=file",
        "Decision D1: treat context-graph as the repository root and ingest as the module serves G1",
    ]
    .join("\n")
}

fn run_repository_capture(store: &Store, context_label: &str) -> Entity {
    let (_report, _harness) = run_capture(
        store,
        CaptureRequest {
            context_label: context_label.to_string(),
            transcript: repository_transcript(),
            model_available: true,
        },
        &super::refactor_extractor::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("repository capture must succeed");
    entity_named_in_context(store, context_label, "context-graph")
}

fn entity_named_in_context(store: &Store, context_label: &str, entity_name: &str) -> Entity {
    let context_id = store
        .list_contexts()
        .expect("list contexts")
        .into_iter()
        .find(|ctx| ctx.name == context_label)
        .map(|ctx| ctx.id)
        .expect("capture context must exist");
    store
        .list_entities(ListEntityFilter {
            context_id: Some(context_id),
            name_contains: Some(entity_name.to_string()),
            ..Default::default()
        })
        .expect("list entities")
        .into_iter()
        .find(|entity| entity.name == entity_name)
        .expect("captured entity must exist")
}

fn correct_entity_type(
    store: &Store,
    entity: &Entity,
    corrected_type: &str,
    skill_id: Option<&str>,
) {
    let item_id = entity.id.to_string();
    let overlay = context_graph::overlay::record_correction(store, &item_id, corrected_type);
    assert_eq!(
        overlay.item_id, item_id,
        "the correction overlay must target the captured entity"
    );
    let correction = apply_correction(store, &item_id, corrected_type, skill_id);
    assert!(
        correction.applied,
        "the correction entrypoint must accept the real captured entity id"
    );
    assert!(
        correction.teaching_event.is_some(),
        "the correction entrypoint must log a teaching event for learning"
    );
}

fn teach_repository_skill_from_real_corrections(store: &Store) -> Vec<LearnedSkill> {
    for index in 0..3 {
        let context_label = format!("repository-label-{index}");
        let entity = run_repository_capture(store, &context_label);
        assert_eq!(
            entity.entity_type,
            EntityType::File,
            "the training fixture must first capture the repository as File so the correction is real"
        );
        correct_entity_type(store, &entity, "Repository", None);
    }
    list_skills_in_store(store)
}

fn learned_repository_retype_skill(skills: &[LearnedSkill]) -> Option<&LearnedSkill> {
    skills.iter().find(|skill| {
        let text = format!("{} {}", skill.description, skill.rule).to_ascii_lowercase();
        skill.active
            && skill.provenance == SkillProvenance::HumanCorrection
            && text.contains("context-graph")
            && text.contains("file")
            && text.contains("repository")
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SkillFiringRow {
    id: String,
    capture_id: String,
    correct: Option<bool>,
    created_at: i128,
    effect_entity_key: Option<String>,
    effect_entity_name: Option<String>,
    effect_from_type: Option<String>,
    effect_to_type: Option<String>,
}

fn firing_traces_for_skill(store: &Store, skill_id: &str) -> Vec<SkillFiringRow> {
    let db = store.sync_database();
    db.scan("cg_skill_firing_trace", db.snapshot())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| {
            let row_skill_id = match row.values.get("skill_id") {
                Some(DbValue::Text(value)) => value,
                _ => return None,
            };
            if row_skill_id != skill_id {
                return None;
            }
            let id = match row.values.get("id") {
                Some(DbValue::Text(id)) => id.clone(),
                Some(DbValue::Uuid(id)) => id.to_string(),
                _ => return None,
            };
            let capture_id = match row.values.get("capture_id") {
                Some(DbValue::Text(capture_id)) => capture_id.clone(),
                Some(DbValue::Uuid(capture_id)) => capture_id.to_string(),
                _ => return None,
            };
            let correct = match row.values.get("correct") {
                Some(DbValue::Bool(value)) => Some(*value),
                Some(DbValue::Int64(value)) => Some(*value != 0),
                Some(DbValue::Null) | None => None,
                _ => return None,
            };
            let created_at = match row.values.get("created_at") {
                Some(DbValue::TxId(tx)) => i128::from(tx.0),
                Some(DbValue::Int64(value)) => i128::from(*value),
                _ => return None,
            };
            let optional_text = |field: &str| match row.values.get(field) {
                Some(DbValue::Text(value)) => Some(value.clone()),
                Some(DbValue::Null) | None => None,
                _ => None,
            };
            Some(SkillFiringRow {
                id,
                capture_id,
                correct,
                created_at,
                effect_entity_key: optional_text("effect_entity_key"),
                effect_entity_name: optional_text("effect_entity_name"),
                effect_from_type: optional_text("effect_from_type"),
                effect_to_type: optional_text("effect_to_type"),
            })
        })
        .collect()
}

fn firing_trace_ids_for_skill(store: &Store, skill_id: &str) -> Vec<String> {
    firing_traces_for_skill(store, skill_id)
        .into_iter()
        .map(|trace| trace.id)
        .collect()
}

fn rewrite_skill_rule(store: &Store, skill_id: &str, rule: &str) {
    let db = store.sync_database();
    let result = db
        .execute(
            "UPDATE cg_learned_skills SET rule = $rule WHERE id = $id",
            &HashMap::from([
                ("id".to_string(), DbValue::Text(skill_id.to_string())),
                ("rule".to_string(), DbValue::Text(rule.to_string())),
            ]),
        )
        .expect("rewrite skill rule");
    assert_eq!(
        result.rows_affected, 1,
        "the skill rule rewrite must update exactly one row for {skill_id}"
    );
}

struct ChainedRetypeCorrection<'a> {
    target_digest_id: &'a str,
    entity_name: &'a str,
    starting_type: &'a str,
    first_skill_id: &'a str,
    second_skill_id: &'a str,
    branch_skill_id: &'a str,
    other_skill_id: &'a str,
}

fn correct_chained_retype_after_reopen_with_rules_rewritten(
    store: &Store,
    fixture: ChainedRetypeCorrection<'_>,
) {
    rewrite_skill_rule(
        store,
        fixture.first_skill_id,
        &format!(
            "when an entity named unrelated-{} is typed Team → Location",
            fixture.entity_name
        ),
    );
    rewrite_skill_rule(
        store,
        fixture.second_skill_id,
        &format!(
            "when an entity named unrelated-{} is typed Database → Team",
            fixture.entity_name
        ),
    );
    rewrite_skill_rule(
        store,
        fixture.branch_skill_id,
        &format!(
            "when an entity named unrelated-{} is typed Configuration → Pipeline",
            fixture.entity_name
        ),
    );
    rewrite_skill_rule(
        store,
        fixture.other_skill_id,
        &format!(
            "when an entity named unrelated-{} is typed Document → Team",
            fixture.entity_name
        ),
    );
    let correction = apply_correction(
        store,
        fixture.target_digest_id,
        &format!("{} {}", fixture.starting_type, fixture.entity_name),
        None,
    );
    assert!(
        correction.applied,
        "the reopened store must accept the digest correction using only persisted trace metadata"
    );
}

fn single_digest_id(report: &CaptureReport) -> String {
    let mut digest_ids: Vec<String> = report
        .provenance_edges
        .iter()
        .map(|edge| edge.digest_id.as_uuid().to_string())
        .collect();
    digest_ids.sort();
    digest_ids.dedup();
    assert_eq!(
        digest_ids.len(),
        1,
        "each capture in this fixture must produce exactly one captured unit; provenance_edges={:?}",
        report.provenance_edges
    );
    digest_ids.pop().expect("digest id")
}

#[test]
fn repeated_real_corrections_promote_persisted_skill_then_run_capture_auto_applies_it() {
    let (_dir, store) = fresh_store();

    let skills = teach_repository_skill_from_real_corrections(&store);
    let teaching_events = list_teaching_events_in_store(&store);
    assert!(
        teaching_events.len() >= 3,
        "the setup must have logged at least three real correction teaching events; got {}",
        teaching_events.len()
    );
    assert!(
        !skills.is_empty(),
        "repeated real corrections through the correction entrypoint must promote and persist an \
         auto-applying learned skill; current code logs lessons but leaves cg_learned_skills empty"
    );

    let skill = learned_repository_retype_skill(&skills).unwrap_or_else(|| {
        panic!(
            "the promoted skill must derive its rule/description from the corrected entity/type \
             labels (context-graph File -> Repository) and carry human-correction provenance; \
             skills={skills:?}"
        )
    });
    let learned_skill_id = skill.id.clone();

    let (_baseline_dir, baseline_store) = fresh_store();
    let baseline = run_repository_capture(&baseline_store, "baseline-no-skill");
    let (_report, harness) = run_capture(
        &store,
        CaptureRequest {
            context_label: "post-learning-applies".to_string(),
            transcript: repository_transcript(),
            model_available: true,
        },
        &super::refactor_extractor::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("post-learning run_capture must succeed");
    let corrected = entity_named_in_context(&store, "post-learning-applies", "context-graph");

    assert_eq!(
        corrected.entity_type,
        EntityType::Repository,
        "the learned skill must auto-apply on a real run_capture path, changing the repeated \
         repository mis-capture from File to Repository without a seeded skill row"
    );
    assert!(
        harness
            .applied_skill_ids
            .iter()
            .any(|id| id == &learned_skill_id),
        "run_capture must report the learned skill as applied; applied_skill_ids={:?}",
        harness.applied_skill_ids
    );
    assert!(
        !firing_trace_ids_for_skill(&store, &learned_skill_id).is_empty(),
        "the auto-applied learned skill must record a capture-produced firing trace for drift"
    );
    assert_eq!(
        baseline.entity_type,
        EntityType::File,
        "sanity: a skill-free capture of the same model output is still File"
    );
}

#[test]
fn correction_after_capture_produced_skill_fire_labels_drift_without_direct_firing_label() {
    let (_noop_dir, noop_store) = fresh_store();
    let noop_skill_id = "skill-cofired-module-repository-noop".to_string();
    assert!(
        persist_skill(
            &noop_store,
            &LearnedSkill {
                id: noop_skill_id.clone(),
                description: "context-graph module to repository co-firing guard".to_string(),
                rule: "when an entity named context-graph is typed Module → Repository".to_string(),
                effect: Some(retype_effect("context-graph", "Module", "Repository")),
                apply_strategy: SkillApplyStrategy::PostHarvestRetype,
                provenance: SkillProvenance::HumanCorrection,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            }
        ),
        "the no-op fixture must persist the co-fired retype skill"
    );
    let (noop_report, noop_harness) = run_capture(
        &noop_store,
        CaptureRequest {
            context_label: "cofired-retype-noop".to_string(),
            transcript: repository_transcript(),
            model_available: true,
        },
        &super::refactor_extractor::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("co-fired no-op capture must succeed");
    assert!(
        noop_harness
            .applied_skill_ids
            .iter()
            .any(|id| id == &noop_skill_id),
        "the co-fired retype skill must be selected so the negative-control trace is real; applied_skill_ids={:?}",
        noop_harness.applied_skill_ids
    );
    let noop_entity = entity_named_in_context(&noop_store, "cofired-retype-noop", "context-graph");
    assert_eq!(
        noop_entity.entity_type,
        EntityType::File,
        "the Module -> Repository co-fired skill must not change a File entity; otherwise the drift \
         negative control is not proving non-changing retype attribution"
    );
    let noop_digest_id = single_digest_id(&noop_report);
    let noop_traces = firing_traces_for_skill(&noop_store, &noop_skill_id);
    assert_eq!(
        noop_traces.len(),
        1,
        "the no-op co-fired skill must record exactly one capture-produced firing trace; \
         noop_traces={noop_traces:?}"
    );
    assert_eq!(
        (noop_traces[0].capture_id.as_str(), noop_traces[0].correct),
        (noop_digest_id.as_str(), None),
        "the no-op co-fired skill trace must be keyed to the digest it selected on"
    );

    let (_dir, store) = fresh_store();
    let unrelated_skill_id = "skill-unrelated-context-graph-note".to_string();
    let cofired_retype_skill_id = "skill-cofired-module-repository".to_string();
    let cofired_effectful_skill_id = "skill-cofired-ingest-module".to_string();
    assert!(
        persist_skill(
            &store,
            &LearnedSkill {
                id: cofired_retype_skill_id.clone(),
                description: "context-graph module to repository co-firing guard".to_string(),
                rule: "when an entity named context-graph is typed Module → Repository".to_string(),
                effect: Some(retype_effect("context-graph", "Module", "Repository")),
                apply_strategy: SkillApplyStrategy::PostHarvestRetype,
                provenance: SkillProvenance::HumanCorrection,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            }
        ),
        "the co-fired retype skill must be older than the learned File -> Repository skill so \
         first-matching-skill shortcuts fail"
    );
    assert!(
        persist_skill(
            &store,
            &LearnedSkill {
                id: cofired_effectful_skill_id.clone(),
                description: "ingest module co-firing guard".to_string(),
                rule: "when an entity named ingest is typed File → Module".to_string(),
                effect: Some(retype_effect("ingest", "File", "Module")),
                apply_strategy: SkillApplyStrategy::PostHarvestRetype,
                provenance: SkillProvenance::HumanCorrection,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            }
        ),
        "the effectful co-fired retype skill must be older than the learned File -> Repository skill \
         so first-effectful-skill shortcuts fail"
    );

    let skills = teach_repository_skill_from_real_corrections(&store);
    let skill = learned_repository_retype_skill(&skills).unwrap_or_else(|| {
        panic!(
            "drift must start from a skill learned by repeated real corrections, not a seeded \
             cg_learned_skills row; skills={skills:?}"
        )
    });
    let learned_skill_id = skill.id.clone();
    let cofired_training_traces = firing_traces_for_skill(&store, &cofired_retype_skill_id);
    assert_eq!(
        cofired_training_traces.len(),
        3,
        "because the co-fired retype skill is older than the learned skill, it should have fired on \
         the three training captures before drift; cofired_training_traces={cofired_training_traces:?}"
    );
    let cofired_training_by_id = cofired_training_traces
        .iter()
        .map(|trace| (trace.id.clone(), (trace.capture_id.clone(), trace.correct)))
        .collect::<BTreeMap<_, _>>();
    assert!(
        cofired_training_by_id
            .values()
            .all(|(_, correct)| correct.is_none()),
        "training co-fired traces must be unresolved before the drift correction; \
         cofired_training_traces={cofired_training_traces:?}"
    );
    let effectful_training_traces = firing_traces_for_skill(&store, &cofired_effectful_skill_id);
    assert!(
        effectful_training_traces.is_empty(),
        "the effectful co-fired skill is older than the learned skill, but the training captures do \
         not mention ingest, so it must not have pre-drift traces; \
         effectful_training_traces={effectful_training_traces:?}"
    );
    assert!(
        persist_skill(
            &store,
            &LearnedSkill {
                id: unrelated_skill_id.clone(),
                description: "context-graph unrelated note for co-firing guard".to_string(),
                // A derivable effect whose source type ("Service") never matches the actual
                // harvested type ("File") — same no-op-co-fire shape as `cofired_retype_skill_id`
                // above: the skill is selected and fires (matches by description/scope) but never
                // changes an entity, so it stays a real co-firing negative control. An
                // effect-less rule would now be withheld under the fold-seam contract (an
                // apply-strategy row whose effect cannot be derived never fires), which would
                // defeat this fixture's whole point.
                rule: "when an entity named context-graph is typed Service → Module".to_string(),
                effect: Some(retype_effect("context-graph", "Service", "Module")),
                apply_strategy: SkillApplyStrategy::PostHarvestRetype,
                provenance: SkillProvenance::HumanAuthored,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            }
        ),
        "the fixture must add an unrelated co-firing skill so correction attribution cannot label \
         every skill that fired on the corrected digest"
    );

    let (stale_report, stale_harness) = run_capture(
        &store,
        CaptureRequest {
            context_label: "drift-stale-capture".to_string(),
            transcript: repository_with_ingest_transcript(),
            model_available: true,
        },
        &super::refactor_extractor::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("stale skill-fire capture must succeed");
    assert!(
        stale_harness
            .applied_skill_ids
            .iter()
            .any(|id| id == &learned_skill_id),
        "the learned skill must fire on a stale earlier capture so oldest-firing shortcuts are caught; applied_skill_ids={:?}",
        stale_harness.applied_skill_ids
    );
    let stale_digest_id = single_digest_id(&stale_report);

    let (target_report, target_harness) = run_capture(
        &store,
        CaptureRequest {
            context_label: "drift-target-capture".to_string(),
            transcript: repository_with_ingest_transcript(),
            model_available: true,
        },
        &super::refactor_extractor::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("target skill-fire capture must succeed");
    assert!(
        target_harness
            .applied_skill_ids
            .iter()
            .any(|id| id == &learned_skill_id),
        "the learned skill must fire on the target capture before drift can be labeled; applied_skill_ids={:?}",
        target_harness.applied_skill_ids
    );
    let target_digest_id = single_digest_id(&target_report);

    let (later_report, later_harness) = run_capture(
        &store,
        CaptureRequest {
            context_label: "drift-later-capture".to_string(),
            transcript: repository_with_ingest_transcript(),
            model_available: true,
        },
        &super::refactor_extractor::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("later skill-fire capture must succeed");
    assert!(
        later_harness
            .applied_skill_ids
            .iter()
            .any(|id| id == &learned_skill_id),
        "the learned skill must also fire on a later capture so a latest-firing shortcut is caught; applied_skill_ids={:?}",
        later_harness.applied_skill_ids
    );
    let later_digest_id = single_digest_id(&later_report);
    assert_ne!(
        target_digest_id, later_digest_id,
        "the drift test needs two distinct captured units"
    );

    let traces_before = firing_traces_for_skill(&store, &learned_skill_id);
    let unrelated_traces_before = firing_traces_for_skill(&store, &unrelated_skill_id);
    let cofired_retype_traces_before = firing_traces_for_skill(&store, &cofired_retype_skill_id);
    let effectful_retype_traces_before =
        firing_traces_for_skill(&store, &cofired_effectful_skill_id);
    assert_eq!(
        traces_before.len(),
        3,
        "this drift fixture should have exactly the stale, target, and later capture-produced \
         traces for the learned skill before correction; traces={traces_before:?}"
    );
    assert_eq!(
        unrelated_traces_before.len(),
        3,
        "the unrelated skill should co-fire on the same three captures before correction; \
         unrelated_traces_before={unrelated_traces_before:?}"
    );
    assert_eq!(
        cofired_retype_traces_before.len(),
        6,
        "the older co-fired retype skill should have three training traces plus traces on the same \
         stale, target, and later captures before correction; \
         before correction; cofired_retype_traces_before={cofired_retype_traces_before:?}"
    );
    assert_eq!(
        effectful_retype_traces_before.len(),
        3,
        "the effectful co-fired retype skill should have traces on the same stale, target, and later \
         captures before correction; effectful_retype_traces_before={effectful_retype_traces_before:?}"
    );
    let stale_trace = traces_before
        .iter()
        .find(|trace| trace.capture_id == stale_digest_id && trace.correct.is_none())
        .unwrap_or_else(|| {
            panic!(
                "the stale capture must have its own unresolved firing trace before correction; traces={traces_before:?}"
            )
        })
        .clone();
    let target_trace = traces_before
        .iter()
        .find(|trace| trace.capture_id == target_digest_id && trace.correct.is_none())
        .unwrap_or_else(|| {
            panic!(
                "the target capture must have its own unresolved firing trace before correction; traces={traces_before:?}"
            )
        })
        .clone();
    let later_trace = traces_before
        .iter()
        .find(|trace| trace.capture_id == later_digest_id && trace.correct.is_none())
        .unwrap_or_else(|| {
            panic!(
                "the later capture must have its own unresolved firing trace so latest-firing shortcuts fail; traces={traces_before:?}"
            )
        })
        .clone();
    let mut trace_ids_before = traces_before
        .iter()
        .map(|trace| trace.id.clone())
        .collect::<Vec<_>>();
    trace_ids_before.sort();
    let expected_before_by_id = BTreeMap::from([
        (stale_trace.id.clone(), (stale_digest_id.clone(), None)),
        (target_trace.id.clone(), (target_digest_id.clone(), None)),
        (later_trace.id.clone(), (later_digest_id.clone(), None)),
    ]);
    let actual_before_by_id = traces_before
        .iter()
        .map(|trace| (trace.id.clone(), (trace.capture_id.clone(), trace.correct)))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        actual_before_by_id, expected_before_by_id,
        "before correction, the learned skill must have only the three fixture traces and all must \
         be unresolved; traces={traces_before:?}"
    );
    assert!(
        unrelated_traces_before
            .iter()
            .any(|trace| trace.capture_id == target_digest_id && trace.correct.is_none()),
        "the unrelated skill must have an unresolved co-firing on the corrected target digest; \
         unrelated_traces_before={unrelated_traces_before:?}"
    );
    let stale_cofired_trace = cofired_retype_traces_before
        .iter()
        .find(|trace| trace.capture_id == stale_digest_id && trace.correct.is_none())
        .unwrap_or_else(|| {
            panic!(
                "the co-fired retype skill must have an unresolved firing on the stale digest; \
                 cofired_retype_traces_before={cofired_retype_traces_before:?}"
            )
        })
        .clone();
    let target_cofired_trace = cofired_retype_traces_before
        .iter()
        .find(|trace| trace.capture_id == target_digest_id && trace.correct.is_none())
        .unwrap_or_else(|| {
            panic!(
                "the co-fired retype skill must have an unresolved firing on the corrected target \
                 digest; cofired_retype_traces_before={cofired_retype_traces_before:?}"
            )
        })
        .clone();
    let later_cofired_trace = cofired_retype_traces_before
        .iter()
        .find(|trace| trace.capture_id == later_digest_id && trace.correct.is_none())
        .unwrap_or_else(|| {
            panic!(
                "the co-fired retype skill must have an unresolved firing on the later digest; \
                 cofired_retype_traces_before={cofired_retype_traces_before:?}"
            )
        })
        .clone();
    let cofired_before_by_id = cofired_retype_traces_before
        .iter()
        .map(|trace| (trace.id.clone(), (trace.capture_id.clone(), trace.correct)))
        .collect::<BTreeMap<_, _>>();
    let mut expected_cofired_before_by_id = cofired_training_by_id.clone();
    expected_cofired_before_by_id.insert(
        stale_cofired_trace.id.clone(),
        (stale_digest_id.clone(), None),
    );
    expected_cofired_before_by_id.insert(
        target_cofired_trace.id.clone(),
        (target_digest_id.clone(), None),
    );
    expected_cofired_before_by_id.insert(
        later_cofired_trace.id.clone(),
        (later_digest_id.clone(), None),
    );
    assert_eq!(
        cofired_before_by_id, expected_cofired_before_by_id,
        "the co-fired retype skill must have exactly the three training traces plus one unresolved \
         trace for each drift fixture digest; \
         cofired_retype_traces_before={cofired_retype_traces_before:?}"
    );
    let stale_effectful_trace = effectful_retype_traces_before
        .iter()
        .find(|trace| trace.capture_id == stale_digest_id && trace.correct.is_none())
        .unwrap_or_else(|| {
            panic!(
                "the effectful co-fired skill must have an unresolved firing on the stale digest; \
                 effectful_retype_traces_before={effectful_retype_traces_before:?}"
            )
        })
        .clone();
    let target_effectful_trace = effectful_retype_traces_before
        .iter()
        .find(|trace| trace.capture_id == target_digest_id && trace.correct.is_none())
        .unwrap_or_else(|| {
            panic!(
                "the effectful co-fired skill must have an unresolved firing on the corrected target \
                 digest; effectful_retype_traces_before={effectful_retype_traces_before:?}"
            )
        })
        .clone();
    let later_effectful_trace = effectful_retype_traces_before
        .iter()
        .find(|trace| trace.capture_id == later_digest_id && trace.correct.is_none())
        .unwrap_or_else(|| {
            panic!(
                "the effectful co-fired skill must have an unresolved firing on the later digest; \
                 effectful_retype_traces_before={effectful_retype_traces_before:?}"
            )
        })
        .clone();
    let effectful_before_by_id = effectful_retype_traces_before
        .iter()
        .map(|trace| (trace.id.clone(), (trace.capture_id.clone(), trace.correct)))
        .collect::<BTreeMap<_, _>>();
    let expected_effectful_before_by_id = BTreeMap::from([
        (
            stale_effectful_trace.id.clone(),
            (stale_digest_id.clone(), None),
        ),
        (
            target_effectful_trace.id.clone(),
            (target_digest_id.clone(), None),
        ),
        (
            later_effectful_trace.id.clone(),
            (later_digest_id.clone(), None),
        ),
    ]);
    assert_eq!(
        effectful_before_by_id, expected_effectful_before_by_id,
        "the effectful co-fired skill must have exactly one unresolved trace for each drift fixture \
         digest; effectful_retype_traces_before={effectful_retype_traces_before:?}"
    );
    assert!(
        list_drift_alerts(&store).is_empty(),
        "unlabeled firing traces alone must not raise drift before the correction labels a specific \
         firing wrong"
    );

    let corrected_entity = entity_named_in_context(&store, "drift-target-capture", "context-graph");
    assert_eq!(
        corrected_entity.entity_type,
        EntityType::Repository,
        "the drift correction must target a value that the learned skill changed on the capture path"
    );
    let other_changed_entity = entity_named_in_context(&store, "drift-target-capture", "ingest");
    assert_eq!(
        other_changed_entity.entity_type,
        EntityType::Module,
        "the target captured unit must also contain a different value changed by a different \
         same-provenance skill, so digest-wide changed-skill blame cannot pass"
    );
    let overlay =
        context_graph::overlay::record_correction(&store, &target_digest_id, "File context-graph");
    assert_eq!(
        overlay.item_id, target_digest_id,
        "the correction overlay must target the captured unit the user addressed"
    );
    let correction = apply_correction(&store, &target_digest_id, "File context-graph", None);
    assert!(
        correction.applied,
        "the correction entrypoint must accept the captured-unit digest id used by cg fix @n"
    );
    assert!(
        correction.teaching_event.is_some(),
        "the correction entrypoint must log a teaching event for the captured-unit correction"
    );

    let alerts = list_drift_alerts(&store);
    let traces_after = firing_traces_for_skill(&store, &learned_skill_id);
    let unrelated_traces_after = firing_traces_for_skill(&store, &unrelated_skill_id);
    let cofired_retype_traces_after = firing_traces_for_skill(&store, &cofired_retype_skill_id);
    let effectful_retype_traces_after =
        firing_traces_for_skill(&store, &cofired_effectful_skill_id);
    let mut trace_ids_after = traces_after
        .iter()
        .map(|trace| trace.id.clone())
        .collect::<Vec<_>>();
    trace_ids_after.sort();
    assert_eq!(
        trace_ids_after, trace_ids_before,
        "a correction must label the existing capture-produced firing row, not insert a synthetic \
         wrong row; before={traces_before:?} after={traces_after:?}"
    );
    let actual_after_by_id = traces_after
        .iter()
        .map(|trace| (trace.id.clone(), (trace.capture_id.clone(), trace.correct)))
        .collect::<BTreeMap<_, _>>();
    let expected_after_by_id = BTreeMap::from([
        (stale_trace.id.clone(), (stale_digest_id.clone(), None)),
        (
            target_trace.id.clone(),
            (target_digest_id.clone(), Some(false)),
        ),
        (later_trace.id.clone(), (later_digest_id.clone(), None)),
    ]);
    assert_eq!(
        actual_after_by_id, expected_after_by_id,
        "the correction must change exactly one persisted firing row: the corrected target digest's \
         row. No stale, later, synthetic, or unrelated row may be labeled; traces_after={traces_after:?}"
    );
    let unrelated_before_by_id = unrelated_traces_before
        .iter()
        .map(|trace| (trace.id.clone(), (trace.capture_id.clone(), trace.correct)))
        .collect::<BTreeMap<_, _>>();
    let unrelated_after_by_id = unrelated_traces_after
        .iter()
        .map(|trace| (trace.id.clone(), (trace.capture_id.clone(), trace.correct)))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        unrelated_after_by_id, unrelated_before_by_id,
        "the correction must not label or insert traces for an unrelated skill that fired on the \
         same corrected digest; unrelated_before={unrelated_traces_before:?} \
         unrelated_after={unrelated_traces_after:?}"
    );
    let cofired_after_by_id = cofired_retype_traces_after
        .iter()
        .map(|trace| (trace.id.clone(), (trace.capture_id.clone(), trace.correct)))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        cofired_after_by_id, cofired_before_by_id,
        "the exact co-fired HumanCorrection retype trace rows must stay unresolved after correcting \
         the target digest because that skill did not change the File value"
    );
    let effectful_after_by_id = effectful_retype_traces_after
        .iter()
        .map(|trace| (trace.id.clone(), (trace.capture_id.clone(), trace.correct)))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        effectful_after_by_id, effectful_before_by_id,
        "the exact same-provenance effectful co-fired trace rows must stay unresolved after correcting \
         context-graph because that different skill changed ingest, not the corrected File value"
    );
    assert!(
        traces_after
            .iter()
            .any(|trace| trace.id == target_trace.id && trace.correct == Some(false)),
        "the correction must label the exact firing trace row for the corrected captured unit, not \
         a generic/latest/oldest skill firing; target_trace={target_trace:?} traces={traces_after:?}"
    );
    assert!(
        traces_after
            .iter()
            .any(|trace| trace.id == stale_trace.id && trace.correct.is_none()),
        "a correction to the target captured unit must not label the earlier unresolved firing; \
         stale_trace={stale_trace:?} traces={traces_after:?}"
    );
    assert!(
        traces_after
            .iter()
            .any(|trace| trace.id == later_trace.id && trace.correct.is_none()),
        "a correction to the earlier captured unit must not label the later unresolved firing; \
         later_trace={later_trace:?} traces={traces_after:?}"
    );
    assert_eq!(
        alerts.len(),
        1,
        "only the learned skill whose exact target trace was labeled wrong may alert; alerts={alerts:?}"
    );
    assert_eq!(
        alerts[0].skill_id, learned_skill_id,
        "a correction after a capture-produced skill firing must label that firing wrong and make \
         list_drift_alerts surface the learned skill; direct record_firing_label-only drift tests \
         do not prove this path. traces={traces_after:?} alerts={alerts:?}"
    );

    record_firing_label(&store, &target_trace.id, true);
    let traces_after_relabel = firing_traces_for_skill(&store, &learned_skill_id);
    let actual_after_relabel_by_id = traces_after_relabel
        .iter()
        .map(|trace| (trace.id.clone(), (trace.capture_id.clone(), trace.correct)))
        .collect::<BTreeMap<_, _>>();
    let expected_after_relabel_by_id = BTreeMap::from([
        (stale_trace.id.clone(), (stale_digest_id.clone(), None)),
        (target_trace.id.clone(), (target_digest_id, Some(true))),
        (later_trace.id.clone(), (later_digest_id, None)),
    ]);
    assert_eq!(
        actual_after_relabel_by_id, expected_after_relabel_by_id,
        "the same target firing row must remain digest-keyed when its label changes; \
         traces_after_relabel={traces_after_relabel:?}"
    );
    assert!(
        list_drift_alerts(&store).is_empty(),
        "drift must be derived from the persisted firing trace labels: when the same target firing \
         is no longer wrong, the alert must disappear"
    );
}

#[test]
fn correction_after_chained_retype_labels_each_effect_that_changed_the_value() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("chained_retype_drift.db");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: db_path.clone(),
        default_text_embedder: None,
        ..Default::default()
    });
    let fixture_uuid = uuid::Uuid::new_v4();
    let fixture_token = fixture_uuid.simple().to_string();
    let entity_name = format!("unit-{fixture_token}");
    let other_entity_name = format!("side-{fixture_token}");
    let entity_key = format!("E{}", 10 + usize::from(fixture_uuid.as_bytes()[4] % 80));
    let other_entity_key = format!("E{}", 100 + usize::from(fixture_uuid.as_bytes()[5] % 80));
    let first_changing_skill_id = format!("sk-{}", uuid::Uuid::new_v4().simple());
    let second_changing_skill_id = format!("sk-{}", uuid::Uuid::new_v4().simple());
    let unapplied_branch_skill_id = format!("sk-{}", uuid::Uuid::new_v4().simple());
    let other_effectful_skill_id = format!("sk-{}", uuid::Uuid::new_v4().simple());
    let ontology_types = [
        ("Service", "service", EntityType::Service),
        ("Person", "person", EntityType::Person),
        ("Device", "device", EntityType::Device),
        ("Module", "module", EntityType::Module),
        ("Repository", "repository", EntityType::Repository),
        ("Configuration", "configuration", EntityType::Configuration),
        (
            "Infrastructure",
            "infrastructure",
            EntityType::Infrastructure,
        ),
        ("Location", "location", EntityType::Location),
        ("File", "file", EntityType::File),
        ("Pipeline", "pipeline", EntityType::Pipeline),
        ("Database", "database", EntityType::Database),
        ("Team", "team", EntityType::Team),
        ("Document", "document", EntityType::Document),
    ];
    let choose_unused = |seed: u8, used: &[usize]| -> usize {
        let mut index = usize::from(seed) % ontology_types.len();
        while used.contains(&index) {
            index = (index + 1) % ontology_types.len();
        }
        index
    };
    let bytes = fixture_uuid.as_bytes();
    let starting_index = choose_unused(bytes[0], &[]);
    let intermediate_index = choose_unused(bytes[1], &[starting_index]);
    let final_index = choose_unused(bytes[2], &[starting_index, intermediate_index]);
    let branch_index = choose_unused(bytes[3], &[starting_index, intermediate_index, final_index]);
    let (starting_type, transcript_type, _starting_entity_type) = ontology_types[starting_index];
    let (intermediate_type, _intermediate_transcript_type, _intermediate_entity_type) =
        ontology_types[intermediate_index];
    let (final_type, _final_transcript_type, expected_final_entity_type) =
        ontology_types[final_index];
    let (branch_type, _branch_transcript_type, _branch_entity_type) = ontology_types[branch_index];
    let other_start_index = choose_unused(bytes[6], &[starting_index]);
    let other_final_index = choose_unused(bytes[7], &[other_start_index]);
    let (other_starting_type, other_transcript_type, _other_starting_entity_type) =
        ontology_types[other_start_index];
    let (other_final_type, _other_final_transcript_type, expected_other_entity_type) =
        ontology_types[other_final_index];
    let transcript = [
        format!("Goal G1: keep {entity_name} classified"),
        format!("Entity {entity_key}: {entity_name} type={transcript_type}"),
        format!("Entity {other_entity_key}: {other_entity_name} type={other_transcript_type}"),
        format!("Decision D1: treat {entity_name} as the runtime unit serves G1"),
    ]
    .join("\n");
    assert!(
        persist_skill(
            &store,
            &LearnedSkill {
                id: first_changing_skill_id.clone(),
                description: format!("{entity_name} {}", uuid::Uuid::new_v4().simple()),
                rule: format!(
                    "when an entity named {entity_name} is typed {starting_type} → {intermediate_type}"
                ),
                effect: Some(retype_effect(
                    &entity_name,
                    starting_type,
                    intermediate_type
                )),
                apply_strategy: SkillApplyStrategy::PostHarvestRetype,
                provenance: SkillProvenance::HumanCorrection,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            }
        ),
        "the first chain skill must persist"
    );
    assert!(
        persist_skill(
            &store,
            &LearnedSkill {
                id: second_changing_skill_id.clone(),
                description: format!("{entity_name} {}", uuid::Uuid::new_v4().simple()),
                rule: format!(
                    "when an entity named {entity_name} is typed {intermediate_type} → {final_type}"
                ),
                effect: Some(retype_effect(&entity_name, intermediate_type, final_type)),
                apply_strategy: SkillApplyStrategy::PostHarvestRetype,
                provenance: SkillProvenance::HumanCorrection,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            }
        ),
        "the second chain skill must persist"
    );
    assert!(
        persist_skill(
            &store,
            &LearnedSkill {
                id: unapplied_branch_skill_id.clone(),
                description: format!("{entity_name} {}", uuid::Uuid::new_v4().simple()),
                rule: format!(
                    "when an entity named {entity_name} is typed {starting_type} → {branch_type}"
                ),
                effect: Some(retype_effect(&entity_name, starting_type, branch_type)),
                apply_strategy: SkillApplyStrategy::PostHarvestRetype,
                provenance: SkillProvenance::HumanCorrection,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            }
        ),
        "the selected-but-unapplied branch skill must persist"
    );
    assert!(
        persist_skill(
            &store,
            &LearnedSkill {
                id: other_effectful_skill_id.clone(),
                description: format!("{other_entity_name} {}", uuid::Uuid::new_v4().simple()),
                rule: format!(
                    "when an entity named {other_entity_name} is typed {other_starting_type} → {other_final_type}"
                ),
                effect: Some(retype_effect(
                    &other_entity_name,
                    other_starting_type,
                    other_final_type,
                )),
                apply_strategy: SkillApplyStrategy::PostHarvestRetype,
                provenance: SkillProvenance::HumanCorrection,
                active: true,
                held_for_review: false,
                fire_right: 0,
                fire_wrong: 0,
            }
        ),
        "the unrelated effectful skill must persist"
    );

    let mut digest_ids = Vec::new();
    for context_label in [
        "chained-retype-stale",
        "chained-retype-target",
        "chained-retype-later",
    ] {
        let (report, harness) = run_capture(
            &store,
            CaptureRequest {
                context_label: context_label.to_string(),
                transcript: transcript.clone(),
                model_available: true,
            },
            &super::refactor_extractor::SpanFaithfulExtractor,
            &CaptureHarnessConfig::default(),
        )
        .expect("chained retype capture must succeed");
        assert!(
            harness
                .applied_skill_ids
                .iter()
                .any(|id| id == &first_changing_skill_id),
            "the first opaque skill must be selected on the capture path; applied_skill_ids={:?}",
            harness.applied_skill_ids
        );
        assert!(
            harness
                .applied_skill_ids
                .iter()
                .any(|id| id == &second_changing_skill_id),
            "the second opaque skill must be selected on the capture path; applied_skill_ids={:?}",
            harness.applied_skill_ids
        );
        assert!(
            harness
                .applied_skill_ids
                .iter()
                .any(|id| id == &unapplied_branch_skill_id),
            "the branch skill must be selected on the same digest so digest-wide blame cannot pass; \
             applied_skill_ids={:?}",
            harness.applied_skill_ids
        );
        assert!(
            harness
                .applied_skill_ids
                .iter()
                .any(|id| id == &other_effectful_skill_id),
            "the unrelated effectful skill must be selected on the same digest so effectful \
             digest-wide blame cannot pass; applied_skill_ids={:?}",
            harness.applied_skill_ids
        );
        let changed_entity = entity_named_in_context(&store, context_label, &entity_name);
        assert_eq!(
            changed_entity.entity_type, expected_final_entity_type,
            "both changing skills must have actually changed the captured starting value before \
             drift is corrected"
        );
        let other_changed_entity =
            entity_named_in_context(&store, context_label, &other_entity_name);
        assert_eq!(
            other_changed_entity.entity_type, expected_other_entity_type,
            "the unrelated skill must actually change another entity on the same capture"
        );
        digest_ids.push(single_digest_id(&report));
    }
    let stale_digest_id = digest_ids[0].clone();
    let target_digest_id = digest_ids[1].clone();
    let later_digest_id = digest_ids[2].clone();
    assert_ne!(
        stale_digest_id, target_digest_id,
        "the stale and target captured units must be distinct so oldest-firing shortcuts fail"
    );
    assert_ne!(
        target_digest_id, later_digest_id,
        "the target and later captured units must be distinct so latest-firing shortcuts fail"
    );
    assert_ne!(
        stale_digest_id, later_digest_id,
        "the stale and later captured units must be distinct"
    );

    let first_before = firing_traces_for_skill(&store, &first_changing_skill_id);
    let second_before = firing_traces_for_skill(&store, &second_changing_skill_id);
    let branch_before = firing_traces_for_skill(&store, &unapplied_branch_skill_id);
    let other_before = firing_traces_for_skill(&store, &other_effectful_skill_id);
    assert_eq!(
        first_before.len(),
        3,
        "the first changing skill must have stale, target, and later traces"
    );
    assert_eq!(
        second_before.len(),
        3,
        "the second changing skill must have stale, target, and later traces"
    );
    assert_eq!(
        branch_before.len(),
        3,
        "the selected branch skill must have stale, target, and later co-firing traces"
    );
    assert_eq!(
        other_before.len(),
        3,
        "the unrelated effectful skill must have stale, target, and later traces"
    );
    let first_target_trace = first_before
        .iter()
        .find(|trace| trace.capture_id == target_digest_id && trace.correct.is_none());
    assert!(
        first_target_trace.is_some(),
        "the first changing skill must have an unresolved trace keyed to the target digest, not an \
         older/latest/synthetic capture id; target_digest_id={target_digest_id} traces={first_before:?}"
    );
    let first_target_trace = first_target_trace.expect("asserted present");
    let second_target_trace = second_before
        .iter()
        .find(|trace| trace.capture_id == target_digest_id && trace.correct.is_none());
    assert!(
        second_target_trace.is_some(),
        "the second changing skill must have an unresolved trace keyed to the target digest, not an \
         older/latest/synthetic capture id; target_digest_id={target_digest_id} traces={second_before:?}"
    );
    let second_target_trace = second_target_trace.expect("asserted present");
    let branch_target_trace = branch_before
        .iter()
        .find(|trace| trace.capture_id == target_digest_id && trace.correct.is_none());
    assert!(
        branch_target_trace.is_some(),
        "the selected branch skill must have an unresolved trace keyed to the same target digest so \
         branch blame can be rejected; target_digest_id={target_digest_id} traces={branch_before:?}"
    );
    let branch_target_trace = branch_target_trace.expect("asserted present");
    let other_target_trace = other_before
        .iter()
        .find(|trace| trace.capture_id == target_digest_id && trace.correct.is_none());
    assert!(
        other_target_trace.is_some(),
        "the unrelated effectful skill must have an unresolved target trace keyed to the same \
         digest; target_digest_id={target_digest_id} traces={other_before:?}"
    );
    let other_target_trace = other_target_trace.expect("asserted present");
    let branch_before_by_digest = branch_before
        .iter()
        .map(|trace| {
            (
                trace.capture_id.clone(),
                (
                    trace.correct,
                    trace.effect_entity_key.clone(),
                    trace.effect_entity_name.clone(),
                    trace.effect_from_type.clone(),
                    trace.effect_to_type.clone(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        branch_before_by_digest,
        BTreeMap::from([
            (stale_digest_id.clone(), (None, None, None, None, None)),
            (target_digest_id.clone(), (None, None, None, None, None)),
            (later_digest_id.clone(), (None, None, None, None, None)),
        ]),
        "all selected branch traces must be unresolved and effect-empty before correction; \
         branch_before={branch_before:?}"
    );
    let first_before_by_digest = first_before
        .iter()
        .map(|trace| {
            (
                trace.capture_id.clone(),
                (
                    trace.correct,
                    trace.effect_entity_key.clone(),
                    trace.effect_entity_name.clone(),
                    trace.effect_from_type.clone(),
                    trace.effect_to_type.clone(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        first_before_by_digest,
        BTreeMap::from([
            (
                stale_digest_id.clone(),
                (
                    None,
                    Some(entity_key.clone()),
                    Some(entity_name.clone()),
                    Some(starting_type.to_string()),
                    Some(intermediate_type.to_string()),
                ),
            ),
            (
                target_digest_id.clone(),
                (
                    None,
                    Some(entity_key.clone()),
                    Some(entity_name.clone()),
                    Some(starting_type.to_string()),
                    Some(intermediate_type.to_string()),
                ),
            ),
            (
                later_digest_id.clone(),
                (
                    None,
                    Some(entity_key.clone()),
                    Some(entity_name.clone()),
                    Some(starting_type.to_string()),
                    Some(intermediate_type.to_string()),
                ),
            ),
        ]),
        "every first-skill trace, not only the target row, must persist capture-time effect \
         metadata; first_before={first_before:?}"
    );
    let second_before_by_digest = second_before
        .iter()
        .map(|trace| {
            (
                trace.capture_id.clone(),
                (
                    trace.correct,
                    trace.effect_entity_key.clone(),
                    trace.effect_entity_name.clone(),
                    trace.effect_from_type.clone(),
                    trace.effect_to_type.clone(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        second_before_by_digest,
        BTreeMap::from([
            (
                stale_digest_id.clone(),
                (
                    None,
                    Some(entity_key.clone()),
                    Some(entity_name.clone()),
                    Some(intermediate_type.to_string()),
                    Some(final_type.to_string()),
                ),
            ),
            (
                target_digest_id.clone(),
                (
                    None,
                    Some(entity_key.clone()),
                    Some(entity_name.clone()),
                    Some(intermediate_type.to_string()),
                    Some(final_type.to_string()),
                ),
            ),
            (
                later_digest_id.clone(),
                (
                    None,
                    Some(entity_key.clone()),
                    Some(entity_name.clone()),
                    Some(intermediate_type.to_string()),
                    Some(final_type.to_string()),
                ),
            ),
        ]),
        "every second-skill trace, not only the target row, must persist capture-time effect \
         metadata; second_before={second_before:?}"
    );
    let other_before_by_digest = other_before
        .iter()
        .map(|trace| {
            (
                trace.capture_id.clone(),
                (
                    trace.correct,
                    trace.effect_entity_key.clone(),
                    trace.effect_entity_name.clone(),
                    trace.effect_from_type.clone(),
                    trace.effect_to_type.clone(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        other_before_by_digest,
        BTreeMap::from([
            (
                stale_digest_id.clone(),
                (
                    None,
                    Some(other_entity_key.clone()),
                    Some(other_entity_name.clone()),
                    Some(other_starting_type.to_string()),
                    Some(other_final_type.to_string()),
                ),
            ),
            (
                target_digest_id.clone(),
                (
                    None,
                    Some(other_entity_key.clone()),
                    Some(other_entity_name.clone()),
                    Some(other_starting_type.to_string()),
                    Some(other_final_type.to_string()),
                ),
            ),
            (
                later_digest_id.clone(),
                (
                    None,
                    Some(other_entity_key.clone()),
                    Some(other_entity_name.clone()),
                    Some(other_starting_type.to_string()),
                    Some(other_final_type.to_string()),
                ),
            ),
        ]),
        "the unrelated effectful skill must carry real capture-time effect metadata on the same \
         digest, so correction cannot label every effectful target row; other_before={other_before:?}"
    );
    assert_eq!(
        (
            first_target_trace.effect_entity_key.as_deref(),
            first_target_trace.effect_entity_name.as_deref(),
            first_target_trace.effect_from_type.as_deref(),
            first_target_trace.effect_to_type.as_deref(),
        ),
        (
            Some(entity_key.as_str()),
            Some(entity_name.as_str()),
            Some(starting_type),
            Some(intermediate_type)
        ),
        "the first target trace must persist the capture-time effect it actually applied before \
         correction runs; replaying current skill rules later is not enough. trace={first_target_trace:?}"
    );
    assert_eq!(
        (
            second_target_trace.effect_entity_key.as_deref(),
            second_target_trace.effect_entity_name.as_deref(),
            second_target_trace.effect_from_type.as_deref(),
            second_target_trace.effect_to_type.as_deref(),
        ),
        (
            Some(entity_key.as_str()),
            Some(entity_name.as_str()),
            Some(intermediate_type),
            Some(final_type),
        ),
        "the second target trace must persist the capture-time effect it actually applied before \
         correction runs; replaying current skill rules later is not enough. trace={second_target_trace:?}"
    );
    assert_eq!(
        (
            branch_target_trace.effect_entity_key.as_deref(),
            branch_target_trace.effect_entity_name.as_deref(),
            branch_target_trace.effect_from_type.as_deref(),
            branch_target_trace.effect_to_type.as_deref(),
        ),
        (None, None, None, None),
        "the branch trace was selected but did not change the captured value, so persisted \
         capture-time effect evidence must stay empty. trace={branch_target_trace:?}"
    );
    assert!(
        first_before
            .iter()
            .any(|trace| trace.capture_id == stale_digest_id && trace.correct.is_none())
            && first_before
                .iter()
                .any(|trace| trace.capture_id == later_digest_id && trace.correct.is_none()),
        "the first changing skill needs stale and later unresolved rows so latest/oldest shortcuts \
         fail; traces={first_before:?}"
    );
    assert!(
        second_before
            .iter()
            .any(|trace| trace.capture_id == stale_digest_id && trace.correct.is_none())
            && second_before
                .iter()
                .any(|trace| trace.capture_id == later_digest_id && trace.correct.is_none()),
        "the second changing skill needs stale and later unresolved rows so latest/oldest shortcuts \
         fail; traces={second_before:?}"
    );

    drop(store);
    let reopened_store = open_store_with_lexical_embedder(StoreConfig {
        db_path,
        default_text_embedder: None,
        ..Default::default()
    });
    correct_chained_retype_after_reopen_with_rules_rewritten(
        &reopened_store,
        ChainedRetypeCorrection {
            target_digest_id: &target_digest_id,
            entity_name: &entity_name,
            starting_type,
            first_skill_id: &first_changing_skill_id,
            second_skill_id: &second_changing_skill_id,
            branch_skill_id: &unapplied_branch_skill_id,
            other_skill_id: &other_effectful_skill_id,
        },
    );
    let first_after = firing_traces_for_skill(&reopened_store, &first_changing_skill_id);
    let second_after = firing_traces_for_skill(&reopened_store, &second_changing_skill_id);
    let branch_after = firing_traces_for_skill(&reopened_store, &unapplied_branch_skill_id);
    let other_after = firing_traces_for_skill(&reopened_store, &other_effectful_skill_id);
    let row_state = |trace: &SkillFiringRow| {
        (
            trace.capture_id.clone(),
            trace.correct,
            trace.created_at,
            trace.effect_entity_key.clone(),
            trace.effect_entity_name.clone(),
            trace.effect_from_type.clone(),
            trace.effect_to_type.clone(),
        )
    };
    let first_after_by_id = first_after
        .iter()
        .map(|trace| (trace.id.clone(), row_state(trace)))
        .collect::<BTreeMap<_, _>>();
    let second_after_by_id = second_after
        .iter()
        .map(|trace| (trace.id.clone(), row_state(trace)))
        .collect::<BTreeMap<_, _>>();
    let branch_after_by_id = branch_after
        .iter()
        .map(|trace| (trace.id.clone(), row_state(trace)))
        .collect::<BTreeMap<_, _>>();
    let other_after_by_id = other_after
        .iter()
        .map(|trace| (trace.id.clone(), row_state(trace)))
        .collect::<BTreeMap<_, _>>();
    let mut expected_first_after = first_before
        .iter()
        .map(|trace| (trace.id.clone(), row_state(trace)))
        .collect::<BTreeMap<_, _>>();
    expected_first_after.insert(
        first_target_trace.id.clone(),
        (
            target_digest_id.clone(),
            Some(false),
            first_target_trace.created_at,
            first_target_trace.effect_entity_key.clone(),
            first_target_trace.effect_entity_name.clone(),
            first_target_trace.effect_from_type.clone(),
            first_target_trace.effect_to_type.clone(),
        ),
    );
    let mut expected_second_after = second_before
        .iter()
        .map(|trace| (trace.id.clone(), row_state(trace)))
        .collect::<BTreeMap<_, _>>();
    expected_second_after.insert(
        second_target_trace.id.clone(),
        (
            target_digest_id.clone(),
            Some(false),
            second_target_trace.created_at,
            second_target_trace.effect_entity_key.clone(),
            second_target_trace.effect_entity_name.clone(),
            second_target_trace.effect_from_type.clone(),
            second_target_trace.effect_to_type.clone(),
        ),
    );
    let expected_branch_after = branch_before
        .iter()
        .map(|trace| (trace.id.clone(), row_state(trace)))
        .collect::<BTreeMap<_, _>>();
    let expected_other_after = other_before
        .iter()
        .map(|trace| (trace.id.clone(), row_state(trace)))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        first_after_by_id, expected_first_after,
        "the File -> Module firing changed the corrected entity away from the owner-confirmed File \
         value, so only the original target trace row must be labeled wrong; traces={first_after:?}"
    );
    assert_eq!(
        second_after_by_id, expected_second_after,
        "the Module -> Service firing also changed the same corrected entity before the final stored \
         type was reached, so final-type inference is not enough; traces={second_after:?}"
    );
    assert_eq!(
        branch_after_by_id, expected_branch_after,
        "the File -> Document branch was selected on the same digest but did not change the value \
         after File -> Module ran first, so digest-wide or inferred-path blame must leave it \
         unresolved; target_branch_trace={branch_target_trace:?} traces={branch_after:?}"
    );
    assert_eq!(
        other_after_by_id, expected_other_after,
        "the unrelated effectful target row must stay unresolved after correcting {entity_name}; \
         effectful digest-wide blame would label it wrong. target_other_trace={other_target_trace:?} \
         traces={other_after:?}"
    );
    let alert_ids = list_drift_alerts(&reopened_store)
        .into_iter()
        .map(|alert| alert.skill_id)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        alert_ids,
        std::collections::BTreeSet::from([
            first_changing_skill_id.clone(),
            second_changing_skill_id.clone()
        ]),
        "drift review must surface exactly the two changing skills and no branch or synthetic \
         attribution; alerts={alert_ids:?}"
    );
}
