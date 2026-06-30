// End-to-end learning-loop tests for the reopened cg-llm-client-refactor correction.
//
// These tests deliberately do not seed cg_learned_skills and do not call the pure promotion
// helpers as proof. They drive the product path: real capture -> correction overlay +
// teaching event -> persisted learned skill -> later run_capture auto-application -> drift
// from the capture-produced firing trace.

use std::collections::BTreeMap;

use context_graph::harness::{CaptureHarnessConfig, run_capture};
use context_graph::ingest::{CaptureReport, CaptureRequest};
use context_graph::skill_store::{
    LearnedSkill, SkillProvenance, apply_correction, list_drift_alerts, list_skills_in_store,
    list_teaching_events_in_store, persist_skill, record_firing_label,
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
            Some(SkillFiringRow {
                id,
                capture_id,
                correct,
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
                rule: "observe context-graph without changing any entity type".to_string(),
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
