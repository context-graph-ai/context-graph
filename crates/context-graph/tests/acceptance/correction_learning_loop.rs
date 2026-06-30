// End-to-end learning-loop tests for the reopened cg-llm-client-refactor correction.
//
// These tests deliberately do not seed cg_learned_skills and do not call the pure promotion
// helpers as proof. They drive the product path: real capture -> correction overlay +
// teaching event -> persisted learned skill -> later run_capture auto-application -> drift
// from the capture-produced firing trace.

use context_graph::harness::{CaptureHarnessConfig, run_capture};
use context_graph::ingest::CaptureRequest;
use context_graph::skill_store::{
    LearnedSkill, SkillProvenance, apply_correction, list_drift_alerts, list_skills_in_store,
    list_teaching_events_in_store,
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

fn firing_trace_ids_for_skill(store: &Store, skill_id: &str) -> Vec<String> {
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
            match row.values.get("id") {
                Some(DbValue::Text(id)) => Some(id.clone()),
                Some(DbValue::Uuid(id)) => Some(id.to_string()),
                _ => None,
            }
        })
        .collect()
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
    let (_dir, store) = fresh_store();

    let skills = teach_repository_skill_from_real_corrections(&store);
    let skill = learned_repository_retype_skill(&skills).unwrap_or_else(|| {
        panic!(
            "drift must start from a skill learned by repeated real corrections, not a seeded \
             cg_learned_skills row; skills={skills:?}"
        )
    });
    let learned_skill_id = skill.id.clone();

    let (_report, harness) = run_capture(
        &store,
        CaptureRequest {
            context_label: "drift-fire-capture".to_string(),
            transcript: repository_transcript(),
            model_available: true,
        },
        &super::refactor_extractor::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("skill-fire capture must succeed");
    assert!(
        harness
            .applied_skill_ids
            .iter()
            .any(|id| id == &learned_skill_id),
        "the learned skill must fire on the capture before drift can be labeled; applied_skill_ids={:?}",
        harness.applied_skill_ids
    );
    let trace_ids = firing_trace_ids_for_skill(&store, &learned_skill_id);
    assert!(
        !trace_ids.is_empty(),
        "drift labeling must have a capture-produced firing trace to blame; no direct \
         record_firing_label call is allowed in this test"
    );

    let corrected_entity = entity_named_in_context(&store, "drift-fire-capture", "context-graph");
    assert_eq!(
        corrected_entity.entity_type,
        EntityType::Repository,
        "the drift correction must target a value that the learned skill changed on the capture path"
    );
    correct_entity_type(&store, &corrected_entity, "File", Some(&learned_skill_id));

    let alerts = list_drift_alerts(&store);
    assert!(
        alerts
            .iter()
            .any(|alert| alert.skill_id == learned_skill_id),
        "a correction after a capture-produced skill firing must label that firing wrong and make \
         list_drift_alerts surface the learned skill; direct record_firing_label-only drift tests \
         do not prove this path. trace_ids={trace_ids:?} alerts={alerts:?}"
    );
}
