use std::sync::Mutex;

use context_graph::harness::{CaptureHarnessConfig, run_capture};
use context_graph::ingest::{
    CaptureRequest, ConversationExtractor, ExtractionRequest, RollingCaptureRequest,
    StructuredHarvest, capture_rolling_conversation_with_extractor,
    record_conversation_with_extractor,
};
use context_graph::overlay;
use context_graph::skill_store::{
    LearnedSkill, SkillProvenance, apply_correction_with_llm_client, list_drift_alerts,
    list_skills_in_store, record_firing_label,
};
use context_graph::{
    Entity, EntityType, ListEntityFilter, LlmClient, ModelInvocation, ModelReply, Store,
    StoreConfig,
};
use contextdb_core::Value as DbValue;

use super::fake_extractor::open_store_with_lexical_embedder;

fn fresh_store() -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = open_store_with_lexical_embedder(StoreConfig {
        db_path: dir.path().join("correction_promotion_production.db"),
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

fn run_repository_capture_with_skills(
    store: &Store,
    context_label: &str,
    skills_enabled: bool,
) -> Entity {
    let (_report, _harness) = run_capture(
        store,
        CaptureRequest {
            context_label: context_label.to_string(),
            transcript: repository_transcript(),
            model_available: true,
        },
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig {
            skills_enabled,
            ..CaptureHarnessConfig::default()
        },
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

#[derive(Debug, Default)]
struct SkillProposalClient {
    observed_inputs: Mutex<Vec<String>>,
}

impl SkillProposalClient {
    fn call_count(&self) -> usize {
        self.observed_inputs
            .lock()
            .expect("observed inputs lock")
            .len()
    }
}

impl LlmClient for SkillProposalClient {
    async fn understand(&self, invocation: &ModelInvocation) -> context_graph::Result<ModelReply> {
        self.understand_native(invocation)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn understand_native(&self, invocation: &ModelInvocation) -> context_graph::Result<ModelReply> {
        self.observed_inputs
            .lock()
            .expect("observed inputs lock")
            .push(
                invocation
                    .request
                    .input_without_output_contract()
                    .to_string(),
            );
        Ok(ModelReply {
            marker: "skill-proposal:context-graph-retype".to_string(),
            phrasing: serde_json::json!({
                "description": "retype context-graph from File to Repository when future captures mention this entity",
                "rule": "when an entity named context-graph is typed File -> Repository"
            })
            .to_string(),
            advisory_flag: false,
            malformed: false,
            usage: None,
        })
    }
}

#[derive(Debug, Default)]
struct GenericSummaryRollingExtractor;

impl ConversationExtractor for GenericSummaryRollingExtractor {
    fn extract(&self, request: ExtractionRequest<'_>) -> context_graph::Result<StructuredHarvest> {
        let mut harvest = super::action_test_extractors::span_faithful_harvest(request.transcript);
        if !harvest.goals.is_empty()
            || !harvest.entities.is_empty()
            || !harvest.decisions.is_empty()
            || !harvest.observations.is_empty()
        {
            harvest.summary = Some("generic rolling import summary".to_string());
        }
        Ok(harvest)
    }
}

fn correct_repository_with_model(
    store: &Store,
    entity: &Entity,
    client: &SkillProposalClient,
) -> context_graph::skill_store::CorrectionResult {
    let item_id = entity.id.to_string();
    let overlay = overlay::record_correction(store, &item_id, "Repository");
    assert_eq!(
        overlay.item_id, item_id,
        "the correction overlay must target the captured entity"
    );
    apply_correction_with_llm_client(
        store,
        &item_id,
        "Repository",
        None,
        entity.context_id,
        client,
    )
    .expect("read succeeds")
}

fn teach_repository_skill_from_model_path(
    store: &Store,
    client: &SkillProposalClient,
    context_prefix: &str,
    skills_enabled_during_training: bool,
) -> context_graph::skill_store::CorrectionResult {
    let mut last = None;
    for index in 0..3 {
        let context_label = format!("{context_prefix}-{index}");
        let entity = run_repository_capture_with_skills(
            store,
            &context_label,
            skills_enabled_during_training,
        );
        assert_eq!(
            entity.entity_type,
            EntityType::File,
            "the training fixture must first capture the repository as File so the correction is real"
        );
        let correction = correct_repository_with_model(store, &entity, client);
        assert!(
            correction.applied,
            "the correction entrypoint must accept the real captured entity id"
        );
        assert!(
            correction.teaching_event.is_some(),
            "the correction entrypoint must log a teaching event for learning"
        );
        last = Some(correction);
    }
    last.expect("three corrections must produce a final correction result")
}

fn learned_llm_repository_retype_skill(skills: &[LearnedSkill]) -> Option<&LearnedSkill> {
    skills.iter().find(|skill| {
        let text = format!("{} {}", skill.description, skill.rule).to_ascii_lowercase();
        skill.active
            && skill.provenance == SkillProvenance::LlmAuthored
            && text.contains("context-graph")
            && text.contains("file")
            && text.contains("repository")
    })
}

fn firing_trace_count_for_skill(store: &Store, skill_id: &str) -> usize {
    firing_trace_ids_for_skill(store, skill_id).len()
}

fn firing_trace_ids_for_skill(store: &Store, skill_id: &str) -> Vec<String> {
    let db = store.sync_database();
    db.scan("cg_skill_firing_trace", db.snapshot())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| {
            matches!(
                row.values.get("skill_id"),
                Some(DbValue::Text(value)) if value == skill_id
            )
            .then(|| match row.values.get("id") {
                Some(DbValue::Text(value)) => value.clone(),
                _ => String::new(),
            })
        })
        .filter(|id| !id.is_empty())
        .collect()
}

#[test]
fn real_corrections_promote_llm_authored_skill_through_heldout_gate() {
    let (_dir, store) = fresh_store();
    let client = SkillProposalClient::default();

    let correction =
        teach_repository_skill_from_model_path(&store, &client, "heldout-promotion", true);
    assert!(
        correction.new_skill_created,
        "the third real correction should create the learned skill"
    );
    assert!(
        correction.heldout_gate_invocations > 0,
        "production promotion must run the held-out gate, not promote on a raw correction count"
    );
    assert!(
        client.call_count() > 0,
        "production promotion must ask the LLM client to author the skill proposal"
    );

    let skills = list_skills_in_store(&store).expect("read succeeds");
    assert!(
        learned_llm_repository_retype_skill(&skills).is_some(),
        "the promoted rule must be LLM-authored and come from the correction path, not a seeded row"
    );
}

#[test]
fn production_promotion_runs_heldout_gate_and_rejects_tied_candidate() {
    let (_dir, store) = fresh_store();
    let client = SkillProposalClient::default();
    let first = teach_repository_skill_from_model_path(&store, &client, "first-skill", true);
    assert!(
        first.new_skill_created,
        "first promotion should create a skill"
    );
    let skill_count_after_first = list_skills_in_store(&store).expect("read succeeds").len();

    let tie = teach_repository_skill_from_model_path(&store, &client, "tied-candidate", false);
    assert!(
        tie.heldout_gate_invocations > 0,
        "the duplicate candidate must still be judged by the held-out gate"
    );
    assert!(
        !tie.new_skill_created,
        "a candidate that only ties the incumbent held-out behavior must not promote"
    );
    assert_eq!(
        list_skills_in_store(&store).expect("read succeeds").len(),
        skill_count_after_first,
        "held-out tie rejection must not persist a second learned skill"
    );
}

#[test]
fn record_conversation_with_extractor_applies_earned_skill_and_records_firing() {
    let (_dir, store) = fresh_store();
    let client = SkillProposalClient::default();
    let correction = teach_repository_skill_from_model_path(&store, &client, "library-path", true);
    assert!(correction.new_skill_created, "training must earn a skill");
    let skills = list_skills_in_store(&store).expect("read succeeds");
    let skill = learned_llm_repository_retype_skill(&skills).expect("learned LLM retype skill");

    record_conversation_with_extractor(
        &store,
        CaptureRequest {
            context_label: "library-import".to_string(),
            transcript: repository_transcript(),
            model_available: true,
        },
        &super::action_test_extractors::SpanFaithfulExtractor,
    )
    .expect("library import capture must succeed");

    let entity = entity_named_in_context(&store, "library-import", "context-graph");
    assert_eq!(
        entity.entity_type,
        EntityType::Repository,
        "record_conversation_with_extractor must run the learned-skill application path"
    );
    assert!(
        firing_trace_count_for_skill(&store, &skill.id) > 0,
        "record_conversation_with_extractor must persist a firing trace for the applied skill"
    );
}

#[test]
fn rolling_public_capture_applies_earned_skill_using_source_transcript_not_digest_summary() {
    let (_dir, store) = fresh_store();
    let client = SkillProposalClient::default();
    let correction = teach_repository_skill_from_model_path(&store, &client, "rolling-path", true);
    assert!(correction.new_skill_created, "training must earn a skill");
    let skills = list_skills_in_store(&store).expect("read succeeds");
    let skill = learned_llm_repository_retype_skill(&skills).expect("learned LLM retype skill");

    let first_chunk = repository_transcript();
    let transcript = format!("{first_chunk}\n{}", "generic filler ".repeat(120));
    let report = capture_rolling_conversation_with_extractor(
        &store,
        RollingCaptureRequest {
            context_label: "rolling-library-import".to_string(),
            transcript,
            chunk_window: first_chunk.len() + 1,
            running_state_cap_tokens: 128,
            model_context_tokens: Some(200_000),
        },
        &GenericSummaryRollingExtractor,
    )
    .expect("rolling public capture must succeed");

    assert!(
        report.import_chunks.unwrap_or(0) > 1,
        "the regression must exercise the public multi-chunk rolling path"
    );
    assert!(
        report.digest_summary.starts_with("rolling import: "),
        "the persisted digest remains the condensed rolling summary, not the full source transcript: {}",
        report.digest_summary
    );
    let entity = entity_named_in_context(&store, "rolling-library-import", "context-graph");
    assert_eq!(
        entity.entity_type,
        EntityType::Repository,
        "rolling capture must select learned skills against the original source transcript"
    );
    assert!(
        firing_trace_count_for_skill(&store, &skill.id) > 0,
        "rolling capture must persist a firing trace for the applied skill"
    );
}

#[test]
fn llm_authored_skill_stays_in_learned_scope_and_drift_alert_does_not_disable() {
    let (_dir, store) = fresh_store();
    let client = SkillProposalClient::default();
    let correction = teach_repository_skill_from_model_path(&store, &client, "guard-path", true);
    assert!(correction.new_skill_created, "training must earn a skill");
    let skills = list_skills_in_store(&store).expect("read succeeds");
    let skill = learned_llm_repository_retype_skill(&skills)
        .expect("learned LLM retype skill")
        .clone();

    let (_report, out_of_scope) = run_capture(
        &store,
        CaptureRequest {
            context_label: "out-of-scope-capture".to_string(),
            transcript: [
                "Goal G_scope: keep the auth-service file classified",
                "Entity E_scope: auth-service type=file",
                "Decision D_scope: treat auth-service as the deployment file serves G_scope",
            ]
            .join("\n"),
            model_available: true,
        },
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("out-of-scope capture must succeed");
    assert!(
        !out_of_scope
            .applied_skill_ids
            .iter()
            .any(|id| id == &skill.id),
        "an LLM-authored context-graph skill must not fire on an unrelated auth-service capture"
    );
    let unrelated = entity_named_in_context(&store, "out-of-scope-capture", "auth-service");
    assert_eq!(
        unrelated.entity_type,
        EntityType::File,
        "out-of-scope capture must keep the original type; the learned skill must not widen itself"
    );

    let before_matching = firing_trace_ids_for_skill(&store, &skill.id).len();
    let (_report, in_scope) = run_capture(
        &store,
        CaptureRequest {
            context_label: "in-scope-drift-capture".to_string(),
            transcript: repository_transcript(),
            model_available: true,
        },
        &super::action_test_extractors::SpanFaithfulExtractor,
        &CaptureHarnessConfig::default(),
    )
    .expect("in-scope capture must succeed");
    assert!(
        in_scope.applied_skill_ids.iter().any(|id| id == &skill.id),
        "the earned LLM-authored skill must still apply on the matching learned scope"
    );

    let trace_ids = firing_trace_ids_for_skill(&store, &skill.id);
    let new_trace = trace_ids
        .get(before_matching)
        .expect("matching capture must persist a new firing trace");
    record_firing_label(&store, new_trace, false);

    let alerts = list_drift_alerts(&store).expect("read succeeds");
    let alert = alerts
        .iter()
        .find(|alert| alert.skill_id == skill.id)
        .expect("wrong-labeled LLM-authored firing must surface a drift alert");
    assert_eq!(
        alert.provenance,
        SkillProvenance::LlmAuthored,
        "drift alert must preserve LLM-authored provenance"
    );
    assert!(
        alert.active,
        "drift is informational; it must not disable the LLM-authored skill"
    );
    let skill_after_alert = list_skills_in_store(&store)
        .expect("read succeeds")
        .into_iter()
        .find(|stored| stored.id == skill.id)
        .expect("skill row must remain present after drift alert");
    assert!(
        skill_after_alert.active,
        "drift alert must not auto-disable the LLM-authored skill"
    );
}
