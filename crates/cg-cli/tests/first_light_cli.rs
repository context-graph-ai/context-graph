mod support;

use std::process::{Command, Stdio};

use context_graph::{
    ContextId, CreateDecision, CreateIntention, DecisionId, DecisionStatus, DiscoveryFilter,
    EntityId, IntentionId, IntentionOrigin, IntentionStatus, Store, StoreConfig,
};
use contextdb_core::Value as DbValue;
use contextdb_engine::Database;
use serde_json::Value;
use support::{CliHarness, CommandOutput, CompletionBehavior, LoginState, Provider, cg_bin};

const AUTH_INITIAL: &str = "Acme needs password-less login to ship. We found the auth module file at src/auth.py and decided the auth module stays at src/auth.py.";
const AUTH_WITH_JWT: &str = "Acme needs password-less login to ship. We found the auth module file at src/auth.py and decided the auth module stays at src/auth.py. Use JWT with 15-min access tokens.";
const AUTH_RENAME: &str = "Following Tuesday, the auth module file moved from src/auth.py to src/authentication.py. Keep password-less login served by the auth module at src/authentication.py.";
const AUTH_CURRENT_WITH_JWT: &str = "Acme password-less login is now served by the auth module at src/authentication.py. Use JWT with 15-min access tokens.";
const CI_PIPELINE_SESSION: &str = "The delivery team needs a CI pipeline. Run CI on every pull request, block merges when CI fails, and include unit tests, clippy, and release build in .github/workflows/ci.yml.";
const PROJECT_X_RECORD: &str = "Project X memory must stay local, so use the local context store.";
const CLIENT_A_SOX_SESSION: &str = "Client A must keep SOX deploy approvals. The publish pipeline is deploy.yml; require SOX approval before promote.";
const CLIENT_B_PUBLISH_SESSION: &str = "Client B must avoid the duplicate-publish race. The publish pipeline is deploy.yml; hold 10 min before promote.";

fn combined(output: &CommandOutput) -> String {
    format!("{}\n{}", output.stdout, output.stderr)
}

fn assert_record_contains(output: &CommandOutput, expected: &str) {
    let text = combined(output);
    assert_eq!(output.code, 0, "{text}");
    assert!(text.contains(expected), "{text}");
    let json = output_json(output).expect("record output json");
    assert_eq!(
        json.pointer("/captured/goals").and_then(Value::as_u64),
        Some(1),
        "{text}"
    );
    assert_eq!(
        json.pointer("/captured/decisions").and_then(Value::as_u64),
        Some(1),
        "{text}"
    );
}

fn assert_record_text(output: &CommandOutput, expected: &str) {
    let text = combined(output);
    assert_eq!(output.code, 0, "{text}");
    assert!(text.contains(expected), "{text}");
}

fn extraction_harness() -> CliHarness {
    let harness = CliHarness::new();
    harness.providers.install(
        Provider::Claude,
        "developer@example.test",
        LoginState::Valid,
        CompletionBehavior::Succeed,
    );
    let login = harness.run(&[
        "model",
        "auth",
        "login",
        "--connection",
        "claude",
        "--consent",
    ]);
    assert_eq!(
        login.code,
        0,
        "fake host login failed: {}",
        combined(&login)
    );
    let selftest = harness.run(&["selftest"]);
    assert_eq!(
        selftest.code,
        0,
        "fake host selftest failed: {}",
        combined(&selftest)
    );
    harness
        .providers
        .set_behavior(Provider::Claude, CompletionBehavior::ExtractHarvest);
    harness
}

fn run_in_dir(harness: &CliHarness, cwd: &std::path::Path, args: &[&str]) -> CommandOutput {
    let output = Command::new(cg_bin())
        .args(args)
        .current_dir(cwd)
        .env("CG_DATA_ROOT", harness.data_root.path())
        .env("CG_USE_TEST_LEXICAL_EMBEDDER", "1")
        .env("HOME", harness.data_root.path())
        .env("PATH", harness.providers.path_env())
        .env_remove("OPENAI_API_KEY")
        .env_remove("CODEX_API_KEY")
        .env_remove("ANTHROPIC_API_KEY")
        .stdin(Stdio::null())
        .output()
        .expect("run cg");
    CommandOutput {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn output_json(output: &CommandOutput) -> Option<Value> {
    serde_json::from_str(&output.stdout).ok()
}

fn stable_id(output: &CommandOutput, keys: &[&str]) -> Option<String> {
    let json = output_json(output)?;
    let stable_ids = json.get("stable_ids").and_then(Value::as_object)?;
    keys.iter().find_map(|key| {
        stable_ids
            .get(*key)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    })
}

fn json_string_at<'a>(json: &'a Value, paths: &[&str]) -> Option<&'a str> {
    paths
        .iter()
        .find_map(|path| json.pointer(path).and_then(Value::as_str))
}

fn store_path_from(outputs: &[&CommandOutput]) -> Option<std::path::PathBuf> {
    outputs.iter().find_map(|output| {
        let json = output_json(output)?;
        json_string_at(
            &json,
            &[
                "/store/db_path",
                "/store_path",
                "/db_path",
                "/context_graph_db",
            ],
        )
        .map(std::path::PathBuf::from)
    })
}

fn context_id_from(outputs: &[&CommandOutput]) -> Option<ContextId> {
    outputs
        .iter()
        .find_map(|output| {
            let json = output_json(output)?;
            json_string_at(
                &json,
                &["/context/id", "/context_id", "/stable_ids/context"],
            )
            .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        })
        .map(ContextId::from)
}

fn memory_counts(output: &CommandOutput) -> Option<(u64, u64, u64)> {
    if let Some(json) = output_json(output) {
        let goals = json
            .pointer("/totals/goals")
            .or_else(|| json.pointer("/goals"))
            .and_then(Value::as_u64);
        let decisions = json
            .pointer("/totals/decisions")
            .or_else(|| json.pointer("/decisions"))
            .and_then(Value::as_u64);
        let observations = json
            .pointer("/totals/observations")
            .or_else(|| json.pointer("/observations"))
            .and_then(Value::as_u64);
        if let (Some(goals), Some(decisions), Some(observations)) = (goals, decisions, observations)
        {
            return Some((goals, decisions, observations));
        }
    }

    let text = combined(output);
    Some((
        count_after_label(&text, "goals:")?,
        count_after_label(&text, "decisions:")?,
        count_after_label(&text, "observations:")?,
    ))
}

fn count_after_label(text: &str, label: &str) -> Option<u64> {
    let tail = text.split_once(label)?.1;
    tail.split_whitespace()
        .next()
        .and_then(|value| value.trim_matches(',').parse().ok())
}

fn stable_uuid_from(outputs: &[&CommandOutput], keys: &[&str]) -> Option<uuid::Uuid> {
    outputs
        .iter()
        .find_map(|output| stable_id(output, keys))
        .and_then(|raw| uuid::Uuid::parse_str(&raw).ok())
}

fn persisted_store(path: Option<std::path::PathBuf>) -> Option<Store> {
    path.and_then(|db_path| {
        Store::open(StoreConfig {
            db_path,
            default_text_embedder: None,
            ..Default::default()
        })
        .ok()
    })
}

fn persisted_decisions_for_intention(
    store: Option<&Store>,
    intention_id: Option<IntentionId>,
) -> Vec<(DecisionId, String, DecisionStatus)> {
    match (store, intention_id) {
        (Some(store), Some(intention_id)) => store
            .by_intention(intention_id, DiscoveryFilter::default())
            .unwrap_or_default()
            .into_iter()
            .map(|decision| (decision.id, decision.description, decision.status))
            .collect(),
        _ => Vec::new(),
    }
}

fn persisted_decisions_for_entity(
    store: Option<&Store>,
    entity_id: Option<EntityId>,
) -> Vec<(DecisionId, String, DecisionStatus)> {
    match (store, entity_id) {
        (Some(store), Some(entity_id)) => store
            .by_entity_decisions(
                entity_id,
                DiscoveryFilter {
                    include_decision_statuses: Some(vec![
                        DecisionStatus::Active,
                        DecisionStatus::Invalidated,
                        DecisionStatus::Superseded,
                    ]),
                    ..Default::default()
                },
            )
            .unwrap_or_default()
            .into_iter()
            .map(|decision| (decision.id, decision.description, decision.status))
            .collect(),
        _ => Vec::new(),
    }
}

fn persisted_invalidated_decisions(
    store: Option<&Store>,
    context_id: Option<ContextId>,
    decision_id: Option<DecisionId>,
) -> Vec<(DecisionId, Vec<IntentionId>)> {
    match (store, context_id, decision_id) {
        (Some(store), Some(context_id), Some(decision_id)) => store
            .list_invalidations(context_id, Some(decision_id), None)
            .unwrap_or_default()
            .into_iter()
            .map(|row| (row.affected_decision_id, row.threatened_intention_ids))
            .collect(),
        _ => Vec::new(),
    }
}

fn persisted_invalidation_decision_ids(
    store: Option<&Store>,
    context_id: Option<ContextId>,
) -> Vec<DecisionId> {
    let mut ids = match (store, context_id) {
        (Some(store), Some(context_id)) => store
            .list_invalidations(context_id, None, None)
            .unwrap_or_default()
            .into_iter()
            .map(|row| row.affected_decision_id)
            .collect(),
        _ => Vec::new(),
    };
    sort_decision_ids(&mut ids);
    ids
}

fn persisted_context_names(path: Option<&std::path::Path>) -> Vec<String> {
    let Some(path) = path else {
        return Vec::new();
    };
    let Ok(db) = Database::open(path) else {
        return Vec::new();
    };
    let mut names = db
        .scan("contexts", db.snapshot())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| match row.values.get("name") {
            Some(DbValue::Text(name)) => Some(name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn invalidation_decision_ids_from_output(
    output: &CommandOutput,
    array_keys: &[&str],
) -> Vec<DecisionId> {
    let Some(json) = output_json(output) else {
        return Vec::new();
    };
    let Some(rows) = array_keys
        .iter()
        .find_map(|key| json.get(*key).and_then(Value::as_array))
    else {
        return Vec::new();
    };
    let mut ids = rows
        .iter()
        .filter_map(|row| {
            row.get("affected_decision_id")
                .or_else(|| row.get("decision_id"))
                .and_then(Value::as_str)
                .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
                .map(DecisionId::from)
        })
        .collect::<Vec<_>>();
    sort_decision_ids(&mut ids);
    ids
}

fn sort_decision_ids(ids: &mut [DecisionId]) {
    ids.sort_by_key(|id| id.as_uuid());
}

fn seed_decision_text_only_decoy(
    store: Option<&Store>,
    context_id: Option<ContextId>,
) -> Option<DecisionId> {
    let (Some(store), Some(context_id)) = (store, context_id) else {
        return None;
    };
    let decoy_goal = store
        .create_intention(CreateIntention {
            description: "keep decision-text decoy out of digest query".to_string(),
            status: IntentionStatus::Active,
            origin: IntentionOrigin::Human,
            context_id,
            ..Default::default()
        })
        .ok()?;
    store
        .create_decision(CreateDecision {
            decision_type: "code_reference".to_string(),
            description: "auth module at src/authentication.py decision-text-only decoy"
                .to_string(),
            confidence: Some(0.99),
            intention_ids: vec![decoy_goal.id],
            context_id,
            ..Default::default()
        })
        .ok()
        .map(|decision| decision.id)
}

fn digest_ids_extracted_from_decision(
    path: Option<&std::path::Path>,
    decision_id: Option<DecisionId>,
) -> Vec<String> {
    let (Some(path), Some(decision_id)) = (path, decision_id) else {
        return Vec::new();
    };
    let Ok(db) = Database::open(path) else {
        return Vec::new();
    };
    let mut ids = db
        .scan("edges", db.snapshot())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| {
            let source_matches = matches!(
                row.values.get("source_id"),
                Some(DbValue::Uuid(source_id)) if *source_id == decision_id.as_uuid()
            );
            let relationship_matches = matches!(
                row.values.get("edge_type"),
                Some(DbValue::Text(edge_type)) if edge_type == "EXTRACTED_FROM"
            );
            match (
                source_matches,
                relationship_matches,
                row.values.get("target_id"),
            ) {
                (true, true, Some(DbValue::Uuid(target_id))) => Some(target_id.to_string()),
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn sorted_present_decision_ids(ids: &[Option<DecisionId>]) -> Vec<DecisionId> {
    let mut present = ids.iter().flatten().copied().collect::<Vec<_>>();
    sort_decision_ids(&mut present);
    present
}

fn no_resolution(output: &CommandOutput) -> bool {
    let text = combined(output);
    if text.contains("not found") || text.contains("no matching") || text.contains("nothing") {
        return true;
    }
    output_json(output)
        .and_then(|json| {
            json.get("decisions")
                .or_else(|| json.get("history"))
                .and_then(Value::as_array)
                .map(Vec::is_empty)
        })
        .unwrap_or(false)
}

#[test]
fn cli_record_observation_emits_invalidations() {
    let harness = extraction_harness();
    let first = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let output = harness.run_with_stdin(&["record"], AUTH_RENAME);
    let store_path = store_path_from(&[&first, &output]);
    let store = persisted_store(store_path.clone());
    let context_id = context_id_from(&[&first, &output]);
    let old_decision =
        stable_uuid_from(&[&first, &output], &["D_old", "auth module at src/auth.py"])
            .map(DecisionId::from);
    let goal = stable_uuid_from(
        &[&first, &output],
        &["G1", "ship password-less login for acme"],
    )
    .map(IntentionId::from);

    assert_eq!(
        (
            output.code,
            store_path.is_some(),
            context_id.is_some(),
            old_decision.is_some(),
            persisted_invalidated_decisions(store.as_ref(), context_id, old_decision),
        ),
        (
            0,
            true,
            true,
            true,
            old_decision
                .zip(goal)
                .map(|(old, goal)| vec![(old, vec![goal])])
                .unwrap_or_default(),
        )
    );
    assert!(combined(&output).contains("src/authentication.py"));
    assert!(combined(&output).contains("ship password-less login"));
}

#[test]
fn cli_query_via_digest() {
    let harness = extraction_harness();
    let first = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let second = harness.run_with_stdin(&["record"], AUTH_RENAME);
    let store_path = store_path_from(&[&first, &second]);
    let store = persisted_store(store_path.clone());
    let context_id = context_id_from(&[&first, &second]);
    let decoy = seed_decision_text_only_decoy(store.as_ref(), context_id);
    drop(store);
    let output = harness.run(&["query", "where is the auth module"]);
    let text = combined(&output);
    let store = persisted_store(store_path.clone());
    let goal = stable_uuid_from(
        &[&first, &second, &output],
        &["G1", "ship password-less login for acme"],
    )
    .map(IntentionId::from);
    let current = stable_uuid_from(
        &[&first, &second, &output],
        &["D_new", "auth module at src/authentication.py"],
    )
    .map(DecisionId::from);
    let persisted = persisted_decisions_for_intention(store.as_ref(), goal);
    drop(store);
    let provenance_digest_ids = digest_ids_extracted_from_decision(store_path.as_deref(), current);

    assert_eq!(
        (
            output.code,
            store_path.is_some(),
            decoy.is_some(),
            goal.is_some(),
            current.is_some(),
            !provenance_digest_ids.is_empty(),
            provenance_digest_ids
                .iter()
                .any(|digest_id| text.contains(digest_id)),
            text.contains("EXTRACTED_FROM"),
            !text.contains("decision-text-only decoy"),
            persisted
                .iter()
                .map(|(id, description, status)| (*id, description.as_str(), *status))
                .collect::<Vec<_>>(),
        ),
        (
            0,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            current
                .map(|id| {
                    vec![(
                        id,
                        "auth module at src/authentication.py",
                        DecisionStatus::Active,
                    )]
                })
                .unwrap_or_default(),
        )
    );
    assert!(text.contains("auth module at src/authentication.py"));
    assert!(text.contains("serves goal: ship password-less login for acme"));
    assert!(!text.contains("prior: auth module at src/auth.py [invalidated"));
    assert!(!text.contains("use JWT with 15-min access tokens [superseded]"));
}

#[test]
fn cli_status() {
    let harness = extraction_harness();
    let empty = harness.run(&["status"]);
    harness.run_with_stdin(&["record"], AUTH_WITH_JWT);
    let first = harness.run(&["status"]);
    harness.run_with_stdin(&["record"], CI_PIPELINE_SESSION);
    let second = harness.run(&["status"]);

    assert!(combined(&empty).contains("goals: 0"));
    assert!(combined(&first).contains("goals: 1"));
    assert!(combined(&first).contains("decisions: 2"));
    assert!(combined(&first).contains("observations: 1"));
    assert!(combined(&second).contains("goals: 2"));
    assert!(combined(&second).contains("decisions: 4"));
    assert!(combined(&second).contains("observations: 5"));
}

#[test]
fn cli_status_idempotent_on_same_session_recaptured() {
    let harness = extraction_harness();
    let first_capture = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let first = harness.run(&["status"]);
    let second_capture = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let second = harness.run(&["status"]);
    let first_decision_id = stable_uuid_from(
        &[&first_capture, &first],
        &["D_old", "auth module at src/auth.py"],
    );
    let second_decision_id = stable_uuid_from(
        &[&second_capture, &second],
        &["D_old", "auth module at src/auth.py"],
    );

    assert_eq!(
        (
            memory_counts(&first),
            memory_counts(&second),
            first_decision_id.is_some(),
            second_decision_id.is_some(),
            first_decision_id == second_decision_id,
        ),
        (Some((1, 1, 1)), Some((1, 1, 1)), true, true, true)
    );
}

#[test]
fn cli_by_intention() {
    let harness = extraction_harness();
    let capture = harness.run_with_stdin(&["record"], AUTH_CURRENT_WITH_JWT);
    let intention_id = stable_id(&capture, &["ship password-less login for acme", "G1"]);
    let store_path = store_path_from(&[&capture]);
    let store = persisted_store(store_path.clone());
    let persisted_intention = intention_id
        .as_ref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(IntentionId::from);
    let persisted = persisted_decisions_for_intention(store.as_ref(), persisted_intention);
    drop(store);
    let intention_arg = intention_id
        .as_deref()
        .unwrap_or("__missing_intention_id__");
    let walk = harness.run(&["by-intention", intention_arg]);
    let wrong_id = format!("{intention_arg}-wrong");
    let wrong = harness.run(&["by-intention", wrong_id.as_str()]);
    let text = combined(&walk);

    assert_eq!(
        (
            intention_id.as_ref().map(|_| "present"),
            store_path.is_some(),
            persisted
                .iter()
                .map(|(_, description, status)| (description.as_str(), *status))
                .collect::<Vec<_>>(),
        ),
        (
            Some("present"),
            true,
            vec![
                ("use JWT with 15-min access tokens", DecisionStatus::Active),
                (
                    "auth module at src/authentication.py",
                    DecisionStatus::Active,
                ),
            ],
        )
    );
    assert!(text.contains("goal: ship password-less login for acme"));
    assert!(text.contains("auth module at src/authentication.py"));
    assert!(text.contains("use JWT with 15-min access tokens"));
    assert!(!text.contains("store refresh tokens in Redis"));
    assert!(no_resolution(&wrong));
}

#[test]
fn cli_by_entity() {
    let harness = extraction_harness();
    let first = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let second = harness.run_with_stdin(&["record"], AUTH_RENAME);
    let entity_id = stable_id(&first, &["auth module file", "E_auth_module"]);
    let store_path = store_path_from(&[&first, &second]);
    let store = persisted_store(store_path.clone());
    let persisted_entity = entity_id
        .as_ref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(EntityId::from);
    let persisted = persisted_decisions_for_entity(store.as_ref(), persisted_entity);
    drop(store);
    let entity_arg = entity_id.as_deref().unwrap_or("__missing_entity_id__");
    let walk = harness.run(&["by-entity", entity_arg]);
    let wrong_id = format!("{entity_arg}-wrong");
    let wrong = harness.run(&["by-entity", wrong_id.as_str()]);
    let text = combined(&walk);

    assert_eq!(
        (
            entity_id.as_ref().map(|_| "present"),
            store_path.is_some(),
            persisted
                .iter()
                .map(|(_, description, status)| (description.as_str(), *status))
                .collect::<Vec<_>>(),
        ),
        (
            Some("present"),
            true,
            vec![
                (
                    "auth module at src/authentication.py",
                    DecisionStatus::Active
                ),
                ("auth module at src/auth.py", DecisionStatus::Invalidated),
            ],
        )
    );
    assert!(text.contains("auth module file: src/authentication.py"));
    assert!(text.contains("auth module at src/authentication.py"));
    assert!(text.contains("auth module at src/auth.py [invalidated"));
    assert!(text.contains("E_auth_module path changed src/auth.py -> src/authentication.py"));
    assert!(!text.contains("Redis"));
    assert!(no_resolution(&wrong));
}

#[test]
fn cli_by_entity_renders_non_auth_entity_history_from_stored_rows() {
    let harness = extraction_harness();
    let capture = harness.run_with_stdin(
        &["record"],
        "Webhook ingestion must stay reliable. The webhook worker file is src/webhook.rs; route webhooks through the signed worker and note the worker path is src/webhook.rs.",
    );
    let entity_id =
        stable_id(&capture, &["E_worker"]).unwrap_or_else(|| "__missing_entity_id__".to_string());
    let walk = harness.run(&["by-entity", &entity_id]);
    let text = combined(&walk);

    assert!(text.contains("webhook worker file: src/webhook.rs"));
    assert!(text.contains("route webhooks through signed worker [current]"));
    assert!(text.contains("E_worker path observed at src/webhook.rs"));
    assert!(!text.contains("auth module"));
    assert!(!text.contains("src/authentication.py"));
}

#[test]
fn cli_walk_verbs_do_not_cross_context_by_default() {
    let harness = extraction_harness();
    let client_a = harness.run_with_stdin(&["record", "--context", "client-a"], AUTH_INITIAL);
    harness.run_with_stdin(
        &["record", "--context", "client-b"],
        "Client B memory must stay separate, so use isolated deploy notes.",
    );
    let intention_id = stable_id(&client_a, &["ship password-less login for acme", "G1"])
        .unwrap_or_else(|| "__missing_intention_id__".to_string());
    let entity_id = stable_id(&client_a, &["auth module file", "E_auth_module"])
        .unwrap_or_else(|| "__missing_entity_id__".to_string());
    let wrong_intention = harness.run(&["by-intention", "--context", "client-b", &intention_id]);
    let wrong_entity = harness.run(&["by-entity", "--context", "client-b", &entity_id]);

    assert!(no_resolution(&wrong_intention));
    assert!(no_resolution(&wrong_entity));
}

#[test]
fn cli_impact_analysis() {
    let harness = extraction_harness();
    harness.run_with_stdin(&["record"], AUTH_INITIAL);
    harness.run_with_stdin(&["record"], AUTH_RENAME);
    let output = harness.run(&["impact"]);
    let text = combined(&output);

    assert!(text.contains("auth module at src/auth.py [INVALIDATED]"));
    assert!(text.contains("based on: src/auth.py -> src/authentication.py"));
    assert!(text.contains("served goal: ship password-less login"));
    assert!(text.contains(
        "goal still served by current decision \"auth module at src/authentication.py\""
    ));
}

#[test]
fn cli_impact_goal_still_served_when_another_decision_serves() {
    let harness = extraction_harness();
    harness.run_with_stdin(
        &["record"],
        "Rollout must stay available. The primary rollout basis is src/d1.py and fallback rollout basis is src/d2.py; D1 uses the primary basis and D2 uses the fallback basis.",
    );
    harness.run_with_stdin(
        &["record"],
        "The primary rollout basis moved from src/d1.py to src/d1_v2.py.",
    );
    let first = harness.run(&["impact"]);
    harness.run_with_stdin(
        &["record"],
        "The fallback rollout basis moved from src/d2.py to src/d2_v2.py.",
    );
    let second = harness.run(&["impact"]);

    assert!(combined(&first).contains("goal still served by current decision \"D2\""));
    assert!(!combined(&first).contains("goal unserved"));
    assert!(combined(&second).contains("goal unserved"));
}

#[test]
fn cli_invalidation_lifecycle() {
    let harness = extraction_harness();
    harness.run_with_stdin(&["record"], AUTH_INITIAL);
    harness.run_with_stdin(&["record"], AUTH_RENAME);
    let output = harness.run(&["impact"]);
    let text = combined(&output);

    assert!(text.contains("auth module at src/auth.py [INVALIDATED]"));
    assert!(text.contains("auth module at src/authentication.py"));
    assert!(!text.contains("resolved"));
}

#[test]
fn cli_changed_surfaces_cross_session_invalidation() {
    let harness = extraction_harness();
    harness.run_with_stdin(&["record"], AUTH_INITIAL);
    harness.run_with_stdin(&["record"], AUTH_RENAME);
    let changed = harness.run(&["changed"]);
    let second = harness.run(&["changed"]);

    assert!(combined(&changed).contains("newly INVALIDATED"));
    assert!(combined(&changed).contains("auth module at src/auth.py"));
    assert!(combined(&changed).contains("ship password-less login"));
    assert!(combined(&second).contains("nothing newly stale"));
}

#[test]
fn impact_standing_set_vs_changed_recent_delta_are_distinct_observables() {
    let harness = extraction_harness();
    let before = harness.run_with_stdin(
        &["record"],
        "Track the before file at src/before.py so the before path stays visible.",
    );
    let before_invalidation = harness.run_with_stdin(
        &["record"],
        "The before file moved from src/before.py to src/before_v2.py.",
    );
    assert_record_contains(&before, "src/before.py");
    assert_record_text(&before_invalidation, "src/before_v2.py");
    let cut = harness.run_with_stdin(
        &["record"],
        "Track the cut file at src/cut.py so the cut path stays visible.",
    );
    let cut_invalidation = harness.run_with_stdin(
        &["record"],
        "The cut file moved from src/cut.py to src/cut_v2.py.",
    );
    assert_record_contains(&cut, "src/cut.py");
    assert_record_text(&cut_invalidation, "src/cut_v2.py");
    let cutoff_changed = harness.run(&["changed"]);
    let after = harness.run_with_stdin(
        &["record"],
        "Track the after file at src/after.py so the after path stays visible.",
    );
    let after_invalidation = harness.run_with_stdin(
        &["record"],
        "The after file moved from src/after.py to src/after_v2.py.",
    );
    assert_record_contains(&after, "src/after.py");
    assert_record_text(&after_invalidation, "src/after_v2.py");
    let changed = harness.run(&["changed"]);
    let repeated = harness.run(&["changed"]);
    let impact = harness.run(&["impact"]);
    let store_path = store_path_from(&[
        &before,
        &before_invalidation,
        &cut,
        &cut_invalidation,
        &cutoff_changed,
        &after,
        &after_invalidation,
        &changed,
        &impact,
    ]);
    let store = persisted_store(store_path.clone());
    let context_id = context_id_from(&[
        &before,
        &before_invalidation,
        &cut,
        &cut_invalidation,
        &after,
        &after_invalidation,
        &changed,
        &impact,
    ]);
    let before_id =
        stable_uuid_from(&[&before, &cutoff_changed, &impact], &["D_before"]).map(DecisionId::from);
    let cut_id =
        stable_uuid_from(&[&cut, &cutoff_changed, &impact], &["D_cut"]).map(DecisionId::from);
    let after_id =
        stable_uuid_from(&[&after, &changed, &impact], &["D_after"]).map(DecisionId::from);
    let expected_initial_delta = sorted_present_decision_ids(&[before_id, cut_id]);
    let expected_recent_delta = sorted_present_decision_ids(&[after_id]);
    let expected_standing = sorted_present_decision_ids(&[before_id, cut_id, after_id]);

    assert_eq!(
        (
            store_path.is_some(),
            context_id.is_some(),
            before_id.is_some(),
            cut_id.is_some(),
            after_id.is_some(),
            persisted_invalidation_decision_ids(store.as_ref(), context_id),
            invalidation_decision_ids_from_output(&cutoff_changed, &["changed"]),
            invalidation_decision_ids_from_output(&changed, &["changed"]),
            invalidation_decision_ids_from_output(&repeated, &["changed"]),
            invalidation_decision_ids_from_output(&impact, &["standing_invalidations"]),
        ),
        (
            true,
            true,
            true,
            true,
            true,
            expected_standing.clone(),
            expected_initial_delta,
            expected_recent_delta,
            Vec::<DecisionId>::new(),
            expected_standing,
        )
    );
}

#[test]
fn cli_query_and_impact_emit_stable_ids_consumed_by_walk_verbs() {
    let harness = extraction_harness();
    let first = harness.run_with_stdin(&["record"], AUTH_INITIAL);
    let second = harness.run_with_stdin(&["record"], AUTH_RENAME);
    let query = harness.run(&["query", "where is the auth module"]);
    let impact = harness.run(&["impact"]);
    let intention_id = stable_id(&query, &["ship password-less login for acme", "G1"]);
    let entity_id = stable_id(&impact, &["auth module file", "E_auth_module"]);
    let store_path = store_path_from(&[&first, &second, &query, &impact]);
    let store = persisted_store(store_path.clone());
    let persisted_intention = intention_id
        .as_ref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(IntentionId::from);
    let persisted_entity = entity_id
        .as_ref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(EntityId::from);
    let by_intention_rows = persisted_decisions_for_intention(store.as_ref(), persisted_intention);
    let by_entity_rows = persisted_decisions_for_entity(store.as_ref(), persisted_entity);
    drop(store);
    let intention_arg = intention_id
        .as_deref()
        .unwrap_or("__missing_intention_id__");
    let entity_arg = entity_id.as_deref().unwrap_or("__missing_entity_id__");
    let by_intention = harness.run(&["by-intention", intention_arg]);
    let by_entity = harness.run(&["by-entity", entity_arg]);
    let wrong_intention_id = format!("{intention_arg}-wrong");
    let wrong_entity_id = format!("{entity_arg}-wrong");
    let wrong_intention = harness.run(&["by-intention", wrong_intention_id.as_str()]);
    let wrong_entity = harness.run(&["by-entity", wrong_entity_id.as_str()]);

    assert_eq!(
        (
            intention_id.is_some(),
            entity_id.is_some(),
            store_path.is_some(),
            by_intention_rows
                .iter()
                .map(|(_, description, _)| description.as_str())
                .collect::<Vec<_>>(),
            by_entity_rows
                .iter()
                .map(|(_, description, _)| description.as_str())
                .collect::<Vec<_>>(),
        ),
        (
            true,
            true,
            true,
            vec!["auth module at src/authentication.py"],
            vec![
                "auth module at src/authentication.py",
                "auth module at src/auth.py",
            ],
        )
    );
    assert!(combined(&by_intention).contains("ship password-less login for acme"));
    assert!(combined(&by_entity).contains("auth module file"));
    assert!(no_resolution(&wrong_intention));
    assert!(no_resolution(&wrong_entity));
}

#[test]
fn record_lands_in_working_location_context_by_default() {
    let harness = extraction_harness();
    let dir_x = tempfile::tempdir().expect("dir x");
    let dir_y = tempfile::tempdir().expect("dir y");
    run_in_dir(&harness, dir_x.path(), &["record", PROJECT_X_RECORD]);
    let x_status = run_in_dir(&harness, dir_x.path(), &["status"]);
    let y_status = run_in_dir(&harness, dir_y.path(), &["status"]);
    run_in_dir(
        &harness,
        dir_x.path(),
        &["record", "--context", "named-client", "override"],
    );
    let named = harness.run(&["status", "--context", "named-client"]);

    assert!(combined(&x_status).contains("context:"));
    assert!(combined(&x_status).contains("goals: 1"));
    assert!(combined(&y_status).contains("goals: 0"));
    assert!(combined(&named).contains("context: named-client"));
}

#[test]
fn cli_query_cross_context_isolation_default_and_opt_in() {
    let harness = extraction_harness();
    let client_a =
        harness.run_with_stdin(&["record", "--context", "client-a"], CLIENT_A_SOX_SESSION);
    let client_b = harness.run_with_stdin(
        &["record", "--context", "client-b"],
        CLIENT_B_PUBLISH_SESSION,
    );
    assert_record_contains(&client_a, "SOX approval required before promote");
    assert_record_contains(&client_b, "hold 10 min before promote");
    let isolated = harness.run(&["query", "--context", "client-b", "duplicate publish"]);
    let crossed = harness.run(&["query", "--cross-context", "client-a,client-b", "publish"]);

    assert!(
        combined(&isolated).contains("hold 10 min before promote"),
        "{}",
        combined(&isolated)
    );
    assert!(!combined(&isolated).contains("SOX"));
    assert!(combined(&crossed).contains("client-a"));
    assert!(combined(&crossed).contains("client-b"));
}

#[test]
fn cross_context_isolation_holds_on_explicit_context_in_shared_directory() {
    let harness = extraction_harness();
    let client_a =
        harness.run_with_stdin(&["record", "--context", "client-a"], CLIENT_A_SOX_SESSION);
    let client_b = harness.run_with_stdin(
        &["record", "--context", "client-b"],
        CLIENT_B_PUBLISH_SESSION,
    );
    assert_record_contains(&client_a, "SOX approval required before promote");
    assert_record_contains(&client_b, "hold 10 min before promote");
    let output = harness.run(&["query", "--context", "client-b", "duplicate publish"]);

    assert!(combined(&output).contains("client-b"));
    assert!(
        combined(&output).contains("hold 10 min before promote"),
        "{}",
        combined(&output)
    );
    assert!(!combined(&output).contains("SOX"));
}

#[test]
fn read_commands_do_not_create_missing_contexts() {
    let harness = extraction_harness();
    let missing_id = uuid::Uuid::now_v7().to_string();
    let query = harness.run(&["query", "--context", "typo-client", "anything"]);
    let impact = harness.run(&["impact", "--context", "typo-client"]);
    let changed = harness.run(&["changed", "--context", "typo-client"]);
    let by_intention = harness.run(&[
        "by-intention",
        "--context",
        "typo-client",
        missing_id.as_str(),
    ]);
    let by_entity = harness.run(&["by-entity", "--context", "typo-client", missing_id.as_str()]);
    let store_path = store_path_from(&[&query, &impact, &changed, &by_intention, &by_entity]);

    assert!(combined(&query).contains("no matching memory"));
    assert!(combined(&impact).contains("nothing stale"));
    assert!(combined(&changed).contains("nothing newly stale"));
    assert!(no_resolution(&by_intention));
    assert!(no_resolution(&by_entity));
    assert_eq!(
        persisted_context_names(store_path.as_deref()),
        Vec::<String>::new()
    );
}

#[test]
fn cli_first_capture_load_is_observable_not_silent() {
    let harness = extraction_harness();
    let first = harness.run_with_stdin(&["record"], "first auth capture");
    let second = harness.run_with_stdin(&["record"], "second auth capture");

    assert!(combined(&first).contains("loading bundled text embedder"));
    assert!(!combined(&second).contains("loading bundled text embedder"));
}
