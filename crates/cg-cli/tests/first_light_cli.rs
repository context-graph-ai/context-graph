mod support;

use std::process::{Command, Stdio};

use context_graph::{
    ContextId, CreateDecision, CreateIntention, DecisionId, DecisionStatus, DiscoveryFilter,
    EntityId, IntentionId, IntentionOrigin, IntentionStatus, Store, StoreConfig,
};
use contextdb_core::Value as DbValue;
use contextdb_engine::Database;
use serde_json::Value;
use support::{CliHarness, CommandOutput, cg_bin};

fn combined(output: &CommandOutput) -> String {
    format!("{}\n{}", output.stdout, output.stderr)
}

fn run_in_dir(harness: &CliHarness, cwd: &std::path::Path, args: &[&str]) -> CommandOutput {
    let output = Command::new(cg_bin())
        .args(args)
        .current_dir(cwd)
        .env("CG_DATA_ROOT", harness.data_root.path())
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
            .by_entity_decisions(entity_id, DiscoveryFilter::default())
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
    let harness = CliHarness::new();
    let first = harness.run_with_stdin(&["record"], "auth module at src/auth.py");
    let output =
        harness.run_with_stdin(&["record"], "auth module renamed to src/authentication.py");
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
    let harness = CliHarness::new();
    let first = harness.run_with_stdin(&["record"], "auth module at src/auth.py");
    let second =
        harness.run_with_stdin(&["record"], "auth module renamed to src/authentication.py");
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
    assert!(text.contains("current: auth module lives at src/authentication.py"));
    assert!(text.contains("serves goal: ship password-less login"));
    assert!(text.contains("prior: auth module at src/auth.py [invalidated"));
    assert!(text.contains("use JWT with 15-min access tokens [superseded]"));
}

#[test]
fn cli_status() {
    let harness = CliHarness::new();
    let empty = harness.run(&["status"]);
    harness.run_with_stdin(&["record"], "session one auth and jwt decisions");
    let first = harness.run(&["status"]);
    harness.run_with_stdin(&["record"], "set up CI pipeline");
    let second = harness.run(&["status"]);

    assert!(combined(&empty).contains("goals: 0"));
    assert!(combined(&first).contains("goals: 3"));
    assert!(combined(&first).contains("decisions: 6"));
    assert!(combined(&first).contains("observations: 11"));
    assert!(combined(&second).contains("goals: 4"));
    assert!(combined(&second).contains("decisions: 8"));
    assert!(combined(&second).contains("observations: 15"));
}

#[test]
fn cli_status_idempotent_on_same_session_recaptured() {
    let harness = CliHarness::new();
    let first_capture = harness.run_with_stdin(&["record"], "auth module at src/auth.py");
    let first = harness.run(&["status"]);
    let second_capture = harness.run_with_stdin(
        &["record"],
        "the authentication module remains in the original path",
    );
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
        (Some((3, 6, 11)), Some((3, 6, 11)), true, true, true)
    );
}

#[test]
fn cli_by_intention() {
    let harness = CliHarness::new();
    let capture = harness.run_with_stdin(
        &["record"],
        "Goal: ship password-less login for acme. Decision: auth module at src/authentication.py. Decision: use JWT with 15-min access tokens.",
    );
    let intention_id = stable_id(&capture, &["ship password-less login for acme", "G1"]);
    let store_path = store_path_from(&[&capture]);
    let store = persisted_store(store_path.clone());
    let persisted_intention = intention_id
        .as_ref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(IntentionId::from);
    let persisted = persisted_decisions_for_intention(store.as_ref(), persisted_intention);
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
                (
                    "auth module at src/authentication.py",
                    DecisionStatus::Active
                ),
                ("use JWT with 15-min access tokens", DecisionStatus::Active),
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
    let harness = CliHarness::new();
    let capture = harness.run_with_stdin(
        &["record"],
        "Entity: auth module file. Decision: auth module at src/auth.py. Later: auth module at src/authentication.py.",
    );
    let entity_id = stable_id(&capture, &["auth module file", "E_auth_module"]);
    let store_path = store_path_from(&[&capture]);
    let store = persisted_store(store_path.clone());
    let persisted_entity = entity_id
        .as_ref()
        .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
        .map(EntityId::from);
    let persisted = persisted_decisions_for_entity(store.as_ref(), persisted_entity);
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
    assert!(text.contains("src/authentication.py (was src/auth.py)"));
    assert!(text.contains("auth module at src/authentication.py [current]"));
    assert!(text.contains("auth module at src/auth.py [invalidated"));
    assert!(text.contains("renamed during the following-Tuesday refactor"));
    assert!(!text.contains("Redis"));
    assert!(no_resolution(&wrong));
}

#[test]
fn cli_impact_analysis() {
    let harness = CliHarness::new();
    harness.run_with_stdin(&["record"], "auth module at src/auth.py");
    harness.run_with_stdin(&["record"], "auth module renamed to src/authentication.py");
    let output = harness.run(&["impact"]);
    let text = combined(&output);

    assert!(text.contains("auth module at src/auth.py [INVALIDATED]"));
    assert!(text.contains("renamed src/auth.py -> src/authentication.py"));
    assert!(text.contains("served goal: ship password-less login"));
    assert!(text.contains(
        "goal still served by current decision \"auth module at src/authentication.py\""
    ));
}

#[test]
fn cli_impact_goal_still_served_when_another_decision_serves() {
    let harness = CliHarness::new();
    harness.run_with_stdin(&["record"], "G served by D1 and D2; D1 invalidated");
    let first = harness.run(&["impact"]);
    harness.run_with_stdin(&["record"], "D2 invalidated too");
    let second = harness.run(&["impact"]);

    assert!(combined(&first).contains("goal still served by D2"));
    assert!(!combined(&first).contains("goal unserved"));
    assert!(combined(&second).contains("goal unserved"));
}

#[test]
fn cli_invalidation_lifecycle() {
    let harness = CliHarness::new();
    harness.run_with_stdin(&["record"], "auth module at src/auth.py");
    harness.run_with_stdin(&["record"], "rename invalidates old decision");
    harness.run_with_stdin(&["record"], "new decision supersedes old auth decision");
    let output = harness.run(&["impact"]);
    let text = combined(&output);

    assert!(text.contains("D_old resolved"));
    assert!(text.contains("auth module at src/authentication.py [current]"));
    assert!(!text.contains("D_old [open threat]"));
}

#[test]
fn cli_changed_surfaces_cross_session_invalidation() {
    let harness = CliHarness::new();
    harness.run_with_stdin(&["record"], "auth module at src/auth.py");
    harness.run_with_stdin(&["record"], "auth module renamed to src/authentication.py");
    let changed = harness.run(&["changed"]);
    let second = harness.run(&["changed"]);

    assert!(combined(&changed).contains("newly INVALIDATED"));
    assert!(combined(&changed).contains("auth module at src/auth.py"));
    assert!(combined(&changed).contains("ship password-less login"));
    assert!(combined(&second).contains("nothing newly stale"));
}

#[test]
fn impact_standing_set_vs_changed_recent_delta_are_distinct_observables() {
    let harness = CliHarness::new();
    let before = harness.run_with_stdin(
        &["record"],
        "Goal G_before. Entity E_before path=src/before.py. Decision D_before: src/before.py serves G_before. Observation: E_before path changed src/before.py -> src/before_v2.py, invalidating D_before.",
    );
    let cut = harness.run_with_stdin(
        &["record"],
        "Goal G_cut. Entity E_cut path=src/cut.py. Decision D_cut: src/cut.py serves G_cut. Observation: E_cut path changed src/cut.py -> src/cut_v2.py, invalidating D_cut.",
    );
    let cutoff_changed = harness.run(&["changed"]);
    let after = harness.run_with_stdin(
        &["record"],
        "Goal G_after. Entity E_after path=src/after.py. Decision D_after: src/after.py serves G_after. Observation: E_after path changed src/after.py -> src/after_v2.py, invalidating D_after.",
    );
    let changed = harness.run(&["changed"]);
    let repeated = harness.run(&["changed"]);
    let impact = harness.run(&["impact"]);
    let store_path = store_path_from(&[&before, &cut, &cutoff_changed, &after, &changed, &impact]);
    let store = persisted_store(store_path.clone());
    let context_id = context_id_from(&[&before, &cut, &after, &changed, &impact]);
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
    let harness = CliHarness::new();
    let capture = harness.run_with_stdin(
        &["record"],
        "Goal: ship password-less login for acme. Entity: auth module file. Decision: auth module moved from src/auth.py to src/authentication.py.",
    );
    let query = harness.run(&["query", "where is the auth module"]);
    let impact = harness.run(&["impact"]);
    let intention_id = stable_id(&query, &["ship password-less login for acme", "G1"]);
    let entity_id = stable_id(&impact, &["auth module file", "E_auth_module"]);
    let store_path = store_path_from(&[&capture, &query, &impact]);
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
    let harness = CliHarness::new();
    let dir_x = tempfile::tempdir().expect("dir x");
    let dir_y = tempfile::tempdir().expect("dir y");
    run_in_dir(
        &harness,
        dir_x.path(),
        &["record", "auth decision in project x"],
    );
    let x_status = run_in_dir(&harness, dir_x.path(), &["status"]);
    let y_status = run_in_dir(&harness, dir_y.path(), &["status"]);
    run_in_dir(
        &harness,
        dir_x.path(),
        &["record", "--context", "named-client", "override"],
    );
    let named = harness.run(&["status", "--context", "named-client"]);

    assert!(combined(&x_status).contains("context:"));
    assert!(combined(&x_status).contains("goals: 3"));
    assert!(combined(&y_status).contains("goals: 0"));
    assert!(combined(&named).contains("context: named-client"));
}

#[test]
fn cli_query_cross_context_isolation_default_and_opt_in() {
    let harness = CliHarness::new();
    harness.run_with_stdin(
        &["record", "--context", "client-a"],
        "SOX AWS deploy control",
    );
    harness.run_with_stdin(
        &["record", "--context", "client-b"],
        "hold 10 min before promote",
    );
    let isolated = harness.run(&["query", "--context", "client-b", "publish race"]);
    let crossed = harness.run(&[
        "query",
        "--cross-context",
        "client-a,client-b",
        "publish race",
    ]);

    assert!(combined(&isolated).contains("hold 10 min before promote"));
    assert!(!combined(&isolated).contains("SOX"));
    assert!(combined(&crossed).contains("client-a"));
    assert!(combined(&crossed).contains("client-b"));
}

#[test]
fn cross_context_isolation_holds_on_explicit_context_in_shared_directory() {
    let harness = CliHarness::new();
    harness.run_with_stdin(
        &["record", "--context", "client-a"],
        "SOX AWS deploy control",
    );
    harness.run_with_stdin(
        &["record", "--context", "client-b"],
        "hold 10 min before promote",
    );
    let output = harness.run(&["query", "--context", "client-b", "deploy delay"]);

    assert!(combined(&output).contains("client-b"));
    assert!(combined(&output).contains("hold 10 min before promote"));
    assert!(!combined(&output).contains("SOX"));
}

#[test]
fn cli_first_capture_load_is_observable_not_silent() {
    let harness = CliHarness::new();
    let first = harness.run_with_stdin(&["record"], "first auth capture");
    let second = harness.run_with_stdin(&["record"], "second auth capture");

    assert!(combined(&first).contains("loading bundled text embedder"));
    assert!(!combined(&second).contains("loading bundled text embedder"));
}
