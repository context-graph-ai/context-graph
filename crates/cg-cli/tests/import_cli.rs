mod support;

use context_graph::{DecisionId, DecisionStatus, DiscoveryFilter, IntentionId, Store, StoreConfig};
use contextdb_core::Value;
use contextdb_engine::Database;
use serde_json::Value as JsonValue;
use support::{CliHarness, CommandOutput, write_file};

fn combined(output: &CommandOutput) -> String {
    format!("{}\n{}", output.stdout, output.stderr)
}

fn output_json(output: &CommandOutput) -> Option<JsonValue> {
    serde_json::from_str(&output.stdout).ok()
}

fn json_string_at<'a>(json: &'a JsonValue, paths: &[&str]) -> Option<&'a str> {
    paths
        .iter()
        .find_map(|path| json.pointer(path).and_then(JsonValue::as_str))
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

fn stable_id(output: &CommandOutput, keys: &[&str]) -> Option<String> {
    let json = output_json(output)?;
    let stable_ids = json.get("stable_ids").and_then(JsonValue::as_object)?;
    keys.iter().find_map(|key| {
        stable_ids
            .get(*key)
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned)
    })
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

fn memory_counts(output: &CommandOutput) -> Option<(u64, u64, u64)> {
    if let Some(json) = output_json(output) {
        let goals = json
            .pointer("/totals/goals")
            .or_else(|| json.pointer("/goals"))
            .and_then(JsonValue::as_u64);
        let decisions = json
            .pointer("/totals/decisions")
            .or_else(|| json.pointer("/decisions"))
            .and_then(JsonValue::as_u64);
        let observations = json
            .pointer("/totals/observations")
            .or_else(|| json.pointer("/observations"))
            .and_then(JsonValue::as_u64);
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct PersistedCounts {
    contexts: usize,
    digests: usize,
    intentions: usize,
    decisions: usize,
    observations: usize,
    extracted_from_edges: usize,
}

fn persisted_counts(path: Option<&std::path::Path>) -> PersistedCounts {
    let Some(path) = path else {
        return PersistedCounts::default();
    };
    let Ok(db) = Database::open(path) else {
        return PersistedCounts::default();
    };
    let snapshot = db.snapshot();
    PersistedCounts {
        contexts: db
            .scan("contexts", snapshot)
            .map(|rows| rows.len())
            .unwrap_or_default(),
        digests: db
            .scan("digests", snapshot)
            .map(|rows| rows.len())
            .unwrap_or_default(),
        intentions: db
            .scan("intentions", snapshot)
            .map(|rows| rows.len())
            .unwrap_or_default(),
        decisions: db
            .scan("decisions", snapshot)
            .map(|rows| rows.len())
            .unwrap_or_default(),
        observations: db
            .scan("observations", snapshot)
            .map(|rows| rows.len())
            .unwrap_or_default(),
        extracted_from_edges: db
            .scan("edges", snapshot)
            .map(|rows| {
                rows.iter()
                    .filter(|row| {
                        matches!(
                            row.values.get("edge_type"),
                            Some(Value::Text(edge_type)) if edge_type == "EXTRACTED_FROM"
                        )
                    })
                    .count()
            })
            .unwrap_or_default(),
    }
}

fn persisted_text_values(path: Option<&std::path::Path>, table: &str, column: &str) -> Vec<String> {
    let Some(path) = path else {
        return Vec::new();
    };
    let Ok(db) = Database::open(path) else {
        return Vec::new();
    };
    let mut values = db
        .scan(table, db.snapshot())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| match row.values.get(column) {
            Some(Value::Text(value)) => Some(value.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    values.sort();
    values
}

const CLAUDE_SESSION: &str = r#"{"type":"system","content":"Goal: raw system decoy should not persist. Decision: raw system decoy should not persist."}
{"type":"user","message":{"content":[{"type":"text","text":"Goal: ship password-less login for acme"}]}}
{"type":"user","message":{"content":[{"type":"text","text":"Goal: keep audit trail for auth migrations"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","name":"grep","input":{"query":"Decision: raw tool-use decoy should not persist"}},{"type":"text","text":"Decision: auth module at src/auth.py"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_result","content":"Observation: raw tool-result decoy should not persist"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Decision: use JWT with 15-min access tokens"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Decision: write auth migration notes to docs/auth-migration.md"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Observation: auth module file observed at src/auth.py"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Observation: auth tests cover password-less login"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Observation: JWT refresh token TTL remains 7 days"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Observation: audit trail lives in docs/auth-migration.md"}]}}
"#;

const EXPECTED_CLEAN_CLAUDE_SESSION: &str = "User: Goal: ship password-less login for acme\nUser: Goal: keep audit trail for auth migrations\nAssistant: Decision: auth module at src/auth.py\nAssistant: Decision: use JWT with 15-min access tokens\nAssistant: Decision: write auth migration notes to docs/auth-migration.md\nAssistant: Observation: auth module file observed at src/auth.py\nAssistant: Observation: auth tests cover password-less login\nAssistant: Observation: JWT refresh token TTL remains 7 days\nAssistant: Observation: audit trail lives in docs/auth-migration.md\n";

const SECOND_CLAUDE_SESSION: &str = r#"{"type":"user","message":{"content":[{"type":"text","text":"Goal: reduce checkout latency"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Decision: cache product catalog in memory"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Observation: checkout latency p95 dropped to 220ms"}]}}
"#;

const EXPECTED_CLEAN_SECOND_CLAUDE_SESSION: &str = "User: Goal: reduce checkout latency\nAssistant: Decision: cache product catalog in memory\nAssistant: Observation: checkout latency p95 dropped to 220ms\n";

#[test]
fn cli_import_claude_session_becomes_memory() {
    let harness = CliHarness::new();
    let session = harness.data_root.path().join("session.jsonl");
    write_file(&session, CLAUDE_SESSION);

    let before = harness.run(&["status"]);
    let dry_run = harness.run(&["import", "--dry-run", session.to_str().expect("utf8 path")]);
    let after_dry_run = harness.run(&["status"]);
    let import = harness.run(&["import", session.to_str().expect("utf8 path")]);
    let status = harness.run(&["status"]);
    let query = harness.run(&["query", "auth module"]);
    let impact = harness.run(&["impact"]);
    let store_path = store_path_from(&[&dry_run, &import, &status, &query, &impact]);
    let store = persisted_store(store_path.clone());
    let goal_id = stable_uuid_from(
        &[&import, &query, &impact],
        &["ship password-less login for acme", "G1"],
    )
    .map(IntentionId::from);
    let audit_goal_id = stable_uuid_from(
        &[&import, &query, &impact],
        &["keep audit trail for auth migrations", "G_auth_audit"],
    )
    .map(IntentionId::from);
    let auth_decision_id = stable_uuid_from(
        &[&import, &query, &impact],
        &["auth module at src/auth.py", "D_old"],
    )
    .map(DecisionId::from);
    let docs_decision_id = stable_uuid_from(
        &[&import, &query, &impact],
        &[
            "write auth migration notes to docs/auth-migration.md",
            "D_auth_docs",
        ],
    )
    .map(DecisionId::from);
    let persisted = persisted_decisions_for_intention(store.as_ref(), goal_id);
    let audit_persisted = persisted_decisions_for_intention(store.as_ref(), audit_goal_id);
    let second_session = harness.data_root.path().join("session-two.jsonl");
    write_file(&second_session, SECOND_CLAUDE_SESSION);
    let second_import = harness.run(&["import", second_session.to_str().expect("utf8 path")]);
    let second_status = harness.run(&["status"]);
    let second_query = harness.run(&["query", "checkout latency"]);
    let second_goal_id = stable_uuid_from(
        &[&second_import, &second_query],
        &["reduce checkout latency", "G_checkout_latency"],
    )
    .map(IntentionId::from);
    let reopened = persisted_store(store_path.clone());
    let second_persisted = persisted_decisions_for_intention(reopened.as_ref(), second_goal_id);
    let store_path_ref = store_path.as_deref();
    let mut expected_digest_summaries = vec![
        EXPECTED_CLEAN_CLAUDE_SESSION.to_string(),
        EXPECTED_CLEAN_SECOND_CLAUDE_SESSION.to_string(),
    ];
    expected_digest_summaries.sort();

    assert_eq!(
        (
            dry_run.code,
            import.code,
            second_import.code,
            memory_counts(&before),
            memory_counts(&after_dry_run),
            memory_counts(&status),
            memory_counts(&second_status),
        ),
        (
            0,
            0,
            0,
            Some((0, 0, 0)),
            Some((0, 0, 0)),
            Some((2, 3, 4)),
            Some((3, 4, 5)),
        )
    );
    assert_eq!(
        (
            store_path.is_some(),
            goal_id.is_some(),
            audit_goal_id.is_some(),
            auth_decision_id.is_some(),
            docs_decision_id.is_some(),
            second_goal_id.is_some(),
        ),
        (true, true, true, true, true, true)
    );
    assert_eq!(
        persisted_counts(store_path_ref),
        PersistedCounts {
            contexts: 1,
            digests: 2,
            intentions: 3,
            decisions: 4,
            observations: 5,
            extracted_from_edges: 12,
        }
    );
    assert_eq!(
        persisted_text_values(store_path_ref, "digests", "summary"),
        expected_digest_summaries
    );
    assert!(
        persisted_text_values(store_path_ref, "digests", "summary")
            .iter()
            .all(|summary| !summary.contains("raw system decoy")
                && !summary.contains("raw tool-use decoy")
                && !summary.contains("raw tool-result decoy"))
    );
    assert_eq!(
        (
            persisted_text_values(store_path_ref, "intentions", "description"),
            persisted_text_values(store_path_ref, "decisions", "description"),
            persisted_text_values(store_path_ref, "observations", "observation_type"),
        ),
        (
            vec![
                "keep audit trail for auth migrations".to_string(),
                "reduce checkout latency".to_string(),
                "ship password-less login for acme".to_string(),
            ],
            vec![
                "auth module at src/auth.py".to_string(),
                "cache product catalog in memory".to_string(),
                "use JWT with 15-min access tokens".to_string(),
                "write auth migration notes to docs/auth-migration.md".to_string(),
            ],
            vec![
                "JWT refresh token TTL remains 7 days".to_string(),
                "audit trail lives in docs/auth-migration.md".to_string(),
                "auth module file observed at src/auth.py".to_string(),
                "auth tests cover password-less login".to_string(),
                "checkout latency p95 dropped to 220ms".to_string(),
            ],
        )
    );
    assert_eq!(
        (
            persisted
                .iter()
                .map(|(_, description, status)| (description.as_str(), *status))
                .collect::<Vec<_>>(),
            second_persisted
                .iter()
                .map(|(_, description, status)| (description.as_str(), *status))
                .collect::<Vec<_>>(),
            audit_persisted
                .iter()
                .map(|(_, description, status)| (description.as_str(), *status))
                .collect::<Vec<_>>(),
        ),
        (
            vec![
                ("auth module at src/auth.py", DecisionStatus::Active),
                ("use JWT with 15-min access tokens", DecisionStatus::Active),
            ],
            vec![("cache product catalog in memory", DecisionStatus::Active)],
            vec![(
                "write auth migration notes to docs/auth-migration.md",
                DecisionStatus::Active,
            )],
        )
    );
    assert!(combined(&query).contains("auth module at src/auth.py"));
    assert!(combined(&query).contains("use JWT with 15-min access tokens"));
    assert!(combined(&query).contains("auth tests cover password-less login"));
    assert!(
        combined(&query).contains(
            goal_id
                .map(|id| id.as_uuid().to_string())
                .unwrap_or_default()
                .as_str()
        )
    );
    assert!(
        combined(&impact).contains(
            auth_decision_id
                .map(|id| id.as_uuid().to_string())
                .unwrap_or_default()
                .as_str()
        )
    );
    assert!(
        combined(&impact).contains(
            docs_decision_id
                .map(|id| id.as_uuid().to_string())
                .unwrap_or_default()
                .as_str()
        )
    );
    assert!(combined(&second_query).contains("cache product catalog in memory"));
}

#[test]
fn cli_import_rejects_bulk_single_file_only() {
    let harness = CliHarness::new();
    let bulk_dir = tempfile::tempdir().expect("bulk dir");
    let before = harness.run(&["status"]);
    let output = harness.run(&["import", bulk_dir.path().to_str().expect("utf8 dir")]);
    let after = harness.run(&["status"]);

    assert_ne!(output.code, 0);
    assert!(combined(&output).contains("single file only"));
    assert_eq!(combined(&before), combined(&after));
}

#[test]
fn cli_record_is_clean_text_escape_hatch_not_a_parser() {
    let harness = CliHarness::new();
    let session = harness.data_root.path().join("session.jsonl");
    write_file(&session, CLAUDE_SESSION);

    let import_harness = CliHarness::new();
    let import_session = import_harness.data_root.path().join("session.jsonl");
    write_file(&import_session, CLAUDE_SESSION);
    let import = import_harness.run(&["import", import_session.to_str().expect("utf8 path")]);
    let import_status = import_harness.run(&["status"]);
    let import_store_path = store_path_from(&[&import, &import_status]);
    let import_store_path_ref = import_store_path.as_deref();
    let import_digest_summaries =
        persisted_text_values(import_store_path_ref, "digests", "summary");

    let before_record = harness.run(&["status"]);
    let record = harness.run(&["record", session.to_str().expect("utf8 path")]);
    let after_record = harness.run(&["status"]);
    let record_store_path = store_path_from(&[&record, &after_record]);
    let record_store_path_ref = record_store_path.as_deref();
    let typed_refusal = record.code != 0
        && combined(&record).contains("use `cg import` for session files")
        && memory_counts(&before_record) == memory_counts(&after_record)
        && persisted_counts(record_store_path_ref) == PersistedCounts::default();
    let raw_verbatim_digest = record.code == 0
        && memory_counts(&after_record) == Some((0, 0, 0))
        && persisted_counts(record_store_path_ref)
            == PersistedCounts {
                contexts: 1,
                digests: 1,
                intentions: 0,
                decisions: 0,
                observations: 0,
                extracted_from_edges: 0,
            }
        && persisted_text_values(record_store_path_ref, "digests", "summary")
            .iter()
            .any(|summary| summary.contains(CLAUDE_SESSION.lines().next().unwrap_or_default()));

    assert_eq!(
        (
            import.code,
            memory_counts(&import_status),
            persisted_counts(import_store_path_ref),
            import_digest_summaries,
            persisted_text_values(import_store_path_ref, "intentions", "description"),
            persisted_text_values(import_store_path_ref, "decisions", "description"),
            typed_refusal || raw_verbatim_digest,
            persisted_counts(record_store_path_ref) != persisted_counts(import_store_path_ref),
            persisted_text_values(record_store_path_ref, "intentions", "description")
                != persisted_text_values(import_store_path_ref, "intentions", "description"),
            persisted_text_values(record_store_path_ref, "decisions", "description")
                != persisted_text_values(import_store_path_ref, "decisions", "description"),
        ),
        (
            0,
            Some((2, 3, 4)),
            PersistedCounts {
                contexts: 1,
                digests: 1,
                intentions: 2,
                decisions: 3,
                observations: 4,
                extracted_from_edges: 9,
            },
            vec![EXPECTED_CLEAN_CLAUDE_SESSION.to_string()],
            vec![
                "keep audit trail for auth migrations".to_string(),
                "ship password-less login for acme".to_string(),
            ],
            vec![
                "auth module at src/auth.py".to_string(),
                "use JWT with 15-min access tokens".to_string(),
                "write auth migration notes to docs/auth-migration.md".to_string(),
            ],
            true,
            true,
            true,
            true,
        )
    );
}

#[test]
fn cli_import_missing_file_typed_error() {
    let harness = CliHarness::new();
    let before = harness.run(&["status"]);
    let output = harness.run(&["import", "./does-not-exist.jsonl"]);
    let after = harness.run(&["status"]);

    assert_ne!(output.code, 0);
    assert!(combined(&output).contains("ImportPathNotFound"));
    assert_eq!(combined(&before), combined(&after));
}

#[test]
fn cli_import_unknown_format_surfaces_typed_error_to_user() {
    let harness = CliHarness::new();
    let path = harness.data_root.path().join("session.unknownext");
    write_file(&path, "unknown format");
    let before = harness.run(&["status"]);
    let output = harness.run(&["import", path.to_str().expect("utf8 path")]);
    let after = harness.run(&["status"]);

    assert_ne!(output.code, 0);
    assert!(combined(&output).contains("UnsupportedImportFormat"));
    assert_eq!(combined(&before), combined(&after));
}
