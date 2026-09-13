mod support;

use contextdb_core::Value;
use contextdb_engine::Database;
use serde_json::Value as JsonValue;
use support::{CliHarness, CommandOutput, CompletionBehavior, LoginState, Provider};

fn combined(output: &CommandOutput) -> String {
    format!("{}\n{}", output.stdout, output.stderr)
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

// EXPECTATION CHANGE (2026-07-04, tool-output preservation, same rationale as the
// import_parser change): tool events now leave a one-line prose HEADER (tool name + sanitized
// artifact target only — free-content keys like this fixture's decoy `query` are never
// rendered, preserving the decoy contract). The old constant pinned "vanish without trace".

#[test]
fn cli_record_auth_replacement_uses_extractor_not_record_shorthand() {
    let harness = extraction_harness();

    let record = harness.run_with_stdin(
        &["record", "--understand-now"],
        "We are migrating the auth handler this week. Replace the auth handler with src/authentication.py.",
    );
    let status = harness.run(&["status"]);
    let store_path = store_path_from(&[&record, &status]);
    let store_path_ref = store_path.as_deref();

    assert_eq!(
        (
            record.code,
            memory_counts(&status),
            persisted_counts(store_path_ref),
            persisted_text_values(store_path_ref, "intentions", "description"),
            persisted_text_values(store_path_ref, "decisions", "description"),
        ),
        (
            0,
            Some((1, 1, 0)),
            PersistedCounts {
                contexts: 1,
                digests: 1,
                intentions: 1,
                decisions: 1,
                observations: 0,
                extracted_from_edges: 3,
            },
            vec!["migrate auth handler".to_string()],
            vec!["replace auth handler with src/authentication.py".to_string()],
        )
    );
}
